"""
backends/llm_backend.py

Provider-agnostic backend abstraction for Prompt-to-Drive.

The Generator needs a *text-to-structured-JSON* capability.
The Verifier needs a *vision+text-to-structured-JSON* capability.

This module defines a small interface for each and two concrete
implementations:

  - OpenAIBackend  -- uses OpenAI's structured outputs (response_format)
                      and GPT-4o vision.
  - HFBackend      -- uses the Hugging Face Inference API (works with a
                      HF PRO account / Inference Endpoints) for both an
                      instruction-tuned text model (Generator) and a
                      vision-language model (Verifier).

Why this exists: running the SAME closed-loop pipeline against two
architecturally different backends (one proprietary, one open-weight) is
a meaningful robustness/generalizability claim for the paper -- it shows
the contribution is the *architecture* (constrained JSON IR + hybrid
verification + targeted reflection), not a property of one specific
model. See eval/run_backend_comparison.py.
"""
from __future__ import annotations

import abc
import base64
import json
import logging
import os
from typing import Optional, Type, TypeVar

from pydantic import BaseModel

logger = logging.getLogger("prompt_to_drive.backends")

T = TypeVar("T", bound=BaseModel)


# --------------------------------------------------------------------- #
# Interfaces
# --------------------------------------------------------------------- #
class TextBackend(abc.ABC):
    """Text-in, structured-JSON-out. Used by the Generator agent."""

    name: str

    @abc.abstractmethod
    def generate_structured(self, system_prompt: str, user_prompt: str, schema: Type[T]) -> T:
        ...

    @abc.abstractmethod
    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        """Free-form text generation, used for the baseline code-gen comparator."""
        ...


class VisionBackend(abc.ABC):
    """Image(s)+text-in, structured-JSON-out. Used by the Verifier agent."""

    name: str

    @abc.abstractmethod
    def verify_frames(self, prompt_text: str, frame_paths: list[str]) -> dict:
        ...


# --------------------------------------------------------------------- #
# OpenAI backend
# --------------------------------------------------------------------- #
class OpenAIBackend(TextBackend, VisionBackend):
    """
    Requires OPENAI_API_KEY. Paid, per-token API -- see README "Cost"
    section for realistic per-run estimates. Uses structured outputs
    (response_format=<pydantic model>) so the Generator never has to
    hand-parse JSON.
    """

    name = "openai"

    def __init__(self, text_model: str = "gpt-4o-2024-08-06", vision_model: str = "gpt-4o"):
        from openai import OpenAI

        self.client = OpenAI()
        self.text_model = text_model
        self.vision_model = vision_model

    def generate_structured(self, system_prompt: str, user_prompt: str, schema: Type[T]) -> T:
        response = self.client.beta.chat.completions.parse(
            model=self.text_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=schema,
            temperature=0.2,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise ValueError("OpenAI structured output returned no parsed object.")
        return parsed

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.text_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
        )
        return response.choices[0].message.content

    def verify_frames(self, prompt_text: str, frame_paths: list[str]) -> dict:
        content = [{"type": "text", "text": prompt_text}]
        for fp in frame_paths:
            content.append(_encode_frame_openai(fp))
        response = self.client.chat.completions.create(
            model=self.vision_model,
            messages=[{"role": "user", "content": content}],
            max_tokens=600,
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()
        return _parse_json_block(raw)


def _encode_frame_openai(frame_path: str) -> dict:
    with open(frame_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}}


def _scenario_spec_hint() -> dict:
    """
    JSON-safe prompt schema for ScenarioSpec.

    Do not call ScenarioSpec.model_json_schema() here: the project's
    ScenarioSpec currently contains Literal typing metadata that is not
    JSON serializable in this environment.
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "weather",
            "town",
            "num_vehicles",
            "pedestrian",
            "cyclist",
            "event_type",
            "duration_s",
            "spawn_point_index",
            "notes",
        ],
        "properties": {
            "weather": {
                "type": "string",
                "enum": ["clear", "rain", "fog", "night", "cloudy"],
            },
            "town": {
                "type": "string",
                "enum": ["Town01", "Town02", "Town03", "Town04", "Town05"],
            },
            "num_vehicles": {
                "type": "integer",
                "minimum": 0,
                "maximum": 20,
            },
            "pedestrian": {"type": "boolean"},
            "cyclist": {"type": "boolean"},
            "event_type": {
                "type": "string",
                "enum": [
                    "crossing",
                    "lane_change",
                    "emergency_brake",
                    "red_light_violation",
                    "obstruction",
                    "overtaking",
                ],
            },
            "duration_s": {
                "type": "integer",
                "minimum": 8,
                "maximum": 30,
            },
            "spawn_point_index": {
                "type": "integer",
                "minimum": 0,
                "maximum": 200,
            },
            "notes": {"type": "string"},
        },
    }


# --------------------------------------------------------------------- #
# Hugging Face backend
# --------------------------------------------------------------------- #
class HFBackend(TextBackend, VisionBackend):
    """
    Uses the Hugging Face Inference API / Inference Endpoints. Works with
    a free or PRO HF account; PRO raises your included monthly inference
    credits and rate limits, and is required for many hosted VLM
    endpoints used here (see README "Cost" section for specifics).

    Defaults to open-weight instruction-tuned models for both roles:
      - text_model:   meta-llama/Meta-Llama-3.1-8B-Instruct (Generator)
      - vision_model: Qwen/Qwen2-VL-7B-Instruct (Verifier)

    Both are swappable via constructor args or the HF_TEXT_MODEL /
    HF_VISION_MODEL environment variables, so you can point this at any
    model your HF account has inference access to (including a private
    Inference Endpoint you deploy yourself).

    Token priority (Bug 2 fix):
      run_scenario.py checks HF_TOKEN (the standard HF CLI env var).
      This class reads BOTH HF_TOKEN and the legacy HUGGINGFACE_API_TOKEN,
      preferring HF_TOKEN so they stay consistent with no user friction.
    """

    name = "huggingface"

    # Vision models known to work on the HF free/PRO inference router.
    # If the configured model returns a 400 "not supported by any provider",
    # we fall back through this list before giving up and returning zeros.
    _VISION_FALLBACKS = [
            "Qwen/Qwen2-7B-Instruct",
            "Qwen/Qwen2.5-VL-7B-Instruct",
            "meta-llama/Llama-3.2-11B-Vision-Instruct",
        ]

    def __init__(
        self,
        text_model: Optional[str] = None,
        vision_model: Optional[str] = None,
        token: Optional[str] = None,
    ):
        from huggingface_hub import InferenceClient

        self.text_model = text_model or os.getenv(
            "HF_TEXT_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct"
        )
        self.vision_model = vision_model or os.getenv(
            "HF_VISION_MODEL", "Qwen/Qwen2-VL-7B-Instruct"
        )

        # Bug 2 fix: accept both HF_TOKEN (standard HF CLI var, also what
        # run_scenario.py validates) and the legacy HUGGINGFACE_API_TOKEN.
        self.token = (
            token
            or os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACE_API_TOKEN")
        )
        if not self.token:
            raise ValueError(
                "No HuggingFace token found. Set HF_TOKEN in your .env file "
                "(generate one at https://huggingface.co/settings/tokens). "
                "A PRO account raises rate limits and unlocks more hosted "
                "endpoints, but a free token works for many smaller models."
            )

        # Bug 4 fix: do NOT pass model= to InferenceClient constructor.
        # The HF router resolves the provider at call time based on the model
        # field in each chat_completion() call. Binding the model to the
        # client constructor bypasses the router's per-call model resolution
        # and causes 400 "model not supported" errors even for valid models.
        self._client = InferenceClient(token=self.token)

    def generate_structured(self, system_prompt: str, user_prompt: str, schema: Type[T]) -> T:
        if schema.__name__ == "ScenarioSpec":
            schema_hint = json.dumps(_scenario_spec_hint(), indent=2)
        else:
            schema_hint = json.dumps(schema.model_json_schema(), indent=2)

        full_prompt = (
            f"{system_prompt}\n\nYou MUST respond with ONLY a single JSON object "
            f"matching this schema (no markdown fences, no prose before or after):\n"
            f"{schema_hint}\n\nUser request: {user_prompt}"
        )
        # Bug 4 fix: model passed explicitly in the call, not the constructor.
        raw = self._client.chat_completion(
            model=self.text_model,
            messages=[{"role": "user", "content": full_prompt}],
            max_tokens=512,
            temperature=0.2,
        )
        text = raw.choices[0].message.content
        data = _parse_json_block(text)
        return schema(**data)

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        raw = self._client.chat_completion(
            model=self.text_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=1024,
            temperature=0.4,
        )
        return raw.choices[0].message.content

    def verify_frames(self, prompt_text: str, frame_paths: list[str]) -> dict:
        """
        Call the vision model with the given frames.

        Bug 3 fix: if the configured vision model returns a 400 "not supported
        by any provider" error, we try each fallback model in order before
        raising. This prevents a misconfigured HF_VISION_MODEL from collapsing
        all predicate scores to False/0 and making the verifier useless.
        If all models fail, we raise so the Verifier's _call_vlm() catches it
        and returns the graceful all-False result with confidence=0, which
        correctly feeds into the score as 0 VLM contribution rather than
        silently scoring everything wrong.
        """
        content = [{"type": "text", "text": prompt_text}]
        for fp in frame_paths:
            content.append({"type": "image_url", "image_url": {"url": _encode_frame_hf(fp)}})

        # Try the configured model first, then any fallbacks.
        models_to_try = [self.vision_model] + [
            m for m in self._VISION_FALLBACKS if m != self.vision_model
        ]

        last_exc = None
        for model in models_to_try:
            try:
                raw = self._client.chat_completion(
                    model=model,
                    messages=[{"role": "user", "content": content}],
                    max_tokens=600,
                    temperature=0.0,
                )
                text = raw.choices[0].message.content
                result = _parse_json_block(text)
                # If we fell back to a different model, log a warning so it's
                # visible without being a hard error.
                if model != self.vision_model:
                    logger.warning(
                        "Vision model '%s' unavailable; fell back to '%s'. "
                        "Set HF_VISION_MODEL to a model your account can access.",
                        self.vision_model,
                        model,
                    )
                return result
            except Exception as exc:
                last_exc = exc
                err_str = str(exc)
                if "not supported by any provider" in err_str or "400" in err_str:
                    logger.warning(
                        "Vision model '%s' not available on this account: %s",
                        model,
                        err_str[:120],
                    )
                    continue
                # Non-provider error (network, parse, etc.) — don't retry
                raise

        # All models exhausted
        raise ValueError(
            f"No usable HF vision model found. Tried: {models_to_try}. "
            f"Last error: {last_exc}. "
            f"Check your HF account's enabled providers at "
            f"https://huggingface.co/settings/inference-providers, or set "
            f"HF_VISION_MODEL to a model you have access to."
        ) from last_exc


def _encode_frame_hf(frame_path: str) -> str:
    with open(frame_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


# --------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------- #
def _parse_json_block(raw: str) -> dict:
    """Strips markdown fences and parses JSON, with one repair attempt
    (extracting the first {...} block) if the model added stray prose --
    open-weight models are more prone to this than OpenAI's structured
    outputs, which is itself a useful robustness data point to report."""
    cleaned = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(cleaned[start: end + 1])
            except json.JSONDecodeError:
                pass
        logger.error("Failed to parse JSON from backend output: %s", cleaned[:300])
        raise


def get_backend(name: str) -> "OpenAIBackend | HFBackend":
    """Factory used by the Generator/Verifier and eval scripts.
    name: "openai" or "huggingface" (case-insensitive)."""
    name = name.lower()
    if name in {"openai", "gpt", "gpt-4o"}:
        return OpenAIBackend()
    if name in {"huggingface", "hf"}:
        return HFBackend()
    raise ValueError(f"Unknown backend '{name}'. Use 'openai' or 'huggingface'.")