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
    """

    name = "huggingface"

    def __init__(
        self,
        text_model: Optional[str] = None,
        vision_model: Optional[str] = None,
        token: Optional[str] = None,
    ):
        from huggingface_hub import InferenceClient

        self.text_model = text_model or os.getenv("HF_TEXT_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
        self.vision_model = vision_model or os.getenv("HF_VISION_MODEL", "Qwen/Qwen2-VL-7B-Instruct")
        self.token = token or os.getenv("HUGGINGFACE_API_TOKEN")
        if not self.token:
            raise ValueError(
                "HUGGINGFACE_API_TOKEN is not set. Generate one at "
                "https://huggingface.co/settings/tokens (a PRO account raises "
                "your rate limits / unlocks more hosted endpoints, but a free "
                "account's token works for many smaller models too)."
            )
        self._text_client = InferenceClient(model=self.text_model, token=self.token)
        self._vision_client = InferenceClient(model=self.vision_model, token=self.token)

    def generate_structured(self, system_prompt: str, user_prompt: str, schema: Type[T]) -> T:
        # Most open-weight chat models don't support OpenAI-style
        # response_format, so we ask for JSON explicitly and parse/repair.
        schema_hint = json.dumps(schema.model_json_schema(), indent=2)
        full_prompt = (
            f"{system_prompt}\n\nYou MUST respond with ONLY a single JSON object "
            f"matching this schema (no markdown fences, no prose before or after):\n"
            f"{schema_hint}\n\nUser request: {user_prompt}"
        )
        raw = self._text_client.chat_completion(
            messages=[{"role": "user", "content": full_prompt}],
            max_tokens=512,
            temperature=0.2,
        )
        text = raw.choices[0].message.content
        data = _parse_json_block(text)
        return schema(**data)

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        raw = self._text_client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=1024,
            temperature=0.4,
        )
        return raw.choices[0].message.content

    def verify_frames(self, prompt_text: str, frame_paths: list[str]) -> dict:
        # HF Inference VLM chat-completion with image content blocks
        content = [{"type": "text", "text": prompt_text}]
        for fp in frame_paths:
            content.append({"type": "image_url", "image_url": {"url": _encode_frame_hf(fp)}})

        raw = self._vision_client.chat_completion(
            messages=[{"role": "user", "content": content}],
            max_tokens=600,
            temperature=0.0,
        )
        text = raw.choices[0].message.content
        return _parse_json_block(text)


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
                return json.loads(cleaned[start : end + 1])
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
