"""
backends/llm_backend.py

Provider-agnostic backend abstraction for Prompt-to-Drive.

Three concrete backends:

  OpenAIBackend      -- GPT-4o via OpenAI API (paid, structured outputs)
  HFBackend          -- HuggingFace Inference API (free/PRO, cloud-hosted)
  LocalVLLMBackend   -- locally served model via vLLM (YOUR SETUP)
                        Text:   meta-llama/Meta-Llama-3.1-8B-Instruct (or any text model)
                        Vision: Qwen/Qwen2.5-VL-7B-Instruct served at localhost:8000

To use your local Qwen2.5-VL, run:
    vllm serve ~/models/Qwen2.5-VL-7B-Instruct --dtype float16 --max-model-len 8192

Then either:
    python run_scenario.py --backend local ...
or set in .env:
    VLLM_BASE_URL=http://localhost:8000/v1
    VLLM_VISION_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
    VLLM_TEXT_MODEL=Qwen/Qwen2.5-VL-7B-Instruct   # can double as text model too
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
    name: str

    @abc.abstractmethod
    def generate_structured(self, system_prompt: str, user_prompt: str, schema: Type[T]) -> T:
        ...

    @abc.abstractmethod
    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        ...


class VisionBackend(abc.ABC):
    name: str

    @abc.abstractmethod
    def verify_frames(self, prompt_text: str, frame_paths: list[str]) -> dict:
        ...


# --------------------------------------------------------------------- #
# OpenAI backend
# --------------------------------------------------------------------- #
class OpenAIBackend(TextBackend, VisionBackend):
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
        return _parse_json_block(response.choices[0].message.content.strip())


def _encode_frame_openai(frame_path: str) -> dict:
    with open(frame_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}}


def _scenario_spec_hint() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "weather", "town", "num_vehicles", "pedestrian", "cyclist",
            "event_type", "duration_s", "spawn_point_index", "notes",
        ],
        "properties": {
            "weather":           {"type": "string", "enum": ["clear", "rain", "fog", "night", "cloudy"]},
            "town":              {"type": "string", "enum": ["Town01", "Town02", "Town03", "Town04", "Town05"]},
            "num_vehicles":      {"type": "integer", "minimum": 0, "maximum": 20},
            "pedestrian":        {"type": "boolean"},
            "cyclist":           {"type": "boolean"},
            "event_type":        {"type": "string", "enum": ["crossing", "lane_change", "emergency_brake",
                                                              "red_light_violation", "obstruction", "overtaking"]},
            "duration_s":        {"type": "integer", "minimum": 8, "maximum": 30},
            "spawn_point_index": {"type": "integer", "minimum": 0, "maximum": 200},
            "notes":             {"type": "string"},
        },
    }


# --------------------------------------------------------------------- #
# HuggingFace cloud backend
# --------------------------------------------------------------------- #
class HFBackend(TextBackend, VisionBackend):
    """
    Uses the HuggingFace Inference API (cloud-hosted models).
    Reads HF_TOKEN first, then HUGGINGFACE_API_TOKEN (legacy).
    """
    name = "huggingface"

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
            "HF_VISION_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct"
        )
        # Accept both HF_TOKEN (standard HF CLI var) and legacy HUGGINGFACE_API_TOKEN
        self.token = (
            token
            or os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACE_API_TOKEN")
        )
        if not self.token:
            raise ValueError(
                "No HuggingFace token found. Set HF_TOKEN in your .env file. "
                "Generate one at https://huggingface.co/settings/tokens"
            )
        # Do NOT bind model= to the client constructor — the HF router
        # resolves provider at call time from the model= in each call.
        self._client = InferenceClient(token=self.token)

    def generate_structured(self, system_prompt: str, user_prompt: str, schema: Type[T]) -> T:
        schema_hint = json.dumps(_scenario_spec_hint() if schema.__name__ == "ScenarioSpec"
                                 else schema.model_json_schema(), indent=2)
        full_prompt = (
            f"{system_prompt}\n\nYou MUST respond with ONLY a single JSON object "
            f"matching this schema (no markdown fences, no prose before or after):\n"
            f"{schema_hint}\n\nUser request: {user_prompt}"
        )
        raw = self._client.chat_completion(
            model=self.text_model,
            messages=[{"role": "user", "content": full_prompt}],
            max_tokens=512,
            temperature=0.2,
        )
        return schema(**_parse_json_block(raw.choices[0].message.content))

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
        content = [{"type": "text", "text": prompt_text}]
        for fp in frame_paths:
            content.append({"type": "image_url", "image_url": {"url": _encode_frame_b64(fp)}})
        raw = self._client.chat_completion(
            model=self.vision_model,
            messages=[{"role": "user", "content": content}],
            max_tokens=600,
            temperature=0.0,
        )
        return _parse_json_block(raw.choices[0].message.content)


# --------------------------------------------------------------------- #
# Local vLLM backend  ← YOUR SETUP
# --------------------------------------------------------------------- #
class LocalVLLMBackend(TextBackend, VisionBackend):
    """
    Points at a locally running vLLM server (OpenAI-compatible API).

    Start your server with:
        vllm serve ~/models/Qwen2.5-VL-7B-Instruct \\
            --dtype float16 \\
            --max-model-len 8192

    Then use --backend local when running run_scenario.py.

    The same Qwen2.5-VL model handles BOTH text generation (Generator)
    and vision verification (Verifier) — it's a vision-language model,
    so it can do both. You can override with env vars if you want a
    separate text-only model:
        VLLM_TEXT_MODEL=<model-name>   # defaults to vision model
        VLLM_BASE_URL=http://localhost:8000/v1
    """
    name = "local"

    def __init__(
        self,
        base_url: Optional[str] = None,
        vision_model: Optional[str] = None,
        text_model: Optional[str] = None,
    ):
        from openai import OpenAI

        self.base_url = base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")

        # The vision model name must match exactly what vLLM reports.
        # vLLM uses the last path component of the model directory by default.
        self.vision_model = (
            vision_model
            or os.getenv("VLLM_VISION_MODEL", "./Qwen2.5-VL-7B-Instruct")
        )
        # Text model defaults to the same vision model (Qwen2.5-VL handles both)
        self.text_model = (
            text_model
            or os.getenv("VLLM_TEXT_MODEL", self.vision_model)
        )

        # vLLM's OpenAI-compatible server does not check the api_key,
        # but the openai client requires a non-empty string.
        self._client = OpenAI(
            base_url=self.base_url,
            api_key="local",        # any non-empty string works
        )
        logger.info(
            "LocalVLLMBackend initialised: base_url=%s vision=%s text=%s",
            self.base_url, self.vision_model, self.text_model,
        )

    def generate_structured(self, system_prompt: str, user_prompt: str, schema: Type[T]) -> T:
        schema_hint = json.dumps(_scenario_spec_hint() if schema.__name__ == "ScenarioSpec"
                                 else schema.model_json_schema(), indent=2)
        full_prompt = (
            f"{system_prompt}\n\nYou MUST respond with ONLY a single JSON object "
            f"matching this schema (no markdown fences, no prose before or after):\n"
            f"{schema_hint}\n\nUser request: {user_prompt}"
        )
        response = self._client.chat.completions.create(
            model=self.text_model,
            messages=[{"role": "user", "content": full_prompt}],
            max_tokens=512,
            temperature=0.2,
        )
        text = response.choices[0].message.content
        return schema(**_parse_json_block(text))

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.text_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=1024,
            temperature=0.4,
        )
        return response.choices[0].message.content

    def verify_frames(self, prompt_text: str, frame_paths: list[str]) -> dict:
        """
        Send frames to the local Qwen2.5-VL model.
        Qwen2.5-VL supports base64 image_url content blocks via the
        OpenAI-compatible API that vLLM exposes.
        We cap at 3 frames to stay well within the 8192-token context window.
        """
        content: list[dict] = [{"type": "text", "text": prompt_text}]
        # On 2GB VRAM, limit to 3 frames max to avoid OOM in the VLM forward pass
        for fp in frame_paths[:3]:
            content.append({
                "type": "image_url",
                "image_url": {"url": _encode_frame_b64(fp)},
            })

        response = self._client.chat.completions.create(
            model=self.vision_model,
            messages=[{"role": "user", "content": content}],
            max_tokens=600,
            temperature=0.0,
        )
        return _parse_json_block(response.choices[0].message.content)


# --------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------- #
def _encode_frame_b64(frame_path: str) -> str:
    """Returns a data URI string suitable for image_url content blocks."""
    with open(frame_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _parse_json_block(raw: str) -> dict:
    """Strips markdown fences and parses JSON, with one repair attempt."""
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


def get_backend(name: str) -> "OpenAIBackend | HFBackend | LocalVLLMBackend":
    """
    Factory. Valid names: 'openai', 'huggingface' / 'hf', 'local' / 'vllm'.
    """
    name = name.lower()
    if name in {"openai", "gpt", "gpt-4o"}:
        return OpenAIBackend()
    if name in {"huggingface", "hf"}:
        return HFBackend()
    if name in {"local", "vllm", "local_vllm"}:
        return LocalVLLMBackend()
    raise ValueError(
        f"Unknown backend '{name}'. Valid options: 'openai', 'huggingface', 'local'."
    )