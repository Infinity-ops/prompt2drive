"""
backends/llm_backend.py

Provider-agnostic backend abstraction for Prompt-to-Drive.
Dynamic version: Completely driven by model strings and parameters specified in your .env file.
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

    def __init__(self, text_model: Optional[str] = None, vision_model: Optional[str] = None):
        from openai import OpenAI
        
        self.text_model = text_model or os.getenv("OPENAI_TEXT_MODEL")
        self.vision_model = vision_model or os.getenv("OPENAI_VISION_MODEL")
        
        if not self.text_model:
            raise ValueError("OPENAI_TEXT_MODEL is missing. Provide it via initialization or set it in your .env file.")
        if not self.vision_model:
            raise ValueError("OPENAI_VISION_MODEL is missing. Provide it via initialization or set it in your .env file.")
        
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL")
        
        if not api_key:
            logger.warning("OPENAI_API_KEY not found in environment variables.")

        self.client = OpenAI(api_key=api_key, base_url=base_url)

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
    name = "huggingface"

    def __init__(
        self,
        text_model: Optional[str] = None,
        vision_model: Optional[str] = None,
        token: Optional[str] = None,
    ):
        from huggingface_hub import InferenceClient

        self.text_model = text_model or os.getenv("HF_TEXT_MODEL")
        self.vision_model = vision_model or os.getenv("HF_VISION_MODEL")
        
        if not self.text_model:
            raise ValueError("HF_TEXT_MODEL is missing. Set it in your .env file (e.g., HF_TEXT_MODEL=\"Qwen/Qwen2.5-7B-Instruct\").")
        if not self.vision_model:
            raise ValueError("HF_VISION_MODEL is missing. Set it in your .env file (e.g., HF_VISION_MODEL=\"Qwen/Qwen2.5-VL-7B-Instruct\").")
        
        self.token = token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_TOKEN")
        if not self.token:
            raise ValueError("No HuggingFace token found. Please define HF_TOKEN in your .env file.")
            
        self._client = InferenceClient(token=self.token)
        
        # Route to local server if hybrid system split is explicitly active
        self.local_vision_url = os.getenv("P2D_VISION_URL")
        if self.local_vision_url:
            from openai import OpenAI
            self._local_vision_client = OpenAI(base_url=self.local_vision_url, api_key="local")
            logger.info("HFBackend routing vision frames locally to: %s", self.local_vision_url)
        else:
            self._local_vision_client = None

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
        
        if self._local_vision_client is not None:
            for fp in frame_paths[:3]:
                content.append({"type": "image_url", "image_url": {"url": _encode_frame_b64(fp)}})
            raw = self._local_vision_client.chat.completions.create(
                model=self.vision_model,
                messages=[{"role": "user", "content": content}],
                max_tokens=600,
                temperature=0.0,
            )
            return _parse_json_block(raw.choices[0].message.content)
            
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
# Local vLLM backend
# --------------------------------------------------------------------- #
class LocalVLLMBackend(TextBackend, VisionBackend):
    name = "local"

    def __init__(
        self,
        base_url: Optional[str] = None,
        vision_model: Optional[str] = None,
        text_model: Optional[str] = None,
    ):
        from openai import OpenAI

        self.base_url = base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        self.vision_model = vision_model or os.getenv("VLLM_VISION_MODEL")
        self.text_model = text_model or os.getenv("VLLM_TEXT_MODEL") or self.vision_model

        if not self.vision_model:
            raise ValueError("VLLM_VISION_MODEL is missing. Set it to your local vLLM model name inside your .env file.")
        if not self.text_model:
            raise ValueError("VLLM_TEXT_MODEL is missing. Set it inside your .env file.")

        self._client = OpenAI(base_url=self.base_url, api_key="local")
        logger.info("LocalVLLMBackend connected: url=%s vision=%s text=%s", self.base_url, self.vision_model, self.text_model)

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
        return schema(**_parse_json_block(response.choices[0].message.content))

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
        content: list[dict] = [{"type": "text", "text": prompt_text}]
        for fp in frame_paths[:3]:
            content.append({"type": "image_url", "image_url": {"url": _encode_frame_b64(fp)}})

        response = self._client.chat.completions.create(
            model=self.vision_model,
            messages=[{"role": "user", "content": content}],
            max_tokens=600,
            temperature=0.0,
        )
        return _parse_json_block(response.choices[0].message.content)


# --------------------------------------------------------------------- #
# Shared helpers & Factories
# --------------------------------------------------------------------- #
def _encode_frame_b64(frame_path: str) -> str:
    with open(frame_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _parse_json_block(raw: str) -> dict:
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
        logger.error("Failed to parse JSON block: %s", cleaned[:300])
        raise


def get_text_backend(name: str) -> TextBackend:
    name = name.lower()
    if name in {"openai", "gpt", "gpt-4o"}: return OpenAIBackend()
    if name in {"huggingface", "hf"}: return HFBackend()
    if name in {"local", "vllm"}: return LocalVLLMBackend()
    raise ValueError(f"Invalid text backend '{name}'")


def get_vision_backend(name: str) -> VisionBackend:
    name = name.lower()
    if name in {"openai", "gpt", "gpt-4o"}: return OpenAIBackend()
    if name in {"huggingface", "hf"}: return HFBackend()
    if name in {"local", "vllm"}: return LocalVLLMBackend()
    raise ValueError(f"Invalid vision backend '{name}'")


def get_backend(name: str) -> OpenAIBackend | HFBackend | LocalVLLMBackend:
    """Kept intact for perfect legacy backwards-compatibility."""
    name = name.lower()
    if name in {"openai", "gpt"}: return OpenAIBackend()
    if name in {"huggingface", "hf"}: return HFBackend()
    if name in {"local", "vllm"}: return LocalVLLMBackend()
    raise ValueError(f"Invalid single backend setup requested: '{name}'")