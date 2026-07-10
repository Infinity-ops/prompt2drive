"""
backends/llm_backend.py
Provider-agnostic backend abstraction for Prompt-to-Drive.
Fixed to correctly route OpenAI to Cloud APIs and handle Llama 3.2 Vision constraints.
"""
from __future__ import annotations

import abc
import base64
import json
import logging
import os
import re
from typing import Optional, Type, TypeVar
from pathlib import Path

from pydantic import BaseModel

logger = logging.getLogger("prompt_to_drive.backends")
T = TypeVar("T", bound=BaseModel)

class TextBackend(abc.ABC):
    name: str
    @abc.abstractmethod
    def generate_structured(self, system_prompt: str, user_prompt: str, schema: Type[T]) -> T: ...
    @abc.abstractmethod
    def generate_text(self, system_prompt: str, user_prompt: str) -> str: ...

class VisionBackend(abc.ABC):
    name: str
    @abc.abstractmethod
    def verify_frames(self, prompt_text: str, frame_paths: list[str]) -> dict: ...


def _format_llama_prompt_if_needed(model_name: str, prompt_text: str, num_images: int) -> str:
    """Helper to dynamically add required image positional tags for Llama 3.2 vLLM setups."""
    if "llama" in model_name.lower():
        placeholders = "<|image|>" * num_images
        return f"{placeholders}\n{prompt_text}"
    return prompt_text


# --------------------------------------------------------------------- #
# OpenAI backend (Fixed to route dynamically based on env configuration)
# --------------------------------------------------------------------- #
class OpenAIBackend(TextBackend, VisionBackend):
    name = "openai"

    def __init__(self, text_model: Optional[str] = None, vision_model: Optional[str] = None):
        from openai import OpenAI
        
        # Pull directly from .env variables instead of hardcoding localhost
        self.text_model = text_model or os.getenv("OPENAI_TEXT_MODEL", "gpt-4o")
        self.vision_model = vision_model or os.getenv("OPENAI_VISION_MODEL", "gpt-4o")
        
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_API_BASE", None) # Default to None lets it reach official OpenAI Cloud endpoints
        
        if api_key == "dummy" or not api_key:
            logger.warning("Using fallback dummy credentials for OpenAI backend.")
            
        logger.info(f"OpenAIBackend tracking -> Model: {self.text_model} Cloud Endpoint")
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def generate_structured(self, system_prompt: str, user_prompt: str, schema: Type[T]) -> T:
        format_instruction = "\n\nCRITICAL: You must return ONLY a valid JSON object matching the requested data structure. Do not wrap code in markdown block formatting."
        full_prompt = f"{system_prompt}\n\nUser request: {user_prompt}{format_instruction}"
        
        response = self._client.chat.completions.create(
            model=self.text_model,
            messages=[{"role": "user", "content": full_prompt}],
            max_tokens=1024,
            temperature=0.1,
        )
        return schema(**_parse_json_block(response.choices[0].message.content))

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.text_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
        )
        return response.choices[0].message.content

    def verify_frames(self, prompt_text: str, frame_paths: list[str]) -> dict:
        selected_frames = list(frame_paths)[:3]
        format_instruction = "\n\nRespond strictly with a raw JSON block containing metrics validation keys."
        
        content = [{"type": "text", "text": prompt_text + format_instruction}]
        for fp in selected_frames:
            content.append({"type": "image_url", "image_url": {"url": _encode_frame_b64(fp)}})
            
        response = self._client.chat.completions.create(
            model=self.vision_model,
            messages=[{"role": "user", "content": content}],
            max_tokens=600,
            temperature=0.0,
        )
        return _parse_json_block(response.choices[0].message.content)


# --------------------------------------------------------------------- #
# HuggingFace cloud backend
# --------------------------------------------------------------------- #
class HFBackend(TextBackend, VisionBackend):
    name = "huggingface"

    def __init__(self, text_model: Optional[str] = None, vision_model: Optional[str] = None, token: Optional[str] = None):
        from huggingface_hub import InferenceClient
        from openai import OpenAI
        
        self.text_model = text_model or os.getenv("HF_TEXT_MODEL", "Qwen/Qwen2.5-7B-Instruct")
        self.vision_model = vision_model or os.getenv("VLLM_VISION_MODEL") or "./Llama-3.2-11B-Vision-Instruct"
        self.token = token or os.getenv("HF_TOKEN")
        self._client = InferenceClient(token=self.token)
        
        raw_url = os.getenv("VLLM_BASE_URL") or "http://127.0.0.1:8000/v1"
        self.local_vision_url = raw_url.strip("'\"")
        self._local_vision_client = OpenAI(base_url=self.local_vision_url, api_key="local")

    def generate_structured(self, system_prompt: str, user_prompt: str, schema: Type[T]) -> T:
        format_instruction = "\n\nReturn output inside standard JSON formatting rules."
        full_prompt = f"{system_prompt}\n\nUser request: {user_prompt}{format_instruction}"
        raw = self._client.chat_completion(
            model=self.text_model, messages=[{"role": "user", "content": full_prompt}], max_tokens=1024, temperature=0.2
        )
        return schema(**_parse_json_block(raw.choices[0].message.content))

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        raw = self._client.chat_completion(
            model=self.text_model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            max_tokens=1024,
            temperature=0.4,
        )
        return raw.choices[0].message.content

    def verify_frames(self, prompt_text: str, frame_paths: list[str]) -> dict:
        # Enforce exactly 1 image slice if utilizing a local llama vision backbone to satisfy constraint parameters
        is_llama = "llama" in self.vision_model.lower()
        selected_frames = list(frame_paths)[-1:] if is_llama else list(frame_paths)[:3]
        
        adjusted_prompt = _format_llama_prompt_if_needed(self.vision_model, prompt_text, len(selected_frames))
        content = [{"type": "text", "text": adjusted_prompt}]
        
        for fp in selected_frames:
            content.append({"type": "image_url", "image_url": {"url": _encode_frame_b64(fp)}})
        
        raw = self._local_vision_client.chat.completions.create(
            model=self.vision_model, 
            messages=[{"role": "user", "content": content}], 
            max_tokens=600, 
            temperature=0.0
        )
        return _parse_json_block(raw.choices[0].message.content)


# --------------------------------------------------------------------- #
# Local vLLM backend
# --------------------------------------------------------------------- #
class LocalVLLMBackend(TextBackend, VisionBackend):
    name = "local"

    def __init__(self, base_url: Optional[str] = None, vision_model: Optional[str] = None, text_model: Optional[str] = None):
        from openai import OpenAI
        
        raw_url = base_url or os.getenv("VLLM_BASE_URL") or "http://127.0.0.1:8000/v1"
        self.base_url = raw_url.strip("'\"")
        
        self.vision_model = vision_model or os.getenv("VLLM_VISION_MODEL", "./Llama-3.2-11B-Vision-Instruct")
        self.text_model = text_model or os.getenv("VLLM_TEXT_MODEL") or self.vision_model
        
        self._client = OpenAI(base_url=self.base_url, api_key="local")
        logger.info("LocalVLLMBackend tracking: url=%s vision=%s text=%s", self.base_url, self.vision_model, self.text_model)

    def generate_structured(self, system_prompt: str, user_prompt: str, schema: Type[T]) -> T:
        format_instruction = "\n\nYou must return a valid JSON data structure structure matching schema requirements."
        full_prompt = f"{system_prompt}\n\nUser request: {user_prompt}{format_instruction}"
        response = self._client.chat.completions.create(
            model=self.text_model, messages=[{"role": "user", "content": full_prompt}], max_tokens=1024, temperature=0.1
        )
        return schema(**_parse_json_block(response.choices[0].message.content))

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.text_model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            max_tokens=1024,
            temperature=0.4,
        )
        return response.choices[0].message.content

    def verify_frames(self, prompt_text: str, frame_paths: list[str]) -> dict:
        # Enforce exactly 1 image slice if utilizing a local llama vision backbone to satisfy constraint parameters
        is_llama = "llama" in self.vision_model.lower()
        selected_frames = list(frame_paths)[-1:] if is_llama else list(frame_paths)[:3]
        
        adjusted_prompt = _format_llama_prompt_if_needed(self.vision_model, prompt_text, len(selected_frames))
        content = [{"type": "text", "text": adjusted_prompt}]
        
        for fp in selected_frames:
            content.append({"type": "image_url", "image_url": {"url": _encode_frame_b64(fp)}})

        response = self._client.chat.completions.create(
            model=self.vision_model, messages=[{"role": "user", "content": content}], max_tokens=600, temperature=0.0
        )
        return _parse_json_block(response.choices[0].message.content)


# --------------------------------------------------------------------- #
# Shared Helpers & Factories
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
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        logger.warning(f"Failed to decode text directly into JSON block structure. Raw stream: {raw[:100]}")
        return {"weather_correct": False, "pedestrian_present": False, "vehicles_present": False, "event_observable": False}

def get_text_backend(name: str) -> TextBackend:
    name = name.lower()
    if name in {"huggingface", "hf"}: return HFBackend()
    if name in {"local", "vllm"}: return LocalVLLMBackend()
    return OpenAIBackend()

def get_vision_backend(name: str) -> VisionBackend:
    name = name.lower()
    if name in {"local", "vllm"}: return LocalVLLMBackend()
    if name in {"huggingface", "hf"}: return HFBackend()
    return OpenAIBackend()

def get_backend(name: str) -> OpenAIBackend | HFBackend | LocalVLLMBackend:
    name = name.lower()
    if name in {"openai", "gpt"}: return OpenAIBackend()
    if name in {"huggingface", "hf"}: return HFBackend()
    if name in {"local", "vllm"}: return LocalVLLMBackend()
    raise ValueError(f"Invalid single backend setup requested: '{name}'")