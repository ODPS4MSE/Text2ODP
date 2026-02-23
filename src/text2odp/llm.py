from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod

import requests


class LLMBackend(ABC):
    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.2, max_tokens: int = 1024) -> str:
        raise NotImplementedError


class OllamaBackend(LLMBackend):
    def __init__(self, model: str = "llama3.1:8b", endpoint: str | None = None) -> None:
        self.model = model
        self.endpoint = endpoint or os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")

    def generate(self, prompt: str, temperature: float = 0.2, max_tokens: int = 1024) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        response = requests.post(f"{self.endpoint}/api/generate", json=payload, timeout=180)
        response.raise_for_status()
        return response.json().get("response", "")


class TransformersBackend(LLMBackend):
    def __init__(self, model: str = "mistralai/Mistral-7B-Instruct-v0.3") -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, device_map="auto")
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    def generate(self, prompt: str, temperature: float = 0.2, max_tokens: int = 1024) -> str:
        result = self.pipe(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            return_full_text=False,
        )
        return result[0]["generated_text"]


class JSONConstrainedMixin:
    @staticmethod
    def parse_json_or_raise(text: str) -> dict:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError("No JSON object found in model output")
        return json.loads(text[start : end + 1])
