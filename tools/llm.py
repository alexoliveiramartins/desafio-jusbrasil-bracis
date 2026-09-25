"""Cliente mínimo para um servidor LLM local compatível com a API OpenAI (Ollama, llama.cpp).

Usado só pelas ferramentas de desenvolvimento (tools.synth_llm), nunca pelo
pipeline. Recusa hosts que não sejam locais, decodifica com temperature=0 e
seed fixa, e guarda as respostas em cache indexado pelo pedido completo.
"""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


class LLMError(RuntimeError):
    pass


class ChatClient:
    def __init__(self, base_url: str, model: str, *, seed: int = 42, max_tokens: int = 1024,
                 timeout_s: float = 120.0, cache_dir: Path | None = None):
        if urlparse(base_url).hostname not in LOCAL_HOSTS:
            raise LLMError(f"servidor LLM precisa ser local: {base_url}")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.seed = seed
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def chat(self, system: str, user: str, *, json_mode: bool = False, temperature: float = 0.0) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature,
            "seed": self.seed,
            "max_tokens": self.max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        body = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        cache_file = self.cache_dir / f"{hashlib.sha256(body).hexdigest()}.txt" if self.cache_dir else None
        if cache_file and cache_file.exists():
            return cache_file.read_text(encoding="utf-8")
        request = urllib.request.Request(f"{self.base_url}/chat/completions", data=body,
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                content = json.loads(response.read().decode("utf-8"))["choices"][0]["message"]["content"]
        except (urllib.error.URLError, TimeoutError) as error:
            raise LLMError(f"falha ao consultar {self.base_url}: {error}") from error
        if cache_file:
            cache_file.write_text(content, encoding="utf-8")
        return content
