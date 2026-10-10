"""A local open-weights model behind the same ``messages.parse`` shape as the Anthropic client.

The model reasoner, critic and judges only need one call: a system prompt, one user message and
a pydantic schema, returning a validated object. ``LocalClient`` answers it with llama.cpp on the
CPU, constraining generation to the schema's JSON grammar so a malformed answer cannot be
produced. No network and no API key; the GGUF weights are a file named by an environment
variable (``DISCOVERYLAB_LOCAL_MODEL``, ``DISCOVERYLAB_LOCAL_JUDGE``). See DECISIONS D17.

Needs the optional ``llama-cpp-python`` package (``uv sync --group local``).
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

# Sampling recommended by the model cards (Qwen3 instruct: 0.7 / 0.8 / 20). Repeats differ by seed.
TEMPERATURE = float(os.environ.get("DISCOVERYLAB_LOCAL_TEMPERATURE", "0.7"))
TOP_P, TOP_K = 0.8, 20
# Context per role. The KV cache grows with it (about 0.15 MB per token for these models), and the
# reasoner and judge share one machine: 20480 for both used 13.5 GB of 15 GB (D19).
N_CTX = int(os.environ.get("DISCOVERYLAB_LOCAL_CTX", "16384"))  # longest prompt (25 abstracts) about 8,500 tokens
JUDGE_CTX = 8192
MAX_OUTPUT = 6000  # a grammar-constrained answer that runs this long is looping; fail it loudly

_LOADED: dict[str, Any] = {}


def model_path(role: str = "reasoner") -> Path | None:
    """The configured weights for a role, or None when not configured or missing."""
    var = "DISCOVERYLAB_LOCAL_JUDGE" if role == "judge" else "DISCOVERYLAB_LOCAL_MODEL"
    p = os.environ.get(var, "")
    return Path(p) if p and Path(p).is_file() else None


def missing(role: str = "reasoner") -> str | None:
    """Why a local model cannot run here, or None if it can."""
    var = "DISCOVERYLAB_LOCAL_JUDGE" if role == "judge" else "DISCOVERYLAB_LOCAL_MODEL"
    if model_path(role) is None:
        return f"{var} is not set to a GGUF file"
    try:
        import llama_cpp  # noqa: F401
    except ImportError:
        return "llama-cpp-python is not installed (uv sync --group local)"
    return None


def _load(path: Path, n_ctx: int = N_CTX) -> Any:
    key = str(path.resolve())
    if key not in _LOADED:  # one copy of the weights per process, shared by reasoner and critic
        from llama_cpp import Llama

        threads = int(os.environ.get("DISCOVERYLAB_LOCAL_THREADS") or os.cpu_count() or 4)
        # no mmap: the CPU backend repacks weights for AMX, and a mapped copy would double resident memory
        _LOADED[key] = Llama(
            model_path=key, n_ctx=n_ctx, n_threads=threads, flash_attn=True, use_mmap=False, verbose=False
        )
    return _LOADED[key]


@dataclass
class _Usage:
    input_tokens: int
    output_tokens: int


@dataclass
class _Response:
    parsed_output: Any
    stop_reason: str
    usage: _Usage
    seed: int


class _Messages:
    def __init__(self, client: LocalClient) -> None:
        self._c = client

    def parse(
        self,
        *,
        model: str,
        max_tokens: int,
        system: str,
        messages: list[dict[str, str]],
        output_format: type[BaseModel],
    ) -> _Response:
        llm = _load(self._c.path, self._c.n_ctx)
        seed = self._c.rng.randrange(2**31)
        out = llm.create_chat_completion(
            messages=[{"role": "system", "content": system}, *messages],
            response_format={"type": "json_object", "schema": output_format.model_json_schema()},
            max_tokens=min(max_tokens, MAX_OUTPUT),
            temperature=self._c.temperature,
            top_p=TOP_P,
            top_k=TOP_K,
            seed=seed,
        )
        choice = out["choices"][0]
        usage = _Usage(out["usage"]["prompt_tokens"], out["usage"]["completion_tokens"])
        if choice.get("finish_reason") == "length":
            return _Response(None, "max_tokens", usage, seed)
        try:
            parsed = output_format.model_validate_json(choice["message"]["content"] or "")
        except ValidationError:  # the grammar enforces shape, not every constraint (e.g. ranges)
            parsed = None
        return _Response(parsed, "end_turn", usage, seed)


class LocalClient:
    """Stands in for ``anthropic.Anthropic()`` with a local GGUF model."""

    def __init__(
        self, path: Path, seed: int | None = None, temperature: float = TEMPERATURE, n_ctx: int = N_CTX
    ) -> None:
        self.path, self.n_ctx = path, n_ctx
        self.temperature = temperature
        self.seed = seed if seed is not None else random.SystemRandom().randrange(2**31)
        self.rng = random.Random(self.seed)  # per-call seeds derive from this, logged with the run
        self.messages = _Messages(self)


def model_id(path: Path) -> str:
    return path.name.removesuffix(".gguf")
