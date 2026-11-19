"""OpenAI-compatible `/v1/chat/completions` FastAPI server with SSE streaming."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from cutile_forge import __version__


@dataclass
class _PendingRequest:
    prompt_ids: Any
    max_tokens: int
    temperature: float
    top_k: int
    top_p: float
    future: asyncio.Future[list[int]]


class DynamicBatcher:
    """Async micro-batching scheduler coalescing concurrent requests onto GPU."""

    def __init__(
        self,
        engine: Any,
        max_batch_size: int = 16,
        batch_wait_ms: float = 8.0,
    ) -> None:
        self.engine = engine
        self.max_batch_size = max(1, max_batch_size)
        self.batch_wait_s = max(0.001, batch_wait_ms / 1000.0)
        self._pending: list[_PendingRequest] = []
        self._flush_task: asyncio.Task[None] | None = None
        self._gpu_lock = asyncio.Lock()

    async def submit(
        self,
        prompt_ids: Any,
        max_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> list[int]:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[list[int]] = loop.create_future()
        self._pending.append(
            _PendingRequest(prompt_ids, max_tokens, temperature, top_k, top_p, fut)
        )
        if self._flush_task is None or self._flush_task.done():
            self._flush_task = asyncio.create_task(self._flush_after_window())
        return await fut

    async def _flush_after_window(self) -> None:
        await asyncio.sleep(self.batch_wait_s)
        while self._pending:
            batch = self._pending[: self.max_batch_size]
            del self._pending[: self.max_batch_size]
            async with self._gpu_lock:
                try:
                    max_toks = max(r.max_tokens for r in batch)
                    prompts = [r.prompt_ids for r in batch]
                    r0 = batch[0]
                    if hasattr(self.engine, "generate_batch"):
                        outs = self.engine.generate_batch(
                            prompts,
                            max_new_tokens=max_toks,
                            temperature=r0.temperature,
                            top_k=r0.top_k,
                            top_p=r0.top_p,
                        )
                    else:
                        outs = [
                            self.engine.generate_ids(
                                r.prompt_ids,
                                max_new_tokens=r.max_tokens,
                                temperature=r.temperature,
                                top_k=r.top_k,
                                top_p=r.top_p,
                            )
                            for r in batch
                        ]
                    for r, out in zip(batch, outs, strict=True):
                        if not r.future.done():
                            r.future.set_result(out[: r.max_tokens])
                except Exception as exc:
                    for r in batch:
                        if not r.future.done():
                            r.future.set_exception(exc)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "mistral-7b-4bit"
    messages: list[ChatMessage]
    max_tokens: int = Field(default=64, ge=1, le=4096)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_k: int = Field(default=50, ge=1)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    stream: bool = False


def _format_prompt(messages: list[ChatMessage]) -> str:
    return "\n".join(f"{m.role}: {m.content}" for m in messages) + "\nassistant:"


def create_app(
    engine: Any,
    tokenizer: Any | None = None,
    model_id: str = "mistral-7b-4bit",
    max_batch_size: int = 16,
    batch_wait_ms: float = 8.0,
) -> FastAPI:
    """Create FastAPI app exposing `/health`, `/v1/models`, and `/v1/chat/completions`."""
    app = FastAPI(title="CuTileForge OpenAI Server", version=__version__)
    batcher = DynamicBatcher(engine, max_batch_size=max_batch_size, batch_wait_ms=batch_wait_ms)
    batch_lock = batcher._gpu_lock

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "target": getattr(engine, "target", "sm_120"),
        }

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "created": 1700000000,
                    "owned_by": "cutile-forge",
                }
            ],
        }

    def _encode_prompt(prompt_text: str) -> Any:
        import torch

        if tokenizer is not None:
            return tokenizer.encode(prompt_text, return_tensors="pt")
        ids = [max(1, ord(c) % max(2, engine.config.vocab_size)) for c in prompt_text[:64]] or [1]
        return torch.tensor([ids], dtype=torch.long)

    def _decode_tokens(token_ids: list[int]) -> str:
        if tokenizer is not None:
            return str(tokenizer.decode(token_ids, skip_special_tokens=True))
        return " ".join(str(t) for t in token_ids)

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest) -> Any:
        prompt_text = _format_prompt(req.messages)
        prompt_ids = _encode_prompt(prompt_text)
        req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        if req.stream:

            async def event_stream() -> AsyncIterator[str]:
                async with batch_lock:
                    for tok_id in engine.stream_ids(
                        prompt_ids,
                        max_new_tokens=req.max_tokens,
                        temperature=req.temperature,
                        top_k=req.top_k,
                        top_p=req.top_p,
                    ):
                        chunk_text = _decode_tokens([tok_id])
                        payload = {
                            "id": req_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": req.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": chunk_text},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(payload)}\n\n"
                    done_payload = {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": req.model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    }
                    yield f"data: {json.dumps(done_payload)}\n\n"
                    yield "data: [DONE]\n\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        out_ids = await batcher.submit(
            prompt_ids,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
        )
        text = _decode_tokens(out_ids)
        return {
            "id": req_id,
            "object": "chat.completion",
            "created": created,
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": int(prompt_ids.shape[1]),
                "completion_tokens": len(out_ids),
                "total_tokens": int(prompt_ids.shape[1]) + len(out_ids),
            },
        }

    return app
