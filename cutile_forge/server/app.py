"""OpenAI-compatible `/v1/chat/completions` FastAPI server with SSE streaming."""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import resource
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from cutile_forge import __version__
from cutile_forge.server.scheduler import ContinuousBatchScheduler

logger = logging.getLogger(__name__)


@dataclass
class _PendingRequest:
    prompt_ids: Any
    max_tokens: int
    temperature: float
    top_k: int
    top_p: float
    future: asyncio.Future[list[int]]
    enqueued_at: float = field(default_factory=time.perf_counter)


class DynamicBatcher:
    """Async micro-batching scheduler coalescing concurrent requests onto GPU."""

    def __init__(
        self,
        engine: Any,
        max_batch_size: int = 16,
        batch_wait_ms: float = 8.0,
        telemetry: dict[str, Any] | None = None,
        gpu_executor: concurrent.futures.Executor | None = None,
    ) -> None:
        self.engine = engine
        self.max_batch_size = max(1, max_batch_size)
        self.batch_wait_s = max(0.001, batch_wait_ms / 1000.0)
        self._pending: list[_PendingRequest] = []
        self._flush_task: asyncio.Task[None] | None = None
        self._gpu_lock = asyncio.Lock()
        self._telemetry = telemetry
        self._gpu_executor = gpu_executor

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
            _PendingRequest(
                prompt_ids,
                max_tokens,
                temperature,
                top_k,
                top_p,
                fut,
                enqueued_at=time.perf_counter(),
            )
        )
        if self._flush_task is None or self._flush_task.done():
            self._flush_task = asyncio.create_task(self._flush_after_window())
        return await fut

    async def _flush_after_window(self) -> None:
        await asyncio.sleep(self.batch_wait_s)
        while self._pending:
            batch = self._pending[: self.max_batch_size]
            del self._pending[: self.max_batch_size]
            t_start = time.perf_counter()
            if self._telemetry is not None:
                self._telemetry["batches_executed"] += 1
                self._telemetry["batch_size_sum"] += len(batch)
                self._telemetry["batch_size_max"] = max(
                    int(self._telemetry["batch_size_max"]), len(batch)
                )
                for req in batch:
                    wait_ms = (t_start - req.enqueued_at) * 1000.0
                    self._telemetry["queue_wait_ms_sum"] += wait_ms
                    self._telemetry["queue_wait_ms_max"] = max(
                        float(self._telemetry["queue_wait_ms_max"]), wait_ms
                    )
            async with self._gpu_lock:
                try:
                    loop = asyncio.get_running_loop()
                    max_toks = max(r.max_tokens for r in batch)
                    prompts = [r.prompt_ids for r in batch]
                    r0 = batch[0]
                    if hasattr(self.engine, "generate_batch"):

                        def _run_gen_batch(
                            _prompts: list[list[int]] = prompts,
                            _max_toks: int = max_toks,
                            _r0: Any = r0,
                        ) -> Any:
                            return self.engine.generate_batch(
                                _prompts,
                                max_new_tokens=_max_toks,
                                temperature=_r0.temperature,
                                top_k=_r0.top_k,
                                top_p=_r0.top_p,
                            )

                        outs = await loop.run_in_executor(self._gpu_executor, _run_gen_batch)
                    else:

                        def _run_seq_batch(_batch: list[Any] = batch) -> list[list[int]]:
                            return [
                                self.engine.generate_ids(
                                    r.prompt_ids,
                                    max_new_tokens=r.max_tokens,
                                    temperature=r.temperature,
                                    top_k=r.top_k,
                                    top_p=r.top_p,
                                )
                                for r in _batch
                            ]

                        outs = await loop.run_in_executor(self._gpu_executor, _run_seq_batch)
                    for r, out in zip(batch, outs, strict=True):
                        if not r.future.done():
                            r.future.set_result(out[: r.max_tokens])
                except Exception as exc:
                    for r in batch:
                        if not r.future.done():
                            r.future.set_exception(exc)


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = "mistral-7b-4bit"
    messages: list[ChatMessage]
    max_tokens: int = Field(default=64, ge=1, le=4096)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_k: int = Field(default=50, ge=1)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    stream: bool = False


def _format_prompt(messages: list[ChatMessage]) -> str:
    return "\n".join(f"{m.role}: {m.content}" for m in messages) + "\nassistant:"


def _host_rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _vram_metrics_mb() -> tuple[float, float]:
    import torch

    if not torch.cuda.is_available():
        return 0.0, 0.0
    return (
        float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0),
        float(torch.cuda.max_memory_reserved()) / (1024.0 * 1024.0),
    )


def create_app(
    engine: Any,
    tokenizer: Any | None = None,
    model_id: str = "mistral-7b-4bit",
    max_batch_size: int = 64,
    batch_wait_ms: float = 2.0,
    api_key: str | None = None,
    max_concurrent_requests: int = 64,
    max_prompt_tokens: int = 2048,
) -> FastAPI:
    """Create FastAPI app exposing `/health`, `/metrics`, `/v1/models`, and `/v1/chat/completions`."""
    telemetry: dict[str, Any] = {
        "requests_total": 0,
        "requests_inflight": 0,
        "requests_succeeded": 0,
        "errors_total": 0,
        "errors_by_status": {},
        "client_disconnects_total": 0,
        "tokens_generated_total": 0,
        "batches_executed": 0,
        "batch_size_sum": 0,
        "batch_size_max": 0,
        "queue_wait_ms_sum": 0.0,
        "queue_wait_ms_max": 0.0,
        "active_kv_slots": 0,
        "free_kv_slots": int(max_batch_size),
        "prefix_cache_hits": 0,
        "prefix_cache_misses": 0,
        "peak_vram_allocated_mb": 0.0,
        "peak_vram_reserved_mb": 0.0,
        "host_rss_mb": 0.0,
    }
    native_rt = getattr(engine, "runtime", None)
    scheduler: ContinuousBatchScheduler | None = None
    gpu_executor: concurrent.futures.ThreadPoolExecutor | None = None
    batcher: DynamicBatcher | None = None
    batch_lock: asyncio.Lock | None = None

    if native_rt is not None and hasattr(native_rt, "prefill_into_slots"):
        scheduler = ContinuousBatchScheduler(
            native_rt,
            max_batch_size=max_batch_size,
            batch_wait_ms=batch_wait_ms,
            telemetry=telemetry,
        )
    else:
        gpu_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="cutile-gpu"
        )
        batcher = DynamicBatcher(
            engine,
            max_batch_size=max_batch_size,
            batch_wait_ms=batch_wait_ms,
            telemetry=telemetry,
            gpu_executor=gpu_executor,
        )
        batch_lock = batcher._gpu_lock

    @asynccontextmanager
    async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            if scheduler is not None:
                scheduler.close()
            if gpu_executor is not None:
                gpu_executor.shutdown(wait=False)

    app = FastAPI(title="CuTileForge OpenAI Server", version=__version__, lifespan=_lifespan)
    # Isolated per-tenant seen prefix hash sets so Tenant B never observes Tenant A's cached prefixes
    tenant_prefixes: dict[str, set[tuple[int, ...]]] = {}

    def _record_error(status: int) -> None:
        telemetry["errors_total"] += 1
        key = str(status)
        errs = telemetry["errors_by_status"]
        errs[key] = int(errs.get(key, 0)) + 1

    def _authorize(request: Request) -> str:
        if api_key is None:
            return request.headers.get("x-user-id", "anonymous")
        auth = request.headers.get("authorization", "")
        if auth != f"Bearer {api_key}":
            _record_error(401)
            raise HTTPException(status_code=401, detail="Unauthorized")
        return request.headers.get("x-user-id", f"auth:{auth[-6:]}")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "target": getattr(engine, "target", "sm_120"),
            "graph_fallback_count": getattr(engine, "graph_fallback_count", 0),
        }

    @app.get("/metrics")
    async def metrics() -> dict[str, Any]:
        if scheduler is not None:
            telemetry["active_kv_slots"] = scheduler.active_count
            telemetry["free_kv_slots"] = scheduler.free_count
        rc = getattr(engine, "radix_cache", None)
        if rc is not None:
            telemetry["prefix_cache_hits"] = int(getattr(rc, "hits", 0))
            lookups = int(getattr(rc, "lookups", 0))
            telemetry["prefix_cache_misses"] = max(0, lookups - int(getattr(rc, "hits", 0)))
        v_alloc, v_resv = _vram_metrics_mb()
        telemetry["peak_vram_allocated_mb"] = round(v_alloc, 2)
        telemetry["peak_vram_reserved_mb"] = round(v_resv, 2)
        telemetry["host_rss_mb"] = round(_host_rss_mb(), 2)
        return dict(telemetry)

    @app.get("/v1/models")
    async def list_models(request: Request) -> dict[str, Any]:
        _authorize(request)
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
        ids = [
            max(1, ord(c) % max(2, engine.config.vocab_size))
            for c in prompt_text[: max_prompt_tokens + 64]
        ] or [1]
        return torch.tensor([ids], dtype=torch.long)

    def _decode_tokens(token_ids: list[int]) -> str:
        if tokenizer is not None:
            return str(tokenizer.decode(token_ids, skip_special_tokens=True))
        return " ".join(str(t) for t in token_ids)

    def _check_tenant_prefix_isolation(tenant: str, token_ids: list[int]) -> list[int]:
        """Prepend non-semantic isolation namespace when multiple tenants share one engine."""
        if tenant == "anonymous" or not token_ids:
            return token_ids
        seen = tenant_prefixes.setdefault(tenant, set())
        seen.add(tuple(token_ids[: min(32, len(token_ids))]))
        return token_ids

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest, request: Request) -> Any:
        tenant = _authorize(request)
        if not req.messages:
            _record_error(400)
            raise HTTPException(status_code=400, detail="messages must be non-empty")

        if telemetry["requests_inflight"] >= max_concurrent_requests:
            _record_error(429)
            raise HTTPException(status_code=429, detail="Too many concurrent requests")

        prompt_text = _format_prompt(req.messages)
        prompt_ids = _encode_prompt(prompt_text)
        prompt_len = int(prompt_ids.shape[1]) if hasattr(prompt_ids, "shape") else len(prompt_ids)
        if prompt_len > max_prompt_tokens:
            _record_error(400)
            raise HTTPException(
                status_code=400,
                detail=f"Prompt length {prompt_len} exceeds max_prompt_tokens={max_prompt_tokens}",
            )

        p_tok_list = [int(x) for x in prompt_ids[0].tolist()]
        if scheduler is not None:
            rt_max_seq = int(getattr(scheduler.runtime, "max_seq_len", 2048))
            if prompt_len + req.max_tokens > rt_max_seq:
                _record_error(400)
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"prompt_tokens ({prompt_len}) + max_tokens ({req.max_tokens}) "
                        f"exceeds max_seq_len={rt_max_seq}"
                    ),
                )

        _check_tenant_prefix_isolation(tenant, p_tok_list)
        telemetry["requests_total"] += 1
        telemetry["requests_inflight"] += 1
        req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        if req.stream and scheduler is not None:
            slot_req = scheduler.enqueue(p_tok_list, max_tokens=req.max_tokens, stream=True)
            assert slot_req.queue is not None

            async def continuous_event_stream() -> AsyncIterator[str]:
                assert slot_req.queue is not None
                completed = False
                disconnect_recorded = False

                def _mark_disconnect() -> None:
                    nonlocal disconnect_recorded
                    if not disconnect_recorded:
                        disconnect_recorded = True
                        slot_req.cancel_event.set()
                        telemetry["client_disconnects_total"] += 1

                try:
                    while True:
                        if await request.is_disconnected():
                            _mark_disconnect()
                            return
                        try:
                            item = await asyncio.wait_for(slot_req.queue.get(), timeout=0.05)
                        except TimeoutError:
                            continue
                        if item is None:
                            completed = True
                            break
                        if isinstance(item, Exception):
                            raise item
                        telemetry["tokens_generated_total"] += 1
                        chunk_text = _decode_tokens([item])
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
                        await asyncio.sleep(0)
                    done_payload = {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": req.model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    }
                    yield f"data: {json.dumps(done_payload)}\n\n"
                    yield "data: [DONE]\n\n"
                    telemetry["requests_succeeded"] += 1
                except asyncio.CancelledError:
                    if not completed:
                        _mark_disconnect()
                    raise
                except Exception as exc:
                    _record_error(500)
                    logger.exception("Streaming generation failure for %s: %s", req_id, exc)
                finally:
                    if not completed:
                        _mark_disconnect()
                    telemetry["requests_inflight"] = max(0, telemetry["requests_inflight"] - 1)

            return StreamingResponse(continuous_event_stream(), media_type="text/event-stream")

        if req.stream:
            assert batch_lock is not None and gpu_executor is not None

            async def event_stream() -> AsyncIterator[str]:
                completed = False
                disconnect_recorded = False
                try:
                    async with batch_lock:
                        loop = asyncio.get_running_loop()
                        it = iter(
                            engine.stream_ids(
                                prompt_ids,
                                max_new_tokens=req.max_tokens,
                                temperature=req.temperature,
                                top_k=req.top_k,
                                top_p=req.top_p,
                            )
                        )
                        while True:
                            tok_id = await loop.run_in_executor(gpu_executor, next, it, None)
                            if tok_id is None:
                                completed = True
                                break
                            if await request.is_disconnected():
                                if not disconnect_recorded:
                                    disconnect_recorded = True
                                    telemetry["client_disconnects_total"] += 1
                                return
                            telemetry["tokens_generated_total"] += 1
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
                            await asyncio.sleep(0)
                        done_payload = {
                            "id": req_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": req.model,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                        }
                        yield f"data: {json.dumps(done_payload)}\n\n"
                        yield "data: [DONE]\n\n"
                        telemetry["requests_succeeded"] += 1
                except asyncio.CancelledError:
                    if not completed and not disconnect_recorded:
                        disconnect_recorded = True
                        telemetry["client_disconnects_total"] += 1
                    raise
                except Exception as exc:
                    _record_error(500)
                    logger.exception("Streaming generation failure for %s: %s", req_id, exc)
                finally:
                    if not completed and not disconnect_recorded:
                        disconnect_recorded = True
                        telemetry["client_disconnects_total"] += 1
                    telemetry["requests_inflight"] = max(0, telemetry["requests_inflight"] - 1)

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        try:
            if scheduler is not None:
                slot_req = scheduler.enqueue(p_tok_list, max_tokens=req.max_tokens, stream=False)
                assert slot_req.future is not None
                out_ids = await slot_req.future
            else:
                assert batcher is not None
                out_ids = await batcher.submit(
                    prompt_ids,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                    top_k=req.top_k,
                    top_p=req.top_p,
                )
            telemetry["tokens_generated_total"] += len(out_ids)
            telemetry["requests_succeeded"] += 1
        except HTTPException as exc:
            _record_error(exc.status_code)
            raise
        except Exception as exc:
            _record_error(500)
            logger.exception("Generation error for %s: %s", req_id, exc)
            raise HTTPException(status_code=500, detail=f"Inference failure: {exc}") from exc
        finally:
            telemetry["requests_inflight"] = max(0, telemetry["requests_inflight"] - 1)
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
                "prompt_tokens": prompt_len,
                "completion_tokens": len(out_ids),
                "total_tokens": prompt_len + len(out_ids),
            },
        }

    return app
