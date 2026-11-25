"""Continuous micro-batching scheduler unifying streaming SSE & non-streaming HTTP inference."""

from __future__ import annotations

import asyncio
import heapq
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class ScheduledRequest:
    """Single queued/active generation request tracked across asyncio and GPU worker thread."""

    prompt_ids: list[int]
    max_tokens: int
    loop: asyncio.AbstractEventLoop
    queue: asyncio.Queue[int | None | Exception] | None
    future: asyncio.Future[list[int]] | None = None
    stream: bool = True
    tokens_acc: list[int] = field(default_factory=list)
    cancel_event: threading.Event = field(default_factory=threading.Event)
    enqueued_at: float = field(default_factory=time.perf_counter)


@dataclass
class _ActiveSlot:
    slot_id: int
    req: ScheduledRequest
    last_tok: int
    pos: int
    generated: int


class ContinuousBatchScheduler:
    """Continuous-batching prefill+decode scheduler over `NativeModelRuntime` KV slots."""

    def __init__(
        self,
        runtime: Any,
        max_batch_size: int = 64,
        batch_wait_ms: float = 1.5,
        telemetry: dict[str, Any] | None = None,
    ) -> None:
        self.runtime = runtime
        self.max_batch_size = max(1, int(max_batch_size))
        self.batch_wait_s = max(0.0, float(batch_wait_ms) / 1000.0)
        self._telemetry = telemetry

        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._pending: deque[ScheduledRequest] = deque()
        self._free_slots: list[int] = list(range(self.max_batch_size))
        heapq.heapify(self._free_slots)
        self._active: dict[int, _ActiveSlot] = {}
        self._stop = False
        self._gpu_inputs_dirty = True

        if self._telemetry is not None:
            self._telemetry.setdefault("active_kv_slots", 0)
            self._telemetry.setdefault("free_kv_slots", self.max_batch_size)

        self._thread = threading.Thread(
            target=self._worker_loop, name="cutile-continuous-batcher", daemon=True
        )
        self._thread.start()

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._active)

    @property
    def free_count(self) -> int:
        with self._lock:
            return len(self._free_slots)

    def enqueue(
        self, prompt_ids: list[int], max_tokens: int, stream: bool = True
    ) -> ScheduledRequest:
        """Enqueue a prompt onto the continuous batching GPU loop."""
        loop = asyncio.get_running_loop()
        q: asyncio.Queue[int | None | Exception] | None = asyncio.Queue() if stream else None
        fut: asyncio.Future[list[int]] | None = None if stream else loop.create_future()
        req = ScheduledRequest(
            prompt_ids=[int(x) for x in prompt_ids],
            max_tokens=max(1, int(max_tokens)),
            loop=loop,
            queue=q,
            future=fut,
            stream=stream,
        )
        with self._lock:
            self._pending.append(req)
            self._wake.notify_all()
        return req

    def close(self, timeout_s: float = 2.0) -> None:
        """Signal worker thread shutdown and wait briefly for drain."""
        with self._lock:
            self._stop = True
            self._wake.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=timeout_s)

    @staticmethod
    def _deliver_token(req: ScheduledRequest, tok: int) -> None:
        if not req.stream:
            req.tokens_acc.append(tok)
            return
        q = req.queue
        assert q is not None
        try:
            if not req.loop.is_closed():
                req.loop.call_soon_threadsafe(q.put_nowait, tok)
        except RuntimeError:
            pass

    @staticmethod
    def _deliver_done(req: ScheduledRequest) -> None:
        if not req.stream:
            fut = req.future
            if fut is not None and not req.loop.is_closed():
                result = list(req.tokens_acc)

                def _resolve() -> None:
                    if not fut.done():
                        fut.set_result(result)

                try:
                    req.loop.call_soon_threadsafe(_resolve)
                except RuntimeError:
                    pass
            return
        q = req.queue
        assert q is not None
        try:
            if not req.loop.is_closed():
                req.loop.call_soon_threadsafe(q.put_nowait, None)
        except RuntimeError:
            pass

    @staticmethod
    def _deliver_error(req: ScheduledRequest, exc: Exception) -> None:
        if not req.stream:
            fut = req.future
            if fut is not None and not req.loop.is_closed():

                def _fail() -> None:
                    if not fut.done():
                        fut.set_exception(exc)

                try:
                    req.loop.call_soon_threadsafe(_fail)
                except RuntimeError:
                    pass
            return
        q = req.queue
        assert q is not None
        try:
            if not req.loop.is_closed():
                req.loop.call_soon_threadsafe(q.put_nowait, exc)
        except RuntimeError:
            pass

    def _release_slot_locked(self, slot_id: int) -> None:
        self._active.pop(slot_id, None)
        if slot_id not in self._free_slots:
            heapq.heappush(self._free_slots, slot_id)
        if hasattr(self.runtime, "clear_slot"):
            self.runtime.clear_slot(slot_id)
        self._gpu_inputs_dirty = True
        self._sync_slot_metrics_locked()

    def _sync_slot_metrics_locked(self) -> None:
        if self._telemetry is not None:
            self._telemetry["active_kv_slots"] = len(self._active)
            self._telemetry["free_kv_slots"] = len(self._free_slots)

    def _worker_loop(self) -> None:
        while True:
            admit: list[tuple[int, ScheduledRequest]] = []
            decode_list: list[_ActiveSlot] = []

            with self._lock:
                if self._stop and not self._pending and not self._active:
                    return

                # Retire cancelled active requests immediately before touching GPU
                for sid in list(self._active.keys()):
                    slot_obj = self._active[sid]
                    if slot_obj.req.cancel_event.is_set():
                        self._release_slot_locked(sid)

                if not self._active and not self._pending:
                    self._sync_slot_metrics_locked()
                    self._wake.wait(timeout=0.05)
                    continue

                if not self._active and self._pending and self.batch_wait_s > 0.0:
                    deadline = time.perf_counter() + self.batch_wait_s
                    max_deadline = time.perf_counter() + max(self.batch_wait_s, 0.015)
                    prev_pending = len(self._pending)
                    while len(self._pending) < len(self._free_slots):
                        rem = deadline - time.perf_counter()
                        if rem <= 0.0 or self._stop:
                            break
                        self._wake.wait(timeout=rem)
                        if len(self._pending) > prev_pending:
                            prev_pending = len(self._pending)
                            deadline = min(max_deadline, time.perf_counter() + 0.002)

                while self._pending and self._free_slots:
                    req = self._pending.popleft()
                    if req.cancel_event.is_set():
                        continue
                    sid = heapq.heappop(self._free_slots)
                    admit.append((sid, req))

                decode_list = list(self._active.values())

            # 1. Execute prefill for newly admitted requests grouped by exact prompt length
            if admit:
                t_admit = time.perf_counter()
                if self._telemetry is not None:
                    self._telemetry["batches_executed"] += 1
                    self._telemetry["batch_size_sum"] += len(admit)
                    self._telemetry["batch_size_max"] = max(
                        int(self._telemetry["batch_size_max"]), len(admit)
                    )
                    for _, req in admit:
                        wait_ms = (t_admit - req.enqueued_at) * 1000.0
                        self._telemetry["queue_wait_ms_sum"] += wait_ms
                        self._telemetry["queue_wait_ms_max"] = max(
                            float(self._telemetry["queue_wait_ms_max"]), wait_ms
                        )

                by_len: dict[int, list[tuple[int, ScheduledRequest]]] = {}
                for sid, req in admit:
                    by_len.setdefault(len(req.prompt_ids), []).append((sid, req))

                new_actives: list[_ActiveSlot] = []
                pending_deliver: list[tuple[int, ScheduledRequest, int]] = []
                try:
                    nxt_groups: list[tuple[list[tuple[int, ScheduledRequest]], torch.Tensor]] = []
                    for _, grp in by_len.items():
                        slot_ids = [sid for sid, _ in grp]
                        inp_t = torch.tensor(
                            [req.prompt_ids for _, req in grp],
                            dtype=torch.long,
                            device=self.runtime.device,
                        )
                        nxt = self.runtime.prefill_into_slots(slot_ids, inp_t)
                        nxt_groups.append((grp, nxt.view(len(grp))))
                    if len(nxt_groups) == 1:
                        grp0, t0 = nxt_groups[0]
                        for (sid, req), t_val in zip(grp0, t0.cpu().tolist(), strict=True):
                            pending_deliver.append((sid, req, int(t_val)))
                    elif nxt_groups:
                        all_toks = torch.cat([t for _, t in nxt_groups], dim=0).cpu().tolist()
                        offset = 0
                        for grp, _ in nxt_groups:
                            g_len = len(grp)
                            for (sid, req), t_val in zip(
                                grp, all_toks[offset : offset + g_len], strict=True
                            ):
                                pending_deliver.append((sid, req, int(t_val)))
                            offset += g_len

                    for sid, req, tok0 in pending_deliver:
                        if req.cancel_event.is_set():
                            with self._lock:
                                if hasattr(self.runtime, "clear_slot"):
                                    self.runtime.clear_slot(sid)
                                heapq.heappush(self._free_slots, sid)
                            continue
                        self._deliver_token(req, tok0)
                        if req.max_tokens <= 1:
                            self._deliver_done(req)
                            with self._lock:
                                if hasattr(self.runtime, "clear_slot"):
                                    self.runtime.clear_slot(sid)
                                heapq.heappush(self._free_slots, sid)
                        else:
                            new_actives.append(
                                _ActiveSlot(
                                    slot_id=sid,
                                    req=req,
                                    last_tok=tok0,
                                    pos=len(req.prompt_ids),
                                    generated=1,
                                )
                            )
                    drain_more = False
                    with self._lock:
                        for slot_obj in new_actives:
                            self._active[slot_obj.slot_id] = slot_obj
                        if new_actives:
                            self._gpu_inputs_dirty = True
                        self._sync_slot_metrics_locked()
                        # Admit remaining queued non-streaming requests before entering decode
                        drain_more = bool(
                            self._pending
                            and self._free_slots
                            and self._active
                            and not any(s.req.stream for s in self._active.values())
                        )
                    if drain_more:
                        continue
                except Exception as exc:
                    with self._lock:
                        for sid, req in admit:
                            self._deliver_error(req, exc)
                            self._active.pop(sid, None)
                            if hasattr(self.runtime, "clear_slot"):
                                self.runtime.clear_slot(sid)
                            if sid not in self._free_slots:
                                heapq.heappush(self._free_slots, sid)
                        self._sync_slot_metrics_locked()
                    continue

            # 2. Execute continuous decode step(s) across all currently active slots
            with self._lock:
                decode_list = [s for s in self._active.values() if not s.req.cancel_event.is_set()]
                if not decode_list:
                    self._sync_slot_metrics_locked()
                    continue
                has_pending = bool(self._pending)

            try:
                active_sids = [s.slot_id for s in decode_list]
                in_toks = [s.last_tok for s in decode_list]
                in_pos = [s.pos for s in decode_list]

                use_multi = (
                    not admit
                    and not has_pending
                    and len(decode_list) >= self.max_batch_size
                    and not any(s.req.stream for s in decode_list)
                    and hasattr(self.runtime, "decode_slots_steps")
                )
                num_steps = 1
                if use_multi:
                    rem_min = min(max(1, s.req.max_tokens - s.generated) for s in decode_list)
                    num_steps = min(32, rem_min)
                    if hasattr(self.runtime, "bucket_len_for_pos"):
                        max_p = max(in_pos)
                        blen = int(self.runtime.bucket_len_for_pos(max_p))
                        num_steps = min(num_steps, max(1, blen - max_p))

                if self._telemetry is not None:
                    self._telemetry["batches_executed"] += num_steps
                    self._telemetry["batch_size_sum"] += len(active_sids) * num_steps
                    self._telemetry["batch_size_max"] = max(
                        int(self._telemetry["batch_size_max"]), len(active_sids)
                    )

                sync_inputs = self._gpu_inputs_dirty
                self._gpu_inputs_dirty = False
                if num_steps > 1:
                    multi_toks = self.runtime.decode_slots_steps(
                        active_sids,
                        in_toks,
                        in_pos,
                        num_steps=num_steps,
                        sync_inputs=sync_inputs,
                    )
                    with self._lock:
                        for slot_obj, tok_seq in zip(decode_list, multi_toks, strict=True):
                            if slot_obj.req.cancel_event.is_set():
                                self._release_slot_locked(slot_obj.slot_id)
                                continue
                            for nxt_tok in tok_seq:
                                slot_obj.last_tok = int(nxt_tok)
                                slot_obj.pos += 1
                                slot_obj.generated += 1
                                self._deliver_token(slot_obj.req, int(nxt_tok))
                                if slot_obj.generated >= slot_obj.req.max_tokens:
                                    self._deliver_done(slot_obj.req)
                                    self._release_slot_locked(slot_obj.slot_id)
                                    break
                        self._sync_slot_metrics_locked()
                else:
                    try:
                        next_toks = self.runtime.decode_slots_step(
                            active_sids, in_toks, in_pos, sync_inputs=sync_inputs
                        )
                    except TypeError:
                        next_toks = self.runtime.decode_slots_step(active_sids, in_toks, in_pos)
                    with self._lock:
                        for slot_obj, nxt_tok in zip(decode_list, next_toks, strict=True):
                            if slot_obj.req.cancel_event.is_set():
                                self._release_slot_locked(slot_obj.slot_id)
                                continue
                            slot_obj.last_tok = int(nxt_tok)
                            slot_obj.pos += 1
                            slot_obj.generated += 1
                            self._deliver_token(slot_obj.req, int(nxt_tok))
                            if slot_obj.generated >= slot_obj.req.max_tokens:
                                self._deliver_done(slot_obj.req)
                                self._release_slot_locked(slot_obj.slot_id)
                        self._sync_slot_metrics_locked()
            except Exception as exc:
                with self._lock:
                    for slot_obj in decode_list:
                        self._deliver_error(slot_obj.req, exc)
                        self._release_slot_locked(slot_obj.slot_id)
                    self._sync_slot_metrics_locked()
