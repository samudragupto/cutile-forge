# Controlled Fault Injection & Recovery Matrix (`CuTileForge v3.1.0`)

| Fault | Detection | User-visible behavior | Recovery | Data/KV integrity | Status |
|---|---|---|---|---|---|
| 1. Invalid JSON & unsupported API fields | FastAPI/Pydantic strict schema (extra=forbid) | HTTP 422 / 422 | 18.11 ms | Untouched | `VERIFIED` |
| 2. Missing / invalid API key (auth enabled) | Bearer check before queueing/execution | HTTP 401 / 401 | 18.58 ms | Auth bypass blocked | `VERIFIED` |
| 3. Oversized prompt & excessive max_tokens | Tokenizer prompt length bound + Field(le=4096) | HTTP 400 / 422 | 20.98 ms | KV bounds protected | `VERIFIED` |
| 4. Client disconnect mid-SSE stream | request.is_disconnected() + finally slot release | Follow-up HTTP 200, inflight=0 | 1087.52 ms | Subsequent completion uncorrupted | `VERIFIED` |
| 5. Queue saturation & backpressure (max_concurrent=2) | requests_inflight >= max_concurrent_requests guard | Status codes=[200, 429, 200, 429, 429, 429] | 434.88 ms | 200 responses valid; rejected clean | `VERIFIED` |
| 9. Owned worker termination (SIGTERM) & restart | Supervisor detects exit, spawns replacement server | Restarted pid 877937->878075 HTTP 200 | 11556.13 ms | Clean cold-start KV pool | `VERIFIED` |
| 6. Missing model path & corrupt checkpoint | Startup FileNotFoundError / JSONDecodeError | exit codes 1 / 1 | 3752.43 ms | Fail-fast prior to serving | `VERIFIED` |
| 7. CUDA Graph capture failure -> eager fallback | RuntimeError catch on _capture_decode_graph | fallback_count=1, logits finite=True | 536.14 ms | Decode output finite & valid | `VERIFIED` |
| 8. KV-cache pressure / capacity exhaustion | Bounds check prior to pool scatter | KV sequence length 32 exceeds max_seq_len=16 | 11.77 ms | No OOB pool overwrite | `VERIFIED` |
| 10. Repeated server start/stop cycles (3 cycles) | Per-cycle server /metrics RSS tracking | RSS per cycle MB=[5440.1, 5438.1, 5437.2] | 31670.26 ms | Owned PIDs cleanly reaped | `VERIFIED` |
