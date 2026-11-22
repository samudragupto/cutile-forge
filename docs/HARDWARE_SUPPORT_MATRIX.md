# Hardware Support Matrix (`CuTileForge v3.6.0`)

| Target Hardware | Execution Scope | Max Verified Context | Status | Evidence Artifact |
|---|---|---:|---|---|
| NVIDIA RTX 5090 (`sm_120`, 32 GB GDDR7) | Dense 7B GPTQ (`desc_act=True`), MoE 8x7B (`47B`), 70B Hybrid PCIe 5.0, Native Cross-Modal | 32,768 (Mistral-7B) / 2K–4K (Mixtral) | `VERIFIED` | `benchmarks/validation_artifacts/v3.5.0/phase18_coverage/compatibility_matrix.json` |
| NVIDIA RTX 5090 (`sm_120`) | `UnifiedModelExecutor` Qwen2.5-0.5B FP16 without KV-cache/CUDA-graph decode loop | 2,048 | `DEGRADED` | `benchmarks/validation_artifacts/v3.6.0/phase20_external_engines/manifest.json` |
| NVIDIA RTX 5090 (`sm_120`) | 64K Context on Mistral-7B-v0.1 | 65,536 | `UNSUPPORTED` | `benchmarks/validation_artifacts/v3.1.0/phase14c_long_context/manifest.json` |
| NVIDIA RTX 4060 Laptop (`sm_89`, 8 GB GDDR6) | One-command laptop profile (`scripts/run_laptop_4060.sh`) + `cutile-forge import-hardware-report` ready; deferred on real device | N/A | `UNVALIDATED` | `scripts/run_laptop_4060.sh` |
| Single-GPU Distributed Loopback (`TP=2/4`, `PP=2`, `EP=2`) | Mathematical sharding parity on 1x RTX 5090 (excluded from performance sections) | N/A | `SIMULATED` | `benchmarks/validation_artifacts/v3.6.0/phase19_multi_gpu/manifest.json` |
| Physical Multi-GPU Cluster (`>= 2` GPUs NVLink/PCIe) | Real multi-rank NCCL / TP / PP / EP / DistributedPagedKVCache scaling (host has 1 GPU) | N/A | `BLOCKED: hardware` | `scripts/run_multi_gpu_validation.sh` + `docs/MULTI_GPU_RUNBOOK.md` |
| CPU-Only CI Runner (`ubuntu-latest`) | Schema, TFG compiler, reliability, report-generation & CLI tests | N/A | `VERIFIED` | `.github/workflows/validation.yml` |
