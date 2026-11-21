# Hardware Support Matrix (`CuTileForge v3.1.0`)

| Target Hardware | Execution Scope | Max Verified Context | Status | Evidence Artifact |
|---|---|---:|---|---|
| NVIDIA RTX 5090 (`sm_120`, 32 GB GDDR7) | Dense 7B GPTQ (`desc_act=True`), MoE 8x7B (`47B`), 70B Hybrid PCIe 5.0 | 32,768 (Mistral-7B) / 2K–4K (Mixtral) | `VERIFIED` | `phase14c_long_context/manifest.json` |
| NVIDIA RTX 5090 (`sm_120`) | `UnifiedModelExecutor` Qwen2.5-0.5B FP16 without KV-cache decode loop | 2,048 | `DEGRADED` | `phase14b_external_engines/manifest.json` |
| NVIDIA RTX 5090 (`sm_120`) | 64K Context on Mistral-7B-v0.1 | 65,536 | `UNSUPPORTED` | `phase14c_long_context/manifest.json` |
| Lenovo LOQ 15APH8 RTX 4060 Laptop (`sm_89`, 8 GB) | Historical `v0.1.0` validated; `v3.1.0` Phase 14 bundle queued (inaccessible from workstation) | 32,768 (KV paged EVICT) | `PENDING_HARDWARE` | `docs/RESULTS.md` + `scripts/reproduce_validation.sh` |
| Single-GPU Distributed Loopback (`TP=2/4`, `PP=2`, `EP=2`) | Mathematical equivalence on 1× RTX 5090 | N/A | `SIMULATED` | `phase14d_multi_gpu/manifest.json` |
| Physical Multi-GPU Cluster (`>= 2` GPUs NVLink/PCIe) | Real multi-rank TP/PP/EP scaling (1 physical GPU on host) | N/A | `PENDING_HARDWARE` | `phase14d_multi_gpu/manifest.json` |
| CPU-Only CI Runner (`ubuntu-latest`) | Schema, TFG compiler, reliability & report-generation tests | N/A | `VERIFIED` | `.github/workflows/validation.yml` |
