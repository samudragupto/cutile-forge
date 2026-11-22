# Physical Multi-GPU, NCCL & Distributed KV Runbook

CuTileForge single-GPU loopback tests (`LoopbackCommTransport`) verify tensor sharding math **only** (`SIMULATED`) and are never valid performance or scaling evidence. Physical multi-GPU validation requires at least two CUDA GPUs visible to both `nvidia-smi -L` and `torch.cuda.device_count()`.

## Current Host Status (`prashant-lab`)

- `nvidia-smi -L`: `GPU 0: NVIDIA GeForce RTX 5090 (UUID: GPU-53a9e748-720f-ea14-df4c-530895df0f88)` (1 physical GPU)
- Status for physical TP/PP/EP/NCCL/DistributedPagedKVCache: **`BLOCKED: hardware`**

## Prerequisites on a Multi-GPU Host (`>= 2` GPUs)

1. Verify GPU visibility and interconnect topology:
   ```sh
   nvidia-smi -L
   nvidia-smi topo -m
   ```
2. Activate the CuTileForge environment (POSIX `dash`-compatible):
   ```sh
   . .venv/bin/activate
   export HF_HOME=/var/tmp/arrhat_models/hf_cache
   export PYTHONPATH=.
   ```
3. Optional NCCL diagnostics / transport overrides (standard PyTorch NCCL variables):
   ```sh
   export NCCL_DEBUG=INFO
   # Only if InfiniBand is unavailable / misconfigured on PCIe hosts:
   # export NCCL_IB_DISABLE=1
   # export NCCL_P2P_LEVEL=PHB
   ```

## Exact Execution Command

Run the unchanged validation wrapper on a `>= 2` GPU machine:

```sh
sh scripts/run_multi_gpu_validation.sh benchmarks/validation_artifacts/v3.6.0/phase19_multi_gpu_physical
```

When `nvidia-smi -L` reports `< 2` GPUs, the script prints the hardware block reason and exits cleanly with code `2`. On `>= 2` GPUs, it records `nvidia-smi topo -m` and launches `mp.spawn(_run_nccl_worker, ...)` inside [`benchmarks/production/bench_multi_gpu_validation.py`](../benchmarks/production/bench_multi_gpu_validation.py) with `reproduction_status="MEASURED"` and `execution_mode="physical-multi-gpu"`.

## Output Artifacts & Importing Back

The run emits a schema-validated directory containing:

- `manifest.json` (`reproduction_status="MEASURED"`, topology matrix, per-rank NCCL all-reduce timings, bytes transferred, exact numerical error)
- `samples.jsonl` (per-rank latency samples)
- `summary.json`, `environment.json`, `commands.log`, `stdout.log`, `stderr.log`

Import a physical multi-GPU report back into the repository with:

```sh
cutile-forge import-hardware-report benchmarks/validation_artifacts/v3.6.0/phase19_multi_gpu_physical/manifest.json --update-matrix docs/HARDWARE_SUPPORT_MATRIX.md
```
