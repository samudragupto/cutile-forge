# Independent & Clean-Room Validation Guide (`CuTileForge v3.6.0`)

This guide allows a second operator or a fresh machine to reproduce CuTileForge `v3.6.0` validation artifacts from a pinned environment without relying on an editable checkout.

## Distinction of Claims

- **Clean-room self-check:** Fresh clone / wheel install in an isolated directory run by the authoring workflow (`v3.1.0/phase14a_clean_room`).
- **Independent reproduction (`PENDING_INDEPENDENT_RUN`):** Execution of the pinned reproduction kit below by a **person other than the agent**. Until an external operator runs it and submits the output manifest, independent reproduction remains `PENDING_INDEPENDENT_RUN`.

## Pinned Reproduction Kit (`benchmarks/repro_kit/`)

- Pinned environment: [`benchmarks/repro_kit/requirements-lock.txt`](../benchmarks/repro_kit/requirements-lock.txt)
- Single-command POSIX `sh` runner: [`benchmarks/repro_kit/run_repro.sh`](../benchmarks/repro_kit/run_repro.sh)
- Reference acceptance ranges: [`benchmarks/repro_kit/expected_ranges.json`](../benchmarks/repro_kit/expected_ranges.json)

```sh
export HF_HOME=/var/tmp/arrhat_models/hf_cache
sh benchmarks/repro_kit/run_repro.sh /path/to/mistral_7b_4bit /var/tmp/arrhat_models/repro_kit_venv
```

## Target Hardware Profiles

### 1. NVIDIA RTX 5090 Workstation (`sm_120`, 32 GB GDDR7)

```sh
export HF_HOME=/var/tmp/arrhat_models/hf_cache
sh scripts/reproduce_validation.sh \
  --profile rtx5090 \
  --model-dir /home/arrhat/models/mistral_7b_4bit \
  --output-root /var/tmp/arrhat_models/repro_rtx5090 \
  --bench-set full
```

### 2. NVIDIA RTX 4060 Laptop (`sm_89`, 8 GB GDDR6) — `UNVALIDATED`

Not yet executed on real laptop hardware; run:

```sh
sh scripts/run_laptop_4060.sh /mnt/d/models/mistral_7b_4bit ./repro_laptop4060 quick
```

### 3. Physical Multi-GPU Cluster (`>= 2` GPUs) — `BLOCKED: hardware` on 1-GPU workstation

See [`docs/MULTI_GPU_RUNBOOK.md`](MULTI_GPU_RUNBOOK.md):

```sh
sh scripts/run_multi_gpu_validation.sh benchmarks/validation_artifacts/v3.6.0/phase19_multi_gpu_physical
```

### 4. CPU-Only CI Verification (`ubuntu-latest`)

```sh
CUDA_VISIBLE_DEVICES="" sh scripts/reproduce_validation.sh --profile cpu_ci
```
