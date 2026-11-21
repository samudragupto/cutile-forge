# Independent & Clean-Room Validation Guide (`CuTileForge`)

This guide allows a second operator or a fresh machine to reproduce the **Phase 14 Production Validation & Hardware Scaling** artifacts without relying on an editable local checkout.

## Distinction of Claims

- **Clean-room reproducibility:** Fresh clone of the pushed release commit in an isolated temporary directory, installed from the built wheel (`.whl`), running unit tests + real-checkpoint generation + baseline comparison on the same host.
- **Independent reproduction (`INDEPENDENTLY_REPRODUCED`):** The exact same validation command executed by a **second operator** or on a **second physical machine**, emitting its own signed/checksummed `manifest.json` + `samples.jsonl`. Any hardware target not physically executed in the current session is labeled `PENDING_HARDWARE` or `PENDING_INDEPENDENT_REPRODUCTION`.

## Target Profiles

### 1. NVIDIA RTX 5090 Workstation (`sm_120`, 32 GB GDDR7)

```sh
export HF_HOME=/var/tmp/arrhat_models/hf_cache
sh scripts/reproduce_validation.sh \
  --profile rtx5090 \
  --model-dir /home/arrhat/models/mistral_7b_4bit \
  --output-root /var/tmp/arrhat_models/repro_rtx5090 \
  --bench-set full
```

### 2. Lenovo LOQ 15APH8 RTX 4060 Laptop (`sm_89`, 8 GB GDDR6)

On Windows 11 + WSL2 Ubuntu (or native Linux with compatible checkpoint on SSD):

```sh
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
sh scripts/reproduce_validation.sh \
  --profile laptop4060 \
  --model-dir /mnt/d/models/mistral_7b_4bit \
  --output-root ./repro_laptop4060 \
  --bench-set quick
```

Note: Because 8 GB VRAM cannot hold all 32 quantized layers + large 32K KV pools simultaneously, Stage -1 PCIe weight streaming (`gpu_vram_gb=8.0`) bounds active memory while reporting honest degraded decode latency at long contexts.

### 3. CPU-Only CI Verification (`ubuntu-latest`)

```sh
CUDA_VISIBLE_DEVICES="" sh scripts/reproduce_validation.sh --profile cpu_ci
```

## Clean-Room Wheel Install Verification

```sh
cd /var/tmp/arrhat_models
git clone https://github.com/samudragupto/cutile-forge.git clean_room_repo
cd clean_room_repo
git checkout v3.1.0
python3 -m venv /var/tmp/arrhat_models/clean_room_venv
. /var/tmp/arrhat_models/clean_room_venv/bin/activate
pip install /path/to/cutile_forge-3.1.0-py3-none-any.whl
python3 -c "import cutile_forge; print(cutile_forge.__file__)"
```

Verify that `cutile_forge.__file__` resolves inside `/var/tmp/arrhat_models/clean_room_venv/lib/python*/site-packages/cutile_forge/__init__.py` and **not** the cloned working tree.
