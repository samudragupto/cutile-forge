# Tuning Guide for 8 GB-VRAM GPUs (sm_89)

## Tile Autotuning (`tile_autotuner.py`)

CuTileForge searches 12 candidate `(TM, TN, TK, occupancy, pipeline_stages)` configs:
- **Decode (`M <= 16`)**: Small `TM=16, TN=64, TK=64` avoids wasted M-padding and keeps high CTA occupancy across the 24 SMs on AD107.
- **Prefill (`M >= 128`)**: `TM=128, TN=128, TK=32` or `(64, 128, 64)` with `occupancy=2` and 2D `GROUP_M=8` L2 cache swizzling maximizes Tensor Core utilization.

## Memory & PCIe Streaming (`config.py`, `kv_cache.py`)

- **Model Weights**: Mistral-7B 4-bit GPTQ uses ~4.16 GB in VRAM when resident, leaving ~3.8 GB for activations and paged KV-cache on an 8 GB RTX 4060.
- **Weight Streaming (`GPUConfig(stream_weights=True)`)**: Keeps weights in pinned DDR5 and prefetches layer `L+1` on `copy_stream` (Stage -1) to overlap PCIe H2D transfer with layer `L` compute.
- **Paged KV-Cache**: Set `gpu_budget_bytes` (e.g., `1.5–2.0 GB`) so long contexts (4K–32K) automatically evict cold pages to pinned DDR5 without OOM.

