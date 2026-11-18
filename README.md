# CuTileForge

**Tile-Fusion Graph (TFG) Compiler and CuTile LLM Inference Engine for 8 GB-VRAM GPUs**

CuTileForge compiles an entire transformer layer into a **Tile Fusion Graph (TFG)** lowered directly to NVIDIA's **CuTile Python DSL** (`cuda.tile`), fusing 8 separate operations (`QKV GEMM`, `GQA Attention`, `O GEMM`, `LayerNorm/RMSNorm`, `FFN-up GEMM`, `SiLU`, `FFN-down GEMM`, `LayerNorm+Residual`) into **3 tile kernels** with 3-stage software pipelining (`latency=3`) and Stage -1 pinned-DDR5 PCIe DMA weight streaming.

## Target Hardware & Tested Environment

- **Laptop**: Lenovo IdeaPad 15APH8
- **GPU**: NVIDIA GeForce RTX 4060 Laptop GPU (`AD107`, `sm_89`, 24 SMs, 8 GB GDDR6)
- **CPU / RAM**: AMD Ryzen 7 7735HS (8C/16T), 32 GB DDR5
- **Model**: `Mistral-7B-v0.1` 4-bit GPTQ (`D:\models\mistral_7b_4bit\`, 4.16 GB)

## Quickstart

```bash
# 1. Install package and CUDA baselines
pip install -e ".[cuda,dev]" --no-build-isolation
python -m cutile_forge.baselines.build

# 2. Download Mistral-7B-v0.1 4-bit GPTQ to D:\models\mistral_7b_4bit\
python models/download.py --output D:/models/mistral_7b_4bit/

# 3. Run unit + GPU correctness tests
pytest tests/ -v
pytest benchmarks/correctness/ -v

# 4. Run text generation
python examples/generate.py --model D:/models/mistral_7b_4bit/ --prompt "The capital of France is" --max-tokens 32

# 5. Run full benchmark suite & generate docs/RESULTS.md + plots
python benchmarks/run_all.py --model D:/models/mistral_7b_4bit/ --output benchmarks/results/
```

## Measured Highlights (RTX 4060 Laptop, sm_89)

| Benchmark | Metric | Measured Result |
| --- | --- | --- |
| **Mistral-7B Decode (256 tokens)** | Throughput / VRAM | **32.43 tok/s** (30.83 ms/tok), **4.30 GB** peak VRAM |
| **Mistral-7B Prefill (2048 tokens)** | Throughput / VRAM | **1,499.0 tok/s**, **4.67 GB** peak VRAM |
| **Long Context (32K tokens)** | Paged KV-Cache | Completes at **6.01 GB** VRAM via DDR5 eviction |
| **Full Layer (3k TFG vs 8k CUDA)** | Speedup (`seq=1 / 64 / 256 / 512`) | **5.47× / 2.22× / 1.64× / 1.96×** |
| **Fused FFN (1k vs 3k CUDA)** | Speedup (`seq=1 / 64 / 256 / 512`) | **5.48× / 2.00× / 1.46× / 1.78×** |
| **GQA FlashAttention-2 (seq=1024)** | CuTile vs PyTorch SDPA | **21.76 TFLOPS** (0.395 ms) vs **13.23 TFLOPS** (0.649 ms) |
| **GEMM+INT4 Dequant (256×12288×4096)** | Fused vs Separate CUDA | **23.37 TFLOPS** (1.10 ms) vs 2.37 ms (**2.15×**) |
| **Stage -1 PCIe DMA Overlap** | Compute-DMA Overlap | **93.8%–100.0%** overlap at **13.30 GB/s** (84.4% util) |

See [docs/RESULTS.md](docs/RESULTS.md) for the complete tables and generated plots, and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for TFG compiler internals.

## License

MIT — see [LICENSE](LICENSE).

