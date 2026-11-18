# Mistral-7B-v0.1 (4-bit GPTQ) Weights

Target location on Windows: `D:\models\mistral_7b_4bit\` (or `/mnt/d/models/mistral_7b_4bit/` under WSL2).

## Download

```bash
python models/download.py --output D:/models/mistral_7b_4bit/
```

## Expected Files

| File | Description | Approximate Size |
|------|-------------|------------------|
| `config.json` | Mistral-7B architecture config | ~600 B |
| `quantize_config.json` | 4-bit GPTQ quantization config (`bits=4, group_size=128`) | ~200 B |
| `tokenizer.json` / `tokenizer_config.json` | SentencePiece BPE tokenizer | ~1.8 MB |
| `model.safetensors` | 4-bit GPTQ weights (`TheBloke/Mistral-7B-v0.1-GPTQ`) | ~4.16 GB |

Total size: **~4.16 GB** (fits alongside KV-cache and activations on an 8 GB VRAM RTX 4060, or streams via Stage -1 pinned DDR5 DMA).
