"""Download Mistral-7B-v0.1 4-bit GPTQ weights to D:\\models\\mistral_7b_4bit\\ with verification."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

from huggingface_hub import snapshot_download

DEFAULT_TARGET_DIR = Path(r"D:\models\mistral_7b_4bit")
DEFAULT_REPO_ID = "TheBloke/Mistral-7B-v0.1-GPTQ"


def sha256_file(path: Path) -> str:
    """Compute SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_model(
    repo_id: str = DEFAULT_REPO_ID,
    output_dir: str | Path = DEFAULT_TARGET_DIR,
    verify_checksums: bool = False,
) -> Path:
    """Download Mistral-7B 4-bit GPTQ files to `output_dir` and validate layout."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    cache_dir = target.parent / ".hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))

    print(f"Downloading {repo_id} -> {target} ...")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target),
        cache_dir=str(cache_dir),
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"],
    )

    files = [p for p in target.iterdir() if p.is_file()]
    total_bytes = 0
    for p in sorted(files):
        sz = p.stat().st_size
        total_bytes += sz
        if verify_checksums:
            print(f"  {p.name}: {sz / 1e6:.1f} MB (sha256={sha256_file(p)[:16]}...)")
        else:
            print(f"  {p.name}: {sz / 1e6:.1f} MB")

    print(f"  Total: {total_bytes / 1e9:.2f} GB")
    if not (target / "config.json").exists():
        raise FileNotFoundError(f"Missing config.json in {target}")
    if not (target / "tokenizer.json").exists():
        raise FileNotFoundError(f"Missing tokenizer.json in {target}")
    if not any(p.suffix == ".safetensors" for p in files):
        raise FileNotFoundError(f"Missing .safetensors weights in {target}")
    if not (3.0e9 < total_bytes < 5.5e9):
        raise RuntimeError(f"Unexpected total size {total_bytes} bytes in {target}")

    print(f"Model downloaded and verified at {target}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Mistral-7B-v0.1 4-bit GPTQ model")
    parser.add_argument("--repo", default=DEFAULT_REPO_ID, help="HuggingFace repo ID")
    parser.add_argument("--output", type=Path, default=DEFAULT_TARGET_DIR, help="Target directory")
    parser.add_argument("--checksum", action="store_true", help="Print SHA256 prefixes")
    args = parser.parse_args()
    download_model(repo_id=args.repo, output_dir=args.output, verify_checksums=args.checksum)


if __name__ == "__main__":
    main()
