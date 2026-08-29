#!/usr/bin/env python3
"""Recognize a Hebrew page in ~10 lines.

Models live on the Hugging Face Hub. First run downloads them (~370 MB) and
caches them; pass --models-dir to use a local snapshot instead.

    pip install -e .            # from the repo root (installs the `ocr` package)
    pip install huggingface_hub # only needed for auto-download
    python examples/quickstart.py examples/sample_images/sample_page.png

For rendering the RTL output correctly in a terminal/GUI, apply python-bidi
get_display() at display time ONLY — never before storing/scoring (the pipeline
returns LOGICAL Unicode order).
"""
import argparse

from ocr import HebrewOCR

HF_REPO = "rivoklabs/paddleocr-hebrew"   # <-- your HF model repo


def resolve_models_dir(models_dir, pipeline):
    if models_dir:
        return models_dir
    from huggingface_hub import snapshot_download
    det = "line-det/*" if pipeline == "line" else "word-det/*"
    return snapshot_download(repo_id=HF_REPO, allow_patterns=[
        "charset_v2f.txt", det, "server-svtrv2/*",
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="image or PDF path")
    ap.add_argument("--models-dir", help="local HF snapshot dir (skip download)")
    ap.add_argument("--pipeline", choices=["word", "line"], default="word",
                    help="word (default, flagship) or line (degraded/hard layouts)")
    ap.add_argument("--cpu", action="store_true", help="force CPU providers")
    args = ap.parse_args()

    providers = ["CPUExecutionProvider"] if args.cpu else None
    models_dir = resolve_models_dir(args.models_dir, args.pipeline)
    factory = HebrewOCR.line if args.pipeline == "line" else HebrewOCR.word
    ocr = factory(models_dir, providers=providers)
    result = ocr.read(args.image)

    print(f"[{result['meta']['n_lines']} lines, "
          f"{result['meta']['n_words']} words, "
          f"{result['meta']['n_nrtr_fallback']} NRTR fallbacks]")
    for line in result["lines"]:
        print(line["text"])


if __name__ == "__main__":
    main()
