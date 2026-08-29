# paddleocr-hebrew

Production-grade **Hebrew OCR** built on PaddleOCR — a family of finetuned
recognizers (CTC + attention) plus a runnable page pipeline. Strong on the two
things generic OCR gets wrong for Hebrew: **real-world documents** (menus,
receipts, invoices, scans) and **bilingual Hebrew+Latin/digit lines**.

The flagship recognizer reads long Hebrew lines *single-pass* (71–80 char lines
at **0.49% CER**) and embedded Latin/number runs inside RTL Hebrew that a plain
CTC model silently drops.

Every model here is a **finetune of a PaddleOCR model** (PaddleOCR v3.7.0,
Apache-2.0) and is released under the same licence — see [NOTICE](NOTICE) and the
lineage column in [Models](#models) below. All reported CER figures are
**in-house evaluation on our own test sets**: measured, but vendor-reported and
not independently verified ([eval/EVAL_RESULTS.md](eval/EVAL_RESULTS.md)).

> **Note on scope.** The Paddle training weights (`.pdparams`) and training
> configs are **not** published — this is a release of inference-ready ONNX
> models, not a reproducible training pipeline. See
> [Fine-tuning & collaboration](#fine-tuning--collaboration).

---

## ⚠️ Read this first — Hebrew is RTL, output is LOGICAL order

Every label and every model output in this project is **logical Unicode order**
(the first character you read is the first character in the string). This is
correct and what you want for storing, searching, scoring, and training.

**Only call `python-bidi`'s `get_display()` at the moment you render** to a
terminal, PDF, or GUI. If you apply `get_display()` before storing or comparing,
you silently corrupt your text and every downstream CER looks broken. This is the
#1 way people get garbage out of Hebrew OCR. See [docs/charset.md](docs/charset.md).

```python
from bidi.algorithm import get_display
print(get_display(line["text"]))   # display ONLY — never before storage/scoring
```

---

## Quickstart (2 minutes)

```bash
git clone https://github.com/RivoksLab/paddleocr-hebrew
cd paddleocr-hebrew
pip install -e .            # installs the `ocr` package (ONNX inference only)
pip install huggingface_hub # to auto-download the models from the Hub

python examples/quickstart.py examples/sample_images/sample_page.png
```

```python
from ocr import HebrewOCR

ocr = HebrewOCR(models_dir="path/to/downloaded/models")   # HF snapshot dir
result = ocr.read("page.png")          # image or PDF
for line in result["lines"]:
    print(line["text"])                # logical order (get_display() to show)
```

- **GPU** (CUDA / Jetson): `pip install onnxruntime-gpu` instead of `onnxruntime`.
- On **Jetson JetPack 6**, run inside a CUDA Docker container — bare-metal
  onnxruntime falls back to CPU. (PaddlePaddle has no aarch64 wheel; everything
  here runs on onnxruntime alone, no Paddle at inference.)

Models are hosted on the **Hugging Face Hub**:
[Rivok/paddleocr-hebrew](https://huggingface.co/Rivok/paddleocr-hebrew)
(~370 MB, ONNX + per-model cards + md5sums). The small detectors and the mobile
rec also ship in this repo under [`models/`](models/) for a zero-download start.

---

## The pipeline

```
image / PDF page
   → word detector (mobile DBNet, Hebrew-finetuned)      models/word-det
   → word crops
   → SVTRv2 recognizer, script-gated CTC→NRTR cascade    server-svtrv2
       · fast single-pass CTC on every crop
       · fall back to the NRTR split-ONNX host loop only on crops with an
         embedded Latin/digit run or low CTC confidence (~10% of crops)
   → reading-order assembly (RTL rows) → Hebrew post-processing
   → {lines, words, meta}   (logical Unicode order)
```

This **word-DET + single-pass** design beats line-level detection on Hebrew
pages (see benchmark). The recognizer is in
[`ocr/plan_e_rec.py`](ocr/plan_e_rec.py); the orchestrator in
[`ocr/pipeline.py`](ocr/pipeline.py).

### Two pipelines (same code + recognizer, different detector)

```python
ocr = HebrewOCR.word(models_dir)   # default, flagship
ocr = HebrewOCR.line(models_dir)   # line detector
```

| pipeline | page-CER | s/page (Jetson) | use it when |
|---|---:|---:|---|
| **`word`** (default) | **7.56%** | **4.95** | almost always — faster **and** more accurate |
| `line` | 10.36% | 6.33 | degraded scans / dense hard layouts where word boxes over-fragment |

`line` is **not** faster (fewer rec calls, but each is a wide line crop; the word
detector's forward pass is also cheaper). Its edge is robustness on hard layouts —
and, because the recognizer reads a whole line's BiDi in one pass, it keeps
mixed Hebrew+Latin word order without post-hoc reordering. `quickstart.py`
takes `--pipeline word|line`.

---

## Models

All recognizers share one byte-identical 120-char charset
([`models/charset_v2f.txt`](models/charset_v2f.txt)). Split-ONNX pairs
(encoder + decstep) are one logical model — the attention decode loop runs on the
host (see [docs/techniques.md](docs/techniques.md)).

| model | role | arch | finetuned from | format / size |
|---|---|---|---|---|
| **server-svtrv2** | **flagship** server REC | SVTRv2 (CTC + NRTR heads) | SVTRv2, Chinese pretrain | `ctc.onnx` 77 MB · NRTR split `enc 72 + dec 27 MB` |
| light-svtrv2small | edge / CPU REC (NRTR-only) | SVTRv2-small (KD student) | distilled from server-svtrv2 | split `enc 28 + dec 27 MB` |
| server-v5 | alt word-level server REC | PPHGNetV2-B4 (CTC) | PP-OCRv5 server rec | `rec.onnx` 73 MB |
| server-v6 | alt word-level server REC | PPLCNetV4-medium (CTC) | PP-OCRv6 | `rec.onnx` 60 MB |
| mobile-word | mobile word REC | PPLCNetV3 (KD student) | PP-OCRv5 mobile rec | `rec.onnx` 7.4 MB |
| word-det | **word detector** (flagship) | mobile DBNet PPLCNetV3 | PP-OCR mobile DBNet | `det.onnx` 4.6 MB |
| line-det | line detector (situational) | mobile DBNet PPLCNetV3 | PP-OCR mobile DBNet | `det.onnx` 4.6 MB |

Which recognizer? **server-svtrv2** for the best quality on long + bilingual
lines. **light-svtrv2small** for a small NRTR model on x86-CPU / edge.
**mobile-word** (7.4 MB) for the smallest word-level footprint. server-v5/v6 are
CTC word-level alternatives (note: v6 **cannot** do sliding-window long lines —
SVTRv2 is the long-line tool). Per-model details + numbers in each
`MODEL_CARD.md`.

---

## Benchmark (headline)

Full tables + methodology + caveats in [eval/EVAL_RESULTS.md](eval/EVAL_RESULTS.md).

**Clean rendered pure-Hebrew** (public-domain text, perfect GT), micro-CER:

| Tesseract | SVTRv2 CTC | cascade | SVTRv2 NRTR |
|---:|---:|---:|---:|
| 1.34% | 0.98% | 0.76% | **0.35%** |

**Real bilingual heb+lat** (n=233 real crops), micro-CER: Tesseract **16.14%**
vs our cascade/NRTR **2.33%** (~7× better).

**Page-level** (71-page GCV-anchored, SVTRv2 cascade held constant), micro-CER:

| word-DET + SVTRv2 | Hebrew line-DET + SVTRv2 | Tesseract |
|---:|---:|---:|
| **7.56%** | 10.36% | 14.20% |

Tesseract narrowly wins *clean academic* Hebrew but loses badly on real domains
(thermal 26% vs ~3%, menus 12% vs ~3%). Cross-domain robustness is the edge — see
[docs/FINDINGS.md](docs/FINDINGS.md).

---

## What's novel here

Two techniques we think are worth reusing (writeup in
[docs/techniques.md](docs/techniques.md)):

1. **Attention decoder on ONNX-only edge via a host-driven split-ONNX loop.**
   A full NRTR decode loop won't export to ONNX (control flow) and Paddle won't
   run on Jetson. Export encoder + one-decode-step as two control-flow-free graphs
   and drive the greedy loop in numpy — reproduces paddle output exactly, runs on
   onnxruntime everywhere.
2. **CTC vs attention on bidirectional script.** A monotonic CTC head *deletes*
   Latin/digit runs embedded in RTL Hebrew (heb+lat 12.6% CER); an attention head
   on the same backbone reads them (2.3%). The script-gated cascade gets NRTR
   quality at ~CTC speed.

---

## Documentation

- [docs/charset.md](docs/charset.md) — the 120-char charset + the RTL logical-order rule
- [docs/techniques.md](docs/techniques.md) — split-ONNX host loop, CTC-vs-attention, sliding-window, the cascade
- [docs/FINDINGS.md](docs/FINDINGS.md) — what we learned (including the negative results)
- [eval/EVAL_RESULTS.md](eval/EVAL_RESULTS.md) · [eval/run_eval.md](eval/run_eval.md) — benchmark + how to reproduce

## Fine-tuning & collaboration

The models are released as **ONNX** — inference-ready and portable (CPU, CUDA,
Jetson) from one set of files. ONNX is not a compromise here: on CPU it runs
~7× faster than native PaddlePaddle, on Jetson it's the only thing that runs
(no Paddle aarch64 wheel), and for TensorRT speed on GPU you can point
onnxruntime's TensorRT execution provider at these same `.onnx` files — no
extra artifacts needed.

The Paddle training weights (`.pdparams`) and training configs are **not
published** — ONNX is all you need to run the models. If you want to **fine-tune
on your own Hebrew data, adapt the charset, or collaborate**, please open an
issue / discussion on this repo or reach out: **ronen@rivoklabs.com**. Happy to
help and to grow this together.

## License

Apache-2.0 (code and model weights). The recognizers are finetuned from
PaddleOCR models (also Apache-2.0); see [NOTICE](NOTICE). See [LICENSE](LICENSE).
