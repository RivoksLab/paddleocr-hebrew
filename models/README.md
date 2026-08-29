# Models

All recognizers share one byte-identical 120-char charset,
[`charset_v2f.txt`](charset_v2f.txt) (md5 `e17ce22e7b4ab8224a3dad9e4c85b6ae`).
Each model has a `MODEL_CARD.md` + `md5sums.txt`.

**Small models ship in this repo** (clone-and-run):

| dir | file | size |
|---|---|---|
| [`word-det/`](word-det) | `det.onnx` | 4.6 MB |
| [`line-det/`](line-det) | `det.onnx` | 4.6 MB |
| [`mobile-word/`](mobile-word) | `rec.onnx` | 7.4 MB |

**Large models live on the Hugging Face Hub** —
[Rivok/paddleocr-hebrew](https://huggingface.co/Rivok/paddleocr-hebrew).
These dirs carry the `MODEL_CARD.md` + `md5sums.txt` only; download the ONNX from HF:

| dir | files on HF | size |
|---|---|---|
| `server-svtrv2/` | `ctc.onnx`, `nrtr-encoder.onnx`, `nrtr-decstep.onnx` | 77 + 72 + 27 MB |
| `light-svtrv2small/` | `nrtr-encoder.onnx`, `nrtr-decstep.onnx` | 28 + 27 MB |
| `server-v5/` | `rec.onnx` | 73 MB |
| `server-v6/` | `rec.onnx` | 60 MB |

```bash
# grab the flagship + charset into a local dir
hf download Rivok/paddleocr-hebrew \
    --include "charset_v2f.txt" "word-det/*" "server-svtrv2/*" \
    --local-dir hebrew-ocr-models
```

Then point the pipeline at it: `HebrewOCR(models_dir="hebrew-ocr-models")`.

Split-ONNX pairs (`nrtr-encoder` + `nrtr-decstep`) are **one logical model** — the
attention decode loop runs on the host (see [../docs/techniques.md](../docs/techniques.md)).
