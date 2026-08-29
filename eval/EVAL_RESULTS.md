# Benchmark Scorecard — Hebrew OCR

> **These are in-house numbers.** Every figure below was produced by our own evaluation
> harness on our own test sets. They are measured rather than estimated, and BiDi-normalised
> on both sides before Levenshtein — but they are **vendor-reported and not independently
> verified**. Methodology is in [`run_eval.md`](run_eval.md) so you can disagree with it
> specifically.

Honest, reproducible benchmark for the release. All CER is **micro-CER (%)**, computed with
BiDi-normalization on both sides before Levenshtein (see [`run_eval.md`](run_eval.md) for the
exact methodology). Systems compared: **Tesseract 5** (heb / heb+eng, psm7), and our
**SVTRv2 ft2 ep7** reader in three modes — **CTC** (fast head), **NRTR** (attention head,
split-ONNX host loop), and **Plan E cascade** (CTC by default + script-gated NRTR fallback =
what production runs).

---

## 1. Clean rendered pure-Hebrew (RELEASABLE)

Mode 1: our renders of public-domain Sefaria text, owned perfect GT, n=600 (short 200 ≤25ch /
mid 200 / long 200), 0 pixel-hash leakage vs train. Clean typography → measures **recognition**,
not scan-robustness.

| system | ALL | exact |
|---|---:|---:|
| Tesseract 5 (heb) | 1.34 | 70.2 |
| SVTRv2 — CTC | 0.98 | 89.5 |
| SVTRv2 — Plan E cascade | 0.76 | 89.5 |
| **SVTRv2 — NRTR** | **0.35** | **97.2** |

We beat Tesseract on pure Hebrew (reverses the old word-level stack's academic loss). NRTR is
flat across length (long bucket 0.36) — the long-line wall is gone on independent text.

---

## 2. Real bilingual heb+lat (n=233, real crops) — the differentiator

| system | real heb+lat CER | exact |
|---|---:|---:|
| Tesseract (heb+eng) | 16.14 | 0.0 |
| SVTRv2 — CTC | 12.63 | 14.6 |
| **SVTRv2 — Plan E cascade / NRTR** | **2.33** | **73.8** |

Cascade/NRTR are **~7× better than Tesseract** on embedded-Latin Hebrew.

> **Honesty note.** On our *own* held-out synth heb+lat the cascade scores ~0.05% — but that
> synth is **in-distribution** (rendered by the generator our model trained on), so it is
> optimistic. We cite the **REAL 2.33%** as our trustworthy bilingual number, never the synth
> ~0.05%. Tesseract sees the synth cold, which is why it is not flattered there either.

---

## 3. SVTRv2 ft2 ep7 — per-slice micro-CER

| slice | CTC | NRTR / cascade |
|---|---:|---:|
| heb_only | 0.64 | — |
| lat_only | 1.20 | — |
| dig_only | 1.41 | — |
| heb+lat | 12.63 | **2.33** (cascade) |
| heb+dig | 4.99 | **0.56** (NRTR) |
| long Hebrew 71–80 char | 0.49 | — |

Pure scripts are solved (all ≤1.4%, long Hebrew 0.49%). The only weak CTC slices are the
embedded-LTR ones, which the cascade/NRTR path fixes.

---

## 4. Page-level, 71-page GCV-anchored (SVTRv2 cascade held constant)

| page pipeline | micro-CER |
|---|---:|
| **word-DET + SVTRv2 cascade** | **7.56** |
| Hebrew-finetuned line-DET + SVTRv2 cascade | 10.36 |
| stock line-DET + SVTRv2 cascade | 10.67 |
| Tesseract | 14.20 |

Word-DET is the flagship page pipeline; the finetuned line-DET result confirms the gap is the
fundamental word-vs-line difference, not detector tuning (see FINDINGS §4).

---

## 5. Real-domain crop CER (older v5 numbers — noted as such)

These are from the older v5 word-level rec (crop-level), kept for the cross-domain robustness
picture; the SVTRv2 reader is at least as good on these domains.

| domain | crop-CER |
|---|---:|
| menus | 2.80 |
| sofer invoices | 2.68 |
| thermal receipts | 2.09 |

---

## 6. Model sizes (deployed ONNX)

| model | role | ONNX size |
|---|---|---:|
| word-DET (phase1_v2b_mild) | word detector (flagship) | 4.6 MB |
| line-DET (Hebrew-finetuned) | line detector (degraded-scan route) | 4.6 MB |
| mobile-word rec (step2_kd_v2 ep18) | mobile short-crop rec | 7.4 MB |
| **server SVTRv2** CTC | flagship reader, fast head | **77 MB** |
| server SVTRv2 NRTR encoder | attention head (split-ONNX) | 72 MB |
| server SVTRv2 NRTR decoder-step | attention head (split-ONNX) | 27 MB |
| light SVTRv2-small NRTR encoder | mobile attention reader | 28 MB |
| light SVTRv2-small NRTR decoder-step | mobile attention reader | 27 MB |
| server v5 rec | prior server rec | 73 MB |
| server v6 rec | alt server rec (short-crop only) | 60 MB |

---

## Caveats (read before quoting any number)

- **Page-CER is 3–15× crop-CER.** Page-level numbers include detection, layout assembly, and
  GT noise; crop-level numbers assume perfect segmentation. Never compare the two directly.
- **GCV pseudo-GT has a noise floor.** The 71-page anchor is GCV-derived; GCV itself carries
  ~7.55% intrinsic error, so page-level absolute numbers sit on top of that floor.
- **The 12.63% heb+lat is a LONG-LINE number.** It is measured on full lines with embedded
  Latin. The production word-DET regime feeds **short crops**, where heb+lat is closer to ~3%;
  the 12.63% only bites the line-DET + SVTRv2 path.
- **§1 is clean rendered Hebrew** — recognition, not scan-robustness.
- **§2 synth (~0.05%) is in-distribution** for our model; cite the real 2.33% instead.
