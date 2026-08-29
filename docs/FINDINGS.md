# Findings — What We Learned Building Production Hebrew OCR

This is a findings document, not a changelog. It distills a multi-month effort into the
handful of results that changed how we build Hebrew OCR. Negative results are treated as
first-class here: several of the most useful things we learned were dead ends that saved
everyone who comes after us the same spend.

Every number below is measured, BiDi-normalized, and reproducible from the eval harness
(see [`../eval/run_eval.md`](../eval/run_eval.md) and [`../eval/EVAL_RESULTS.md`](../eval/EVAL_RESULTS.md)).
All of it is **in-house evaluation on our own test sets** — vendor-reported, not
independently verified.

---

## 1. The differentiator is real-domain + bilingual robustness, not pure academic Hebrew

The intuitive goal — "beat Tesseract on clean Hebrew" — is the wrong target. On clean
academic Hebrew, Tesseract is genuinely strong, and the older word-level stack narrowly
*lost* to it (3.20% vs 1.57% page-CER). Chasing that margin is low-value.

The real edge is **cross-domain robustness**. Tesseract collapses on the documents people
actually scan:

| domain | Tesseract | ours (crop-CER) |
|---|---:|---:|
| thermal receipts | 26.32% | **2.09%** |
| menus | 12.41% | **2.80%** |
| sofer invoices | 7.99% | **2.68%** |

A model that holds ~3% across thermal, menu, and invoice inputs — where Tesseract is at
8–26% — wins the use cases that matter, even when it is a hair behind on pristine text.

That said, we *also* closed the academic gap. On a clean, releasable, public-domain
rendered pure-Hebrew set (Sefaria text, perfect GT, n=600), the flagship SVTRv2 reader now
beats Tesseract outright: **NRTR 0.35% vs Tesseract 1.34% micro-CER** (CTC 0.98%,
cascade 0.76%). So the honest framing is: *we now win pure Hebrew too, but robustness is
what makes the tool worth shipping.*

---

## 2. CTC deletes embedded LTR islands; attention reads them — the headline technique

This is the single most important recognition finding. When a run of Latin letters or
digits is **embedded inside RTL Hebrew** (an institution name, a gene, a citation, a date, a
page reference), a CTC decoder tends to **blank-collapse the whole island** — it drops the
LTR run at the RTL↔LTR boundary. Pure scripts are fine (isolated Latin 1.20%, isolated
digits 1.41%); the failure is specifically the mixed boundary.

An attention decoder (NRTR) reads the same island natively — it sees the whole line and is
not bound to CTC's monotonic left-to-right alignment.

Same backbone, swap the head:

| slice | CTC | NRTR / cascade |
|---|---:|---:|
| real heb+lat (n=233) | 12.63% | **2.33%** (~5.4×) |
| heb+dig | 4.99% | **0.56%** (~9×) |

The failure is **not** glyph inability and **not** an ordering bug (pure-reorder rate is
near zero) — it is a CTC compression bias at the boundary. We deploy this as a
**confidence + script-gated cascade**: fast CTC by default (~90% of crops, pure script),
fall back to NRTR only when the CTC output carries a Latin/digit char or is low-confidence.
That buys the NRTR ceiling on mixed text at ~1.25–1.36× CTC latency, no retrain.
Full mechanism and gate design: [`techniques.md`](techniques.md).

---

## 3. Long Hebrew lines: a ~52% wall for years, broken single-pass by SVTRv2

Reading long Hebrew lines (61–80 chars) was the hardest sub-problem. Every line-level model
**we** trained plateaued around **52–55% CER** on that bucket, no matter the data volume.

Long-sequence degradation is not Hebrew-specific, which was the clue that ours was not a
Hebrew data problem. PaddleOCR
[#11482](https://github.com/PaddlePaddle/PaddleOCR/issues/11482) reports that long-sequence
recognition "never matches the accuracy of single words" across CRNN / SAR / ABINET / SVTR
(script unspecified), and
[#13938](https://github.com/PaddlePaddle/PaddleOCR/issues/13938) documents a **Chinese**
`ch_PP-OCRv4_rec` model stuck near 50% *accuracy* on ~135-char lines. Neither is Hebrew and
neither reports CER, so they are corroboration that the failure mode is general — not
external measurements of our number.

Two things broke it:

- **Sliding-window inference** on the old 25-char CTC model was a **−45pt patch**
  (52% → 6.93% on 61–80 chars) — process the line in overlapping ~320px windows, stitch by
  Levenshtein alignment of the seams. Real, but **model-specific and seam-prone**: it only
  worked on the exact v5 model at the exact scale, added 3–5× rec passes, and every seam was
  a chance for duplication garbage. The v6 backbone could not slide at all (architecturally
  verified). Sliding was a workaround, never the answer.
- **SVTRv2** (a native long-text reader) made it obsolete: it reads the whole line
  **single-pass, no seams**, and drops 71–80 char Hebrew to **0.49% CER**. The wall is
  simply gone, on independent public-domain text, with a standard architecture.

Lesson: for long lines, the fix was a reader that natively handles the sequence length, not
an inference trick bolted onto a short-crop model.

---

## 4. Word-DET beats line-DET for Hebrew pages — fairly tested

For page-level pipelines, word-level detection (detect words, assemble into RTL lines) beats
line-level detection. We tested this **fairly**, because our first comparison was
tuning-confounded — the word detector was Hebrew-finetuned while the line detector was stock.
So we trained a Hebrew line detector with the *same* mobile DBNet architecture and the *same*
warm-start base as the word detector (only the GT differs: line polygons vs word boxes),
converged it to val hmean 0.913, and re-ran the 71-page GCV-anchored benchmark with the
SVTRv2 cascade held constant:

| page pipeline (SVTRv2 cascade held constant) | micro-CER |
|---|---:|
| **word-DET** | **7.56%** |
| Hebrew-finetuned line-DET | 10.36% |
| stock line-DET | 10.67% |
| Tesseract | 14.20% |

The Hebrew finetune closed only 0.31pt vs stock, and a threshold sweep closes ~0.25pt more.
So **~2.5pt of the ~2.8pt gap is neither detector-config nor detector-weights** — it is the
fundamental word-vs-line difference. Word-level segmentation plus RTL word-assembly handles
Hebrew page layout (especially multi-column academic) better than line-level, even with a
good line detector and a reader that reads whole lines natively. This retires the
tuning-confound caveat.

**Exception (honest):** line-DET wins degraded scans (pd_phase1: 28.8% vs word-DET 35.7%,
where word-DET over-fragments broken print) and helps bilingual. A line-DET route is a
candidate for degraded / hard-layout domains only; word-DET is the general flagship. The
finetune also left us a deployable 4.8 MB Hebrew line detector for that route.

---

## 5. Negative results worth publishing

These cost real compute and are worth writing down so nobody repeats them.

- **Line-level rec from a word-specialized base cliffs at 21–30 chars.** Abandoned **four
  times** (Stage A/B, v5, line_rec_v2). The word-trained backbone hits its horizon exactly at
  its 25-char training cap; three independent levers (learning rate, width, data up-weighting)
  all confirmed it is a capability limit, not a tuning miss. Long-line reading requires either
  from-stock line training or a native long-text reader — never a warm-start off a short-crop
  base.

- **A light RepSVTR student is a capacity wall, not a recipe problem.** RepSVTR plateaued at
  ~12–18% CER across **three** separate runs (from-scratch, warm-start, and full
  longtext-pretrain), while its Chinese pretrain proves the glyph capacity exists. Pivoting to
  a **same-family SVTRv2-small** student distilled cleanly (heb_only 3.51%, heb+lat 3.81%,
  heb+dig 1.85%) — same-family KD aligns student↔teacher features in a way cross-arch KD does
  not. Match the student architecture to the teacher.

- **The light-CTC quadrant is not viable.** A small backbone cannot produce cleanly
  CTC-alignable (per-column monotonic) Hebrew features — a light-CTC head plateaued ~54% on
  pure Hebrew even though the *same* small backbone reads Hebrew at 3.5% through NRTR. The
  heavy teacher does Hebrew CTC at 0.63%; the small one cannot. **The light tier ships
  NRTR-only**; the pure-script fast CTC path stays on the heavy teacher.

- **INT8 dynamic quant of the transformer is quality-perfect but slower on non-VNNI x86.**
  Quantizing the SVTRv2 NRTR encoder was lossless (2.53% vs 2.54% CER) but ran ~1.2× *slower*
  on an AVX2 (no VNNI) CPU — dynamic-INT8 `MatMulInteger` needs AVX-512-VNNI to pay off. INT8
  is a win only on VNNI-capable x86 (Ice Lake+ / Xeon Scalable 2nd gen+), not a free speedup.

---

## 6. The RTL logical-order footgun

The most expensive silent bug in Hebrew OCR: storing or scoring text in **visual** order.

**Rule: store and score Hebrew in LOGICAL Unicode order** (reading order — first character of
the word is the first character of the string). `python-bidi`'s `get_display()` is for
**rendering only** (GUI, PDF) and must never be applied before storage, training, or CER
scoring. Applying it early once cost 114,755 corrupted annotations.

Detection heuristic: the fraction of Hebrew words whose first character is a final-form letter
(ם ן ך ף ץ). Logical-order data sits at 1–6%; true visual-order pools show 20–27%. Any CER
computed on bilingual text must BiDi-normalize **both** sides before Levenshtein, or the
metric is meaningless. Charset and ordering details: [`charset.md`](charset.md).
