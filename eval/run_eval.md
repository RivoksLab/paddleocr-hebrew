# Reproducing the Benchmark

This is a practical how-to for reproducing the numbers in [`EVAL_RESULTS.md`](EVAL_RESULTS.md).
The important part is the **CER methodology** — Hebrew OCR metrics are easy to get wrong, and a
metric computed the wrong way is worse than no metric.

---

## What ships in this release

The release includes **synthetic and public-domain eval material only**:

- **Pure Hebrew (Mode 1):** our renders of public-domain **Sefaria** text (CC0 / CC-BY), with
  perfect GT = the source text itself. Reproducible from the render script + a fixed seed.
- **Bilingual synth:** held-out realistic heb+lat / heb+dig renders (disjoint seed,
  pixel-hash de-leaked vs training).

**Real crops (real heb+lat n=233, real-domain menus/invoices/thermal) are NOT shipped** — they
are GCV/vision-anchored and their sources' ToS prohibit republication. Their numbers are
reported in `EVAL_RESULTS.md` for honesty, but you cannot rerun them from this repo. Eval
manifests live in [`manifests/`](manifests/) (placeholder for now — populated as releasable
sets are snapshotted).

---

## CER methodology (get this right or the numbers are meaningless)

### Crop / line level

1. **BiDi-normalize BOTH sides before Levenshtein.** GT and prediction both pass through
   `bidi.algorithm.get_display()` before the edit-distance is computed. Hebrew GT is stored in
   **logical** order; some systems (Tesseract) emit logical order too, but display-order
   mismatches otherwise inflate CER wildly. For digit-leading mixed strings, take the
   **min of the L-ordered and R-ordered** distances.
2. **Strip nikud + BiDi control characters** from both sides (our models never emit nikud;
   scoring against vocalized GT measures a domain mismatch, not recognition).
3. **Drop empty / whitespace-only** GT rows.
4. **Filter to labels within the model's `max_text_length`** (25 for word-level, 80 for
   line-level) — a longer label is an architectural out-of-vocab, not a recognition error.
5. **Dynamic-shape preprocessing is mandatory** (aspect-preserving resize, dyn16 round-to-16,
   `max_w=1280`). Fixed-width padding alone cost 24 CER points in one measurement.

micro-CER = (total edit distance) / (total GT chars) across the slice. Report **exact-match %**
alongside — it is the ship metric; micro-CER is the training/diagnosis signal.

### Page level

Page-level uses **cleaned-CER**, because raw sorted-token CER over-counts GT tokenization
artifacts (compound tokens, attached punctuation). Cleaned-CER =

- normalize + BiDi-normalize both sides,
- **split at script boundaries** (Hebrew ↔ Latin ↔ digit),
- **strip terminal punctuation**,
- **lowercase Latin**,

then Levenshtein. This roughly halves the apparent CER on Hebrew pages vs raw sorted-token and
is the only fair page-level metric. Use **GCV as a secondary pseudo-GT** where available, and
report that it carries a ~7.55% intrinsic noise floor.

---

## Running the harness

### Pure Hebrew (Mode 1, releasable)

```bash
# 1. Build the GT text from Sefaria (public domain)
python scripts/build_clean_eval_sefaria.py        # -> data/clean_eval/heb_text.tsv (n=600)

# 2. Render to crops (seeded = pixel-reproducible)
python scripts/render_clean_eval.py --seed 618    # -> data/clean_eval/images/ + manifest

# 3. Score every system on the same crops
python scripts/eval_clean_benchmark.py            # -> results/clean_eval/RESULTS.md
```

The benchmark script runs Tesseract (heb, psm7), SVTRv2 CTC, SVTRv2 NRTR (split-ONNX host
loop), and the Plan E cascade, and prints the per-length micro-CER + exact table.

### Bilingual synth (releasable)

```bash
python scripts/gen_heblat_realistic_synth.py --eval --seed 4242 --n 300   # held-out heb+lat
python scripts/eval_clean_benchmark.py --slice heb_lat
```

### Tesseract runner

Tesseract is run heb (pure) or heb+eng (bilingual), `--psm 7` (single line). Its output is
**BiDi-normalized against display-order GT** before scoring — Tesseract emits logical order, GT
is stored logical, but the normalization must be applied consistently to both.

### GPU inference

All ONNX inference on Jetson runs **inside Docker with `--runtime=nvidia`** (bare-metal
onnxruntime silently falls back to CPU, 10–30× slower). Tesseract, CER computation, and string
processing run on the host CPU.

---

## Notes

- The NRTR head deploys as **two control-flow-free ONNX graphs** (encoder + decoder-step) driven
  by a numpy host greedy loop — this is what makes the attention reader deployable on Jetson
  (a full-loop NRTR ONNX crashes the exporter on `while`/`if`).
- The Plan E cascade = CTC by default, fall back to NRTR when the CTC output contains a
  Latin/digit character or confidence < 0.50. On pure Hebrew the gate is mostly silent (CTC
  speed); on mixed text it fires (NRTR quality). ~11% blended fallback, ~1.25–1.36× CTC latency.
- Every releasable render is seeded; images are not committed to keep the repo small — snapshot
  them into a release bundle if you need byte-identical pixel reproduction.
