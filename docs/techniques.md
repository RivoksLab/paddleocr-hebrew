# Techniques

Reusable and novel techniques from building a production Hebrew OCR stack on
PaddleOCR. This is a findings/tech document — every claim carries a measured
number and the mechanism behind it. It is the seed of a paper, not marketing.

All CER numbers are micro-CER after the standard eval normalization (BiDi
`get_display()` on both sides before Levenshtein, nikud/BiDi-control stripping,
label-length filtering). "heb+lat" = Hebrew lines with embedded Latin runs;
"heb+dig" = Hebrew lines with embedded digit runs.

---

## 1. Autoregressive decoders on ONNX-only edge via a host-driven split-ONNX loop

**Problem.** An attention decoder (NRTR / any autoregressive head) decodes with a
`while` loop and per-step `if` (stop on EOS). Exporting that loop to ONNX as a
single graph fails: `paddle2onnx` SIGABRTs on the control-flow ops. And on the
deployment target of record — a Jetson AGX Orin on JetPack 6 (aarch64) — there is
**no PaddlePaddle aarch64 wheel**, so "just run paddle" is not an option. Together
these close off the obvious paths to running an attention decoder on the edge.

**Solution.** Split the model into **two control-flow-free graphs** and move the
loop to the host:

- `encoder.onnx` — runs **once** per crop, produces the memory/context tensor.
- `decstep.onnx` — computes **one** decode step: `(memory, tokens_so_far) -> next-token logits`.
- A **numpy greedy loop on the host** drives them: seed with `BOS = 2`, run
  `decstep` once per step, `argmax` the logits, append the token, feed the
  extended sequence back in, stop on `EOS = 3` or a max-length cap.

Because neither exported graph contains control flow, both convert cleanly and run
on **onnxruntime alone** (CPU or CUDA EP). The host loop is paddle-free.

Reference implementation: `paddleocr_hebrew/plan_e_rec.py`, method `PlanECascade._nrtr_infer`.
The full mechanism in ~15 lines:

```python
memory = self.enc.run(None, {self.ein: tensor})[0]     # encoder.onnx, once
seq = np.array([[2]], dtype=np.int64)                  # BOS = 2
for _ in range(self.maxlen):
    wp  = self.dec.run(None, {mem: memory, tok: seq})[0]   # decstep.onnx, one step
    nxt = int(wp[0].argmax())
    if nxt == 3:                                       # EOS = 3
        break
    seq = np.concatenate([seq, [[nxt]]], axis=1)
```

**Payoff (measured).** The host loop reproduces the paddle-native output
**exactly**:

- heb+lat 233-sample CER **2.33%** on Jetson `onnxruntime-gpu` — identical to the
  paddle-native probe (2.37%) and to the x86 native-paddle number.
- Latency **~184 ms/crop** on long bilingual lines: encoder ~65 ms + decode
  ~119 ms over ~56 decode steps. Shorter crops are proportionally faster (fewer
  steps). KV-cache / batching / TensorRT are unexploited headroom.

**Consequence.** An attention decoder is now deployable on *every* target: x86 via
native paddle **or** split-ONNX; Jetson via split-ONNX + host loop. The "CTC-only
on the edge" limitation that motivates so many OCR architecture choices is not
actually a constraint.

---

## 2. CTC vs. attention on bidirectional (RTL + LTR) script: the embedded-LTR-island deletion

This is the core empirical finding of the project.

**Setup.** Hebrew is RTL. Real Hebrew documents interleave **LTR islands** — a
Latin word (institution, drug, gene name), a digit run (date, page ref, currency,
range). We compared a **CTC** head and an **NRTR attention** head trained on the
**same SVTRv2 backbone**, same charset, same data.

**Finding.** The CTC head, whose alignment is strictly monotonic left-to-right,
**systematically deletes the embedded LTR island** — it blank-collapses the Latin
or digit run — and does so **at high confidence**. The attention head over the
same backbone reads the island correctly.

| slice | CTC head | NRTR head |
|---|---:|---:|
| heb+lat (embedded Latin) | 12.63% | **2.33%** |
| heb+dig (embedded digits) | 4.99% | **0.56%** |
| heb_only (pure Hebrew) | 0.64% | 0.36% |
| lat_only (isolated Latin) | 1.20% | ~1.2% |
| dig_only (isolated digits) | 1.41% | ~1.3% |

**What it is NOT.**

- **Not an ordering / BiDi bug.** The pure-reorder rate is ~0% — the model does
  not scramble the logical order, it *drops* characters. (An earlier PPLCNetV4 v6
  model *did* have a digit-run reordering bug; that is a different, fixed defect.)
- **Not glyph inability.** Isolated Latin reads at **1.20%** and isolated digits at
  **1.41%** under the *same CTC head* — the glyphs are learned. Failure appears
  only when the LTR run is embedded inside RTL Hebrew.
- **Not data scarcity fixable by more synth.** Deletion is the dominant edit
  operation: **≈77% of heb+lat edits are deletions** (906 deletions vs. 209
  substitutions in one breakdown). Warm-start finetunes on enriched embedded-LTR
  synth moved the number a couple of points (−1.87pt on real heb+lat) but never
  changed the *mechanism*.

**Can CTC be taught to read the island?** No. We self-distilled the CTC head on the
attention head's own pseudo-labels (Plan D: warm-start, MultiHead, 8 epochs on
NRTR pseudo-labels). CTC heb+lat **plateaued ~13.5%** (orbit 13.07–15.43 across
epochs) against the NRTR ceiling of ~1.6–2.3%. CTC's monotonic decode can absorb a
little of the reordering at the RTL↔LTR boundary (≈−2.5pt) but **cannot reorder
across it**.

**Conclusion.** For bidirectional script, **CTC alone is a false economy on mixed
lines**. Use an attention decoder (or a cascade, §3) whenever LTR islands appear.
On pure single-script text CTC is at parity and cheaper — so pay for attention
only where it earns its keep.

---

## 3. Script-gated CTC → NRTR cascade (combines §1 + §2)

The production lever. Default to the **fast single-pass CTC** head; fall back to
the **NRTR split-ONNX host loop** (§1) only where CTC fails (§2).

**The gate is script-conditioned, not confidence-only.** This is the crucial
detail. Because CTC is *confident-but-wrong* on the crops it blank-collapses, a
confidence-only gate misses them — the mean softmax over emitted frames stays high
while a whole Latin/digit run has silently vanished. The fix: fall back whenever

- the CTC output **contains a digit or Latin character** (`[0-9A-Za-z]`), **OR**
- the CTC **mean confidence < 0.50**.

The script condition catches the high-confidence deletions; the confidence
condition catches ordinary low-confidence noise.

Reference: `paddleocr_hebrew/plan_e_rec.py`, `PlanECascade.rec_crops`. The CTC pass runs in
width-bucketed batches; the gated minority goes to the per-crop NRTR loop. Both
heads use SVTRv2-native preprocessing (`quantize_width` + `prep_dynamic_target_w`,
`max_w = 1280`), so cascade output is byte-identical to the validated eval.

```python
if mode == "cascade" and (conf < self.conf_threshold or bool(_LTR.search(text))):
    fallback.append(k)     # -> NRTR split-ONNX host loop
else:
    preds[k] = text        # keep the fast CTC read
```

**Payoff (measured).** The cascade hits the NRTR quality ceiling on mixed slices
while paying NRTR's cost on only a minority of crops:

- heb+lat **12.63% → 2.33%** (= NRTR ceiling)
- heb+dig **4.99% → 0.56%** (= NRTR ceiling)
- pure Hebrew pays only **~1.3% fallback** (0.63% → 0.61%, no regression)
- blended production fallback **~10–11%** of crops ⇒ **~1.25–1.36× CTC latency**,
  no retraining.

An optional image-side script pre-router (a small CNN script classifier) drops the
blended cost further by skipping the wasted CTC pass on crops it will hand to NRTR
anyway (~1.07× CTC on x86 in one measurement).

---

## 4. Sliding-window inference for long lines on a short-crop CTC model

A documented technique **and** a negative result.

**Technique.** A CTC rec model trained at a 25-char cap can still read 61–80 char
Hebrew lines *at inference time*, with no retraining, by sliding a fixed-width
window across the line:

- Process **right-to-left in pixel space** (Hebrew RTL) so outputs concatenate in
  logical order.
- Window **w ≈ 320 px**, overlap **200 px**, no training-parity padding at window
  scale.
- **Levenshtein-align the seam**: match the last K≈8 chars of the logical-prior
  window against the first ~2K chars of the logical-next window and cut at the
  best alignment (always take the best alignment — never a naive concat).
- Base direction = document RTL; flip to LTR only for lines with zero Hebrew chars
  (UAX#9 first-strong is *wrong* for Hebrew document lines like
  `EU REGULATION ABC מסמך`).

**Payoff (measured).** On the v5 server rec (25-char-trained, 75 MB), 61–80 char
Hebrew CER **~52% → 6.93%** — a pure inference patch, ~0.32 s/line on Jetson. A
3-way shifted-window ensemble (offsets 0/60/100, pick min seam edit distance)
pushed the combined-eval CER further (8.44% → 5.70%).

**Negative result — it is model-sensitive and does not generalize.** Sliding works
only for a model that has *not* been trained with crop-concatenation augmentation:

- v5 (no RecConAug): slides to 6.93% on 61–80 Hebrew.
- v6 / SVTRv2 (RecConAug crop-concat aug): **cannot slide**. Their augmentation
  makes them intolerant of narrow out-of-distribution sub-windows; seams produce
  garbage. v6 61–80 Hebrew sliding measured *worse* than its own single-pass, and
  got monotonically worse with more edge-crop exposure. Confirmed architecturally
  dead for these models. A `HorizontalEdgeCrop` augmentation intended to fix this
  failed its purpose (still couldn't slide) though it acted as a useful
  regularizer.

**Why the flagship makes sliding obsolete.** SVTRv2 reads long lines **natively,
single-pass, with no seams**: 71–80 char Hebrew = **0.49%** CER, breaking the
~52% line-level wall that every prior CTC line model hit. So the flagship pipeline
uses SVTRv2 single-pass and never invokes sliding. Sliding remains the right tool
only for a short-crop CTC model on long clean Hebrew crops where no line-native
model is available.

---

## Summary of the reusable ideas

1. **Split-ONNX + host loop** makes autoregressive decoders deployable on
   ONNX-only edge runtimes — exact parity with native paddle.
2. **CTC deletes embedded LTR islands in RTL script**; attention reads them. A
   general result for any bidirectional-script OCR, not Hebrew-specific.
3. **A script-conditioned gate** (not confidence) is what makes a CTC→attention
   cascade correct, because the failure mode is confident-but-wrong.
4. **Sliding-window inference** rescues long lines on a short-crop CTC model, but
   is model-sensitive; a line-native model (SVTRv2) is strictly better where
   available.
