"""Plan E — script-gated CTC->NRTR cascade recognizer.

Default = fast single-pass SVTRv2 CTC ONNX. Fall back to the NRTR split-ONNX
host loop (encoder + decstep + numpy greedy) on any crop whose CTC output
carries an embedded left-to-right run (a digit / Latin char) OR whose CTC
mean-confidence < threshold.

The gate is SCRIPT-CONDITIONED, not confidence-only: CTC is confident-but-WRONG
on the embedded-LTR crops it blank-collapses (it drops a Latin/digit run inside
RTL Hebrew at high confidence), so a pure confidence gate can't catch them — the
script gate can. Falling back only there lets us hit the NRTR quality ceiling on
mixed slices (heb+lat 12.6 -> 2.3, heb+dig 5.0 -> 0.6 CER) while paying NRTR's
cost on only ~10% of crops on typical pages.

Both heads use SVTRv2-native preprocessing (quantize_width + prep_dynamic_target_w,
max_w 1280), so recognized text is byte-identical to the validated eval.

The NRTR head deploys as TWO control-flow-free ONNX graphs (encoder + one decode
step) driven by a numpy greedy loop on the host. This is what makes an
autoregressive decoder runnable on ONNX-only edge runtimes (e.g. Jetson, where
PaddlePaddle has no aarch64 wheel) — a full-loop NRTR export SIGABRTs on the
while/if control flow.
"""
import re
from collections import defaultdict

import numpy as np
import onnxruntime as ort

from .rec_decode import ctc_greedy, load_dict, prep_dynamic_target_w, quantize_width

# an embedded left-to-right run (digit or Latin) in the CTC output — the failure
# mode CTC blank-collapses; presence of one triggers the NRTR fallback regardless
# of confidence.
_LTR = re.compile(r"[0-9A-Za-z]")


def _softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


class PlanECascade:
    """Script-gated CTC->NRTR cascade over a list of BGR crops.

    Usage:
        casc = PlanECascade(ctc_onnx, enc_onnx, dec_onnx, dict_path)
        preds = casc.rec_crops(list_of_bgr_crops)   # -> list[str]
        casc.last_n_fallback                          # NRTR fallbacks in that call
    """

    def __init__(self, ctc_onnx, enc_onnx, dec_onnx, dict_path,
                 conf_threshold=0.50, max_w=1280, maxlen=100,
                 batch_size=16, providers=None):
        prov = providers or ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.ctc = ort.InferenceSession(ctc_onnx, providers=prov)
        self.enc = ort.InferenceSession(enc_onnx, providers=prov)
        self.dec = ort.InferenceSession(dec_onnx, providers=prov)
        self.cin = self.ctc.get_inputs()[0].name
        self.ein = self.enc.get_inputs()[0].name
        self.dn = [i.name for i in self.dec.get_inputs()]
        # CTC vocab: ["blank"] + chars + [" "]
        self.vocab = load_dict(dict_path)
        # NRTR label list: ["blank","<unk>","<s>","</s>"] + chars + [" "]  (BOS=2, EOS=3)
        chars = open(dict_path, encoding="utf-8").read().splitlines()
        self.FULL = ["blank", "<unk>", "<s>", "</s>"] + chars + [" "]
        self.conf_threshold = conf_threshold
        self.max_w = max_w
        self.maxlen = maxlen
        self.batch_size = batch_size
        self.last_n_fallback = 0

    # ---- CTC decode + per-crop confidence (mean max-softmax over EMITTED frames) ----
    def _ctc_decode_conf(self, logits_tv):
        row = logits_tv
        s = float(row[0].sum())
        # PP-OCR heads usually emit post-softmax probs; avoid double-softmax.
        probs = row if (row.min() >= 0.0 and abs(s - 1.0) < 0.05) else _softmax(row)
        ids = probs.argmax(axis=-1)
        text = ctc_greedy(ids, self.vocab)
        confs, prev = [], -1
        for t, i in enumerate(ids):
            i = int(i)
            if i != prev and i != 0 and i < len(self.vocab):
                confs.append(float(probs[t, i]))
            prev = i
        return text, (float(np.mean(confs)) if confs else 0.0)

    # ---- NRTR split-ONNX host loop (paddle-free numpy greedy) ----
    def _nrtr_infer(self, tensor):
        memory = self.enc.run(None, {self.ein: tensor})[0]
        seq = np.array([[2]], dtype=np.int64)   # BOS=2
        for _ in range(self.maxlen):
            wp = self.dec.run(None, {self.dn[0]: memory, self.dn[1]: seq})[0]
            nxt = int(wp[0].argmax())
            if nxt == 3:                         # EOS
                break
            seq = np.concatenate([seq, [[nxt]]], axis=1)
        out = []
        for idx in list(seq[0])[1:]:
            if idx == 3:
                break
            if idx < 4 or idx >= len(self.FULL):
                continue
            out.append(self.FULL[idx])
        return "".join(out)

    def rec_crops(self, crops_bgr, mode="cascade"):
        """Recognize a list of BGR crops. Returns list[str] aligned to crops_bgr.

        mode: 'cascade' (production: CTC + script-gated NRTR fallback),
              'ctc' (CTC only, never fall back — benchmark the fast head),
              'nrtr' (NRTR on every crop — benchmark the quality head).
        """
        self.last_n_fallback = 0
        n = len(crops_bgr)
        preds = [""] * n
        tens = [None] * n
        widths = [None] * n
        for i, c in enumerate(crops_bgr):
            if c is None or c.shape[0] < 2 or c.shape[1] < 2:
                continue
            widths[i] = quantize_width(c, 48, 16, max_w=self.max_w)
            tens[i] = prep_dynamic_target_w(c, 48, widths[i])[None].astype(np.float32)

        # mode 'nrtr': every valid crop goes through NRTR (benchmark the quality head)
        if mode == "nrtr":
            for i, t in enumerate(tens):
                if t is not None:
                    preds[i] = self._nrtr_infer(t)
                    self.last_n_fallback += 1
            return preds

        # 1) fast CTC pass, width-bucketed batches; collect fallback indices
        buckets = defaultdict(list)
        for i, w in enumerate(widths):
            if w is not None:
                buckets[w].append(i)
        fallback = []
        for w, idxs in buckets.items():
            for j in range(0, len(idxs), self.batch_size):
                bidx = idxs[j:j + self.batch_size]
                batch = np.concatenate([tens[k] for k in bidx], axis=0)
                out = self.ctc.run(None, {self.cin: batch})[0]   # (B, T, V)
                for bi, k in enumerate(bidx):
                    text, conf = self._ctc_decode_conf(out[bi])
                    # mode 'ctc': keep CTC always; 'cascade': gate to NRTR on script/conf
                    if mode == "cascade" and (conf < self.conf_threshold or bool(_LTR.search(text))):
                        fallback.append(k)
                    else:
                        preds[k] = text

        # 2) NRTR fallback (autoregressive; per-crop) on the gated minority
        for k in fallback:
            preds[k] = self._nrtr_infer(tens[k])
            self.last_n_fallback += 1
        return preds
