"""Recognition preprocessing + CTC decode helpers (SVTRv2-native).

Carved from the project's eval harness so the release pipeline reproduces the
exact validated preprocessing: aspect-preserving resize to height 48, width
quantized to a multiple of 16 (capped at max_w), white right-pad, [-1, 1]
normalization. CTC greedy decode collapses repeats + blanks.

The vocab layout for CTC is ["blank"] + charset_chars + [" "]; index 0 is the
CTC blank. `load_dict` builds it from the shared 120-char charset file.
"""
import cv2
import numpy as np


def load_dict(path):
    """Load the CTC vocab: ['blank'] + charset chars + [' ']."""
    with open(path, encoding="utf-8") as f:
        chars = [ln.rstrip("\n") for ln in f]
    return ["blank"] + chars + [" "]


def ctc_greedy(ids, vocab):
    """Greedy CTC decode of an argmax id sequence (collapse repeats, drop blank=0)."""
    out, prev = [], -1
    for i in ids:
        i = int(i)
        if i != prev and i != 0 and i < len(vocab):
            out.append(vocab[i])
        prev = i
    return "".join(out)


def prep_dynamic_target_w(img, h, target_w):
    """Resize aspect-preserving to height h, then right-pad with WHITE to target_w.

    All crops sharing a target_w stack into one batch. Returns CHW float32 in [-1, 1].
    """
    ih, iw = img.shape[:2]
    ratio = iw / max(ih, 1)
    new_w = max(1, min(target_w, int(round(h * ratio))))
    r = cv2.resize(img, (new_w, h), interpolation=cv2.INTER_LINEAR)
    if new_w < target_w:
        pad = np.full((h, target_w - new_w, 3), 255, dtype=np.uint8)
        r = np.concatenate([r, pad], axis=1)
    r = r.astype(np.float32) / 255.0
    r = (r - 0.5) / 0.5
    return r.transpose(2, 0, 1)


def quantize_width(img, h, round_to, min_w=32, max_w=3200):
    """Aspect-preserving width for a height-h crop, rounded up to `round_to`."""
    ih, iw = img.shape[:2]
    nw = int(round(h * iw / max(ih, 1)))
    return max(min_w, min(max_w, ((nw + round_to - 1) // round_to) * round_to))
