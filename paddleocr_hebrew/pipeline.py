"""Hebrew OCR pipeline — word-DET + SVTRv2 Plan-E cascade (the flagship page path).

image/PDF -> word detector (mobile DBNet) -> crops -> Plan-E cascade recognizer
(SVTRv2 CTC fast path + script-gated NRTR fallback) -> reading-order assembly ->
Hebrew post-processing (gershayim canon + geresh<-yod repair).

This is the page pipeline the benchmark uses: word-level detection + single-pass
recognition (the recognizer reads each word crop once; no sliding window). On the
71-page GCV-anchored eval it scores 7.56% page-CER, ahead of a Hebrew-finetuned
line detector (10.36%) and Tesseract (14.20%).

Output is a dict of recognized LINES (logical Unicode order — apply python-bidi
get_display() only when rendering) plus the underlying word boxes.

    from paddleocr_hebrew import HebrewOCR
    ocr = HebrewOCR(models_dir="hebrew-ocr-models")   # a downloaded HF snapshot
    result = ocr.read("page.png")
    for line in result["lines"]:
        print(line["text"])
"""
import os

import cv2
import numpy as np

from .detect import crop_bbox, detect_words
from .heb_postprocess import cluster_rows, fix_geresh_yod_text, order_row, post_process
from .plan_e_rec import PlanECascade


def render_page(path, dpi=150, page=1):
    """Load an image, or render one PDF page, to a BGR numpy array."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        try:
            import pymupdf  # pymupdf >= 1.24.3
        except ImportError:
            import fitz as pymupdf  # older pymupdf, where `fitz` is the only name
        doc = pymupdf.open(path)
        p = doc[page - 1]
        zoom = dpi / 72
        pix = p.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"):
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(f"Cannot open image: {path}")
        return img
    raise RuntimeError(f"Unsupported input: {path}")


class HebrewOCR:
    """End-to-end Hebrew page OCR: DBNet detector + Plan-E cascade recognizer.

    Two pipelines, same code + same recognizer, only the detector differs:

      HebrewOCR.word(models_dir)   # DEFAULT, flagship. Word-level detection.
                                   # Faster AND more accurate at page level
                                   # (7.56% page-CER, 4.95 s/page on Jetson).
      HebrewOCR.line(models_dir)   # Line-level detection. Slower + a bit less
                                   # accurate on clean pages (10.36%, 6.33 s), but
                                   # WINS on degraded scans / dense hard layouts
                                   # where word detection over-fragments.

    Or construct directly and pass your own `det` ONNX + thresholds.
    """

    @classmethod
    def word(cls, models_dir, **kw):
        """Flagship word-level pipeline (the default; recommended)."""
        kw.setdefault("det", "word-det/det.onnx")
        kw.setdefault("det_thresh", 0.40)
        kw.setdefault("det_unclip", 1.3)
        return cls(models_dir, **kw)

    @classmethod
    def line(cls, models_dir, **kw):
        """Line-level pipeline — for degraded scans / hard layouts (see class doc)."""
        kw.setdefault("det", "line-det/det.onnx")
        kw.setdefault("det_thresh", 0.30)
        kw.setdefault("det_unclip", 1.8)
        return cls(models_dir, **kw)

    def __init__(self, models_dir, *,
                 det="word-det/det.onnx",
                 ctc="server-svtrv2/ctc.onnx",
                 nrtr_encoder="server-svtrv2/nrtr-encoder.onnx",
                 nrtr_decstep="server-svtrv2/nrtr-decstep.onnx",
                 charset="charset_v2f.txt",
                 providers=None,
                 det_thresh=0.40, det_unclip=1.3, det_max_side=1280,
                 dilate_w=0, dilate_h=0,
                 conf_threshold=0.50,
                 gershayim=True, geresh_yod=True):
        import onnxruntime as ort
        prov = providers or ['CUDAExecutionProvider', 'CPUExecutionProvider']

        def mp(rel):
            return rel if os.path.isabs(rel) else os.path.join(models_dir, rel)

        self.det_sess = ort.InferenceSession(mp(det), providers=prov)
        self.det_inp = self.det_sess.get_inputs()[0].name
        self.rec = PlanECascade(mp(ctc), mp(nrtr_encoder), mp(nrtr_decstep),
                                mp(charset), conf_threshold=conf_threshold,
                                providers=prov)
        self.det_thresh = det_thresh
        self.det_unclip = det_unclip
        self.det_max_side = det_max_side
        self.dilate_w = dilate_w
        self.dilate_h = dilate_h
        self.gershayim = gershayim
        self.geresh_yod = geresh_yod

    def read(self, path, *, page=1, dpi=150, rec_mode="cascade"):
        """Recognize one image or PDF page. Returns {lines, words, meta}."""
        img = render_page(path, dpi=dpi, page=page)
        return self.read_array(img, rec_mode=rec_mode)

    def read_array(self, img_bgr, *, rec_mode="cascade"):
        """Recognize an in-memory BGR image array."""
        bboxes = detect_words(img_bgr, self.det_sess, self.det_inp,
                              thresh=self.det_thresh, unclip_ratio=self.det_unclip,
                              max_side=self.det_max_side,
                              dilate_w=self.dilate_w, dilate_h=self.dilate_h)
        crops, keep = [], []
        for bb in bboxes:
            c = crop_bbox(img_bgr, bb)
            if c is not None:
                crops.append(c)
                keep.append(bb)
        preds = self.rec.rec_crops(crops, mode=rec_mode)

        words = []
        for bb, txt in zip(keep, preds):
            if self.gershayim:
                txt = post_process(txt)
            if txt.strip():
                words.append({"bbox": list(bb), "text": txt})

        # assemble lines in reading order (logical Unicode order)
        lines = []
        for row in cluster_rows(words):
            ordered = order_row(row)
            text = " ".join(w["text"] for w in ordered).strip()
            if self.geresh_yod:
                text = fix_geresh_yod_text(text)
            if not text:
                continue
            xs1 = min(w["bbox"][0] for w in ordered); ys1 = min(w["bbox"][1] for w in ordered)
            xs2 = max(w["bbox"][2] for w in ordered); ys2 = max(w["bbox"][3] for w in ordered)
            lines.append({"text": text, "bbox": [xs1, ys1, xs2, ys2],
                          "n_words": len(ordered)})

        return {
            "lines": lines,
            "words": words,
            "meta": {
                "n_words": len(words),
                "n_lines": len(lines),
                "n_nrtr_fallback": self.rec.last_n_fallback,
                "rec_mode": rec_mode,
            },
        }
