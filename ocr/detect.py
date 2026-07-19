"""DBNet text detector (word-level or line-level) — ONNX inference.

Carved from the production pipeline. Runs a mobile PPLCNetV3 DBNet ONNX, decodes
the probability map into boxes via contour-finding + unclip. The SAME code drives
both the word detector (word-level boxes) and the line detector; passing
dilate_w/dilate_h > 0 bridges inter-word gaps so words on a baseline merge into a
line blob (line-mode). The flagship pipeline uses word-level (dilate off).
"""
import cv2
import numpy as np
import pyclipper


def _resize_for_det(img_bgr, max_side=960):
    h, w = img_bgr.shape[:2]
    s = max(h, w)
    if s > max_side:
        scale = max_side / s
        img_r = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
    else:
        scale = 1.0
        img_r = img_bgr
    h, w = img_r.shape[:2]
    nh = ((h + 31) // 32) * 32
    nw = ((w + 31) // 32) * 32
    pad = np.zeros((nh, nw, 3), dtype=np.uint8)
    pad[:h, :w] = img_r
    return pad, scale, h, w


def _normalize_for_det(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_rgb = (img_rgb - mean) / std
    return img_rgb.transpose(2, 0, 1)[None].astype(np.float32)


def _polygon_area_perimeter(pts):
    """Shoelace area + perimeter (no shapely dependency)."""
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    n = len(pts)
    if n < 3:
        return 0.0, 0.0
    area = perim = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i, 0] * pts[j, 1] - pts[j, 0] * pts[i, 1]
        perim += float(np.linalg.norm(pts[j] - pts[i]))
    return abs(area) / 2.0, perim


def _unclip_contour(cnt, ratio=1.3):
    pts = cnt[:, 0]
    area, perim = _polygon_area_perimeter(pts)
    if area < 1:
        return None
    distance = area * ratio / max(perim, 1e-6)
    pco = pyclipper.PyclipperOffset()
    pco.AddPath(pts.tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = pco.Execute(distance)
    if not expanded:
        return None
    return np.array(expanded[0])


def detect_words(img_bgr, det_sess, det_inp, thresh=0.40, unclip_ratio=1.3,
                 min_area=3, min_wh=6, max_side=960, dilate_w=0, dilate_h=0):
    """Return a list of (x1, y1, x2, y2) boxes in original image coordinates.

    dilate_w/dilate_h > 0 -> line-mode (merge words on a baseline into a line).
    """
    pad, scale, valid_h, valid_w = _resize_for_det(img_bgr, max_side)
    arr = _normalize_for_det(pad)
    out = det_sess.run(None, {det_inp: arr})[0]
    prob = out[0, 0, :valid_h, :valid_w]
    mask = (prob > thresh).astype(np.uint8) * 255

    if dilate_w > 0 or dilate_h > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, dilate_w), max(1, dilate_h)))
        mask = cv2.dilate(mask, k, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    bboxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        unclipped = _unclip_contour(cnt, unclip_ratio)
        if unclipped is None:
            continue
        x, y, w, h = cv2.boundingRect(unclipped)
        if w < min_wh or h < min_wh:
            continue
        bboxes.append((int(x / scale), int(y / scale),
                       int((x + w) / scale), int((y + h) / scale)))
    return bboxes


def crop_bbox(img_bgr, bbox, pad=0):
    """Crop (x1,y1,x2,y2) from a BGR image, clamped to bounds, optional pad px."""
    H, W = img_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(W, x2 + pad); y2 = min(H, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return img_bgr[y1:y2, x1:x2]
