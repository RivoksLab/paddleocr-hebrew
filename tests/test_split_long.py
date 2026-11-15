"""Long-line splitting before NRTR (plan_e_rec.split_column, join_halves, PlanECascade._nrtr_read).

No models needed: the ONNX heads are replaced by stubs.

    python -m unittest discover -s tests
"""
import unittest

import numpy as np

from paddleocr_hebrew.plan_e_rec import PlanECascade, _SPLIT_LEVELS, join_halves, split_column


def line_image(width=600, height=40, gap=(330, 350), dark_on_light=True):
    """A fake text line: ink everywhere in the middle band except one gap."""
    bg, ink = (255, 0) if dark_on_light else (0, 255)
    img = np.full((height, width, 3), bg, np.uint8)
    img[10:30, 5:width - 5] = ink
    img[:, gap[0]:gap[1]] = bg
    return img


class SplitColumn(unittest.TestCase):
    def test_cuts_in_the_gap(self):
        self.assertTrue(330 <= split_column(line_image()) < 350)

    def test_light_on_dark(self):
        self.assertTrue(330 <= split_column(line_image(dark_on_light=False)) < 350)

    def test_grayscale(self):
        self.assertTrue(330 <= split_column(line_image()[..., 0]) < 350)

    def test_stays_in_the_middle(self):
        # the only gap is near the edge: the cut must still be in the middle 30-70%
        x = split_column(line_image(gap=(20, 40)))
        self.assertTrue(180 <= x <= 420)


class JoinHalves(unittest.TestCase):
    def test_hebrew_right_half_first(self):
        self.assertEqual(join_halves("עולם", "שלום"), "שלום עולם")

    def test_mixed_line_counts_as_hebrew(self):
        self.assertEqual(join_halves("123", "סה״כ"), "סה״כ 123")

    def test_latin_left_half_first(self):
        self.assertEqual(join_halves("hello", "world"), "hello world")

    def test_empty_half(self):
        self.assertEqual(join_halves(" שלום ", ""), "שלום")


class Stub(PlanECascade):
    """PlanECascade without models: CTC 'reads' 1 char per 5 px, NRTR returns a Hebrew letter + the crop width."""

    def __init__(self, split_long=80):
        self.split_long = split_long
        self.max_w = 1280
        self.calls = []
        self.last_n_split = 0

    def _tensor(self, crop):
        return crop

    def _ctc_text(self, tensor):
        return "א" * (tensor.shape[1] // 5)

    def _nrtr_infer(self, tensor):
        self.calls.append(tensor.shape[1])
        return f"ב{tensor.shape[1]}"


class Threshold(unittest.TestCase):
    def test_short_line_not_cut(self):
        s = Stub()
        img = line_image(width=395)
        self.assertEqual(s._nrtr_read(img, img, "א" * 79), "ב395")
        self.assertEqual(s.calls, [395])
        self.assertEqual(s.last_n_split, 0)

    def test_long_line_cut_once(self):
        s = Stub()
        img = line_image(width=600)                 # halves ~340 and ~260 px: 68 and 52 chars
        s._nrtr_read(img, img, "א" * 120)
        self.assertEqual(len(s.calls), 2)
        self.assertEqual(sum(s.calls), 600)
        self.assertEqual(s.last_n_split, 1)

    def test_hebrew_join_order(self):
        s = Stub()
        img = line_image(width=600)
        x = split_column(img)
        self.assertEqual(s._nrtr_read(img, img, "א" * 120), f"ב{600 - x} ב{x}")

    def test_off(self):
        for off in (0, None):
            s = Stub(split_long=off)
            img = line_image(width=600)
            s._nrtr_read(img, img, "א" * 120)
            self.assertEqual(s.calls, [600])

    def test_depth_limit(self):
        s = Stub(split_long=1)                      # everything is "long": cut until the limit
        img = line_image(width=3200)
        s._nrtr_read(img, img, "א" * 640)
        self.assertEqual(len(s.calls), 2 ** _SPLIT_LEVELS)
        self.assertEqual(sum(s.calls), 3200)


if __name__ == "__main__":
    unittest.main()
