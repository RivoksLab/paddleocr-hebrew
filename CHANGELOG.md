# Changelog

## 0.1.2

- **Long lines are cut in two before NRTR reads them.** NRTR was trained on lines of up to
  80 characters and loses its place on longer ones. A crop going to NRTR whose CTC reading is
  80 characters or more is now cut at the emptiest column in its middle 30–70%, each half is
  read on its own (cut again if still long, up to 3 levels), and the texts are joined in
  reading order (right half first on Hebrew lines). On 1,861 real line crops (production
  cascade) CER went from 8.17% to 7.27%, with no category worse:

  | slice | before | after |
  |---|---:|---:|
  | Hebrew + numbers | 6.22 | 4.33 |
  | Hebrew + English | 14.40 | 13.53 |
  | English + numbers | 3.98 | 2.82 |
  | English only | 2.66 | 2.04 |
  | Hebrew only | 2.14 | 2.14 |
  | numbers only | 26.70 | 26.70 |
  | **all** | **8.17** | **7.27** |

  Only line-level reading changes (`HebrewOCR.line`, or long crops passed to
  `PlanECascade.rec_crops`); word crops are never that long. New `split_long` argument on
  `HebrewOCR` and `PlanECascade` (default 80; 0 turns it off) and `PlanECascade.last_n_split`.
- First unit tests: `python -m unittest discover -s tests`.

Earlier versions: see the git history.
