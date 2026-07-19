# Eval manifests

This release ships **synthetic / public-domain** eval material only. The real-crop
evaluation sets (menus, receipts, invoices, and GCV-anchored pages) stay private
because their labels derive from third-party OCR output whose terms of service
forbid redistribution.

A manifest is a TSV: `relative_image_path <TAB> ground_truth_label` (one row per
crop), with ground truth in **logical Unicode order** (see
[../../docs/charset.md](../../docs/charset.md)).

To reproduce the numbers in [../EVAL_RESULTS.md](../EVAL_RESULTS.md) on your own
data, follow [../run_eval.md](../run_eval.md). Public-domain rendered eval sets
(e.g. Sefaria/Wikisource text) can be regenerated from source text; drop the
resulting `images/` + `manifest.tsv` here.
