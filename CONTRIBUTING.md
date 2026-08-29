# Contributing

Thanks for looking. This project is small and deliberately scoped, so here is an honest
map of what is easy to contribute and what is not.

## What we would genuinely welcome

- **Bug reports on real Hebrew documents.** The most useful issue you can file is a page
  we read badly, with the expected text. Domain coverage (rabbinic, handwriting, forms,
  historical print) is where this model is weakest and where we have the least data.
- **Independent evaluation.** Our numbers are in-house and vendor-reported. If you
  benchmark these models against anything else on a set we have never seen, we want to
  know — including, especially, if we come out worse.
- **Bidirectional-script results beyond Hebrew.** The CTC-deletes-embedded-LTR-islands
  finding should apply to Arabic, Farsi and Urdu. Nobody has checked. If you do, tell us.
- **Inference fixes and packaging**: the `ocr/` package, ONNX runtime compatibility,
  platform issues, dependency pins.
- **Docs.** If something in `docs/` is wrong or unclear, that is a real bug.

## What we cannot currently accept

- **Retraining or new model weights.** The training weights (`.pdparams`), the training
  configs and the benchmark harness are not published, and training did not run on
  commodity hardware. We cannot review or reproduce a training PR.
- **Reproducing our benchmark.** See [`eval/run_eval.md`](eval/run_eval.md) — the harness
  is not shipped. Please do not open PRs that assume it is.

## Before you open a PR

1. Open an issue first for anything beyond a small fix, so we can agree the direction.
2. Keep the RTL rule: **all text is stored, compared and scored in logical Unicode
   order.** `get_display()` is for rendering only. See [`docs/charset.md`](docs/charset.md).
   A PR that BiDi-flips text before storage or scoring will be rejected on sight.
3. Test with the shipped sample:
   ```bash
   pip install -e .
   python examples/quickstart.py examples/sample_images/sample_page.png
   ```
4. Match the surrounding code style. There is no linter config; just be consistent.

## Licence

By contributing you agree your contribution is licensed under Apache-2.0, matching the
rest of the project.

## Contact

Issues and discussions on this repo, or **ronen@rivoklabs.com**.
