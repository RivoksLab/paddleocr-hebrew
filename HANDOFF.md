# Handoff — pushing paddleocr-hebrew to GitHub + Hugging Face

Everything is built and committed locally. You push (no credentials touched this
session). Two artifacts:

- **GitHub repo** (code + docs + small models): `/mnt/shared_drive/claude_projects/heb_OCR/paddleocr-hebrew/`
- **Hugging Face model repo** (all ONNX weights): `/mnt/shared_drive/claude_projects/heb_OCR/paddleocr-hebrew-hf/`

> Namespaces (confirmed 2026-08-29): GitHub owner = `RivoksLab`, Hugging Face
> namespace = `Rivok`. All READMEs, `examples/quickstart.py` (`HF_REPO`), and the
> HF `README.md` already use these.

## 1. GitHub (create private, then push)

Create an **empty private** repo named `paddleocr-hebrew` on github.com (no README/
license/gitignore — the repo already has them). Then:

```bash
cd /mnt/shared_drive/claude_projects/heb_OCR/paddleocr-hebrew
git remote add origin git@github.com:Rivok/paddleocr-hebrew.git   # or https://…
git push -u origin main
```

HTTPS auth needs a Personal Access Token (github.com → Settings → Developer
settings → Fine-grained tokens, `Contents: read/write` on this repo) as the
password. SSH needs your key on github.com. Flip the repo to public when you've
reviewed.

## 2. Hugging Face (models)

```bash
pip install huggingface_hub          # provides the `hf` CLI
hf auth login                              # paste an HF write token
hf repos create paddleocr-hebrew --type model --private
cd /mnt/shared_drive/claude_projects/heb_OCR/paddleocr-hebrew-hf
hf upload Rivok/paddleocr-hebrew . . --repo-type model
```

Uploads ~370 MB (7 model folders + shared charset + cards + md5sums). Make the HF
repo public when ready. The GitHub links to the HF repo will then resolve.

## 3. Verify after push

- GitHub: clone fresh, `pip install -e .`, `python examples/quickstart.py --cpu <image>`.
- HF: `hf download Rivok/paddleocr-hebrew --include "server-svtrv2/*"`
  and check the md5s against each `md5sums.txt`.

## What's in the release

- `paddleocr_hebrew/` — runnable inference package (word-DET + SVTRv2 Plan-E cascade), verified
  end-to-end on GPU (the flagship path).
- `models/` — shared charset + small ONNX in-repo (word-det, line-det, mobile-word);
  large models are card-only (download from HF).
- `docs/` — techniques (split-ONNX host loop, CTC-vs-attention), FINDINGS, charset/RTL rule.
- `eval/` — benchmark scorecard + reproduction guide (synth/public-domain material only).
- `examples/` — 10-line quickstart + a public-domain sample page.

## Known open items (safe to ship private; revisit before public)

- **Sample image** `examples/sample_images/sample_page.png` is a PIL render of
  public-domain text (Genesis + demo lines). Fine to ship. For a glossier demo you
  may swap in a real page you own the rights to.
- **Mixed-line word ordering**: the reading-order assembly uses paragraph-direction
  sorting (not full UAX#9 BiDi), so a Hebrew-dominant line with a long embedded
  Latin run can order the runs by position rather than strict logical BiDi. Fine for
  most documents; documented in `docs/charset.md`.
- **eval/manifests/** ships a README only (real crops are private; regenerate
  public-domain eval sets to populate).
