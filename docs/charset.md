# Charset & the RTL logical-order rule

Every rec model in this release shares **one byte-identical 120-character
charset**, `models/charset_v2f.txt` (md5 `e17ce22e7b4ab8224a3dad9e4c85b6ae`).
Before anything else, read the rule below — it is the single most common way to
silently corrupt a Hebrew OCR dataset.

---

## 1. The RTL logical-order rule (read this first — it is the top footgun)

**All labels and all model output are in LOGICAL Unicode order.** The first
character you *read* is the first character in the string. This is the order
Unicode calls "logical" (as opposed to "visual"). Store, score, and train on
logical order, always.

**`python-bidi`'s `get_display()` is for display only.** Apply it **only** at the
final render step — printing to a terminal, drawing to a PDF, showing in a GUI —
where the display engine will not itself reorder RTL text. **Never** apply
`get_display()` before storing, scoring (CER), or training:

- Applying `get_display()` before storage **silently corrupts your data**. It
  reverses Hebrew runs into visual order; the bytes look plausible and nothing
  errors, but every downstream model learns backwards text.
- CER scoring **does** call `get_display()` on *both* sides (hypothesis and
  reference) immediately before the Levenshtein comparison — that is a transient
  display normalization for the metric, not a change to stored data.

**Detection heuristic.** Compute the percentage of Hebrew words whose **first
character is a final-form letter** (ם ן ך ף ץ). Final forms legitimately occur
only at the *end* of a word, so in correct logical order this rate is **1–6%**
(residual noise). A pool in true **visual order** shows **20–27%**. If you see 20%+,
your data has been `get_display()`-ed before storage — stop and fix it.

Related normalization: **strip Hebrew nikud (vowel points) and all BiDi format
control characters** before storage/training. They are not in the charset and must
never reach the model.

---

## 2. The charset (120 characters)

`models/charset_v2f.txt` — one character per line, index = line number. Composition:

**Hebrew letters (27)** — the 22 base letters plus 5 final forms:
`א ב ג ד ה ו ז ח ט י ך כ ל ם מ ן נ ס ע ף פ ץ צ ק ר ש ת`
(final forms: **ך ם ן ף ץ**).

**Hebrew punctuation (1):** `־` (maqaf, the Hebrew hyphen).

**Latin letters (53):** `A–Z`, `a–z`, and `é` (one accented form seen in loanwords).

**Digits (10):** `0 1 2 3 4 5 6 7 8 9`.

**ASCII punctuation & symbols (22):**
`! " # $ % ' ( ) * + , - . / : ; = ? @ [ ] _`

**Typographic / currency / math symbols (7):**
`– • ₪ € ■ ± &`
(en dash, bullet, shekel sign, euro sign, black square, plus-minus, ampersand).

Total: 27 + 1 + 53 + 10 + 22 + 7 = **120**.

Notes on deliberate inclusions/exclusions:
- `₪` (new shekel) and `€` are in-charset; other currency marks are not.
- `±` was added because scanned academic Hebrew renders `±` where naive GT had a
  plain `+`.
- `|` (pipe) is deliberately **excluded** — it was column-separator/noise content
  the model should never emit.

---

## 3. How the charset maps into each head's vocabulary

The 120 characters are shared, but the two decoder heads wrap them differently.

**CTC head vocabulary** — a blank at index 0, the 120 chars, then a trailing space:

```
["blank"] + <120 chars> + [" "]
```

**NRTR (attention) head label list** — four special tokens first, then the same
120 chars, then a trailing space:

```
["blank", "<unk>", "<s>", "</s>"] + <120 chars> + [" "]
```

Here **`<s>` = BOS = index 2** and **`</s>` = EOS = index 3**. The host-driven
decode loop (see `docs/techniques.md` §1 and `paddleocr_hebrew/plan_e_rec.py`) seeds the
sequence with token 2 and stops on token 3; indices `< 4` are special tokens and
are dropped from the emitted string. Because both heads are built from the same
`charset_v2f.txt`, a crop recognized by either head maps to the identical Unicode
characters.

---

## 4. Invariants for anyone adding a model to this release

- Use `models/charset_v2f.txt` verbatim; verify md5 `e17ce22e7b4ab8224a3dad9e4c85b6ae`.
- Labels in logical Unicode order; `get_display()` only at render time.
- Strip nikud + BiDi controls before training/storage.
- Keep space as the trailing vocab entry for both heads; keep BOS=2 / EOS=3 for
  the NRTR label list.
