"""Hebrew OCR output post-processors.

Consumer-side normalizations to apply to recognized text before display or
downstream use (they do not change the model):

  - normalize_quote_styles / normalize_gershayim: canonicalize apostrophe/quote
    forms to ASCII ' and " (e.g. two adjacent apostrophes between Hebrew letters
    -> a single gershayim ").
  - fix_geresh_yod_*: repair the SVTRv2 residual defect where the ~1px abbreviation
    marks geresh ' / gershayim " are read as a yod (yod resembles a thin mark),
    e.g. עמ'->עמי, ס"ק->סיק. Conservative by design (over-correcting a real yod is
    worse than a missed mark): a whole-token whitelist of yod-forms that are NOT
    valid Hebrew words, plus numeric-gated page/section refs.
  - reading_order: sort a word list into BiDi-friendly reading order (row cluster
    + per-row RTL/LTR direction).

NB Hebrew text must be stored/scored in LOGICAL order. Apply python-bidi
get_display() only at the final rendering step (GUI/PDF), never before.
"""
import re

HEB = r"֐-׿"


def normalize_gershayim(s):
    """Two adjacent ASCII apostrophes between Hebrew letters -> a single ASCII
    double-quote (canonical gershayim). 'ס''מ' -> 'ס"מ'."""
    s = re.sub(rf"([{HEB}])''([{HEB}])", r'\1"\2', s)
    s = re.sub(rf"([{HEB}])''(?=[\s\W]|$)", r'\1"', s)
    return s


def normalize_quote_styles(s):
    """Map curly quotes / Hebrew presentation quotes to canonical ASCII."""
    s = s.replace("״", '"')   # U+05F4 Hebrew gershayim -> "
    s = s.replace("׳", "'")   # U+05F3 Hebrew geresh -> '
    s = s.replace("“", '"').replace("”", '"')
    s = s.replace("‘", "'").replace("’", "'")
    return s


def post_process(s):
    """Apply all canonical quote/gershayim post-processors to OCR output."""
    s = normalize_quote_styles(s)
    s = normalize_gershayim(s)
    return s


# ──────────────────────────────────────────────────────────────────────────
# Geresh/gershayim <- yod repair (SVTRv2 residual defect)
# ──────────────────────────────────────────────────────────────────────────
_HEBSET = "֐-׿"

# yod-form -> correct mark-form. Every KEY is a non-word (verified), so an exact
# standalone-token match is safe. Values use canonical ASCII ' and ".
GERESH_YOD_TOKENS = {
    "עויד": "עו\"ד",    # lawyer  (עו"ד)
    "פרופי": "פרופ'",   # professor (פרופ')
    "וכוי": "וכו'",      # etc. (וכו')
    "וגוי": "וגו'",      # and so forth (וגו')
    "בגיץ": "בג\"ץ",    # High Court of Justice (בג"ץ)
    "דויח": "דו\"ח",    # report (דו"ח)
    "צהיל": "צה\"ל",    # IDF (צה"ל)
    "ארהיב": "ארה\"ב",  # USA (ארה"ב)
}

_GY_TOKEN_RE = re.compile(
    rf"(?<![{_HEBSET}])(" + "|".join(re.escape(k) for k in GERESH_YOD_TOKENS) + rf")(?![{_HEBSET}])"
)

# numeric-gated page/section references (yod-form IS a real word -> require a following number)
_GY_PAGEREF = [
    (re.compile(rf"(?<![{_HEBSET}])עמי(?=\s+\d)"), "עמ'"),   # page:  עמ' 131
    (re.compile(rf"(?<![{_HEBSET}])מסי(?=\s+\d)"), "מס'"),   # number: מס' 5
    (re.compile(rf"(?<![{_HEBSET}])סיק(?=\s+\d)"), "ס\"ק"),  # sub-section: ס"ק 3
]


def fix_geresh_yod_token(s):
    """Whole-token geresh/gershayim<-yod repair (whitelist only; safe on isolated words)."""
    if not s:
        return s
    return _GY_TOKEN_RE.sub(lambda m: GERESH_YOD_TOKENS[m.group(1)], s)


def fix_geresh_yod_text(s):
    """Full geresh/gershayim<-yod repair for TEXT with token order: whitelist tokens
    PLUS numeric-gated page/section refs (עמ'/מס'/ס"ק NN). Use on joined line text."""
    if not s:
        return s
    s = fix_geresh_yod_token(s)
    for rx, rep in _GY_PAGEREF:
        s = rx.sub(rep, s)
    return s


# ──────────────────────────────────────────────────────────────────────────
# Reading-order assembly (operates on lists of {bbox, text, ...} dicts)
# ──────────────────────────────────────────────────────────────────────────

def _row_direction(row_words):
    """Paragraph-level direction for a row by counting strong chars (default RTL)."""
    rtl, ltr = 0, 0
    for w in row_words:
        for c in w.get("text", ""):
            if 0x0590 <= ord(c) <= 0x05FF or 0x0600 <= ord(c) <= 0x077F:
                rtl += 1
            elif c.isalpha() and ord(c) < 0x0590:
                ltr += 1
    if rtl > ltr:
        return "rtl"
    if ltr > rtl:
        return "ltr"
    return "rtl"  # default to RTL for Hebrew documents


def cluster_rows(words, row_tolerance=0.5):
    """Group words into rows by y-center (row_tolerance x median word height).

    Returns a list of rows, each a list of word dicts, top-to-bottom.
    """
    if not words:
        return []
    heights = sorted([w["bbox"][3] - w["bbox"][1] for w in words])
    median_h = heights[len(heights) // 2] if heights else 30
    tol = max(median_h * row_tolerance, 5)

    indexed = [((w["bbox"][1] + w["bbox"][3]) / 2, w) for w in words]
    indexed.sort(key=lambda x: x[0])
    rows = []
    current = [indexed[0]]
    cur_y = indexed[0][0]
    for y, w in indexed[1:]:
        if abs(y - cur_y) <= tol:
            current.append((y, w))
            cur_y = sum(t[0] for t in current) / len(current)
        else:
            rows.append(current)
            current = [(y, w)]
            cur_y = y
    rows.append(current)
    return [[t[1] for t in row] for row in rows]


def order_row(row_words):
    """Order one row's words by reading direction (RTL -> rightmost first)."""
    dir_ = _row_direction(row_words)
    key = (lambda w: -((w["bbox"][0] + w["bbox"][2]) / 2)) if dir_ == "rtl" \
        else (lambda w: ((w["bbox"][0] + w["bbox"][2]) / 2))
    return sorted(row_words, key=key)


def reading_order(words, row_tolerance=0.5):
    """Sort words into a flat BiDi-friendly reading order (rows top-to-bottom,
    each row ordered by its paragraph direction)."""
    out = []
    for row in cluster_rows(words, row_tolerance):
        out.extend(order_row(row))
    return out


if __name__ == "__main__":
    tests = [
        ("ס''מ", "ס\"מ"), ("ס\"מ", "ס\"מ"), ("בע''מ", "בע\"מ"),
        ("פרופ'", "פרופ'"), ("הבורסה”", "הבורסה\""),
    ]
    for inp, expected in tests:
        got = post_process(inp)
        ok = "OK " if got == expected else "XX "
        print(f"  {ok} {inp!r:14s} -> {got!r:14s} (want {expected!r})")
    ws = [
        {"bbox": [400, 100, 500, 130], "text": "אוניברסיטת"},
        {"bbox": [200, 100, 350, 130], "text": "חיפה"},
        {"bbox": [600, 100, 700, 130], "text": "Haifa"},
    ]
    print("reading order (Hebrew-dominant row -> RTL):",
          [w["text"] for w in reading_order(ws)])
