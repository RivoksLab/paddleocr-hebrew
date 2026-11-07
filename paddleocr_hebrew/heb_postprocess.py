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


def _word_script(text):
    """'R' for a word with any Hebrew/Arabic letter (prefixed tokens such as ו-Twitter or ה-CEO are
    Hebrew words), 'L' for a word with Latin letters only, 'N' for numbers and punctuation."""
    if any(0x0590 <= ord(c) <= 0x05FF or 0x0600 <= ord(c) <= 0x077F for c in text):
        return "R"
    if any(c.isalpha() and ord(c) < 0x0590 for c in text):
        return "L"
    return "N"


def _is_number(text):
    return any(c.isdigit() for c in text) and not any(c.isalpha() for c in text)


# Numbers next to an embedded run are ambiguous on the page ("Windows 11" and "50 USD" look alike).
EDGE_NUMBERS_JOIN_RUN = False


def _restore_runs(visual_words, direction):
    """Put each embedded run of the other script back in its own order.

    `visual_words` are in visual order for `direction` (rightmost first for "rtl"). A run starts and
    ends on a word of the other script; numbers and punctuation between its words belong to it;
    punctuation at its edges stays with the surrounding text, and numbers there join it only when
    EDGE_NUMBERS_JOIN_RUN is set.
    """
    base, other = ("R", "L") if direction == "rtl" else ("L", "R")
    out, seg = [], []

    def flush():
        idx = [i for i, (_, s) in enumerate(seg) if s == other]
        if idx:
            a, b = idx[0], idx[-1]
            if EDGE_NUMBERS_JOIN_RUN:
                while a > 0 and _is_number(seg[a - 1][0].get("text", "")):
                    a -= 1
                while b < len(seg) - 1 and _is_number(seg[b + 1][0].get("text", "")):
                    b += 1
            out.extend(w for w, _ in seg[:a])
            out.extend(w for w, _ in reversed(seg[a:b + 1]))
            out.extend(w for w, _ in seg[b + 1:])
        else:
            out.extend(w for w, _ in seg)
        seg.clear()

    for w in visual_words:
        s = _word_script(w.get("text", ""))
        if s == base:
            flush()
            out.append(w)
        else:
            seg.append((w, s))
    flush()
    return out


def order_row(row_words, direction=None):
    """Order one row's words in logical (reading) order.

    direction=None decides from the row: any Hebrew word makes it right to left (rightmost word first),
    Latin without Hebrew makes it left to right, and a row of numbers only stays right to left as in a
    Hebrew document. One guard: a row that starts and ends with Latin-only words and has at least
    3x more Latin than Hebrew letters stays left to right (an English line where one short word was
    misread as Hebrew, e.g. "in" read as "מו"). Inside the row, each embedded run of the other script
    keeps its own order, so an English phrase in a Hebrew line reads "אוניברסיטת חיפה University of
    Haifa", not reversed. Pass "ltr"/"rtl" to force the paragraph direction (see row_directions).
    """
    cx = lambda w: (w["bbox"][0] + w["bbox"][2]) / 2  # noqa: E731
    if direction is None:
        scripts = [_word_script(w.get("text", "")) for w in sorted(row_words, key=cx)]
        if "R" in scripts:
            strong = [s for s in scripts if s != "N"]
            heb = sum(1 for w in row_words for c in w.get("text", "")
                      if 0x0590 <= ord(c) <= 0x05FF or 0x0600 <= ord(c) <= 0x077F)
            lat = sum(1 for w in row_words for c in w.get("text", "") if c.isalpha() and ord(c) < 0x0590)
            english_line = strong[0] == "L" and strong[-1] == "L" and lat >= 3 * heb
            direction = "ltr" if english_line else "rtl"
        else:
            direction = "ltr" if "L" in scripts else "rtl"
    visual = sorted(row_words, key=(lambda w: -cx(w)) if direction == "rtl" else cx)
    return _restore_runs(visual, direction)


def row_directions(rows, window=2, latin_ratio=4.0, min_rows=3):
    """Paragraph direction per row (rows as returned by cluster_rows), or None to let order_row decide.

    Safeguard for English text that quotes Hebrew: a row with Hebrew in it is "ltr" when the row and its
    neighbours (window rows above and below) hold at least `min_rows` rows with letters and
    `latin_ratio` times more Latin than Hebrew letters. Every other row gets None.
    """
    counts = []
    for row in rows:
        h = l = 0
        for w in row:
            for c in w.get("text", ""):
                if 0x0590 <= ord(c) <= 0x05FF or 0x0600 <= ord(c) <= 0x077F:
                    h += 1
                elif c.isalpha() and ord(c) < 0x0590:
                    l += 1
        counts.append((h, l))
    out = []
    for i, (h_i, _) in enumerate(counts):
        near = counts[max(0, i - window): i + window + 1]
        h = sum(c[0] for c in near)
        l = sum(c[1] for c in near)
        lettered = sum(1 for c in near if c[0] or c[1])
        out.append("ltr" if h_i and lettered >= min_rows and l >= latin_ratio * h else None)
    return out


def reading_order(words, row_tolerance=0.5):
    """Sort words into a flat BiDi-friendly reading order (rows top-to-bottom,
    each row in logical order; see order_row)."""
    rows = cluster_rows(words, row_tolerance)
    out = []
    for row, direction in zip(rows, row_directions(rows)):
        out.extend(order_row(row, direction))
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
    # word order: (word, x1, x2) as the boxes sit on the page -> expected logical text
    order_tests = [
        ([("University", 400, 580), ("of Haifa", 600, 780), ("2024", 800, 880), ("חיפה", 900, 980),
          ("אוניברסיטת", 1000, 1100)], ["אוניברסיטת חיפה 2024 University of Haifa",
                                        "אוניברסיטת חיפה University of Haifa 2024"]),
        ([("היום", 0, 120), ("הודיעה", 150, 280), ("Microsoft", 300, 440), ("Corporation", 460, 600)],
         ["Microsoft Corporation הודיעה היום"]),
        ([("ו-Twitter", 0, 140), ("Facebook", 160, 300), ("הודיעו", 320, 450)], ["הודיעו Facebook ו-Twitter"]),
        ([("החדשות", 0, 120), ("–", 130, 150), ("mako", 160, 240)], ["mako – החדשות"]),
        ([("אורתופדיה", 150, 380), ("7.1.4", 400, 480)], ["7.1.4 אורתופדיה"]),
        ([("Total", 0, 80), ("due", 90, 140)], ["Total due"]),
        ([("from", 0, 60), ("his", 70, 110), ("post", 120, 170), ("מו", 180, 200), ("accordance", 210, 330),
          ("with", 340, 390)], ["from his post מו accordance with"]),
    ]
    for ws, expected in order_tests:
        row = [{"bbox": [x1, 100, x2, 130], "text": t} for t, x1, x2 in ws]
        got = " ".join(w["text"] for w in order_row(row))
        want = expected[1] if EDGE_NUMBERS_JOIN_RUN and len(expected) > 1 else expected[0]
        print(f"  {'OK ' if got == want else 'XX '} {got}   (want {want})")
