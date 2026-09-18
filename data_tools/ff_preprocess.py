"""Python port of Firefox's ML field-context preprocessing.

Faithfully reproduces FormAutofillHeuristics.sys.mjs (Bug 2035232 / D305753):
turn a form's fields into the per-field `mlData` context string the autofill
model consumes -- lowercased, camelCase-split, word-tokenized (words >=3 chars),
with an input-type marker and `bb`(previous)/`aa`(next) neighbor baking.

Ported 1:1 from toolkit/components/formautofill/shared/FormAutofillHeuristics.sys.mjs:
  WORD_RE = /\\s*([\\p{L}\\p{N}]+)/u          (letters/digits only)
  ADJACENT_BEFORE_PREFIX = "bb"  (previous field)
  ADJACENT_AFTER_PREFIX  = "aa"  (next field)
  tokenizeWords / splitMixedCase / tokenizeAttributes / tokenizeElements

Use split_context() (dotraining) on the result to get the three bare strings for
the 'triple' format.

NOTE (train/inference skew): the shipped inference routes the "**"+type marker
through tokenizeWords, whose WORD_RE strips the "**" -> a plain type word
("**email" -> "email"). The training .txt files, however, contain literal
"**email"/"**select-one" tokens, so the generator that produced them preserved
the marker. `type_marker` selects which behavior to reproduce:
  "inference" (default) -> matches the DEPLOYED model's actual input
  "training"            -> matches the tokens in the training .txt
"""
from __future__ import annotations

import regex  # \p{...} unicode classes, matching the JS /u regex
from bs4 import NavigableString, Comment, Tag

WORD_RE = regex.compile(r"[\p{L}\p{N}]+")
_MIXED_RE = regex.compile(r"([\p{Ll}\p{N}]*)(\p{Lu}*)")
ADJACENT_BEFORE_PREFIX = "bb"
ADJACENT_AFTER_PREFIX = "aa"
_NONFIELD_TYPES = {"hidden", "submit", "image", "button", "reset"}
# FF getFormInfo only tokenizes credit-card/address-eligible fields
# (FormAutofillUtils.isCreditCardOrAddressFieldType): inputs of these types
# (unknown type -> "text", which is eligible), plus all <textarea>/<select>.
ELIGIBLE_INPUT_TYPES = {"text", "email", "tel", "number", "month", "search"}

# --- input_attributes / select_option feature constants (ported 1:1 from
# FormAutofillHeuristics.sys.mjs, branch fr_perf) ---
SELECT_OPTION_RANGE_MAX = 16   # chars kept per side of the "<first>...<last>" token
INPUT_MAXLENGTH_CAP = 16       # only emit **maxlen<N> for small explicit maxlengths
DIGIT_PATTERN_RE = regex.compile(
    r"^\^?(?:\\d|\[0-9\]|\[\\d\])(?:(\{(\d+)(?:,\d*)?\})|[+*])?\$?$"
)
# input.type IDL "limited to only known values": unknown -> "text".
_VALID_INPUT_TYPES = {
    "hidden", "text", "search", "tel", "url", "email", "password", "date",
    "month", "week", "time", "datetime-local", "number", "range", "color",
    "checkbox", "radio", "file", "submit", "image", "reset", "button",
}
# inputmode IDL "limited to only known values": unknown -> "".
_VALID_INPUTMODES = {
    "none", "text", "tel", "url", "email", "numeric", "decimal", "search",
}
# Features toggled by extensions.formautofill.useml.features. The -iaor/-relabel
# datasets use both.
DEFAULT_ML_FEATURES = frozenset({"select_option", "input_attributes"})


def tokenize_words(text: str, words: list[str]) -> None:
    """Lowercase, split into letter/digit runs, keep words with >=3 chars."""
    if not text:
        return
    text = text.lower()
    for m in WORD_RE.finditer(text):
        w = m.group(0)
        if len(w) >= 3:            # FF: "Ignore short words"
            words.append(w)


def split_mixed_case(text: str) -> str:
    """Insert a space between a lower/digit run and an Upper run (addressLine
    -> address Line). FF applies this to id and name only."""
    if not text:
        return text
    return _MIXED_RE.sub(r"\1 \2", text)


def element_type(el) -> str:
    if el.name == "select":
        return "select-multiple" if el.has_attr("multiple") else "select-one"
    if el.name == "textarea":
        return "textarea"
    t = (el.get("type") or "text").strip().lower()
    # HTMLInputElement.type is "limited to only known values": unknown -> "text".
    return t if t in _VALID_INPUT_TYPES else "text"


# LabelUtils.sys.mjs constants (ported 1:1).
_EXCLUDED_TAGS = {"script", "noscript", "option", "style"}   # extractLabelStrings
_STOP_TAGS = {                                               # shouldStopIterating
    "button", "input", "label", "meter", "output", "progress", "select",
    "textarea", "form", "fieldset", "script", "style",
}


def _is_text_node(node) -> bool:
    """A DOM Text node: a bs4 NavigableString that is not a Comment/Declaration/
    ProcessingInstruction/CData (all NavigableString subclasses)."""
    return type(node) is NavigableString


def extract_label_strings(element) -> list[str]:
    """LabelUtils.extractLabelStrings: depth-first collect trimmed text of leaf
    nodes, skipping EXCLUDED_TAGS subtrees."""
    strings: list[str] = []

    def rec(el):
        if isinstance(el, Tag) and el.name and el.name.lower() in _EXCLUDED_TAGS:
            return
        if _is_text_node(el) or (isinstance(el, Tag) and not el.contents):
            trimmed = (str(el) if _is_text_node(el) else el.get_text()).strip()
            if trimmed:
                strings.append(trimmed)
            return
        for node in el.contents:
            if isinstance(node, Tag) or _is_text_node(node):
                rec(node)

    rec(element)
    return strings


def _should_stop_iterating(el) -> bool:
    return isinstance(el, Tag) and (el.name or "").lower() in _STOP_TAGS


def _iterate_nodes(element, reverse: bool, filt):
    """LabelUtils.iterateNodes: walk siblings (descending into their subtrees)
    then up to parents, in `reverse` (backwards) order; return filt()'s node, or
    None when filt/stop-tag/DOM-end halts it."""
    while element is not None:
        nxt = element.previous_sibling if reverse else element.next_sibling
        if nxt is None:
            element = element.parent
            if element is not None and _should_stop_iterating(element):
                return None
        else:
            child = nxt
            while child is not None:
                res = filt(child)
                if res is not False:
                    return res
                if isinstance(child, Tag) and _should_stop_iterating(child):
                    return None
                element = child
                kids = child.contents if isinstance(child, Tag) else []
                child = (kids[-1] if reverse else kids[0]) if kids else None
    return None


def find_nearby_text(element) -> str:
    """LabelUtils.findNearbyText: for an element with no <label>, walk backwards
    collecting inline text (guard of 10 nodes; stop-tag boundaries end it),
    prepending each run, then collapse whitespace. Ported 1:1 (incl. the
    `current`-based div short-circuit)."""
    state = {"count": 10, "current": element, "txt": ""}

    def return_text_node(node):
        cur = state["current"]
        c = state["count"]
        state["count"] = c - 1                       # JS post-decrement: !count--
        if not c or (isinstance(cur, Tag) and (cur.name or "") == "div"
                     and len(state["txt"]) > 0):
            return None
        return node if _is_text_node(node) else False

    current = element
    while True:
        current = _iterate_nodes(current, True, return_text_node)
        if current is None:
            break
        state["current"] = current
        text_content = str(current)
        if text_content:
            state["txt"] = text_content + state["txt"]
    return regex.sub(r"\s{2,}", " ", state["txt"]).strip()


def find_label_elements(el, soup) -> list:
    """Approximate LabelUtils.findLabelElements: <label for=id> and ancestor
    <label> (the associated-control cases). Adjacent-control association is not
    modeled (rare, and absent from the synthetic forms this is used for)."""
    labels = []
    fid = el.get("id")
    if fid:
        labels.extend(soup.find_all("label", attrs={"for": fid}))
    parent = el.find_parent("label")
    if parent is not None:
        labels.append(parent)
    return labels


def get_label_strings(el, soup) -> list[str]:
    """FormAutofillHeuristics._getElementLabelStrings: strings of associated
    <label> elements; if none, the single findNearbyText result; then aria-label."""
    out: list[str] = []
    labels = find_label_elements(el, soup)
    for lab in labels:
        out.extend(extract_label_strings(lab))
    if not labels:
        out.append(find_nearby_text(el))
    aria = el.get("aria-label")
    if aria:
        out.append(aria)
    return out


def _element_max_length(el) -> int:
    """DOM element.maxLength: parsed `maxlength` content attr, or -1 when unset/
    invalid. undefined (no such attr) for <select> -> treated as -1."""
    raw = el.get("maxlength")
    if raw is None:
        return -1
    try:
        return int(str(raw).strip())
    except ValueError:
        return -1


def _element_input_mode(el) -> str:
    """DOM element.inputMode: lowercased `inputmode` content attr limited to
    known values; "" when unset or invalid."""
    im = (el.get("inputmode") or "").strip().lower()
    return im if im in _VALID_INPUTMODES else ""


def tokenize_input_attributes(el, words: list[str]) -> None:
    """FF tokenizeInputAttributes: append **inputmode<mode> and **maxlen<N>.
    A digit-only `pattern` maps onto numeric inputmode + a fixed {N} length."""
    max_length = _element_max_length(el)
    input_mode = _element_input_mode(el)

    pattern = (el.get("pattern") or "").strip()
    if pattern:
        m = DIGIT_PATTERN_RE.match(pattern)
        if m:
            if not input_mode:
                input_mode = "numeric"
            if m.group(2) and not (max_length > 0):
                max_length = int(m.group(2))

    if input_mode and input_mode != "text":
        words.append("**inputmode" + input_mode)
    if 0 < max_length < INPUT_MAXLENGTH_CAP:
        words.append("**maxlen" + str(max_length))


def _option_label(option) -> str:
    """FF: (option.text || option.value || "").toLowerCase, all whitespace
    removed, truncated to SELECT_OPTION_RANGE_MAX chars."""
    text = option.get_text() or option.get("value") or ""
    text = text.lower()
    text = regex.sub(r"\s+", "", text)
    return text[:SELECT_OPTION_RANGE_MAX]


def tokenize_select_option_range(el, words: list[str]) -> None:
    """FF tokenizeSelectOptionRange: append "<first>...<last>" from the first/
    last non-blank options (skipping empty placeholder options at each end)."""
    if el.name != "select":
        return
    options = el.find_all("option")
    if not options:
        return
    lo, hi = 0, len(options) - 1
    while lo < hi and not _option_label(options[lo]):
        lo += 1
    while hi > lo and not _option_label(options[hi]):
        hi -= 1
    first = _option_label(options[lo])
    last = _option_label(options[hi])
    if first or last:
        words.append(f"{first}...{last}")


def tokenize_attributes(el, soup, type_marker: str = "inference",
                        features=DEFAULT_ML_FEATURES) -> list[str]:
    """FF tokenizeAttributes: id, name (both split-mixed-case), placeholder,
    labels, the input-type marker for non-text fields, then (when enabled) the
    select_option range token and input_attributes tokens."""
    words: list[str] = []
    tokenize_words(split_mixed_case(el.get("id", "")), words)
    tokenize_words(split_mixed_case(el.get("name", "")), words)
    tokenize_words(el.get("placeholder", ""), words)
    for label in get_label_strings(el, soup):
        tokenize_words(label, words)
    etype = element_type(el)
    if etype != "text":
        if type_marker == "training":
            words.append("**" + etype)          # literal marker, as in the .txt
        else:
            tokenize_words("**" + etype, words)  # inference: WORD_RE strips "**"
    if features and "select_option" in features:
        tokenize_select_option_range(el, words)
    if features and "input_attributes" in features:
        tokenize_input_attributes(el, words)
    return words


def fillable_fields(soup) -> list:
    """Fields FF's getFormInfo tokenizes, in document order: <textarea> and
    <select> always, <input> only for the credit-card/address-eligible types
    (element_type maps unknown/missing input types to 'text', which is
    eligible). Matches FormAutofillUtils.isCreditCardOrAddressFieldType."""
    out = []
    for el in soup.find_all(["input", "select", "textarea"]):
        if el.name == "input" and element_type(el) not in ELIGIBLE_INPUT_TYPES:
            continue
        out.append(el)
    return out


def tokenize_elements(soup, type_marker: str = "inference",
                      features=DEFAULT_ML_FEATURES) -> list[str]:
    """FF tokenizeElements: each field's own words + bb-prefixed previous +
    aa-prefixed next, whitespace-joined. Returns one mlData string per field,
    in document order."""
    fields = fillable_fields(soup)
    per_field_words = [tokenize_attributes(el, soup, type_marker, features)
                       for el in fields]
    out = []
    for i, words in enumerate(per_field_words):
        combined = list(words)
        if i > 0:
            combined += [ADJACENT_BEFORE_PREFIX + w for w in per_field_words[i - 1]]
        if i < len(per_field_words) - 1:
            combined += [ADJACENT_AFTER_PREFIX + w for w in per_field_words[i + 1]]
        out.append(" ".join(combined))
    return out
