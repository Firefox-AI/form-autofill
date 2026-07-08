#!/usr/bin/env python3
"""Build an interactive HTML report of label-validation errors.

Reads the per-field verdict JSONL produced by `evaluate_labels.py --dump` plus
the source HTML directory, and emits a single self-contained page:

  * left  — the forms that have errors, each expandable to its individual errors
  * right — the selected form rendered with ALL fields visible (hidden inputs
            revealed, the site's own CSS stripped so nothing is concealed), and
            the errored field(s) outlined in red with a badge (assigned -> suggested)

Usage:
  python build_error_report.py --dump verdicts.jsonl --dir /path/to/html \\
      --out error_report.html
  python -m http.server 8000            # then open http://localhost:8000/error_report.html
"""

from __future__ import annotations

import argparse
import ast
import collections
import html as _html
import json
import os
import sys

from bs4 import BeautifulSoup, Comment

from gen.validate import LABEL_ATTR, _find_dotraining, load_taxonomy

# Extra synonym pairs beyond dotraining's fieldNamesCloseDict — tokens the
# project treats (or should treat) as interchangeable, so the auditor flagging
# one as the other is not a real error.
_EXTRA_CLOSE = [("address-line2", "address-extra"), ("address-line3", "address-extra")]


def load_close_pairs() -> set:
    """Symmetric set of {a, b} token pairs considered equivalent — from
    dotraining.py's fieldNamesCloseDict (the model's own 'close accuracy'
    definition) plus _EXTRA_CLOSE."""
    src = open(_find_dotraining(), encoding="utf-8").read()
    close = {}
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", "") == "fieldNamesCloseDict" for t in node.targets):
            close = ast.literal_eval(node.value)
    pairs = {frozenset((a, b)) for a, lst in close.items() for b in lst}
    pairs |= {frozenset(p) for p in _EXTRA_CLOSE}
    return pairs

_STRIP_TAGS = ("script", "style", "link", "noscript", "iframe", "svg", "canvas")

# Clean, complete rendering for the form inside the iframe: everything visible,
# errored field in red. The site's own CSS is removed so nothing stays hidden.
_FORM_CSS = """
* { box-sizing: border-box; }
body { font: 14px system-ui, -apple-system, sans-serif; margin: 16px; color: #111; background: #fff; }
form, fieldset { max-width: 820px; }
fieldset { border: 1px solid #eee; margin: 10px 0; padding: 10px 12px; }
label { display: block; margin: 10px 0 2px; font-weight: 600; color: #333; }
span, td, th, p, li, div { color: #222; }
input, select, textarea { display: inline-block; min-width: 240px; max-width: 100%;
  padding: 4px 6px; margin: 2px 0; border: 1px solid #bbb; border-radius: 4px; font: inherit; }
button { margin: 6px 0; padding: 4px 10px; }
h1, h2, h3, h4 { font-size: 15px; margin: 12px 0 4px; }
.__eval_err { outline: 3px solid #e5484d !important; outline-offset: 1px;
  background: #ffecec !important; }
.__eval_badge { display: inline-block; margin: 3px 0 10px; padding: 2px 8px;
  background: #e5484d; color: #fff; border-radius: 4px; font-size: 12px; font-weight: 700; }
.__flash { animation: __f 1.3s ease; }
@keyframes __f { 30% { box-shadow: 0 0 0 8px rgba(229,72,77,.35); } }
/* hover chicklet: shows the field's autofill label (or unlabeled) */
.af-wrap { position: relative; display: inline-block; max-width: 100%; }
.af-wrap::after { content: attr(data-af); position: absolute; left: 0; top: calc(100% + 3px);
  z-index: 10; background: #1f2937; color: #fff; font: 600 11px system-ui, sans-serif;
  padding: 2px 8px; border-radius: 10px; white-space: nowrap; opacity: 0;
  pointer-events: none; transition: opacity .08s; }
.af-wrap:hover::after { opacity: 1; }
.af-wrap[data-af="(unlabeled)"]::after { background: #9ca3af; }
"""


def prepare_form_html(raw_html: str, errors: list[dict]) -> str:
    """Return an iframe srcdoc: the form with all fields visible and errored
    fields highlighted. `errors` is a list of {index, assigned, suggested, label}.
    """
    soup = BeautifulSoup(raw_html, "html.parser")

    for tag in soup.find_all(_STRIP_TAGS):
        tag.decompose()
    for c in soup.find_all(string=lambda s: isinstance(s, Comment)):
        c.extract()

    # Drop hidden form fields entirely (CSRF tokens etc. — noise in the report).
    for el in soup.find_all("input"):
        if (el.get("type") or "text").lower() == "hidden":
            el.decompose()
    # Reveal the rest: strip inline hiding + hidden attrs so real (CSS-hidden)
    # fields are visible for review.
    for el in soup.find_all(True):
        if el.has_attr("style"):
            del el["style"]
        if el.has_attr("hidden"):
            del el["hidden"]
        if el.get("aria-hidden") == "true":
            del el["aria-hidden"]

    # Highlight errored fields (by position among labeled elements).
    labeled = soup.select(f"[{LABEL_ATTR}]")
    for err in errors:
        i = err["index"]
        if not (0 <= i < len(labeled)):
            continue
        el = labeled[i]
        el["class"] = el.get("class", []) + ["__eval_err"]
        el["id"] = f"__err{i}"
        badge = soup.new_tag("div")
        badge["class"] = ["__eval_badge"]
        sug = err.get("suggested") or "?"
        badge.string = f"✗ {err['assigned']} → {sug}"
        el.insert_after(badge)

    # Wrap every control so hovering shows a chicklet with its label (autofill
    # type) or "(unlabeled)". Done last so it also wraps the highlighted fields.
    for el in soup.find_all(["input", "select", "textarea", "button"]):
        wrapper = soup.new_tag("span")
        wrapper["class"] = ["af-wrap"]
        wrapper["data-af"] = el.get(LABEL_ATTR) or "(unlabeled)"
        el.wrap(wrapper)

    body = soup.body or soup
    return (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<style>{_FORM_CSS}</style></head>{body}</html>"
            if soup.body else
            f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<style>{_FORM_CSS}</style></head><body>{soup}</body></html>")


PAGE_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Label validation errors — {title}</title>
<style>
  * {{ box-sizing: border-box; }}
  html, body {{ height: 100%; margin: 0; }}
  body {{ font: 14px system-ui, -apple-system, sans-serif; color: #111; display: flex; }}
  #sidebar {{ width: 360px; min-width: 300px; border-right: 1px solid #e4e4e7;
    display: flex; flex-direction: column; height: 100vh; }}
  #head {{ padding: 12px 14px; border-bottom: 1px solid #e4e4e7; }}
  #head h1 {{ font-size: 15px; margin: 0 0 6px; }}
  #head .sub {{ color: #666; font-size: 12px; }}
  #filter {{ width: 100%; margin-top: 8px; padding: 6px 8px; border: 1px solid #ccc; border-radius: 6px; font: inherit; }}
  #list {{ overflow-y: auto; flex: 1; }}
  .form-group {{ border-bottom: 1px solid #f0f0f1; }}
  .form-hdr {{ padding: 8px 14px; font-weight: 600; font-size: 13px; cursor: pointer;
    display: flex; justify-content: space-between; gap: 8px; }}
  .form-hdr:hover {{ background: #fafafa; }}
  .count {{ color: #e5484d; font-weight: 700; }}
  .err-row {{ padding: 6px 14px 6px 22px; font-size: 12px; color: #333; cursor: pointer;
    border-top: 1px dashed #f0f0f1; }}
  .err-row:hover {{ background: #fff5f5; }}
  .err-row.active {{ background: #ffe8e8; }}
  .err-row .t {{ font-weight: 600; }}
  .err-row .arrow {{ color: #e5484d; }}
  .err-row .lbl {{ color: #777; }}
  #detail {{ flex: 1; display: flex; flex-direction: column; height: 100vh; }}
  #detail-head {{ padding: 10px 16px; border-bottom: 1px solid #e4e4e7; background: #fafafa; }}
  #detail-head .fname {{ font-weight: 700; font-size: 14px; }}
  #detail-head .errs {{ margin-top: 4px; font-size: 12px; color: #555; }}
  #frame {{ flex: 1; border: 0; width: 100%; background: #fff; }}
  #empty {{ margin: auto; color: #999; }}
</style></head>
<body>
  <div id="sidebar">
    <div id="head">
      <h1>Label validation errors</h1>
      <div class="sub">{title} · {nforms} forms · {nerrors} errors</div>
      <input id="filter" placeholder="filter forms…">
    </div>
    <div id="list"></div>
  </div>
  <div id="detail">
    <div id="detail-head"><span class="fname">Select an error on the left</span>
      <div class="errs"></div></div>
    <iframe id="frame" sandbox="allow-same-origin"></iframe>
    <div id="empty"></div>
  </div>
<script>
const DATA = {data_json};
const listEl = document.getElementById('list');
const frame = document.getElementById('frame');
const dHead = document.querySelector('#detail-head .fname');
const dErrs = document.querySelector('#detail-head .errs');
let curFile = null, pending = null;

function scrollTo(idx) {{
  try {{
    const doc = frame.contentDocument;
    const el = doc.getElementById('__err' + idx);
    if (el) {{ el.scrollIntoView({{block:'center'}}); el.classList.remove('__flash');
      void el.offsetWidth; el.classList.add('__flash'); }}
  }} catch (e) {{}}
}}
frame.addEventListener('load', () => {{ if (pending != null) {{ scrollTo(pending); pending = null; }} }});

function show(fi, idx) {{
  const f = DATA[fi];
  document.querySelectorAll('.err-row.active').forEach(e => e.classList.remove('active'));
  const row = document.querySelector(`.err-row[data-f="${{fi}}"][data-i="${{idx}}"]`);
  if (row) row.classList.add('active');
  dHead.textContent = f.file;
  dErrs.innerHTML = f.errors.map(e =>
    `<div>field #${{e.index}}: <b>${{e.assigned}}</b> <span style="color:#e5484d">→ ${{e.suggested||'?'}}</span> · <span style="color:#777">${{(e.label||'').replace(/</g,'&lt;')}}</span></div>`).join('');
  if (curFile !== fi) {{ curFile = fi; pending = idx; frame.srcdoc = f.html; }}
  else {{ scrollTo(idx); }}
}}

function render(filter) {{
  listEl.innerHTML = '';
  DATA.forEach((f, fi) => {{
    if (filter && !f.file.toLowerCase().includes(filter)) return;
    const g = document.createElement('div'); g.className = 'form-group';
    const h = document.createElement('div'); h.className = 'form-hdr';
    h.innerHTML = `<span>${{f.file}}</span><span class="count">${{f.errors.length}}</span>`;
    h.onclick = () => show(fi, f.errors[0].index);
    g.appendChild(h);
    f.errors.forEach(e => {{
      const r = document.createElement('div'); r.className = 'err-row';
      r.dataset.f = fi; r.dataset.i = e.index;
      r.innerHTML = `<span class="t">${{e.assigned}}</span> <span class="arrow">→ ${{e.suggested||'?'}}</span><br><span class="lbl">${{(e.label||'').replace(/</g,'&lt;')||'(no label)'}}</span>`;
      r.onclick = () => show(fi, e.index);
      g.appendChild(r);
    }});
    listEl.appendChild(g);
  }});
}}
document.getElementById('filter').addEventListener('input', ev => render(ev.target.value.toLowerCase().trim()));
render('');
</script>
</body></html>
"""


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dump", required=True, help="verdict JSONL from evaluate_labels.py --dump")
    p.add_argument("--dir", required=True, help="directory of the source HTML forms")
    p.add_argument("--out", default="error_report.html")
    p.add_argument("--include-unsure", action="store_true",
                   help="also show 'unsure' verdicts, not just 'incorrect'")
    p.add_argument("--keep-close", action="store_true",
                   help="do NOT drop close-match/synonym pairs (fieldNamesCloseDict)")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    keep = {"incorrect"} | ({"unsure"} if args.include_unsure else set())

    def actionable(rec: dict) -> bool:
        # Drop judge "hedging": an 'incorrect' that suggests the same token (or
        # none) is non-actionable noise, not a real relabeling.
        if rec.get("verdict") == "incorrect":
            s = (rec.get("suggested") or "").strip()
            return bool(s) and s != rec["assigned"]
        return True

    def is_password(rec: dict) -> bool:
        # Password is out of scope (not implemented), so ignore any error whose
        # assigned or suggested token is password-related.
        blob = (rec.get("assigned", "") + " " + (rec.get("suggested") or "")).lower()
        return "password" in blob

    close_pairs = set() if args.keep_close else load_close_pairs()
    valid_tokens = set(load_taxonomy())

    def is_close(rec: dict) -> bool:
        return frozenset((rec["assigned"], (rec.get("suggested") or "").strip())) in close_pairs

    def bad_suggestion(rec: dict) -> bool:
        # The auditor sometimes returns an explanatory sentence instead of a
        # token. A real relabel is a single valid taxonomy token; anything else
        # (spaces, prose, or a non-taxonomy word like 'text') is not actionable.
        s = (rec.get("suggested") or "").strip()
        return bool(s) and (" " in s or s not in valid_tokens)

    by_file: dict[str, list[dict]] = collections.defaultdict(list)
    dropped = dropped_pw = dropped_close = dropped_bad = 0
    for line in open(args.dump, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("verdict") not in keep:
            continue
        if is_password(rec):
            dropped_pw += 1
        elif is_close(rec):
            dropped_close += 1
        elif bad_suggestion(rec):
            dropped_bad += 1
        elif actionable(rec):
            by_file[rec["file"]].append(rec)
        else:
            dropped += 1
    if dropped:
        print(f"  (dropped {dropped} non-actionable same-token/empty 'incorrect' verdicts)")
    if dropped_pw:
        print(f"  (dropped {dropped_pw} password-related verdicts — out of scope)")
    if dropped_close:
        print(f"  (dropped {dropped_close} close-match/synonym pairs — treated as equivalent)")
    if dropped_bad:
        print(f"  (dropped {dropped_bad} verdicts with non-token/prose suggestions)")

    forms = []
    n_errors = 0
    for fname in sorted(by_file):
        path = os.path.join(args.dir, fname)
        if not os.path.exists(path):
            print(f"  ! source not found, skipping: {fname}", file=sys.stderr)
            continue
        errors = sorted(by_file[fname], key=lambda r: r["index"])
        n_errors += len(errors)
        raw = open(path, encoding="utf-8", errors="replace").read()
        forms.append({
            "file": fname,
            "errors": [{"index": e["index"], "assigned": e["assigned"],
                        "suggested": e.get("suggested", ""), "label": e.get("label", "")}
                       for e in errors],
            "html": prepare_form_html(raw, errors),
        })

    title = os.path.basename(os.path.normpath(args.dir))
    page = PAGE_TEMPLATE.format(
        title=_html.escape(title), nforms=len(forms), nerrors=n_errors,
        data_json=json.dumps(forms, ensure_ascii=False))
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(page)
    print(f"Wrote {args.out}: {len(forms)} forms, {n_errors} errors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
