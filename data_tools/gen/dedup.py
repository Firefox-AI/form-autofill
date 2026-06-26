"""Near-duplicate detection over the sample corpus.

A form's signature is a hash of (sorted field types, ordered lowercased label
texts). Rejecting signature collisions prevents thousands of near-identical
synthetic forms AND guards against accidentally minting a near-copy of a
testing/ file into the training set (an eval-integrity leak).
"""

from __future__ import annotations

import glob
import hashlib
import os

from bs4 import BeautifulSoup

from gen.validate import LABEL_ATTR


def _signature_from_pairs(type_label_pairs: list[tuple[str, str]]) -> str:
    types = ",".join(sorted(t for t, _ in type_label_pairs))
    labels = "|".join(lbl.lower().strip() for _, lbl in type_label_pairs)
    return hashlib.sha1(f"{types}##{labels}".encode("utf-8")).hexdigest()


def signature_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    pairs: list[tuple[str, str]] = []
    for el in soup.select(f"[{LABEL_ATTR}]"):
        autofill_type = el.get(LABEL_ATTR, "")
        # Best-effort nearby label: preceding span/label text or placeholder.
        label = ""
        prev = el.find_previous(["span", "label"])
        if prev and prev.get_text(strip=True):
            label = prev.get_text(strip=True)
        elif el.get("placeholder"):
            label = el.get("placeholder")
        elif el.get("aria-label"):
            label = el.get("aria-label")
        pairs.append((autofill_type, label))
    return _signature_from_pairs(pairs)


def signature_from_spec(spec: dict) -> str:
    pairs = [
        (f["autofill_type"], f.get("label_text") or "")
        for f in spec.get("fields", [])
        if f.get("autofill_type") and f["autofill_type"] != "__nonfill__"
    ]
    return _signature_from_pairs(pairs)


class DedupIndex:
    def __init__(self) -> None:
        self._seen: set[str] = set()

    def add(self, signature: str) -> None:
        self._seen.add(signature)

    def is_duplicate(self, signature: str) -> bool:
        return signature in self._seen

    def add_if_new(self, signature: str) -> bool:
        """Return True if added (new), False if it was already a duplicate."""
        if signature in self._seen:
            return False
        self._seen.add(signature)
        return True

    @property
    def size(self) -> int:
        return len(self._seen)

    def load_corpus(self, samples_dir: str) -> int:
        """Index every existing sample HTML so we never duplicate or leak them."""
        count = 0
        for path in glob.glob(os.path.join(samples_dir, "**", "*.html"), recursive=True):
            try:
                with open(path, encoding="utf-8", errors="replace") as fh:
                    self._seen.add(signature_from_html(fh.read()))
                    count += 1
            except OSError:
                continue
        return count
