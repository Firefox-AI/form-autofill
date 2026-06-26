#!/usr/bin/env python
"""Compare per-field accuracy across models for a chosen quantization.

Joins the ``quantization_field_accuracy.csv`` files produced by
eval_quantized.py (one per model) on the field name, pulling out a single
quantization column from each, and writes a side-by-side comparison CSV. When
exactly two models are given, a ``delta`` column (second minus first) is added.

Example:

    python compare_field_accuracy.py --variant fp32 \
        --output field_accuracy_jt4qd_vs_server.csv \
        jt4qd=quantization/autofill-tiny-supported-argo-autofillflow-jt4qd/quantization_field_accuracy.csv \
        server_v0.1.0=/Users/Rrando/autofill_model_from_server/v0.1.0/quantization_field_accuracy.csv
"""

import argparse
import csv


def read_field_csv(path, variant):
    """Return ({field: accuracy_float}, {field: support_int}) for one model."""
    accs, support = {}, {}
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        if variant not in reader.fieldnames:
            raise SystemExit(
                f"Variant '{variant}' not in {path}. Available: "
                f"{[c for c in reader.fieldnames if c not in ('field', 'support')]}")
        for r in reader:
            field = r["field"]
            val = r[variant]
            accs[field] = float(val) if val not in ("", None) else None
            support[field] = int(r["support"])
    return accs, support


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+",
                    help="name=path entries pointing at quantization_field_accuracy.csv files.")
    ap.add_argument("--variant", default="fp32",
                    help="Quantization column to compare on (default: fp32).")
    ap.add_argument("--output", required=True, help="Path to write the comparison CSV.")
    ap.add_argument("--sort-by", default="support", choices=["support", "delta", "field"],
                    help="Sort order for the rows (default: support, descending).")
    args = ap.parse_args()

    names, data, support = [], {}, {}
    for entry in args.models:
        if "=" not in entry:
            raise SystemExit(f"Expected name=path, got: {entry}")
        name, path = entry.split("=", 1)
        names.append(name)
        data[name], sup = read_field_csv(path, args.variant)
        support.update(sup)  # support is identical across models (same test set)

    fields = sorted(support, key=lambda f: support[f])
    rows = []
    for field in fields:
        row = {"field": field, "support": support[field]}
        for name in names:
            row[name] = data[name].get(field, "")
        if len(names) == 2:
            a, b = data[names[0]].get(field), data[names[1]].get(field)
            row["delta"] = round(b - a, 6) if (a is not None and b is not None) else ""
        rows.append(row)

    if args.sort_by == "support":
        rows.sort(key=lambda r: r["support"], reverse=True)
    elif args.sort_by == "delta" and len(names) == 2:
        rows.sort(key=lambda r: (r["delta"] if isinstance(r["delta"], (int, float)) else 0))
    else:
        rows.sort(key=lambda r: r["field"])

    fieldnames = ["field", "support"] + names + (["delta"] if len(names) == 2 else [])
    with open(args.output, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} field rows comparing {names} on '{args.variant}' -> {args.output}")


if __name__ == "__main__":
    main()
