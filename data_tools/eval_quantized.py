#!/usr/bin/env python
"""Evaluate each ONNX quantization of the autofill model against the test set.

For every ``*.onnx`` file in ``<model_dir>/onnx`` this runs the test data
through ONNX Runtime, compares the argmax prediction to the expected label, and
writes the high-level metrics (kappa, accuracy, weighted/balanced accuracy) for
each quantization to a CSV.

The metric definitions mirror dotraining.compute_standard_metrics so the
numbers are directly comparable to the PyTorch evaluation: weighted averages use
sklearn ``average="weighted"`` and kappa is Cohen's kappa.

Run it with the transformers.js eval venv (which has onnxruntime), as in the
README -- scikit-learn must be installed into that venv:

    /tmp/transformers.js/scripts/venv/bin/pip install scikit-learn
    /tmp/transformers.js/scripts/venv/bin/python eval_quantized.py \
        --model_dir quantization/autofill-tiny-supported-argo-autofillflow-kjth9 \
        --test_file testing-supported.txt \
        --output_csv quantization/autofill-tiny-supported-argo-autofillflow-kjth9/quantization_eval.csv
"""

import argparse
import csv
import glob
import os
import time

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
)

# The dataset rows are "<filename>,<fieldname>,<label_int>,<text>"; the first
# two columns are reference-only, matching dotraining.ignoreLineCount = 2.
IGNORE_LINE_COUNT = 2

# Maps the ONNX filename suffix (model_<suffix>.onnx) to the quantization label.
# "model.onnx" (no suffix) is the unquantized fp32 export.
SUFFIX_TO_MODE = {
    "": "fp32",
    "fp16": "fp16",
    "quantized": "q8",
    "int8": "int8",
    "uint8": "uint8",
    "q4": "q4",
    "q4f16": "q4f16",
    "bnb4": "bnb4",
}


def read_test_file(path):
    """Return (texts, expected_label_ints) from a *-supported.txt test file."""
    texts, labels = [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", IGNORE_LINE_COUNT + 1)
            try:
                labels.append(int(parts[IGNORE_LINE_COUNT]))
                texts.append(parts[IGNORE_LINE_COUNT + 1])
            except (IndexError, ValueError):
                # Skip malformed lines (same tolerance as dotraining.readFile,
                # which only guards the label parse).
                print(f"  skipping malformed line: {line!r}")
    return texts, labels


def mode_for(onnx_path):
    """Derive the quantization label from an onnx filename."""
    stem = os.path.splitext(os.path.basename(onnx_path))[0]  # e.g. model_int8
    suffix = stem[len("model"):].lstrip("_")                 # e.g. int8 (or "")
    return SUFFIX_TO_MODE.get(suffix, suffix or "fp32")


def predict(session, tokenizer, texts, batch_size=32):
    """Return the argmax class index for every text using the ONNX session."""
    input_names = {i.name for i in session.get_inputs()}
    preds = np.empty(len(texts), dtype=np.int64)
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        enc = tokenizer(
            batch,
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors="np",
        )
        feeds = {"input_ids": enc["input_ids"].astype(np.int64),
                 "attention_mask": enc["attention_mask"].astype(np.int64)}
        if "token_type_ids" in input_names:
            tt = enc.get("token_type_ids")
            if tt is None:
                tt = np.zeros_like(enc["input_ids"])
            feeds["token_type_ids"] = tt.astype(np.int64)
        # Only pass inputs the model actually declares.
        feeds = {k: v for k, v in feeds.items() if k in input_names}
        logits = session.run(None, feeds)[0]
        preds[start:start + len(batch)] = np.asarray(logits).argmax(axis=-1)
    return preds


def evaluate(onnx_path, tokenizer, texts, expected):
    """Run one ONNX model; return (metrics_row, predictions or None on error)."""
    mode = mode_for(onnx_path)
    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    row = {
        "quantization": mode,
        "onnx_file": os.path.basename(onnx_path),
        "size_mb": round(size_mb, 2),
        "num_samples": len(texts),
        "kappa": "",
        "accuracy": "",
        "balanced_accuracy": "",
        "weighted_precision": "",
        "weighted_recall": "",
        "weighted_f1": "",
        "infer_seconds": "",
        "status": "ok",
        "error": "",
    }
    preds = None
    try:
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        t0 = time.time()
        preds = predict(sess, tokenizer, texts)
        row["infer_seconds"] = round(time.time() - t0, 2)
        y_true, y_pred = expected, preds
        row["kappa"] = round(float(cohen_kappa_score(y_true, y_pred)), 6)
        row["accuracy"] = round(float(accuracy_score(y_true, y_pred)), 6)
        row["balanced_accuracy"] = round(float(balanced_accuracy_score(y_true, y_pred)), 6)
        row["weighted_precision"] = round(float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), 6)
        row["weighted_recall"] = round(float(recall_score(y_true, y_pred, average="weighted", zero_division=0)), 6)
        row["weighted_f1"] = round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 6)
    except Exception as exc:  # noqa: BLE001 - record per-model failures, keep going
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"
        preds = None
    return row, preds


def load_id2label(model_dir, onnx_dir):
    """Return {int_label: field_name} from config.json (onnx dir, then model dir)."""
    import json
    for cfg_path in (os.path.join(onnx_dir, "config.json"),
                     os.path.join(model_dir, "config.json")):
        if os.path.exists(cfg_path):
            id2label = json.load(open(cfg_path)).get("id2label", {})
            return {int(k): v for k, v in id2label.items()}
    return {}


def per_field_accuracy(expected, preds, id2label):
    """Per true field: (accuracy, support). Accuracy = fraction predicted correctly."""
    import numpy as np
    stats = {}
    expected = np.asarray(expected)
    preds = np.asarray(preds)
    for label in sorted(set(expected.tolist())):
        name = id2label.get(int(label), str(label))
        mask = expected == label
        support = int(mask.sum())
        acc = float((preds[mask] == label).mean()) if support else 0.0
        stats[name] = (round(acc, 6), support)
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_dir", required=True,
                    help="Model dir containing an onnx/ subfolder and tokenizer files.")
    ap.add_argument("--test_file", required=True,
                    help="Path to the *-supported.txt test dataset.")
    ap.add_argument("--output_csv", default=None,
                    help="Where to write the summary CSV (default: <model_dir>/quantization_eval.csv).")
    ap.add_argument("--field_csv", default=None,
                    help="Where to write the per-field-accuracy CSV "
                         "(default: <model_dir>/quantization_field_accuracy.csv).")
    args = ap.parse_args()

    onnx_dir = os.path.join(args.model_dir, "onnx")
    output_csv = args.output_csv or os.path.join(args.model_dir, "quantization_eval.csv")
    field_csv = args.field_csv or os.path.join(args.model_dir, "quantization_field_accuracy.csv")

    # Tokenizer files live in the onnx/ folder (written by the ONNX export); fall
    # back to the model dir itself.
    tok_dir = onnx_dir if os.path.exists(os.path.join(onnx_dir, "tokenizer.json")) else args.model_dir
    tokenizer = AutoTokenizer.from_pretrained(tok_dir)

    id2label = load_id2label(args.model_dir, onnx_dir)
    texts, expected = read_test_file(args.test_file)
    print(f"Loaded {len(texts)} test rows from {args.test_file}")

    onnx_files = sorted(glob.glob(os.path.join(onnx_dir, "*.onnx")))
    if not onnx_files:
        raise SystemExit(f"No .onnx files found in {onnx_dir}")

    rows = []
    field_by_mode = {}   # mode -> {field_name: (accuracy, support)}
    for path in onnx_files:
        print(f"Evaluating {mode_for(path):>6} ({os.path.basename(path)}) ...", flush=True)
        row, preds = evaluate(path, tokenizer, texts, expected)
        if row["status"] == "ok":
            print(f"   kappa={row['kappa']}  acc={row['accuracy']}  "
                  f"weighted_f1={row['weighted_f1']}  balanced_acc={row['balanced_accuracy']}")
            field_by_mode[row["quantization"]] = per_field_accuracy(expected, preds, id2label)
        else:
            print(f"   FAILED: {row['error']}")
        rows.append(row)

    # Order rows fp32 -> fp16 -> q8 -> int8 -> uint8 -> q4 -> q4f16 -> bnb4.
    order = list(SUFFIX_TO_MODE.values())
    rows.sort(key=lambda r: order.index(r["quantization"]) if r["quantization"] in order else 99)

    fieldnames = list(rows[0].keys())
    with open(output_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} summary rows to {output_csv}")

    # Per-field accuracy CSV: one row per field, a column per quantization mode,
    # plus the support (number of test samples for that field).
    modes = [m for m in order if m in field_by_mode]
    all_fields = sorted({f for stats in field_by_mode.values() for f in stats})
    with open(field_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["field", "support"] + modes)
        for field in all_fields:
            # support is the same across modes (it depends only on the test set).
            support = next((field_by_mode[m][field][1] for m in modes if field in field_by_mode[m]), 0)
            accs = [field_by_mode[m].get(field, ("", ""))[0] for m in modes]
            writer.writerow([field, support] + accs)
    print(f"Wrote {len(all_fields)} field rows to {field_csv}")


if __name__ == "__main__":
    main()
