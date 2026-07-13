#!/usr/bin/env python
"""Per-class (support, accuracy=recall, f1) for the int8 variant of each model.

Runs the int8 ONNX of every model (triple = encoder+head split; bb = standard
classifier) over a test set, then writes a wide CSV: one row per field class
with support and, per model, accuracy(recall) + f1. int8 only.

    uv run --with onnxruntime python onnx/classwise_int8.py --test-file testing-supported.txt \
        --out class_wise_int8_comparison.csv
"""
import argparse, csv, json, os, sys
import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from safetensors.numpy import load_file
from transformers import AutoTokenizer
from sklearn.metrics import precision_recall_fscore_support

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from dotraining import split_context, ignoreLineCount, fieldTypesReversedDict  # noqa: E402

# (label, quant filename) -> use the q8 export
QF = "model_quantized.onnx"

MODELS = [
    ("sj546",         "triple", "quantization/autofill-tiny-supported-argo-autofillflow-sj546"),
    ("2xsjf",         "triple", "quantization/autofill-tiny-supported-argo-autofillflow-2xsjf"),
    ("grqhq",         "triple", "quantization/autofill-tiny-supported-argo-autofillflow-grqhq"),
    ("f7wzn",         "bb",     "outputs/autofill-tiny-supported-argo-autofillflow-f7wzn"),
    ("sbsd5",         "bb",     "outputs/autofill-tiny-supported-argo-autofillflow-sbsd5"),
    ("scjmd",         "bb",     "outputs/autofill-tiny-supported-argo-autofillflow-scjmd"),
    ("mozilla-1.0.1", "bb",     "outputs/mozilla-tinybert-1.0.1"),
]


def read_rows(path):
    rows = []
    with open(os.path.join(REPO, path), encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", ignoreLineCount + 1)
            if len(parts) <= ignoreLineCount + 1:
                continue
            try:
                lab = int(parts[ignoreLineCount])
            except ValueError:
                continue
            raw = parts[ignoreLineCount + 1]
            rows.append((lab, raw, split_context(raw)))
    return rows


def _feeds(enc, in_names):
    f = {"input_ids": enc["input_ids"].astype(np.int64),
         "attention_mask": enc["attention_mask"].astype(np.int64)}
    if "token_type_ids" in in_names:
        f["token_type_ids"] = np.zeros_like(f["input_ids"])
    return {k: v for k, v in f.items() if k in in_names}


def bb_predict(model_dir, rows, bs=64):
    sess = ort.InferenceSession(os.path.join(REPO, model_dir, "onnx", QF), providers=["CPUExecutionProvider"])
    names = {i.name for i in sess.get_inputs()}
    tok = AutoTokenizer.from_pretrained(os.path.join(REPO, model_dir))
    texts = [r[1] for r in rows]
    preds = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], truncation=True, max_length=512, padding=True, return_tensors="np")
        logits = sess.run(None, _feeds(enc, names))[0]
        preds.extend(np.asarray(logits).argmax(-1).tolist())
    return preds


def triple_predict(split_dir, rows, bs=64):
    enc_dir = os.path.join(REPO, split_dir, "encoder")
    sess = ort.InferenceSession(os.path.join(enc_dir, "onnx", QF), providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]
    tok = AutoTokenizer.from_pretrained(enc_dir)
    head = {k: torch.from_numpy(v) for k, v in load_file(os.path.join(REPO, split_dir, "head", "head.safetensors")).items()}
    hj = json.load(open(os.path.join(REPO, split_dir, "head", "head.json")))

    def pooled(texts):
        out = []
        for i in range(0, len(texts), bs):
            e = tok(texts[i:i + bs], truncation=True, max_length=512, padding=True, return_tensors="np")
            lhs = sess.run(None, _feeds(e, names))[0]
            m = e["attention_mask"][..., None].astype(np.float32)
            out.append((lhs * m).sum(1) / np.clip(m.sum(1), 1e-9, None))
        return torch.from_numpy(np.concatenate(out, 0))

    cur = pooled([r[2][0] for r in rows]); prev = pooled([r[2][1] for r in rows]); nxt = pooled([r[2][2] for r in rows])
    with torch.no_grad():
        if "proj.weight" in head:
            cur = F.linear(cur, head["proj.weight"], head["proj.bias"])
            prev = F.linear(prev, head["proj.weight"], head["proj.bias"])
            nxt = F.linear(nxt, head["proj.weight"], head["proj.bias"])
        secs = [cur, prev, nxt]
        if hj.get("head_interactions"):
            secs += [cur - prev, cur - nxt]
        h = F.gelu(F.linear(torch.cat(secs, -1), head["fusion.weight"], head["fusion.bias"]))
        logits = F.linear(h, head["classifier.weight"], head["classifier.bias"])
    return logits.argmax(-1).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-file", default="testing-supported.txt")
    ap.add_argument("--out", default="class_wise_int8_comparison.csv")
    args = ap.parse_args()

    rows = read_rows(args.test_file)
    y_true = [r[0] for r in rows]
    labels = sorted(set(y_true))
    support = {l: y_true.count(l) for l in labels}
    print(f"{len(rows)} rows, {len(labels)} classes")

    metrics = {}
    for name, typ, path in MODELS:
        preds = triple_predict(path, rows) if typ == "triple" else bb_predict(path, rows)
        _, recall, f1, _ = precision_recall_fscore_support(y_true, preds, labels=labels, zero_division=0)
        metrics[name] = (recall, f1)
        print(f"  scored {name}")

    order = sorted(range(len(labels)), key=lambda i: -support[labels[i]])
    with open(os.path.join(REPO, args.out), "w", newline="") as fh:
        w = csv.writer(fh)
        header = ["class", "support"]
        for name, _, _ in MODELS:
            header += [f"{name}_acc", f"{name}_f1"]
        w.writerow(header)
        for i in order:
            lab = labels[i]
            row = [fieldTypesReversedDict.get(lab, str(lab)), support[lab]]
            for name, _, _ in MODELS:
                r, f = metrics[name]
                row += [round(float(r[i]), 4), round(float(f[i]), 4)]
            w.writerow(row)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
