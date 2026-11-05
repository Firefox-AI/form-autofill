"""Eval a quantized bb (standard classifier) ONNX end-to-end vs its fp32 export.
Usage: eval_int8_bb.py <quant_dir> <test_file>
<quant_dir> holds onnx/model.onnx + model_int8.onnx and the tokenizer/config."""
import sys, os, numpy as np, onnxruntime as ort
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoTokenizer
from dotraining import ignoreLineCount, strip_type_marker, reformat_context, Config

qdir, test_file = sys.argv[1], sys.argv[2]
# bb uses its base model's tokenizer (the quant dir may omit tokenizer.json).
base = sys.argv[3] if len(sys.argv) > 3 else "rolf-mozilla/minilm-pruned-notest-iaor"
tok = AutoTokenizer.from_pretrained(base)
cfg = Config(contextFormat="bb")
rows = [l.rstrip("\n") for l in open(test_file, encoding="utf-8") if l.strip()]
texts, labels = [], []
for l in rows:
    d = l.split(",", ignoreLineCount + 1)
    labels.append(int(d[ignoreLineCount]))
    texts.append(reformat_context(strip_type_marker(d[ignoreLineCount + 1], cfg), "bb"))
labels = np.array(labels)
for name in ("model.onnx", "model_int8.onnx"):
    p = os.path.join(qdir, "onnx", name)
    if not os.path.exists(p):
        print(f"{name}: missing"); continue
    sess = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
    innames = [i.name for i in sess.get_inputs()]
    correct = 0
    for i in range(0, len(texts), 64):
        enc = tok(texts[i:i + 64], truncation=True, max_length=512, padding=True, return_tensors="np")
        feed = {k: enc[k].astype(np.int64) for k in innames if k in enc}
        if "token_type_ids" in innames and "token_type_ids" not in feed:
            feed["token_type_ids"] = np.zeros_like(feed["input_ids"])
        pred = sess.run(None, feed)[0].argmax(-1)
        correct += int((pred == labels[i:i + 64]).sum())
    print(f"{name:20s} {os.path.getsize(p)/1e6:6.1f}MB  acc={correct/len(labels):.4f}")
