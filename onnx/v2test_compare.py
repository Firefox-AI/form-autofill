#!/usr/bin/env python
"""Per-field accuracy/f1 on the real-site (v2) test set: our exported q8 models
vs the previously shipped models. Each model is scored on the real-site test in
its matching token format (ours=popup/optrange, shipped=no-popup/baseline; same
4749 fields). Writes a merged CSV keyed by field."""
import argparse, csv, json, os, sys
import numpy as np, onnxruntime as ort, torch, torch.nn.functional as F
from safetensors.numpy import load_file
from transformers import AutoTokenizer
from sklearn.metrics import precision_recall_fscore_support
REPO=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,REPO)
from dotraining import split_context, ignoreLineCount, fieldTypesReversedDict  # noqa
QF="model_quantized.onnx"

# name, type, quant/onnx dir, tokenizer dir (None=use enc dir), test file
MODELS=[
 ("our_bb_4sjmf","bb","onnx/quantization/autofill-tiny-supported-argo-autofillflow-4sjmf",
  "onnx/outputs/autofill-tiny-supported-argo-autofillflow-4sjmf","testing-optrange.txt"),
 ("our_multihead_2mrjb","triple","onnx/quantization/autofill-tiny-supported-argo-autofillflow-2mrjb",
  None,"testing-optrange.txt"),
 ("shipped_bb_mozilla","bb","onnx/outputs/mozilla-tinybert-address-autofill",
  "onnx/outputs/mozilla-tinybert-address-autofill","testing-baseline.txt"),
 ("shipped_multihead_sj546","triple","onnx/quantization/autofill-tiny-supported-argo-autofillflow-sj546",
  None,"testing-baseline.txt"),
]

def read_rows(path):
    rows=[]
    for line in open(os.path.join(REPO,path),encoding="utf-8"):
        line=line.strip()
        if not line: continue
        p=line.split(",",ignoreLineCount+1)
        if len(p)<=ignoreLineCount+1: continue
        try: lab=int(p[ignoreLineCount])
        except ValueError: continue
        rows.append((lab,p[ignoreLineCount+1],split_context(p[ignoreLineCount+1])))
    return rows

def _feeds(enc,names):
    f={"input_ids":enc["input_ids"].astype(np.int64),"attention_mask":enc["attention_mask"].astype(np.int64)}
    if "token_type_ids" in names: f["token_type_ids"]=np.zeros_like(f["input_ids"])
    return {k:v for k,v in f.items() if k in names}

def bb_predict(qdir,tokdir,rows,bs=64):
    sess=ort.InferenceSession(os.path.join(REPO,qdir,"onnx",QF),providers=["CPUExecutionProvider"])
    names={i.name for i in sess.get_inputs()}; tok=AutoTokenizer.from_pretrained(os.path.join(REPO,tokdir))
    preds=[]
    for i in range(0,len(rows),bs):
        enc=tok([r[1] for r in rows[i:i+bs]],truncation=True,max_length=512,padding=True,return_tensors="np")
        preds.extend(np.asarray(sess.run(None,_feeds(enc,names))[0]).argmax(-1).tolist())
    return preds

def triple_predict(split_dir,rows,bs=64):
    enc_dir=os.path.join(REPO,split_dir,"encoder")
    sess=ort.InferenceSession(os.path.join(enc_dir,"onnx",QF),providers=["CPUExecutionProvider"])
    names=[i.name for i in sess.get_inputs()]; tok=AutoTokenizer.from_pretrained(enc_dir)
    head={k:torch.from_numpy(v) for k,v in load_file(os.path.join(REPO,split_dir,"head","head.safetensors")).items()}
    hj=json.load(open(os.path.join(REPO,split_dir,"head","head.json")))
    def pooled(texts):
        out=[]
        for i in range(0,len(texts),bs):
            e=tok(texts[i:i+bs],truncation=True,max_length=512,padding=True,return_tensors="np")
            lhs=sess.run(None,_feeds(e,names))[0]; m=e["attention_mask"][...,None].astype(np.float32)
            out.append((lhs*m).sum(1)/np.clip(m.sum(1),1e-9,None))
        return torch.from_numpy(np.concatenate(out,0))
    cur=pooled([r[2][0] for r in rows]); prev=pooled([r[2][1] for r in rows]); nxt=pooled([r[2][2] for r in rows])
    with torch.no_grad():
        if "proj.weight" in head:
            cur=F.linear(cur,head["proj.weight"],head["proj.bias"]); prev=F.linear(prev,head["proj.weight"],head["proj.bias"]); nxt=F.linear(nxt,head["proj.weight"],head["proj.bias"])
        secs=[cur,prev,nxt]
        if hj.get("head_interactions"): secs+=[cur-prev,cur-nxt]
        h=F.gelu(F.linear(torch.cat(secs,-1),head["fusion.weight"],head["fusion.bias"]))
        logits=F.linear(h,head["classifier.weight"],head["classifier.bias"])
    return logits.argmax(-1).tolist()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",required=True); a=ap.parse_args()
    # labels + support from the baseline test (identical fields across formats)
    base=read_rows("testing-baseline.txt"); y_ref=[r[0] for r in base]
    labels=sorted(set(y_ref)); support={l:y_ref.count(l) for l in labels}
    res={}; overall={}
    for name,typ,qdir,tokdir,tf in MODELS:
        rows=read_rows(tf); y=[r[0] for r in rows]
        preds=triple_predict(qdir,rows) if typ=="triple" else bb_predict(qdir,tokdir,rows)
        overall[name]=float(np.mean(np.array(preds)==np.array(y)))
        _,rec,f1,_=precision_recall_fscore_support(y,preds,labels=labels,zero_division=0)
        res[name]=(rec,f1); print(f"  {name}: overall acc={overall[name]:.4f} (on {tf})")
    order=sorted(range(len(labels)),key=lambda i:-support[labels[i]])
    with open(os.path.join(REPO,a.out),"w",newline="") as fh:
        w=csv.writer(fh); hdr=["field","support"]
        for name,_,_,_,_ in MODELS: hdr+=[f"{name}_acc",f"{name}_f1"]
        w.writerow(hdr)
        for i in order:
            lab=labels[i]; row=[fieldTypesReversedDict.get(lab,str(lab)),support[lab]]
            for name,_,_,_,_ in MODELS:
                rec,f1=res[name]; row+=[round(float(rec[i]),4),round(float(f1[i]),4)]
            w.writerow(row)
        w.writerow([]); orow=["OVERALL",len(y_ref)]
        for name,_,_,_,_ in MODELS: orow+=[round(overall[name],4),""]
        w.writerow(orow)
    print("wrote",a.out)

if __name__=="__main__": main()
