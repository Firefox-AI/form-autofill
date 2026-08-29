"""Does training on non-address negatives improve rejection?
Eval models trained WITHOUT non-address (v1-dataset base arm) vs WITH
non-address (v3-dataset base arm) on the held-out non-address test slice only.
Base arm (no option-range) so we isolate the non-address effect."""
import io, os, json, zipfile, random, contextlib, re, ast
from collections import defaultdict, Counter
from metaflow import Run, namespace
from dotraining import Config, evaluate_model, CLASSIFICATION_REPORT
namespace(None)

FT=ast.literal_eval(re.search(r"fieldTypesDict\s*=\s*(\{.*?\n\})",open("dotraining.py").read(),re.S).group(1))
rev={v:k for k,v in FT.items()}

# reconstruct the non-address test split (same seed/logic as v3 assembly)
def load(p):
    d=defaultdict(list)
    for ln in open(p,encoding="utf-8"):
        ln=ln.rstrip("\n")
        if ln: d[ln.split(",",1)[0]].append(ln)
    return d
na_base=load("data_tools/ffgen/non_address.baseline.ffgen.txt")
na_opt =load("data_tools/ffgen/non_address.optrange.ffgen.txt")
files=sorted(f for f in na_opt if len(na_opt[f])>=3 and f in na_base)
rng=random.Random(20260826); sh=files[:]; rng.shuffle(sh)
n=len(sh); n_tr=int(n*0.8); n_va=int(n*0.1)
test_files=set(sh[n_tr+n_va:])
test_rows=[r for f in files if f in test_files for r in na_base[f]]  # base format
open("testing-nateb.txt","w").write("\n".join(test_rows)+"\n")
comp=Counter(rev.get(int(r.split(",",3)[2]),"?") for r in test_rows)
print(f"non-address TEST slice: {len(test_files)} files, {len(test_rows)} rows  labels={dict(comp)}", flush=True)

TRI=dict(contextFormat="triple", encoderLayers=4, headInteractions=True, headProjDim=0)
# (base label, model params) ; run-ids: trained-WITHOUT vs trained-WITH non-address
BASES={
 "bb":       (dict(modelName="huawei-noah/TinyBERT_General_4L_312D", contextFormat="bb"), "qlnt5","t8vl7"),
 "tri-v2":   (dict(modelName="rolf-mozilla/minilm-pruned-notest-v2", **TRI),              "4s8fm","6r99j"),
 "tri-unpr": (dict(modelName="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", **TRI),"s5hv2","swft8"),
 "tri-v3":   (dict(modelName="rolf-mozilla/minilm-pruned-notest-v3", **TRI),              "mwtc8","j4jjx"),
}
def eval_run(rid, params):
    cfg=Config(modelSuffix=f"naeval-{rid}", dataVariant="-nateb", **params)
    blob=Run(f"AutofillFlow/argo-autofillflow-{rid}").data.model_artifact
    os.makedirs(cfg.saveModelDir,exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(blob)) as zf: zf.extractall(cfg.saveModelDir)
    with contextlib.redirect_stdout(io.StringIO()):
        m=evaluate_model(cfg,"testing")
    rep=m[CLASSIFICATION_REPORT]
    other_rec=rep.get("other",{}).get("recall")   # rejection rate on negatives
    # positive recall = weighted over labeled (non-other) classes
    pos=[(k,v) for k,v in rep.items() if isinstance(v,dict) and k not in ("other","accuracy","macro avg","weighted avg")]
    tot=sum(v.get("support",0) for _,v in pos)
    pos_rec=sum(v["recall"]*v.get("support",0) for _,v in pos)/tot if tot else None
    return m.get("accuracy"), other_rec, pos_rec

out={}
for base,(params,r_without,r_with) in BASES.items():
    aw,ow,pw=eval_run(r_without,params); print(f"{base} WITHOUT non-addr ({r_without}) done",flush=True)
    ay,oy,py=eval_run(r_with,params);    print(f"{base} WITH non-addr    ({r_with}) done",flush=True)
    out[base]={"without":{"acc":aw,"other_recall":ow,"pos_recall":pw},
               "with":{"acc":ay,"other_recall":oy,"pos_recall":py}}
json.dump(out,open("/tmp/nonaddr_rejection.json","w"),indent=2)
print("WROTE /tmp/nonaddr_rejection.json",flush=True)
