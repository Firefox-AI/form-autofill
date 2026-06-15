"""Form-context classification: a second-stage model over neighbouring fields.

The base model in dotraining.py classifies each form field in isolation. This
module builds a *context* model that looks at each field together with its
neighbours within the same form, to test whether that improves accuracy.

Pipeline:
  1. Group the dataset rows into forms (column 0 -- the source filename -- is
     the form id) preserving the in-file element order.
  2. Run the trained base model over every element to get a probability vector
     over the field-type classes.
  3. For each form, stack the per-element probability vectors into a matrix and
     append two extra columns -- a 'first item' and a 'last item' flag. Then pad
     the stack with `window` all-zero marker rows on each side (leading rows
     flagged 'first item', trailing rows flagged 'last item') so boundary
     elements still have a full window of neighbours.
  4. Build a feature row per real element by concatenating its own vector with
     its +/-window neighbours, in fixed left-to-right order so the classifier
     always knows which block is to the left vs the right. window=1 gives the
     prev/current/next trio; larger windows look further out.
  5. Train MLP / Logistic Regression / Random Forest on the training forms and
     evaluate on the test forms, comparing against the no-context base model.

The dataset files do not ship a separate per-form id or element-order column,
so the form id is the filename and the order is the in-file row order. See
build_datasets() / validate_forms() which check that grouping/ordering is sane.
"""

import io
import os
import urllib.request
import zipfile

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from dotraining import (
    Config,
    DATA_BASE_URL,
    compute_standard_metrics,
    ensure_dataset,
    fieldTypesDict,
    fieldTypesReversedDict,
    ignoreLineCount,
)

# Per-element vector layout: one probability per field-type class, then the two
# extra position-marker dimensions.
NUM_CLASSES = len(fieldTypesDict)        # 66
FIRST_ITEM = NUM_CLASSES                 # index of the 'first item' flag
LAST_ITEM = NUM_CLASSES + 1              # index of the 'last item' flag
ROW_DIM = NUM_CLASSES + 2                # 68
DEFAULT_WINDOW = 1                       # neighbours considered on each side
                                         # (window=1 -> the prev/current/next trio)

# The three second-stage methods we compare.
MODEL_KINDS = ["mlp", "logreg", "rf"]

# Files needed to load the base classifier for inference. Downloaded from the
# source repo (same place as the datasets) when not already present locally.
MODEL_FILES = [
    "config.json",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
]


def ensure_model(cfg):
  """Ensure the base model directory is present, downloading it if needed."""
  os.makedirs(cfg.saveModelDir, exist_ok=True)
  for name in MODEL_FILES:
    local = os.path.join(cfg.saveModelDir, name)
    if not os.path.exists(local):
      url = DATA_BASE_URL + cfg.saveModelDir + "/" + name
      print("Downloading " + url + " -> " + local)
      urllib.request.urlretrieve(url, local)
  return cfg.saveModelDir


def model_from_run(cfg, flow_name="AutofillFlow", run_id="", ns=""):
  """Unzip the base model from an AutofillFlow run's `model_artifact`.

  Uses the latest successful run of `flow_name` when `run_id` is empty. The
  artifact is the zip the AutofillFlow train step stored, so extracting it into
  cfg.saveModelDir reconstructs the model files the same way that flow saved
  them. `ns` selects the Metaflow namespace -- needed because a remotely-trained
  run lives in a production namespace, not the caller's user namespace. Returns
  the pathspec of the run the model came from.
  """
  from metaflow import Flow, Run, namespace
  if ns:
    namespace(ns)
  run = Run(f"{flow_name}/{run_id}") if run_id else Flow(flow_name).latest_successful_run
  if run is None:
    raise RuntimeError(f"no successful run found for {flow_name}")
  os.makedirs(cfg.saveModelDir, exist_ok=True)
  with zipfile.ZipFile(io.BytesIO(run.data.model_artifact)) as zf:
    zf.extractall(cfg.saveModelDir)
  print(f"Loaded base model from {run.pathspec} -> {cfg.saveModelDir}")
  return run.pathspec


def read_form_rows(filename, cfg):
  """Return [(form_id, label_id, text)] for a dataset file, preserving order."""
  rows = []
  path = ensure_dataset(filename + cfg.dataVariant + ".txt")
  for line in open(path, encoding="utf-8"):
    line = line.strip()
    if not line:
      continue
    parts = line.split(",", ignoreLineCount + 1)
    form_id = parts[0]
    label_id = int(parts[ignoreLineCount])
    text = parts[ignoreLineCount + 1]
    rows.append((form_id, label_id, text))
  return rows


def group_into_forms(rows):
  """Group rows into [(form_id, [(label_id, text), ...])].

  Forms are kept in first-appearance order and elements in within-form order.
  This is what lets the validation step in val/test (where forms are
  interleaved) reconstruct each whole form from the scattered rows.
  """
  forms = {}
  order = []
  for form_id, label_id, text in rows:
    if form_id not in forms:
      forms[form_id] = []
      order.append(form_id)
    forms[form_id].append((label_id, text))
  return [(fid, forms[fid]) for fid in order]


def validate_forms(name, forms, rows):
  """Sanity-check the grouping/ordering and return a small stats dict."""
  sizes = [len(elems) for _, elems in forms]
  total_elems = sum(sizes)
  stats = {
    "dataset": name,
    "forms": len(forms),
    "elements": total_elems,
    "min_form_size": min(sizes) if sizes else 0,
    "max_form_size": max(sizes) if sizes else 0,
    "mean_form_size": (total_elems / len(forms)) if forms else 0.0,
  }
  # Invariants: grouping loses no rows, every form is non-empty, no duplicate
  # form id in the grouped output.
  assert total_elems == len(rows), f"{name}: grouped {total_elems} != {len(rows)} rows"
  assert all(s > 0 for s in sizes), f"{name}: empty form found"
  ids = [fid for fid, _ in forms]
  assert len(ids) == len(set(ids)), f"{name}: duplicate form id after grouping"
  print(f"[validate] {name}: {stats['forms']} forms, {stats['elements']} elements, "
        f"size min/mean/max = {stats['min_form_size']}/"
        f"{stats['mean_form_size']:.1f}/{stats['max_form_size']}")
  return stats


def first_stage_probs(texts, cfg, batch_size=64):
  """Run the base model over texts -> ([len(texts), NUM_CLASSES] probs, id2label).

  We call the model directly rather than via the text-classification pipeline:
  the model was trained with 1-indexed labels so its config.id2label has keys
  1..66 while the 66 output neurons are 0..65, and the pipeline's all-scores
  path raises KeyError mapping index 0. The raw softmax over output neurons is
  exactly what we want for features; baseline names come from id2label.
  """
  import torch
  from transformers import AutoModelForSequenceClassification, AutoTokenizer
  tokenizer = AutoTokenizer.from_pretrained(cfg.saveModelDir)
  model = AutoModelForSequenceClassification.from_pretrained(cfg.saveModelDir)
  model.eval()
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model.to(device)

  probs = np.zeros((len(texts), model.config.num_labels), dtype=np.float32)
  with torch.no_grad():
    for start in range(0, len(texts), batch_size):
      batch = texts[start:start + batch_size]
      enc = tokenizer(batch, truncation=True, max_length=512,
                      padding=True, return_tensors="pt").to(device)
      logits = model(**enc).logits
      probs[start:start + len(batch)] = torch.softmax(logits, dim=-1).cpu().numpy()
  return probs, model.config.id2label


def build_form_stack(form_probs, window=DEFAULT_WINDOW):
  """Pad a form's [n, NUM_CLASSES] probs into a [n + 2*window, ROW_DIM] stack.

  `window` marker rows are added on each side so every real element has a full
  window of neighbours even at the form boundaries: the leading rows carry the
  'first item' flag, the trailing rows the 'last item' flag.
  """
  n = form_probs.shape[0]
  stack = np.zeros((n + 2 * window, ROW_DIM), dtype=np.float32)
  stack[window:window + n, :NUM_CLASSES] = form_probs   # real elements
  stack[:window, FIRST_ITEM] = 1.0                      # leading 'first item' markers
  stack[window + n:, LAST_ITEM] = 1.0                   # trailing 'last item' markers
  return stack


def windowed_features(stack, window=DEFAULT_WINDOW):
  """Concatenate each real element with its +/-window neighbours, left to right.

  The result for element k is [row(k-window), ..., row(k), ..., row(k+window)]
  flattened, so each relative position lives in a fixed slot of the feature
  vector -- the classifier always knows which block is to the left vs right.
  """
  n = stack.shape[0] - 2 * window
  feats = np.zeros((n, (2 * window + 1) * ROW_DIM), dtype=np.float32)
  for k in range(n):
    i = k + window                             # padded index of the element
    feats[k] = stack[i - window:i + window + 1].reshape(-1)
  return feats


def build_dataset(rows, cfg, window=DEFAULT_WINDOW):
  """From raw rows build (X, y, baseline_pred) for the context models.

  X: [n_elements, (2*window+1)*ROW_DIM] context features.
  y: field-type names (str) of each element.
  baseline_pred: field-type names from the base model alone (no context),
                 for comparison.
  """
  forms = group_into_forms(rows)
  texts = [text for _, elems in forms for (_, text) in elems]
  probs, id2label = first_stage_probs(texts, cfg)
  assert probs.shape[1] == NUM_CLASSES, \
      f"model emits {probs.shape[1]} classes, expected {NUM_CLASSES}"

  X_parts, y = [], []
  offset = 0
  for _fid, elems in forms:
    n = len(elems)
    stack = build_form_stack(probs[offset:offset + n], window)
    offset += n
    X_parts.append(windowed_features(stack, window))
    y.extend(fieldTypesReversedDict[label_id] for (label_id, _t) in elems)

  X = np.vstack(X_parts).astype(np.float32)
  # No-context baseline: the base model's own top-1 prediction per element.
  baseline_pred = [id2label.get(int(j), "other") for j in np.argmax(probs, axis=1)]
  return X, np.array(y), baseline_pred


def make_model(kind):
  """Construct one of the second-stage classifiers."""
  if kind == "mlp":
    return MLPClassifier(hidden_layer_sizes=(256,), max_iter=300, random_state=189)
  if kind == "logreg":
    return LogisticRegression(max_iter=1000, random_state=189)
  if kind == "rf":
    return RandomForestClassifier(n_estimators=200, random_state=189)
  raise ValueError(f"unknown model kind: {kind}")


def train_and_eval(kind, X_train, y_train, X_test, y_test):
  """Fit one method on the training features and score it on the test set."""
  model = make_model(kind)
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  print(f"=== {kind} ===")
  return compute_standard_metrics(y_test, y_pred, print_report=True)
