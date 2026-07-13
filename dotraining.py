from dataclasses import dataclass, field

from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    PreTrainedModel,
    TrainingArguments,
    Trainer,
    pipeline,
    set_seed,
    AutoConfig
)
from transformers.modeling_outputs import SequenceClassifierOutput

import torch.nn as nn

from datasets import Dataset

import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
)

# Keys used for the standard classification metrics returned by evaluate_model.
KAPPA = "kappa"
PRECISION = "precision"
RECALL = "recall"
F1 = "f1"
ACCURACY = "accuracy"
CLASSIFICATION_REPORT = "classification_report"
# Per-class and locale breakdowns added to the eval metrics dict.
FIELD_ACCURACY = "fieldAccuracy"
LOCALE_ACCURACY = "localeAccuracy"
LOCALE_SUPPORT = "localeSupport"
ENGLISH_ACCURACY = "englishAccuracy"
NON_ENGLISH_ACCURACY = "nonEnglishAccuracy"
ENGLISH_COUNT = "englishCount"
NON_ENGLISH_COUNT = "nonEnglishCount"

# Locale tokens that we treat as English. QA data is keyed by country code and
# the GEN/EN data by language, so this covers English-speaking countries plus
# the bare "EN" language tag.
ENGLISH_LOCALES = {"US", "GB", "UK", "AU", "NZ", "IE", "CA", "EN"}


def classify_locale(filename):
  """Best-effort (locale, language, is_english) parsed from a dataset filename.

  Dataset rows are named by source, which encodes the locale a few ways:
    GEN_<bcp47>_...  generated data, e.g. GEN_en-US_... / GEN_de-DE_...
    QA_<CC>_...      QA data keyed by country code, e.g. QA_ES_... / QA_NZ_...
    EN_... FR_... DE_...   language/locale prefix
    <cc>.html        bare country file, e.g. us.html / nl.html
  Anything unrecognized is bucketed as 'other' and treated as non-English.
  """
  name = (filename or "").strip()
  parts = name.split("_")
  head = parts[0]
  if head == "GEN" and len(parts) >= 2:
    tag = parts[1]
    lang = tag.split("-")[0].lower()
    return tag, lang, lang == "en"
  if head == "QA" and len(parts) >= 2:
    cc = parts[1].upper()
    return "QA-" + cc, cc.lower(), cc in ENGLISH_LOCALES
  if len(parts) >= 2 and head.isalpha() and head.isupper() and 2 <= len(head) <= 3:
    # e.g. EN_..., FR_..., DE_...
    return head, head.lower(), head in ENGLISH_LOCALES
  # bare filename like us.html / nl.html / creditcards.html
  stem = name.split(".")[0]
  if 2 <= len(stem) <= 3 and stem.isalpha():
    return stem.lower(), stem.lower(), stem.upper() in ENGLISH_LOCALES
  return "other", "other", False


# Supported context encodings for reformat_context (the --context_format flag).
# "bb"/"sep"/"symbol" are single-string re-encodings fed to a standard
# AutoModelForSequenceClassification. "triple" is a different model architecture
# entirely: the current/previous/next tokens are split into three separate
# strings (aa/bb prefixes removed), each run through a shared base transformer,
# and the three pooled outputs fused by an MLP head (see
# TripleEncoderForSequenceClassification). "triple" is only supported by the
# train / evaluate_model / infer paths in this file -- not the pipeline-based
# ONNX export, quantization, or two-stage form_context flow.
CONTEXT_FORMATS = ("bb", "sep", "symbol", "triple")

# Pooling modes for the single-sequence path (context_format bb/sep/symbol);
# see Config.pooling. "cls" uses the base model's standard classification head
# (the [CLS]/pooler token); "mean" uses a custom head over attention-masked
# mean-pooled token embeddings. Ignored for "triple" (always mean-pools).
POOLING_MODES = ("cls", "mean")

# Single-token markers for the "symbol" format. Both are already single tokens
# in the bert-base-uncased vocab (so no tokenizer/embedding changes are needed),
# and using two distinct symbols gives the previous vs next sections their own
# learned embeddings -- a directional signal a shared [SEP] can't provide.
PREV_SYMBOL = "•"   # bullet: marks the start of the previous-field section
NEXT_SYMBOL = "§"   # section sign: marks the start of the next-field section


def _split_context_words(text):
  """Bucket a raw context string into (current, previous, next) word lists.

  The datasets encode the field's own tokens with no prefix, the *previous*
  field's tokens with a 'bb' prefix, and the *next* field's tokens with an 'aa'
  prefix, all whitespace-joined, e.g.:

      street house number bblast bbname aapostcode aapostcode

  The 'bb'/'aa' prefixes are stripped so the words keep their normal (plain)
  form. Shared by reformat_context (single-string encodings) and split_context
  (the three-string 'triple' encoding).
  """
  current, previous, nxt = [], [], []
  for word in text.split():
    if len(word) > 2 and word.startswith("bb"):
      previous.append(word[2:])
    elif len(word) > 2 and word.startswith("aa"):
      nxt.append(word[2:])
    else:
      current.append(word)
  return current, previous, nxt


def split_context(text):
  """Split a raw context string into three plain strings for the 'triple' format.

  Returns (current, previous, next) as whitespace-joined strings with the
  'bb'/'aa' per-word prefixes removed, e.g.:

      "street house number bblast bbname aapostcode aapostcode"
        -> ("street house number", "last name", "postcode postcode")

  Each string is encoded independently by the shared base transformer in
  TripleEncoderForSequenceClassification. Empty sections yield "".
  """
  current, previous, nxt = _split_context_words(text)
  return " ".join(current), " ".join(previous), " ".join(nxt)


def reformat_context(text, fmt="bb"):
  """Re-encode the bb/aa per-word context prefixes at load time.

  The datasets encode the field's own tokens with no prefix, the *previous*
  field's tokens with a 'bb' prefix, and the *next* field's tokens with an 'aa'
  prefix, all whitespace-joined, e.g.:

      street house number bblast bbname aapostcode aapostcode

  WordPiece shreds 'bblast' into 'bb ##la ##st' -- which both inflates the
  sequence (~3x) and hides the real word from the model. This regroups the
  tokens by section and marks each section once, keeping the words plain so they
  share their normal embeddings. Two encodings are offered:

      sep:    <field tokens> [SEP] <previous tokens> [SEP] <next tokens>
      symbol: <field tokens>  •  <previous tokens>  §  <next tokens>

  'sep' uses one shared [SEP] for both boundaries (direction is positional);
  'symbol' uses two distinct single-token markers so previous vs next each get
  their own embedding. Both markers are always emitted so the sections stay
  positionally distinguishable even when one is empty. fmt='bb' returns the text
  unchanged (the original behavior). fmt='triple' also returns the text
  unchanged here -- that format splits into three strings via split_context in
  readFile / eval rather than producing a single sequence. Applied on the fly;
  data files are not modified.
  """
  if fmt in ("bb", "triple"):
    return text
  if fmt not in ("sep", "symbol"):
    raise ValueError(f"Unknown context_format {fmt!r}; expected one of {CONTEXT_FORMATS}")
  current, previous, nxt = _split_context_words(text)
  prev_marker, next_marker = ("[SEP]", "[SEP]") if fmt == "sep" else (PREV_SYMBOL, NEXT_SYMBOL)
  return (f"{' '.join(current)} {prev_marker} {' '.join(previous)} "
          f"{next_marker} {' '.join(nxt)}")


# Seed for reproducible synthetic-data subsampling (so the kept rows are the
# same across runs and comparable).
SUBSAMPLE_SEED = 189


def dataset_source(filename):
  """Classify a dataset row by its source from the filename prefix.

  'gen'  -> GEN_*  (generated/synthetic forms)
  'cc'   -> CC_*   (generated credit-card forms)
  'real' -> everything else (QA crawled sites, language/country, bare files)
  """
  head = (filename or "").split("_", 1)[0]
  if head == "GEN":
    return "gen"
  if head == "CC":
    return "cc"
  return "real"


def _subsample_sources(records, cfg):
  """Subsample synthetic sources (gen/cc) down to a target ratio of real rows.

  records is a list of (source, example) tuples. genToRealRatio / ccToRealRatio
  cap that source at ratio * (#real rows); a ratio <= 0 disables subsampling for
  that source (keep all). Only ever downsamples -- never upsamples. Real rows are
  always kept. Reproducible via SUBSAMPLE_SEED.
  """
  ratios = {"gen": cfg.genToRealRatio, "cc": cfg.ccToRealRatio}
  if all(r <= 0 for r in ratios.values()):
    return records
  by_src = {"gen": [], "cc": [], "real": []}
  for src, rec in records:
    by_src[src].append((src, rec))
  n_real = len(by_src["real"])
  rng = random.Random(SUBSAMPLE_SEED)
  kept = list(by_src["real"])
  for src in ("gen", "cc"):
    rows = by_src[src]
    ratio = ratios[src]
    if ratio > 0:
      target = int(ratio * n_real)
      if len(rows) > target:
        rows = rng.sample(rows, target)
      print(f"  subsample {src}: {len(by_src[src])} -> {len(rows)} "
            f"(ratio {ratio} x {n_real} real rows)")
    kept.extend(rows)
  return kept


def compute_standard_metrics(y_test, y_pred, print_report=False):
  """Standard weighted classification metrics + classification report dict.

  Shared by evaluate_model and the form-context models so every model is
  scored the same way.
  """
  if print_report:
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
  return {
    KAPPA: cohen_kappa_score(y_test, y_pred),
    PRECISION: precision_score(y_test, y_pred, average="weighted", zero_division=0),
    RECALL: recall_score(y_test, y_pred, average="weighted", zero_division=0),
    F1: f1_score(y_test, y_pred, average="weighted", zero_division=0),
    ACCURACY: accuracy_score(y_test, y_pred),
    CLASSIFICATION_REPORT: classification_report(y_test, y_pred, zero_division=0, output_dict=True),
  }

# The first two fields of the CSV datasets (filename and field name) are ignored
# and are used only for reference when debugging.
ignoreLineCount = 2

# This script is used to train the address autofill model for Firefox.
# The configuration below allows two modes, 'all' and 'supported'. All mode
# handles all of the field types in the fieldTypesDict list whereas supported
# mode only handles the field types that are supported by Firefox autofill.
# Generally, we have been using supported mode.
#
# It is expected that there is a file 'training-supported.txt',
# 'validation-supported.txt' and 'testing-supported.txt' in the same directory
# as this script which contains the training, validation and test data in CSV
# format. In 'all' mode, the '-supported' should be removed from the filenames,
# allowing both datasets to coexist.
#
# This training, validation and test data is generated separately from
# sample forms from various regions.
#
# To train:
#   python dotraining.py train
# To test:
#   python dotraining.py test
#
# There is also a random forest classifier that can be tried out with:
#   python dotraining.py forest
# It requires sklearn.ensemble.RandomForestClassifier.
#
# A special case 'together' is also supported to handle a file named
# 'together-supported.txt' intended to be a concatenation of all three of
# the input files.
#   python dotraining.py together
#
#
# If a different string is passed to this script as an argument, then it is
# treated as a single token list to test with.
#
# Trained models are saved in the output-models directory.
#
# The CSV data has four fields: source filename, expected fieldname,
# expected fieldname index (from fieldTypesDict), and the set of tokens.
#
# The training and evaluation logic is also exposed as importable functions
# (train / evaluate) so that it can be orchestrated from a Metaflow flow. See
# autofill_flow.py for the Metaflow wrapper that runs training and evaluation
# as separate steps.

# ---- Configuration Section ----

# This section allow configuration of the training and testing.

# Source model to use for training
DEFAULT_MODEL_NAME = "huawei-noah/TinyBERT_General_4L_312D"

# There are two modes: 'all' and 'supported'. All mode handles all of the
# field types in the fieldTypesDict list whereas supported mode only handles
# the field types that are supported by Firefox autofill. This python script
# will use different datasets in each mode. Select the desired one by setting
# the value of dataVariant to "" for all mode and "-supported" for supported mode.
#DEFAULT_DATA_VARIANT = ""
DEFAULT_DATA_VARIANT = "-supported"

# Number of epochs to train. This is the number of passes through the training
# data that are performed.
DEFAULT_NUM_EPOCHS = 15

# ---- End Configuration Section ----


@dataclass
class Config:
    """Holds the training/evaluation configuration.

    All of the values that used to be module-level globals now live here so
    that the same logic can be driven either from the command line or from a
    Metaflow flow with different parameter values.
    """

    modelName: str = DEFAULT_MODEL_NAME
    dataVariant: str = DEFAULT_DATA_VARIANT
    numEpochs: int = DEFAULT_NUM_EPOCHS
    # Append an extra string to the filename to test variations, e.g.
    # "-updated". Applied to both the saved model name and the dataVariant.
    modelSuffix: str = ""

    # Override for the training dataset filename. When empty, the default
    # "training<dataVariant>.txt" is used; set it to train on a different file
    # such as "training-supported-expanded.txt".
    trainFile: str = ""

    # When True, training/validation rows are filtered to English-only sources
    # (see classify_locale). Evaluation still runs on the full test set so the
    # English vs non-English breakdown remains visible.
    englishOnly: bool = False

    # How the bb/aa previous/next-field context tokens are encoded at load time
    # (see reformat_context). "bb" keeps the raw per-word prefixes; "sep"
    # regroups them into [SEP]-delimited sections with plain words. Applied to
    # training, validation and evaluation so train/eval stay consistent.
    contextFormat: str = "bb"

    # Pooling for the single-sequence path (context_format bb/sep/symbol):
    # "cls" (default) uses the base model's standard classification head (the
    # [CLS]/pooler token); "mean" uses a custom head over attention-masked
    # mean-pooled token embeddings -- a better match for encoders pretrained
    # with mean pooling (e.g. sentence-transformers models). Ignored when
    # contextFormat="triple" (that architecture always mean-pools).
    pooling: str = "cls"

    # Shrink the base encoder to this many evenly-spaced transformer layers
    # before fine-tuning (e.g. 4 keeps ~[0,4,7,11] of a 12-layer model). 0
    # (default) keeps all layers. Applies to the standard, mean, and triple
    # paths -- the smaller depth is saved so eval/reload rebuilds it. Combine
    # with vocab pruning (prune_vocab.py) to approach TinyBERT size.
    encoderLayers: int = 0

    # Separate learning rate for the classification head (the triple fusion
    # head / mean pooler+classifier / standard classifier), leaving the
    # pretrained encoder on the main learningRate. 0.0 (default) uses a single
    # LR for everything. A higher head LR (e.g. 1e-3) lets the randomly
    # initialized head converge without pushing the encoder off its pretrained
    # weights -- most useful for the triple head.
    headLearningRate: float = 0.0

    # Triple-head variants (context_format='triple' only; both default off = the
    # original 3H->H->num_labels head). headInteractions also feeds neighbor
    # difference features (cur-prev, cur-next) to the fusion; headProjDim > 0
    # inserts a shared Linear(H->d) that projects each pooled section to d dims
    # before fusion (a regularizing bottleneck; smaller cached per-field vectors).
    headInteractions: bool = False
    headProjDim: int = 0

    # Cap synthetic training data relative to real (crawled) data to curb
    # overfitting to templated synthetic forms. Each ratio bounds that source at
    # ratio * (#real rows); <= 0 disables subsampling for that source (keep all).
    # Applied to the training set only (see readFile / _subsample_sources).
    genToRealRatio: float = 0.0   # GEN_* generated forms
    ccToRealRatio: float = 0.0    # CC_* credit-card forms

    # Training hyperparameters. Defaults match the HF Trainer defaults so that
    # leaving them unset reproduces the previous behavior. learningRate of 0.0
    # means "auto": the HF default (5e-5), or 2e-4 when LoRA is enabled.
    learningRate: float = 0.0
    trainBatchSize: int = 8
    evalBatchSize: int = 8
    weightDecay: float = 0.0
    # Fraction of total training steps spent linearly warming the learning rate
    # up from 0 before it decays. 0.0 (default) reproduces the previous
    # no-warmup behavior. A small value (~0.06-0.1) eases in a randomly
    # initialized head (e.g. the context_format='triple' fusion head) so its
    # early gradients don't disturb the pretrained encoder.
    warmupRatio: float = 0.0

    # Close-label smoothing: leak this fraction of the target mass onto a class's
    # fieldNamesCloseDict neighbors, so predicting a "close" type is penalized less
    # than a genuinely wrong one. 0.0 (default) reproduces plain cross-entropy;
    # 0.1-0.2 is a good range. Aims to lift both Total and Close accuracy.
    closeLabelEps: float = 0.0

    # LoRA / parameter-efficient fine-tuning. When useLora is set, the model is
    # wrapped with a PEFT LoRA adapter for training and the adapter is merged
    # back into the base weights before saving (so eval/inference is unchanged).
    useLora: bool = False
    loraR: int = 8
    loraAlpha: int = 16
    loraDropout: float = 0.1

    # Weights & Biases logging. If wandbProject is set, training reports to W&B
    # and the eval metrics are logged to the same run. Left empty (the default)
    # W&B is disabled entirely, which keeps the plain CLI path offline.
    wandbProject: str = ""
    wandbRunId: str = ""
    wandbRunName: str = ""

    @property
    def modelExtra(self) -> str:
        base = "supported" if self.dataVariant.startswith("-supported") else "all"
        return base + self.modelSuffix

    @property
    def saveModelName(self) -> str:
        return "autofill-tiny-" + self.modelExtra

    @property
    def saveModelDir(self) -> str:
        return "output-models/" + self.saveModelName


# Other models that were experimented with.
#modelName = "nhull/random-forest-model"
#modelName = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
#modelName = "distilbert/distilbert-base-uncased"
#modelName = "Mozilla/tinybert-uncased-autofill"
#modelName = "microsoft/MiniLM-L12-H384-uncased"
#modelName = "Xenova/distilbert-base-uncased-finetuned-sst-2-english"
#modelName = "Xenova/bert-base-uncased"

#transformers.set_seed(189)

# The weights could be used, but are not for now.
weights = [
 1,1,1,1,3,
 2,2,3,2,2,
 2,4,3,4,3,
 3,1,1,1,1,
 1,1,1,2,1,
 2,2,2,1,1,
 1,1,1,2,2,
 2,2,2,1,1,
 1,1,1,1,1,
 1,2,2,2,2,
 1,1,1,1,1,
 1,1,1,1,1,
 1,1,1,1,1,
 1
]

# List of fields with their ids
fieldTypesDict = {
  'other': 1,
  'given-name': 2,
  'family-name': 3,
  'name': 4,
  'additional-name': 5,
  'phonetic-given-name': 6,
  'phonetic-family-name': 7,
  'phonetic-name': 8,
  'honorific-prefix': 9,
  'honorific-suffix': 10,
  'nickname': 11,
  'street-address': 12,
  'address-lookup': 13,
  'address-line1': 14,
  'address-line2': 15,
  'address-line3': 16,
  'address-level1': 17,
  'address-level2': 18,
  'address-level3': 19,
  'address-level4': 20,
  'street': 21,
  'address-streetname': 22,
  'address-housenumber': 23,
  'address-extra-housesuffix': 24,
  'postal-code': 25,
  'postal-code-lookup': 26,
  'postal-code-and-city': 27,
  'postal-code-or-suburb': 28,
  'country': 29,
  'country-name': 30,
  'tel': 31,
  'tel-country-code': 32,
  'tel-national': 33,
  'tel-area-code': 34,
  'tel-local': 35,
  'tel-local-prefix': 36,
  'tel-local-suffix': 37,
  'tel-extension': 38,
  'organization': 39,
  'organization-title': 40,
  'bday': 41,
  'bday-day': 42,
  'bday-month': 43,
  'bday-year': 44,
  'email': 45,
  'apartment': 46,
  'floor': 47,
  'stair': 48,
  'building': 49,
  'block': 50,
  'address-extra': 51,
  'cc-name': 52,
  'cc-given-name': 53,
  'cc-additional-name': 54,
  'cc-family-name': 55,
  'cc-number': 56,
  'cc-exp': 57,
  'cc-exp-month': 58,
  'cc-exp-year': 59,
  'cc-csc': 60,
  'cc-type': 61,
  'sex': 62,
  'id-number': 63,
  'vat-number': 64,
  'reference-point': 65,
  'loginname': 66,
}
fieldTypesReversedDict = {v: k for k,v in fieldTypesDict.items()}

fieldNamesCloseDict = {
  "street-address": ["address-line1", "street"],
  "address-line1": ["street-address", "street"],
  "address-line2": ["apartment"],
  "street": ["street-address", "address-line1"],
  "postal-code-and-city": ["postal-code"],
  "postal-code-and-suburb": ["postal-code"],
  "tel": ["tel-national"],
  "tel-national": ["tel"],
  "apartment": ["address-line2"],
  "given-name": ["cc-given-name"],
  "additional-name": ["cc-additonal-name"],
  "family-name": ["cc-family-name"],
  "name": ["cc-name"],
  "cc-given-name": ["given-name"],
  "cc-additional-name": ["additonal-name"],
  "cc-family-name": ["family-name"],
  "cc-name": ["name"],
  "loginname": ["email"],
  "email": ["loginname"],
  "country": ["country-name"],
  "country-name": ["country"],
}

def dataset_path(filename):
  """Return a dataset filename, requiring it to be present locally.

  The dataset .txt files travel with the run (Metaflow packages them when
  .txt is included in METAFLOW_DEFAULT_PACKAGE_SUFFIXES), so they are read from
  the working directory rather than downloaded at runtime.
  """
  if not os.path.exists(filename):
    raise FileNotFoundError(
      f"Dataset file '{filename}' not found in the working directory. "
      "Ensure it is present locally and that .txt is included in the Metaflow "
      "package suffixes so it ships with the run."
    )
  return filename

def build_close_targets(label2id, close_dict, num_labels, eps, symmetric=False):
  """Soft-target matrix (num_labels x num_labels): row i is the target
  distribution for true class i. The true class keeps (1-eps); the remaining eps
  is split over its fieldNamesCloseDict neighbors. Classes with no close entry
  stay one-hot. Sized to num_labels and indexed by the model's own label ids."""
  S = torch.zeros(num_labels, num_labels)
  close = {k: set(v) for k, v in close_dict.items()}
  if symmetric:                       # fieldNamesCloseDict isn't fully symmetric
    for a, vs in list(close.items()):
      for b in vs:
        close.setdefault(b, set()).add(a)
  for name, i in label2id.items():
    if not (0 <= i < num_labels):
      continue
    cs = [label2id[c] for c in close.get(name, ())
          if c in label2id and 0 <= label2id[c] < num_labels]
    if cs:
      S[i, i] = 1.0 - eps
      for c in cs:
        S[i, c] = eps / len(cs)
    else:
      S[i, i] = 1.0
  return S


class CloseAwareTrainer(Trainer):
  """Trainer with similarity-aware soft-label cross-entropy. Predicting a class
  that is 'close' (per fieldNamesCloseDict) to the true one incurs less loss than
  predicting a far, wrong class."""

  def __init__(self, *args, soft_targets=None, **kwargs):
    super().__init__(*args, **kwargs)
    self.soft_targets = soft_targets      # (num_labels, num_labels) CPU tensor

  def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
    labels = inputs.pop("labels")
    outputs = model(**inputs)
    logp = F.log_softmax(outputs.logits, dim=-1)
    target = self.soft_targets.to(logp.device)[labels]        # (batch, num_labels)
    loss = -(target * logp).sum(dim=-1).mean()
    return (loss, outputs) if return_outputs else loss


# ---- Triple-encoder architecture (context_format="triple") ----------------
#
# A different model from the single-sequence formats: the current / previous /
# next tokens are split into three separate strings (see split_context), each
# encoded independently by ONE shared base transformer, and the three pooled
# outputs are fused by an MLP head into the classification. This is not a
# standard HF sequence-classification model, so it is trained/evaluated through
# the custom paths in this file rather than pipeline() / optimum / form_context.

def masked_mean_pool(last_hidden_state, attention_mask):
  """Attention-mask-weighted mean over tokens (excludes padding).

  last_hidden_state: (B, T, H); attention_mask: (B, T) -> pooled (B, H). Shared
  by the triple-encoder and the mean-pooling single-sequence model.
  """
  mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # (B, T, 1)
  summed = (last_hidden_state * mask).sum(dim=1)                   # (B, H)
  counts = mask.sum(dim=1).clamp(min=1e-9)                         # (B, 1)
  return summed / counts


def select_encoder_layers(encoder, num_keep):
  """Keep `num_keep` evenly-spaced transformer layers (incl. first & last), in place.

  Shrinks a pretrained BERT/XLM-R-style encoder (a `.encoder.layer` ModuleList)
  before fine-tuning, to cut size and latency. Evenly-spaced selection keeps the
  first (lexical) and last (task-level; the mean-pool reads it) layers. Updates
  config.num_hidden_layers so the smaller architecture round-trips through
  save/load. Returns the kept indices, or None if num_keep is falsy or >= depth
  (left unchanged).
  """
  base = getattr(encoder, "base_model", encoder)
  enc = getattr(base, "encoder", None)
  layers = getattr(enc, "layer", None)
  if layers is None:
    raise ValueError(
        f"Could not locate transformer layers (.encoder.layer) on "
        f"{type(base).__name__}; layer-drop supports BERT/XLM-R-style encoders.")
  total = len(layers)
  if not num_keep or num_keep <= 0 or num_keep >= total:
    return None
  if num_keep == 1:
    idx = [total - 1]
  else:
    idx = sorted({round(i * (total - 1) / (num_keep - 1)) for i in range(num_keep)})
  enc.layer = nn.ModuleList([layers[i] for i in idx])
  base.config.num_hidden_layers = len(idx)
  return idx


def _resolve_encoder_config(config):
  """AutoConfig for the shared encoder of the custom triple/mean models.

  Prefers the encoder config stored on our config (so a pruned vocab or dropped
  layers round-trip through save/load without re-fetching the base model);
  falls back to fetching base_model_name for older saved configs.
  """
  enc = getattr(config, "encoder_config", None)
  if enc:
    enc = dict(enc)
    model_type = enc.pop("model_type", None)
    return AutoConfig.for_model(model_type, **enc)
  return AutoConfig.from_pretrained(config.base_model_name)


class TripleEncoderConfig(PretrainedConfig):
  """Config for TripleEncoderForSequenceClassification.

  `base_model_name` names the shared encoder (e.g. the TinyBERT checkpoint);
  `encoder_config` is that encoder's resolved config dict (captured at build
  time so a pruned vocab / dropped layers round-trip without re-fetching the
  base); `hidden_dropout` is the dropout in the fusion head. num_labels /
  id2label / label2id are handled by PretrainedConfig so the saved model carries
  the field names, exactly like the standard classifier.
  """
  model_type = "triple_encoder"

  def __init__(self, base_model_name=DEFAULT_MODEL_NAME, encoder_config=None,
               hidden_dropout=0.1, head_interactions=False, head_proj_dim=0, **kwargs):
    super().__init__(**kwargs)
    self.base_model_name = base_model_name
    self.encoder_config = encoder_config
    self.hidden_dropout = hidden_dropout
    # Head variants (both default off -> the original 3H->H->num_labels head):
    #   head_interactions: also feed neighbor differences (cur-prev, cur-next),
    #     so the fusion sees 5 sections instead of 3.
    #   head_proj_dim > 0: a SHARED Linear(H -> d) projects each pooled section
    #     before fusion, so the head (and cached per-field embeddings) work in d
    #     dims instead of H.
    self.head_interactions = head_interactions
    self.head_proj_dim = head_proj_dim


class TripleEncoderForSequenceClassification(PreTrainedModel):
  """Shared-encoder triple model with an MLP fusion head.

  forward() takes the three tokenized sections (current/previous/next), runs the
  shared encoder over each, attention-mask-mean-pools each to a fixed vector,
  concatenates the three (3H), then:

      3H -> Linear(3H, H) -> GELU -> dropout -> Linear(H, num_labels)

  Returns a SequenceClassifierOutput so it plugs into the HF Trainer (and
  CloseAwareTrainer, which reads outputs.logits) unchanged. labels is optional
  so the loss-popping path in CloseAwareTrainer works too.
  """
  config_class = TripleEncoderConfig

  def __init__(self, config):
    super().__init__(config)
    # Build the encoder architecture (no weights) from the stored/base encoder
    # config; from_base_pretrained() swaps in pretrained weights for training,
    # and from_pretrained() fills them from the saved state_dict on reload.
    enc_cfg = _resolve_encoder_config(config)
    self.encoder = AutoModel.from_config(enc_cfg)
    hidden = enc_cfg.hidden_size
    # Optional shared per-section projection H -> d; d is the head's working dim.
    self.head_proj_dim = int(getattr(config, "head_proj_dim", 0) or 0)
    self.proj = nn.Linear(hidden, self.head_proj_dim) if self.head_proj_dim > 0 else None
    d = self.head_proj_dim if self.head_proj_dim > 0 else hidden
    # Fusion input sections: [cur, prev, next] (+ [cur-prev, cur-next] if enabled).
    self.head_interactions = bool(getattr(config, "head_interactions", False))
    n_sections = 5 if self.head_interactions else 3
    self.dropout = nn.Dropout(config.hidden_dropout)
    self.fusion = nn.Linear(n_sections * d, d)
    self.act = nn.GELU()
    self.classifier = nn.Linear(d, config.num_labels)
    self.post_init()

  @classmethod
  def from_base_pretrained(cls, base_model_name, num_labels, id2label=None,
                           label2id=None, hidden_dropout=0.1, encoder_layers=0,
                           head_interactions=False, head_proj_dim=0):
    """Construct for training: random fusion head + pretrained shared encoder.

    encoder_layers > 0 drops the encoder down to that many evenly-spaced layers
    before training (see select_encoder_layers). head_interactions / head_proj_dim
    select the head variant (see TripleEncoderConfig).
    """
    encoder = AutoModel.from_pretrained(base_model_name)
    total = encoder.config.num_hidden_layers
    kept = select_encoder_layers(encoder, encoder_layers)
    if kept is not None:
      print(f"  layer-drop: {total} -> {len(kept)} encoder layers {kept}")
    config = cls.config_class(
        base_model_name=base_model_name,
        encoder_config=encoder.config.to_dict(),
        hidden_dropout=hidden_dropout,
        head_interactions=head_interactions,
        head_proj_dim=head_proj_dim,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    model = cls(config)
    # Replace the freshly-initialized encoder with the pretrained weights.
    model.encoder = encoder
    return model

  def _encode(self, input_ids, attention_mask):
    out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
    return masked_mean_pool(out.last_hidden_state, attention_mask)

  def forward(self, input_ids_current=None, attention_mask_current=None,
              input_ids_previous=None, attention_mask_previous=None,
              input_ids_next=None, attention_mask_next=None, labels=None,
              **kwargs):
    cur = self._encode(input_ids_current, attention_mask_current)
    prev = self._encode(input_ids_previous, attention_mask_previous)
    nxt = self._encode(input_ids_next, attention_mask_next)
    if self.proj is not None:                               # shared H -> d projection
      cur, prev, nxt = self.proj(cur), self.proj(prev), self.proj(nxt)
    sections = [cur, prev, nxt]
    if self.head_interactions:                              # neighbor difference features
      sections += [cur - prev, cur - nxt]
    fused = torch.cat(sections, dim=-1)                     # (B, n_sections * d)
    hidden = self.dropout(self.act(self.fusion(fused)))
    logits = self.classifier(hidden)
    loss = None
    if labels is not None:
      loss = F.cross_entropy(logits, labels)
    return SequenceClassifierOutput(loss=loss, logits=logits)


# Register so AutoConfig / AutoModelForSequenceClassification can round-trip the
# saved model. Guarded because re-import (e.g. in the same process) would raise.
try:
  AutoConfig.register("triple_encoder", TripleEncoderConfig)
  AutoModelForSequenceClassification.register(
      TripleEncoderConfig, TripleEncoderForSequenceClassification)
except (ValueError, KeyError):
  pass


class TripleDataCollator:
  """Dynamic-pads the three tokenized sections independently.

  Mirrors DataCollatorWithPadding but for the triple format: each feature
  carries input_ids_/attention_mask_ for current/previous/next; each section is
  padded to its own per-batch max via tokenizer.pad. Non-tensor columns (the raw
  text_* fields) are ignored, so remove_unused_columns can stay off.
  """

  def __init__(self, tokenizer):
    self.tokenizer = tokenizer

  def __call__(self, features):
    batch = {}
    for section in ("current", "previous", "next"):
      group = [{"input_ids": f[f"input_ids_{section}"],
                "attention_mask": f[f"attention_mask_{section}"]} for f in features]
      padded = self.tokenizer.pad(group, return_tensors="pt")
      batch[f"input_ids_{section}"] = padded["input_ids"]
      batch[f"attention_mask_{section}"] = padded["attention_mask"]
    if "label" in features[0]:
      batch["labels"] = torch.tensor([f["label"] for f in features], dtype=torch.long)
    return batch


# ---- Mean-pooling single-sequence model (pooling="mean") ------------------
#
# Same single input as the standard AutoModelForSequenceClassification path,
# but the classification head reads an attention-masked mean of the token
# embeddings instead of the base model's [CLS]/pooler token. This matches how
# sentence-transformers encoders were pretrained. The forward uses the standard
# input_ids/attention_mask/labels names, so it reuses the ordinary
# tokenization, DataCollatorWithPadding, pipeline() eval, and infer() paths.
# The head mirrors the BERT pooler+classifier (dense+tanh -> dropout ->
# classifier) so the only difference from the CLS baseline is the pooling.

class MeanPoolConfig(PretrainedConfig):
  """Config for MeanPoolForSequenceClassification (see POOLING_MODES).

  Stores the resolved `encoder_config` for the same round-trip reasons as
  TripleEncoderConfig (pruned vocab / dropped layers).
  """
  model_type = "mean_pool_encoder"

  def __init__(self, base_model_name=DEFAULT_MODEL_NAME, encoder_config=None,
               hidden_dropout=0.1, **kwargs):
    super().__init__(**kwargs)
    self.base_model_name = base_model_name
    self.encoder_config = encoder_config
    self.hidden_dropout = hidden_dropout


class MeanPoolForSequenceClassification(PreTrainedModel):
  config_class = MeanPoolConfig

  def __init__(self, config):
    super().__init__(config)
    enc_cfg = _resolve_encoder_config(config)
    self.encoder = AutoModel.from_config(enc_cfg)
    hidden = enc_cfg.hidden_size
    self.pooler = nn.Linear(hidden, hidden)      # mirrors the BERT pooler dense
    self.act = nn.Tanh()
    self.dropout = nn.Dropout(config.hidden_dropout)
    self.classifier = nn.Linear(hidden, config.num_labels)
    self.post_init()

  @classmethod
  def from_base_pretrained(cls, base_model_name, num_labels, id2label=None,
                           label2id=None, hidden_dropout=0.1, encoder_layers=0):
    """Construct for training: random head + pretrained shared encoder.

    encoder_layers > 0 drops the encoder to that many evenly-spaced layers first.
    """
    encoder = AutoModel.from_pretrained(base_model_name)
    total = encoder.config.num_hidden_layers
    kept = select_encoder_layers(encoder, encoder_layers)
    if kept is not None:
      print(f"  layer-drop: {total} -> {len(kept)} encoder layers {kept}")
    config = cls.config_class(
        base_model_name=base_model_name,
        encoder_config=encoder.config.to_dict(),
        hidden_dropout=hidden_dropout,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    model = cls(config)
    model.encoder = encoder
    return model

  def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
    out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
    pooled = masked_mean_pool(out.last_hidden_state, attention_mask)
    hidden = self.dropout(self.act(self.pooler(pooled)))
    logits = self.classifier(hidden)
    loss = None
    if labels is not None:
      loss = F.cross_entropy(logits, labels)
    return SequenceClassifierOutput(loss=loss, logits=logits)


try:
  AutoConfig.register("mean_pool_encoder", MeanPoolConfig)
  AutoModelForSequenceClassification.register(
      MeanPoolConfig, MeanPoolForSequenceClassification)
except (ValueError, KeyError):
  pass


def wandb_config(cfg):
  """The hyperparameters logged to W&B run config (what sweeps compare on)."""
  return {
    "model_name": cfg.modelName,
    "data_variant": cfg.dataVariant,
    "num_epochs": cfg.numEpochs,
    "model_suffix": cfg.modelSuffix,
    "train_file": cfg.trainFile,
    "english_only": cfg.englishOnly,
    "context_format": cfg.contextFormat,
    "pooling": cfg.pooling,
    "encoder_layers": cfg.encoderLayers,
    "head_learning_rate": cfg.headLearningRate,
    "head_interactions": cfg.headInteractions,
    "head_proj_dim": cfg.headProjDim,
    "gen_to_real_ratio": cfg.genToRealRatio,
    "cc_to_real_ratio": cfg.ccToRealRatio,
    "learning_rate": cfg.learningRate,
    "train_batch_size": cfg.trainBatchSize,
    "eval_batch_size": cfg.evalBatchSize,
    "weight_decay": cfg.weightDecay,
    "warmup_ratio": cfg.warmupRatio,
    "close_label_eps": cfg.closeLabelEps,
    "use_lora": cfg.useLora,
    "lora_r": cfg.loraR,
    "lora_alpha": cfg.loraAlpha,
    "lora_dropout": cfg.loraDropout,
  }

def readFile(filetype, cfg):
  if cfg.contextFormat not in CONTEXT_FORMATS:
    raise ValueError(f"Unknown context_format {cfg.contextFormat!r}; expected one of {CONTEXT_FORMATS}")

  # Allow training to read from an explicit file (e.g. an expanded dataset);
  # otherwise derive the name from the filetype and data variant.
  if filetype == "training" and cfg.trainFile:
    filename = cfg.trainFile
  else:
    filename = filetype + cfg.dataVariant + ".txt"

  file = open(dataset_path(filename), encoding="utf-8")
  lines = file.readlines()

  records = []  # (source, {"label", "text"})
  for line in lines:
    line = line.strip()
    lineData = line.split(",", ignoreLineCount + 1)
    src = lineData[0] if len(lineData) > ignoreLineCount else ""
    if cfg.englishOnly and not classify_locale(src)[2]:
      continue
    try:
      raw = lineData[ignoreLineCount + 1]
      if cfg.contextFormat == "triple":
        # Three separate strings (aa/bb removed); tokenized per-section in train.
        cur, prev, nxt = split_context(raw)
        rec = {
          "label": int(lineData[ignoreLineCount]),
          "text_current": cur,
          "text_previous": prev,
          "text_next": nxt,
        }
      else:
        rec = {
          "label": int(lineData[ignoreLineCount]),
          "text": reformat_context(raw, cfg.contextFormat),
        }
    except Exception:
      print(filetype + ".txt : " + line)
      raise
    records.append((dataset_source(src), rec))

  # Only the training set is rebalanced; validation/testing keep their full
  # distribution so eval/model-selection stay comparable across runs.
  if filetype == "training":
    records = _subsample_sources(records, cfg)

  dataset = Dataset.from_list([rec for _src, rec in records])
  return dataset

def select_device():
  """Pick the compute device for training/inference.

  Honors an explicit AUTOFILL_DEVICE override (e.g. "cuda", "mps", "cpu");
  otherwise auto-detects CUDA, then Apple-silicon MPS, then CPU. When MPS is
  chosen we enable the CPU fallback so any op MPS hasn't implemented still runs
  rather than erroring out.
  """
  override = os.environ.get("AUTOFILL_DEVICE", "").strip().lower()
  if override:
    device = override
  elif torch.cuda.is_available():
    device = "cuda"
  elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
    device = "mps"
  else:
    device = "cpu"
  if device == "mps":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
  return device

def build_optimizer(model, encoder, base_lr, head_lr, weight_decay):
  """AdamW with the encoder on base_lr and the classification head on head_lr.

  `encoder` is the shared transformer module; every other trainable parameter is
  treated as head. Within each group, bias / LayerNorm parameters are excluded
  from weight decay, matching the HF Trainer's default grouping. Empty groups are
  dropped so AdamW doesn't choke.
  """
  no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight")
  enc_ids = {id(p) for p in encoder.parameters()}
  buckets = {"enc_decay": [], "enc_nodecay": [], "head_decay": [], "head_nodecay": []}
  for name, p in model.named_parameters():
    if not p.requires_grad:
      continue
    where = "enc" if id(p) in enc_ids else "head"
    decay = "nodecay" if any(nd in name for nd in no_decay) else "decay"
    buckets[f"{where}_{decay}"].append(p)
  specs = [
      ("enc_decay", base_lr, weight_decay),
      ("enc_nodecay", base_lr, 0.0),
      ("head_decay", head_lr, weight_decay),
      ("head_nodecay", head_lr, 0.0),
  ]
  groups = [{"params": buckets[k], "lr": lr, "weight_decay": wd}
            for k, lr, wd in specs if buckets[k]]
  from torch.optim import AdamW
  return AdamW(groups, lr=base_lr)


def train(cfg):
  device = select_device()
  print(f"Training on device: {device}")
  is_triple = cfg.contextFormat == "triple"
  if cfg.pooling not in POOLING_MODES:
      raise ValueError(f"Unknown pooling {cfg.pooling!r}; expected one of {POOLING_MODES}")
  # 'pooling' only applies to the single-sequence path; triple always mean-pools.
  use_mean = (not is_triple) and cfg.pooling == "mean"
  if (is_triple or use_mean) and cfg.useLora:
      raise ValueError(
          "LoRA is only supported with the standard CLS classification head "
          "(context_format in bb/sep/symbol and pooling='cls'). The "
          f"{'triple-encoder' if is_triple else 'mean-pooling'} model is a "
          "custom architecture without the SEQ_CLS head PEFT expects.")
  tokenizer = AutoTokenizer.from_pretrained(cfg.modelName)

  def preprocess_function(examples):
      return tokenizer(examples["text"], truncation=True, max_length=512)

  def preprocess_triple(examples):
      out = {}
      for section, col in (("current", "text_current"),
                           ("previous", "text_previous"),
                           ("next", "text_next")):
          enc = tokenizer(examples[col], truncation=True, max_length=512)
          out[f"input_ids_{section}"] = enc["input_ids"]
          out[f"attention_mask_{section}"] = enc["attention_mask"]
      return out

  preprocess = preprocess_triple if is_triple else preprocess_function

  ds = readFile("training", cfg)
  train_ds = ds.map(preprocess, batched=True)

  ds = readFile("validation", cfg)
  validate_ds = ds.map(preprocess, batched=True)

  data_collator = (TripleDataCollator(tokenizer) if is_triple
                   else DataCollatorWithPadding(tokenizer=tokenizer))

  def compute_metrics(eval_pred):
      predictions, labels = eval_pred
      predictions = np.argmax(predictions, axis=1)
      # Accuracy computed directly to avoid evaluate.load("accuracy"), which
      # downloads a metric script from the HF hub at runtime and breaks against
      # newer huggingface_hub (the removed HfFolder API).
      return {"accuracy": float(np.mean(predictions == labels))}

  if is_triple:
      # Shared-encoder triple model: pretrained encoder + randomly-init MLP head.
      model = TripleEncoderForSequenceClassification.from_base_pretrained(
          cfg.modelName, num_labels=len(fieldTypesDict),
          id2label=fieldTypesReversedDict, label2id=fieldTypesDict,
          encoder_layers=cfg.encoderLayers,
          head_interactions=cfg.headInteractions, head_proj_dim=cfg.headProjDim)
      encoder = model.encoder
  elif use_mean:
      # Single-sequence, but classify off mean-pooled tokens instead of [CLS].
      model = MeanPoolForSequenceClassification.from_base_pretrained(
          cfg.modelName, num_labels=len(fieldTypesDict),
          id2label=fieldTypesReversedDict, label2id=fieldTypesDict,
          encoder_layers=cfg.encoderLayers)
      encoder = model.encoder
  else:
      model = AutoModelForSequenceClassification.from_pretrained(
          cfg.modelName, num_labels=len(fieldTypesDict), ignore_mismatched_sizes=True,
          id2label=fieldTypesReversedDict, label2id=fieldTypesDict
      )
      # Standard HF classifier: drop layers on the base encoder in place (config
      # num_hidden_layers is updated, so save/reload rebuilds the smaller model).
      total = model.config.num_hidden_layers
      kept = select_encoder_layers(model, cfg.encoderLayers)
      if kept is not None:
          print(f"  layer-drop: {total} -> {len(kept)} encoder layers {kept}")
      encoder = model.base_model

  if cfg.useLora:
      from peft import LoraConfig, get_peft_model, TaskType
      # task_type=SEQ_CLS lets PEFT auto-target the attention projections for
      # the base architecture (e.g. query/value for BERT-family models), so we
      # don't hardcode module names per model.
      lora_cfg = LoraConfig(
          task_type=TaskType.SEQ_CLS,
          r=cfg.loraR,
          lora_alpha=cfg.loraAlpha,
          lora_dropout=cfg.loraDropout,
      )
      model = get_peft_model(model, lora_cfg)
      model.print_trainable_parameters()

  # Enable Weights & Biases when a project is configured. We start the run
  # ourselves (rather than letting the Trainer auto-init) so it carries the
  # hyperparameters as config -- this is what parameter sweeps group/compare on
  # -- and so the eval step can resume the same run to log the final metrics.
  report_to = "none"
  if cfg.wandbProject:
      import wandb
      os.environ["WANDB_PROJECT"] = cfg.wandbProject
      wandb.init(
          project=cfg.wandbProject,
          id=cfg.wandbRunId or None,
          name=cfg.wandbRunName or None,
          resume="allow",
          config=wandb_config(cfg),
      )
      report_to = ["wandb"]

  # learningRate of 0.0 means "auto": LoRA wants a higher rate than full
  # fine-tuning, so default to 2e-4 with LoRA and the HF default (5e-5) without.
  lr = cfg.learningRate if cfg.learningRate > 0 else (2e-4 if cfg.useLora else 5e-5)

  training_args = TrainingArguments(
      seed=189,
      data_seed=189,
      output_dir=cfg.saveModelName,
      learning_rate=lr,
      per_device_train_batch_size=cfg.trainBatchSize,
      per_device_eval_batch_size=cfg.evalBatchSize,
      num_train_epochs=cfg.numEpochs,
      weight_decay=cfg.weightDecay,
      warmup_ratio=cfg.warmupRatio,
      # The Trainer auto-selects CUDA/MPS when available; this only forces CPU
      # when select_device() resolved to it (e.g. AUTOFILL_DEVICE=cpu).
      use_cpu=(device == "cpu"),
      eval_strategy="epoch",
      save_strategy="epoch",
      load_best_model_at_end=True,
      report_to=report_to,
      run_name=cfg.wandbRunName or None,
      # The triple model's forward takes per-section columns and its collator
      # reads the raw text_* columns off each feature, so keep all columns.
      remove_unused_columns=(not is_triple),
  )

  trainer_kwargs = dict(
      model=model,
      args=training_args,
      train_dataset=train_ds,
      eval_dataset=validate_ds,
      processing_class=tokenizer,
      data_collator=data_collator,
      compute_metrics=compute_metrics,
  )
  if cfg.headLearningRate > 0:
      # Discriminative LR: encoder on `lr`, classification head on headLearningRate.
      # `encoder` was captured above per model type. Trainer builds the scheduler
      # from this optimizer, so warmup_ratio still applies to both groups.
      optimizer = build_optimizer(model, encoder, lr, cfg.headLearningRate, cfg.weightDecay)
      print(f"Using separate head LR: encoder={lr}, head={cfg.headLearningRate}.")
      trainer_kwargs["optimizers"] = (optimizer, None)
  if cfg.closeLabelEps > 0:
      # Similarity-aware soft-label loss: 'close' predictions cost less than wrong ones.
      soft_targets = build_close_targets(
          fieldTypesDict, fieldNamesCloseDict,
          num_labels=model.config.num_labels, eps=cfg.closeLabelEps)
      print(f"Using close-aware soft-label loss (eps={cfg.closeLabelEps}).")
      trainer = CloseAwareTrainer(**trainer_kwargs, soft_targets=soft_targets)
  else:
      trainer = Trainer(**trainer_kwargs)

  trainer.train()

  if cfg.useLora:
      # (Not reached for context_format='triple'; guarded above.) Merge the LoRA
      # adapter into the base weights so the saved model is a plain
      # AutoModelForSequenceClassification. This keeps the evaluate / inference
      # path unchanged -- no PEFT needed to load the result.
      merged = trainer.model.merge_and_unload()
      trainer.model = merged

  trainer.save_model(cfg.saveModelDir)

  if cfg.wandbProject:
      import wandb
      # Flush the training run; the eval step resumes it to add final metrics.
      wandb.finish()

  return cfg.saveModelDir

def _triple_classify(texts, cfg, batch_size=64):
  """Inference for context_format='triple', mirroring the HF text-classification
  pipeline output: a list of {"label", "score"} dicts (one per input).

  The saved TripleEncoderForSequenceClassification is not a standard HF
  classifier, so it can't go through pipeline(). This loads it once, splits each
  raw text into (current, previous, next) via split_context, tokenizes each
  section, runs the shared encoder + fusion head, and returns the argmax label
  (via id2label) with its softmax probability -- so evaluate_model's downstream
  scoring/printing is unchanged.
  """
  device = select_device()
  tokenizer = AutoTokenizer.from_pretrained(cfg.modelName)
  model = TripleEncoderForSequenceClassification.from_pretrained(cfg.saveModelDir)
  model.to(device)
  model.eval()
  # Saved id2label keys arrive as strings from JSON; normalize to ints.
  id2label = {int(k): v for k, v in model.config.id2label.items()}

  results = []
  with torch.no_grad():
    for start in range(0, len(texts), batch_size):
      batch = texts[start:start + batch_size]
      splits = [split_context(t) for t in batch]
      enc = {}
      for i, section in enumerate(("current", "previous", "next")):
        tok = tokenizer([s[i] for s in splits], truncation=True, max_length=512,
                        padding=True, return_tensors="pt")
        enc[f"input_ids_{section}"] = tok["input_ids"].to(device)
        enc[f"attention_mask_{section}"] = tok["attention_mask"].to(device)
      probs = F.softmax(model(**enc).logits, dim=-1)
      scores, idxs = probs.max(dim=-1)
      for idx, score in zip(idxs.tolist(), scores.tolist()):
        results.append({"label": id2label[idx], "score": float(score)})
  return results


def evaluate_model(cfg, filename="testing"):
  # Without an explicit device the pipeline runs on CPU; select_device() routes
  # it to CUDA/MPS when available (override with AUTOFILL_DEVICE).
  device = select_device()
  print(f"Evaluating on device: {device}")
  is_triple = cfg.contextFormat == "triple"
  # The triple model isn't pipeline-loadable; it uses the custom path below.
  classifier = None if is_triple else pipeline(
      "text-classification", model=cfg.saveModelDir, truncation=True,
      max_length=512, device=device)

  list = []
  expectedList = []
  autocompleteList = []
  detailsList = []

  file = open(dataset_path(filename + cfg.dataVariant + ".txt"), encoding="utf-8")
  lines = file.readlines()

  count = 0
  for line in lines:
    line = line.strip()
    lineData = line.split(",", ignoreLineCount + 1)
    raw_text = lineData[ignoreLineCount + 1]
    # Re-encode bb/aa context the same way as training so eval stays consistent;
    # autocomplete detection below still works off the raw text.
    list.append(reformat_context(raw_text, cfg.contextFormat))

    if raw_text.startswith("a-c-"):
      actype = raw_text.split(" ", maxsplit=1)[0][4:]
      if actype in fieldTypesDict:
        autocompleteList.append(actype)
      else:
        autocompleteList.append(None)
    else:
      autocompleteList.append(None)

    print (lineData)
    expectedList.append(fieldTypesReversedDict[int(lineData[ignoreLineCount])])
    if ignoreLineCount == 2:
      detailsList.append([lineData[0], lineData[1]]);
    else:
      detailsList.append(["", ""]);

    count = count + 1

  for l in list:
    print(l)

  # For 'triple', `list` holds the raw text (reformat_context is a no-op for it);
  # _triple_classify splits each row itself. Both paths yield {"label","score"}.
  results = _triple_classify(list, cfg) if is_triple else classifier(list, truncation=True)

  correct = 0
  close = 0
  blank = 0
  fieldCorrect = {}
  localeCorrect = {}            # locale -> [correct, total]
  englishCorrect = [0, 0]       # [correct, total] for English-locale rows
  nonEnglishCorrect = [0, 0]    # [correct, total] for everything else

  for result in zip(results, expectedList, autocompleteList, detailsList):
    actualresult = None
    suffix = ""

#    if result[2] is not None:
#      actualresult = result[2]
#    else:
    actualresult = result[0]["label"]

    match = 0
    if actualresult == result[1]:
      correct += 1
      close += 1
      match = 1
    elif actualresult in fieldNamesCloseDict and result[1] in fieldNamesCloseDict[actualresult]:
      suffix = " -"
      close += 1
    else:
      suffix = " X"

    if result[2] is not None:
      suffix += " AC: " + result[2]

    if result[1] in fieldCorrect:
      fieldCorrect[result[1]] = (fieldCorrect[result[1]][0] + match, fieldCorrect[result[1]][1] + 1)
    else:
      fieldCorrect[result[1]] = (match, 1)

    if result[1] == "other" and actualresult != "other":
      blank += 1

    # Accuracy by locale / language, with an English vs non-English split. The
    # filename (detailsList[i][0]) encodes the source locale.
    locale, _language, is_english = classify_locale(result[3][0])
    lc = localeCorrect.setdefault(locale, [0, 0])
    lc[0] += match
    lc[1] += 1
    bucket = englishCorrect if is_english else nonEnglishCorrect
    bucket[0] += match
    bucket[1] += 1

    probability = result[0]["score"]
    if ignoreLineCount == 2:
      print(result[3][0] + "," + result[3][1] + "  ", end="")
    print(result[1].ljust(26, " ") + "  " + result[0]["label"].ljust(26, " ") + " " + f"{probability:.4f}" + suffix)

  totalAccuracy = correct / len(results)
  closeAccuracy = close / len(results)
  blankRate = blank / len(results)

  print(f"Total Accuracy: {totalAccuracy:.4f} {correct}/{len(results)}")
  print(f"Close Accuracy: {closeAccuracy:.4f} {close}/{len(results)}")
  print(f"Expect Blank: {blankRate:.4f} {blank}/{len(results)}")

  fieldAccuracy = {}
  print("Field Accuracy:")
  for field in sorted(fieldCorrect.keys()):
    fieldAccuracy[field] = fieldCorrect[field][0] / fieldCorrect[field][1]
    print("  " + field + " : " + str(fieldAccuracy[field]))

  localeAccuracy = {loc: c / n for loc, (c, n) in localeCorrect.items()}
  localeSupport = {loc: n for loc, (c, n) in localeCorrect.items()}
  englishAccuracy = englishCorrect[0] / englishCorrect[1] if englishCorrect[1] else 0.0
  nonEnglishAccuracy = nonEnglishCorrect[0] / nonEnglishCorrect[1] if nonEnglishCorrect[1] else 0.0
  print("Locale Accuracy:")
  for loc in sorted(localeAccuracy, key=lambda k: localeSupport[k], reverse=True):
    print(f"  {loc} : {localeAccuracy[loc]:.4f} ({localeSupport[loc]})")
  print(f"English Accuracy: {englishAccuracy:.4f} ({englishCorrect[1]})")
  print(f"Non-English Accuracy: {nonEnglishAccuracy:.4f} ({nonEnglishCorrect[1]})")

  for result in zip(results, expectedList, autocompleteList):
    if result[1] == "other" and result[0] == "other" and result[2] is None:
      print("SPECIAL: " + result[0] + " " + result.probability + "\n");

  # Standard classification metrics over the predicted vs. expected labels.
  y_test = expectedList
  y_pred = [result["label"] for result in results]
  metrics = compute_standard_metrics(y_test, y_pred, print_report=True)

  # Retain the existing autofill-specific measures alongside the standard ones.
  metrics.update({
    "total": correct,
    "count": len(results),
    "totalAccuracy": totalAccuracy,
    "closeAccuracy": closeAccuracy,
    "blankRate": blankRate,
    FIELD_ACCURACY: fieldAccuracy,
    LOCALE_ACCURACY: localeAccuracy,
    LOCALE_SUPPORT: localeSupport,
    ENGLISH_ACCURACY: englishAccuracy,
    NON_ENGLISH_ACCURACY: nonEnglishAccuracy,
    ENGLISH_COUNT: englishCorrect[1],
    NON_ENGLISH_COUNT: nonEnglishCorrect[1],
  })
  return metrics

def infer(text, cfg):
  if cfg.contextFormat == "triple":
    # Custom triple-encoder path (not pipeline-loadable).
    results = _triple_classify([text], cfg)
  else:
    classifier = pipeline("text-classification", model=cfg.saveModelDir)
    results = classifier([text])
  for result in results:
    print(result)

def forest(cfg):
  import pandas as pd
  from sklearn.feature_extraction.text import CountVectorizer
  from sklearn import metrics
  from sklearn.model_selection import train_test_split
  from sklearn.ensemble import RandomForestClassifier

  trainingDS = readFile("training", cfg)
  trainingDS = pd.DataFrame(trainingDS)
  trainingDS_X = trainingDS["text"]
  trainingDS_Y = trainingDS["label"]

  testingDS = readFile("testing", cfg)
  testingDS = pd.DataFrame(testingDS)
  testingDS_X = testingDS["text"]
  testingDS_Y = testingDS["label"]

  vectorizer = CountVectorizer();
  trainingXCount = vectorizer.fit_transform(trainingDS_X);
  testingXCount = vectorizer.transform(testingDS_X);

  random_forest_model = RandomForestClassifier()
  random_forest_model.fit(trainingXCount, trainingDS_Y)

  before = time.perf_counter()

  yprediction = random_forest_model.predict(testingXCount)

  duration = time.perf_counter() - before

  accuracy = metrics.accuracy_score(testingDS_Y, yprediction)

  print(f"  Random Forest: {accuracy:.2f}% Time: {duration:.2f}")

  correct = 0
  close = 0
  fieldCorrect = {}

  for result in zip(yprediction, testingDS_Y.values, testingDS_X.values):
    suffix = ""

    found_result = result[0]
#    if result[2].startswith("ac-"):
#      fieldtype = result[2][3:result[2].find(" ")]
#      if fieldtype in fieldTypesDict:
#        found_result = fieldTypesDict[fieldtype]

    found_field_type = fieldTypesReversedDict[found_result]

    match = 0
    if found_result == result[1]:
      correct += 1
      close += 1
      match = 1
    elif found_field_type in fieldNamesCloseDict and fieldTypesReversedDict[result[1]] in fieldNamesCloseDict[found_field_type]:
      suffix = " -"
      close += 1
    else:
      suffix = " X"

    fieldtype = fieldTypesReversedDict[result[1]]
    if fieldtype in fieldCorrect:
      fieldCorrect[fieldtype] = (fieldCorrect[fieldtype][0] + match, fieldCorrect[fieldtype][1] + 1)
    else:
      fieldCorrect[fieldtype] = (match, 1)

#    print(found_result + "  " + result[1] + suffix)
    print(fieldTypesReversedDict[found_result] + "  " + fieldTypesReversedDict[result[1]] + suffix)

  print("Total Accuracy: " + str(correct / len(yprediction)))
  print("Close Accuracy: " + str(close / len(yprediction)))

  print("Field Accuracy:")
  for field in sorted(fieldCorrect.keys()):
    print("  " + field + " : " + str(float(fieldCorrect[field][0]) / float(fieldCorrect[field][1])))


def main(argv):
  cfg = Config()
  if len(argv) == 2 and argv[1] == "train":
    train(cfg)
    evaluate_model(cfg, "testing")
  elif len(argv) == 2 and argv[1] == "test":
    evaluate_model(cfg, "testing")
  elif len(argv) == 2 and argv[1] == "together":
    evaluate_model(cfg, "together")
  elif len(argv) == 2 and argv[1] == "forest":
    forest(cfg)
  else:
    infer(" ".join(argv[1:]), cfg)


if __name__ == "__main__":
  main(sys.argv)
