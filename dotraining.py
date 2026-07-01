from dataclasses import dataclass, field

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    pipeline,
    set_seed,
    AutoConfig
)

from datasets import Dataset

import os
import sys
import time
import random
import numpy as np
import torch

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
CONTEXT_FORMATS = ("bb", "sep", "symbol")

# Single-token markers for the "symbol" format. Both are already single tokens
# in the bert-base-uncased vocab (so no tokenizer/embedding changes are needed),
# and using two distinct symbols gives the previous vs next sections their own
# learned embeddings -- a directional signal a shared [SEP] can't provide.
PREV_SYMBOL = "•"   # bullet: marks the start of the previous-field section
NEXT_SYMBOL = "§"   # section sign: marks the start of the next-field section


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
  unchanged (the original behavior). Applied on the fly; data files are not
  modified.
  """
  if fmt == "bb":
    return text
  if fmt not in ("sep", "symbol"):
    raise ValueError(f"Unknown context_format {fmt!r}; expected one of {CONTEXT_FORMATS}")
  current, previous, nxt = [], [], []
  for word in text.split():
    if len(word) > 2 and word.startswith("bb"):
      previous.append(word[2:])
    elif len(word) > 2 and word.startswith("aa"):
      nxt.append(word[2:])
    else:
      current.append(word)
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
    "gen_to_real_ratio": cfg.genToRealRatio,
    "cc_to_real_ratio": cfg.ccToRealRatio,
    "learning_rate": cfg.learningRate,
    "train_batch_size": cfg.trainBatchSize,
    "eval_batch_size": cfg.evalBatchSize,
    "weight_decay": cfg.weightDecay,
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
      rec = {
        "label": int(lineData[ignoreLineCount]),
        "text": reformat_context(lineData[ignoreLineCount + 1], cfg.contextFormat),
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

def train(cfg):
  device = select_device()
  print(f"Training on device: {device}")
  tokenizer = AutoTokenizer.from_pretrained(cfg.modelName)

  def preprocess_function(examples):
      return tokenizer(examples["text"], truncation=True, max_length=512)

  ds = readFile("training", cfg)
  train_ds = ds.map(preprocess_function, batched=True)

  ds = readFile("validation", cfg)
  validate_ds = ds.map(preprocess_function, batched=True)

  data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

  def compute_metrics(eval_pred):
      predictions, labels = eval_pred
      predictions = np.argmax(predictions, axis=1)
      # Accuracy computed directly to avoid evaluate.load("accuracy"), which
      # downloads a metric script from the HF hub at runtime and breaks against
      # newer huggingface_hub (the removed HfFolder API).
      return {"accuracy": float(np.mean(predictions == labels))}

  model = AutoModelForSequenceClassification.from_pretrained(
      cfg.modelName, num_labels=len(fieldTypesDict), ignore_mismatched_sizes=True,
      id2label=fieldTypesReversedDict, label2id=fieldTypesDict
  )

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
      # The Trainer auto-selects CUDA/MPS when available; this only forces CPU
      # when select_device() resolved to it (e.g. AUTOFILL_DEVICE=cpu).
      use_cpu=(device == "cpu"),
      eval_strategy="epoch",
      save_strategy="epoch",
      load_best_model_at_end=True,
      report_to=report_to,
      run_name=cfg.wandbRunName or None,
  )

  trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=train_ds,
      eval_dataset=validate_ds,
      processing_class=tokenizer,
      data_collator=data_collator,
      compute_metrics=compute_metrics,
  )

  trainer.train()

  if cfg.useLora:
      # Merge the LoRA adapter into the base weights so the saved model is a
      # plain AutoModelForSequenceClassification. This keeps the evaluate /
      # inference path unchanged -- no PEFT needed to load the result.
      merged = trainer.model.merge_and_unload()
      trainer.model = merged

  trainer.save_model(cfg.saveModelDir)

  if cfg.wandbProject:
      import wandb
      # Flush the training run; the eval step resumes it to add final metrics.
      wandb.finish()

  return cfg.saveModelDir

def evaluate_model(cfg, filename="testing"):
  # Without an explicit device the pipeline runs on CPU; select_device() routes
  # it to CUDA/MPS when available (override with AUTOFILL_DEVICE).
  device = select_device()
  print(f"Evaluating on device: {device}")
  classifier = pipeline("text-classification", model=cfg.saveModelDir, truncation=True, max_length=512, device=device)

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

  results = classifier(list, truncation=True)

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
