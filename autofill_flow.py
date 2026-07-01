"""Metaflow flow for training and evaluating the Firefox address autofill model.

This wraps the logic in dotraining.py so it runs as a Metaflow job, with
training and evaluation as separate steps. Splitting them means the (expensive)
training step runs once and the evaluation step can be resumed/re-run on its
own against the already-trained model.

Run it locally with:

    python autofill_flow.py run

Override any of the configuration parameters, e.g.:

    python autofill_flow.py run --num_epochs 5 --data_variant -supported

Inspect the metrics produced by the eval step afterwards:

    python autofill_flow.py dump <run-id>/evaluate

Notes on data and artifacts:
  * The dataset files (training<variant>.txt, validation<variant>.txt,
    testing<variant>.txt) are read with paths relative to the working
    directory, so launch the flow from the repository root.
  * The trained model is written to disk under output-models/ and the eval
    step reloads it from there. For local (and --with conda/local-metadata)
    runs the filesystem is shared between steps so this works directly. If you
    move this to a remote/distributed backend (e.g. @kubernetes), persist the
    model directory to a shared store (S3 via the Metaflow S3 datastore, or
    zip it into a self artifact) instead of relying on the local filesystem.
"""

import io
import os
import zipfile

# Ensure the dataset .txt files are packaged with the run. The dotraining code
# reads them from the working directory (no longer downloading them), so remote
# @kubernetes steps only receive them if .txt is in the package suffixes. This
# is set before importing metaflow -- which reads the value at import time --
# and intentionally augments (rather than trusts) any narrower
# METAFLOW_DEFAULT_PACKAGE_SUFFIXES inherited from the shell.
_pkg_suffixes = [
    s for s in os.environ.get("METAFLOW_DEFAULT_PACKAGE_SUFFIXES", ".py,.json").split(",") if s
]
if ".txt" not in _pkg_suffixes:
    _pkg_suffixes.append(".txt")
os.environ["METAFLOW_DEFAULT_PACKAGE_SUFFIXES"] = ",".join(_pkg_suffixes)

from metaflow import FlowSpec, Parameter, step, kubernetes, current

# Set AUTOFILL_LOCAL=1 to run every step as a local process instead of on
# Kubernetes -- this turns the @kubernetes decorators below into no-ops so the
# flow runs entirely on this machine (fastest for quick local iteration). Pair
# it with a local Metaflow profile, e.g.:
#     AUTOFILL_LOCAL=1 METAFLOW_PROFILE=local python autofill_flow.py run ...
_RUN_LOCAL = os.environ.get("AUTOFILL_LOCAL") == "1"


def maybe_kubernetes(**kwargs):
    """@kubernetes(...) normally, but a no-op when AUTOFILL_LOCAL=1."""
    def decorator(func):
        return func if _RUN_LOCAL else kubernetes(**kwargs)(func)
    return decorator

from dotraining import (
    Config,
    DEFAULT_DATA_VARIANT,
    DEFAULT_MODEL_NAME,
    DEFAULT_NUM_EPOCHS,
    train,
    evaluate_model,
    wandb_config,
    ACCURACY,
    CLASSIFICATION_REPORT,
    F1,
    KAPPA,
    PRECISION,
    RECALL,
    FIELD_ACCURACY,
    LOCALE_ACCURACY,
    LOCALE_SUPPORT,
    ENGLISH_ACCURACY,
    NON_ENGLISH_ACCURACY,
    ENGLISH_COUNT,
    NON_ENGLISH_COUNT,
)

# The headline scalar metrics logged to W&B and surfaced in the end step.
# (Everything in the eval result except the nested classification report.)
SUMMARY_METRIC_KEYS = [ACCURACY, F1, PRECISION, RECALL, KAPPA,
                       "totalAccuracy", "closeAccuracy", "blankRate",
                       ENGLISH_ACCURACY, NON_ENGLISH_ACCURACY]

# Image built from ml-services/ml_shared/dockers/metaflow_autofill_generation.
# Swap the tag to your branch name to test a branch build, e.g.
# metaflow_autofill_generation:my-branch-name.
GPU_IMAGE = (
    "us-docker.pkg.dev/moz-fx-mozsoc-ml-nonprod/metaflow-dockers/"
    "metaflow_autofill_generation:autofill_docker"
)


def _zip_dir(path):
    """Return the contents of a directory as the bytes of a zip archive."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(path):
            for name in files:
                filepath = os.path.join(root, name)
                zf.write(filepath, os.path.relpath(filepath, path))
    return buf.getvalue()


def _unzip_to(blob, path):
    """Extract a zip archive (as bytes) into the given directory."""
    os.makedirs(path, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        zf.extractall(path)


def _wandb_detail_payload(metrics, wandb):
    """Build per-class and per-locale W&B metrics from an eval result.

    Returns (scalars, tables): scalars are namespaced so each field/locale is a
    chartable metric (e.g. ``per_class_f1/address-line1``,
    ``locale_accuracy/en-US``); tables give a sortable single-run view.
    """
    aggregate_keys = {"accuracy", "macro avg", "weighted avg"}
    report = metrics.get(CLASSIFICATION_REPORT, {}) or {}
    field_acc = metrics.get(FIELD_ACCURACY, {}) or {}
    locale_acc = metrics.get(LOCALE_ACCURACY, {}) or {}
    locale_sup = metrics.get(LOCALE_SUPPORT, {}) or {}

    scalars = {}
    # Per-class scalars -- one chartable series per field type.
    for field, acc in field_acc.items():
        scalars[f"per_class_accuracy/{field}"] = acc
    for field, stats in report.items():
        if field in aggregate_keys or not isinstance(stats, dict):
            continue
        if "precision" in stats:
            scalars[f"per_class_precision/{field}"] = stats["precision"]
        if "recall" in stats:
            scalars[f"per_class_recall/{field}"] = stats["recall"]
        if "f1-score" in stats:
            scalars[f"per_class_f1/{field}"] = stats["f1-score"]
    # Per-locale accuracy + the English split.
    for loc, acc in locale_acc.items():
        scalars[f"locale_accuracy/{loc}"] = acc
    scalars["accuracy_english"] = metrics.get(ENGLISH_ACCURACY, 0.0)
    scalars["accuracy_non_english"] = metrics.get(NON_ENGLISH_ACCURACY, 0.0)

    tables = {}
    pc = wandb.Table(columns=["field", "support", "precision", "recall", "f1", "accuracy"])
    for field in sorted(field_acc):
        stats = report.get(field) if isinstance(report.get(field), dict) else {}
        pc.add_data(field, stats.get("support", ""), stats.get("precision", ""),
                    stats.get("recall", ""), stats.get("f1-score", ""), field_acc[field])
    tables["per_class_metrics"] = pc

    lt = wandb.Table(columns=["locale", "support", "accuracy"])
    for loc in sorted(locale_acc, key=lambda k: locale_sup.get(k, 0), reverse=True):
        lt.add_data(loc, locale_sup.get(loc, ""), locale_acc[loc])
    tables["locale_accuracy_table"] = lt

    return scalars, tables


class AutofillFlow(FlowSpec):
    """Train and evaluate the autofill field-type classifier."""

    model_name = Parameter(
        "model_name",
        help="Source model to fine-tune.",
        default=DEFAULT_MODEL_NAME,
    )
    data_variant = Parameter(
        "data_variant",
        help="Dataset variant suffix: '-supported' for supported mode, '' for all mode.",
        default=DEFAULT_DATA_VARIANT,
    )
    num_epochs = Parameter(
        "num_epochs",
        help="Number of training epochs.",
        default=DEFAULT_NUM_EPOCHS,
        type=int,
    )
    model_suffix = Parameter(
        "model_suffix",
        help="Extra suffix appended to the saved model name to test variations, e.g. '-updated'.",
        default="",
    )
    train_file = Parameter(
        "train_file",
        help="Training dataset filename. Empty uses 'training<data_variant>.txt'; "
             "set e.g. 'training-supported-expanded.txt' to train on a different file.",
        default="",
    )
    english_only = Parameter(
        "english_only",
        help="Train/validate on English-locale rows only. Evaluation still runs on "
             "the full test set so the English vs non-English split stays visible.",
        default=False,
        type=bool,
    )
    context_format = Parameter(
        "context_format",
        help="How prev/next-field context is encoded at load time: 'bb' keeps the "
             "raw bb/aa per-word prefixes; 'sep' regroups them into [SEP]-delimited "
             "sections; 'symbol' uses distinct single-token markers (• previous, "
             "§ next). Reformatted on the fly; data files unchanged.",
        default="bb",
    )
    gen_to_real_ratio = Parameter(
        "gen_to_real_ratio",
        help="Cap GEN_* (synthetic) training rows at this ratio of real rows to "
             "reduce overfit, e.g. 1.0 = at most 1x as many GEN as real. "
             "<=0 keeps all. Subsamples on the fly; data files unchanged.",
        default=0.0,
        type=float,
    )
    cc_to_real_ratio = Parameter(
        "cc_to_real_ratio",
        help="Cap CC_* (common crawl) training rows at this ratio of real rows. "
             "<=0 keeps all. Subsamples on the fly; data files unchanged.",
        default=0.0,
        type=float,
    )
    eval_dataset = Parameter(
        "eval_dataset",
        help="Base name of the dataset to evaluate against (e.g. 'testing' or 'together').",
        default="testing",
    )
    wandb_project = Parameter(
        "wandb_project",
        help="Weights & Biases project to log to. Empty disables W&B logging.",
        default="form_autofill_flow",
    )
    learning_rate = Parameter(
        "learning_rate",
        help="Learning rate. 0 = auto (HF default 5e-5, or 2e-4 with --use_lora).",
        default=0.0,
        type=float,
    )
    train_batch_size = Parameter(
        "train_batch_size",
        help="Per-device training batch size.",
        default=8,
        type=int,
    )
    eval_batch_size = Parameter(
        "eval_batch_size",
        help="Per-device eval batch size.",
        default=8,
        type=int,
    )
    weight_decay = Parameter(
        "weight_decay",
        help="Weight decay.",
        default=0.0,
        type=float,
    )
    use_lora = Parameter(
        "use_lora",
        help="Enable LoRA (parameter-efficient) fine-tuning.",
        default=False,
        type=bool,
    )
    lora_r = Parameter(
        "lora_r",
        help="LoRA rank.",
        default=8,
        type=int,
    )
    lora_alpha = Parameter(
        "lora_alpha",
        help="LoRA alpha (scaling).",
        default=16,
        type=int,
    )
    lora_dropout = Parameter(
        "lora_dropout",
        help="LoRA dropout.",
        default=0.1,
        type=float,
    )

    def _config(self):
        cfg = Config(
            modelName=self.model_name,
            dataVariant=self.data_variant,
            numEpochs=self.num_epochs,
            modelSuffix=self.model_suffix,
            trainFile=self.train_file,
            englishOnly=self.english_only,
            contextFormat=self.context_format,
            genToRealRatio=self.gen_to_real_ratio,
            ccToRealRatio=self.cc_to_real_ratio,
            learningRate=self.learning_rate,
            trainBatchSize=self.train_batch_size,
            evalBatchSize=self.eval_batch_size,
            weightDecay=self.weight_decay,
            useLora=self.use_lora,
            loraR=self.lora_r,
            loraAlpha=self.lora_alpha,
            loraDropout=self.lora_dropout,
        )
        if self.wandb_project:
            # One W&B run per Metaflow run, shared across the train and eval
            # steps so training curves and final metrics land together -- this
            # is what parameter sweeps compare across.
            cfg.wandbProject = self.wandb_project
            cfg.wandbRunId = f"{current.flow_name}-{current.run_id}"
            cfg.wandbRunName = f"{cfg.saveModelName}-{current.run_id}"
        return cfg

    @maybe_kubernetes(image=GPU_IMAGE)
    @step
    def start(self):
        """Resolve configuration and record where the model will be saved."""
        cfg = self._config()
        self.save_model_dir = cfg.saveModelDir
        self.save_model_name = cfg.saveModelName
        train_file = self.train_file or f"training{self.data_variant}.txt"
        print(f"Training {self.model_name} -> {self.save_model_dir} "
              f"({self.num_epochs} epochs, variant '{self.data_variant}', "
              f"train_file '{train_file}', english_only={self.english_only}, "
              f"context_format={self.context_format}, "
              f"gen_ratio={self.gen_to_real_ratio}, cc_ratio={self.cc_to_real_ratio}, "
              f"lora={self.use_lora}, lr={self.learning_rate or 'auto'}, "
              f"train_batch={self.train_batch_size})")
        self.next(self.train)

    @maybe_kubernetes(image=GPU_IMAGE,
                gpu_vendor="nvidia",
                gpu=1,
              )
    @step
    def train(self):
        """Fine-tune the model and save it to the output-models directory."""
        if self.wandb_project:
            # Loads WANDB_API_KEY (and other job secrets) into the environment.
            # Imported lazily so the deploy-time flow import doesn't require the
            # GCP secret-manager deps that common.secrets pulls in.
            from common.secrets import load_env
            load_env()
        cfg = self._config()
        self.save_model_dir = train(cfg)
        # Persist the trained model as a Metaflow artifact so the separate
        # evaluate step can reload it. Remote @kubernetes steps don't share a
        # filesystem, so we ship the model through the datastore rather than
        # relying on output-models/ being present in the next pod.
        self.model_artifact = _zip_dir(self.save_model_dir)
        print(f"Saved trained model to {self.save_model_dir} "
              f"({len(self.model_artifact)} bytes archived)")
        self.next(self.evaluate)

    @maybe_kubernetes(image=GPU_IMAGE,
                gpu_vendor="nvidia",
                gpu=1
                )
    @step
    def evaluate(self):
        """Evaluate the trained model, store metrics, and log them to W&B."""
        if self.wandb_project:
            # Loads WANDB_API_KEY before we resume the run to log metrics.
            from common.secrets import load_env
            load_env()
        cfg = self._config()
        # Restore the model packaged by the train step into the directory the
        # classifier pipeline expects to load from.
        _unzip_to(self.model_artifact, cfg.saveModelDir)
        self.metrics = evaluate_model(cfg, self.eval_dataset)
        self.classification_report = self.metrics[CLASSIFICATION_REPORT]
        # Headline scalars, stored as their own artifacts and logged to W&B.
        self.summary_metrics = {k: self.metrics[k] for k in SUMMARY_METRIC_KEYS}
        # Per-class and per-locale breakdowns, also kept as artifacts for `dump`.
        self.field_accuracy = self.metrics[FIELD_ACCURACY]
        self.locale_accuracy = self.metrics[LOCALE_ACCURACY]
        self.locale_support = self.metrics[LOCALE_SUPPORT]

        if self.wandb_project:
            import wandb
            # Resume the run started in train so the eval metrics attach to the
            # same run (and the same hyperparameter config) for sweeps.
            wandb.init(
                project=cfg.wandbProject,
                id=cfg.wandbRunId,
                name=cfg.wandbRunName,
                resume="allow",
                config=wandb_config(cfg),
            )
            wandb.log(self.summary_metrics)
            wandb.summary.update(self.summary_metrics)
            # Per-class (incl. address-line1) and per-locale metrics as chartable
            # scalars + sortable tables.
            detail_scalars, detail_tables = _wandb_detail_payload(self.metrics, wandb)
            wandb.log({**detail_scalars, **detail_tables})
            wandb.summary.update(detail_scalars)
            wandb.finish()

        print(f"Accuracy: {self.metrics[ACCURACY]:.4f}  F1: {self.metrics[F1]:.4f}  "
              f"English: {self.metrics[ENGLISH_ACCURACY]:.4f} "
              f"({self.metrics[ENGLISH_COUNT]})  "
              f"Non-English: {self.metrics[NON_ENGLISH_ACCURACY]:.4f} "
              f"({self.metrics[NON_ENGLISH_COUNT]})")
        self.next(self.end)

    @step
    def end(self):
        """Done. Print the headline metrics (also stored as artifacts / in W&B)."""
        print("Flow complete. Headline metrics:")
        for key in SUMMARY_METRIC_KEYS:
            print(f"  {key}: {self.summary_metrics[key]:.4f}")


if __name__ == "__main__":
    AutofillFlow()
