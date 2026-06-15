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

from metaflow import FlowSpec, Parameter, step, kubernetes, current

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
)

# The headline scalar metrics logged to W&B and surfaced in the end step.
# (Everything in the eval result except the nested classification report.)
SUMMARY_METRIC_KEYS = [ACCURACY, F1, PRECISION, RECALL, KAPPA,
                       "totalAccuracy", "closeAccuracy", "blankRate"]

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
    eval_dataset = Parameter(
        "eval_dataset",
        help="Base name of the dataset to evaluate against (e.g. 'testing' or 'together').",
        default="testing",
    )
    wandb_project = Parameter(
        "wandb_project",
        help="Weights & Biases project to log to. Empty disables W&B logging.",
        default="",
    )

    def _config(self):
        cfg = Config(
            modelName=self.model_name,
            dataVariant=self.data_variant,
            numEpochs=self.num_epochs,
            modelSuffix=self.model_suffix,
        )
        if self.wandb_project:
            # One W&B run per Metaflow run, shared across the train and eval
            # steps so training curves and final metrics land together -- this
            # is what parameter sweeps compare across.
            cfg.wandbProject = self.wandb_project
            cfg.wandbRunId = f"{current.flow_name}-{current.run_id}"
            cfg.wandbRunName = f"{cfg.saveModelName}-{current.run_id}"
        return cfg

    @kubernetes(image=GPU_IMAGE)
    @step
    def start(self):
        """Resolve configuration and record where the model will be saved."""
        cfg = self._config()
        self.save_model_dir = cfg.saveModelDir
        self.save_model_name = cfg.saveModelName
        print(f"Training {self.model_name} -> {self.save_model_dir} "
              f"({self.num_epochs} epochs, variant '{self.data_variant}')")
        self.next(self.train)

    @kubernetes(image=GPU_IMAGE,
                gpu_vendor="nvidia",
                gpu=1
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

    @kubernetes(image=GPU_IMAGE,
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
            wandb.finish()

        print(f"Accuracy: {self.metrics[ACCURACY]:.4f}  F1: {self.metrics[F1]:.4f}")
        self.next(self.end)

    @step
    def end(self):
        """Done. Print the headline metrics (also stored as artifacts / in W&B)."""
        print("Flow complete. Headline metrics:")
        for key in SUMMARY_METRIC_KEYS:
            print(f"  {key}: {self.summary_metrics[key]:.4f}")


if __name__ == "__main__":
    AutofillFlow()
