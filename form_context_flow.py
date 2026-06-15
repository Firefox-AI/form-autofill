"""Metaflow flow for the form-context experiment.

Tests whether classifying each form field together with its neighbours (rather
than in isolation) improves accuracy. Steps:

  start      -> resolve config
  validate   -> group rows into forms (train, and merged val+test) and validate
                grouping/ordering
  features   -> run the base model over every element, build per-form padded
                probability stacks, and turn them into windowed feature matrices
                (GPU step)
  train_one  -> foreach over [mlp, logreg, rf]: fit on the training features and
                score on the test features
  join       -> gather per-method metrics + the no-context baseline, log to W&B
  end        -> print the comparison

Run it:
    python form_context_flow.py run
    python form_context_flow.py run --wandb_project autofill-form-context

The test set is built by merging validation+testing and regrouping by form id,
because those two files are element-wise slices of the same QA forms (see
form_context.py / the validate step).
"""

from metaflow import FlowSpec, Parameter, step, current

import form_context as fc
from dotraining import (
    Config,
    DEFAULT_DATA_VARIANT,
    DEFAULT_MODEL_NAME,
    compute_standard_metrics,
    wandb_config,
    ACCURACY,
    F1,
    KAPPA,
    PRECISION,
    RECALL,
)

# Scalar metrics surfaced per method (and for the baseline) in W&B / the end step.
SCALAR_KEYS = [ACCURACY, F1, PRECISION, RECALL, KAPPA]


class FormContextFlow(FlowSpec):
    """Compare neighbour-aware field classification against the base model."""

    model_name = Parameter("model_name", help="Base model name (for reference).",
                           default=DEFAULT_MODEL_NAME)
    data_variant = Parameter("data_variant", help="Dataset variant suffix.",
                            default=DEFAULT_DATA_VARIANT)
    wandb_project = Parameter("wandb_project",
                             help="W&B project to log to. Empty disables logging.",
                             default="")
    window = Parameter("window",
                      help="Neighbours considered on each side (1 = prev/current/next).",
                      default=fc.DEFAULT_WINDOW, type=int)
    autofill_flow_name = Parameter("autofill_flow_name",
                                  help="Flow whose latest successful run supplies the base model.",
                                  default="AutofillFlow")
    autofill_run_id = Parameter("autofill_run_id",
                               help="Specific AutofillFlow run id for the base model. "
                                    "Empty uses the latest successful run.",
                               default="")
    autofill_namespace = Parameter("autofill_namespace",
                                  help="Metaflow namespace the AutofillFlow run lives in "
                                       "(remote/production runs are not in your user namespace).",
                                  default="production:autofillflow-0-egrc")

    def _config(self):
        return Config(modelName=self.model_name, dataVariant=self.data_variant)

    @step
    def start(self):
        """Resolve config."""
        cfg = self._config()
        print(f"Form-context experiment using base model {cfg.saveModelDir}")
        self.next(self.validate)

    @step
    def validate(self):
        """Group rows into forms and validate grouping/ordering for train + test."""
        cfg = self._config()
        train_rows = fc.read_form_rows("training", cfg)
        # Merge validation+testing: they are element-wise slices of the same QA
        # forms, so regrouping by form id reassembles whole, ordered forms.
        test_rows = fc.read_form_rows("validation", cfg) + fc.read_form_rows("testing", cfg)
        self.train_stats = fc.validate_forms("train", fc.group_into_forms(train_rows), train_rows)
        self.test_stats = fc.validate_forms("test(merged val+test)",
                                            fc.group_into_forms(test_rows), test_rows)
        self.next(self.features)

    @step
    def features(self):
        """Base-model inference -> per-form stacks -> windowed feature matrices."""
        cfg = self._config()
        # Prefer the freshly-trained model from the AutofillFlow run; fall back
        # to the committed model on GitHub if no run/metadata is available.
        try:
            self.base_model_source = fc.model_from_run(
                cfg, self.autofill_flow_name, self.autofill_run_id,
                self.autofill_namespace)
        except Exception as e:
            print(f"Could not load model from {self.autofill_flow_name} run "
                  f"({type(e).__name__}: {e}); downloading committed model instead.")
            self.base_model_source = fc.ensure_model(cfg)
        train_rows = fc.read_form_rows("training", cfg)
        test_rows = fc.read_form_rows("validation", cfg) + fc.read_form_rows("testing", cfg)
        self.X_train, self.y_train, _ = fc.build_dataset(train_rows, cfg, self.window)
        self.X_test, self.y_test, baseline_pred = fc.build_dataset(test_rows, cfg, self.window)
        # No-context baseline scored the same way as the context models.
        self.baseline_metrics = compute_standard_metrics(self.y_test, baseline_pred)
        print(f"features (window={self.window}): X_train={self.X_train.shape} "
              f"X_test={self.X_test.shape} baseline acc={self.baseline_metrics[ACCURACY]:.4f}")
        self.model_kinds = fc.MODEL_KINDS
        self.next(self.train_one, foreach="model_kinds")

    @step
    def train_one(self):
        """Fit and score one method (mlp / logreg / rf)."""
        self.kind = self.input
        self.metrics = fc.train_and_eval(self.kind, self.X_train, self.y_train,
                                          self.X_test, self.y_test)
        print(f"{self.kind}: acc={self.metrics[ACCURACY]:.4f} f1={self.metrics[F1]:.4f}")
        self.next(self.join)

    @step
    def join(self, inputs):
        """Collect per-method metrics + baseline; log the comparison to W&B."""
        self.baseline_metrics = inputs[0].baseline_metrics
        self.model_metrics = {inp.kind: inp.metrics for inp in inputs}

        # Flat scalar comparison: baseline + each method.
        self.comparison = {"baseline": {k: self.baseline_metrics[k] for k in SCALAR_KEYS}}
        for kind, m in self.model_metrics.items():
            self.comparison[kind] = {k: m[k] for k in SCALAR_KEYS}

        if self.wandb_project:
            from common.secrets import load_env
            load_env()
            import wandb
            cfg = self._config()
            cfg.wandbProject = self.wandb_project
            wandb.init(project=self.wandb_project,
                       id=f"{current.flow_name}-{current.run_id}",
                       name=f"form-context-{current.run_id}",
                       resume="allow",
                       config={**wandb_config(cfg), "window": self.window})
            flat = {f"{group}/{k}": v
                    for group, metrics in self.comparison.items()
                    for k, v in metrics.items()}
            wandb.log(flat)
            wandb.summary.update(flat)
            wandb.finish()

        self.next(self.end)

    @step
    def end(self):
        """Print the baseline-vs-context comparison."""
        print("Form-context comparison (test set):")
        header = "  {:10s} " + " ".join("{:>10s}" for _ in SCALAR_KEYS)
        print(header.format("method", *SCALAR_KEYS))
        for group in ["baseline"] + list(self.model_metrics.keys()):
            vals = self.comparison[group]
            print(("  {:10s} " + " ".join("{:10.4f}" for _ in SCALAR_KEYS)).format(
                group, *[vals[k] for k in SCALAR_KEYS]))


if __name__ == "__main__":
    FormContextFlow()
