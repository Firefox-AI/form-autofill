#!/usr/bin/env python
"""Extract a trained model from a Metaflow run into a local directory.

The training step stores the saved Hugging Face model directory as a zipped
artifact (``model_artifact``). This pulls that artifact for a given run and
unzips it so the model can be exported / quantized / evaluated locally.

Example:

    uv run python extract_model.py \
        --run-id argo-autofillflow-jt4qd \
        --namespace production:autofillflow-0-egrc \
        --output outputs/autofill-tiny-supported-argo-autofillflow-jt4qd
"""

import argparse
import io
import os
import zipfile

from metaflow import Run, namespace as set_namespace


# Artifacts whose values are worth printing as provenance after extraction.
_PROVENANCE_KEYS = [
    "model_name", "use_lora", "learning_rate", "train_batch_size",
    "eval_batch_size", "weight_decay", "num_epochs", "data_variant", "train_file",
]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True,
                    help="Metaflow run id (e.g. argo-autofillflow-jt4qd) or full FlowName/RunID pathspec.")
    ap.add_argument("--flow", default="AutofillFlow",
                    help="Flow name, used when --run-id is not a full pathspec.")
    ap.add_argument("--namespace", default=None,
                    help="Metaflow namespace (e.g. production:autofillflow-0-egrc).")
    ap.add_argument("--output", required=True, help="Directory to unzip the model into.")
    ap.add_argument("--step", default="train", help="Step that holds the model artifact.")
    ap.add_argument("--artifact", default="model_artifact",
                    help="Name of the zipped-model artifact on the step.")
    args = ap.parse_args()

    if args.namespace:
        set_namespace(args.namespace)

    pathspec = args.run_id if "/" in args.run_id else f"{args.flow}/{args.run_id}"
    run = Run(pathspec)
    if not run.successful:
        print(f"WARNING: run {pathspec} is not marked successful (finished={run.finished}).")

    task = run[args.step].task
    blob = task[args.artifact].data

    os.makedirs(args.output, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        zf.extractall(args.output)
    print(f"Extracted {len(blob)} bytes -> {args.output}")

    data = task.data
    print("Run provenance:")
    for key in _PROVENANCE_KEYS:
        if hasattr(data, key):
            print(f"  {key}: {getattr(data, key)}")


if __name__ == "__main__":
    main()
