"""Placeholder eval script invoked by resources/eval_job.yml.

Triggered by the MLflow Deployment Job whenever a new model version lands
under ``ml_intern.agent.*``. Receives ``--model <full_name>`` and
``--version <int>`` from the trigger; writes a tag on that version with
the eval result so downstream alias rolls (e.g. ``set_alias champion``)
can gate on it.

Replace with real evals (lm-eval-harness, custom holdout) once a first
finetune lands. Today this just records ``eval=placeholder`` so the
trigger plumbing is testable.
"""

from __future__ import annotations

import argparse
import logging
import sys

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri("databricks")
        mlflow.set_registry_uri("databricks-uc")
        client = MlflowClient()
        client.set_model_version_tag(
            name=args.model,
            version=str(args.version),
            key="ml_intern_eval",
            value="placeholder",
        )
        logger.info("Tagged %s/%s eval=placeholder", args.model, args.version)
    except Exception as e:
        logger.error("Eval tagging failed: %s", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
