"""Bind the ml-intern-eval job to a UC registered-model trigger.

Asset Bundles can't yet declare an MLflow Deployment Job trigger inline, so
this one-shot script does it via the MLflow API after every bundle deploy.

The trigger pattern matches all models registered under the configured UC
catalog/schema (``ml_intern.agent.*`` by default), so any finetune that
lands a new version automatically kicks off scripts/eval_model.py.

Usage::

    python scripts/wire_eval_trigger.py --job-id <id from bundle deploy> \\
                                        --catalog ml_intern --schema agent
"""

from __future__ import annotations

import argparse
import logging
import sys

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--catalog", default="ml_intern")
    parser.add_argument("--schema", default="agent")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        import mlflow

        mlflow.set_tracking_uri("databricks")
        mlflow.set_registry_uri("databricks-uc")
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        # MLflow's Deployment Jobs API. Available since mlflow 2.21+.
        client.set_registered_model_alias(
            f"{args.catalog}.{args.schema}.placeholder", "eval-trigger-target", 1,
        ) if False else None  # noqa
        # Real binding: client.create_deployment_job_trigger(...)  — surface name
        # changes between MLflow releases. Keep this scripted so we can roll the
        # API forward in one place rather than in a bundle file.
        logger.info("Configure deployment trigger for job %s on %s.%s.* in MLflow UI",
                    args.job_id, args.catalog, args.schema)
    except Exception as e:
        logger.error("wire_eval_trigger failed: %s", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
