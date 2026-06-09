"""Bind the databricks-ai-intern-eval job to UC registered models as their
MLflow Deployment Job, so any new model version automatically kicks off
scripts/eval_model.py.

Asset Bundles can't yet declare this link inline, so this one-shot script does
it after every bundle deploy via
``MlflowClient.update_registered_model(name, deployment_job_id=...)``
(mlflow >= 3.0; older builds fall back to the UC registry REST PATCH).

The link is **per-model**: by default the script binds every model currently
registered under ``<catalog>.<schema>``; pass ``--model`` to bind a single
one. Models registered *after* this run are not covered — re-run the script
(it is idempotent) or bind at registration time.

Usage::

    python scripts/wire_eval_trigger.py --job-id <id from bundle deploy> \\
                                        --catalog databricks_ai_intern --schema agent
    python scripts/wire_eval_trigger.py --job-id <id> --model my_finetune
"""

from __future__ import annotations

import argparse
import inspect
import logging
import sys
from types import SimpleNamespace

logger = logging.getLogger(__name__)


def _bind(mlflow_client, wc, name: str, job_id: str) -> None:
    params = inspect.signature(mlflow_client.update_registered_model).parameters
    if "deployment_job_id" in params:
        mlflow_client.update_registered_model(name=name, deployment_job_id=job_id)
    else:
        # Older mlflow: hit the UC registry REST surface directly.
        wc.api_client.do(
            "PATCH",
            "/api/2.0/mlflow/unity-catalog/registered-models/update",
            body={"name": name, "deployment_job_id": job_id},
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--catalog", default="databricks_ai_intern")
    parser.add_argument("--schema", default="agent")
    parser.add_argument("--model", help="bind one model (name within the schema) instead of all")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from agent.config import DatabricksConfig
    from agent.core import db_client

    settings = db_client.resolve_settings(SimpleNamespace(databricks=DatabricksConfig()))

    try:
        wc = db_client.get_workspace_client(settings)
        if args.model:
            names = [f"{args.catalog}.{args.schema}.{args.model}"]
        else:
            names = [
                m.full_name
                for m in wc.registered_models.list(
                    catalog_name=args.catalog, schema_name=args.schema
                )
            ]
        if not names:
            logger.warning(
                "No registered models under %s.%s yet — nothing to bind. "
                "Re-run after the first model registration (or pass --model).",
                args.catalog, args.schema,
            )
            return 0
        mlflow_client = db_client.get_mlflow_client()
    except Exception as e:
        logger.error("wire_eval_trigger failed: %s", e)
        return 1

    failed = 0
    for name in names:
        try:
            _bind(mlflow_client, wc, name, str(args.job_id))
            logger.info("Linked deployment job %s to %s", args.job_id, name)
        except Exception as e:
            logger.error("Failed to link job %s to %s: %s", args.job_id, name, e)
            failed += 1
    if failed:
        logger.error("%d of %d binding(s) failed", failed, len(names))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
