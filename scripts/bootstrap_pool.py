"""Bootstrap (or refresh) the ml-intern warm-GPU instance pool.

Asset Bundles don't yet expose ``instance_pools`` as a first-class resource
type, so the pool is created via the SDK on first deploy. Idempotent —
re-running with the same name updates the existing pool's capacity / node
type rather than creating a duplicate.

Usage::

    python scripts/bootstrap_pool.py                        # defaults
    python scripts/bootstrap_pool.py --name ml-intern-warm \\
        --node-type-id g5.xlarge \\
        --runtime 15.4.x-gpu-ml-scala2.12 \\
        --max-capacity 4
"""

from __future__ import annotations

import argparse
import logging
import sys

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="ml-intern-warm")
    parser.add_argument("--node-type-id", default="g5.xlarge")
    parser.add_argument("--runtime", default="15.4.x-gpu-ml-scala2.12")
    parser.add_argument("--min-idle", type=int, default=0)
    parser.add_argument("--max-capacity", type=int, default=4)
    parser.add_argument("--idle-autoterminate-min", type=int, default=15)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from databricks.sdk import WorkspaceClient

    wc = WorkspaceClient()
    body = {
        "instance_pool_name": args.name,
        "node_type_id": args.node_type_id,
        "min_idle_instances": args.min_idle,
        "max_capacity": args.max_capacity,
        "idle_instance_autotermination_minutes": args.idle_autoterminate_min,
        "preloaded_spark_versions": [args.runtime],
        "custom_tags": {"ml_intern_purpose": "sandbox-pool"},
    }

    existing = None
    for p in wc.api_client.do("GET", "/api/2.0/instance-pools/list").get("instance_pools", []):
        if p.get("instance_pool_name") == args.name:
            existing = p
            break

    if existing:
        body["instance_pool_id"] = existing["instance_pool_id"]
        wc.api_client.do("POST", "/api/2.0/instance-pools/edit", body=body)
        logger.info("Updated pool %s (id=%s)", args.name, existing["instance_pool_id"])
        print(existing["instance_pool_id"])
    else:
        resp = wc.api_client.do("POST", "/api/2.0/instance-pools/create", body=body)
        pool_id = resp.get("instance_pool_id")
        logger.info("Created pool %s (id=%s)", args.name, pool_id)
        print(pool_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
