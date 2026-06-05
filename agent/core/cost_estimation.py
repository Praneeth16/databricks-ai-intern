"""Pre-call cost estimates for billable agent actions (issue #16).

Two cost flows live in the Databricks deployment:

  * **Static price catalog** (this module) — per-token FMAPI rates +
    per-hour compute rates. Used as a **pre-call estimate** for the YOLO
    auto-approval gate (#17) and to accumulate ``Session.total_cost_usd``
    in near-real-time. Compile-time constants; refresh by editing the
    dicts below when Databricks publishes new rates.
  * **``system.serving.endpoint_usage`** + **``system.billing.usage``**
    (queried by ``Session.reconcile_actual_cost`` below) — actual DBUs +
    USD consumed. ~15 min lag, so it can't gate the YOLO call site, but
    we run it in the background to keep ``Session.actual_cost_usd``
    honest and to detect drift in the static catalog.

Failure mode contract: every estimator returns a ``CostEstimate`` with
``estimated_cost_usd=None`` when it cannot price the call confidently.
``None`` means "billable but unknown" — the policy at #17 must NOT
auto-approve in that case (fall back to human approval).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ── Pricing catalogs ────────────────────────────────────────────────────


# FMAPI per-million-token rates. Per the Databricks AI Gateway listed
# rates as of 2026-05; refresh when the gateway publishes new prices.
# Keys are the served-model name segment after ``databricks/`` so the
# resolver can take ``databricks/databricks-claude-opus-4-7`` and look
# up ``databricks-claude-opus-4-7`` directly.
FMAPI_PRICE_USD_PER_MTOK: dict[str, dict[str, float]] = {
    "databricks-claude-opus-4-7":          {"input": 15.0, "output": 75.0},
    "databricks-claude-opus-4-6":          {"input": 15.0, "output": 75.0},
    "databricks-claude-opus-4":            {"input": 15.0, "output": 75.0},
    "databricks-claude-sonnet-4":          {"input":  3.0, "output": 15.0},
    "databricks-claude-3-7-sonnet":        {"input":  3.0, "output": 15.0},
    "databricks-meta-llama-3-3-70b-instruct": {"input": 1.00, "output": 1.00},
    "databricks-meta-llama-3-1-70b-instruct": {"input": 1.00, "output": 1.00},
    "databricks-dbrx-instruct":            {"input": 0.75, "output": 2.25},
    "databricks-gpt-oss-120b":             {"input": 0.50, "output": 1.50},
}

# Per-hour compute rates for the node types ``databricks_jobs`` and the
# sandbox tool most commonly select. Numbers are workspace list price for
# AWS regions; on-cluster usage may carry an instance-pool discount.
DATABRICKS_NODE_PRICE_USD_PER_HOUR: dict[str, float] = {
    # AI Runtime serverless GPU (env_version=4)
    "GPU_1xA10":    3.10,    # 1× NVIDIA A10G
    "GPU_2xA10":    6.20,
    "GPU_4xA10":   12.40,
    "GPU_1xA100": 11.40,
    "GPU_2xA100": 22.80,
    "GPU_1xH100": 22.80,
    "GPU_2xH100": 45.60,
    # Classic clusters (most common GPU flavors)
    "g5.xlarge":   1.20,
    "g5.2xlarge":  2.40,
    "g5.4xlarge":  4.80,
    "g5.12xlarge":14.40,
    "p4d.24xlarge": 32.80,
    "p5.48xlarge": 80.00,
    # Classic CPU clusters
    "i3.xlarge":   0.35,
    "i3.2xlarge":  0.70,
    "m5d.xlarge":  0.30,
    "m5d.large":   0.15,
    "m5d.2xlarge": 0.60,
    "m5d.4xlarge": 1.20,
    # Serverless CPU (post-paid hourly approximate)
    "serverless":  0.55,
}


# Default budgets when the caller didn't pin a timeout / duration.
DEFAULT_JOB_TIMEOUT_HOURS = 0.5
DEFAULT_SANDBOX_RESERVATION_HOURS = 1.0

# Tool names that NEVER auto-approve regardless of YOLO policy —
# destructive or registry-mutating ops where a budget gate alone isn't
# safety enough. ``databricks_jobs`` is deliberately NOT here: the budget
# gate is exactly what YOLO is supposed to provide for it. Consumed by
# #17's ``should_auto_approve`` predicate.
NEVER_AUTO_APPROVE: frozenset[str] = frozenset({
    "uc_volume_rm",
    "uc_model_set_alias",
    "uc_model_delete_alias",
})

_DURATION_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([smhd]?)\s*$", re.IGNORECASE)


# ── dataclass ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CostEstimate:
    """One pre-call cost estimate.

    ``estimated_cost_usd=None`` signals "billable but unable to price";
    the YOLO policy MUST treat that as "fall back to human approval"
    rather than approving silently.
    """

    estimated_cost_usd: Optional[float]
    billable: bool
    block_reason: Optional[str] = None
    label: Optional[str] = None


# ── duration parsing ───────────────────────────────────────────────────


def parse_duration_hours(
    value: Any, *, default_hours: float = DEFAULT_JOB_TIMEOUT_HOURS,
) -> Optional[float]:
    """Parse a Databricks-style timeout into hours.

    Accepts the shapes the agent commonly uses:
      * ``int`` / ``float`` → seconds (matches ``timeout_seconds`` on
        ``runs/submit`` and Mosaic AI's job spec).
      * ``str`` with ``s|m|h|d`` suffix (e.g. ``"30m"``, ``"2h"``).
      * Empty / None → ``default_hours``.

    Returns ``None`` for unparseable input so the caller can surface a
    block_reason instead of silently auto-approving with a wrong budget.
    """
    if value is None or value == "":
        return default_hours
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        seconds = float(value)
        return seconds / 3600 if seconds > 0 else None
    if not isinstance(value, str):
        return None
    match = _DURATION_RE.match(value)
    if not match:
        return None
    amount = float(match.group(1))
    unit = match.group(2).lower() or "s"
    if amount <= 0:
        return None
    return {
        "s": amount / 3600,
        "m": amount / 60,
        "h": amount,
        "d": amount * 24,
    }[unit]


# ── per-call estimators ────────────────────────────────────────────────


def _strip_model_prefix(model_name: str) -> str:
    """Return the served-model name segment after ``databricks/``."""
    if model_name.startswith("databricks/"):
        return model_name.split("/", 1)[1]
    return model_name


def estimate_llm_cost(
    model_name: str, prompt_tokens: int, completion_tokens: int,
) -> CostEstimate:
    """Pre-call estimate for an FMAPI LLM call.

    Pure function — callers extract token counts from the response usage
    block (post-call) and accumulate the result on the session.
    Returning a ``CostEstimate`` keeps the surface symmetric with the
    tool estimators below.
    """
    endpoint = _strip_model_prefix(model_name)
    rates = FMAPI_PRICE_USD_PER_MTOK.get(endpoint)
    if not rates:
        return CostEstimate(
            estimated_cost_usd=None,
            billable=True,
            block_reason=f"No FMAPI price catalog entry for {endpoint!r}.",
            label=endpoint,
        )
    input_cost = (prompt_tokens / 1_000_000) * rates["input"]
    output_cost = (completion_tokens / 1_000_000) * rates["output"]
    return CostEstimate(
        estimated_cost_usd=round(input_cost + output_cost, 6),
        billable=(input_cost + output_cost) > 0,
        label=endpoint,
    )


def estimate_jobs_cost(args: dict[str, Any]) -> CostEstimate:
    """Estimate a single ``databricks_jobs`` submission's cost.

    Args dict mirrors the tool's input. Picks duration from
    ``timeout_seconds`` (preferred) or ``timeout`` (string with unit).
    Picks node from explicit ``node_type_id`` first, falls back to
    ``hardware_flavor`` then a serverless-default.
    """
    timeout_hours = parse_duration_hours(
        args.get("timeout_seconds") or args.get("timeout"),
    )
    if timeout_hours is None:
        return CostEstimate(
            estimated_cost_usd=None,
            billable=True,
            block_reason=(
                f"Could not parse jobs timeout: "
                f"{args.get('timeout_seconds') or args.get('timeout')!r}."
            ),
        )

    node = (
        args.get("node_type_id")
        or args.get("hardware_flavor")
        or ("serverless" if args.get("kind") == "serverless" else "GPU_1xA10")
    )
    node = str(node)
    rate = DATABRICKS_NODE_PRICE_USD_PER_HOUR.get(node)
    if rate is None:
        return CostEstimate(
            estimated_cost_usd=None,
            billable=True,
            block_reason=f"No price catalog entry for node type {node!r}.",
            label=node,
        )
    return CostEstimate(
        estimated_cost_usd=round(rate * timeout_hours, 4),
        billable=rate > 0,
        label=node,
    )


def estimate_sandbox_cost(
    args: dict[str, Any], *, session: Any = None,
) -> CostEstimate:
    """Estimate a ``sandbox_create`` call's cost.

    Sandbox cost is treated as ``0.0`` when the session already has one
    attached (re-use, no new allocation). Otherwise priced as the
    reserved-hours rate of the selected hardware.
    """
    if session is not None and getattr(session, "sandbox", None):
        return CostEstimate(estimated_cost_usd=0.0, billable=False, label="existing")
    hardware = str(args.get("hardware") or args.get("node_type_id") or "GPU_1xA10")
    rate = DATABRICKS_NODE_PRICE_USD_PER_HOUR.get(hardware)
    if rate is None:
        return CostEstimate(
            estimated_cost_usd=None,
            billable=True,
            block_reason=f"No price catalog entry for sandbox hardware {hardware!r}.",
            label=hardware,
        )
    return CostEstimate(
        estimated_cost_usd=round(rate * DEFAULT_SANDBOX_RESERVATION_HOURS, 4),
        billable=rate > 0,
        label=hardware,
    )


def estimate_tool_cost(
    tool_name: str, args: dict[str, Any], *, session: Any = None,
) -> CostEstimate:
    """Top-level dispatcher mirroring the upstream signature.

    Free / read-only tools (uc_volume_read, uc_inspect_dataset, plan_tool,
    docs_*, github_*) return ``CostEstimate(0.0, False)`` so the YOLO
    policy auto-approves them without further checks.
    """
    if tool_name == "databricks_jobs":
        return estimate_jobs_cost(args)
    if tool_name in {"sandbox_create", "sandbox"}:
        return estimate_sandbox_cost(args, session=session)
    return CostEstimate(estimated_cost_usd=0.0, billable=False)


# ── post-hoc reconciliation against system tables ──────────────────────


def query_actual_usd_for_user(
    settings: Any, user_email: str, *, lookback_minutes: int = 60,
) -> Optional[float]:
    """Sum the user's actual USD spend over the last hour via system
    tables. Returns ``None`` when the query fails (no warehouse, no
    permissions, table missing on the workspace).

    Used by ``Session.reconcile_actual_cost`` so we can periodically
    rebase ``Session.actual_cost_usd`` against billing reality. Latency
    of system.billing.usage is ~15 min so we never use this on the YOLO
    hot path — it's an after-the-fact honesty check on the catalog.
    """
    try:
        from agent.core import db_client

        conn = db_client.get_sql_connection(settings)
    except Exception as e:
        logger.debug("query_actual_usd_for_user: no SQL connection (%s)", e)
        return None

    sql = """
        SELECT COALESCE(SUM(u.usage_quantity * p.pricing.default), 0.0) AS usd
          FROM system.billing.usage u
          JOIN system.billing.list_prices p
            ON p.sku_name = u.sku_name
           AND u.usage_start_time BETWEEN p.price_start_time
                                    AND COALESCE(p.price_end_time, current_timestamp())
         WHERE u.identity_metadata.run_as = %(user)s
           AND u.usage_start_time >= current_timestamp() - INTERVAL %(mins)s MINUTES
    """
    try:
        with conn.cursor() as cur:
            cur.execute(sql, {"user": user_email, "mins": lookback_minutes})
            row = cur.fetchone()
        return float(row[0]) if row and row[0] is not None else 0.0
    except Exception as e:
        logger.debug("query_actual_usd_for_user query failed: %s", e)
        return None
