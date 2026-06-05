"""Unity Catalog dataset/table inspection tool.

Replaces ``hf_inspect_dataset`` for UC-resident tables. Issues read-only SQL
through the configured warehouse. Mutating SQL is rejected up front — write
paths go through training jobs / explicit ETL.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any, Dict, Optional

from agent.core import db_client
from agent.tools.types import ToolResult

logger = logging.getLogger(__name__)


_FORBIDDEN_SQL = re.compile(
    r"\b(INSERT|UPDATE|DELETE|MERGE|DROP|TRUNCATE|ALTER|CREATE|GRANT|REVOKE|COPY)\b",
    re.IGNORECASE,
)


def _ok(formatted: str, n: int = 1) -> ToolResult:
    return {"formatted": formatted, "totalResults": n, "resultsShared": n}


def _err(msg: str) -> ToolResult:
    return {"formatted": f"Error: {msg}", "totalResults": 0, "resultsShared": 0, "isError": True}


def _validate_table(name: str) -> Optional[str]:
    if not isinstance(name, str) or not name:
        return "table is required"
    if not re.match(r"^[A-Za-z_][\w]*\.[A-Za-z_][\w]*\.[A-Za-z_][\w]*$", name):
        return f"table must be fully-qualified <catalog>.<schema>.<table> (got {name!r})"
    return None


def _markdown_rows(cols: list[str], rows: list[tuple]) -> str:
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    out = [header, sep]
    for r in rows:
        out.append("| " + " | ".join(_cell(v) for v in r) + " |")
    return "\n".join(out)


def _cell(v: Any) -> str:
    if v is None:
        return ""
    s = str(v)
    return s.replace("|", "\\|").replace("\n", " ")[:80]


class UCDatasetTool:
    def __init__(self, wc, settings: db_client.DatabricksSettings,
                 user_token: str | None = None):
        self.wc = wc
        self.settings = settings
        # OBO token for SQL warehouse auth. When set, audit log names the
        # invoking user instead of the App SP.
        self.user_token = user_token

    async def execute(self, args: Dict[str, Any]) -> ToolResult:
        op = (args.get("operation") or "").lower().strip()
        try:
            if op == "list_tables":
                return await self._list_tables(args)
            if op == "describe":
                return await self._describe(args)
            if op == "sample":
                return await self._sample(args)
            if op == "query":
                return await self._query(args)
            return _err(f"Unknown operation {op!r}. Use list_tables | describe | sample | query.")
        except Exception as e:
            logger.exception("uc_dataset %s failed", op)
            return _err(f"{op} failed: {e}")

    async def _execute_sql(self, sql: str) -> tuple[list[str], list[tuple]]:
        def _run():
            conn = db_client.get_sql_connection(self.settings, user_token=self.user_token)
            try:
                cur = conn.cursor()
                cur.execute(sql)
                cols = [d[0] for d in cur.description] if cur.description else []
                rows = cur.fetchall()
                return cols, rows
            finally:
                conn.close()

        return await asyncio.to_thread(_run)

    async def _list_tables(self, args: Dict[str, Any]) -> ToolResult:
        catalog = args.get("catalog") or self.settings.uc_catalog
        schema = args.get("schema") or self.settings.uc_schema
        sql = f"SHOW TABLES IN `{catalog}`.`{schema}`"
        cols, rows = await self._execute_sql(sql)
        if not rows:
            return _ok(f"No tables in {catalog}.{schema}.")
        body = _markdown_rows(cols, rows[:200])
        return _ok(f"**Tables in {catalog}.{schema}** ({len(rows)} total):\n\n{body}", n=len(rows))

    async def _describe(self, args: Dict[str, Any]) -> ToolResult:
        table = args.get("table")
        err = _validate_table(table)
        if err:
            return _err(err)
        cols, rows = await self._execute_sql(f"DESCRIBE TABLE EXTENDED `{table.split('.')[0]}`.`{table.split('.')[1]}`.`{table.split('.')[2]}`")
        body = _markdown_rows(cols, rows)
        # Row count (best-effort).
        try:
            _, count_rows = await self._execute_sql(f"SELECT COUNT(*) AS n FROM {table}")
            n = count_rows[0][0] if count_rows else "?"
        except Exception as e:
            n = f"(count failed: {e})"
        return _ok(f"**{table}** — {n} rows\n\n{body}", n=1)

    async def _sample(self, args: Dict[str, Any]) -> ToolResult:
        table = args.get("table")
        err = _validate_table(table)
        if err:
            return _err(err)
        limit = max(1, min(int(args.get("limit", 10)), 100))
        cols, rows = await self._execute_sql(f"SELECT * FROM {table} LIMIT {limit}")
        if not rows:
            return _ok(f"{table} is empty.")
        body = _markdown_rows(cols, rows)
        return _ok(f"**Sample from {table}** ({len(rows)} rows):\n\n{body}", n=len(rows))

    async def _query(self, args: Dict[str, Any]) -> ToolResult:
        sql = args.get("sql") or ""
        if not isinstance(sql, str) or not sql.strip():
            return _err("sql is required")
        if _FORBIDDEN_SQL.search(sql):
            return _err("Only read-only SQL allowed (no INSERT/UPDATE/DELETE/MERGE/DROP/CREATE/ALTER).")
        cols, rows = await self._execute_sql(sql)
        if not rows:
            return _ok("Query returned 0 rows.", n=0)
        body = _markdown_rows(cols, rows[:200])
        return _ok(f"**Query result** ({len(rows)} rows; first {min(len(rows),200)} shown):\n\n{body}", n=len(rows))


UC_DATASET_TOOL_SPEC = {
    "name": "uc_inspect_dataset",
    "description": (
        "Inspect Unity Catalog tables. Use this BEFORE submitting training jobs to validate "
        "dataset format, columns, and row counts.\n\n"
        "Operations:\n"
        "- list_tables: enumerate tables in <catalog>.<schema> (defaults to ml_intern.agent)\n"
        "- describe: column types, partitioning, location, properties + row count for a fully-qualified table\n"
        "- sample: SELECT * LIMIT <n> for quick eyeballing\n"
        "- query: arbitrary read-only SELECT (write SQL is rejected).\n\n"
        "Tables are referenced as <catalog>.<schema>.<table> (three-level UC names)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": ["list_tables", "describe", "sample", "query"]},
            "catalog": {"type": "string", "description": "(list_tables) Override default catalog."},
            "schema": {"type": "string", "description": "(list_tables) Override default schema."},
            "table": {"type": "string", "description": "(describe, sample) Three-level name <catalog>.<schema>.<table>."},
            "limit": {"type": "integer", "description": "(sample) Default 10, max 100."},
            "sql": {"type": "string", "description": "(query) Read-only SELECT statement."},
        },
        "required": ["operation"],
    },
}


async def uc_inspect_dataset_handler(arguments: Dict[str, Any], session: Any = None,
                                     tool_call_id: str | None = None) -> tuple[str, bool]:
    try:
        cfg = session.config if session and getattr(session, "config", None) else _load_default_config()
        settings = db_client.resolve_settings(cfg)
        token = getattr(session, "databricks_user_token", None) if session else None
        if token and settings.host:
            wc = db_client.get_workspace_client_for_user(token, settings.host)
        else:
            wc = db_client.get_workspace_client(settings)
        tool = UCDatasetTool(wc=wc, settings=settings, user_token=token)
        result = await tool.execute(arguments)
        return result["formatted"], not result.get("isError", False)
    except Exception as e:
        logger.exception("uc_inspect_dataset handler crashed")
        return f"Error: {e}", False


def _load_default_config():
    from agent.config import load_config
    cfg_path = os.environ.get(
        "ML_INTERN_CONFIG_PATH",
        os.path.join(os.path.dirname(__file__), "..", "..", "configs", "main_agent_config.json"),
    )
    return load_config(cfg_path)
