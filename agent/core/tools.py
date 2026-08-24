"""
Tool system for the agent
Provides ToolSpec and ToolRouter for managing both built-in and MCP tools
"""

import json
import logging
import warnings
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

from fastmcp import Client
from fastmcp.exceptions import ToolError
from mcp.types import EmbeddedResource, ImageContent, TextContent

from agent.config import MCPServerConfig
from agent.tools.databricks_jobs_tool import (
    DATABRICKS_JOBS_TOOL_SPEC,
    databricks_jobs_handler,
)
from agent.tools.docs_tools import (
    EXPLORE_HF_DOCS_TOOL_SPEC,
    HF_DOCS_FETCH_TOOL_SPEC,
    explore_hf_docs_handler,
    hf_docs_fetch_handler,
)
from agent.tools.github_find_examples import (
    GITHUB_FIND_EXAMPLES_TOOL_SPEC,
    github_find_examples_handler,
)
from agent.tools.github_list_repos import (
    GITHUB_LIST_REPOS_TOOL_SPEC,
    github_list_repos_handler,
)
from agent.tools.github_read_file import (
    GITHUB_READ_FILE_TOOL_SPEC,
    github_read_file_handler,
)
from agent.tools.hf_to_uc_tool import HF_TO_UC_TOOL_SPEC, hf_to_uc_handler
from agent.tools.papers_tool import HF_PAPERS_TOOL_SPEC, hf_papers_handler
from agent.tools.read_skill_tool import READ_SKILL_TOOL_SPEC, read_skill_handler
from agent.tools.experiment_tool import EXPERIMENT_TOOL_SPEC, experiment_handler
from agent.tools.sweep_tool import SWEEP_TOOL_SPEC, sweep_handler
from agent.core.compute_advisor import (
    COMPUTE_ADVICE_TOOL_SPEC,
    compute_advice_handler,
)
from agent.tools.tool_search import (
    TOOL_SEARCH_TOOL_SPEC,
    build_catalog,
    tool_search_handler,
)
from agent.tools.truncation import MODEL_FACING_CHAR_CAP, truncate_output
from agent.tools.critic_tool import CRITIC_TOOL_SPEC, critic_handler
from agent.tools.research_loop_tool import RESEARCH_LOOP_TOOL_SPEC, research_loop_handler
from agent.tools.model_serving_tool import MODEL_SERVING_TOOL_SPEC, model_serving_handler
from agent.tools.web_search_tool import WEB_SEARCH_TOOL_SPEC, web_search_handler
from agent.tools.plan_tool import PLAN_TOOL_SPEC, plan_tool_handler
from agent.tools.repos_tool import REPOS_TOOL_SPEC, repos_handler
from agent.tools.research_tool import RESEARCH_TOOL_SPEC, research_handler
from agent.tools.sandbox_tool import get_sandbox_tools
from agent.tools.uc_dataset_tools import (
    UC_DATASET_TOOL_SPEC,
    uc_inspect_dataset_handler,
)
from agent.tools.uc_model_tools import UC_MODEL_TOOL_SPEC, uc_model_handler
from agent.tools.uc_volume_tools import UC_VOLUME_TOOL_SPEC, uc_volume_handler

# Suppress aiohttp deprecation warning
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module="aiohttp.connector"
)

# MCP tools we deliberately refuse to import even when surfaced. The HF
# server's ``hf_doc_search`` / ``hf_doc_fetch`` / ``hf_whoami`` collide with
# our own docs tools and the Databricks identity surface.
NOT_ALLOWED_TOOL_NAMES = ["hf_doc_search", "hf_doc_fetch", "hf_whoami"]


def convert_mcp_content_to_string(content: list) -> str:
    """
    Convert MCP content blocks to a string format compatible with LLM messages.

    Based on FastMCP documentation, content can be:
    - TextContent: has .text field
    - ImageContent: has .data and .mimeType fields
    - EmbeddedResource: has .resource field with .text or .blob

    Args:
        content: List of MCP content blocks

    Returns:
        String representation of the content suitable for LLM consumption
    """
    if not content:
        return ""

    parts = []
    for item in content:
        if isinstance(item, TextContent):
            # Extract text from TextContent blocks
            parts.append(item.text)
        elif isinstance(item, ImageContent):
            # TODO: Handle images
            # For images, include a description with MIME type
            parts.append(f"[Image: {item.mimeType}]")
        elif isinstance(item, EmbeddedResource):
            # TODO: Handle embedded resources
            # For embedded resources, try to extract text
            resource = item.resource
            if hasattr(resource, "text") and resource.text:
                parts.append(resource.text)
            elif hasattr(resource, "blob") and resource.blob:
                parts.append(
                    f"[Binary data: {resource.mimeType if hasattr(resource, 'mimeType') else 'unknown'}]"
                )
            else:
                parts.append(
                    f"[Resource: {resource.uri if hasattr(resource, 'uri') else 'unknown'}]"
                )
        else:
            # Fallback: try to convert to string
            parts.append(str(item))

    return "\n".join(parts)


@dataclass
class ToolSpec:
    """Tool specification for LLM"""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Optional[Callable[[dict[str, Any]], Awaitable[tuple[str, bool]]]] = None
    # When True the schema is withheld from the LLM request until `tool_search`
    # fetches it. The tool stays callable either way — see RESIDENT_TOOLS.
    defer_loading: bool = False


# Tools whose schemas are always sent. Everything else is deferred behind
# `tool_search`, because sending all 25 builtin schemas costs ~52 KB (~13k tokens) on
# every request while a typical turn touches only a few.
#
# The bar for being resident: needed in the first few turns of almost any task, or
# needed to recover when something goes wrong. Discovery/research and
# specialised-subsystem tools do not clear it.
RESIDENT_TOOLS: frozenset[str] = frozenset(
    {
        # Filesystem / execution — the core loop.
        "bash",
        "read",
        "write",
        "edit",
        "sandbox_create",
        # Task structure.
        "plan_tool",
        # Databricks essentials: reaching data and submitting work.
        "uc_volume",
        "databricks_jobs",
        # Domain playbooks; cheap and steers everything after it.
        "read_skill",
        # Tiny schema, and must be consulted BEFORE the first job submission.
        "compute_advice",
        # Always resident by construction.
        "tool_search",
    }
)


def _normalize_tool_result(tool_name: str, result: Any) -> tuple[str, bool]:
    """Coerce whatever a handler returned into the ``(output, success)`` contract.

    Most handlers return ``tuple[str, bool]``, but six — read_skill, critic, experiment,
    sweep, research_loop, model_serving — return a ``ToolResult`` dict instead. Python
    unpacks a 2-key dict into its *keys*, so ``out, ok = handler(...)`` silently yielded
    ``("formatted", "isError")``: the model received the literal string "formatted" as
    the tool's entire output, with a truthy "isError" as the success flag. That killed
    the skills system, the critic, the experiment ledger, and sweeps at the agent
    boundary while looking like success in the logs.

    Normalising here rather than in each tool fixes all five at once and stops a future
    handler from reintroducing it, since the dict shape is a reasonable thing to write.
    """
    if isinstance(result, tuple) and len(result) == 2:
        output, ok = result
        return output if isinstance(output, str) else str(output), bool(ok)

    if isinstance(result, dict):
        output = result.get("formatted")
        if output is None:
            output = json.dumps(result, default=str)
        logger.debug("Normalized dict result from %s into the (output, ok) contract", tool_name)
        return str(output), not bool(result.get("isError", False))

    logger.warning(
        "Tool %s returned %s, expected tuple[str, bool] or a ToolResult dict",
        tool_name, type(result).__name__,
    )
    return str(result), True


def sanitize_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Drop `$defs`/`definitions` entries that nothing `$ref`s.

    Mirrors the definition-pruning stage of codex's schema pipeline
    (`codex-rs/tools/src/json_schema.rs`). MCP servers in particular ship whole shared
    definition blocks per tool, and unreferenced ones are pure token cost.
    """
    if not isinstance(schema, dict):
        return schema

    def refs(node: Any) -> set[str]:
        found: set[str] = set()
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str):
                found.add(ref.rsplit("/", 1)[-1])
            for key, value in node.items():
                if key not in ("$defs", "definitions"):
                    found |= refs(value)
        elif isinstance(node, list):
            for item in node:
                found |= refs(item)
        return found

    out = dict(schema)
    for bucket in ("$defs", "definitions"):
        defs = out.get(bucket)
        if not isinstance(defs, dict):
            continue
        # Iterate to a fixed point: a kept definition may reference another.
        keep = refs({k: v for k, v in out.items() if k != bucket})
        while True:
            grown = set(keep)
            for name in keep:
                if name in defs:
                    grown |= refs(defs[name])
            if grown == keep:
                break
            keep = grown
        pruned = {k: v for k, v in defs.items() if k in keep}
        if pruned:
            out[bucket] = pruned
        else:
            out.pop(bucket)
    return out


class ToolRouter:
    """
    Routes tool calls to appropriate handlers.
    Based on codex-rs/core/src/tools/router.rs
    """

    def __init__(self, mcp_servers: dict[str, MCPServerConfig], hf_token: str | None = None, local_mode: bool = False):
        self.tools: dict[str, ToolSpec] = {}
        self.mcp_servers: dict[str, dict[str, Any]] = {}
        # Names whose schemas go on the wire. Deferred tools join this as
        # `tool_search` fetches them, or on first direct call.
        self.active_tools: set[str] = set()

        for tool in create_builtin_tools(local_mode=local_mode):
            self.register_tool(tool)
        self._install_tool_search()

        self.mcp_client: Client | None = None
        if mcp_servers:
            mcp_servers_payload = {}
            for name, server in mcp_servers.items():
                data = server.model_dump()
                if hf_token:
                    data.setdefault("headers", {})["Authorization"] = f"Bearer {hf_token}"
                mcp_servers_payload[name] = data
            self.mcp_client = Client({"mcpServers": mcp_servers_payload})
        self._mcp_initialized = False

    def register_tool(self, tool: ToolSpec) -> None:
        tool.parameters = sanitize_schema(tool.parameters)
        self.tools[tool.name] = tool
        if not tool.defer_loading:
            self.active_tools.add(tool.name)

    def _install_tool_search(self) -> None:
        """Register `tool_search`, with the deferred-tool catalog in its description.

        The catalog is what makes deferral safe: the model still sees every deferred
        tool's name and one-line summary, so it can tell what exists and ask for it.
        Without it, deferral would just hide capability.
        """
        deferred = self.deferred_tools()
        if not deferred:
            return
        self.register_tool(
            ToolSpec(
                name=TOOL_SEARCH_TOOL_SPEC["name"],
                description=TOOL_SEARCH_TOOL_SPEC["description"] + build_catalog(deferred),
                parameters=TOOL_SEARCH_TOOL_SPEC["parameters"],
                handler=tool_search_handler,
            )
        )

    def get_tool_specs_by_name(self, names: set[str]) -> list[dict[str, Any]]:
        """Specs for named tools, ignoring whether they're advertised to the main loop.

        Deferral exists to keep the *main* loop's per-request schema budget down. A
        sub-agent that is handed an explicit, curated tool list (see
        ``research_tool.RESEARCH_TOOL_NAMES``) has its own context and should get every
        tool on that list — filtering it through the advertised set would silently strip
        it down to whatever the main loop happened to have loaded.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for name, tool in self.tools.items()
            if name in names
        ]

    def deferred_tools(self) -> dict[str, str]:
        """Registered-but-not-advertised tools, as {name: description}."""
        return {
            name: spec.description
            for name, spec in self.tools.items()
            if name not in self.active_tools
        }

    def promote(self, names: list[str]) -> list[str]:
        """Move deferred tools into the advertised set. Returns those actually promoted."""
        promoted = [n for n in names if n in self.tools and n not in self.active_tools]
        self.active_tools.update(promoted)
        if promoted:
            logger.info("Promoted deferred tools: %s", ", ".join(promoted))
        return promoted

    async def register_mcp_tools(self) -> None:
        tools = await self.mcp_client.list_tools()
        registered_names = []
        skipped_count = 0
        for tool in tools:
            if tool.name in NOT_ALLOWED_TOOL_NAMES:
                skipped_count += 1
                continue
            registered_names.append(tool.name)
            self.register_tool(
                ToolSpec(
                    name=tool.name,
                    description=tool.description,
                    parameters=tool.inputSchema,
                    handler=None,
                )
            )
        logger.info(
            f"Loaded {len(registered_names)} MCP tools: {', '.join(registered_names)} ({skipped_count} disabled)"
        )

    async def register_openapi_tool(self) -> None:
        """Register the OpenAPI search tool (requires async initialization)"""
        from agent.tools.docs_tools import (
            _get_api_search_tool_spec,
            search_openapi_handler,
        )

        try:
            openapi_spec = await _get_api_search_tool_spec()
            self.register_tool(
                ToolSpec(
                    name=openapi_spec["name"],
                    description=openapi_spec["description"],
                    parameters=openapi_spec["parameters"],
                    handler=search_openapi_handler,
                )
            )
            logger.info(f"Loaded OpenAPI search tool: {openapi_spec['name']}")
        except Exception as e:
            logger.warning("Failed to load OpenAPI search tool: %s", e)

    def get_tool_specs_for_llm(self) -> list[dict[str, Any]]:
        """Tool specifications in OpenAI format, limited to the advertised set."""
        specs = []
        for tool in self.tools.values():
            if tool.name not in self.active_tools:
                continue
            specs.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
            )
        return specs

    async def __aenter__(self) -> "ToolRouter":
        if self.mcp_client is not None:
            try:
                await self.mcp_client.__aenter__()
                await self.mcp_client.initialize()
                await self.register_mcp_tools()
                self._mcp_initialized = True
            except Exception as e:
                logger.warning("MCP connection failed, continuing without MCP tools: %s", e)
                self.mcp_client = None

        await self.register_openapi_tool()

        total_tools = len(self.tools)
        logger.info(f"Agent ready with {total_tools} tools total")

        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self.mcp_client is not None:
            await self.mcp_client.__aexit__(exc_type, exc, tb)
            self._mcp_initialized = False

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        session: Any = None,
        tool_call_id: str | None = None,
    ) -> tuple[str, bool]:
        """Call a tool, then apply the global output cap before the result enters history.

        The cap is a backstop, not the primary limit: tools that already trim themselves
        (bash/read at 25k, research at 8k) stay under it untouched. What it catches is
        everything that never set a limit — MCP tools especially, whose output volume we
        do not control.
        """
        output, ok = _normalize_tool_result(
            tool_name, await self._dispatch_tool(tool_name, arguments, session, tool_call_id)
        )
        if isinstance(output, str) and len(output) > MODEL_FACING_CHAR_CAP:
            original = len(output)
            output = truncate_output(output, prefix=f"{tool_name}_output_")
            logger.info(
                "Capped %s output for the model: %d -> %d chars",
                tool_name, original, len(output),
            )
        return output, ok

    async def _dispatch_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        session: Any = None,
        tool_call_id: str | None = None,
    ) -> tuple[str, bool]:
        """
        Call a tool and return (output_string, success_bool).

        For MCP tools, converts the CallToolResult content blocks to a string.
        For built-in tools, calls their handler directly.
        """
        # A deferred tool called directly still runs — deferral withholds the schema,
        # never the capability. Promote it so the next request carries its schema.
        if tool_name in self.tools and tool_name not in self.active_tools:
            self.promote([tool_name])

        # Check if this is a built-in tool with a handler
        tool = self.tools.get(tool_name)
        if tool and tool.handler:
            import inspect

            # Check if handler accepts session argument
            sig = inspect.signature(tool.handler)
            if "session" in sig.parameters:
                # Check if handler also accepts tool_call_id parameter
                if "tool_call_id" in sig.parameters:
                    return await tool.handler(
                        arguments, session=session, tool_call_id=tool_call_id
                    )
                return await tool.handler(arguments, session=session)
            return await tool.handler(arguments)

        # Otherwise, use MCP client
        if self._mcp_initialized:
            try:
                result = await self.mcp_client.call_tool(tool_name, arguments)
                output = convert_mcp_content_to_string(result.content)
                return output, not result.is_error
            except ToolError as e:
                # Catch MCP tool errors and return them to the agent
                error_msg = f"Tool error: {str(e)}"
                return error_msg, False

        return "MCP client not initialized", False


# ============================================================================
# BUILT-IN TOOL HANDLERS
# ============================================================================


def create_builtin_tools(local_mode: bool = False) -> list[ToolSpec]:
    """Create built-in tool specifications"""
    # in order of importance
    tools = [
        # Research sub-agent (delegates to read-only tools in independent context)
        ToolSpec(
            name=RESEARCH_TOOL_SPEC["name"],
            description=RESEARCH_TOOL_SPEC["description"],
            parameters=RESEARCH_TOOL_SPEC["parameters"],
            handler=research_handler,
        ),
        # Documentation search tools
        ToolSpec(
            name=EXPLORE_HF_DOCS_TOOL_SPEC["name"],
            description=EXPLORE_HF_DOCS_TOOL_SPEC["description"],
            parameters=EXPLORE_HF_DOCS_TOOL_SPEC["parameters"],
            handler=explore_hf_docs_handler,
        ),
        ToolSpec(
            name=HF_DOCS_FETCH_TOOL_SPEC["name"],
            description=HF_DOCS_FETCH_TOOL_SPEC["description"],
            parameters=HF_DOCS_FETCH_TOOL_SPEC["parameters"],
            handler=hf_docs_fetch_handler,
        ),
        # Paper discovery and reading
        ToolSpec(
            name=HF_PAPERS_TOOL_SPEC["name"],
            description=HF_PAPERS_TOOL_SPEC["description"],
            parameters=HF_PAPERS_TOOL_SPEC["parameters"],
            handler=hf_papers_handler,
        ),
        # Current-web lookup (DuckDuckGo HTML; no API key)
        ToolSpec(
            name=WEB_SEARCH_TOOL_SPEC["name"],
            description=WEB_SEARCH_TOOL_SPEC["description"],
            parameters=WEB_SEARCH_TOOL_SPEC["parameters"],
            handler=web_search_handler,
        ),
        # Domain-specific playbooks (Kaggle tabular, CV, NLP, ...) — read on demand.
        ToolSpec(
            name=READ_SKILL_TOOL_SPEC["name"],
            description=READ_SKILL_TOOL_SPEC["description"],
            parameters=READ_SKILL_TOOL_SPEC["parameters"],
            handler=read_skill_handler,
        ),
        # Which compute a training job belongs on (CPU vs GPU) — advice only.
        ToolSpec(
            name=COMPUTE_ADVICE_TOOL_SPEC["name"],
            description=COMPUTE_ADVICE_TOOL_SPEC["description"],
            parameters=COMPUTE_ADVICE_TOOL_SPEC["parameters"],
            handler=compute_advice_handler,
        ),
        # Planning and job management tools
        ToolSpec(
            name=PLAN_TOOL_SPEC["name"],
            description=PLAN_TOOL_SPEC["description"],
            parameters=PLAN_TOOL_SPEC["parameters"],
            handler=plan_tool_handler,
        ),
        ToolSpec(
            name=DATABRICKS_JOBS_TOOL_SPEC["name"],
            description=DATABRICKS_JOBS_TOOL_SPEC["description"],
            parameters=DATABRICKS_JOBS_TOOL_SPEC["parameters"],
            handler=databricks_jobs_handler,
        ),
        ToolSpec(
            name=UC_VOLUME_TOOL_SPEC["name"],
            description=UC_VOLUME_TOOL_SPEC["description"],
            parameters=UC_VOLUME_TOOL_SPEC["parameters"],
            handler=uc_volume_handler,
        ),
        ToolSpec(
            name=UC_DATASET_TOOL_SPEC["name"],
            description=UC_DATASET_TOOL_SPEC["description"],
            parameters=UC_DATASET_TOOL_SPEC["parameters"],
            handler=uc_inspect_dataset_handler,
        ),
        ToolSpec(
            name=UC_MODEL_TOOL_SPEC["name"],
            description=UC_MODEL_TOOL_SPEC["description"],
            parameters=UC_MODEL_TOOL_SPEC["parameters"],
            handler=uc_model_handler,
        ),
        ToolSpec(
            name=HF_TO_UC_TOOL_SPEC["name"],
            description=HF_TO_UC_TOOL_SPEC["description"],
            parameters=HF_TO_UC_TOOL_SPEC["parameters"],
            handler=hf_to_uc_handler,
        ),
        ToolSpec(
            name=REPOS_TOOL_SPEC["name"],
            description=REPOS_TOOL_SPEC["description"],
            parameters=REPOS_TOOL_SPEC["parameters"],
            handler=repos_handler,
        ),
        ToolSpec(
            name=GITHUB_FIND_EXAMPLES_TOOL_SPEC["name"],
            description=GITHUB_FIND_EXAMPLES_TOOL_SPEC["description"],
            parameters=GITHUB_FIND_EXAMPLES_TOOL_SPEC["parameters"],
            handler=github_find_examples_handler,
        ),
        ToolSpec(
            name=GITHUB_LIST_REPOS_TOOL_SPEC["name"],
            description=GITHUB_LIST_REPOS_TOOL_SPEC["description"],
            parameters=GITHUB_LIST_REPOS_TOOL_SPEC["parameters"],
            handler=github_list_repos_handler,
        ),
        ToolSpec(
            name=GITHUB_READ_FILE_TOOL_SPEC["name"],
            description=GITHUB_READ_FILE_TOOL_SPEC["description"],
            parameters=GITHUB_READ_FILE_TOOL_SPEC["parameters"],
            handler=github_read_file_handler,
        ),
        # Experiment ledger — the agent's durable record of what it tried + scored.
        ToolSpec(
            name=EXPERIMENT_TOOL_SPEC["name"],
            description=EXPERIMENT_TOOL_SPEC["description"],
            parameters=EXPERIMENT_TOOL_SPEC["parameters"],
            handler=experiment_handler,
        ),
        # Parallel hypothesis sweep — fan out N training-job variants, rank, dedup.
        ToolSpec(
            name=SWEEP_TOOL_SPEC["name"],
            description=SWEEP_TOOL_SPEC["description"],
            parameters=SWEEP_TOOL_SPEC["parameters"],
            handler=sweep_handler,
        ),
        # Result critic — audit for overfit / target confusion / leakage before a win.
        ToolSpec(
            name=CRITIC_TOOL_SPEC["name"],
            description=CRITIC_TOOL_SPEC["description"],
            parameters=CRITIC_TOOL_SPEC["parameters"],
            handler=critic_handler,
        ),
        # Autonomous research loop — burns down a hypothesis pool hands-off.
        ToolSpec(
            name=RESEARCH_LOOP_TOOL_SPEC["name"],
            description=RESEARCH_LOOP_TOOL_SPEC["description"],
            parameters=RESEARCH_LOOP_TOOL_SPEC["parameters"],
            handler=research_loop_handler,
        ),
        # Custom LLM Serving — plan/build/deploy/query model serving endpoints.
        ToolSpec(
            name=MODEL_SERVING_TOOL_SPEC["name"],
            description=MODEL_SERVING_TOOL_SPEC["description"],
            parameters=MODEL_SERVING_TOOL_SPEC["parameters"],
            handler=model_serving_handler,
        ),
    ]

    # Sandbox or local tools (highest priority)
    if local_mode:
        from agent.tools.local_tools import get_local_tools
        tools = get_local_tools() + tools
    else:
        tools = get_sandbox_tools() + tools

    # Defer everything outside the resident set. MCP tools are left resident on
    # purpose: the operator configured them explicitly, so hiding them would be
    # surprising, and there are usually few of them.
    for tool in tools:
        tool.defer_loading = tool.name not in RESIDENT_TOOLS

    resident = [t.name for t in tools if not t.defer_loading]
    deferred = [t.name for t in tools if t.defer_loading]
    logger.info(
        "Loaded %d built-in tools: %d resident (%s), %d deferred behind tool_search (%s)",
        len(tools), len(resident), ", ".join(resident), len(deferred), ", ".join(deferred),
    )

    return tools
