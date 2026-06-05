"""
Context management for conversation history
"""

import logging
import os
import zoneinfo
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Template
from litellm import Message, acompletion

from agent.core.prompt_caching import with_prompt_caching

logger = logging.getLogger(__name__)

_HF_WHOAMI_URL = "https://huggingface.co/api/whoami-v2"
_HF_WHOAMI_TIMEOUT = 5  # seconds


def _get_hf_username(hf_token: str | None = None) -> str:
    """Return the HF username for the given token.

    Uses subprocess + curl to avoid Python HTTP client IPv6 issues that
    cause 40+ second hangs (httpx/urllib try IPv6 first which times out
    at OS level before falling back to IPv4 — the "Happy Eyeballs" problem).
    """
    import json
    import subprocess
    import time as _t

    if not hf_token:
        logger.warning("No hf_token provided, using 'unknown' as username")
        return "unknown"

    t0 = _t.monotonic()
    try:
        result = subprocess.run(
            [
                "curl",
                "-s",
                "-4",  # force IPv4
                "-m",
                str(_HF_WHOAMI_TIMEOUT),  # max time
                "-H",
                f"Authorization: Bearer {hf_token}",
                _HF_WHOAMI_URL,
            ],
            capture_output=True,
            text=True,
            timeout=_HF_WHOAMI_TIMEOUT + 2,
        )
        t1 = _t.monotonic()
        if result.returncode == 0 and result.stdout:
            data = json.loads(result.stdout)
            username = data.get("name", "unknown")
            logger.info(f"HF username resolved to '{username}' in {t1 - t0:.2f}s")
            return username
        else:
            logger.warning(
                f"curl whoami failed (rc={result.returncode}) in {t1 - t0:.2f}s"
            )
            return "unknown"
    except Exception as e:
        t1 = _t.monotonic()
        logger.warning(f"HF whoami failed in {t1 - t0:.2f}s: {e}")
        return "unknown"


_COMPACT_PROMPT = (
    "Please provide a concise summary of the conversation above, focusing on "
    "key decisions, the 'why' behind the decisions, problems solved, and "
    "important context needed for developing further. Your summary will be "
    "given to someone who has never worked on this project before and they "
    "will be have to be filled in."
)

# Per-message ceiling. A single message above this in the preserved tail
# (first user turn or untouched recent window) defeats compaction because
# the body passes through verbatim — the post-compact context stays above
# the threshold and the agent_loop retries compaction indefinitely. We
# replace such messages with a small placeholder before summarization so
# the next pass has a chance of bringing usage below threshold.
_MAX_TOKENS_PER_MESSAGE = 50_000


class CompactionFailedError(Exception):
    """Raised when compaction can't reduce context below the threshold.

    Means a preserved message (system, first user, or untouched tail) is
    intrinsically too large for one truncation+summarize pass. The caller
    must terminate the session — retrying just burns LLM budget on the
    same useless compaction call (cost-amplification pattern from
    huggingface/ml-intern#213).
    """

# Used when seeding a brand-new session from prior browser-cached messages.
# Here we're writing a note to *ourselves* — so preserve the tool-call trail,
# files produced, and planned next steps in first person. Optimized for
# continuity, not brevity.
_RESTORE_PROMPT = (
    "You're about to be restored into a fresh session with no memory of the "
    "conversation above. Write a first-person note to your future self so "
    "you can continue right where you left off. Include:\n"
    "  • What the user originally asked for and what progress you've made.\n"
    "  • Every tool you called, with arguments and a one-line result summary.\n"
    "  • Any code, files, scripts, or artifacts you produced (with paths).\n"
    "  • Key decisions and the reasoning behind them.\n"
    "  • What you were planning to do next.\n\n"
    "Don't be cute. Be specific. This is the only context you'll have."
)


async def summarize_messages(
    messages: list[Message],
    model_name: str,
    hf_token: str | None = None,
    max_tokens: int = 2000,
    tool_specs: list[dict] | None = None,
    prompt: str = _COMPACT_PROMPT,
) -> tuple[str, int]:
    """Run a summarization prompt against a list of messages.

    ``prompt`` defaults to the compaction prompt (terse, decision-focused).
    Callers seeding a new session after a restart should pass ``_RESTORE_PROMPT``
    instead — it preserves the tool-call trail so the agent can answer
    follow-up questions about what it did.

    Returns ``(summary_text, completion_tokens)``.
    """
    from agent.core.llm_params import _resolve_llm_params

    prompt_messages = list(messages) + [Message(role="user", content=prompt)]
    llm_params = _resolve_llm_params(model_name, hf_token, reasoning_effort="high")
    prompt_messages, tool_specs = with_prompt_caching(
        prompt_messages, tool_specs, llm_params.get("model")
    )
    response = await acompletion(
        messages=prompt_messages,
        max_completion_tokens=max_tokens,
        tools=tool_specs,
        **llm_params,
    )
    summary = response.choices[0].message.content or ""
    completion_tokens = response.usage.completion_tokens if response.usage else 0
    return summary, completion_tokens


class ContextManager:
    """Manages conversation context and message history for the agent"""

    def __init__(
        self,
        model_max_tokens: int = 180_000,
        compact_size: float = 0.1,
        untouched_messages: int = 5,
        tool_specs: list[dict[str, Any]] | None = None,
        prompt_file_suffix: str = "system_prompt_v3.yaml",
        hf_token: str | None = None,
        local_mode: bool = False,
    ):
        self.system_prompt = self._load_system_prompt(
            tool_specs or [],
            prompt_file_suffix="system_prompt_v3.yaml",
            hf_token=hf_token,
            local_mode=local_mode,
        )
        # The model's real input-token ceiling (from litellm.get_model_info).
        # Compaction triggers at _COMPACT_THRESHOLD_RATIO below it — see
        # the compaction_threshold property.
        self.model_max_tokens = model_max_tokens
        self.compact_size = int(model_max_tokens * compact_size)
        # Running count of tokens the last LLM call reported. Drives the
        # compaction gate; updated in add_message() with each response's
        # usage.total_tokens.
        self.running_context_usage = 0
        self.untouched_messages = untouched_messages
        self.items: list[Message] = [Message(role="system", content=self.system_prompt)]

    def _load_system_prompt(
        self,
        tool_specs: list[dict[str, Any]],
        prompt_file_suffix: str = "system_prompt.yaml",
        hf_token: str | None = None,
        local_mode: bool = False,
    ):
        """Load and render the system prompt.

        Resolution order:
            1. MLflow Prompt Registry under
               ``$DATABRICKS_AI_INTERN_PROMPT_NAME`` (default ``databricks_ai_intern.agent.system_prompt``)
               at the alias / version named by ``$DATABRICKS_AI_INTERN_PROMPT_VERSION``
               (default ``latest``).
            2. Bundled YAML ``agent/prompts/<prompt_file_suffix>`` — used in
               local CLI runs, unit tests, and any environment where the
               registry isn't reachable.

        The result is fed through Jinja2 (`{{ tools }}` etc.) and decorated
        with session context (date, user) below.
        """
        import os
        from agent.core.prompt_registry import load_system_prompt as _load_prompt

        prompt_name = os.environ.get("DATABRICKS_AI_INTERN_PROMPT_NAME", "databricks_ai_intern.agent.system_prompt")
        prompt_version = os.environ.get("DATABRICKS_AI_INTERN_PROMPT_VERSION") or None
        prompt_file = Path(__file__).parent.parent / "prompts" / f"{prompt_file_suffix}"

        try:
            template_str = _load_prompt(
                prompt_name, version=prompt_version, yaml_path=prompt_file,
            )
        except Exception:
            with open(prompt_file, "r") as f:
                prompt_data = yaml.safe_load(f)
                template_str = prompt_data.get("system_prompt", "")

        # Get current date and time
        tz = zoneinfo.ZoneInfo("Europe/Paris")
        now = datetime.now(tz)
        current_date = now.strftime("%d-%m-%Y")
        current_time = now.strftime("%H:%M:%S.%f")[:-3]
        current_timezone = f"{now.strftime('%Z')} (UTC{now.strftime('%z')[:3]}:{now.strftime('%z')[3:]})"

        # Get HF user info from OAuth token
        hf_user_info = _get_hf_username(hf_token)

        # Pull the skill catalog (one-liner per registered skill). Lives
        # outside the YAML so contributors can drop a skill into
        # ``agent/skills/<name>/`` and the agent picks it up on next boot.
        try:
            from agent.skills import skill_catalog_for_prompt

            skill_catalog = skill_catalog_for_prompt()
        except Exception:
            skill_catalog = ""

        template = Template(template_str)
        static_prompt = template.render(
            tools=tool_specs,
            num_tools=len(tool_specs),
            skill_catalog=skill_catalog,
        )

        if skill_catalog and "{{ skill_catalog }}" not in template_str:
            # YAML template was authored before the skill loader existed
            # (or the user-supplied YAML doesn't include the placeholder).
            # Append the catalog so the agent still sees it.
            static_prompt = f"{static_prompt}\n\n{skill_catalog}\n"

        # CLI-specific context for local mode
        if local_mode:
            import os
            cwd = os.getcwd()
            local_context = (
                f"\n\n# CLI / Local mode\n\n"
                f"You are running as a local CLI tool on the user's machine. "
                f"There is NO sandbox — bash, read, write, and edit operate directly "
                f"on the local filesystem.\n\n"
                f"Working directory: {cwd}\n"
                f"Use absolute paths or paths relative to the working directory. "
                f"Do NOT use /app/ paths — that is a sandbox convention that does not apply here.\n"
                f"The sandbox_create tool is NOT available. Run code directly with bash."
            )
            static_prompt += local_context

        return (
            f"{static_prompt}\n\n"
            f"[Session context: Date={current_date}, Time={current_time}, "
            f"Timezone={current_timezone}, User={hf_user_info}, "
            f"Tools={len(tool_specs)}]"
        )

    def add_message(self, message: Message, token_count: int = None) -> None:
        """Add a message to the history"""
        if token_count:
            self.running_context_usage = token_count
        self.items.append(message)

    def get_messages(self) -> list[Message]:
        """Get all messages for sending to LLM.

        Patches any dangling tool_calls (assistant messages with tool_calls
        that have no matching tool-result message) so the LLM API doesn't
        reject the request.
        """
        self._patch_dangling_tool_calls()
        return self.items

    @staticmethod
    def _normalize_tool_calls(msg: Message) -> None:
        """Ensure msg.tool_calls contains proper ToolCall objects, not dicts.

        litellm's Message has validate_assignment=False (Pydantic v2 default),
        so direct attribute assignment (e.g. inside litellm's streaming handler)
        can leave raw dicts.  Re-assigning via the constructor fixes this.
        """
        from litellm import ChatCompletionMessageToolCall as ToolCall

        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            return
        needs_fix = any(isinstance(tc, dict) for tc in tool_calls)
        if not needs_fix:
            return
        msg.tool_calls = [
            tc if not isinstance(tc, dict) else ToolCall(**tc) for tc in tool_calls
        ]

    def _patch_dangling_tool_calls(self) -> None:
        """Add stub tool results for any tool_calls that lack a matching result.

        Scans backwards to find the last assistant message with tool_calls,
        which may not be items[-1] if some tool results were already added.
        """
        if not self.items:
            return

        # Find the last assistant message with tool_calls
        assistant_msg = None
        for i in range(len(self.items) - 1, -1, -1):
            msg = self.items[i]
            if getattr(msg, "role", None) == "assistant" and getattr(
                msg, "tool_calls", None
            ):
                assistant_msg = msg
                break
            # Stop scanning once we hit a user message — anything before
            # that belongs to a previous (complete) turn.
            if getattr(msg, "role", None) == "user":
                break

        if not assistant_msg:
            return

        self._normalize_tool_calls(assistant_msg)
        answered_ids = {
            getattr(m, "tool_call_id", None)
            for m in self.items
            if getattr(m, "role", None) == "tool"
        }
        for tc in assistant_msg.tool_calls:
            if tc.id not in answered_ids:
                self.items.append(
                    Message(
                        role="tool",
                        content="Tool was not executed (interrupted or error).",
                        tool_call_id=tc.id,
                        name=tc.function.name,
                    )
                )

    def undo_last_turn(self) -> bool:
        """Remove the last complete turn (user msg + all assistant/tool msgs that follow).

        Pops from the end until the last user message is removed, keeping the
        tool_use/tool_result pairing valid. Never removes the system message.

        Returns True if a user message was found and removed.
        """
        if len(self.items) <= 1:
            return False

        while len(self.items) > 1:
            msg = self.items.pop()
            if getattr(msg, "role", None) == "user":
                return True

        return False

    def truncate_to_user_message(self, user_message_index: int) -> bool:
        """Truncate history to just before the Nth user message (0-indexed).

        Removes that user message and everything after it.
        System message (index 0) is never removed.

        Returns True if the target user message was found and removed.
        """
        count = 0
        for i, msg in enumerate(self.items):
            if i == 0:
                continue  # skip system message
            if getattr(msg, "role", None) == "user":
                if count == user_message_index:
                    self.items = self.items[:i]
                    return True
                count += 1
        return False

    # Compaction fires at 90% of model_max_tokens so there's headroom for
    # the next turn's prompt + response before we actually hit the ceiling.
    _COMPACT_THRESHOLD_RATIO = 0.9

    @property
    def compaction_threshold(self) -> int:
        """Token count at which `compact()` kicks in."""
        return int(self.model_max_tokens * self._COMPACT_THRESHOLD_RATIO)

    @property
    def needs_compaction(self) -> bool:
        return self.running_context_usage > self.compaction_threshold and bool(self.items)

    def _truncate_oversized(
        self, messages: list[Message], model_name: str
    ) -> list[Message]:
        """Replace any message > _MAX_TOKENS_PER_MESSAGE with a placeholder.

        Typically the offender is a tool output (CSV dump, file contents,
        bash output) sitting in the untouched tail or first-user position.
        Those bodies pass through compaction verbatim and keep usage above
        threshold — the same shape that drove the upstream HF#213 infinite
        compaction loop.

        System messages are never touched: the slice math in compact() can,
        in edge cases (items < untouched_messages), let items[0] leak into
        recent_messages. Defense-in-depth.
        """
        from litellm import token_counter

        out: list[Message] = []
        for msg in messages:
            if msg.role == "system":
                out.append(msg)
                continue
            try:
                n = token_counter(model=model_name, messages=[msg.model_dump()])
            except Exception:
                # token_counter occasionally raises on edge-case content.
                # Don't drop the message — keep it; if it's the offender,
                # compact() will raise CompactionFailedError below.
                out.append(msg)
                continue
            if n <= _MAX_TOKENS_PER_MESSAGE:
                out.append(msg)
                continue
            placeholder = (
                f"[truncated for compaction — original was {n} tokens, "
                f"removed to keep context under {self.compaction_threshold} tokens]"
            )
            logger.warning(
                "Truncating %s message: %d -> %d tokens for compaction",
                msg.role, n, len(placeholder) // 4,
            )
            # Preserve all known assistant-side fields when content is replaced.
            # Anthropic extended-thinking models reject the next request with
            # "Invalid signature in thinking block" if thinking_blocks is
            # dropped from a prior assistant message.
            kept = {
                k: getattr(msg, k, None)
                for k in (
                    "tool_call_id",
                    "tool_calls",
                    "name",
                    "thinking_blocks",
                    "reasoning_content",
                    "provider_specific_fields",
                )
                if getattr(msg, k, None) is not None
            }
            out.append(Message(role=msg.role, content=placeholder, **kept))
        return out

    def _recompute_usage(self, model_name: str) -> None:
        """Refresh ``running_context_usage`` from current items via real tokenizer.

        On ``token_counter`` failure we set usage to ``model_max_tokens + 1``
        so the post-compact threshold check in ``compact()`` raises
        ``CompactionFailedError``. A char-based fallback undercounts messages
        that carry their weight in metadata fields the truncation path
        preserves (``thinking_blocks``, ``tool_calls``,
        ``provider_specific_fields``): the next LLM call would then exceed
        the context window anyway and re-enter the compaction path, which
        is exactly the loop this whole patch is meant to prevent. Failing
        loud is consistent with the PR's "never retry on uncertain state"
        invariant.
        """
        from litellm import token_counter

        try:
            self.running_context_usage = token_counter(
                model=model_name,
                messages=[m.model_dump() for m in self.items],
            )
        except Exception as e:
            logger.warning(
                "token_counter failed (%s); marking usage over-threshold so "
                "compact() raises CompactionFailedError instead of returning "
                "with an undercount that would re-loop on the next LLM call.",
                e,
            )
            self.running_context_usage = self.model_max_tokens + 1

    async def compact(
        self,
        model_name: str,
        tool_specs: list[dict] | None = None,
        hf_token: str | None = None,
        session=None,
    ) -> None:
        """Remove old messages to keep history under target size.

        Raises ``CompactionFailedError`` when the post-compact context is
        still over the threshold — typically because an individual
        preserved message exceeds what truncation can fix in one pass.
        The caller (``agent_loop._compact_and_notify``) must terminate
        the session; retrying produces the infinite-loop cost-amplification
        pattern that this method is designed to prevent.

        ``session`` is accepted but currently unused — kept on the
        signature so future telemetry hooks (cost-attribution for the
        compaction LLM call) can land without changing every caller.
        """
        if not self.needs_compaction:
            return

        system_msg = (
            self.items[0] if self.items and self.items[0].role == "system" else None
        )

        # Preserve the first user message (task prompt) — never summarize it
        first_user_msg = None
        first_user_idx = 1
        for i in range(1, len(self.items)):
            if getattr(self.items[i], "role", None) == "user":
                first_user_msg = self.items[i]
                first_user_idx = i
                break

        # Don't summarize a certain number of just-preceding messages
        # Walk back to find a user message to make sure we keep an assistant -> user ->
        # assistant general conversation structure
        idx = len(self.items) - self.untouched_messages
        while idx > 1 and self.items[idx].role != "user":
            idx -= 1
        # The real invariant is "idx must be strictly after first_user_idx",
        # otherwise recent_messages overlaps with what we place in head.
        # Two failure modes the walk-back alone doesn't cover:
        #   * len(items) <= untouched_messages -> idx initialises <= 0, the
        #     loop is a no-op, and recent_messages starts at the system
        #     message. Rebuild then duplicates system.
        #   * idx lands at first_user_idx -> first_user ends up in both
        #     head and recent_messages. Two consecutive user turns ->
        #     Anthropic API 400.
        if idx <= first_user_idx:
            idx = first_user_idx + 1

        recent_messages = self.items[idx:]
        messages_to_summarize = self.items[first_user_idx + 1:idx]

        # Truncate oversized messages in the parts we PRESERVE through
        # compaction (first_user + recent_tail). Messages inside
        # ``messages_to_summarize`` are folded into the summary, so their
        # individual size doesn't drive the loop.
        if first_user_msg is not None:
            truncated = self._truncate_oversized([first_user_msg], model_name)
            first_user_msg = truncated[0]
        recent_messages = self._truncate_oversized(recent_messages, model_name)

        # If there's nothing to summarize but the preserved messages are
        # now truncated, rebuild and recompute. Without this branch the
        # method returns silently with the old over-threshold usage and
        # the agent_loop retries forever.
        if not messages_to_summarize:
            head = [system_msg] if system_msg else []
            if first_user_msg:
                head.append(first_user_msg)
            self.items = head + recent_messages
            self._recompute_usage(model_name)
            if self.running_context_usage > self.compaction_threshold:
                self._aggressive_reduce(model_name)
            if self.running_context_usage > self.compaction_threshold:
                raise CompactionFailedError(
                    f"Nothing to summarize but context ({self.running_context_usage}) "
                    f"still over threshold ({self.compaction_threshold}) after truncation "
                    f"and one aggressive reduction pass. "
                    f"System prompt or first user message likely exceeds the budget."
                )
            return

        summary, completion_tokens = await summarize_messages(
            messages_to_summarize,
            model_name=model_name,
            hf_token=hf_token,
            max_tokens=self.compact_size,
            tool_specs=tool_specs,
            prompt=_COMPACT_PROMPT,
        )
        summarized_message = Message(role="assistant", content=summary)

        # Reconstruct: system + first user msg + summary + recent messages
        head = [system_msg] if system_msg else []
        if first_user_msg:
            head.append(first_user_msg)
        self.items = head + [summarized_message] + recent_messages

        # Count the actual post-compact context — system prompt + first user
        # turn + summary + the preserved tail all contribute, not just the
        # summary. litellm.token_counter uses the model's real tokenizer.
        self._recompute_usage(model_name)

        # Hard verify: if compaction still couldn't drop us below the
        # threshold, attempt ONE more aggressive reduction (drop the largest
        # non-essential preserved messages) before giving up. Only if that
        # still can't fit do we raise — retrying the full summarize path
        # just burns budget on the same useless call.
        if self.running_context_usage > self.compaction_threshold:
            self._aggressive_reduce(model_name)
        if self.running_context_usage > self.compaction_threshold:
            raise CompactionFailedError(
                f"Compaction ineffective: {self.running_context_usage} tokens "
                f"still over threshold {self.compaction_threshold} after summarize, "
                f"truncation, and one aggressive reduction pass. Likely the system "
                f"prompt + first user + summary still exceeds budget."
            )

    def _aggressive_reduce(self, model_name: str) -> None:
        """One-shot last-resort reduction when normal compaction can't fit.

        Runs exactly once (the caller checks the threshold again afterward and
        raises if still over — no looping). Keeps the essential spine —
        system message, first user turn, and the last ``untouched_messages``
        — and drops every other message in between, then recomputes usage.

        This is strictly more aggressive than ``_truncate_oversized`` (which
        only shrinks messages above a per-message ceiling): here we remove
        whole mid-conversation messages whose combined weight, not any single
        one, keeps the context over threshold. Tool-call/result pairing in the
        preserved tail is left intact because we never split it.
        """
        items = self.items
        if len(items) <= 1:
            return

        system_msg = items[0] if items[0].role == "system" else None
        start = 1 if system_msg else 0

        first_user_idx = None
        for i in range(start, len(items)):
            if getattr(items[i], "role", None) == "user":
                first_user_idx = i
                break

        # Indices that form the essential spine and must be preserved.
        keep_idx: set[int] = set()
        if system_msg is not None:
            keep_idx.add(0)
        if first_user_idx is not None:
            keep_idx.add(first_user_idx)
        tail_start = max(0, len(items) - self.untouched_messages)
        keep_idx.update(range(tail_start, len(items)))

        reduced = [items[i] for i in sorted(keep_idx)]
        if len(reduced) == len(items):
            # Nothing left to drop — every message is already essential.
            return

        logger.warning(
            "Aggressive compaction reduction: %d -> %d messages "
            "(dropped non-essential mid-conversation turns).",
            len(items), len(reduced),
        )
        self.items = reduced
        self._recompute_usage(model_name)
