"""One truncation policy for every tool result that reaches the model.

Before this, truncation was per-tool and inconsistent: `local_tools` capped bash/read at
25,000 chars, `research_tool` capped itself at 8,000, and most tools — including every
MCP tool, whose output we do not control at all — had no cap. A single unbounded result
could push the conversation into compaction, which costs an LLM call and loses history
that was worth keeping.

The policy is tail-biased on purpose: when a command fails, the error is at the end.

`spill_path` is the durable-vs-model-facing split from deepseek's `ToolExecutionResult`
(`packages/core/tools/src/index.ts`), where `content` is what the model sees and `value`
stays execution-local. Here the full output goes to a file the model can `read` on
demand, and only the trimmed view enters history.
"""

from __future__ import annotations

import re
import tempfile

# Per-tool caps (bash/read at 25k) sit below this. This is the backstop for everything
# that never set one, so it should not fight a tool that already trimmed itself.
MODEL_FACING_CHAR_CAP = 30_000
DEFAULT_HEAD_RATIO = 0.25

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]|\x1b\].*?\x07")


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _spill(output: str, prefix: str) -> str | None:
    """Write full output to a temp file so nothing is actually lost."""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", prefix=prefix, delete=False, encoding="utf-8"
        ) as f:
            f.write(output)
            return f.name
    except OSError:
        return None


def truncate_output(
    output: str,
    max_chars: int = MODEL_FACING_CHAR_CAP,
    head_ratio: float = DEFAULT_HEAD_RATIO,
    *,
    prefix: str = "tool_output_",
    finished_note: bool = True,
) -> str:
    """Tail-biased truncation with temp-file spillover for full output access."""
    if len(output) <= max_chars:
        return output

    spill_path = _spill(output, prefix)
    head_budget = int(max_chars * head_ratio)
    tail_budget = max_chars - head_budget
    total = len(output)
    omitted = total - max_chars

    meta = (
        f"\n\n... ({omitted:,} of {total:,} chars omitted, showing first "
        f"{head_budget:,} + last {tail_budget:,}) ...\n"
    )
    if spill_path:
        meta += (
            f"Full output saved to {spill_path} — use the read tool with offset/limit "
            "to inspect specific sections.\n"
        )
    if finished_note:
        meta += (
            "IMPORTANT: The command has finished. Analyze the output above and continue "
            "with your next action.\n"
        )
    return output[:head_budget] + meta + output[-tail_budget:]
