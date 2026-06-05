"""CLI rendering tests for terminal_display (issue #11).

Port of upstream HF#248 ``test_help_output_keeps_descriptions_aligned``
and ``test_help_output_recomputes_widths_from_rows`` so the description
column stays column-aligned even as rows get added / args grow.
"""

from __future__ import annotations

from io import StringIO

from rich.console import Console

from agent.utils import terminal_display


def test_help_output_keeps_descriptions_aligned(monkeypatch):
    output = StringIO()
    console = Console(
        file=output,
        color_system=None,
        theme=terminal_display._THEME,
        width=120,
    )
    monkeypatch.setattr(terminal_display, "_console", console)

    terminal_display.print_help()

    lines = [line.rstrip() for line in output.getvalue().splitlines() if line.strip()]
    description_columns = []
    for command, args, description in terminal_display.HELP_ROWS:
        line = next(line for line in lines if command in line)
        if args:
            assert args in line
        description_columns.append(line.index(description))

    # All description fields land at the same column index — the entire
    # point of the format_help_text width derivation.
    assert len(set(description_columns)) == 1


def test_help_output_recomputes_widths_from_rows():
    rows = terminal_display.HELP_ROWS + (
        ("/longer-command", "[longer-args]", "Synthetic help row"),
    )
    output = StringIO()
    Console(
        file=output,
        color_system=None,
        theme=terminal_display._THEME,
        width=140,
    ).print(terminal_display.format_help_text(rows))

    lines = [line.rstrip() for line in output.getvalue().splitlines() if line.strip()]
    description_columns = [
        next(line for line in lines if command in line).index(description)
        for command, _args, description in rows
    ]
    # Even after adding a long synthetic row, the descriptions stay aligned.
    assert len(set(description_columns)) == 1
