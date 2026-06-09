"""Model-switching logic for the interactive CLI's ``/model`` command.

Split out of ``agent.main`` so the REPL dispatcher stays focused on input
parsing. Exposes:

* ``SUGGESTED_MODELS`` — the short list shown by ``/model`` with no arg.
* ``is_valid_model_id`` — loose format check on user input.
* ``probe_and_switch_model`` — async: validates routing, fires a 1-token
  probe to resolve the effort cascade, then commits the switch (or
  rejects on hard error).
"""

from __future__ import annotations

from agent.core.effort_probe import ProbeInconclusive, probe_effort


# Suggested models shown by ``/model`` (not a gate). Defaults to Databricks
# Foundation Model API endpoints; users can paste any ``databricks/<endpoint>``
# their workspace exposes, or ``anthropic/`` / ``openai/`` for direct API.
# HF router ids still work for research-tool fallback paths.
SUGGESTED_MODELS = [
    {"id": "databricks/databricks-claude-opus-4-6",  "label": "Claude Opus 4.6 (Databricks FMAPI)"},
    {"id": "databricks/databricks-claude-sonnet-4-6", "label": "Claude Sonnet 4.6 (Databricks FMAPI)"},
    {"id": "databricks/databricks-claude-haiku-4-5", "label": "Claude Haiku 4.5 (Databricks FMAPI)"},
    {"id": "databricks/databricks-gpt-oss-120b", "label": "GPT-OSS 120B (Databricks FMAPI)"},
    {"id": "databricks/databricks-meta-llama-3-3-70b-instruct", "label": "Llama 3.3 70B (Databricks FMAPI)"},
]


def is_valid_model_id(model_id: str) -> bool:
    """Loose format check — lets users pick any provider-prefixed model id.

    Accepts:
      • databricks/<endpoint>
      • anthropic/<model>
      • openai/<model>
      • bedrock/<model>
      • <org>/<model>[:<tag>]            (HF router; tag = provider or policy)
      • huggingface/<org>/<model>[:<tag>] (legacy HF prefix)
    """
    if not model_id or "/" not in model_id:
        return False
    head = model_id.split(":", 1)[0]
    parts = head.split("/")
    return len(parts) >= 2 and all(parts)


def _print_routing_info(model_id: str, console) -> bool:
    """Show endpoint info for ``databricks/<endpoint>`` ids; HF-router info
    for HF ids; no output for direct-API ids. Returns True to proceed.
    """
    if model_id.startswith(("anthropic/", "openai/", "bedrock/")):
        return True

    if model_id.startswith("databricks/"):
        from agent.core import model_catalog as cat

        info = cat.lookup(model_id)
        if info is None:
            console.print(
                f"[bold red]Warning:[/bold red] '{model_id}' isn't in this "
                "workspace's serving endpoints. First call may fail."
            )
            suggestions = cat.fuzzy_suggest(model_id)
            if suggestions:
                console.print(
                    "[dim]Did you mean: "
                    + ", ".join(f"databricks/{s}" for s in suggestions)
                    + "[/dim]"
                )
            return True
        if not info.is_ready:
            console.print(
                f"[bold red]Warning:[/bold red] endpoint '{info.name}' is "
                f"in state {info.state}. First call may fail."
            )
        if not info.is_chat:
            console.print(
                f"[bold red]Warning:[/bold red] endpoint '{info.name}' "
                f"is task={info.task!r}; this agent expects llm/v1/chat."
            )
        served = ", ".join(info.served_entities) or "?"
        console.print(
            f"  [dim]{info.name}: {info.state}, "
            f"task={info.task or '?'}, entities=[{served}], "
            f"type={info.endpoint_type or '?'}[/dim]"
        )
        return True

    # HF-router path — still live: users can paste HF router ids, and
    # research-tool sub-calls inherit non-Anthropic main models verbatim.
    # No routing info to print for these.
    return True


def print_model_listing(config, console) -> None:
    current = config.model_name if config else ""
    console.print("[bold]Current model:[/bold]")
    console.print(f"  {current}")
    console.print("\n[bold]Suggested:[/bold]")
    for m in SUGGESTED_MODELS:
        marker = " [dim]<-- current[/dim]" if m["id"] == current else ""
        console.print(f"  {m['id']}  [dim]({m['label']})[/dim]{marker}")
    console.print(
        "\n[dim]Paste a 'databricks/<endpoint>' from this workspace, "
        "an 'anthropic/<model>' / 'openai/<model>' for direct API, "
        "or any HF router id.[/dim]"
    )


def print_invalid_id(arg: str, console) -> None:
    console.print(f"[bold red]Invalid model id format:[/bold red] {arg}")
    console.print(
        "[dim]Expected:\n"
        "  • databricks/<endpoint>     (Foundation Model API)\n"
        "  • anthropic/<model>\n"
        "  • openai/<model>\n"
        "  • bedrock/<model>\n"
        "  • <org>/<model>[:tag]       (HF router)[/dim]"
    )


async def probe_and_switch_model(
    model_id: str,
    config,
    session,
    console,
    hf_token: str | None,
) -> None:
    """Validate model+effort with a 1-token ping, cache the effective effort,
    then commit the switch.

    Three visible outcomes:

    * ✓ ``effort: <level>`` — model accepted preferred effort (or fallback).
    * ✓ ``effort: off`` — model doesn't support thinking; we strip it.
    * ✗ hard error (auth, not-found, quota) — we reject the switch and keep
      the current model so the user isn't stranded.

    On a transient (inconclusive) probe we do NOT switch — the active model
    and effort stay as-is and we emit a yellow warning that the switch was
    deferred. Committing the switch on an unresolved probe risks a hard
    failure on the very next turn of an unattended run; better to keep the
    known-good model than to gamble.
    """
    preference = config.reasoning_effort
    if not _print_routing_info(model_id, console):
        return

    if not preference:
        _commit_switch(model_id, config, session, effective=None, cache=False)
        console.print(f"[green]Model switched to {model_id}[/green] [dim](effort: off)[/dim]")
        return

    console.print(f"[dim]checking {model_id} (effort: {preference})...[/dim]")
    try:
        outcome = await probe_effort(model_id, preference, hf_token)
    except ProbeInconclusive as e:
        # Transient probe failure → leave the active model/effort untouched.
        # Switching now could hard-fail the next turn of an unattended run.
        current = config.model_name
        console.print(
            f"[yellow]Switch to {model_id} deferred:[/yellow] "
            f"[dim]couldn't validate ({e}); keeping current model {current}. "
            f"Try again once the provider is reachable.[/dim]"
        )
        return
    except Exception as e:
        console.print(f"[bold red]Switch failed:[/bold red] {e}")
        console.print(f"[dim]Keeping current model: {config.model_name}[/dim]")
        return

    _commit_switch(
        model_id, config, session,
        effective=outcome.effective_effort, cache=True,
    )
    effort_label = outcome.effective_effort or "off"
    suffix = f" — {outcome.note}" if outcome.note else ""
    console.print(
        f"[green]Model switched to {model_id}[/green] "
        f"[dim](effort: {effort_label}{suffix}, {outcome.elapsed_ms}ms)[/dim]"
    )


def _commit_switch(model_id, config, session, effective, cache: bool) -> None:
    if session is not None:
        session.update_model(model_id)
        if cache:
            session.model_effective_effort[model_id] = effective
        else:
            session.model_effective_effort.pop(model_id, None)
    else:
        config.model_name = model_id
