"""Eval harness for the auto-researcher.

Phase 1 spine: load a task spec, drive an agent against it, score the
result deterministically, record the run to the experiment ledger, and
write a scorecard. Everything else in the auto-researcher is validated
against this harness — "can't improve what you can't measure."

Public surface:
    - ``task_spec.EvalTask`` / ``task_spec.load_task``
    - ``scorers`` — pure deterministic metric functions
    - ``runner.run_eval`` / ``runner.EvalResult``
    - ``report.write_report``
"""

from __future__ import annotations
