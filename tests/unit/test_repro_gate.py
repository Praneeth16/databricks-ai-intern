"""Tests for the reproduction-gap gate — block premature complexity escalation."""

from dataclasses import dataclass

from agent.core.repro_gate import GateDecision, evaluate_gap, gate_from_row


# ── Lightweight stand-in so we don't depend on the full ExperimentRow ───


@dataclass
class _FakeRow:
    expected_metric: float | None
    actual_metric: float | None


# ── unknown: no expectation to hold against ─────────────────────────────


def test_unknown_when_expected_missing():
    d = evaluate_gap(expected_metric=None, actual_metric=0.94)
    assert d == GateDecision(blocked=False, severity="unknown", gap=None, directive=None)


def test_unknown_when_actual_missing():
    d = evaluate_gap(expected_metric=0.95, actual_metric=None)
    assert d.severity == "unknown"
    assert d.blocked is False
    assert d.gap is None
    assert d.directive is None


# ── ok: met or within the minor threshold ───────────────────────────────


def test_ok_when_beats_expectation():
    # actual > expected => negative shortfall => comfortably ok.
    d = evaluate_gap(expected_metric=0.94, actual_metric=0.96)
    assert d.severity == "ok"
    assert d.blocked is False
    assert d.directive is None
    assert d.gap is not None and d.gap < 0


def test_ok_just_within_minor_threshold():
    # shortfall just under minor_threshold stays ok.
    d = evaluate_gap(expected_metric=0.95, actual_metric=0.946)  # shortfall 0.004
    assert d.severity == "ok"
    assert d.blocked is False
    assert d.directive is None


# ── minor: advisory, not blocked ────────────────────────────────────────


def test_minor_is_advisory_not_blocked():
    d = evaluate_gap(expected_metric=0.95, actual_metric=0.94)  # shortfall 0.01
    assert d.severity == "minor"
    assert d.blocked is False
    assert d.directive is not None
    assert abs(d.gap - 0.01) < 1e-9


def test_minor_just_within_major_threshold():
    # shortfall just under major_threshold is still minor, not blocked.
    d = evaluate_gap(expected_metric=0.95, actual_metric=0.931)  # shortfall 0.019
    assert d.severity == "minor"
    assert d.blocked is False


# ── major: blocked with reproduction-first directive ────────────────────


def test_major_blocks_and_names_the_gap():
    d = evaluate_gap(expected_metric=0.95, actual_metric=0.90)  # shortfall 0.05
    assert d.severity == "major"
    assert d.blocked is True
    assert abs(d.gap - 0.05) < 1e-9
    text = d.directive.lower()
    assert "reproduce" in text
    assert "do not add" in text
    # The concrete shortfall value is named in the directive.
    assert "0.05" in d.directive


def test_major_directive_lists_reproduction_checks():
    d = evaluate_gap(expected_metric=0.95, actual_metric=0.85)
    text = d.directive.lower()
    assert "leakage" in text
    assert "learning rate" in text or "schedule" in text
    assert "preprocessing" in text or "dataset format" in text


# ── higher_is_better=False (e.g. eval_loss, lower is better) ─────────────


def test_lower_is_better_ok_when_below_expected():
    # eval_loss 0.30 vs expected 0.35: we beat it -> ok.
    d = evaluate_gap(
        expected_metric=0.35, actual_metric=0.30, higher_is_better=False
    )
    assert d.severity == "ok"
    assert d.blocked is False
    assert d.gap is not None and d.gap < 0


def test_lower_is_better_major_when_far_above_expected():
    # eval_loss 0.45 vs expected 0.35: 0.10 worse -> major, blocked.
    d = evaluate_gap(
        expected_metric=0.35, actual_metric=0.45, higher_is_better=False
    )
    assert d.severity == "major"
    assert d.blocked is True
    assert abs(d.gap - 0.10) < 1e-9


# ── custom thresholds ────────────────────────────────────────────────────


def test_custom_thresholds_shift_severity():
    # With wide thresholds a 0.05 shortfall is only minor, not major.
    d = evaluate_gap(
        expected_metric=0.95,
        actual_metric=0.90,
        minor_threshold=0.01,
        major_threshold=0.10,
    )
    assert d.severity == "minor"
    assert d.blocked is False


# ── gate_from_row: reads expected/actual off a row-like object ──────────


def test_gate_from_row_major():
    row = _FakeRow(expected_metric=0.95, actual_metric=0.88)
    d = gate_from_row(row)
    assert d.severity == "major"
    assert d.blocked is True


def test_gate_from_row_unknown_when_expected_none():
    row = _FakeRow(expected_metric=None, actual_metric=0.91)
    d = gate_from_row(row)
    assert d.severity == "unknown"
    assert d.blocked is False


def test_gate_from_row_passes_through_kwargs():
    # Lower-is-better row routed through gate_from_row.
    row = _FakeRow(expected_metric=0.35, actual_metric=0.50)
    d = gate_from_row(row, higher_is_better=False)
    assert d.severity == "major"
    assert d.blocked is True
    assert abs(d.gap - 0.15) < 1e-9
