"""Tests for the critic/verifier — audit a result for false-win failure modes."""

from agent.core.critic import (
    Finding,
    audit,
    check_correlation_floor,
    check_leakage,
    check_overfit,
    check_target_confusion,
)


# ── check_overfit ────────────────────────────────────────────────────────


def test_overfit_warn_when_cv_beats_lb():
    f = check_overfit(0.96, 0.945)  # gap 0.015, between 1x and 2x threshold
    assert isinstance(f, Finding)
    assert f.kind == "overfit"
    assert f.severity == "warn"
    assert abs(f.evidence["gap"] - 0.015) < 1e-9


def test_overfit_block_when_gap_large():
    f = check_overfit(0.97, 0.93)  # gap 0.04 > 2*0.01
    assert f is not None
    assert f.severity == "block"


def test_overfit_clean_when_lb_tracks_cv():
    # gap within threshold -> no finding.
    assert check_overfit(0.95, 0.948) is None


def test_overfit_none_when_score_missing():
    assert check_overfit(None, 0.94) is None
    assert check_overfit(0.95, None) is None


def test_overfit_lower_is_better_direction():
    # eval_loss: CV "better" (lower) than LB by 0.03 -> overfit, block.
    f = check_overfit(0.30, 0.33, higher_is_better=False)
    assert f is not None
    assert f.kind == "overfit"
    assert f.severity == "block"
    assert abs(f.evidence["gap"] - 0.03) < 1e-9


def test_overfit_lower_is_better_clean():
    # CV loss worse than LB loss -> not overfit.
    assert check_overfit(0.33, 0.30, higher_is_better=False) is None


# ── check_target_confusion ─────────────────────────────────────────────────


def test_target_confusion_positive():
    f = check_target_confusion(["id", "PitStop"], "Position")
    assert f is not None
    assert f.kind == "target_confusion"
    assert f.severity == "block"
    assert f.evidence["predicted"] == "Position"


def test_target_confusion_clean():
    assert check_target_confusion(["id", "PitStop"], "PitStop") is None


def test_target_confusion_respects_id_columns():
    # "id" is an id column; predicting it is wrong even though it's a column.
    f = check_target_confusion(["id", "PitStop"], "id", id_columns=["id"])
    assert f is not None
    assert f.severity == "block"
    assert "id" not in f.evidence["expected"]
    assert f.evidence["expected"] == ["PitStop"]


# ── check_leakage ────────────────────────────────────────────────────────


def test_leakage_dominance_warn():
    f = check_leakage({"a": 0.8, "b": 0.1, "c": 0.1})  # share 0.8 >= 0.6
    assert f is not None
    assert f.kind == "leakage"
    assert f.severity == "warn"
    assert f.evidence["top_feature"] == "a"
    assert abs(f.evidence["share"] - 0.8) < 1e-9


def test_leakage_dominance_clean():
    assert check_leakage({"a": 0.4, "b": 0.3, "c": 0.3}) is None


def test_leakage_adversarial_warn():
    f = check_leakage(None, adversarial_auc=0.85)  # >= 0.8, < 0.95
    assert f is not None
    assert f.kind == "leakage"
    assert f.severity == "warn"
    assert abs(f.evidence["adversarial_auc"] - 0.85) < 1e-9


def test_leakage_adversarial_block_at_high_auc():
    f = check_leakage(None, adversarial_auc=0.97)  # >= 0.95
    assert f is not None
    assert f.severity == "block"


def test_leakage_adversarial_clean_below_threshold():
    assert check_leakage(None, adversarial_auc=0.6) is None


def test_leakage_none_when_no_inputs():
    assert check_leakage(None) is None
    assert check_leakage({}) is None


# ── check_correlation_floor ─────────────────────────────────────────────────


def test_correlation_floor_warn():
    f = check_correlation_floor(0.999)
    assert f is not None
    assert f.kind == "correlation_floor"
    assert f.severity == "warn"
    assert abs(f.evidence["spearman"] - 0.999) < 1e-9


def test_correlation_floor_clean_real_diversity():
    assert check_correlation_floor(0.99) is None


def test_correlation_floor_none_when_missing():
    assert check_correlation_floor(None) is None


# ── audit: dispatch only present detectors, aggregate findings ──────────────


def test_audit_empty_signals_returns_no_findings():
    assert audit({}) == []


def test_audit_partial_signals_skips_incomplete_detectors():
    # Only cv_score (no lb_score) -> overfit can't run; nothing else present.
    assert audit({"cv_score": 0.96}) == []


def test_audit_aggregates_overfit_and_correlation_floor():
    findings = audit(
        {
            "cv_score": 0.97,
            "lb_score": 0.93,  # overfit, block
            "spearman": 0.999,  # correlation_floor, warn
        }
    )
    kinds = {f.kind for f in findings}
    assert kinds == {"overfit", "correlation_floor"}
    assert len(findings) == 2


def test_audit_runs_all_detectors_when_inputs_present():
    findings = audit(
        {
            "cv_score": 0.97,
            "lb_score": 0.93,
            "sample_submission_columns": ["id", "PitStop"],
            "predicted_target": "Position",
            "feature_importances": {"a": 0.9, "b": 0.1},
            "spearman": 0.999,
        }
    )
    kinds = {f.kind for f in findings}
    assert kinds == {"overfit", "target_confusion", "leakage", "correlation_floor"}


def test_audit_clean_signals_returns_no_findings():
    findings = audit(
        {
            "cv_score": 0.95,
            "lb_score": 0.949,
            "sample_submission_columns": ["id", "PitStop"],
            "predicted_target": "PitStop",
            "id_columns": ["id"],
            "feature_importances": {"a": 0.4, "b": 0.3, "c": 0.3},
            "spearman": 0.99,
        }
    )
    assert findings == []


def test_audit_passes_through_optional_kwargs():
    # Wide overfit_threshold makes a 0.015 gap fall below the floor.
    assert audit({"cv_score": 0.96, "lb_score": 0.945, "overfit_threshold": 0.05}) == []
    # higher_is_better=False routes the loss direction.
    findings = audit(
        {"cv_score": 0.30, "lb_score": 0.33, "higher_is_better": False}
    )
    assert [f.kind for f in findings] == ["overfit"]
