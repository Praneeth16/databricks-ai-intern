"""CPU-vs-GPU compute selection and the script/compute mismatch checks."""

import pytest

from agent.core.compute_advisor import (
    GPU_WORTH_IT_ROWS,
    advise_from_script,
    check_runtime_budget,
    recommend_compute,
)

# The competition this was built for: 691,369 rows x 12 features, tabular GBM.
S6E8 = dict(task_shape="tabular", model_family="lightgbm", n_rows=691_369, n_features=12)


# --------------------------------------------------------------- recommendations


def test_playground_scale_tabular_goes_to_cpu():
    rec = recommend_compute(**S6E8)
    assert rec.accelerator == "cpu"
    assert rec.kind == "serverless"
    assert "CPU" in rec.render()


def test_cpu_flavor_scales_with_row_count():
    small = recommend_compute(task_shape="tabular", model_family="xgboost", n_rows=5_000, n_features=8)
    large = recommend_compute(task_shape="tabular", model_family="xgboost", n_rows=500_000, n_features=8)
    assert small.hardware_flavor == "cpu-basic"
    assert large.hardware_flavor == "cpu-upgrade"


def test_tabular_escalates_to_gpu_only_when_genuinely_large():
    rec = recommend_compute(
        task_shape="tabular", model_family="xgboost",
        n_rows=GPU_WORTH_IT_ROWS * 2, n_features=40,
    )
    assert rec.accelerator == "gpu"
    assert rec.kind == "serverless_gpu"


def test_large_tabular_lightgbm_warns_about_missing_opencl():
    """The s6e5 run lost a job to exactly this."""
    rec = recommend_compute(
        task_shape="tabular", model_family="lightgbm",
        n_rows=GPU_WORTH_IT_ROWS * 2, n_features=40,
    )
    assert rec.accelerator == "gpu"
    assert any("OpenCL" in w for w in rec.warnings)


def test_large_tabular_xgboost_gets_no_opencl_warning():
    rec = recommend_compute(
        task_shape="tabular", model_family="xgboost",
        n_rows=GPU_WORTH_IT_ROWS * 2, n_features=40,
    )
    assert not any("OpenCL" in w for w in rec.warnings)


@pytest.mark.parametrize("shape", ["nlp", "cv", "finetune"])
def test_gradient_descent_workloads_go_to_gpu(shape):
    assert recommend_compute(task_shape=shape).accelerator == "gpu"


def test_finetune_uses_the_finetune_kind():
    assert recommend_compute(task_shape="finetune").kind == "finetune"


@pytest.mark.parametrize("family", ["torch", "transformers", "tabm", "resnet", "keras"])
def test_neural_family_forces_gpu_even_when_shape_says_tabular(family):
    """A neural net on tabular data is still gradient descent."""
    rec = recommend_compute(task_shape="tabular", model_family=family, n_rows=1_000, n_features=5)
    assert rec.accelerator == "gpu"


def test_unknown_shape_defaults_to_gpu_and_flags_itself():
    rec = recommend_compute(task_shape="quantum-basket-weaving")
    assert rec.accelerator == "gpu"
    assert rec.warnings


def test_missing_size_is_treated_as_small():
    rec = recommend_compute(task_shape="tabular", model_family="lgbm")
    assert rec.accelerator == "cpu"


def test_render_includes_reason_and_warnings():
    rec = recommend_compute(
        task_shape="tabular", model_family="lightgbm",
        n_rows=GPU_WORTH_IT_ROWS * 2, n_features=40,
    )
    text = rec.render()
    assert "Why:" in text
    assert "WARNING:" in text


# --------------------------------------------------------------- script checks


def test_gbm_script_on_gpu_is_flagged():
    warnings = advise_from_script("import lightgbm as lgb\nlgb.train({})", "serverless_gpu")
    assert any("tree ensembles" in w for w in warnings)


@pytest.mark.parametrize(
    "src",
    [
        'import lightgbm as lgb\np = {"device": "gpu"}',      # dict-literal spelling
        'import lightgbm as lgb\nlgb.train(device="gpu")',    # kwarg spelling
        'import lightgbm as lgb\np = {"device_type": "gpu"}',  # older param name
        "import lightgbm as lgb\nlgb.train(device='cuda')",   # single quotes
    ],
)
def test_lightgbm_gpu_request_is_flagged_in_every_spelling(src):
    warnings = advise_from_script(src, "serverless_gpu")
    assert any("OpenCL" in w for w in warnings), f"missed: {src}"


def test_lightgbm_on_cpu_is_not_flagged_for_opencl():
    warnings = advise_from_script('import lightgbm as lgb\np = {"device": "cpu"}', "serverless_gpu")
    assert not any("OpenCL" in w for w in warnings)


def test_neural_script_on_cpu_is_flagged():
    warnings = advise_from_script(
        "import torch\nm = torch.nn.Linear(4, 4)", "serverless", hardware_flavor="cpu-upgrade"
    )
    assert any("GPU-bound" in w for w in warnings)


def test_correct_pairings_produce_no_warnings():
    assert advise_from_script("import torch", "serverless_gpu") == []
    assert advise_from_script(
        "import lightgbm as lgb", "serverless", hardware_flavor="cpu-basic"
    ) == []


def test_mixed_gbm_and_neural_script_on_gpu_is_left_alone():
    """A script that also trains a net legitimately wants the GPU."""
    src = "import lightgbm as lgb\nimport torch\n"
    assert advise_from_script(src, "serverless_gpu") == []


def test_gpu_node_type_is_detected_without_a_flavor():
    warnings = advise_from_script(
        "import lightgbm as lgb", "script", hardware_flavor=None, node_type_id="g5.4xlarge"
    )
    assert any("tree ensembles" in w for w in warnings)


def test_cpu_node_type_is_not_treated_as_gpu():
    assert advise_from_script(
        "import lightgbm as lgb", "script", hardware_flavor=None, node_type_id="m5.2xlarge"
    ) == []


def test_empty_script_is_not_second_guessed():
    assert advise_from_script(None, "serverless_gpu") == []
    assert advise_from_script("", "serverless_gpu") == []


# ---- runtime budget -------------------------------------------------------------
#
# From the S6E8 run: the agent smoke-tested at 87s on 20k rows / 100 trees, then submitted
# the full run at 34x rows and 20x trees with a 45-minute timeout. Killed at 45.4 min in
# fold 1 of 5. It had both numbers and never multiplied them.


def test_the_s6e8_timeout_is_refused_with_the_arithmetic_shown():
    warnings, refusal = check_runtime_budget(45 * 60, {"measured_seconds": 87, "scale_factor": 680})
    assert warnings == []
    assert refusal is not None
    assert "16h" in refusal and "45m" in refusal


def test_a_timeout_that_covers_the_estimate_passes():
    warnings, refusal = check_runtime_budget(21 * 3600, {"measured_seconds": 87, "scale_factor": 680})
    assert refusal is None
    assert warnings == []


def test_the_headroom_factor_is_enforced_not_just_the_bare_estimate():
    """A run that fits with 0% to spare is a run that dies in its last fold."""
    _, refusal = check_runtime_budget(3600, {"estimated_seconds": 3400})
    assert refusal is not None and "1.25x" in refusal


def test_an_asserted_estimate_is_accepted_without_a_measurement():
    _, refusal = check_runtime_budget(6 * 3600, {"estimated_seconds": 4 * 3600})
    assert refusal is None


def test_measured_seconds_alone_means_scale_one():
    _, refusal = check_runtime_budget(600, {"measured_seconds": 87})
    assert refusal is None


def test_a_short_timeout_without_an_estimate_warns_but_does_not_block():
    warnings, refusal = check_runtime_budget(45 * 60, None)
    assert refusal is None
    assert any("runtime_estimate" in w for w in warnings)


def test_a_long_timeout_without_an_estimate_is_left_alone():
    assert check_runtime_budget(6 * 3600, None) == ([], None)


def test_no_timeout_with_an_estimate_warns_about_the_workspace_default():
    warnings, refusal = check_runtime_budget(0, {"estimated_seconds": 4 * 3600})
    assert refusal is None
    assert any("No timeout set" in w for w in warnings)


def test_an_unusable_estimate_is_refused_rather_than_ignored():
    _, refusal = check_runtime_budget(3600, {"scale_factor": 10})
    assert refusal is not None and "estimated_seconds" in refusal


def test_a_nonsense_estimate_is_refused():
    _, refusal = check_runtime_budget(3600, {"estimated_seconds": 0})
    assert refusal is not None
