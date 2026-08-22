"""CPU-vs-GPU compute selection and the script/compute mismatch checks."""

import pytest

from agent.core.compute_advisor import (
    GPU_WORTH_IT_ROWS,
    advise_from_script,
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
