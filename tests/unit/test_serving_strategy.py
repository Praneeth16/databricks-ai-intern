"""Unit tests for the Custom LLM Serving deployment-strategy math.

Load-bearing assertions reproduce the FE reference verdicts:
- Qwen3-4B (fp16) fits a single A10 (GPU_MEDIUM, TP=1).
- Qwen3.5-27B-FP8 (~27 GB) does NOT fit one A10 — it escalates to a single H100,
  or falls back to TP=4 across A10×4 (MULTIGPU_MEDIUM) when no single GPU fits.
"""

from __future__ import annotations

import pytest

from agent.core import serving_strategy as ss
from agent.core.serving_strategy import ModelFacts, feasible_configs, render_entrypoint

QWEN3_4B = ModelFacts(params_billions=4.0, num_layers=36, hidden_size=2560, kv_dim=1024,
                      native_precision="fp16")
QWEN35_27B_FP8 = ModelFacts(params_billions=27.0, num_layers=64, hidden_size=5120, kv_dim=1024,
                            native_precision="fp8")


# --- VRAM math ---------------------------------------------------------------

def test_4b_fp16_fits_single_a10():
    est = ss.estimate_vram_per_gpu(
        params_billions=4.0, bytes_per_param=2.0, num_layers=36, hidden_size=2560,
        kv_dim=1024, max_model_len=16384, sizing_concurrency=1, tensor_parallel_size=1,
    )
    assert 8.0 <= est < 20.4  # ~8 GB weights + one-sequence KV, under A10 usable


def test_27b_fp8_overflows_one_a10_but_tp4_fits():
    kw = dict(params_billions=27.0, bytes_per_param=1.0, num_layers=64, hidden_size=5120,
              kv_dim=1024, max_model_len=8192, sizing_concurrency=1)
    usable = 24 * 0.88
    assert ss.estimate_vram_per_gpu(tensor_parallel_size=1, **kw) > usable  # ~27 GB — no fit
    assert ss.estimate_vram_per_gpu(tensor_parallel_size=4, **kw) < usable  # TP=4 fits


def test_vram_scales_inversely_with_tp():
    kw = dict(params_billions=27.0, bytes_per_param=1.0, num_layers=64, hidden_size=5120,
              kv_dim=1024, max_model_len=8192, sizing_concurrency=1)
    assert ss.estimate_vram_per_gpu(tensor_parallel_size=1, **kw) > \
           ss.estimate_vram_per_gpu(tensor_parallel_size=4, **kw)


# --- feasibility / tier selection -------------------------------------------

def test_4b_feasible_on_single_a10_tp1():
    cfgs = feasible_configs(model=QWEN3_4B, available_workload_types=["GPU_MEDIUM"],
                            accuracy_budget="max")
    assert cfgs and all(c.tensor_parallel_size == 1 and c.fits for c in cfgs)
    assert {c.precision for c in cfgs} <= {"bf16", "fp16"}
    assert all(c.quant_source == "native" for c in cfgs)


def test_27b_fp8_needs_multigpu_not_single():
    single = feasible_configs(model=QWEN35_27B_FP8, available_workload_types=["GPU_MEDIUM"],
                              accuracy_budget="balanced")
    multi = feasible_configs(model=QWEN35_27B_FP8, available_workload_types=["MULTIGPU_MEDIUM"],
                             accuracy_budget="balanced")
    assert single == []                                  # nothing fits one A10
    assert any(c.tensor_parallel_size == 4 and c.precision == "fp8" for c in multi)


def test_native_fp8_not_offered_as_bf16():
    cfgs = feasible_configs(model=QWEN35_27B_FP8, available_workload_types=["GPU_XLARGE"],
                            accuracy_budget="max")
    assert cfgs == []  # max = bf16/fp16 only; can't upcast an fp8 checkpoint


def test_fp8_excluded_on_turing_t4():
    tiny = ModelFacts(params_billions=1.0, num_layers=24, hidden_size=2048, kv_dim=512,
                      native_precision="fp16")
    cfgs = feasible_configs(model=tiny, available_workload_types=["GPU_SMALL"],
                            accuracy_budget="aggressive")
    assert cfgs and all(c.precision == "fp16" for c in cfgs)  # bf16/fp8 need Ampere+; only fp16 on T4


# --- P0: quantized formats require an artifact, not a free flag --------------

def test_awq_requires_an_existing_artifact():
    # An fp16 checkpoint cannot be planned as AWQ at serve time without the artifact.
    without = feasible_configs(model=QWEN3_4B, available_workload_types=["GPU_MEDIUM"],
                               accuracy_budget="aggressive")
    assert all(c.precision != "awq" for c in without)

    have_artifact = ModelFacts(params_billions=4.0, num_layers=36, hidden_size=2560,
                               kv_dim=1024, native_precision="fp16",
                               available_quant_formats=("awq",))
    with_art = feasible_configs(model=have_artifact, available_workload_types=["GPU_MEDIUM"],
                                accuracy_budget="aggressive")
    awq = [c for c in with_art if c.precision == "awq"]
    assert awq and awq[0].quant_source == "artifact"


def test_awq_offered_when_build_allowed():
    cfgs = feasible_configs(model=QWEN3_4B, available_workload_types=["GPU_MEDIUM"],
                            accuracy_budget="aggressive", allow_quantization_build=True)
    awq = [c for c in cfgs if c.precision == "awq"]
    assert awq and awq[0].quant_source == "build"
    assert any("build step" in n for n in awq[0].notes)


def test_fp8_is_online_for_fp16_checkpoint():
    # fp8 IS a free choice: vLLM quantizes a 16-bit checkpoint online at load.
    cfgs = feasible_configs(model=QWEN3_4B, available_workload_types=["GPU_MEDIUM"],
                            accuracy_budget="balanced")
    fp8 = [c for c in cfgs if c.precision == "fp8"]
    assert fp8 and fp8[0].quant_source == "online"


# --- P1a: explicit empty capability means nothing fits -----------------------

def test_empty_capability_yields_no_configs():
    assert feasible_configs(model=QWEN3_4B, available_workload_types=[]) == []


# --- P1c: T4 default-excluded; size-driven escalation; objectives ------------

def test_t4_excluded_by_default_but_optin():
    tiny = ModelFacts(params_billions=1.0, num_layers=24, hidden_size=2048, kv_dim=512,
                      native_precision="fp16")
    default = feasible_configs(model=tiny, accuracy_budget="max")  # available=None
    assert all(c.workload_type != "GPU_SMALL" for c in default)
    optin = feasible_configs(model=tiny, accuracy_budget="max", allow_small_gpu=True)
    assert any(c.workload_type == "GPU_SMALL" for c in optin)


def test_27b_prefers_h100_single_gpu_over_a10_tp4():
    cfgs = feasible_configs(model=QWEN35_27B_FP8,
                            available_workload_types=["MULTIGPU_MEDIUM", "GPU_XLARGE"],
                            accuracy_budget="balanced")
    assert cfgs[0].workload_type == "GPU_XLARGE" and cfgs[0].tensor_parallel_size == 1
    assert any(c.workload_type == "MULTIGPU_MEDIUM" for c in cfgs)  # offered as fallback


def test_small_model_prefers_cheaper_a10_over_h100():
    cfgs = feasible_configs(model=QWEN3_4B,
                            available_workload_types=["GPU_MEDIUM", "GPU_XLARGE"],
                            accuracy_budget="max")
    assert cfgs[0].workload_type == "GPU_MEDIUM" and cfgs[0].tensor_parallel_size == 1


def test_cost_first_objective_picks_cheapest():
    cfgs = feasible_configs(model=QWEN35_27B_FP8,
                            available_workload_types=["MULTIGPU_MEDIUM", "GPU_XLARGE"],
                            accuracy_budget="balanced", objective="cost_first")
    # A10×4 (4×$1.7=$6.8) is cheaper than one H100 ($9).
    assert cfgs[0].est_cost_per_hr_usd == min(c.est_cost_per_hr_usd for c in cfgs)
    assert cfgs[0].workload_type == "MULTIGPU_MEDIUM"


# --- TP head divisibility ----------------------------------------------------

def test_tp_blocked_when_kv_heads_not_divisible():
    # 27B-FP8 with 6 KV heads: 6 % 4 != 0 -> A10×4 (TP=4) is invalid, dropped.
    indivisible = ModelFacts(params_billions=27.0, num_layers=64, hidden_size=5120,
                             kv_dim=1024, num_kv_heads=6, native_precision="fp8")
    cfgs = feasible_configs(model=indivisible, available_workload_types=["MULTIGPU_MEDIUM"],
                            accuracy_budget="balanced")
    assert cfgs == []


# --- P1b: serving capacity, max_num_seqs clamp -------------------------------

def test_max_num_seqs_clamped_by_kv_budget():
    # A long context + a big request should clamp max_num_seqs below the request.
    cfgs = feasible_configs(model=QWEN3_4B, available_workload_types=["GPU_MEDIUM"],
                            accuracy_budget="max", max_model_len=16384, max_num_seqs=256)
    c = cfgs[0]
    assert c.max_num_seqs == c.est_max_concurrent_seqs < 256
    assert any("clamped" in n for n in c.notes)


def test_provisioned_concurrency_snaps_to_multiple_of_four():
    cfgs = feasible_configs(model=QWEN3_4B, available_workload_types=["GPU_MEDIUM"],
                            provisioned_concurrency=5)
    assert all(c.provisioned_concurrency == 8 for c in cfgs)
    assert ss._snap_to_multiple_of_four(1) == 4
    assert ss._snap_to_multiple_of_four(9) == 12


def test_bad_accuracy_budget_and_objective_raise():
    with pytest.raises(ValueError):
        feasible_configs(model=QWEN3_4B, accuracy_budget="nonsense")
    with pytest.raises(ValueError):
        feasible_configs(model=QWEN3_4B, objective="nonsense")


# --- entrypoint rendering ----------------------------------------------------

def _cfg(**o):
    base = dict(workload_type="GPU_MEDIUM", tensor_parallel_size=1, precision="fp16",
                quant_source="native", max_model_len=16384, gpu_memory_utilization=0.85,
                max_num_seqs=64, provisioned_concurrency=4, est_vram_per_gpu_gb=10.0,
                vram_headroom_gb=10.0, est_max_concurrent_seqs=8, quality_delta_pct=0.0,
                est_cost_per_hr_usd=1.7, fits=True)
    base.update(o)
    return ss.ServingConfig(**base)


def test_entrypoint_always_uninstalls_opencv_even_for_plain_fp16():
    # FIPS-enabled Databricks GPU runtimes crash on opencv during vLLM model
    # inspection regardless of precision (live-confirmed), so the uninstall is
    # unconditional. Single-GPU fp16 still skips the TP fork.
    ep = render_entrypoint(_cfg(), artifacts_path="qwen3", served_model_name="qwen")
    assert ep.startswith("bash -lc ")
    assert "opencv-python" in ep
    assert "VLLM_WORKER_MULTIPROC_METHOD=fork" not in ep   # TP=1
    assert "--tensor-parallel-size" not in ep and "--quantization" not in ep
    assert "--dtype float16" in ep and "exec python -u -m vllm" in ep


def test_entrypoint_bakes_tp_opencv_fork_for_native_fp8_tp4():
    cfg = _cfg(precision="fp8", quant_source="native", tensor_parallel_size=4,
               workload_type="MULTIGPU_MEDIUM")
    ep = render_entrypoint(cfg, artifacts_path="qwen35_27b_fp8", served_model_name="qwen")
    assert ep.startswith("bash -lc ")
    assert "pip uninstall -y opencv-python-headless opencv-python" in ep
    assert "VLLM_WORKER_MULTIPROC_METHOD=fork" in ep
    assert "--tensor-parallel-size 4" in ep and "--dtype bfloat16" in ep
    assert "--quantization" not in ep  # native fp8 is auto-detected, no flag
    assert "exec python" in ep


def test_entrypoint_emits_quantization_flag_only_for_online_fp8():
    online = _cfg(precision="fp8", quant_source="online")
    ep = render_entrypoint(online, artifacts_path="m", served_model_name="x")
    assert "--quantization fp8" in ep        # online path needs the flag
    artifact = _cfg(precision="awq", quant_source="artifact")
    ep2 = render_entrypoint(artifact, artifacts_path="m", served_model_name="x")
    assert "--quantization" not in ep2       # artifact is auto-detected
    assert "bash -lc" in ep2                  # but still needs the opencv fix


def test_entrypoint_bf16_tp4_has_both_fork_and_opencv():
    cfg = _cfg(precision="bf16", quant_source="native", tensor_parallel_size=4,
               workload_type="MULTIGPU_MEDIUM")
    ep = render_entrypoint(cfg, artifacts_path="m", served_model_name="x")
    assert "VLLM_WORKER_MULTIPROC_METHOD=fork" in ep
    assert "opencv-python" in ep            # opencv fix is now unconditional


def test_entrypoint_shell_quotes_paths():
    # Path + served name carry a space; they must survive shell-quoting. (The
    # whole command is bash -lc wrapped, so the inner quoting is escaped, but the
    # raw substrings remain contiguous.)
    ep = render_entrypoint(_cfg(), artifacts_path="/Volumes/c s/m", served_model_name="a b")
    assert "/Volumes/c s/m" in ep and "a b" in ep
