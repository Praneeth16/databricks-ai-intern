"""Deployment-strategy math for Databricks Custom LLM Serving.

The agent must decide *how* to host a (fine-tuned) model on Custom LLM Serving:
which GPU tier, single-GPU or tensor-parallel across several, what precision /
quantization, and the fixed provisioned concurrency. The public docs describe a
``workload_size``/``scale_to_zero`` autoscaling path that the **entrypoint-based**
deploy actually rejects — the real surface is a Serverless Optimized Deployment
whose vLLM ``entrypoint`` command is the only thing Serving runs.

This module is the *deterministic mechanism* half of that decision (mirrors how
``research_loop`` owns control flow while the LLM supplies content): pure VRAM /
tensor-parallel / precision math plus the entrypoint renderer. It does NOT pick
the winner — ``feasible_configs`` returns the set of configs that *fit*, ranked
by an explicit ``objective``, each annotated with quality-delta band, cost,
headroom, and serving capacity, so the agent can choose per the team's accuracy /
latency / cost priorities.

A key correctness boundary (see ``Precision`` / ``quant_source``): a target
precision is only offered when it is actually achievable for the given checkpoint
— served as-is, produced online by vLLM at load (fp8 dynamic), backed by an
existing pre-quantized artifact, or explicitly built. vLLM does NOT turn an fp16
checkpoint into AWQ/GPTQ/int8 at serve time.

No Databricks contact, no I/O — fully unit-testable offline. Validated against
two known points (see ``tests``): Qwen3-4B fits one A10 (TP=1); Qwen3.5-27B-FP8
(~27 GB) does not fit one A10 and escalates to a single H100, or falls back to
TP=4 across A10×4 when no single GPU is big enough.
"""

from __future__ import annotations

import logging
import shlex
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# GPU micro-architecture rank — higher = newer. Precision support gates on this.
# bf16 needs Ampere+; weight-only/online fp8 serves on Ampere+ via Marlin (hardware
# fp8 *compute* would need Ada/Hopper, but that's a throughput detail, not a
# serveability gate). Only ranks we actually serve on.
_ARCH = {"turing": 1, "ampere": 2, "ada": 3, "hopper": 4}


@dataclass(frozen=True)
class GpuTier:
    workload_type: str   # Custom LLM Serving workload_type
    gpu: str             # marketing name
    gpu_count: int       # GPUs in the tier (== tensor_parallel_size when split across all)
    vram_gb: int         # per-GPU VRAM
    arch: str            # key into _ARCH
    cost_per_gpu_hr: float  # rough always-on USD/hr per GPU (no scale-to-zero) — order-of-magnitude


# Confirmed serving workload_types (FE reference + docs). MULTIGPU_* enables
# tensor parallelism for models that don't fit one GPU. GPU_SMALL (T4) is below
# the LLM-serving baseline (16 GB, no bf16) and is opt-in only. Costs are rough.
GPU_TIERS: dict[str, GpuTier] = {
    "GPU_SMALL": GpuTier("GPU_SMALL", "T4", 1, 16, "turing", 1.0),
    "GPU_MEDIUM": GpuTier("GPU_MEDIUM", "A10", 1, 24, "ampere", 1.7),
    "MULTIGPU_MEDIUM": GpuTier("MULTIGPU_MEDIUM", "A10", 4, 24, "ampere", 1.7),
    "GPU_XLARGE": GpuTier("GPU_XLARGE", "H100", 1, 80, "hopper", 9.0),
}


@dataclass(frozen=True)
class Precision:
    name: str
    bytes_per_param: float   # weight footprint per parameter in VRAM
    min_arch: str            # minimum GPU arch that supports it
    quality_delta_pct: float  # rough accuracy loss vs a 16-bit baseline
    vllm_dtype: str          # --dtype (compute dtype)
    is_quant: bool           # True for sub-16-bit quantized formats
    online_capable: bool     # vLLM can produce it at load from a 16-bit checkpoint (no artifact)
    online_quant_flag: str | None  # --quantization value for the online path (None otherwise)


# Ordered best-accuracy-first. bf16/fp16 are served as-is (a --dtype choice, not a
# quantization). fp8 can be produced online by vLLM (`--quantization fp8`) from a
# 16-bit checkpoint, OR served from a native FP8 checkpoint (auto-detected, no flag).
# int8/AWQ/GPTQ require a matching pre-quantized artifact (or a build step) — vLLM
# auto-detects them from the checkpoint's config and needs no flag.
PRECISIONS: dict[str, Precision] = {
    "bf16": Precision("bf16", 2.0, "ampere", 0.0, "bfloat16", False, False, None),
    "fp16": Precision("fp16", 2.0, "turing", 0.0, "float16", False, False, None),
    "fp8": Precision("fp8", 1.0, "ampere", 0.8, "bfloat16", True, True, "fp8"),
    "int8": Precision("int8", 1.0, "turing", 1.2, "auto", True, False, None),
    "awq": Precision("awq", 0.5, "turing", 2.5, "float16", True, False, None),
    "gptq": Precision("gptq", 0.5, "turing", 2.5, "float16", True, False, None),
}

# Accuracy budget -> allowed precision set (the team's tolerance, not hardware).
ACCURACY_BUDGETS: dict[str, tuple[str, ...]] = {
    "max": ("bf16", "fp16"),
    "balanced": ("bf16", "fp16", "fp8", "int8"),
    "aggressive": ("bf16", "fp16", "fp8", "int8", "awq", "gptq"),
}

# objective -> sort key over fitting configs. "balanced" is size-driven: smallest
# GPU that fits (TP asc, so single-GPU beats multi-GPU TP), then accuracy, then cost.
_OBJECTIVES = {
    "balanced": lambda c: (c.tensor_parallel_size, c.quality_delta_pct, c.est_cost_per_hr_usd, -c.vram_headroom_gb),
    "cost_first": lambda c: (c.est_cost_per_hr_usd, c.tensor_parallel_size, c.quality_delta_pct, -c.vram_headroom_gb),
    "accuracy_first": lambda c: (c.quality_delta_pct, c.tensor_parallel_size, c.est_cost_per_hr_usd, -c.vram_headroom_gb),
    "latency_first": lambda c: (c.tensor_parallel_size, -c.est_max_concurrent_seqs, c.quality_delta_pct, c.est_cost_per_hr_usd),
}


@dataclass(frozen=True)
class ModelFacts:
    params_billions: float
    num_layers: int
    hidden_size: int
    kv_dim: int | None = None             # num_kv_heads * head_dim; None => assume == hidden_size (no GQA, conservative)
    num_attention_heads: int | None = None  # for TP head-divisibility gate (optional)
    num_kv_heads: int | None = None          # for TP head-divisibility gate (optional)
    native_precision: str = "fp16"           # the checkpoint's on-disk format
    available_quant_formats: tuple[str, ...] = ()  # pre-quantized artifacts that already exist


@dataclass(frozen=True)
class ServingConfig:
    workload_type: str
    tensor_parallel_size: int
    precision: str
    quant_source: str                # "native" | "online" | "artifact" | "build"
    max_model_len: int
    gpu_memory_utilization: float
    max_num_seqs: int                # clamped to the KV budget
    provisioned_concurrency: int     # fixed, multiple of 4
    est_vram_per_gpu_gb: float       # startup working set (weights + one sequence)
    vram_headroom_gb: float          # usable VRAM - startup estimate; >=0 means it loads
    est_max_concurrent_seqs: int     # how many max-len sequences the KV budget holds
    quality_delta_pct: float
    est_cost_per_hr_usd: float
    fits: bool
    notes: tuple[str, ...] = field(default_factory=tuple)


def estimate_vram_per_gpu(
    *,
    params_billions: float,
    bytes_per_param: float,
    num_layers: int,
    hidden_size: int,
    max_model_len: int,
    sizing_concurrency: int,
    tensor_parallel_size: int = 1,
    kv_dim: int | None = None,
    kv_bytes: int = 2,
    overhead: float = 1.18,
) -> float:
    """Estimate per-GPU VRAM (GB) for a vLLM deployment.

    Weights and KV cache both shard across ``tensor_parallel_size`` GPUs. The
    ``overhead`` factor (~18%) covers activations, CUDA context, and allocator
    fragmentation. ``kv_dim`` is ``num_kv_heads * head_dim`` (much smaller than
    ``hidden_size`` under GQA); when unknown we assume no GQA (== hidden_size),
    a conservative upper bound. This is a sizing heuristic, not an allocator;
    the weights term dominates the fit verdict. (Uses decimal GB / 1e9 — slightly
    optimistic vs GiB, covered by the overhead margin.)
    """
    tp = max(1, tensor_parallel_size)
    return _vram_breakdown(
        params_billions=params_billions, bytes_per_param=bytes_per_param,
        num_layers=num_layers, hidden_size=hidden_size, max_model_len=max_model_len,
        sizing_concurrency=sizing_concurrency, tp=tp, kv_dim=kv_dim,
        kv_bytes=kv_bytes, overhead=overhead,
    )[0]


def _vram_breakdown(
    *, params_billions, bytes_per_param, num_layers, hidden_size, max_model_len,
    sizing_concurrency, tp, kv_dim, kv_bytes, overhead,
) -> tuple[float, float, float]:
    """Return (total_per_gpu_gb, weights_per_gpu_gb, kv_per_seq_per_gpu_gb)."""
    kv_dim = kv_dim if kv_dim is not None else hidden_size
    weights_gb = params_billions * bytes_per_param / tp  # 1B params * 2 B/param = 2 GB
    kv_per_token_bytes = 2 * num_layers * kv_dim * kv_bytes  # 2 = K and V
    kv_per_seq_gb = kv_per_token_bytes * max_model_len / 1e9 / tp
    total = (weights_gb + kv_per_seq_gb * sizing_concurrency) * overhead
    return total, weights_gb * overhead, kv_per_seq_gb * overhead


def feasible_configs(
    *,
    model: ModelFacts,
    available_workload_types: list[str] | None = None,
    accuracy_budget: str = "balanced",
    objective: str = "balanced",
    max_model_len: int = 8192,
    max_num_seqs: int = 64,
    provisioned_concurrency: int = 4,
    gpu_memory_utilization: float = 0.88,
    allow_small_gpu: bool = False,
    allow_quantization_build: bool = False,
) -> list[ServingConfig]:
    """Return serving configs that fit, ranked by ``objective``.

    For every (precision allowed by ``accuracy_budget`` AND achievable for this
    checkpoint) × (available GPU tier), check arch compatibility, TP head
    divisibility, and per-GPU fit after the tier's tensor-parallel split. A
    precision is *achievable* when it is the native format, served as-is (16-bit),
    produced online by vLLM (fp8), backed by an existing artifact in
    ``model.available_quant_formats``, or (when ``allow_quantization_build``) built.

    Tier selection is size-driven under the default ``objective="balanced"``:
    smallest GPU that fits stays put; too big for an A10 escalates to a single
    H100; too big for one H100 falls back to TP across A10×4. H100 is the
    escalation step, not a default. T4 (GPU_SMALL) is below the LLM baseline and
    excluded unless ``allow_small_gpu`` or named explicitly.

    Passing an explicit ``available_workload_types`` (incl. ``[]``) is honored
    verbatim; ``None`` means "use the sensible default set". ``fits=False`` rows
    are dropped. Every config carries quality-delta / cost / headroom / capacity
    so the caller can re-sort for a different objective.
    """
    if accuracy_budget not in ACCURACY_BUDGETS:
        raise ValueError(
            f"unknown accuracy_budget {accuracy_budget!r}; expected one of {list(ACCURACY_BUDGETS)}"
        )
    if objective not in _OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}; expected one of {list(_OBJECTIVES)}")

    if available_workload_types is None:
        tiers = [t for t in GPU_TIERS if t != "GPU_SMALL" or allow_small_gpu]
    else:
        tiers = available_workload_types  # honor exactly, including []

    pc = _snap_to_multiple_of_four(provisioned_concurrency)
    native_bytes = _bytes_for(model.native_precision)

    out: list[ServingConfig] = []
    for tier_name in tiers:
        tier = GPU_TIERS.get(tier_name)
        if tier is None:
            logger.debug("unknown workload_type %s — skipping", tier_name)
            continue
        tp = tier.gpu_count

        # TP requires the (kv) heads to divide evenly across GPUs.
        if tp > 1 and not _tp_heads_divide(model, tp):
            continue

        for prec_name in ACCURACY_BUDGETS[accuracy_budget]:
            prec = PRECISIONS[prec_name]
            if _ARCH[tier.arch] < _ARCH[prec.min_arch]:
                continue  # GPU too old for this precision

            source = _quant_source(prec, model, native_bytes, allow_quantization_build)
            if source is None:
                continue  # not achievable for this checkpoint

            total, weights_oh, kv_per_seq_oh = _vram_breakdown(
                params_billions=model.params_billions, bytes_per_param=prec.bytes_per_param,
                num_layers=model.num_layers, hidden_size=model.hidden_size,
                max_model_len=max_model_len, sizing_concurrency=1, tp=tp,
                kv_dim=model.kv_dim, kv_bytes=2, overhead=1.18,
            )
            usable = tier.vram_gb * gpu_memory_utilization
            headroom = usable - total
            if headroom < 0:
                continue  # can't even load one sequence

            # KV budget left after weights -> how many max-len sequences we can batch.
            kv_budget = usable - weights_oh
            cap = max(1, int(kv_budget / kv_per_seq_oh)) if kv_per_seq_oh > 0 else max_num_seqs
            seqs = min(max_num_seqs, cap)

            notes: list[str] = []
            if tp > 1:
                notes.append(f"tensor-parallel across {tp}× {tier.gpu}")
            if source == "online":
                notes.append(f"{prec_name} produced online by vLLM from the {model.native_precision} checkpoint")
            elif source == "artifact":
                notes.append(f"requires the existing {prec_name} pre-quantized artifact")
            elif source == "build":
                notes.append(f"requires a {prec_name} quantization build step before deploy")
            if prec.is_quant:
                notes.append("opencv FIPS uninstall required in entrypoint")
            if seqs < max_num_seqs:
                notes.append(f"max_num_seqs clamped {max_num_seqs}→{seqs} by KV budget")

            out.append(
                ServingConfig(
                    workload_type=tier.workload_type,
                    tensor_parallel_size=tp,
                    precision=prec_name,
                    quant_source=source,
                    max_model_len=max_model_len,
                    gpu_memory_utilization=gpu_memory_utilization,
                    max_num_seqs=seqs,
                    provisioned_concurrency=pc,
                    est_vram_per_gpu_gb=round(total, 2),
                    vram_headroom_gb=round(headroom, 2),
                    est_max_concurrent_seqs=cap,
                    quality_delta_pct=(0.0 if not prec.is_quant else prec.quality_delta_pct),
                    est_cost_per_hr_usd=round(tier.cost_per_gpu_hr * tier.gpu_count, 2),
                    fits=True,
                    notes=tuple(notes),
                )
            )

    out.sort(key=_OBJECTIVES[objective])
    return out


def render_entrypoint(
    cfg: ServingConfig,
    *,
    artifacts_path: str,
    served_model_name: str,
    serving_port: int = 8080,
) -> str:
    """Render the exact command Custom LLM Serving runs for ``cfg``.

    Bakes in the FE-discovered fixes (see plan.md Phase 7 gotchas): the opencv
    FIPS uninstall on the quantized path, ``VLLM_WORKER_MULTIPROC_METHOD=fork``
    for tensor-parallel workers, and a ``bash -lc`` wrap so those run before
    ``exec python``. ``--quantization`` is emitted ONLY for the online path
    (e.g. fp8 dynamic) — native and artifact-backed quantized checkpoints are
    auto-detected by vLLM from their config. Paths are shell-quoted.
    """
    prec = PRECISIONS[cfg.precision]
    base = [
        "python", "-u", "-m", "vllm.entrypoints.openai.api_server",
        "--model", shlex.quote(artifacts_path),
        "--served-model-name", shlex.quote(served_model_name),
        "--host", "0.0.0.0", "--port", str(serving_port),
        "--dtype", prec.vllm_dtype,
        "--max-model-len", str(cfg.max_model_len),
        "--gpu-memory-utilization", str(cfg.gpu_memory_utilization),
        "--max-num-seqs", str(cfg.max_num_seqs),
    ]
    if cfg.tensor_parallel_size > 1:
        base += ["--tensor-parallel-size", str(cfg.tensor_parallel_size)]
    if cfg.quant_source == "online" and prec.online_quant_flag:
        base += ["--quantization", prec.online_quant_flag]
    cmd = " ".join(base)

    # opencv's bundled libcrypto aborts a FIPS self-test (`crypto/fips/fips.c:154:
    # FATAL FIPS SELFTEST FAILURE`) when vLLM imports it during model inspection,
    # crashing the server on FIPS-enabled Databricks GPU runtimes REGARDLESS of
    # precision (live-confirmed on a bf16 model — not just the quantized/gguf path).
    # Text inference doesn't need cv2, so always uninstall it. Vision/multimodal
    # models would need an opt-out.
    prefix = "python -m pip uninstall -y opencv-python-headless opencv-python >/dev/null 2>&1 || true; "
    if cfg.tensor_parallel_size > 1:
        # fork avoids a libstdc++ CXXABI mismatch in the TP>1 worker subprocess.
        prefix += "VLLM_WORKER_MULTIPROC_METHOD=fork "
    return "bash -lc " + shlex.quote(f"{prefix}exec {cmd}")


def _quant_source(
    prec: Precision, model: ModelFacts, native_bytes: float, allow_build: bool
) -> str | None:
    """How (if at all) this precision is achievable for the checkpoint.

    Returns "native" | "online" | "artifact" | "build", or None if unachievable.
    """
    if not prec.is_quant:
        # 16-bit served as-is — only valid if the checkpoint is itself >= 16-bit
        # (can't upcast a quantized checkpoint back to bf16/fp16).
        return "native" if native_bytes >= 2.0 else None
    if prec.name == model.native_precision:
        return "native"
    if prec.name in model.available_quant_formats:
        return "artifact"
    if prec.bytes_per_param > native_bytes:
        return None  # would need to "upcast" a smaller checkpoint
    if prec.online_capable:
        return "online"
    if allow_build:
        return "build"
    return None


def _tp_heads_divide(model: ModelFacts, tp: int) -> bool:
    """vLLM tensor parallelism requires attention/KV heads to divide across GPUs.

    Unknown head counts -> permissive (assume divisible); the sizing layer can't
    invent architecture it wasn't given.
    """
    for heads in (model.num_attention_heads, model.num_kv_heads):
        if heads is not None and heads % tp != 0:
            return False
    return True


def _bytes_for(precision_name: str) -> float:
    p = PRECISIONS.get(precision_name)
    return p.bytes_per_param if p else 2.0


def _snap_to_multiple_of_four(n: int) -> int:
    """Custom LLM Serving requires fixed provisioned concurrency, multiple of 4 (min 4)."""
    if n <= 4:
        return 4
    return ((n + 3) // 4) * 4
