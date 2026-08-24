"""Pick CPU or GPU compute for a training job, and say why.

The agent had no notion of this: it reached for `serverless_gpu` because GPU reads as
"the fast one". For the workload this repo spends most of its time on — gradient-boosted
trees on a few hundred thousand tabular rows — GPU is slower *and* more expensive, and on
Databricks serverless it can fail outright.

Two facts drive the rules below, both from this repo's own history rather than folklore:

1. **LightGBM cannot use GPU on the serverless GPU image.** Its GPU build needs OpenCL,
   which that image doesn't ship. The s6e5 run burned a whole job discovering this
   (documented in `examples/kaggle-f1-pitstops-s6e5/README.md`). XGBoost is different —
   it uses CUDA directly via `device="cuda"` and does work there.
2. **GBM histogram building on a few hundred thousand rows does not saturate a GPU.**
   The kernel launch and host-to-device transfer costs dominate, so multicore CPU
   `hist` wins until the data is roughly an order of magnitude larger.

Pure functions, no I/O — so this is unit-testable without a workspace.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Tree-ensemble libraries: CPU-first.
GBM_FAMILIES = frozenset({"lgbm", "lightgbm", "xgboost", "xgb", "catboost", "cat", "gbm", "sklearn"})
# Anything gradient-descent on dense tensors: GPU-first.
NEURAL_FAMILIES = frozenset(
    {"torch", "pytorch", "transformers", "nn", "mlp", "resnet", "tabm", "tabnet",
     "ft-transformer", "keras", "tensorflow", "jax"}
)

# Above roughly this much data, GBM GPU training starts to pay for its transfer cost.
# Deliberately conservative: being wrong toward CPU costs some wall-clock, being wrong
# toward GPU costs money and, for LightGBM, a failed job.
GPU_WORTH_IT_ROWS = 10_000_000
GPU_WORTH_IT_CELLS = 200_000_000

_CPU_SMALL, _CPU_LARGE = "cpu-basic", "cpu-upgrade"
_CPU_UPGRADE_ROWS = 200_000


@dataclass(frozen=True)
class ComputeRecommendation:
    accelerator: str          # "cpu" | "gpu"
    kind: str                 # databricks_jobs `kind`
    hardware_flavor: str | None
    reason: str
    warnings: list[str] = field(default_factory=list)

    def render(self) -> str:
        out = [
            f"Recommended compute: {self.accelerator.upper()} "
            f"(kind={self.kind}"
            + (f", hardware_flavor={self.hardware_flavor}" if self.hardware_flavor else "")
            + ")",
            f"Why: {self.reason}",
        ]
        out += [f"WARNING: {w}" for w in self.warnings]
        return "\n".join(out)


def recommend_compute(
    *,
    task_shape: str,
    model_family: str | None = None,
    n_rows: int | None = None,
    n_features: int | None = None,
) -> ComputeRecommendation:
    """Recommend compute for a training task.

    `task_shape` is one of tabular / nlp / cv / finetune (anything unrecognised is
    treated as neural, which is the safe direction — an unknown workload that wants a
    GPU and gets CPU merely runs slowly).
    """
    shape = (task_shape or "").strip().lower()
    family = (model_family or "").strip().lower()
    warnings: list[str] = []

    if shape == "finetune" or family in NEURAL_FAMILIES or shape in ("nlp", "cv"):
        reason = {
            "finetune": "fine-tuning updates weights by gradient descent, which is GPU-bound",
            "nlp": "transformer text models are GPU-bound",
            "cv": "convolutional and vision-transformer models are GPU-bound",
        }.get(shape, f"{family or 'neural'} models train by gradient descent on dense tensors, which is GPU-bound")
        return ComputeRecommendation(
            accelerator="gpu",
            kind="finetune" if shape == "finetune" else "serverless_gpu",
            hardware_flavor=None,
            reason=reason,
            warnings=warnings,
        )

    if shape == "tabular":
        cells = (n_rows or 0) * (n_features or 0)
        big = (n_rows or 0) >= GPU_WORTH_IT_ROWS or cells >= GPU_WORTH_IT_CELLS
        size_note = _describe_size(n_rows, n_features)

        if not big:
            return ComputeRecommendation(
                accelerator="cpu",
                kind="serverless",
                hardware_flavor=(
                    _CPU_LARGE if (n_rows or 0) >= _CPU_UPGRADE_ROWS else _CPU_SMALL
                ),
                reason=(
                    f"tabular gradient boosting on {size_note} — multicore CPU `hist` beats GPU "
                    f"below ~{GPU_WORTH_IT_ROWS:,} rows, because kernel-launch and host-to-device "
                    "transfer cost more than the histogram work saved. CPU is also far cheaper per DBU"
                ),
                warnings=warnings,
            )

        if family in ("lgbm", "lightgbm"):
            warnings.append(
                "LightGBM GPU requires OpenCL, which the Databricks serverless GPU image does "
                "not ship — a LightGBM GPU job will fail there. Use XGBoost with "
                'device="cuda", or stay on CPU.'
            )
        return ComputeRecommendation(
            accelerator="gpu",
            kind="serverless_gpu",
            hardware_flavor=None,
            reason=(
                f"tabular data is large enough ({size_note}) that GPU histogram construction "
                "amortises its transfer cost"
            ),
            warnings=warnings,
        )

    return ComputeRecommendation(
        accelerator="gpu",
        kind="serverless_gpu",
        hardware_flavor=None,
        reason=f"unrecognised task_shape {task_shape!r}; defaulting to GPU so an accelerated workload is not starved",
        warnings=["task_shape was not one of tabular / nlp / cv / finetune — confirm this is right"],
    )


def _describe_size(n_rows: int | None, n_features: int | None) -> str:
    if n_rows and n_features:
        return f"{n_rows:,} rows x {n_features} features"
    if n_rows:
        return f"{n_rows:,} rows"
    return "an unspecified size (assumed small)"


# --------------------------------------------------------------------------- script check

_GBM_IMPORT = re.compile(r"\b(?:import|from)\s+(lightgbm|xgboost|catboost)\b")
_NEURAL_IMPORT = re.compile(r"\b(?:import|from)\s+(torch|transformers|tensorflow|keras|jax)\b")
# Matches both keyword form (device="gpu") and dict-literal form ("device": "gpu"), so the
# optional quote after the key name is what makes the JSON-style spelling match too.
_LGBM_GPU = re.compile(
    r"""\b(?:device|device_type)["']?\s*[=:]\s*["'](?:gpu|cuda)["']""",
    re.IGNORECASE,
)

_GPU_KINDS = {"serverless_gpu", "finetune"}


def advise_from_script(
    script: str | None,
    kind: str | None,
    hardware_flavor: str | None = None,
    node_type_id: str | None = None,
) -> list[str]:
    """Warn when a submitted job's compute contradicts what the script actually does.

    Called on the job-submission path, and advisory only — it never blocks a run. The
    point is that the agent finds out before spending job-minutes, instead of after.
    """
    if not script:
        return []

    text = script
    gbm = set(_GBM_IMPORT.findall(text))
    neural = set(_NEURAL_IMPORT.findall(text))
    on_gpu = (kind or "") in _GPU_KINDS or _is_gpu_flavor(hardware_flavor, node_type_id)
    warnings: list[str] = []

    if on_gpu and gbm and not neural:
        warnings.append(
            f"This script trains tree ensembles ({', '.join(sorted(gbm))}) with no neural "
            "framework imported, but it was submitted to GPU compute. Tabular gradient "
            "boosting is normally faster and cheaper on multicore CPU — consider "
            'kind="serverless" with hardware_flavor="cpu-upgrade".'
        )
    if on_gpu and "lightgbm" in gbm and _LGBM_GPU.search(text):
        warnings.append(
            "This script asks LightGBM to train on GPU. The Databricks serverless GPU image "
            "ships no OpenCL, so LightGBM's GPU build cannot initialise and the job will "
            'fail. Use XGBoost with device="cuda" for GPU, or run LightGBM on CPU.'
        )
    if not on_gpu and neural and not gbm:
        warnings.append(
            f"This script imports {', '.join(sorted(neural))} but was submitted to CPU "
            'compute. Gradient-descent training is GPU-bound — consider kind="serverless_gpu".'
        )
    return warnings


def check_runtime_budget(
    timeout_seconds: int | None,
    estimate: dict | None,
) -> tuple[list[str], str | None]:
    """Check a job's timeout against what the submitter says the job will cost.

    Returns ``(warnings, refusal)``. A refusal means do not submit: the run cannot finish
    inside its own timeout, so submitting it buys nothing but spent compute.

    This exists because of a specific, recorded failure. On the S6E8 competition the agent
    smoke-tested a training script at 87s on 20k rows and 100 trees, then submitted the
    full run — 34x the rows, 20x the trees — with a 45-minute timeout. It was killed at
    45.4 minutes, in fold 1 of 5, having done nothing wrong except the multiplication. It
    held both numbers and never multiplied them, and no code asked it to.

    `estimate` is the caller's arithmetic, either measured-and-scaled
    (``{"measured_seconds": 87, "scale_factor": 680}``) or asserted outright
    (``{"estimated_seconds": 8700}``). Given one, this checks it. Given none, it can only
    warn — which is why a short timeout on a training job is worth warning about at all.
    """
    warnings: list[str] = []
    timeout = int(timeout_seconds or 0)

    if not estimate:
        if timeout and timeout < 2 * 3600:
            warnings.append(
                f"Timeout is {_dur(timeout)} and no runtime_estimate was given. Full "
                "training runs at competition scale usually exceed 2h. If you have "
                "smoke-tested this script, pass runtime_estimate={\"measured_seconds\": "
                "<smoke wall clock>, \"scale_factor\": <rows ratio x trees ratio x folds>} "
                "and this gets checked instead of guessed."
            )
        return warnings, None

    measured = _as_float(estimate.get("measured_seconds"))
    scale = _as_float(estimate.get("scale_factor"))
    asserted = _as_float(estimate.get("estimated_seconds"))

    if asserted is not None:
        expected = asserted
        basis = f"estimated_seconds={asserted:g}"
    elif measured is not None:
        expected = measured * (scale if scale is not None else 1.0)
        basis = (
            f"measured_seconds={measured:g} x scale_factor={scale:g}"
            if scale is not None
            else f"measured_seconds={measured:g} (no scale_factor, so scale 1x)"
        )
    else:
        return warnings, (
            "runtime_estimate needs either estimated_seconds, or measured_seconds "
            "(optionally with scale_factor). Got: "
            f"{sorted(estimate)}."
        )

    if expected <= 0:
        return warnings, f"runtime_estimate resolves to {expected:g}s, which cannot be right ({basis})."

    if not timeout:
        warnings.append(
            f"No timeout set, and this run is expected to take {_dur(expected)} ({basis}). "
            "The workspace default may be shorter than that."
        )
        return warnings, None

    # 1.25x, not 1.0x: fold times vary, and a run killed at 99% of the work is as useless
    # as one killed at 10%.
    needed = expected * 1.25
    if timeout < needed:
        return warnings, (
            f"This run will not finish inside its timeout. Expected runtime {_dur(expected)} "
            f"({basis}); timeout is {_dur(timeout)}. Raise timeout to at least "
            f"{_dur(needed)} (1.25x expected, since fold times vary), or shrink the run. "
            "A job killed at its timeout returns nothing."
        )
    return warnings, None


def _as_float(v: object) -> float | None:
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _dur(seconds: float) -> str:
    """Human duration. Hours carry their minutes: a "3.0h expected vs 3.0h timeout"
    refusal reads like a contradiction, "3h 1m vs 3h" does not."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 7200:
        return f"{seconds / 60:.0f}m"
    h, m = divmod(int(round(seconds / 60)), 60)
    return f"{h}h" if m == 0 else f"{h}h {m}m"


COMPUTE_ADVICE_TOOL_SPEC: dict[str, object] = {
    "name": "compute_advice",
    "description": (
        "Ask whether a training job should run on CPU or GPU, and get the reason plus any "
        "warnings. Call this BEFORE submitting a databricks_jobs run so you pick the right "
        "`kind` first time. Tabular gradient boosting at normal competition scale belongs on "
        "CPU: it is faster than GPU below roughly 10M rows and much cheaper, and LightGBM "
        "cannot use GPU on the Databricks serverless GPU image at all (no OpenCL). Reach for "
        "GPU when the model does gradient descent on dense tensors (transformer, CNN, "
        "MLP/TabM, any fine-tune)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task_shape": {
                "type": "string",
                "enum": ["tabular", "nlp", "cv", "finetune"],
                "description": "Shape of the learning problem.",
            },
            "model_family": {
                "type": "string",
                "description": "Library or architecture, e.g. lightgbm, xgboost, catboost, torch, transformers, tabm.",
            },
            "n_rows": {"type": "integer", "description": "Training row count, if known."},
            "n_features": {"type": "integer", "description": "Feature count after engineering, if known."},
        },
        "required": ["task_shape"],
    },
}


async def compute_advice_handler(args: dict, **_kw) -> tuple[str, bool]:
    """Tool wrapper over `recommend_compute`. Pure advice — it launches nothing."""
    try:
        rec = recommend_compute(
            task_shape=args.get("task_shape") or "",
            model_family=args.get("model_family"),
            n_rows=args.get("n_rows"),
            n_features=args.get("n_features"),
        )
    except Exception as e:  # keep a bad argument from ending the turn
        return f"compute_advice failed: {e}", False
    return rec.render(), True


def _is_gpu_flavor(hardware_flavor: str | None, node_type_id: str | None) -> bool:
    flavor = (hardware_flavor or "").lower()
    if flavor:
        return not flavor.startswith("cpu-")
    node = (node_type_id or "").lower()
    # AWS accelerated families: g4dn/g5/g6/g6e (T4, A10, L4, L40S) and p3/p4/p5 (V100..H100).
    return bool(re.match(r"^(g4dn|g5|g6|g6e|p2|p3|p4d?|p5)\.", node))
