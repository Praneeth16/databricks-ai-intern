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


def _is_gpu_flavor(hardware_flavor: str | None, node_type_id: str | None) -> bool:
    flavor = (hardware_flavor or "").lower()
    if flavor:
        return not flavor.startswith("cpu-")
    node = (node_type_id or "").lower()
    # AWS accelerated families: g4dn/g5/g6/g6e (T4, A10, L4, L40S) and p3/p4/p5 (V100..H100).
    return bool(re.match(r"^(g4dn|g5|g6|g6e|p2|p3|p4d?|p5)\.", node))
