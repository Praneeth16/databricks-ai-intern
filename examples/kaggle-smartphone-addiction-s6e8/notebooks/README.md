# Public notebooks

## `lookup-transformer.ipynb`

Published to Kaggle as
[S6E8: Correlation Does Not Predict Contribution](https://www.kaggle.com/code/paiky1995/s6e8-correlation-does-not-predict-contribution).

Standalone teaching notebook built from this example's second pass. It is not the competition
pipeline. It isolates one question into a controlled experiment that runs end to end on a Kaggle
T4 with no external dependencies, and publishes the result even though the result contradicted
the hypothesis it was built to demonstrate.

### The hypothesis, and how it died

Going in, the claim from our own second pass was: **blend diversity comes from input
representation, not architecture.** The lookup transformer decorrelated from our GBDTs
(Spearman 0.9645) while TabM did not (0.9945), and TabM was fed target-encoded features while
the lookup read exact-value token identities.

The notebook tests that with a strength-matched control: the same trainer, same optimizer,
schedule, augmentation and EMA, fed the trees' own feature matrix. If representation drove
diversity, that control would sit on top of the trees. It did not.

| member | reads | solo OOF AUC | Spearman vs trees | blend weight | added on top of xgb+lgb |
|---|---|---|---|---|---|
| `xgboost` | TE + freq + raw + budget | 0.968454 | 0.9990 (vs lgbm) | 0.3251 | |
| `lightgbm` | same | 0.968331 | 0.9990 (vs xgb) | 0.1627 | |
| `mlp_magnitudes` | rank-gauss magnitudes + masks only | 0.941125 | 0.9416 | 0.0000 | +0.000008 |
| `mlp_encoded` | the trees' exact matrix, rank-gauss scaled | 0.965032 | 0.9665 | 0.0159 | +0.000013 |
| `lookup_transformer` | learned vector per exact value + gated magnitudes | 0.968305 | 0.9658 | **0.4963** | **+0.000577** |

`mlp_encoded` measured 0.9665 against the trees, indistinguishable from the lookup's 0.9658, and
the two neural nets correlate 0.9782 with **each other**, more tightly than either does with
either tree. The decorrelation boundary runs between trees and neural networks, not between
magnitudes and identities. Representation bought no measurable diversity.

### What replaced it

Two members equally decorrelated from the trees (0.9665 vs 0.9658) differ more than **40-fold** in what
they contribute (+0.000013 vs +0.000577). The most decorrelated member in the pool
(`mlp_magnitudes`, 0.9415) contributes 0.000000. So correlation predicts nothing on its own.

What separates them is strength at equal diversity. The neural family was already decorrelated
from the trees; that was free. What was scarce was a neural member strong enough for the
decorrelation to cash out, and `mlp_encoded` at 0.965032 is 0.0033 behind the trees it is meant
to complement. The lookup is level with them. The corrected claim:

> Per-value embeddings did not buy diversity. They bought strength inside a family that was
> already diverse, and that is what made the diversity cash out.

Practical form: when adding a neural network to a GBDT blend, the binding constraint is its solo
score, not how exotic the architecture is. It is already unlike your trees. It needs to be as
good as them.

### The methodological point

`mlp_magnitudes` alone would have *confirmed* the hypothesis. It correlates 0.9415 with the
trees, lower than the lookup's 0.9658, which reads as strong support. It is nothing of the kind:
it is 0.027 AUC weaker than everything else, and any sufficiently weak model decorrelates because
it is largely producing noise. Correlation is only interpretable between models of comparable
strength. One extra strength-matched control reversed the conclusion.

This also means our own `versions.yaml` note on `v15_lookup_wide`, which asserts that
"decorrelation here comes from the input representation, not the architecture", is not supported
by a controlled measurement. The lookup is still the most valuable single member we have; the
stated mechanism was wrong.

### Other content

Cardinality evidence measured before modelling (`gaming_hours` is a float column with 401
distinct values over 691,369 rows, so exact values are levels). The EMA warmup bug
(`0.999 ** 500 = 0.61`, which made a smoke run score AUC 0.45 and look like a flipped label).
Leave-one-out and incremental ablation. Negative results covering capacity, augmentation, Optuna
and row-NaN experiments. Every member's OOF and test predictions exported on the frozen folds so
readers can stack them without refitting.

One number left unresolved and flagged as such in the notebook: our offline TabM measured 0.9945
against XGBoost on these folds, against `mlp_encoded`'s 0.9665 here. Plausible mechanism is that
TabM averages 32 internal sub-models and averaging pulls a ranking toward consensus, but that is
untested.

### Operational notes

Two Kaggle-specific things cost real time:

- Competition data mounts at `/kaggle/input/competitions/<slug>/`, not `/kaggle/input/<slug>/`.
- `enable_gpu` alone can land a **P100 (sm_60)**, and the preinstalled torch build ships kernels
  for sm_70 and up only. XGBoost bundles its own CUDA build including sm_60, so the tree sections
  pass and torch dies 25 minutes later with `no kernel image is available for execution on the
  device`. Set `machine_shape: NvidiaTeslaT4`. The notebook probes CUDA with a real operation at
  setup so this surfaces immediately.
- Changing a kernel's `title` moves its slug; the old URL 404s. Update `id` in the metadata to
  match afterwards or the next push creates a second kernel. Push with the *current* `id` and the
  new `title`: Kaggle renames in place and warns that the title does not resolve to the id, which
  is expected. Slugs are predictable (lowercase, every non-alphanumeric run to one hyphen,
  trimmed), so cross-links can be written before the rename.
- **Kaggle truncates titles at 50 characters, silently.** Every notebook in this competition's
  top 30 is at or under it. Both titles here are checked against that cap.
- `build.py` no longer asserts pure ASCII, it only reports non-ASCII. Emoji in titles are allowed
  and are the local convention in this competition; they are dropped from the slug.

### Files

| file | what |
|---|---|
| `lookup-transformer.src.py` | jupytext percent-format source; the single source of truth |
| `build.py` | assembles the source into the `.ipynb` and asserts the text is pure ASCII |
| `lookup-transformer.ipynb` | generated, what gets pushed |
| `kernel-metadata.json` | Kaggle kernel config (T4, internet off, competition attached) |

Edit the `.src.py`, run `build.py`, then `kaggle kernels push -p .` from the directory holding the
`.ipynb` plus `kernel-metadata.json`. Note `build.py` reads from `/tmp/s6e8nb`; point `HERE` at
this directory to run it here.

### Attribution

The architecture is [tamerlanomralinov's](https://www.kaggle.com/code/tamerlanomralinov/s6e8-lookup-transformer-insights-lb-0-97041),
ported and retrained from scratch on our folds. The controls, the EMA fix, and the ablation
analysis are ours. The PLR numeric embedding is from Gorishniy et al., *On Embeddings for
Numerical Features in Tabular Deep Learning*. See the parent `README.md` for the full attribution
table covering the competition submissions.


---

## Companion dataset

[S6E8 OOF Library: 11 Neural Members](https://www.kaggle.com/datasets/paiky1995/s6e8-oof-library-11-members)
(CC0) publishes the OOF and test vectors for the 11 neural members from the competition pipeline
(`v10`, `v13`-`v17`, `v19`, `v21`-`v24`), staged out of the UC Volume. Deliberately the complement
to the tree-heavy OOF libraries already public in this competition. Both notebooks link to it.

Rebuild it from `/tmp/s6e8_oof_dataset` with `kaggle datasets version -p .`; note that
`kaggle datasets create` is private by default, so pass `-u` for public, and that version history
stays downloadable, so never let an internal path into v1.

## `tabfm/tabfm-deep-dive.ipynb`

Published to Kaggle as
[S6E8: TabFM zero-shot on 0.7% of the data](https://www.kaggle.com/code/paiky1995/s6e8-tabfm-zero-shot-on-0-7-of-the-data).

A deep dive into [google-research/tabfm](https://github.com/google-research/tabfm), Google
Research's zero-shot tabular foundation model (released 2026-06-16), applied to this competition.
Architecture is read out of `tabfm/src/pytorch/model.py`, and the notebook asserts its structural
claims against the loaded module tree so the section is falsifiable rather than prose.

### The measurement

TabFM does no gradient training. It reads labelled rows as context in a single forward pass. On a
16GB T4 that context tops out at 8,000 rows, which is **1.4% of the 553,095 rows a GBDT gets per
fold**, and the AUC curve is still climbing when memory runs out.

| context rows | estimators | ROC AUC | sec / 3,000 predictions |
|---|---|---|---|
| 250 | 1 | 0.914793 | 5.9 |
| 1,000 | 1 | 0.934674 | 7.6 |
| 4,000 | 4 | 0.949890 | 70.8 |
| 8,000 | 1 | 0.950420 | 34.8 |
| 8,000 | 4 | 0.953135 | 138.3 |
| 16,000 | any | out of memory | wanted 7.63 GiB |

Head-to-head on 100,000 identical rows of the frozen folds, against an XGBoost given exactly the
same twelve columns and all 553,095 rows per fold:

| | value |
|---|---|
| xgboost (matched raw features) | 0.966113 |
| tabfm (8,000-row context, 4 estimators, no training) | 0.955027 |
| Spearman between them | 0.9732 |
| hill-climb weight for tabfm | 0.0625 |
| gain from adding tabfm | **+0.000050** |

### Why this notebook exists

It puts the previous notebook's conclusion at risk instead of restating it. That notebook found
that decorrelation only pays at competitive strength. TabFM is the most decorrelated member
obtainable (different family, zero gradient steps, 1.4% of the rows) and is also 0.011 behind, so
the earlier claim predicts it contributes ~nothing. It contributes +0.000050 on 100,000 rows, which
is not distinguishable from zero. Placing all three tested members side by side:

| member | behind baseline | correlation with trees | gain |
|---|---|---|---|
| Lookup-Transformer | 0.000149 | 0.9658 | **+0.000577** |
| MLP on encoded features | 0.003422 | 0.9665 | +0.000013 |
| TabFM, zero-shot | 0.011086 | 0.9732 | +0.000050 |

Correlation barely moves; the strength gap moves two orders of magnitude; contribution tracks
strength. The prediction held.

### Architecture notes worth keeping

- 1,639.44M parameters, `embed_dim=256`, 3 row-axis blocks, **256 inducing vectors** per block,
  3 col-axis blocks, 8 CLS columns, 32 Fourier frequencies.
- Cells become tokens via **Fourier features over a 3-column group** (offsets 0, 1, 3, wrapping
  modulo the active feature count), with separate learned bases for numeric and categorical cells,
  computed in fp32 because the arguments reach magnitude ~30.
- In-context learning is one line: `where(row < train_size, cell + y_emb, cell)`. The label
  embedding is added to context rows and not to query rows. No special tokens, no architectural
  split.
- **Rows are a permutation-invariant set** (induced-point attention, no positional encoding);
  **columns are a positional sequence** (RoPE). That asymmetry is why long context is tractable
  (`O(T * num_inds)`), why context is cacheable (inducing hiddens plus an int8-quantised ICL K/V
  cache), and why the built-in ensembling shuffles *columns*: test-time augmentation against the
  model's own positional bias.

### Upstream problems found

- **The package will not import on Kaggle.** `tabfm/__init__.py` probes the JAX backend inside
  `except ImportError`, but Kaggle's flax is older than the pinned 0.12.7 and raises
  `AttributeError: Module flax.nnx has no attribute 'dataclass'`. Uncaught, so `import tabfm`
  fails even for PyTorch-only use. Widening both guards to `except Exception:` matches their stated
  intent and fixes it.
- **The README contradicts the code.** The FAQ says `max_num_rows` defaults to 100 context rows;
  the code defaults it to `None`, meaning no cap, which on 553,095 rows is an immediate
  out-of-memory error rather than a conservative default.
- **The recommended `ensemble()` preset does not fit.** At an 8,000-row context it OOMs on a 16GB
  T4. That preset is what the repository's own example uses, so reproducing TabFM's published
  numbers needs more memory than Kaggle's free tier provides.

### Traps I hit myself

- **Query rows cost memory too.** Sizing the context is not enough: a 30,000-row `predict_proba`
  call OOMs at a context size that is fine with 3,000 query rows, because the attention over
  `[queries x context]` is materialised per batch. `tabfm_predict` fits the context once and
  predicts in 3,000-row blocks.
- **The baseline was truncated.** XGBoost's `best_iteration` hit its 4,000-tree cap, so early
  stopping never triggered and the reference was still improving. Comparing a zero-shot model
  against an under-trained baseline flatters the zero-shot model. Raised to 9,000.

### Licence and provenance

The TabFM *code* is Apache-2.0. The pretrained weights that `load()` downloads are
`tabfm-non-commercial-v1.0`, **non-commercial and non-production only**. There is no technical
report, so the pretraining corpus is unknown and contamination on a synthetic Playground dataset
cannot be ruled out. Both are stated in the notebook itself.

### Files

| file | what |
|---|---|
| `tabfm-deep-dive.src.py` | jupytext percent-format source; the single source of truth |
| `build.py` | assembles the source into the `.ipynb`, asserting the text is pure ASCII |
| `tabfm-deep-dive.ipynb` | generated, what gets pushed |
| `kernel-metadata.json` | Kaggle config (T4, **internet on** for the clone and HF weights) |
| `measured_context_sweep.csv` | the context curve above, as produced by the run |
