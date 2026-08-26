# Notebooks

Two notebooks, each generated from a jupytext percent-format `.src.py` by its own
`build.py`. Never hand-edit the `.ipynb`, because the next build overwrites it.

| Directory | Notebook | Purpose |
|---|---|---|
| `.` | `label-supply-chain` | The audit. CPU only, internet off. |
| `submit/` | `knee-submit` | The minimal baseline and the submission. |

```bash
python build.py          # from either directory
```

`build.py` asserts the two things that silently break a push. Kaggle truncates a kernel
title at 50 characters without warning, and every push queues a full re-run, so a syntax
error in one cell costs a wasted run. Non-ASCII is reported rather than rejected, because
the emoji in the title and the diacritics in the language detector are both wanted.

The audit notebook attaches seven published label-set datasets and reads the competition
data from `/kaggle/input/competitions/rsna-knee-abnormality-detection`, which is where
competition data mounts. It is not `/kaggle/input/<slug>`.
