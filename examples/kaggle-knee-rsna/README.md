# RSNA Knee Abnormality Detection

Third Kaggle example in this repo. Competition:
[rsna-knee-abnormality-detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection).
Research track, $77,000, closes 2026-10-22, about 2,450 teams. The score is the mean of
twelve ROC AUC values, one per finding.

Three pieces of work live here.

## 1. The published notebook

[RSNA Knee: the data, then a simple baseline](https://www.kaggle.com/code/paiky1995/rsna-knee-the-data-then-a-simple-baseline)

One notebook that reads the data, explains it, trains one small image model, and writes a
submission. Runs end to end on one T4 in about 15 minutes with the internet off. Source of
truth is `notebooks/eda-baseline.src.py` in jupytext percent format. `notebooks/build.py`
generates the `.ipynb` and asserts the two things that silently break a push, which are a
title over 50 characters and a code cell that does not parse.

Twelve sections: the metric, the files, the 58 labelled studies, the reports, where the
training labels come from, the DICOM headers, building one input per study, the model, what it
learned, what it looks at, the submission, and what to try next.

What it measures, all reproduced by the Kaggle run:

| Fact | Value |
|---|---|
| `Fluid_Sensitive` equals `Fat_Suppression` | on all 24,371 series rows, so the CSV carries one axis under two names |
| 95% interval width of one finding's AUC on 58 studies | 0.19 to 0.32, against 0.07 for the mean over twelve |
| Rank correlation, file name order against position along the stack | mean 0.043, so sorting a series by file name randomises the anatomy |
| Studies where the header gives the knee's side | 100%, against 42% that carry the `Laterality` tag; the two agree on 95.9% of the studies that have both |
| Word list written in the notebook, mean AUC on the 58 | 0.731 |
| Published language model label set, mean AUC on the 58 | 0.899 |
| Four sets merged, mean AUC on the 58 | 1.000, and it contains the 58 answers verbatim |
| Reports not written in English | 59.8%, across nine languages and three alphabets |

The label supply chain finding survives from the earlier version of this notebook, cut to the
part that changes what a reader does. Writing real labels in where real labels exist is a
sensible training choice. It only breaks when the same 58 studies are then used to measure a
labeler or a model, which is the one thing they are otherwise for.

`notebooks/label-supply-chain.src.py` is the previous version of the same published kernel,
kept for reference. It is not built or pushed any more. Everything in it that changes what a
reader does was carried into section 5 of the new notebook.

## 2. The audit script

`scripts/audit_labels.py` measures every published weak-label set against the 58 annotated
studies, with `--assert` to turn each finding into a check so an upstream dataset revision is
noticed rather than silently changing the story. It needs no GPU, no pixels, and no Databricks.
`FINAL_RESULTS.md` records what it found.

```bash
mkdir -p /tmp/rsnaknee && cd /tmp/rsnaknee
for f in train.csv train_series.csv test.csv test_series.csv sample_submission.csv; do
  kaggle competitions download -c rsna-knee-abnormality-detection -f "$f" -p . --force
done

mkdir -p /tmp/rsnallm && cd /tmp/rsnallm
for d in pilkwang/rsna-knee-report-labels pilkwang/rsna-knee-llm-labels \
         stevenleehans/rsna-knee-llm-report-labels lixin73/rsna-knee-llm-report-labels-sol56 \
         flight0234/rsna-knee-hybrid-report-labels \
         yunusgmsoy/rsna-knee-abnormality-3-source-merged-labels \
         yunusgmsoy/rsna-knee-llm-labels-4-source-merged; do
  kaggle datasets download -d "$d" -p "$(echo $d | tr '/' '_')" --unzip
done

uv run python scripts/audit_labels.py --root /tmp/rsnaknee --label-root /tmp/rsnallm --assert
```

## 3. Model v2: reproduce raptor, fine-tune a second arm, blend

[🦵 RSNA Knee: 0.926 LB, CoaTNet + fine-tune blend](https://www.kaggle.com/code/paiky1995/rsna-knee-0-926-lb-coatnet-fine-tune-blend)

One notebook that ports the public 0.924-LB raptor pipeline (CoaTNet-2, 64-slice stacks,
CC0 checkpoints), verifies it against the 58 radiologist studies (0.9128 reproduced),
fine-tunes a second arm from the same checkpoint on the yunus v5 labels, and blends the
two arms by weighted rank-mean (w\* = 0.35 chosen on a report-hash holdout, never on the
58; blend 0.9170 on the 58, reported not selected; **public leaderboard 0.926**, rank
665 of 2,676). Runs end to end on one T4 in about
3.7 hours with the internet off, with time guards that degrade to the arm-1-only
submission rather than time out. All numbers in `FINAL_RESULTS.md`.

Source of truth is `notebooks/model-v2/model-v2.src.py`. Its own `build.py` adds two
assertions on top of the shared ones: every call to a notebook-defined function must
match its signature, positional and keyword — a trimmed helper signature is invisible to
a name-binding check and costs one GPU re-run each time.

Two-push protocol: push once with `INCLUDE_GOLD = False` (the honest measurement run,
still a valid submission), then flip to `True` and push the final submission version
(the 58 and their report twins go into fine-tuning; the printed gold numbers carry a
contamination banner).

v3 (kernel v13/v14, 2026-08-31) adds a third arm from raptor's SWA checkpoint, 8
fine-tune epochs, and test-only TTA. The holdout's 3-way blend drops arm 1 entirely
(w\* = 0.0/0.3/0.7, holdout weak 0.9110 vs v2's 0.9093): the SWA twin is a strictly
better base model. Extra epochs bought arm 2 nothing — epoch 3 still selected. The
v14 gold-in run is the final v3 submission candidate.

## Building and pushing the notebook

```bash
cd notebooks && python build.py eda-baseline && kaggle kernels push -p .
python scripts/kaggle_submit.py --dir notebooks/model-v2 --kernel --verify \
  --timeout 14400 --expect submission.csv     # model-v2, both flag modes
```

Push with the current `id` in `kernel-metadata.json`. A title change renames the kernel in
place and moves its slug, so the `id` has to be updated to the new slug before the next push
or Kaggle answers 409.

## Layout

```
scripts/
  audit_labels.py          every finding above, with --assert to catch upstream drift
  enumerate_manifest.py    pages the Kaggle file listing into a local manifest
  ingest_pilot.py          downloads the pilot studies into a UC Volume
  extract_headers.py       DICOM headers into a Delta table
  train.py                 2.5D baseline, one model per versions.yaml row
  kaggle_submit.py         builds, pushes, and verifies the notebook and dataset
notebooks/
  eda-baseline.src.py         jupytext percent format, the single source of truth
  label-supply-chain.src.py   the previous version of the same kernel, superseded
  build.py                    generates the .ipynb and asserts what silently breaks
  kernel-metadata.json
  model-v2/                   raptor reproduction + fine-tune + blend (own build.py,
                              model-v2.src.py, kernel-metadata.json)
artifacts/                 submission, leaderboard snapshot, figures
```

## Reproducing the audit

The audit needs no GPU, no pixels, and no Databricks. It reads the competition CSVs and
the published label sets.

```bash
mkdir -p /tmp/rsnaknee && cd /tmp/rsnaknee
for f in train.csv train_series.csv test.csv sample_submission.csv; do
  kaggle competitions download -c rsna-knee-abnormality-detection -f "$f" -p . --force
done

mkdir -p /tmp/rsnallm && cd /tmp/rsnallm
for d in pilkwang/rsna-knee-report-labels pilkwang/rsna-knee-llm-labels \
         stevenleehans/rsna-knee-llm-report-labels lixin73/rsna-knee-llm-report-labels-sol56 \
         flight0234/rsna-knee-hybrid-report-labels \
         yunusgmsoy/rsna-knee-abnormality-3-source-merged-labels \
         yunusgmsoy/rsna-knee-llm-labels-4-source-merged; do
  kaggle datasets download -d "$d" -p "$(echo $d | tr '/' '_')" --unzip
done

uv run python scripts/audit_labels.py \
  --root /tmp/rsnaknee --label-root /tmp/rsnallm --assert
```

The `--assert` flag turns each finding into a check, so if an upstream dataset is
revised the script fails instead of reporting a stale number as fact.

## Building the notebook

```bash
cd notebooks && python build.py
```

`build.py` asserts the two things that silently break a push. Kaggle truncates a kernel
title at 50 characters without telling you, and a syntax error in one cell wastes a full
re-run because every push queues one. Non-ASCII is reported rather than rejected, because
the emoji in the title is wanted.

## The pilot on Databricks

The full DICOM corpus is about 512 GB across 24,371 series. The audit needs none of it.
The model needs some, so we ingested a pilot of one sagittal series for each of about 700
studies, roughly 15 GB.

Taking studies in listing order is not a biased sample. Study identifiers are hash-like,
and the pilot's language mix matches the rest of the corpus at a chi-square p of 0.501.
Eleven of the 58 annotated studies fall inside it.

```bash
export DATABRICKS_CONFIG_PROFILE=fe-vm-lakebase-praneeth
python scripts/enumerate_manifest.py            # resumable, ~1.8 h for ~700k paths
python scripts/ingest_pilot.py --studies 700
python scripts/extract_headers.py
python scripts/train.py --version v1_sagittal_r18
```

## What we did not do

We did not compete seriously on the leaderboard. We did not ingest the full corpus, use
more than one imaging plane, or publish a thirteenth label set. The board already has
twelve label sets and had no measurement of any of them, so we measured instead of adding
one more.
