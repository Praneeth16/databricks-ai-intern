# RSNA Knee Abnormality Detection: auditing the label supply chain

Third Kaggle example in this repo. Unlike the two before it, the work here is not a
leaderboard climb. The competition has a structural problem in how everyone builds
training labels, and measuring that problem is worth more than another model.

Competition: [rsna-knee-abnormality-detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection).
Research track, $77,000, closes 2026-10-22, 2,394 teams at the time of writing.
The score is the mean of twelve ROC AUC values, one per finding.

## The problem in one paragraph

There are 4,407 training studies and labels for 58 of them. Everyone else builds labels
by reading the free-text radiology report that ships with each study. About a dozen of
those report-derived label sets are now published as Kaggle Datasets, and people merge
them into each other. Nobody had measured any of them. The test set has no report
column, so text can only ever help you build training labels, never predict.

## What we measured

| # | Finding |
|---|---|
| 1 | Two published label sets score AUC 1.000 on all twelve findings against the 58 annotated studies. On those 58 rows every value is exactly 0.0 or 1.0, and on the other 4,349 rows not one value is. The annotations were written in. |
| 2 | A 58 row AUC has a 95% interval about 0.23 wide, while the gap between the best and worst honest label set is 0.087. The 58 studies cannot rank label sets. |
| 3 | Agreement between independent labelers needs no annotations, so it covers all 4,407 studies. It falls from 0.819 in English to 0.685 in Bulgarian. The labelers fail together, so merging sources does not rescue the hard languages. |
| 4 | The reports are in nine languages, and two are not written in the Latin alphabet. A Latin alphabet keyword list fires on 99.8% of English reports and 0.9% of Bulgarian ones. |
| 5 | A keyword list implies 27.7% of unmentioned findings are present. A language model allowed to answer "cannot tell" puts it at 8.2%. For Synovitis it is 34.1%, and Synovitis is the finding every label set scores worst on. |
| 6 | The 58 annotated studies are not a fair sample. Their reports are longer, at 1,305 characters against 1,095 with p = 0.029, and 60.3% of them have an effusion. |

Finding 1 is not cheating. Writing real labels in where real labels exist is a sensible
training choice. It only breaks when the same 58 studies are then used to measure a
labeler or a model, which is the one thing they are otherwise for.

Finding 5 corrects our own first measurement. A keyword list put silence at 27.7%, and
using a better instrument dropped it to 8.2%. Most of what looked like a radiologist's
silence was the keyword list failing to read the report. The residual is concentrated in
Synovitis rather than spread evenly, and that is the useful part.

## What shipped

1. A public notebook, `notebooks/label-supply-chain.src.py`, built into
   `label-supply-chain.ipynb`. Runs on CPU with the internet switched off.
2. A CC0 Kaggle Dataset holding a label confidence table. One row per study and finding
   for all 4,407 studies, with the cross-labeler agreement for that language and finding,
   a weight, and a flag for entries to mask rather than set to zero.
3. A minimal image model and submission, trained on Databricks serverless GPU.

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
  label-supply-chain.src.py   jupytext percent format, the single source of truth
  build.py                    generates the .ipynb and asserts what silently breaks
  kernel-metadata.json
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
