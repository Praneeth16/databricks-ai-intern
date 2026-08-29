# Measurements

Every number here comes from `scripts/audit_labels.py`, which reads the competition CSVs
and the published label sets and needs no GPU. Run it with `--assert` to check the
findings still hold against the current versions of those datasets.

Measured 2026-08-26.

## The corpus

| Quantity | Value |
|---|---|
| Training studies | 4,407 |
| Studies with annotations | 58, which is 1.32% |
| Series | 24,371 |
| DICOM files | about 700,000 |
| Corpus size on disk | about 512 GB |
| Findings per study | 12 |
| Score | mean of 12 ROC AUC values |
| Test set | hidden, and it has no report column |

## Languages in the reports

Two methods, agreeing on 99.3% of the 4,407 reports. The first is the detector in the
notebook, which scores report section headers and function words and decides Greek and
Bulgarian by alphabet. The second is `langdetect`, used offline as a check only.

| Language | Studies | Share |
|---|---|---|
| English | 1,763 | 40.0% |
| Spanish | 682 | 15.5% |
| Turkish | 545 | 12.4% |
| Croatian | 379 | 8.6% |
| Greek | 321 | 7.3% |
| German | 262 | 5.9% |
| Bulgarian | 220 | 5.0% |
| Dutch | 153 | 3.5% |
| French | 82 | 1.9% |

Greek and Bulgarian are decided by Unicode block, so those 541 studies cannot be
misassigned. The only real disagreement between the two methods is 26 Croatian reports
that the notebook's detector reads as English, which is 0.6% of the corpus.

## Published label sets against the 58 annotated studies

Mean over the twelve per finding AUC values, with a 2,000 sample bootstrap interval.

| Label set | Mean AUC | 95% interval | Contains the annotations |
|---|---|---|---|
| yunus_3src | 1.000 | n/a | yes |
| yunus_4src | 1.000 | n/a | yes |
| flight_hybrid | 0.899 | 0.79 to 0.96 | no |
| stevenleehans_v4 | 0.893 | 0.79 to 0.96 | no |
| stevenleehans_full | 0.878 | 0.77 to 0.96 | no |
| pilkwang_v2_llm | 0.870 | 0.77 to 0.95 | no |
| lixin73_gpt56sol | 0.835 | 0.73 to 0.92 | no |
| pilkwang_v1_regex | 0.813 | 0.69 to 0.92 | no |

The widest interval is 0.230 across, and the spread between the best and worst honest
set is 0.087. The ordering of the honest sets is not significant on 58 studies.

A language model reading the reports is worth about 0.086 AUC of label quality over a
keyword list, comparing `pilkwang_v1_regex` at 0.813 with `flight_hybrid` at 0.899. That
gap is larger than the spread among the language model sets, so it survives the interval
problem in a way the ranking does not.

## The two sets that contain the annotations

| Check | yunus_3src | lixin73_gpt56sol |
|---|---|---|
| Values exactly 0.0 or 1.0 on the 58 rows | 100.0% | 95.4% |
| Values exactly 0.0 or 1.0 on the other 4,349 rows | 0.0% | 96.2% |
| Reproduces the annotations | 100.0% | 79.9% |
| Distinct values on the other rows | 344 | 2 |
| Verdict | contains them | clean |

The second column is the reason the test needs both parts. `lixin73_gpt56sol` is a set
that gives hard 0 and 1 answers everywhere, which looks alarming on the annotated rows
alone and is fine once you compare the two populations.

Files: `yunusgmsoy/rsna-knee-abnormality-3-source-merged-labels/report_labels_v3.csv` and
`yunusgmsoy/rsna-knee-llm-labels-4-source-merged/report_labels_v5.csv`.

Writing real labels in where real labels exist is a reasonable training choice, and the
competition invites building labels from reports. It only breaks when those 58 studies
are then used to measure a labeler or a model.

## Agreement between labelers, by language

Mean rank correlation over all 15 pairs of the 6 honest label sets, across 4,406 studies.
This needs no annotations, so unlike the table above it is not limited to 58 studies.

| Language | Studies | Agreement | Worst finding | Worst |
|---|---|---|---|---|
| Bulgarian | 220 | 0.685 | Synovitis | 0.336 |
| Turkish | 546 | 0.689 | Fracture | 0.483 |
| Croatian | 406 | 0.706 | Fracture | 0.476 |
| Dutch | 153 | 0.717 | Synovitis | 0.471 |
| Greek | 321 | 0.730 | Fracture | 0.528 |
| French | 81 | 0.734 | Fracture | 0.429 |
| German | 262 | 0.748 | Fracture | 0.516 |
| Spanish | 682 | 0.767 | Fracture | 0.563 |
| English | 1,735 | 0.818 | Fracture | 0.648 |

`Fracture` is the worst agreed finding in six of the nine languages.

The labelers get worse together rather than independently, which is what makes merging
sources misleading. Voting removes noise when sources fail independently. Here they are
all reading a language none of them was built for, so a merge returns several copies of a
similar mistake with a confident average.

## Keyword list coverage, by language

How often a keyword list covering the six Latin alphabet languages fires at all.

| Language | Fires on |
|---|---|
| Bulgarian | 0.9% |
| Greek | 41.1% |
| Croatian | 70.9% |
| Spanish | 84.8% |
| Dutch | 91.5% |
| German | 98.5% |
| English | 99.8% |
| Turkish | 99.8% |
| French | 100.0% |

A study where nothing fires becomes twelve zeros. Bulgarian and Greek together are 541
studies, so a keyword labeler quietly records about 500 studies as healthy. The Greek
41.1% comes from Latin abbreviations and measurements surviving inside a Greek report
rather than from any word the list knows.

## Three things that look like silence

Measured on the 58 annotated studies.

| Question | Mean P(finding is present) |
|---|---|
| The keyword list never fired | 0.277 |
| A language model answered "cannot tell" | 0.082 |
| The best set concluded the finding is absent | 0.087 |

The first row is the confounded one. A keyword list cannot separate a radiologist's
silence from its own failure to read the report, and the coverage table above shows those
failures are large. Our first pass reported 0.277 as the silence rate, and the better
instrument put it at 0.082. Most of what looked like silence was the keyword list.

The residual 8.2% is concentrated rather than spread out.

| Finding | Model cannot tell | Of those, share present |
|---|---|---|
| Synovitis | 70.7% | 34.1% |
| Fracture | 44.8% | 15.4% |
| PF OA | 24.1% | 21.4% |
| Lateral Meniscus | 12.1% | 14.3% |
| Effusion | 3.4% | 0.0% |

Synovitis is the whole effect. Across every published label set it also scores worst, at
0.628 to 0.790. Those are the same fact. Synovitis scores badly from text because the
radiologists in this corpus mostly do not write it down, so no better prompt recovers it
and the image model has to carry that finding alone.

## Are the 58 a fair sample

| Quantity | The 58 | The other 4,349 | p |
|---|---|---|---|
| Report length in characters | 1,305 | 1,095 | 0.029 |
| Series per study | 5.79 | 5.53 | 0.091 |

Findings present in the 58: Effusion 60.3%, Synovitis 46.6%, Medial Meniscus 44.8%,
ACL 41.4%, Lateral Meniscus 39.7%, PF OA 36.2%, Contusion 32.8%, Fracture 31.0%,
Medial OA 25.9%, Baker's 20.7%, Lateral OA 19.0%, MCL 15.5%. An average annotated study
carries 4.1 of the 12 findings.

The reports are longer and the findings are dense, so the 58 look selected for studies
with something to see. The number of series per study is about the same, so the imaging
protocol is not what separates them. We do not know how the 58 were chosen, and we cannot
say which population the hidden test set matches.

## The pilot ingest

The audit needs no pixels. The model does, so we took one sagittal fluid sensitive series
for each of about 700 studies, roughly 15 GB of the 512 GB corpus.

Taking studies in listing order is not a biased sample. Study identifiers are hash-like,
and the pilot's language mix matches the rest of the corpus at a chi-square p of 0.501.
Eleven of the 58 annotated studies fall inside the pilot, which is a smoke test rather
than a measurement.

## What we could not settle

The 8.2% figure rests on the UNK verdicts of one label set, `pilkwang/rsna-knee-llm-labels`,
because it is the only published set that reports YES, NO and UNK instead of collapsing
uncertainty into a number. A second three way set would let someone check it.

We do not know how the 58 annotated studies were chosen, so we can describe how they
differ from the rest but not why.

The notebook's language detector reads 26 Croatian reports as English. That pulls the
Croatian agreement figure slightly toward the English one, so 0.706 for Croatian is a
little optimistic.

## The baseline model in the published notebook

Measured by the Kaggle run of `notebooks/eda-baseline.src.py`, version 7, the version submitted,
on one T4 with the internet off. Whole run about 850 seconds, of which about 670 seconds is
reading DICOM files at 6.5 studies a second. All 4,407 studies were read and none failed to
decode.

One input per study is twelve slices at 224 by 224, taken at even spacing from the middle 80% of
one sagittal fluid sensitive series, in medial to lateral order. The model is a ResNet18 with
ImageNet weights, its first layer widened from three channels to twelve, and twelve outputs. The
training target is the `flight0234/rsna-knee-hybrid-report-labels` set, which scores 0.899 on the
58 annotated studies and does not contain them. Eight epochs, batch 32, one cycle schedule, no
horizontal flip. Predictions are the rank average of the three epochs that scored best on the
held out fifth, which were epochs 5, 6 and 7.

| Measurement | Average AUC over twelve findings |
|---|---|
| Best single epoch, held out fifth (883 studies) | 0.771 |
| Rank average of the best three epochs, same holdout | 0.777 |
| Rank average against the radiologist, 58 studies, none trained on | 0.778 |
| Public leaderboard, submission 55800687 | **0.798** |
| Report labels against the radiologist, same 58 | 0.899 |

The 58 study check predicted the leaderboard within 0.02. That is the useful result about the
method rather than about the model. 58 studies cannot rank two models that are 0.02 apart, which
is what section 3 of the notebook says, but they do tell you whether a model works at all, and
here the estimate landed inside the interval. Top of the public board is about 0.952.

Per finding, the submitted model against the radiologist on the 58: Baker's 0.871, Medial OA
0.864, Effusion 0.857, Medial Meniscus 0.823, ACL 0.817, Contusion 0.785, PF OA 0.775, Synovitis
0.761, Lateral OA 0.753, Lateral Meniscus 0.699, Fracture 0.681, MCL 0.644.

Three things the runs established that are not about the score.

Holding the 58 out of training is worth 0.066. An earlier version let 52 of the 58 into the
training set, and the model then scored 0.859 against the radiologist instead of 0.793. Nothing
warned about it, because the label the model trained on was the report label and the label it was
scored against was the annotation, so the two never looked like the same number.

A 58 study AUC moves about 0.03 between runs of the same recipe. Moving one study between the two
sides of the split changed the mean from 0.793 to 0.763. That is the size the bootstrap in
section 3 of the notebook predicts from the sample alone, so the notebook's own reruns confirm
its own warning. The held out number moved 0.005 over the same change.

The geometry rule for the knee's side works. `ImagePositionPatient` gives a side for 100% of
studies against 42% that carry the `Laterality` tag, and the two agree on 95.9% of the 49
studies in the sample that have both.

## The word list written inside the notebook

A clause level reader over nine languages, about 60 lines of patterns, scores 0.731 on the 58
against 0.899 for a language model reading the same reports. Per language it is uneven in a way
an overall coverage rate hides. Bulgarian looks fine at 97% of reports where something fires,
but only the term for an effusion fires, at 95.5%, while the other eleven findings sit near
zero. Turkish is the weakest overall at 59.7%, with 217 studies where nothing fires at all.

## Model v2: reproduce raptor, fine-tune a second arm, blend

The second model does not start from the 0.798 baseline. A public CC0 dataset
(`dreaddevelopment/raptor-knee-widedense`) ships two CoaTNet checkpoints
(`coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`, per-finding attention pooling head, 73.2M
parameters) that score 0.924 on the public board as a single model, and the author's
inference notebook is the exact pipeline. The strategy: port the pipeline, verify it
against the 58 radiologist studies, fine-tune a second arm from the same checkpoint on
the yunus v5 labels (different labels are the only diversity the arms get), and blend
the two arms by weighted rank-mean. Split discipline: a 20% report-hash holdout (832
studies) stays out of fine-tuning in both flag modes, and epoch choice plus blend weight
come from it, never from the 58.

Measured on the honest run (`INCLUDE_GOLD=False`, kernel version 7):

- **The port reproduces their pipeline: 0.9128 mean AUC on the 58** — between the 0.9167
  stored inside the checkpoint and the 0.9054 in their notebook comment for the same
  file. Their blends added about +0.001 live, so a single-arm 0.924 LB is consistent
  with a ~0.913 gate plus the usual gate-to-LB offset.
- Arm 2 (three epochs over 3,522 studies, loss 0.4904 -> 0.4712, epoch 3 selected):
  **0.9158 on the 58** (95% CI half-width 0.052) against arm 1's 0.9128 (0.051).
- Holdout weak-label AUC: arm 1 0.9083, arm 2 0.9045. Arm 2 agrees less with v5; the
  disagreement is the diversity doing its job.
- Blend weight chosen on the holdout: **w\* = 0.35**, holdout weak 0.9097 (best of the
  three, +0.0014 over the better arm). Blend against the radiologist, reported not
  selected on: **0.9170**.
- Per finding on the 58, arm 1: ACL 0.979, MCL 0.973, Medial Meniscus 0.960, Lateral
  Meniscus 0.840, Medial OA 0.981, Lateral OA 0.843, PF OA 0.820, Effusion 0.984,
  Synovitis 0.777, Baker's 0.971, Contusion 0.934, Fracture 0.892. Synovitis and PF OA
  are the weak findings, as they were for the baseline.
- End-to-end 3.7 hours on one T4 (decode 48 min, three fine-tune epochs 2.4 h, evals
  about 30 min) inside the 9-hour session, with time guards that degrade to the
  arm-1-only submission instead of timing out.

Two defects found while porting:

- Their `_pick_series_for_slot` calls `int(r.get('Fluid_Sensitive', 0) or 0)`. A NaN is
  truthy, `int(nan)` raises, their per-study try/except swallows it, and the study is
  scored 0.5 with no log line. This port treats NaN as no-preference and logs the
  column's presence.
- Submitting all 4,407 decode jobs as one list of futures retains every 7.2 MB decoded
  stack inside its Future object: 31.8 GB of results, and the kernel is SIGKILLed at
  study ~3,300 with no traceback. Bounded batches (`ex.map` over slices) keep the
  in-flight footprint at 24 stacks.

The submission run (`INCLUDE_GOLD=True`, kernel version 8) is the same notebook with
the 58 radiologist studies and their 53 report twins admitted to fine-tuning. Its
printed gold numbers are contaminated by design, the log carries the banner, and the
blend weight still comes from the holdout.

The gold-in run completed clean (3.7 h): fine-tune set 3,575 studies (58 gold + 53 twins
in), epoch 3 selected, w\* = 0.3, holdout weak 0.9092 — statistically identical to the
honest run's 0.9097, so including the 52 non-holdout gold studies in training moved
nothing measurable out-of-sample. Its arm-2 gold mean of 0.9403 is exactly the
contamination the memory on weak-label inflation predicts (0.066 AUC is the measured
inflation rate; 0.9403 - 0.9158 = 0.025 on a different split, same direction), and the
log carries the banner. The final submission is this run's blend.
