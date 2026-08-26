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
