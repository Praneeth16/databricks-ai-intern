# Examples

Worked end-to-end runs of `databricks-ai-intern` on real problems, kept in the repo so
claims about the agent are checkable rather than asserted.

| Example | Task | Metric | Result |
|---|---|---|---|
| [`kaggle-f1-pitstops-s6e5`](kaggle-f1-pitstops-s6e5/) | Kaggle Playground S6E5 — predict whether an F1 driver pits next lap | ROC-AUC | public LB **0.94924** (rank 1189/2457 on the first submission) |
| [`kaggle-smartphone-addiction-s6e8`](kaggle-smartphone-addiction-s6e8/) | Kaggle Playground S6E8 — predict smartphone addiction from usage features | ROC-AUC | see that folder's `FINAL_RESULTS.md` |
| [`kaggle-knee-rsna`](kaggle-knee-rsna/) | RSNA Knee Abnormality Detection — audit the report-derived label sets everyone trains on | mean of 12 ROC-AUC | found the 58 annotations written into two published label sets; see that folder's `FINAL_RESULTS.md` |

## The folder standard

`kaggle-smartphone-addiction-s6e8` is the reference layout. `kaggle-f1-pitstops-s6e5`
predates it and has not been migrated.

```
examples/<slug>/
├── README.md          # narrative: what was tried, what worked, what broke and how
├── FINAL_RESULTS.md   # every attempt ranked, with the lessons locked
├── versions.yaml      # the attempt ledger — see below
├── scripts/
│   ├── agent-prompt.txt   # the prompt handed to the agent
│   ├── train.py           # ONE parameterized trainer, driven by versions.yaml
│   └── kaggle_submit.py   # submit + poll for the leaderboard score
├── artifacts/         # submission.csv, oof/, leaderboard snapshot, charts
└── logs/              # agent transcript
```

### One trainer, many versions

**Each attempt is a row in `versions.yaml`, not a copy of the script.** s6e5 accumulated
26 near-duplicate training scripts (`..._v5_2_optuna.py`, `..._v13_2_year_pseudo.py`, and
so on). That made two things impossible: seeing what actually differed between two
attempts, and trusting that a shared fix had reached all of them.

So `train.py` takes `--version <id>` and reads the feature set, model family,
hyperparameters, and CV geometry from `versions.yaml`. A new attempt is a new row.

`versions.yaml` also carries a **`rejected:`** block — ideas that were tried and lost,
each with the number that killed it. This is the part that compounds: it stops the next
run (agent or human) from re-deriving a dead end, and it is where the
`kaggle-tabular-classification` skill's falsified-hypotheses table comes from.

### Rules that make an example worth keeping

- **Real numbers or none.** Leaderboard scores are actual submissions. If something was
  estimated rather than measured, it says so.
- **Record the negative results.** The ranked table of what *didn't* work is usually more
  valuable than the winning config, because the winning config is one line of params.
- **No borrowing other competitors' submissions.** Blending public submissions will lift
  a public-LB number and teaches nothing about the agent. Both examples decline it, and
  say where that leaves them relative to the top of the board.
- **Say who did what.** Where a human iterated locally and the agent ran the canonical
  pipeline, the README distinguishes the two rather than implying full autonomy.
