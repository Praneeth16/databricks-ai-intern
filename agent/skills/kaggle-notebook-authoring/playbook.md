# Skill: kaggle-notebook-authoring

How to publish a Kaggle notebook that gets read. This is the *packaging* half;
`kaggle-tabular-classification` is the *winning* half. Load both when the task is
"compete and then write it up".

**The founding measurement.** Two S6E8 notebooks with real content — a controlled
strength-matched experiment and a 1.6B-parameter-model teardown, both running end to
end on a T4 — sat at **0 votes for 20 hours**. Nothing was wrong with the science. The
titles hid the claim, the results were deliberately withheld until section 8, and no
reusable asset shipped alongside. The first upvote arrived within hours of changing
only the title and the opening screen. Treat packaging as a first-class deliverable,
not decoration on a finished analysis.

## Phase 0 — Decide whether to publish at all (cheap, always do it)

Publish when at least one is true. If none are, keep it private and move on.

1. You have a **reusable asset**: OOF vectors, a working port of an architecture,
   a dataset, a bug fix in an upstream package.
2. You have a **falsified belief**: something the community repeats that you measured
   and it did not hold. This is the highest-value kind and the rarest.
3. You have a **negative result with a threshold**: a direction that is now closed,
   with the number that closed it.

> **Anti-pattern: publishing an unmodified upstream template.** Four kernels on the
> account we audited were verbatim Unsloth examples with placeholder tokens
> (`push_to_hub("hf/model", token = "")`) and stock gsm8k. They cannot earn votes
> because the original is better known, and giving one a real title reads as claimed
> authorship. Leave templates auto-named, or keep them private.

## Phase 1 — The title (hard limits, get these exactly right)

**Kaggle truncates titles at 50 characters, silently.** Verified across 302 top-voted
kernels: none exceeds 50, and one reads `Exercise: Airline Price Optimization Microchalleng`.
Assert `len(title) <= 50` in the build script.

**Slugs are deterministic**: lowercase, every run of non-alphanumerics collapses to one
hyphen, then trim. `"Lookup-Transformer | AUC: 96.91%"` -> `lookup-transformer-auc-96-91`.
Emoji are dropped. So you can write cross-links between two notebooks *before* the
rename that creates their slugs, and verify against the push output afterwards.

What the title must carry, in priority order:

1. **The competition token** (`S6E8`, `S5E7`). It is the scan keyword; 24 of the top 30
   S6E8 titles carry it. Without it your notebook is invisible to people browsing the
   competition.
2. **A public LB score, or a claim.** Both work. Measured S6E8 vote counts:
   `S6E8 Addiction LB 0.97113` (82), `Why Every S6E8 Notebook Above 0.97110 Overfits`
   (64), `S6E8 Diversity Beats Strength` (54), `will your 0.971 survive the private
   split?` (47), `What Moved the Score, and What Didn't` (46). Score-titles and
   claim-titles both reach the top; vague-titles do not.
3. **Emoji, optional.** Local convention near the top of Playground boards
   (`📱`, `🔥🔥🔥`, `🥇`). They cost 1-2 characters against the cap and vanish from the slug.

> **Anti-pattern: a CV score in the title.** `Lookup-Transformer | AUC: 96.91%` advertises
> 0.9691 on a board whose top is 0.97113. Readers compare it to the leaderboard, not to
> your folds, and it reads as a losing notebook. Put an LB number or no number.

> **Anti-pattern: burying a good headline in the H1.** That same notebook's H1 was
> "The most decorrelated model in my ensemble was worth nothing", which is an excellent
> title, and it was invisible because the *kernel* title was different. Listings show
> the kernel title only. Promote the H1, or shorten it to fit the cap and keep the long
> form as the H1.

## Phase 2 — The first screen decides the vote

The vote decision happens in roughly fifteen seconds of the first screen. Structure
the opening cell in this order, above a `---`:

1. **H1** matching the kernel title.
2. **One sentence** on what you measured and what broke.
3. **The results table.** Immediately. Not after the setup, not after the narrative.
4. **"What you get if you fork this"** — 3 to 5 numbered concrete items.
5. **A contents table** with in-notebook anchor links.

Then the narrative, the method, the code.

> **Anti-pattern: the withheld punchline.** Both audited notebooks said it out loud —
> "I am not going to open with the results table... jump to section 8" and "I am not
> going to put the scores in this paragraph". That is a blog-post technique and it kills
> a Kaggle notebook, because the reader who would have voted has already left. The 64-vote
> notebook opens with its claim in line 1 and "produces a 0.97115 submission" in line 4.
> The 54-vote notebook has "Public LB 0.97113" in line 3 and "This runs end to end. Fork
> it and you get the same number" in line 5. Leading with the ending does not spoil the
> reasoning; it earns the scroll that gets the reasoning read.

**Length is not the enemy, but front-loading is mandatory.** The 82-vote S6E8 notebook is
5 cells and 5 KB. The 54-vote one is 20 cells and 29 KB and works fine, because its
conclusion is on screen one. A 76 KB notebook with the answer in the middle scores zero.

## Phase 3 — The body

- **Charts are what get screenshotted.** Put a figure on the thesis, not only on the
  inputs. One audited notebook had a strength-vs-correlation scatter but no plot of the
  quantity the notebook was *about* (contribution), which was the one number readers
  needed. If a claim is the point of the notebook, it needs a panel.
- **State the rules that would invalidate your own measurements**, up front: the fold
  definition, what is refit inside folds, what is held constant across models, the float
  precision. At AUC 0.968 over ~300k rows, float32 rounding reorders enough ties to move
  the fourth decimal, and the fourth decimal is the competition.
- **Publish negatives with thresholds.** "40-trial Optuna: 0.966850 vs the incumbent's
  0.966853 on an identical fold" closes a direction for every reader. This is the
  cheapest goodwill available and it is what the `What Moved the Score, and What Didn't`
  genre (46 votes) is made of.
- **Flag what you could not settle.** One audited notebook reported a correlation it
  could not reconcile with its own offline run and said so. That is a feature: it invites
  comments, which is distribution.
- **Assert structural claims against the loaded object** where you can, rather than
  asserting them in prose. Claims about an architecture read out of source should be
  checked against the module tree in the notebook itself, so the section is falsifiable.
- **Credit ported architectures by name and link, and ask readers to upvote the original.**
  It is correct, and it is the highest-conversion comment you will write in Phase 6.

## Phase 4 — Ship a reusable asset, not only an argument

**This is the highest-leverage single act and the one most often skipped.** In S6E8 the
82-vote notebook is 5 cells that publish OOF predictions; the 51- and 54-vote notebooks
are built on published OOF libraries. A dataset is separately discoverable, earns its own
votes, and every borrower links back to you.

Publish OOF **and** test predictions as a separate CC0 Kaggle Dataset:

- Row-align everything to `train.csv` in its original row order, and **state the exact
  fold definition** (`StratifiedKFold(shuffle=True, random_state=42)`, and which members
  used how many folds). Without it the arrays are unusable.
- float64. Ship both `.npy` and a single wide CSV; let people choose.
- **Position against what is already public.** If the public pools are boosted-tree-heavy,
  a neural-only library is the scarce complement, and saying so in the subtitle is worth
  more than listing your AUCs.
- Link the dataset from the notebook and the notebook from the dataset.

### Dataset API traps

- **`kaggle datasets create` is private by default.** Pass `-u/--public`. There is no CLI
  command to flip visibility afterwards.
- **Version history stays downloadable once public.** If v1 contains something that must
  not ship, `kaggle datasets delete` and recreate; a new version does not retract v1.
- **Audit the staging directory before upload.** A generated report containing an internal
  volume path and workspace profile name was swept into one upload. Caught while still
  private. Enumerate exactly what is in the folder, do not glob and hope.

## Phase 5 — Build and push mechanics

Keep a **jupytext percent-format `.src.py` as the single source of truth** and generate
the `.ipynb` from it with a `build.py`. Never hand-edit the notebook JSON. Have `build.py`
assert the things that silently break:

- `len(title) <= 50`.
- every code cell parses (`ast.parse`).
- non-ASCII is *reported*, not rejected — emoji in titles are wanted. An ASCII-only
  assert is over-tight and blocks the local convention.

Then the platform traps:

- **`enable_gpu` alone can land a P100 (sm_60)**, and the preinstalled torch ships kernels
  for sm_70 and up. XGBoost bundles its own CUDA build including sm_60, so the tree
  sections pass and torch dies 25 minutes later with `no kernel image is available for
  execution on the device`. Set `machine_shape: NvidiaTeslaT4`, and probe CUDA with a real
  operation in the setup cell so it fails in 10 seconds instead of 25 minutes.
- **Competition data mounts at `/kaggle/input/competitions/<slug>/`**, not
  `/kaggle/input/<slug>/`.
- **Every `kaggle kernels push` queues a full re-run.** Batch your edits and push once; a
  packaging fix that costs a 4-hour GPU re-run is a packaging fix you should have caught
  in `build.py`.
- **Renaming moves the slug and 404s the old URL.** Push with the *current* `id` and the
  *new* `title`. Kaggle renames in place and warns "title does not resolve to the
  specified id", which is expected and not an error. Setting `id` to the new slug first
  creates a *second* kernel. Update `id` in the metadata after the rename lands.
- **The CLI cannot change privacy on an existing kernel.** `is_private` is honored at
  creation only, and `kernels update` is an alias for `push`. 22 pushes across 6 kernels
  confirmed it changes nothing. The web UI is the only path. `kernels delete` does work,
  so the automatable option is the destructive one — ask before using it.

### Verify the published run, do not assume it

A pushed notebook whose run failed shows no outputs and reads as broken. After every push:

```
kaggle kernels status <ref>                 # wait for COMPLETE, not just pushed
kaggle kernels output <ref> -p <dir>        # expected files present?
```

Then parse the downloaded `.log` for `Error`/`Traceback`, and confirm
`__results___files/` was created once per figure cell — that is the evidence your plots
actually rendered.

## Phase 6 — Distribution (without it, none of the above matters)

**An account with ~20 lifetime votes gets no browse traffic.** The first ten votes come
from the comment and discussion surface, not from people finding you. Budget real time
for this; it is not optional polish.

Ranked by conversion:

1. **Comment on the notebook whose architecture you ported**, reporting what you measured
   when you retrained it, including anything that would save that author time. Highest
   conversion on the list, and it is simply correct.
2. **Report a clean negative on a widely-forked starter notebook.** Short, factual, high
   visibility.
3. **A competition discussion post** carrying your headline table and the dataset link.
4. **Engage the notebook that claims the opposite of your result** — as a reconciliation,
   not a rebuttal. Both measurements are usually right in their own regime, and naming the
   regime that separates them is a genuine contribution. Propose the test that would
   distinguish them.
5. Complementary negative results on a "what didn't work" notebook.

> **Anti-pattern: six comments in one hour.** From a low-vote account that reads as spam
> to moderators and to readers. Spread over two days, lead with the courtesy comments, and
> post the argumentative ones while you still have time to reply.

**Timing.** Playground notebooks stop accruing votes almost immediately after the
competition closes. A notebook published with six days left still reaches 30-44 votes; one
published after close reaches nothing. Publish before you have finished optimising.

## Pre-publish checklist

```
[ ] title <= 50 chars, carries the competition token, carries a score or a claim
[ ] kernel title == H1 (or H1 is the longer form of it)
[ ] results table on the first screen, above any narrative
[ ] "what you get if you fork this" block, 3-5 concrete items
[ ] contents table with anchor links
[ ] a figure on the notebook's actual thesis
[ ] fold definition, refit boundaries, and float precision stated
[ ] negatives included, each with the number that closed the direction
[ ] ported work credited by name and link
[ ] machine_shape pinned; CUDA probed with a real op in setup
[ ] OOF + test predictions exported, and published as a separate CC0 dataset
[ ] dataset staging dir enumerated file-by-file, no internal paths
[ ] kernel-metadata id matches the live slug
[ ] run reached COMPLETE; output files and __results___files/ verified
[ ] distribution plan written down, spread over two days
```

## Already measured — do not re-derive

| Idea | Verdict | Evidence |
|---|---|---|
| Withhold the result to preserve the narrative turn | **Wrong** | 0 votes / 20 h on two strong notebooks; first vote after front-loading |
| Put your CV score in the title | **Wrong** | `AUC: 96.91%` reads as losing against a 0.97113 board |
| Titles can be as long as needed | **Wrong** | Hard 50-char truncation, verified over 302 kernels |
| Emoji break the slug | **False** | Dropped from the slug; safe anywhere in the title |
| Long notebooks cannot score | **False** | 20 cells / 29 KB reached 54 votes with a front-loaded conclusion |
| Quality alone earns votes | **Wrong** | Content was never the problem in the founding measurement |
| CLI can privatize an existing kernel | **No** | 22 pushes, 6 kernels, zero effect; UI only |
| `datasets create` publishes | **No** | Private by default; needs `-u` |
| A new dataset version retracts v1 | **No** | v1 stays downloadable; delete and recreate |
