# Outreach drafts - review before posting

Nothing below has been posted. Comment on the notebook, do not DM.

---

## 1. adarsh1077 / s6e8-diversity-beats-strength (54 votes) - HIGHEST VALUE

Why: his title claims the opposite of my result. Handled as a reconciliation, not a rebuttal,
because both measurements are correct in their own regime. This is the comment most likely to
produce a real exchange.

> Your logistic regression at 0.9589 outranking eight stronger trees is the part I keep thinking
> about, because I measured what looks like the opposite and I think both results are right.
>
> I ran a strength-matched control from a 2-tree base: same trainer, same schedule, one fed the
> trees' own matrix and one reading exact-value embeddings. They came out equally decorrelated from
> the trees (0.9665 vs 0.9658) and differed 44x in what they added (+0.000013 vs +0.000577). The
> most decorrelated member in my pool contributed 0.000000.
>
> The difference is pool saturation. You are adding to 177 members that already cover the strong
> direction densely, so the marginal thing worth having is a new direction at almost any strength.
> I was adding to two trees, where a weak member has nothing to be marginal to. If that is right
> then "diversity beats strength" is a statement about pool size rather than about models, and it
> should reverse as the pool shrinks. Worth testing: refit your hill-climb on a 5-member subset and
> see whether the LR keeps its weight.
>
> Full measurement here if useful: <NB1 URL>

---

## 2. raykkretzschmar / why-every-s6e8-notebook-above-0-97110-overfits (64 votes)

Why: I already changed my own submission strategy because of this notebook. I can hand him a
positive control he does not have - evidence that OOF-fit weights *do* generalise, which is the
other half of his argument.

> This changed which two submissions I am selecting, so thank you for posting it.
>
> One measurement you might want as a positive control, since your notebook establishes that
> public-fit selection does not generalise but not that OOF-fit weights do. On a 7-member pool I
> fit hill-climb weights on folds 0-2 only and scored them on folds 3-4: 0.969656, against
> 0.969668 for weights fit on all five folds. A 0.000012 penalty for never seeing the evaluation
> rows. So the weight-fitting step itself is honest at this pool size, which means the overfitting
> you are describing is specifically located in the *selection among candidate submissions* step,
> not in the blending.
>
> That distinction matters practically: it says keep blending on OOF, just stop choosing between
> near-twins on public feedback.

---

## 3. tamerlanomralinov / s6e8-lookup-transformer-insights-lb-0-97041 (37 votes)

Why: I ported his architecture. Reporting results back is basic courtesy and the highest-
conversion comment on the list.

> I ported your Lookup-Transformer and retrained it from scratch on my own folds. It came out level
> with a tuned XGBoost (0.968305 vs 0.968454 OOF) and it is by a wide margin the most valuable
> member in my blend: +0.000577 on top of xgboost + lightgbm, where an MLP on the same encoded
> features that the trees use gave +0.000013.
>
> Three things I measured that might save you time:
>
> - More capacity hurt. d=192, L=6 gave 0.968147 against 0.968293 for the smaller config. The
>   lookup table is the part doing the work.
> - Heavier augmentation hurt. aug=0.25 gave 0.968154.
> - If you use an EMA warm-started from initialisation, warm it up. 0.999^500 = 0.61, so a short
>   run evaluated through a cold EMA scores below chance and looks exactly like a flipped label.
>
> Credited and linked in my writeup: <NB1 URL>

---

## 4. cdeotte / simple-xgb-starter (34 votes)

Why: a clean negative on his suggested features. Short, factual, high visibility.

> Tested your three ratio features (opens_per_screen_hour, engagement_events,
> sleep_minus_screen) on a frozen 5-fold split against an otherwise identical baseline. Fold 0
> went 0.968028 -> 0.968001, and the version with them needed about 1,200 more trees to get there.
> So on this dataset they are a wash at best, at least on top of per-level target encoding, which
> may already be capturing the same interactions.
>
> Posting the negative since the starter is where most people will begin.

---

## 5. tomasa2 / s6e8-what-moved-the-score-and-what-didn-t (46 votes)

Why: complementary negative results, no overlap with his.

> Adding four negatives from my own runs, all on one frozen 5-fold split so they are comparable to
> each other:
>
> - 40-trial Optuna sweep on XGBoost: 0.966850 against the incumbent's 0.966853 on an identical
>   fold and subsample. The hyperparameter surface here is flat.
> - Row-level NaN-count features: +0.00002.
> - Wider/deeper Lookup-Transformer (d=192, L=6): -0.00015.
> - Google's TabFM zero-shot, the most decorrelated model I could obtain: contributes exactly
>   0.000000 to my pool.
>
> The one that did move: 3 seeds averaged inside each fold, +0.00043, and 5 folds to 10 folds,
> another +0.00018. Both variance reduction rather than insight, which is why they are dependable.

---

## 6. Competition discussion post

Title: `Correlation does not predict what a blend member is worth (and 11 free neural OOF vectors)`

> Most member-selection advice in this competition is some form of "add something decorrelated from
> your trees". I built the control that tests it and the advice did not survive.
>
> Same trainer, same optimizer, schedule, augmentation and EMA. Two neural members, differing only
> in what they are allowed to read:
>
> | member | solo OOF AUC | Spearman vs trees | added on top of xgb+lgbm |
> |---|---|---|---|
> | MLP on the trees' own matrix | 0.965032 | 0.9665 | +0.000013 |
> | Lookup-Transformer, per-value embeddings | 0.968305 | 0.9658 | **+0.000577** |
> | MLP on magnitudes + masks only | 0.941125 | **0.9416** | +0.000008 |
>
> Equally decorrelated, 44x apart in contribution. And the *most* decorrelated member in the pool
> contributes nothing, because any sufficiently weak model decorrelates by producing noise.
> Correlation is only interpretable between models of comparable strength.
>
> What this means in practice: when you add a neural network to a GBDT blend, the binding
> constraint is its solo score, not how exotic the architecture is. It is already unlike your
> trees. That part was free. It needs to be as good as them.
>
> Two things you can take:
>
> - The notebook, which runs end to end on a T4 with internet off, including the leave-one-out and
>   incremental ablation you would need to run this test on your own pool: <NB1 URL>
> - **S6E8 OOF Library: 11 Neural Members** (CC0): out-of-fold and test predictions for 11 neural
>   models - Lookup-Transformers, RealMLP-TD, TabM - on a frozen
>   StratifiedKFold(shuffle=True, random_state=42) split, row-aligned to train.csv. Most published
>   OOF libraries here are boosted-tree-heavy, so this is deliberately the other half:
>   https://www.kaggle.com/datasets/paiky1995/s6e8-oof-library-11-members
>
> Architecture credit to @tamerlanomralinov and @zhenruiweng, whose notebooks I ported and
> retrained.

---

## URLs to substitute

- NB1: https://www.kaggle.com/code/paiky1995/s6e8-correlation-does-not-predict-contribution
- NB2: https://www.kaggle.com/code/paiky1995/s6e8-tabfm-zero-shot-on-0-7-of-the-data
- Dataset: https://www.kaggle.com/datasets/paiky1995/s6e8-oof-library-11-members

## Posting order and timing

Post 3 (tamerlanomralinov) and 4 (cdeotte) first - courtesy comments, no argument, and both
authors are active. Then 6 (discussion post). Then 1 (adarsh) and 2 (rayk), which are the ones
that start conversations and are worth having in your feed while you can reply. Then 5.

Spread over two days. Six comments in one hour from an account with 21 lifetime votes reads as
spam to Kaggle and to the readers.
