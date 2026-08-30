# Distribution

An account with few lifetime votes gets no browse traffic, so the first votes come from
the comment and discussion surface rather than from people finding the notebook. Spread
this over two days and lead with the courtesy comments. Six comments in one hour from a
low-vote account reads as spam.

## 1. On pilkwang/rsna-knee-llm-labels

Highest priority, and it is simply correct. That dataset is the only published label set
that reports YES, NO and UNK instead of collapsing uncertainty into a number, and the
whole silence measurement depends on that choice.

> Your three way verdicts turned out to be the only way to measure something I could not
> otherwise get at. Using UNK as the definition of "the report does not say", the finding
> is present 8.2% of the time on average across the 58 annotated studies, and 34.1% of
> the time for Synovitis over 41 studies. A keyword list puts the same number at 27.7%,
> so most of what looks like radiologist silence is the keyword list failing to read the
> report. No other published set lets you separate those two causes. Details and the code
> are in [notebook]. Thank you for shipping the UNK column.

## 2. On the two merged label sets

Not a rebuttal. The author did something reasonable and the consequence is easy to miss.

> Heads up that report_labels_v3.csv reproduces all 58 annotated studies exactly, every
> value 0.0 or 1.0, while none of the other 4,349 rows are exactly 0 or 1. Substituting
> the real labels where they exist is a sensible thing to do for training, so this is not
> a criticism. The catch is that anyone who merges this set and then validates on those
> 58 studies will measure AUC 1.000 and learn nothing, because the 58 are the only
> validation data in the competition. Might be worth a note in the dataset description.
> Two line check is in [notebook].

## 3. Competition discussion post

Title: The 58 annotated studies cannot rank your label set

Carry the two headline tables, the language agreement table and the bootstrap interval
table, plus the link to the CC0 confidence dataset. The interval result is the part that
applies to everyone regardless of which label set they use.

## 4. On the widely forked baseline

Report the clean negative with its number, which is the cheapest goodwill available.

> Worth knowing before you tune on the 58: a 58 row AUC has a 95% bootstrap interval
> about 0.23 wide, while the honest published label sets span only 0.087. So the ordering
> between label sets is inside the noise. Cross-labeler agreement needs no annotations
> and covers all 4,407 studies, and it falls from 0.819 in English to 0.685 in Bulgarian.

## 5. Where a reconciliation is owed

One published training notebook already says "NO flips (laterality is signal)" in a
comment. That is the same conclusion reached here about horizontal flips, arrived at
independently. Credit it rather than presenting the point as new.

## Timing

The competition closes 2026-10-22. Notebooks stop accruing votes almost immediately after
a competition closes, so publish before the work feels finished.
