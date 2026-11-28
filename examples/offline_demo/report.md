# MOCK — pipeline validation only

**Synthetic responses and scores test the software. They are not measurements of any real model and must not be reported as experimental findings.**

The frozen plan contains **88 generation requests** across 1 model configurations and 10 scenarios. The generation count excludes judge calls and does not duplicate shared A responses.

All three conditions provide the full unchanged scenario packet. E1 supplies no preceding answers, E2 supplies the accepted reference answers for the defined parents, and E3 supplies the same model's generated parent answers from the same repetition. A is generated once and shared. The table entries below are percentages, while condition differences are percentage points.

## Coverage

Execution failures, blocked descendants and unresolved judging remain in the planned denominators. The observed score uses fully scoreable responses. The lower-bound score assigns zero to every unscored planned response, combining content performance and availability. Neither view should be reported alone when coverage is incomplete.

| Model | Condition | Planned | Generated | Scored | Scoring coverage | Generation failed | Blocked | Judge failed | Judge unscorable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen35_9b | E1 | 44 | 44 | 0 | 0.0% | 0 | 0 | 0 | 44 |
| qwen35_9b | E2 | 22 | 22 | 0 | 0.0% | 0 | 0 | 0 | 22 |
| qwen35_9b | E3 | 22 | 22 | 0 | 0.0% | 0 | 0 | 0 | 22 |


## Scenario-macro question performance

Question criteria are averaged within each stage, scenario and repetition. Scenarios and repetitions then receive equal weight. BF is an equal-weight average of B and F, excluding A, so the condition comparison has the same target. SD is the standard deviation across complete repetition-level scenario means, not a confidence interval.

| Model | Condition | Stage | Observed criterion score | Lower-bound criterion score | Run SD | Fully correct, lower bound | Any material error, observed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| qwen35_9b | E1 | A | — | 0.0% | — | 0.0% | — |
| qwen35_9b | E1 | B | — | 0.0% | — | 0.0% | — |
| qwen35_9b | E1 | BF | — | 0.0% | — | 0.0% | — |
| qwen35_9b | E1 | F | — | 0.0% | — | 0.0% | — |
| qwen35_9b | E2 | B | — | 0.0% | — | 0.0% | — |
| qwen35_9b | E2 | BF | — | 0.0% | — | 0.0% | — |
| qwen35_9b | E2 | F | — | 0.0% | — | 0.0% | — |
| qwen35_9b | E3 | B | — | 0.0% | — | 0.0% | — |
| qwen35_9b | E3 | BF | — | 0.0% | — | 0.0% | — |
| qwen35_9b | E3 | F | — | 0.0% | — | 0.0% | — |


## Paired condition differences

The table shows the availability-adjusted difference of lower-bound criterion scores for BF in percentage points. This difference is not itself a bound on the unknown complete-data effect. The full CSV additionally reports B and F, complete-case criterion differences and fully-correct differences. Positive oracle-minus-sequential values favour expert preceding answers. The contrast changes the complete preceding answer, including phrasing and completeness, and does not identify an error-only causal effect.

| Model | Contrast | Difference, pp | 95% bootstrap interval, pp | Paired responses | Scenarios |
| --- | --- | --- | --- | --- | --- |
| qwen35_9b | oracle_minus_sequential | +0.00 | [+0.00, +0.00] | 22 | 10 |
| qwen35_9b | sequential_minus_independent | +0.00 | [+0.00, +0.00] | 22 | 10 |
| qwen35_9b | oracle_minus_independent | +0.00 | [+0.00, +0.00] | 22 | 10 |


## Complete chains and transitions

`chain_scores.csv` contains one matched record per selected chain, model, repetition and condition. Only E3 is a generated end-to-end chain. E1 records joint success of independently elicited tasks. E2 records joint task success with expert-parent assistance, even when the model's own preceding response was wrong. Shared nodes are reused rather than counted as extra generations.

`transitions.csv` records whether all required parents and the child are correct. Incorrect-to-incorrect is labelled an association. It is not automatically attributed to propagation of the same error. Unscorable parent or child sets are retained separately.

## Statistical interpretation

Paired intervals use 100 scenario-cluster bootstrap draws with seed 20261127. A scenario contributes its complete graph, and repetition blocks are resampled jointly for the compared conditions. Questions and chain memberships are not resampled as independent units.

- MOCK / PIPELINE TEST ONLY — these records are synthetic and are not empirical model performance.
- Confidence intervals describe resampling of this fixed benchmark's scenarios and repetition blocks; the ten inherited scenarios are not a random sample of military staff work.
- Shared nodes and chains do not create independent observations. Node-level scores count each planned question once.
- Oracle-minus-sequential is the effect of replacing the entire preceding answer with an expert reference, including correctness, completeness, phrasing and length.
- Incorrect-to-incorrect transitions are associations. Attribution of error propagation requires a separate content-linked adjudication.
- Lower-bound scores assign zero to unscored planned requests and therefore mix availability with content success. Interpret them together with observed scores and coverage.
- A difference of two lower-bound scores is an availability-adjusted contrast, not a mathematical lower bound on the unknown complete-data contrast.
- At least one model has only one repetition; run-to-run variability cannot be estimated for that model.
- Some planned outputs are unavailable or unscorable. Complete-case contrasts use a selected subset and must be read with lower-bound contrasts and coverage.


## Files

The JSON report retains the complete metadata and all tables. CSV files expose question scores, execution coverage, scenario/repetition means, stage summaries, paired contrasts, chain outcomes and transitions. No original MA-Bench severity weighting is applied.
