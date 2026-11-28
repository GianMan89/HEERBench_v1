# Metrics, estimands and report fields

`analysis.py` joins primary-judge records to the **frozen request plan**. Requests
that failed, are blocked by an upstream failure or remain pending stay in that
plan. Secondary-judge records are excluded from the primary estimate. Distinct
chain memberships never duplicate a question score.

## Question scores

For question \(q\), let \(K_q\) be its required atomic criteria and \(J_q\) its
applicable material-error checks. The positive score is

\[
s_q=\frac{1}{|K_q|}\sum_{k\in K_q}\mathbf{1}[k\text{ satisfied}].
\]

All required criteria have equal weight. Optional criteria do not enter the
denominator. An answer satisfies complete-answer success only if every required
criterion is satisfied and every applicable error check is absent.

\[
c_q=\mathbf{1}[\forall k\in K_q,\ k\text{ satisfied}]
     \mathbf{1}[\forall j\in J_q,\ j\text{ absent}].
\]

The judge allows `satisfied`, `not_satisfied` and `unscorable` for a positive
criterion, and `error_present`, `error_absent` and `unscorable` for an error check.
An unscorable item is not automatically a failed criterion or absent error.
The strict primary analysis requires a successful generation, a valid completed
judge record and resolved verdicts for all applicable required criteria and
error checks. A partially resolved answer can contain diagnostic information in
the raw judge record but is excluded from the observed primary score.

| Field | Meaning |
|---|---|
| `criterion_score` | Fraction of required criteria satisfied, on fully scoreable answers. |
| `fully_correct` | All required criteria satisfied and no applicable material error. |
| `has_error` | At least one applicable local or inherited error is present. |
| `negative_error_rate` | Fraction of applicable inherited MA-Bench checks triggered. Null if no inherited checks apply. |
| `n_local_errors_present` | Count of triggered question-specific local errors. |
| `n_inherited_errors_present` | Count of triggered inherited MA-Bench checks. |
| `scoreable` | All required conditions for the strict content score are met. |
| `generation_status` | `succeeded`, `failed`, `blocked` or `pending`. |
| `judge_status` | `succeeded`, `failed`, `unscorable` or `pending`. |

Negative error checks are only applied where linked to that question. Original
MA-Bench penalty/severity values remain provenance, not new scoring weights.
Questions without an applicable inherited error check are not assigned an
artificial zero error rate. Omission and assertion errors remain distinct.

## Missingness and denominators

Let \(R_q=1\) when the planned answer is strictly scoreable. Two summaries are
reported together.

\[
\bar s_{\mathrm{observed}}=
\frac{\sum_q R_qs_q}{\sum_q R_q},\qquad
\bar s_{\mathrm{lower}}=
\frac{\sum_q R_qs_q}{N_{\mathrm{planned}}}.
\]

The observed score is conditional on scoring availability. The conservative
`lower_bound` score assigns zero to every unscored planned request and is a lower
bound on the score if the unknown values lie in [0,1]. It combines content
success and execution availability. It is not a pure measure of model reasoning.
The same definitions apply to complete-answer success. A difference between two
lower-bound scores is an **availability-adjusted contrast**, not necessarily a
lower bound on the complete-data contrast. Failures are additionally listed by
generation and judge status rather than being relabelled as content errors.

`scoring_coverage` is the number of strictly scored answers divided by planned
answers. `any_error_observed` and `inherited_error_rate_observed` are conditional
on available judgments and should not be read as if missing answers contained no
errors. `questions_with_inherited_error_rate` identifies the smaller denominator
for inherited-check incidence.

## Stage and scenario weighting

Within each model, condition, scenario, repetition and stage, average scores over
unique planned questions. For the common B/F target, average the B and F means
equally, then average scenarios and repetitions equally.

\[
\bar s_{m,e}=
\frac{1}{T}\sum_{t=1}^{T}\frac{1}{S}\sum_{s=1}^{S}
\frac{\bar s_{m,e,s,t,B}+\bar s_{m,e,s,t,F}}{2}.
\]

This definition avoids giving large scenarios, stages with more questions or
questions with more criteria automatic extra weight. A is reported separately
under E1 because the identical generated A answer is shared across chain views.
The `coverage` table also gives pooled request counts for operational accounting.
Its pooled score is not the principal scenario-balanced performance estimate.

For conditional observed scores, a scenario/repetition block must have an
observed value for both B and F to enter a B/F mean. Empty stages do not silently
receive zero or disappear from the equal-stage formula. The run tables record
the number of contributing scenarios. Always accompany a conditional estimate
with coverage and the all-planned conservative estimate.

## Paired comparisons and intervals

Condition differences are formed for the same model, question and repetition.
The differences, rather than two independently averaged model scores, are then
aggregated with the stage/scenario scheme above.

| Contrast | Interpretation |
|---|---|
| `oracle_minus_sequential`, E2−E3 | Benefit or harm of replacing generated parents with expert reference outputs. |
| `sequential_minus_independent`, E3−E1 | Benefit or harm of explicitly supplying generated parents when the packet remains available. |
| `oracle_minus_independent`, E2−E1 | Benefit or harm of reference assistance over independent elicitation. |

`paired_effects.csv` includes a conservative criterion contrast, a paired
complete-case criterion contrast and a conservative complete-answer contrast for
B, F and the stage-balanced B/F target. `planned_pairs`, `used_pairs`, and
`unmatched_left_requests` disclose the matched and analysed denominators. For a
complete-case B/F estimate, pairs in a block without either B or F observations
are not counted as used pairs. Null means unestimable, not zero.

The implementation first forms a paired difference for each
scenario/repetition block. For each of 10,000 seeded bootstrap draws it samples
whole scenario blocks with replacement, then samples the paired repetition
blocks within each drawn scenario with replacement. Every constituent question,
shared node and chain stays together within the selected block. The 2.5th and
97.5th percentiles are reported. Individual criteria and shared chain memberships
are never independent bootstrap draws. A single scenario with a single
repetition cannot supply a meaningful resampling interval and returns null.

These are descriptive intervals for the fixed benchmark's scenario composition
and repetitions. They do not establish representative sampling of military
problems. Sparse repetitions and ten scenarios limit interval stability. No
multiple-comparison-adjusted hypothesis-testing claim is produced automatically.

Run-to-run variability uses the sample standard deviation of per-repetition
scenario-balanced means. It is null with one repetition. It is not a confidence
interval or a measure of inter-scenario variation. Plot error bars for stage
performance are labelled ±1 run SD, while paired-condition plots use bootstrap
intervals.

## Chain scores

For a chain \(h\) with unique member questions \(V_h\), joint success requires
every member answer to satisfy complete-answer success.

\[
C_h=\prod_{q\in V_h}c_q.
\]

If any member is unavailable or unscorable, observed success is null and
conservative success is zero. A shared node can affect several chains, which is
appropriate for the graph but does not increase the effective sample size.

| Condition | Exported label | Meaning |
|---|---|---|
| E1 | `independent_joint_task_success` | All members correct when elicited independently. |
| E2 | `oracle_assisted_joint_task_success` | All member outputs correct, with downstream tasks receiving expert parents. |
| E3 | `sequential_chain_success` | All members correct in the actual generated handoff sequence. |

E2's generated A and B outputs are assessed, but its children receive expert
references. Its joint score therefore differs from actual sequential chain
success. Compare task and chain results without presenting the three labels as
identical execution constructs. The chain selection comes from the frozen plan,
not from whichever additional chains happen to be covered by overlapping pilot
nodes.

## Dependency transitions

Each normalized E3 dependency set is counted once per model and repetition.
`parent_set_correct` requires all its parents to be correct. The export records
correct→correct, correct→incorrect, incorrect→correct,
`incorrect_to_incorrect_association`, and unscorable transitions.

Incorrect→incorrect is not an automatic error-propagation rate. To support a
claim of propagated errors, a separately defined annotation would need to link
the specific parent error to the specific child error. This repository does not
manufacture that semantic link from binary correctness. Similarly, a correct
child with an incorrect parent does not prove that the child detected and
repaired the mistake.

## Generated artifacts

| File | Content |
|---|---|
| `analysis.json` | All tables, design metadata and interpretation warnings. |
| `report.md` | Readable coverage, stage scores and primary contrasts. |
| `coverage.csv` | Planned/generated/scored counts and failure categories. |
| `question_scores.csv` | One primary-score record per planned generation. |
| `scenario_run_scores.csv` | Stage and B/F scores within each scenario and repetition. |
| `run_scores.csv` | Scenario-balanced score for each repetition. |
| `stage_scores.csv` | Mean and run-to-run SD for each stage and condition. |
| `paired_effects.csv` | Matched contrasts, coverage and bootstrap intervals. |
| `chain_scores.csv` | Success and availability of each selected chain. |
| `chain_scenario_run_scores.csv` | Chain summaries within scenario and repetition. |
| `transitions.csv` | Observed E3 parent-set/child correctness associations. |
| `stage_performance.svg` | Optional stage/condition plot with run SD. |
| `paired_conditions.svg` | Optional paired-condition contrasts with intervals. |

The JSON export uses null for unavailable values and rejects non-finite scores.
CSV files use UTF-8 with a byte-order mark for convenient spreadsheet import.
Mock runs carry `MOCK_PIPELINE_TEST_NOT_MODEL_RESULTS` in metadata and prominent
warnings in prose and figure titles. Such outputs test the pipeline and must not
be inserted into a paper as empirical model findings.
