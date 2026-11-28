# Experimental protocol

This protocol fixes the measurement target and experimental comparisons for the
accepted HEERBench MA-Bench adaptation. It describes the intended study. A working
pipeline, a completed pilot and an expert-validated automated judge are separate
milestones. Repository tests and mock runs do not constitute experimental results.

## 1. Research questions and scope

**RQ1. Individual task performance.** How do the selected LLMs perform on military
staff reasoning tasks when each task is elicited independently from the scenario
packet?

**RQ2. Dependence on preceding outputs.** How does downstream performance change
when the model receives its own preceding analytical outputs or expert reference
outputs, while the original scenario packet remains available?

The study evaluates observable answers to bounded training-scenario tasks. It does
not estimate effectiveness of a human–AI team, validate autonomous command, or
recover a model's internal reasoning. The present configuration excludes the
earlier RQ3 on degraded or conflicting source information and excludes
reasoning-mode ablations.

The task construct preserves the distinction agreed with the military reviewers.

| Stage | Required function of the answer | Boundary |
|---|---|---|
| A, *Ansprechen* / Identify | Identify relevant information stated in the supplied material. | An existing assessment or order can itself be an explicitly stated fact. |
| B, *Beurteilen* / Assess | Assess the significance of information, such as an implication, risk, dependency or mismatch. | Retrieving a stated implication is not automatically a new assessment. |
| F, *Folgern* / Conclude | Derive a concrete consequence for planning or support, such as a task, resource requirement, coordination requirement or decision prerequisite. | A summary or generic expression of uncertainty is insufficient. An information requirement must identify the missing fact and the decision that depends on it. |

The original MA-Bench extraction, synthesis and inference tiers remain separate
provenance metadata. They are not recoded into A/B/F or treated as equivalent
constructs. The adaptation measures explicitly elicited tasks rather than the
ability to produce a complete mission-analysis brief. The source and construction
records are described in `data/heerbench_v05/method/CONSTRUCTION_METHOD.md`.

## 2. Frozen corpus and units

The bundled v0.5 snapshot contains 488 questions across ten scenarios and 112 source
documents, comprising 175 A, 147 B and 166 F questions. There are 150 complete
chains and 313 conjunctive dependency sets. Each of the 313 B/F questions has one
defined parent set. Sixty-four questions occur in more than one chain. These
counts are derived from the bundled records and checked by the dataset validator.

Every experimental request is identified by the frozen configuration, model,
repetition, question and condition. The same question is generated once per
condition and repetition, regardless of the number of chains that contain it.
Chain memberships are analysis views over this shared graph. They are not
independent replications and do not create additional billable responses.

The acceptance record documents the user's reported joint review with a military
colleague. Historical construction statuses are preserved as provenance. The
repository does not invent independent annotations, inter-rater agreement or
human performance scores from that acceptance.

Before generation, freeze the dataset fingerprint, selected chain IDs, question
IDs, dependencies, model configurations and generator prompt. A pilot selection
must contain complete chains and all their required parents. Any later correction
to a question, source, criterion or dependency creates a new version and requires
identifying the affected responses and comparisons.

## 3. Conditions

| Condition | Current question | Original scenario packet | Direct parent outputs |
|---|---|---|---|
| E1, independent | Unchanged | Complete, unchanged | None |
| E2, reference supported | Unchanged | Complete, unchanged | Accepted expert reference answer for each defined parent |
| E3, sequential | Unchanged | Complete, unchanged | Same model's generated answer for each defined parent in the same repetition |

A questions have no supplied analytical parents. They are generated once under
E1 and reused as the A outputs of E3. Their scores can appear in E1/E2/E3 chain
views, but they are not three independent observations. In E2 the downstream
input is the expert reference, not the reused model-generated A answer.

Each request starts from a fresh message context. The packet, question wording,
generator instruction and output allowance remain fixed within a model across
conditions. Only the specified parent-output field changes. Parent labels identify
the preceding question without revealing whether its answer is generated or an
expert reference. E3 carries the actual parent text, including mistakes, without
correction, selection of the best run or substitution from another model.

All parents in a conjunctive dependency set are supplied together. Only defined
direct parents are inserted. A chain is not implemented as an accumulating chat
transcript. B is generated after its A parents and F after its required parents.
The model can consult the full packet and can reject an erroneous preceding
answer. Consequently, the treatment measures susceptibility to or benefit from a
supplied analytical result when the underlying information remains accessible.

An upstream API failure is not an erroneous analytical answer. The dependent E3
request is blocked until a valid parent response exists. It is not silently given
an empty string, a reference answer or a parent from a different repetition.
Blocked calls remain visible in the planned denominator.

For the complete corpus, the number of generation requests per model and
repetition is

\[
N_A+3(N_B+N_F)=175+3(147+166)=1,114.
\]

For the eleven default models and three repetitions this is 36,762 planned generation
requests. This arithmetic does not include judge calls, validation calls,
retries, token differences or provider discounts. The executable plan is the
authoritative count for any selected model panel or pilot.

## 4. Models, repetitions and execution

The model panel should include different model families and, where documented,
different parameter scales and dense or mixture-of-experts architectures. Record
parameter count and architecture as *undisclosed* where the provider does not
publish them. A comparison between different families cannot isolate the causal
effect of parameter count or architecture because training data, optimisation,
tokenisation and post-training also differ.

The default pool comprises eleven cost-oriented target profiles. An optional
extended pool supports later frontier comparisons; Grok is outside the default
pool. Exact model identifiers, checkpoints where available, serving precision,
architecture disclosures and source links are recorded in
`configs/provider_catalog.yaml` and [the model catalogue](model_catalog.md).
Selecting additional profiles is explicit and does not submit the entire catalogue.

Use the same predeclared generation settings for a given model across all
conditions. Record the provider, exact model identifier, endpoint, date, exposed
sampling settings, output budget, context allowance and returned model/version
metadata. A provider default or unsupported setting must be recorded explicitly.
Do not silently replace a model, change its reasoning setting or shorten packets
to make a run finish. Input packets must fit the selected model's context budget.

The main configuration uses **three complete repetitions**. A repetition produces
a new response for every planned question and condition. It does not reuse an
answer from an earlier repetition. Five repetitions are an optional design
change to be fixed before the main run. Three repetitions permit a basic report
of run-to-run variation but do not guarantee a particular statistical power or
effect-size resolution. Even nominally deterministic API settings can produce
variation. Where an API supports seeds, the seed is recorded but is not claimed
to make different models' random draws equivalent.

Request ordering is reproducibly shuffled within execution waves. Archive
request/response timestamps and retry events. Transport failures can be retried
under the same scientific request ID. A retry is not a new repetition, and a
response must not be retried merely because its content is poor. Persist responses
before continuing the dependency graph, so resuming a run does not rerun successful
generation requests.

Collection may proceed in immutable study segments, adding selected models or
global repetition identifiers later. The default main configuration requests native
batch execution where implemented, while the pilot uses synchronous execution.
Compatible endpoints without a native batch API use synchronous calls at standard
rates. Within a study, preceding analytical outputs always come from the same
model and global repetition. Study names do not override scientific compatibility.
Combined reporting verifies the complete declared task set and frozen deployment,
prompt, corpus and judge settings. See [incremental reporting](incremental_reporting.md).

## 5. Judge preparation and use

An automated judge evaluates one candidate answer at a time against the accepted
required criteria, acceptable alternatives and applicable local/inherited error
checks. It receives the question, reference exemplar and the associated source
excerpts with their document/line locators. It does not receive the generating
model, condition, repetition or another candidate's score. Model and condition
metadata remain in coordinator records outside the blinded judge payload.

The reference answer is an exemplar, not a required exact string. A supported
alternative that meets the criteria is acceptable. The judge returns a structured
verdict for every criterion and error check, with a short justification. A schema
validator checks IDs, completeness and allowed verdicts before scoring. Malformed
JSON, missing verdicts and provider failures are execution failures. A valid
`unscorable` verdict instead identifies a substantive ambiguity or insufficient
reference coverage requiring review. Neither is automatically called a model
content error.

The default preparation panel contains **60 distinct generated answers**, selected
without consulting their scores. Marginal coverage is sought across scenarios,
stages, models and conditions. Whole connected components of the chain graph are
assigned to development or held-out use, with **10 development answers and 50
held-out answers**. Shared questions and linked chain members cannot cross the
split. The achieved balance must be reported. Disconnected components can still
share scenario passages. Select whole-scenario grouping if transfer to unseen
packets is the intended claim. An infeasible sample is an error, not permission
to silently reduce the requested size.

**One military domain expert** records all selected criterion and error-check
labels in `reviewer_1.csv`, before inspecting automated verdicts and without
model/condition metadata. These labels form the single-expert reference. There
is no second expert, no adjudicated consensus and no estimate of inter-human
agreement in this mode. The expert's participation in corpus construction must
be disclosed, since blinding model identity does not make the reference assessment
independent of benchmark design.

Only development results inform changes to the judge prompt or interpretation
rules. Freeze the judge before examining held-out agreement. The default makes
one judge call per selected answer, so it does not measure repeated-call stability.
Report agreement with the single-expert reference, Cohen's kappa where defined,
class-specific detection of failed criteria and errors, coverage and abstentions.
High agreement on common satisfied criteria does not establish sensitivity to
rare detrimental assertions. Multiple labels within an answer are dependent and
must not be presented as additional independent validation cases.

This smaller design supports a scoped preliminary check. Neither the sample
size nor a universal agreement threshold automatically establishes suitability.
The expert records the actual decision and limitations after reviewing the
held-out report. If those results drive another revision, a fresh held-out panel
is required. Reference defects require an explicitly versioned corpus correction.

The optional two-expert mode retains separate independent worksheets and an
adjudicated reference. The earlier 200-answer design, with 50 development and
150 held-out answers and three judge calls per answer, can be requested explicitly.
Only that mode supports inter-human agreement, and judge repeatability requires
at least two actual scoring calls on the same answers.

The primary judge profile is GPT-6.1 Terra and the secondary profile is Claude
Sonnet 5.5. Their exact serving and inference settings are frozen in the catalogue.
These are candidate scoring instruments, whose suitability still requires the
held-out expert validation described above.

For the main run, use one frozen primary judge per answer. A reproducibly selected
20% sample within each target-model/global-repetition group is scored by the second
judge from a different model family for
sensitivity analysis. The second judge does not replace expert calibration and
its verdicts are not pooled as additional repetitions. The primary model's
responses receive the same blinded scoring protocol as every other model. Assess
possible family-related scoring differences on the human panel and secondary
sample. LLM-judge biases documented by Zheng et al. motivate these checks, but
their general chat evaluation does not validate a judge for military staff work
([paper](https://arxiv.org/abs/2306.05685)).

The secondary sample is selected before inspecting primary scores, after the
relevant generation group has finished. Marginal balance is sought within the
group. Adding a model or repetition does not redraw previously frozen samples.
Its nominal size is `ceil(0.2 * eligible_answers)` for each group, so rounding and
failed generation requests affect the achieved overall percentage.

## 6. Measurements and statistical analysis

The primary content score is the fraction of required atomic criteria satisfied
for each question. Complete-answer success requires all required criteria and no
applicable material error. Local errors and inherited MA-Bench error-check
incidence remain separately available. Original MA-Bench severity weights are not
used to construct a new weighted aggregate. See `metrics.md` for formulas,
denominators and output columns.

Report A, B and F performance separately. The common target for comparing all
three conditions comprises B and F questions. Average questions within each
stage/scenario/repetition, give B and F equal weight, then average scenarios and
repetitions equally. Larger scenarios and questions with more criteria therefore
do not automatically dominate the principal score.

The primary paired comparison is **E2 minus E3**, with **E3 minus E1** as the
complementary comparison of sequential and independent elicitation. E2 minus E1 is
also exported. Pair exactly the same question, model and repetition. Report effect
sizes, complete-case coverage and the corresponding conservative scores that keep
every planned request in the denominator. Report per-repetition means and sample
standard deviations. The 95% percentile intervals use 10,000 seeded resamples of
whole scenarios and joint repetition blocks. Questions and chain memberships are
not resampled as independent observations.

These intervals describe sensitivity to the composition of the fixed benchmark
and stochastic repetitions. Ten inherited scenarios are not a random sample of
military staff work. Additional repetitions do not increase the number of
independent scenarios. The code does not turn the many criteria or overlapping
chains into a large independent sample or provide unadjusted significance tests
for every possible model comparison.

Report all-node success for each chain, retaining the shared-node structure. E3
provides an actual sequential chain score. E1 provides joint success on independently
elicited tasks. E2 provides joint task success with reference-supported downstream
answers, not an end-to-end chain built from the model's own preceding answers.

## 7. Interpretation of dependencies

Distinguishing individual competence from composition has methodological precedent
in the compositionality gap studied by Press et al.
([paper](https://aclanthology.org/2023.findings-emnlp.378/)). The present treatment
replaces a complete preceding answer with an expert reference. The resulting
E2–E3 difference includes changes in correctness, completeness, phrasing, length
and salience. It is not a pure causal estimate of the effect of a single upstream
error. References may also supply more useful information than a correct but
shorter generated answer.

For each E3 dependency, report whether all parents and the child are correct.
An incorrect parent set followed by an incorrect child is an association. Calling
it propagated error requires a separately adjudicated content link between the
specific upstream mistake and the downstream failure. A correct child after an
incorrect parent set may reflect correction from the full packet or non-use of
the parent, rather than verified error recovery. A correct parent set followed by
a wrong child is a downstream failure under correct supplied parents, rather than
proof of a particular internal mechanism.

## 8. Pilot, cost controls and release

The default pilot selects two complete chains per scenario, giving 20 chains,
and uses Qwen3.5-9B and Gemma 4 31B with one repetition. For the fixed default
selection this requires 168 generation calls per model, or 336 in total. The
executable plan verifies the actual count. The selection is fixed before
viewing model scores. It checks context limits, parent handoffs, API records,
grading clarity, costs and engineering failures. It cannot estimate run-to-run
variation with one repetition. If pilot inspection changes the corpus, prompt,
judge or settings, version those changes and keep the earlier run separate from
the main experiment.

Scientifically compatible savings include reusing shared question outputs within
a repetition, generating A only once, native provider input caching and supported
batch endpoints. A cached input prefix does not mean caching model answers across
repetitions. Batch submissions must obey the graph's dependencies, and pricing
discounts are only applied for the actual supported endpoint and returned usage.
Judge payloads use the validated excerpts for the relevant question rather than
sending the entire packet repeatedly. Reducing the model panel or repetitions is
an explicit design change, not an invisible optimisation.

For publication, archive the frozen configuration and manifest, corpus and prompt
fingerprints, selected chains, raw generations, parent request links, judge
verdicts, human validation records, unresolved failures, provider usage and price
snapshot, analysis tables and scripts. Keep model-identifying assignment files
separate while human review is blinded. A public release must retain source
attribution and the applicable dataset rights described in the repository's data
documentation. Report mock and real runs separately. No API experiment or judge
validation is asserted to have been completed by distributing this code.
