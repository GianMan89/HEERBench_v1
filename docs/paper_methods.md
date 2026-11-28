# Methods text to adapt after the pilot

The following is a protocol description. Replace planned settings with the actual
frozen experimental configuration and add the observed judge-validation results
before presenting it as a completed empirical study.

The accepted HEERBench adaptation comprises 488 questions in ten MA-Bench scenarios
with 112 source documents. The questions consist of 175 information-identification
tasks, 147 assessment tasks and 166 conclusion tasks. They form 150 complete ABF
chains with 313 conjunctive parent sets. Questions can belong to several chains.
Each question is generated once per applicable experimental condition, model and
repetition, and its response is reused wherever that question occurs in the graph.

Three conditions distinguish independent task performance from the effect of
preceding analytical outputs. In the independent condition, the model receives
the original scenario packet and the current question. The reference-assisted
condition additionally supplies the accepted expected answers to all specified
direct parent questions. The sequential condition instead supplies the same
model's corresponding generated answers from the same repetition. Each request
uses a fresh context. The original scenario packet remains available and unchanged
in every condition. Supplied parent answers have identical labels across the two
assisted conditions. A tasks have no parents and are generated once under the
independent condition. All B and F tasks are evaluated in all three conditions.

The planned full evaluation uses three fresh generations for each model and
applicable condition. The full corpus requires 1,114 generation requests per model
and repetition. The default configuration contains eleven target models and
therefore 36,762 generation requests. Optional frontier profiles can be selected
separately. The pilot configuration selects two complete chains per scenario
before observing model performance and uses one repetition, with 168 generation
requests per selected model. Its default target profiles are Qwen3.5-9B and
Gemma 4 31B. The final model set,
provider configurations, output limits, reasoning settings and deviations from the
planned settings must be reported from the frozen run manifests.

Models and repetitions may be collected in separate immutable study segments.
Global repetition identifiers are retained, and resumed runs do not regenerate
completed answers. Combined analyses require identical corpus and task subsets,
condition definitions, generator and judge prompts, global generation settings
and scoring configurations. Different deployments under one model label and
overlapping logical observations are rejected. Prices and credential locations
are excluded from scientific identity. Collection dates and returned model
identifiers remain part of the record because provider-managed model versions
may change between sessions.

The planned primary judge is GPT-6.1 Terra, with Claude Sonnet 5.5 as the secondary
judge. Their settings must be confirmed during expert validation. An automated
judge receives the current question, a candidate answer, accepted
reference information, atomic assessment criteria, applicable error checks and
their supporting source passages. Target-model identity and experimental condition
are withheld. Each criterion and error check requires an explicit verdict.
Unsupported reference assumptions or defensible alternatives outside the reference
coverage can be marked unscorable for human review. The software verifies the
schema, criterion coverage and quoted answer passages before calculating scores.
Local error checks and inherited MA-Bench negative items are reported separately
and are not combined into an arbitrary weighted penalty.

The proposed judge-validation sample comprises 60 real generated answers,
including 10 development and 50 held-out answers. Connected question components
are kept within one partition. One military expert labels each selected criterion
and error check without target-model identity, experimental-condition metadata
or automated verdicts. The expert also participated in corpus construction, which
limits the independence of the reference assessment. Development results may
inform judge revisions; the chosen prompt and configuration are frozen before
held-out comparison. Each answer receives one judge call. Agreement rate, Cohen's
kappa where defined, failure-detection precision/recall/F1, coverage and abstention
are reported against this single-expert reference, separately by label kind.
The design provides neither inter-human agreement nor judge-repeatability estimates.
Its limited sample size and error-category coverage constrain the validation claim;
criterion decisions within the same answer are not independent validation cases.

The secondary judge scores a balanced 20-percent sample within each target-model/
global-repetition group as a sensitivity check. This selection precedes inspection
of primary scores and remains fixed when other models or repetitions are added.
Agreement between the two LLM judges does not establish correctness independently
of the expert reference. Actual sample composition, observed agreement, acceptance
reasoning and deviations must be inserted after the evaluation has been conducted.

Question-level scores are aggregated within scenario, stage and repetition before
macro-averaging. Comparisons between conditions use matched B and F questions.
Scenario-cluster bootstrap intervals preserve all shared questions and dependency
relationships within the resampled scenario. Paired repetition blocks remain
aligned across conditions. Coverage, generation failures, blocked descendants,
scoring failures and unscorable answers are reported alongside content scores.
Incorrect-parent/incorrect-child associations are analysed as diagnostic patterns
and are not automatically classified as propagation of the same error. A difference
between reference-assisted and sequential performance changes the supplied answer
as a whole and does not isolate correctness from its completeness or formulation.

The protocol does not include source-degradation experiments or reasoning-mode
ablations. It evaluates bounded analytical outputs and their dependencies. It does
not establish operational effectiveness or the effectiveness of a human--AI team.
