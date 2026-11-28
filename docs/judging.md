# LLM judging and validation

The automatic judge checks a generated answer against the approved task criteria and
the supplied scenario passages. A program then calculates the scores. The judge does
not invent a global quality rating. Acceptance of the benchmark by the military experts
does not establish agreement between an automatic judge and those experts. That second
validation requires actual generated answers and human annotations recorded before
viewing automated verdicts. The default uses one expert, with the limits described below.

This repository supplies the judging implementation, blinded annotation export,
reproducible sample selection and agreement calculations. It contains no completed
human validation, no approved judge and no real model performance results.

The primary profile is GPT-6.1 Terra (`gpt61_terra`), and the secondary
profile is Claude Sonnet 5.5 (`claude_sonnet55`). Their exact API identities,
inference settings and price snapshots are recorded in the model catalogue.
These selections are provisional scoring instruments until the designated expert completes
the validation below. Keys are read from the corresponding ignored local files.

## One answer, one transparent assessment

`build_judge_request` supplies the following information to the judge.

1. The question and the candidate's complete answer.
2. The reference answer as an exemplar, its acceptable alternatives and every atomic
   positive criterion, including the required/optional flag.
3. The question's local error conditions and the applicable inherited MA-Bench negative
   checks, with their applicability statements.
4. The validated source excerpts, document identifiers, titles and line ranges.

The target model, experimental condition, repetition, previous score and original
MA-Bench assessment wording are excluded. Model and condition metadata remain in the
coordinator's result file for analysis. Self-identification that happens to occur inside
a generated answer is not edited out, because that would change the object being judged.
The judge is instructed to ignore that information. Full blinding therefore cannot be
guaranteed from answer style or content alone.

The judge does not receive preceding generated answers separately. It judges the final
observable answer against the same task reference and original scenario passages under
E1, E2 and E3. Consequently, accepting an incorrect preceding assessment is not excused
by changing the reference. Analyses of dependencies use the recorded upstream and
downstream scores afterwards.

All embedded text is marked as untrusted evaluation data. The prompt forbids instructions
inside an answer or source from changing the rubric. This separation and strict output
validation reduce straightforward evaluation manipulation, but are not a proof of
immunity to prompt injection. Suspicious cases remain reviewable from stored inputs.

For each positive criterion, the judge returns `satisfied`, `not_satisfied`, or
`unscorable`. For every error check, it returns `error_present`, `error_absent`, or
`unscorable`. Every verdict includes a short justification and, where the answer text is
the basis, an exact quotation from that answer. Source identifiers must come from the
request. Private chain-of-thought is neither requested nor scored.

A required but omitted statement is `not_satisfied`. A plausible alternative that the
supplied sources do not resolve, a defective reference, or an ambiguous interpretation
is `unscorable` and requires review. The reference is not an exact-match answer. Correct
equivalent wording and the approved alternatives can satisfy a criterion. Outside facts
must not be used to repair the fictional scenario.

A provider can successfully return an empty answer. It remains a content outcome,
including any omission errors, and is eligible for the validation panel. It is not
relabeled as a transport failure. The judge evaluates its missing content explicitly.

### Source scope

The default judge sees the validated task-specific source excerpts, not the complete
scenario packet. This keeps judge input bounded while exposing the actual passages used
to construct the task. The generator may use the complete packet. If an alternative
answer depends on a passage outside the judge bundle, the judge should abstain and the
reviewer should inspect the full packet. Such cases must not be counted automatically as
hallucinations. A systematic need for additional passages requires a versioned source
bundle and renewed validation of the affected judge configuration. The accepted sources
must not be expanded selectively only for one evaluated model.

## Programmatic scores and missingness

`validate_judgement` requires exactly one result for every expected criterion and error
identifier. It rejects missing checks, duplicate or invented IDs, unexpected fields,
invalid labels, invented source IDs and quotations absent from the candidate answer.
Malformed responses are judge protocol failures, not wrong model answers.

`score_judgement` calculates the following values.

| Field | Meaning |
|---|---|
| `criterion_score` | Fraction of required positive criteria satisfied. Undefined if any required criterion is unscorable. |
| `fully_correct` | All required criteria satisfied and every applicable error absent. A known failure gives false. Unresolved items without a known failure give undefined. |
| `has_error` | Any applicable error present. Undefined if no error is known to be present but at least one error check is unscorable. |
| `negative_error_rate` | Fraction of inherited negative checks triggered. Undefined when none apply or an inherited check is unscorable. |
| `scoreable` | Every positive and error check has a scorable verdict. |
| lower/upper criterion bounds | Descriptive bounds obtained by treating unknown required criteria as unsatisfied/satisfied. They are not substituted for observed scores. |

Local and inherited error counts are reported separately. Overlapping checks are not
summed into a new severity-weighted penalty. A positive omission, a detrimental
assertion, a failed generator request and a failed judge request remain distinct events.

The runner may retry transport failures or invalid judge JSON under the fixed retry
policy. A valid `unscorable` verdict must not be retried until a favourable label appears.
The denominator and coverage must include unavailable judgements in the operational
report even where a conditional metric uses only scorable answers. See `analysis.py`
and the analysis documentation for final aggregation.

`mock_judgement` deliberately marks every item unscorable. Offline smoke runs verify
the software workflow and cannot be read as scientific scores.

## Single-expert preparation and held-out validation

The default panel contains **60 actual generated answers**, with **10 development
answers** and **50 held-out answers**. One military expert labels the selected
criteria and error checks. One automated judge call is made for each selected
answer. These are answer counts, not counts of independent rubric decisions.
Several criteria within an answer, related questions and repeated scenario material
remain dependent. This is a limited preliminary check, not a formal power calculation
or a guarantee that every rare error category is sufficiently represented.

1. **Generate real pilot answers.** Include the relevant target models, ABF stages,
   scenarios and E1/E2/E3 conditions where applicable. A tasks are generated once
   under E1, so this is not a fully crossed stage-by-condition design. Mock answers,
   transport failures and reused aliases do not enter the annotation sample. Retain
   their separate coverage and transport records.
2. **Select the panel before judging its correctness.** The seeded sampler attempts
   marginal balance across scenario, stage, target model and condition and favours
   distinct questions. It assigns whole connected components of the accepted chain
   graph to one partition. Shared questions and connected chain members cannot cross
   the development/held-out split. Disconnected components can still share source
   passages. Whole-scenario grouping is available for transfer to unseen packets.
3. **Record blinded human labels.** The expert uses the blinded answer packet and
   fills `reviewer_1.csv` without model/condition metadata or automated verdicts.
   Enter a verdict for every selected criterion and error check. Preserve the initial
   labels. This file is the single-expert reference, without a second reviewer or
   an adjudicated-consensus claim. The private assignment key contains model,
   condition, repetition and partition metadata and is not an annotation aid.
4. **Develop using the 10 development answers.** Use development-only agreement to
   identify prompt problems. Keep held-out labels and comparisons out of prompt
   selection and revision. Freeze the chosen judge model, inference settings,
   instructions, source bundle, schema and corpus before examining held-out results.
5. **Assess the 50 held-out answers.** Compare the frozen judge with the recorded
   expert labels. One call per answer supplies no estimate of repeated-judge
   stability. If the held-out comparison drives another revision, a fresh held-out
   panel is needed for the revised scoring instrument.
6. **Record the decision and limitations.** Review agreement, coverage, abstention,
   failure detection and concrete disagreements. Record the actual acceptance
   decision and its scope. The code imposes no universal kappa, agreement or F1
   threshold. A favourable number alone does not establish adequacy for every claim.

The expert also participated in corpus construction. That role must be disclosed.
Blinding the target-model identity and automated verdicts reduces some sources of
bias but does not make the human reference independent of the benchmark's design.
There is no inter-human agreement estimate, expert consensus or external certification
in this mode. Report the results as agreement with the single military expert's labels.

The sampler rejects an infeasible requested panel rather than silently reducing it.
If rare detrimental assertions are absent or too sparse, their detection cannot be
estimated reliably. Any deliberately selected challenge set should be reported
separately because its prevalence differs from the sampled answer population.

The generated UTF-8 CSV worksheet can be opened in Excel and is blank by design.
Existing annotation files are not overwritten. Reference defects require a recorded,
versioned corpus correction, not selective changes made to favour one model.

### Optional two-expert mode

`calibration-prepare --reviewers 2` creates `reviewer_1.csv`, `reviewer_2.csv` and
`adjudicated.csv`. The experts annotate independently before discussing disagreements.
Keep the original labels and put final decisions in the separate adjudicated file.
Two reviewers do not create an automatic majority. Unresolved reference problems
remain unscorable. The two initial files permit inter-human agreement to be reported;
judge agreement is then measured against the adjudicated reference.

To reproduce the earlier larger design, explicitly request `--n 200
--development-n 50` and `judge-prepare --repetitions 3`. Sample size, reviewer count
and judge repetitions are separate choices. Repeated scoring calls do not create
additional independent validation answers.

## Agreement and stability reports

`compare_label_records` reports positive criteria, local errors and inherited errors
separately. For positive criteria, the event of interest is criterion failure. For
error checks, it is error presence. Results include confusion counts, agreement rate,
Cohen's kappa, failure precision, recall/sensitivity, failure F1, specificity and
coverage. Undefined metrics remain null. In particular, absent event classes and
degenerate marginals must not be reported as perfect detection or agreement.

Agreement conditions on both labels being scorable. Reports also retain human and
judge abstentions, missing verdicts and failure detection when abstentions are treated
as missed known failures. High conditional agreement with low coverage is not a claim
of uniformly reliable scoring. Single-expert mode compares against `reviewer_1.csv`;
it does not calculate inter-human agreement by comparing that file with a copy.

The default one-call procedure reports repeatability as unavailable. When at least
two independent judge calls are requested, `repeatability_report` describes their
agreement on the same answers, separating missing responses from semantic abstentions.
The number of unique answers is retained. The included agreement metrics are point
estimates with explicit denominators, not confidence intervals based on an invented
independence of every criterion decision.

Use `calibration-report --split development` during prompt development. This reports
the development comparison without exposing held-out agreement. After the judge is
frozen, use `--split heldout`. `--split all` produces both comparisons and should only
be used after that freeze. Record the actual sample composition and which classes
have adequate labelled support before interpreting the metrics.

## Main experiment and secondary judge

After validation, use one primary judgement per physical generated answer. Score a
seeded, marginally balanced **20% subset within each target-model/global-repetition
group** with a second judge. Finish the group's generations before freezing its
selection. Its requested size is `ceil(0.2 * eligible_answers_in_group)`. This
grouped sampling scheme preserves earlier selections when models or repetitions
are added later. Select the subset before inspecting primary scores. Keep `judge_role=primary` and
`judge_role=secondary` explicit. The secondary judgement measures sensitivity to judge
choice and is not pooled into a majority vote or treated as an additional model answer.
If cost requires a different fraction, freeze it before scoring and report the achieved
coverage. The secondary sample excludes aliases so shared A answers are not repeatedly
charged or counted.

Blinding does not eliminate shared model-family preferences. Report the target and judge
model families, inspect disagreements by family and include results from the independent
judge on its actual sample. A secondary judge is not an independent military expert.

Use the same judge instructions, source bundle policy, criteria and output budget rule
for all target models and conditions. Save the full request, response, prompt and corpus
hashes, provider model identifiers, attempts, tokens and role. Cached results are reusable
only when the exact request and judge configuration match. Validation repetitions must
make fresh judge calls rather than reuse a cached judgement as a new observation.

### Output budget

Every explicit verdict consumes output tokens. The accepted v0.5 corpus requires
4–16 verdicts per answer, with mean 6.90. A 1,000-token output cap is not a safe universal
assumption for the full schema with quotations and justifications. A conservative initial
rule used by the runner is `max(4096, 256 * number_of_verdicts + 512)`, subject to the provider's supported
limit. Measure actual output tokens and truncation in the pilot and freeze the chosen
rule before the main study. Treat truncation as a judge failure. Do not drop error checks
to make a response fit. Cost estimates must include primary judging, secondary judging,
validation repetitions and transport retries separately.

## Command-line workflow

The study interface supports separately collected model segments. The default pilot
uses Qwen3.5-9B and Gemma 4 31B. Their inclusion does not establish judge validity for
every optional model family. Add relevant families before selecting the validation
panel if the intended validation scope requires them. Supply local API keys before
the credential check.

```bash
heerbench run --study pilot --pilot --group pilot --repetitions 1 --plan-only
heerbench preflight --study pilot --credentials
heerbench run --study pilot --pilot --group pilot --repetitions 1
heerbench status --study pilot
```

The pilot uses synchronous requests. Reissue the same run command to resume it.
Completed responses are retained. Finish the selected source generations before
preparing the panel. The calibration bundle copies existing target answers and is
excluded from model-report consolidation, so these copies are not extra trials.

```bash
heerbench calibration-prepare --study pilot --out reviews/judge_v1 \
  --reviewers 1 --n 60 --development-n 10 --group-by component
```

All four flags on the second line are defaults and can be omitted. The expert completes
`reviews/judge_v1/reviewer_1.csv` against `answers_blinded.md` or the equivalent JSONL
packet before viewing automated verdicts. No `reviewer_2.csv` or `adjudicated.csv` is
required in single-expert mode. The coordinator keeps assignment metadata private.

```bash
heerbench judge-prepare reviews/judge_v1/pooled_run \
  --selection reviews/judge_v1/selection.private.json --repetitions 1
heerbench judge reviews/judge_v1/pooled_run
heerbench calibration-report reviews/judge_v1/pooled_run \
  --panel reviews/judge_v1 --split development
```

One judge repetition is the default. It is independent of the number of target-model
repetitions. Prepared calls are idempotent on resume. Judge calls incur their own API
charges; panel construction and reporting are offline. A pending or failed judge call
reduces coverage rather than becoming a favourable verdict.

### Judge development without regenerating target answers

If development shows that revised instructions are needed, fork the fixed target
answers into a new judge run.

```bash
heerbench fork-judge reviews/judge_v1/pooled_run \
  --prompt prompts/judge_revised.txt --out reviews/judge_v2_run
heerbench judge-prepare reviews/judge_v2_run \
  --selection reviews/judge_v1/selection.private.json --repetitions 1
heerbench judge reviews/judge_v2_run
heerbench calibration-report reviews/judge_v2_run \
  --panel reviews/judge_v1 --split development
```

The fork retains target response IDs and exact candidate answers, and copies no old
judge scores or approval. It prohibits new target generation. Keep the same original
expert labels and grouping. Preserve each version's reports separately when comparing
development candidates. Do not compare prompt versions on held-out agreement and then
call the selected version's results held out. Such a selection needs a fresh held-out
panel. The expert's awareness of the scenario material also remains a limitation.

### Held-out assessment and production scoring

After choosing and freezing the judge, evaluate the held-out comparison for that exact
run. The example below uses the original judge; substitute the chosen fork path if it
was revised.

```bash
heerbench calibration-report reviews/judge_v1/pooled_run \
  --panel reviews/judge_v1 --split heldout
```

Review the actual result and limitations. Complete the matching generated
`judge_approval.template.json` into an acceptance record, preserving the original
pending template if a separate output name is used. Record the real reviewer, date,
scope, rationale and report hash from `judge_approval.hashes.json`. A revised-prompt
fork has its own matching template. Do not reuse an approval for different judge
instructions or imply that a generated template is an expert decision.

The main study must use the same accepted corpus, judge deployment and instructions.
After main target generation finishes, scoring remains attached to its original
segments.

```bash
heerbench judge --study conference \
  --approval reviews/judge_v1/judge_approval.json \
  --heldout-report reviews/judge_v1/judge_heldout_report.json
heerbench report --study conference --out reports/conference
```

Use the actual acceptance/report paths from the chosen validation run. `--models ID`
can restrict judging to selected completed models. Full scoring requires a matching
recorded acceptance. Provisional pilot scoring remains available without an acceptance
claim, and resulting reports remain explicitly unvalidated.

### Direct segment and two-expert alternatives

The same calibration commands accept a frozen run directory instead of `--study`.
For example, `calibration-prepare runs/pilot --out reviews/panel` exports a panel from
one completed run. Its judge calls then use `runs/pilot`, while the private selection
and expert worksheet remain under `reviews/panel`.

The earlier two-expert design can be selected explicitly in a new output directory.

```bash
heerbench calibration-prepare --study pilot --out reviews/two_experts \
  --reviewers 2 --n 200 --development-n 50
heerbench judge-prepare reviews/two_experts/pooled_run \
  --selection reviews/two_experts/selection.private.json --repetitions 3
```

It requires independent `reviewer_1.csv` and `reviewer_2.csv` labels and final labels
in `adjudicated.csv`. It does not substitute copied labels when another expert is
unavailable. The default one-expert workflow does not require these extra files.

## Python interfaces

The CLI wraps these functions, which can also be called without provider access.

```python
from heerbench.judge import build_judge_request, validate_judgement, score_judgement

request = build_judge_request(question, candidate_text, question_sources,
                              inherited_error_checks)
# Send request.system and request.user_text to the configured judge.
validated = validate_judgement(raw_judge_response, request)
score = score_judgement(validated, request)
```

```python
from heerbench.judge_validation import (
    select_validation_sample, export_validation_package, load_label_csv,
    compare_label_records,
)

selection = select_validation_sample(
    generation_records, all_questions_by_id, full_accepted_chains,
    n=60, development_n=10, seed=20261127, group_by="component",
)
export_validation_package(selection, all_questions_by_id, all_sources,
                          all_inherited_error_checks, "reviews/panel", reviewers=1)

# Use only the development split during judge-prompt development.
reference = load_label_csv("reviews/panel/reviewer_1.csv")
split_by_answer = {r["blind_answer_id"]: r["split"] for r in selection["assignment"]}
report = compare_label_records(reference, automated_label_rows,
                               split_by_answer=split_by_answer, split="development")
```

The CLI additionally freezes human-review mode and the review-reference filename.
Input generation records need `task_id`, `question_id`, `model_id`, `condition`,
`repetition`, `generation_status` and answer text. The sampler never uses scores.
`judgement_label_rows` converts a validated judgement into the comparison schema.
`select_secondary_sample` returns actual records for secondary-judge scoring.

`configs/judge_protocol.yaml` documents the reference protocol. Runtime judge profiles
are selected in the experiment configuration. The acceptance template is explicitly
unapproved. `judge_configuration_digest` binds the judge prompt, deployment/inference
settings, corpus and protocol version while excluding credential paths and prices.
`validate_judge_approval` checks matching report provenance and real decision fields.
It does not establish scientific validity independently of the recorded assessment.

## Methodological grounding

Rubric-based automated evaluation and empirical validation against humans are established
approaches. This implementation adapts them to the accepted HEERBench criteria. It does
not claim that a general-purpose judge paper validates the present military tasks.

- Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*, NeurIPS 2023.
  [Official proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/file/91f18a1287b398d378ef22505bf41832-Paper-Datasets_and_Benchmarks.pdf).
  Motivates explicit human comparison and attention to systematic judge preferences.
- Liu et al., *G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment*, EMNLP 2023.
  [ACL Anthology](https://aclanthology.org/2023.emnlp-main.153/).
  Provides a precedent for criterion-guided LLM evaluation. Its particular scoring
  procedure and reasoning prompt are not reproduced here.
- The supplied MA-Bench data and HEERBench adaptation preserve original item lineage,
  required determinations and applicable detrimental assertions. The present protocol
  scores bounded ABF answers, supplies source passages to the judge and must be validated
  for this different measurement target. It is not a claim of exact replication of the
  original MA-Bench automated judge.
