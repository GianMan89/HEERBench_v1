# Reproduction and release record

## What is frozen

`plan` freezes the selected corpus fingerprint, generator and judge prompts,
selected full chains, unique question set, all direct parents, model catalog
entries, price snapshot, explicit repetition identifiers, output caps, sampling options,
selection seed and request order. Its manifest records the Python/platform and
source-code fingerprint. Changed code, plans, configuration or data are rejected
on resume. Dataset manifests are verified before planning and execution. API key
values are never frozen. File locations are recorded as configuration, and key
contents are read only for a requested check or provider operation.

The incremental study interface creates immutable segments for missing
model/repetition pairs. `run --study conference --models ID --repetitions N --plan-only`
prepares the selected model up to a total of N repetitions without API calls.
Repeating the command preserves existing segments. Removing `--plan-only` executes
or resumes the selected pending work. A later model or repetition is added in a new
segment, with its global repetition identifiers unchanged. Explicit configuration
changes are rejected within that study rather than applied to earlier work.

There is one physical generation for each question/condition/model/repetition
combination, with A represented under E1 only. Generation IDs retain the complete
logical identity. The original packet remains in every request. Source passages,
references, rubrics and negative checks are supplied to the judge, not inserted
into target-model prompts. Only explicitly allowed E2 parent references enter a
target-model prompt.

The source dataset, construction audit and acceptance overlay have different
provenance roles. Do not overwrite historical construction statuses to manufacture
an earlier validation record. Any corpus correction changes its version and hash.

## What remains stochastic or provider controlled

The selection seed controls deterministic selection and request ordering. It is
not a claim that provider inference is deterministic. Temperature/seed parameters
are omitted when their support is not established. A model-specific inference
setting remains fixed across conditions and repetitions. Its default reasoning
behaviour can differ from another model's behaviour. Output tokens can include
provider-billed internal reasoning. Record the chosen settings and actual usage.

An immutable public checkpoint revision is not available for every hosted model.
The catalog records unknown or provider-managed revisions explicitly. The returned
model identity, backend, precision where known, run timestamps and raw response
are retained. An unexpected identity is a recorded failure and is not automatically
whitelisted. Resolve aliases during the pilot and create a new, fixed main plan.

The local context check is conservative but is not a vendor tokenizer by default.
It estimates from UTF-8 length and includes a framing margin. Where a frozen
`tokenizer_file` and `tokenizer_sha256` are provided, the local tokenizer replaces
that estimate and retains a framing allowance. API rejection of an oversized full
prompt stays visible. The harness never silently truncates documents or parents.

## Accounting and failures

Token costs are dated rate-card estimates, not invoices. Standard and batch prices,
cache reads, cache writes and long-context tiers are applied separately. A planned
cache hit is never treated as observed. Native batching and prefix caching do not
reuse outputs between repetitions. Gemini explicit cache storage is not enabled
by the default configuration.

`max_cost_usd` is an optional conservative cap on cumulative token reservations.
Reservations use full byte bounds, output caps and the highest configured input
or cache-write rate. They are intentionally higher than expected invoices and
are not released after cheaper responses. They cover only operations recorded by
this run and the rate-card assumptions, not unrelated account usage, GPU rental,
taxes or unobserved provider charges. An incomplete rate card cannot support a
hard cap.

A confirmed rate-limit rejection can retry with bounded backoff and an attempt
record. Transport uncertainty after submission, an interrupted running request or
a server response that may hide successful completion is not silently retried.
Inspect the provider account and either adopt the existing native batch or record
an explicit recovery. Every failed attempt remains in the event history. Missing
cost information is reported, not treated as zero.

Generation success means that a valid provider response was obtained. Empty,
refused, truncated and substantively wrong answers are retained with flags and
evaluated once. A missing upstream API response blocks its E3 descendants. It is
not inserted as an analytical premise. Scoring failures and semantic unscorability
are distinct from failed content criteria.

## Suggested publication archive

Archive the exact code release, dependency constraints, data manifest, configurations,
prompts and final frozen run manifests. Include the complete planned denominator,
generation/usage ledger, primary and secondary judge outputs, calibration sample
design, single-expert labels where consent permits (or separate initial labels and
adjudication in the two-expert mode), judge acceptance record and derived tables. Preserve raw outputs needed to reproduce
scores and explain omissions. Record all deviations from the initial plan.

Review publication artifacts for credentials or restricted project material before
publishing them. The repository itself contains only the supplied releasable MA-Bench
material and local research artifacts. No public upload is performed by the CLI.

The local `secrets/*.key` files are ignored by Git. A public archive must omit real
key contents even when it includes blank key-file placeholders. Authentication
never falls back to an environment variable. Provider proxy configuration is
separate from API credential handling.

The paper should report selected models and serving configurations, counts by
scenario/stage, repetitions, effective conditions, criteria, failure treatment,
judge-validation results, observed cost/usage, and uncertainty intervals. Explain
that there are ten scenario clusters, not thousands of independent scenarios.
Publish provisional pilot results separately from the final evaluation.

## Consolidation of incremental runs

`report --all-runs --runs-root runs --out reports` discovers source segments
without opening their journals through the execution runner. It verifies frozen
configuration, plan and corpus integrity, then creates separate reports for each
compatible protocol. Add `--study conference` to restrict discovery to one study.
Different selected model sets and repetition counts can contribute to the same
comparison. Changed corpus/task subsets, prompts, judge settings and pilot/full
scope cannot. Deployment changes under the same target-model label and overlapping
logical observations are errors. Repetitions are never renumbered during reporting.

The source-run ledger preserves code hashes, collection times, frozen plans,
observed model versions, validation status and exclusions. Historical code is not
executed to read its records. Different source software hashes require review even
when their recorded protocol fields match. Exact copied snapshots can be counted
once, but conflicting copies are not resolved by selecting better scores. Missing
planned answers remain in coverage and conservative denominators. Mock runs and
judge-only calibration bundles do not supply additional empirical observations.

All contributing source runs must have matching held-out judge approval for the
combined report to record completed expert acceptance. This label records the
author's acceptance decision, not an independent certification of scientific validity. A mixed approved/unapproved pool is
marked provisional. Scientific identity excludes API credential locations and
price metadata, so relocating a key or refreshing prices does not change the
accepted judge's measurement procedure. See [incremental reporting](incremental_reporting.md)
for the complete compatibility and duplicate-handling rules.

## Environment and offline checks

The release was exercised with Python 3.12 using the supplied pinned direct
dependencies. CI also defines Python 3.11 and 3.13 jobs. Only actually executed
local test results are claimed in `VALIDATION.md`. CI configuration is not proof
that a remote CI service has run.

The software uses POSIX file locking. Run on Linux/macOS or WSL2. Installed package
code is small, while corpus and configuration files live in the source repository.
Keep the source tree available for the CLI or configure explicit data paths.

For an archival public release, add the actual Git commit and repository URL to
the paper and assign an archival DOI if desired. No DOI, conference acceptance or
public repository location has been invented in this package.

## Human-reference scope in v0.3

The default validation uses one military expert, 10 development answers, 50 held-out
answers and one judge call per answer. Archive the review mode, reference filename,
actual sample composition and original labels. Do not create a second worksheet by
copying the same labels or describe single-expert labels as adjudicated consensus.
The expert helped construct the corpus, so the paper must disclose that connection.
Inter-human agreement and judge repeatability are unavailable under these defaults.
Use development-only reporting before freezing the judge and held-out reporting
only after that freeze. Preserve any prompt revisions and their development rationale.
