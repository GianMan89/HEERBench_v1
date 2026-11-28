# Changes in v0.3.0

## Single-expert judge validation

- Changed the default preparation panel to 60 actual generated answers, with 10 development and 50 held out, and one automated judge call per selected answer.
- Added an explicit one-expert mode. It uses `reviewer_1.csv` as the human reference and does not require or manufacture a second review or adjudicated labels.
- Retained the optional two-expert workflow, independently configurable sample sizes and judge repetitions.
- Added development-only and held-out-only reporting so prompt development need not reveal held-out agreement.
- Reports and documentation distinguish agreement with a single expert from inter-human agreement or expert consensus. One judge call does not supply repeatability results.
- Documented the limited validation sample, dependent criterion decisions and the expert's participation in corpus construction. No universal acceptance threshold or guarantee of adequacy is asserted.
- Kept the accepted corpus, ABF graph, E1/E2/E3 conditions, target-model catalogue, primary/secondary judges and 20% secondary sampling policy unchanged.

## Migration

Keep the original software revision for executing an already-frozen v0.2 run.
Do not edit frozen configuration or fabricate extra expert labels to satisfy the
older workflow. The new calibration mode is explicit in the exported panel and
assessment record. Existing two-expert panels retain their original reference
semantics. Historic target generations may be reused through the documented
calibration-bundle workflow without becoming new model repetitions.

No real human labels, held-out results or expert approval are included in the
release. The code supports the agreed procedure; it does not perform that review.

# Changes in v0.2.0

## Model and judge configuration

- Replaced the previous eight-model default with eleven lower-cost target configurations.
- Added all sixteen requested optional frontier/MA-Bench selections. Gemma is shared by the two pools, giving twenty-six distinct target models. Grok remains optional only.
- Verified published endpoint IDs and archived public provider metadata. Pinned router backend/precision and disabled fallback where supported.
- Changed the primary judge to GPT-6.1 Terra and the secondary to Claude Sonnet 5.5.
- Corrected backend-specific context and pricing details, including Haiku long-context tiers, Nemotron serving capacity, Gemma precision, Qwen snapshot identity and optional Kimi rates. GPT-5.6 Sol's promotional expiry is explicit.

## Incremental execution

- Added named studies composed of immutable run segments.
- Models can be selected individually or as groups. Desired repetitions are totals per model, and new work includes only missing model/repetition pairs.
- Existing completed answers, prompts, scoring records and global repetition IDs are retained. Subset execution never dispatches pending work for unselected models or repetitions.
- Preserved the direct per-directory CLI workflow.
- Added read-only consolidated reports with explicit compatibility groups, deployment/version checks, source ledgers and duplicate-observation safeguards.
- Added fixed calibration bundles combining completed pilot segments without new target-model calls or copied judge scores.

## Credentials and packaging

- API credentials are read exclusively from local key files. Environment-variable authentication has been removed.
- Added four empty Git-ignored key placeholders, recursive inline-secret rejection, lazy validation and provider-response redaction.
- Added a safe release packager that never reads local key contents and recreates empty key placeholders.

## Scientific continuity

- The accepted v0.5 corpus, questions, references, criteria, graph and MA-Bench provenance are unchanged.
- E1/E2/E3 definitions, all direct-parent dependencies, full scenario context, fixed output budget and RQ1/RQ2 scope are unchanged.
- Secondary judging is sampled at 20% within each model/global repetition, so adding later models does not reselect previous samples.
- Approval digests now use schema version 2 and bind judge deployment, prompt, corpus and held-out report while excluding key paths and price metadata. Old-format approvals cannot silently migrate.
- No paid model calls or new human-validation results were produced for this release.

## Migration from v0.1.0

Keep the original source revision for resuming any already-frozen v0.1 run. Changed software fingerprints are intentionally rejected during execution. Historical runs remain readable for consolidation, but old and new scientific protocols receive separate groups. Gemma's old DeepInfra FP8 deployment and new CoreWeave FP4 deployment must not be treated as one unchanged model configuration. Do not copy credentials into frozen run configurations. New experiments should use the v0.2 key files and study interface.
