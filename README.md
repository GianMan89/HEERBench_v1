# HEERBench experiment repository v0.3.0

**Scenario-based evaluation of large language models for military command and staff decision support.**

This repository contains the accepted HEERBench adaptation of MA-Bench and the software to run independent, reference-assisted and sequential ABF experiments. It includes a small paid-pilot configuration and a separate **zero-cost, offline software demonstration**. No model results or judge-validation outcomes are claimed by this release.

The scientific protocol is in [docs/protocol.md](docs/protocol.md). Provider-specific limitations are in [docs/providers.md](docs/providers.md), the scoring procedure in [docs/judging.md](docs/judging.md), and the analysis definitions in [docs/metrics.md](docs/metrics.md).

## Included corpus

| Component | Count |
|---|---:|
| Original scenarios | 10 |
| Unchanged source documents | 112 |
| Original rubric items retained in the provenance ledger | 1,257 |
| Constructed questions | 488 |
| A — Ansprechen / Identify | 175 |
| B — Beurteilen / Assess | 147 |
| F — Folgern / Conclude | 166 |
| Complete ABF chains | 150 |
| Distinct parent sets / B and F children | 313 |
| Atomic positive criteria | 2,235 |
| Question-specific links to original negative checks | 492 |

Questions, expected answers, alternatives, local errors, source passages, original MA-Bench tiers, adaptation cases, inclusion/exclusion decisions, chain memberships and dependencies are included in `data/heerbench_v05/`. Your subsequent joint expert acceptance is recorded in `acceptance.json`. Historical construction files remain unchanged and retain their original pending-review fields. The acceptance overlay does not imply independently collected annotation statistics.

This release contains the MA-Bench adaptation. A separate German Army scenario has not been invented or added. The data schema supports adding it as a new, explicitly versioned release once it is available and approved.

## Three experimental conditions

| Condition | Input to the current question |
|---|---|
| E1 — Independent | Complete original scenario packet and current question |
| E2 — Reference-assisted | The same packet and question, plus expert expected answers to all specified direct parents |
| E3 — Sequential | The same packet and question, plus the evaluated model's generated answers to all specified direct parents in the same repetition |

An A answer is generated once per model and repetition under E1 and reused as the input to dependent E3 tasks. A shared question is not regenerated merely because it belongs to several chains. B and F are generated separately in all three conditions. E2 uses reference parents throughout, including when a preceding E2 answer was wrong. No reference criteria or expected current answer are exposed to the evaluated model. The supplied-parent labels are identical in E2 and E3.

The full corpus requires **1,114 generation requests per model per repetition**. The default pool of 11 models and three repetitions requires **36,762 generation requests**, before judging, calibration and retries. Each repetition creates fresh responses. This protocol covers RQ1 and RQ2. Source-degradation experiments for RQ3 and reasoning-mode ablations remain outside this release.

## Install and inspect without API calls

Use Python 3.11 or later on Linux/macOS. On Windows, use WSL2 or a Linux container because experiment locking uses POSIX locks.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -c constraints.txt -e '.[test,plots]'
heerbench validate
python -m pytest -q
heerbench models --group default
heerbench models --group extended
heerbench demo --out runs/offline_demo
```

`python -m heerbench` and `heerbench` are equivalent. The offline demo makes no network requests and produces explicitly unscorable mock judgements. Its output is software verification, not a model result. Keep `data/`, `configs/` and `prompts/` alongside the source package. Optional `tokenizers` and `proxy` extras support a frozen local tokenizer and SOCKS proxy environments.

## Local key files

Put the raw API keys into the supplied empty files, one key per file, without quotes or an assignment.

| File | Account |
|---|---|
| `secrets/openai.key` | OpenAI models and the primary judge |
| `secrets/anthropic.key` | Claude models and the secondary judge |
| `secrets/google.key` | Gemini Developer API |
| `secrets/openrouter.key` | Pinned hosted open-weight and other router models |

Keys are read exclusively from these files when executing or explicitly checking credentials. There is no environment-variable fallback. Only the keys needed for the selected model or judge are required. Contents are never frozen in configurations or written to logs. The paths resolve against the repository, independently of the working directory. `.gitignore` excludes the credential files. See [docs/credentials.md](docs/credentials.md).

## Choose a model pool or individual models

The catalogue records the exact API ID, provider/backend, serving precision where disclosed, context and output limits, fixed inference settings, rate card and source URLs. Public availability was checked against provider documentation and endpoint metadata. This does not certify live account permissions or quota.

| Pool | Models |
|---|---|
| `default` | GPT-6 Luna, Claude Haiku 5.5, Gemini 3.5 Flash-Lite, Qwen3.5-9B, Qwen3.6-27B, Qwen3.6-35B-A3B, Gemma 4 31B, DeepSeek V4 Flash, Mistral Small 4, Llama 4 Maverick, Nemotron 3.5 Lightning |
| `pilot` | Qwen3.5-9B and Gemma 4 31B |
| `extended` / `frontier` | Muse Spark 1.3, Grok 4.6, Claude Opus 5, GLM-5.3, GPT-6 Astra, Kimi K3, GPT-5.6 Sol, GPT-5.6 Luna, Nemotron 3 Ultra, MiniMax M3, Claude Fable 5.1, Qwen3.8-Max, DeepSeek V4 Pro, Gemini 3.1 Pro, Mistral Medium 3.5, Gemma 4 31B |

There are **26 distinct targets** across the default and extended pools. Grok is optional and is never selected by the default or pilot configuration. Legacy selectors remain available separately to make historical configurations interpretable. Selecting a published model name never silently substitutes a successor. Friendly names can be quoted, such as `--models "GPT-6 Astra"`.

The primary judge is **GPT-6.1 Terra**. **Claude Sonnet 5.5** judges a predetermined 20% secondary sample, independently selected within each target model and repetition. Judge validity still requires expert comparison. The full catalogue and documented pricing corrections are in [docs/model_catalog.md](docs/model_catalog.md).

## Start with a small pilot

The pilot selects two complete chains in each of the ten scenarios before observing model performance. Their dependency-closed union contains **84 unique questions**, producing **168 generation requests per model and repetition**. Two pilot models therefore generate 336 answers in total. The pilot is for engineering checks and judge preparation, not an estimate of the full corpus with guaranteed statistical precision.

Plan, check capacity and estimate costs first. These commands make no model requests.

```bash
heerbench run --pilot --study pilot --repetitions 1 --plan-only
heerbench preflight --study pilot --credentials
heerbench estimate --study pilot
```

Then execute one or both models whenever required.

```bash
heerbench run --pilot --study pilot --models qwen35_9b --repetitions 1
heerbench run --pilot --study pilot --models gemma4_31b --repetitions 1
heerbench status --study pilot
```

An optional `--max-requests 5` limits a real endpoint smoke test to at most five new dispatches. Resume with the same model, study and repetition total afterwards. The output budget stays fixed at 4,096 tokens. It includes billed reasoning tokens where the provider counts them within that budget. Truncated, incorrect or empty answers are retained. Responses are never retried because their quality is poor.

For provisional pilot scoring, run `heerbench judge --study pilot`, then `heerbench report --study pilot --out reports/pilot`. The report marks judging as unvalidated until a genuine held-out expert acceptance record exists. Secondary scores never replace unfavourable primary scores.

## Add models or repetitions over time

A study is a collection of immutable run segments. Choose a study name and request the **total** repetitions desired for the selected models. The software creates only missing model/repetition pairs and resumes unfinished requests from their original segments.

```bash
# Prepare one model and one full-corpus repetition, without calling an API.
heerbench run --study conference --models qwen35_9b --repetitions 1 --plan-only
heerbench estimate --study conference

# Execute or resume it.
heerbench run --study conference --models qwen35_9b --repetitions 1

# Add another model later, leaving previous work intact.
heerbench run --study conference --models gpt6_astra --repetitions 1 --plan-only
heerbench run --study conference --models gpt6_astra --repetitions 1

# Increase only Qwen from one to three total repetitions.
heerbench run --study conference --models qwen35_9b --repetitions 3

# Alternatively prepare the whole affordable pool.
heerbench run --study affordable --group default --repetitions 3 --plan-only
```

Repeated use of `--repetitions 1` does not create another repetition. Changing it to `3` adds repetitions 2 and 3. Selecting one model does not dispatch pending work belonging to another model, even if both were originally planned together. Each sequential parent comes from the same target model and global repetition as its child.

The study rejects changes to the corpus, selected tasks, prompts, scoring policy, judge deployment, or an existing model's serving configuration. Use an explicitly new study or model label for a different scientific protocol or deployment. A model name shared across several runs is insufficient to establish comparability. Pilot and full-corpus records remain separate.

Native batches are nonblocking. Invoke the same command after submitted jobs finish to collect results and submit the next dependency-ready wave. Models without a native batch adapter use standard calls and standard prices. `status` lists provider jobs. No generic batch discount is assumed for router endpoints. A run with a changed software fingerprint cannot execute under the new code, while historical records can still be read for reporting.

The legacy per-directory workflow remains supported with `plan --config ... --out ...`, `run PATH`, `preflight PATH`, `estimate PATH`, `judge-prepare PATH`, `judge PATH` and `analyse PATH`. A frozen directory cannot be edited to append models. Use the study interface for that purpose.

## Judge preparation and expert validation

The judge receives the question, candidate answer, expected answer, acceptable alternatives, atomic criteria, relevant negative checks and cited source passages. Model identity and experimental condition are not supplied. It returns structured criterion and error-check verdicts, with an abstention option. The parser checks the schema and criterion IDs before calculating scores.

The default validation procedure uses **one military expert and 60 real answers**, comprising **10 development answers and 50 held-out answers**. Each selected answer is judged once. Connected question components cannot occur in both partitions. This is a constrained preliminary validation design, not a guarantee of adequate precision or coverage of every error type.

After completing the pilot, prepare a fixed calibration snapshot and the blank expert worksheet. Pooling separately run pilot models reuses existing answers without generating new target responses or copying previous judge labels.

```bash
heerbench calibration-prepare --study pilot --out reviews/judge_v1
```

The defaults are equivalent to `--reviewers 1 --n 60 --development-n 10`. Complete **only `reviewer_1.csv`**, using the blinded answer packet. Enter a label for every criterion and error check before inspecting automated verdicts. No second reviewer file or adjudication file is required or manufactured. Keep model/condition assignment keys private. As the annotator also helped construct the corpus, report that role and the resulting limitation on independent validation.

Prepare and run one judge call for each selected answer, then inspect development agreement only.

```bash
heerbench judge-prepare reviews/judge_v1/pooled_run \
  --selection reviews/judge_v1/selection.private.json --repetitions 1
heerbench judge reviews/judge_v1/pooled_run
heerbench calibration-report reviews/judge_v1/pooled_run \
  --panel reviews/judge_v1 --split development
```

Use only the development comparison to decide whether the judge prompt needs revision. If it does, `fork-judge` preserves the same candidate answers and partition while keeping the new judge outputs separate. Freeze the chosen judge before inspecting held-out agreement.

```bash
heerbench calibration-report reviews/judge_v1/pooled_run \
  --panel reviews/judge_v1 --split heldout
```

The report describes agreement with the **single-expert reference**, including coverage, abstention, agreement rate, Cohen's kappa and failure-detection precision/recall/F1 by label kind. Undefined statistics remain null. It does not claim expert consensus, inter-human agreement or judge repeatability from one call. Multiple criteria per answer do not increase the number of independent validation answers.

Record the actual acceptance decision, rationale and limitations in the generated `judge_approval.template.json`, using the supplied hashes. No approval is inferred from a metric. The small held-out sample and any unrepresented error categories limit the validation claim. After that review, score completed main-study models.

```bash
heerbench judge --study conference --models qwen35_9b \
  --approval reviews/judge_v1/judge_approval.template.json \
  --heldout-report reviews/judge_v1/judge_heldout_report.json
```

The accepted approval is copied into each scored segment. If the judge prompt changed during development, create the main study with the accepted prompt/configuration. Full scoring requires a recorded acceptance of the actual held-out assessment. That requirement does not prevent target generation or calibration calls.

The two-expert option remains available. Use a new output directory with `calibration-prepare --reviewers 2`; this exports `reviewer_1.csv`, `reviewer_2.csv` and `adjudicated.csv`. Explicit `--n 200 --development-n 50` and `judge-prepare --repetitions 3` reproduce the earlier, larger design. It requires two independently completed expert worksheets and a separate adjudicated reference. See [docs/judging.md](docs/judging.md) for the complete workflow.

## Consolidated reports and scientific comparability

```bash
heerbench report --all-runs --runs-root runs --out reports --plots
# Or restrict discovery to one study.
heerbench report --study conference --out reports/conference --plots
```

Reports preserve model identifiers, repetition counts, planned denominators, failed requests, blocked descendants, unscorable judgements and missing observations. Compatible segments are combined. Different protocols receive separate report groups. Mocks and judge-only reuse bundles are excluded from performance aggregation. Duplicate logical observations or conflicting deployments are rejected instead of choosing an answer. Read-only reporting does not change the original journals. See [docs/incremental_reporting.md](docs/incremental_reporting.md).

Paired condition comparisons use B/F questions and preserve the dependency graph. Scenario-level bootstrap intervals reflect the ten purposively assembled scenarios. They do not establish generalisation to every military task. Architecture, parameter count, serving precision and inference policy are reported as descriptive factors. Comparisons between differently trained and hosted models do not isolate a causal effect of architecture or model size.

## Logs, cost accounting and recovery

Every segment records its frozen configuration and task plan, dataset and software hashes, exact prompt components, supplied parent responses and their IDs, complete returned answers, raw provider responses, token usage, finish flags, judge contracts and verdicts, token-rate estimates, latency, attempts, native batch handles and event history. Repeated packet text is stored once by content hash. Secret contents are redacted, so an exceptional provider response echoing a credential is deliberately not retained verbatim.

`generation.jsonl` and `judgements.jsonl` are readable exports. `journal.sqlite` and `blobs/` retain the authoritative structured records and prompt text. API credentials are not stored there. Actual invoices can differ from rate-card estimates, especially for failed or uncertain attempts without usage metadata.

The offline estimator includes complete scenario packets and supplied parent text, distinct judge prompts, long-context tiers, configured cache writes and actual transport modes. It assumes no cache hits. `estimate --study` reports the entire planned budget, not the amount still to spend. Optional calibration is reported separately and is needed once for a validated judge configuration, not once per model segment. A configured `execution.max_cost_usd` is a conservative **per-segment** admission cap. It is not an account-wide or study-wide spending limit. Expired price cards do not produce a current total and cannot support a hard cap.

Preflight checks complete requests against the selected endpoint's documented limits. A byte-based token estimate is not a tokenizer proof. A frozen local tokenizer can tighten admission; actual sequential parent outputs are checked again before dispatch. The software never truncates scenario information to fit a smaller context. Provider context errors are visible failures.

Successful responses survive interruption. An uncertain submission is not automatically resubmitted. Use `adopt-batch` to attach a confirmed provider job after an interruption. `recover` requires an exact request ID and a reason, preserves the audit trail, and cannot overwrite a returned answer. Do not hand-edit frozen files or journal entries.

## Build a distributable archive

```bash
python scripts/package_release.py \
  --output ../HEERBench_experiments_v0.3.0.zip --prefix heerbench_experiments_v03
```

The packager excludes credentials, local experiment outputs, environments, caches and symlinks. It creates empty key-file placeholders without reading local keys and regenerates the release checksum manifest. Keep local experiments outside distributable archives. An ordinary ZIP of a working directory is not protected by `.gitignore`.

## Repository contents

| Directory | Contents |
|---|---|
| `src/heerbench/` | Planning, study registry, providers, execution, judging and aggregation |
| `configs/` | Default, pilot and demo profiles, catalogue and judge protocol |
| `secrets/` | Gitignored local credential files and setup instructions |
| `prompts/` | Generator and judge instructions frozen at planning |
| `data/heerbench_v05/` | Accepted corpus, references, dependency graph and provenance |
| `docs/` | Scientific protocol, provider sources, metrics and workflows |
| `tests/` | Offline unit, provider-contract and integration tests |
| `scripts/` | Validation and safe release packaging |
| `examples/` | Explicitly labelled planning estimates and mock demonstration reports |

## Citation and licensing

Cite the HEERBench paper when its publication identifier is available, and cite MA-Bench for the reused scenarios and original rubrics. `CITATION.cff` identifies this software release without inventing a DOI. MA-Bench attribution and its BibTeX are preserved in the provenance directory and `docs/references.bib`.

New experiment software is MIT licensed. Reused MA-Bench material is attributed under its supplied CC BY 4.0 declaration. Annotation provenance and data terms are described in `NOTICE.md` and `LICENSES/`. This release has not been pushed to GitHub. No paid model evaluation, human scoring exercise or empirical judge validity is claimed by the software tests.
