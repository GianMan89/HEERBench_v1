# Incremental experiments and combined reports

Models and repetitions can be collected in separate immutable run segments. A report combines compatible observations without rewriting any source run, issuing a model call or renumbering repetitions.

## Collect observations under one study

Use the same study identifier when adding models or repetitions to one experiment. The study scheduler creates a new segment for the missing model/repetition pairs. Each segment retains its own frozen configuration, request plan, journal and software fingerprint. An existing segment is resumed using its original plan.

Repetitions are study-wide identifiers for each model. For example, adding repetition 3 after repetitions 1 and 2 produces observations labelled 3. It does not create another repetition 1 and rename it during analysis. A model may be added after the other models have finished without changing their observations.

The study is an organisational unit. Scientific compatibility is checked separately. A common study name does not permit different prompts, task subsets or judge configurations to be combined.

## Produce the combined report

From the repository root, run

```bash
python -m heerbench report --all-runs --runs-root runs --out reports --study paper
```

Omit `--study paper` to discover all eligible runs beneath `runs`. The scanner also recognises nested study segment directories. Use `--bootstrap 10000` to specify the number of paired bootstrap resamples, and add `--plots` if the plotting dependencies are installed.

The command produces `reports/index.md` and `reports/index.json`. Each compatible protocol has a separate `protocol_<digest>/` directory containing

- `report.md` and `analysis.json`, with coverage, task scores, matched condition comparisons, chain outcomes and dependency transitions.
- CSV tables written by the existing analysis/reporting modules.
- `compatibility.json`, the exact scientific protocol used to form that group.
- `model_configurations.json`, deployment and generation settings for the compared models, excluding credentials and descriptive catalogue metadata.
- `source_run_ledger.json`, including collection times, source directories, study identifiers, configuration/plan/software hashes, observed model identities, explicit repetition identifiers and validation status.
- `merged_plan.jsonl` and `merged_primary_results.jsonl`, the actual analysis inputs.
- `duplicate_snapshots.json`, an explicit record of identical recorded scientific snapshots excluded from double counting.

The original source run retains the full prompts, raw responses, reference contracts, judge outputs, token usage, costs and execution history. Combined reports refer back to those records instead of creating additional model responses. Reports can be regenerated without loading API keys.

## What may be combined

| Property | Pooling rule |
|---|---|
| Dataset | The same corpus fingerprint, with local dataset integrity verified. |
| Task subset | Identical selected chains and their complete question union. |
| Pilot/full scope | Separate protocol groups, even if an explicitly selected subset happens to coincide. |
| Conditions | Identical condition set and protocol version. |
| Prompts | Identical frozen generator and judge system prompts. |
| Global generation settings | Identical settings, including output budgets and sampling configuration. |
| Judges | Identical primary/secondary deployment settings, judge prompt/schema versions, scoring version and secondary sampling policy. |
| Target models | Different models may be added. The same model label must retain the same deployment and inference settings. |
| Repetitions | Different explicit repetition identifiers may be added. Overlapping logical observations are rejected. |
| Prices and credentials | Price changes, API key locations and display metadata do not define scientific compatibility. Actual inference parameters and pinned backend settings do. |
| Software version | Different source hashes are recorded and warned about. The historical execution code is not run during reporting. Review code differences before publication. |

Models are not silently replaced by another endpoint. Changed API model IDs, pinned backends, precision, checkpoint revision or inference parameters under one model label cause consolidation to fail. Use distinct model labels and document the comparison. If returned model version identifiers change within a proposed pool, consolidation also fails. This conservative check can require separate reporting even when a provider claims two names are aliases.

Resolved judge endpoints are included in grouping. Observed judge version drift is rejected rather than allowing a change of scoring instrument to masquerade as a model performance difference. Stable reported IDs do not prove that a provider-managed model stayed unchanged internally. Collection dates and provider revision policies remain part of the publication record.

The index identifies which protocol fields differ between groups. It does not average scores across these groups.

## Missing observations and shared questions

Every frozen planned request remains in the analysis. Missing journal rows are pending requests, and a run that has not started is still represented in coverage. Transport failures and unscorable judgements do not become content errors. Reports show observed scores, scoring coverage and the existing conservative scores that assign zero to unscored planned requests.

An A question is generated once per model/repetition under E1 and reused in the E2/E3 chain views. Shared questions belonging to several chains do not become additional generations or additional independent observations. Sequential dependencies must point to the correct question, model, repetition and upstream condition.

The consolidation step verifies that each source plan contains its declared models, repetitions, conditions and complete task subset. A missing planned question cannot disappear from the denominator merely because an entire row was omitted.

## Duplicates, pilot development and judge approval

An exact copied run is counted once only when its frozen configuration, plan and journal observations are identical. The duplicate is recorded explicitly. Conflicting copies, independent runs with the same model/condition/repetition/question key, and two different observations sharing a request ID are rejected. The consolidator never selects the higher score, the latest answer or the most complete run automatically. Resolve genuine collection mistakes using documented provenance, or select the intended study explicitly.

Mock/demo runs and journal entries marked as mock are excluded. Judge-development and calibration bundles marked `reused_generation_only` are excluded because their generation records were already collected elsewhere. Such bundles support judge development and validation, not extra repetitions. Production scoring should remain associated with the original generation segment.

Each included run must have matching accepted held-out judge validation for the combined report to record completed expert acceptance. This records the stated review decision and does not certify the scientific adequacy of that validation. If any source lacks approval, the combined report is marked provisional. A held-out metrics file without human acceptance is still provisional. A single-expert acceptance remains a single-expert assessment and must not be described as independent expert consensus. A present approval with missing or mismatched supporting material is an error.

## Scientific interpretation

Adding models does not turn this fixed benchmark into a random sample of military staff work. Several chains may share questions, and repetitions do not add scenarios. The existing analysis therefore retains scenario-level aggregation and matched comparisons between E1, E2 and E3.

The oracle-minus-sequential contrast measures replacement of the preceding answer with an expert reference, including differences in correctness, completeness, wording and length. Incorrect-to-incorrect transitions remain associations until content-linked analysis establishes propagation of a particular error. Collecting runs incrementally does not change those interpretation limits.

Unequal numbers of repetitions are retained and reported for each model. They are not hidden through duplication or downsampling. For a final comparison, complete the intended replication schedule or explain the imbalance, missingness and corresponding uncertainty.

The v0.3 default validation scope is 50 held-out answers assessed by one military
expert after development on 10 separate answers. The report and paper must retain
that scope, the expert's role in corpus construction and the absence of inter-human
agreement. A larger combined target-model report does not enlarge the human
validation sample or create evidence of transfer to unrepresented model families.
