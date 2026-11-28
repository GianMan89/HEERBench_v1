# Data dictionary

## Canonical and review formats

`reference/*.jsonl` is the canonical constructed dataset. Each line is a JSON record. `data/*.csv` provides equivalent review/export views in UTF-8 with a byte-order mark for Excel. Arrays and objects are JSON-encoded in normalized CSV fields. The question CSV uses readable line-separated lists. The workbook is an editable review view with explicit acceptance and comment columns. Reviewing the workbook does not automatically modify the canonical JSONL or freeze a new release.

`provenance/construction_records/C*.json` preserves the case construction and disposition records before export normalization. `provenance/rubric_items.jsonl` and `provenance/documents.jsonl` preserve the original supplied data. `provenance/accepted_v04_items.json` includes the accepted screening columns and earlier audit history. The earlier audit flags remain historical metadata and do not reverse the user's acceptance.

## Questions

- `question_id` is a stable identifier within this version. It contains scenario and ABF stage.
- `case_id` identifies one of the ten source scenarios.
- `abf_stage` is A, B or F. It is independent of the original MA-Bench reasoning tier.
- `question` is the neutral task prompt.
- `expected_answer` is an exemplar, not an exact-match string.
- `criteria` contains separately identified content requirements. `required` determines mandatory content.
- `acceptable_alternatives` describes permitted formulations or solutions consistent with the assessed construct.
- `error_conditions` records local material errors introduced during construction.
- `source_refs` identifies the unchanged released source text and explains what it contributes.
- `source_item_uids` records directly associated original rubric items. Association does not imply that a companion or derived answer already existed in that rubric.
- `origin` distinguishes `adapted`, `decomposed` and `constructed_companion`.
- `source_answer_status` is `explicit` for retrieval and `derived` for an additional judgement or consequence.
- `derivation_note` explains the construct, inference or construction-stage correction.
- `chain_ids` lists complete-chain memberships.
- `lineage_item_uids` adds ancestral or chain-anchor provenance when a constructed companion has no direct original item. It is not a second count of retained items or a claim of direct conversion.
- `original_tiers_by_item` and `original_adaptation_cases_by_item` retain item-level metadata without imposing a single tier on a multi-item question.
- `new_reference_status` remains `constructed_pending_expert_validation`.

The model-facing export includes only `question_id`, `case_id` and `question`. The prompt builder does not show task IDs in the generated prompt. It does not expose reference answers or criteria in the independent condition.

## Sources

A source reference uses `doc_order` to construct IDs such as `C10:D09`. This can differ from the document's own printed number. `doc_number`, title and original filename remain present for disambiguation. Line ranges are one-based and inclusive in `documents.jsonl` after `splitlines()`. The source excerpt is copied exactly from that range. Its document SHA-256 hashes the original UTF-8 text, not the wrapper added to the model packet.

A source link supports a question-level reference. Its `support` description identifies the premise supplied. It is not a claim that every sentence in an exemplar appears verbatim in the excerpt. B/F derivations are documented separately. Long source spans may be displayed in consecutive workbook rows for readability; canonical JSONL retains the full span.

## Dependencies and chains

`dependencies.jsonl` stores unique parent-set/child relationships. `parent_ids` is an indivisible set for an `ALL` edge. `child_id` is the task being supported. `chain_ids` lists the complete chains that use the relationship. `rationales` and `counterfactual_tests` retain construction explanations when a shared edge appears in more than one chain.

`chains.jsonl` retains the full scenario-local chain, including A/B/F membership and the original edge records. `chain_memberships.jsonl` is a normalized one-row-per-membership view. A shared question is counted once as a question and several times as a membership. A direct A prerequisite at F remains explicit alongside B when the consequence needs additional source context.

The counterfactual field is an expert-review prompt. No altered-source or altered-answer experiment was performed. A reference-supported run must choose a specific dependency set rather than silently take the union of every predecessor from all chains.

## Item ledger

Every original MA-Bench item has exactly one ledger row with its original wording, conditions, tier, polarity, step, severity, weight, fact and source fields.

`accepted_v04_decision`, `accepted_v04_role` and `accepted_v04_case` preserve the agreed baseline. Construction status is one of the following.

| Value | Meaning |
|---|---|
| `BASELINE_EXCLUDED` | Excluded in the accepted v0.4 corpus, before question construction. |
| `RETAINED_IN_CHAIN` | At least a stated facet supports constructed questions in a complete chain. |
| `EXCLUDED_NO_FULL_CHAIN` | Accepted positive item not retained after applying the complete-chain and construct rules. |
| `ERROR_CHECK_LINKED` | Accepted negative check attached to at least one applicable question. |
| `ERROR_CHECK_NOT_APPLICABLE` | Accepted negative check preserved but no applicable retained question was found. |

`coverage` is `full`, `partial` or `none`. Partial means the explicitly identified facets are retained while `uncovered_content` states what is not assessed. It is not a numerical coverage estimate. `construction_rationale` explains the decision. Original items, constructed questions and criteria have different denominators.

## Negative checks

`error_item_uid` points to an original negative item. `trigger` describes when the error is present in the particular task. `applicability` states why the check is attached. The original question and original error-presence condition are preserved.

An original negative pass condition is not the desired answer. No credit is earned just because an error is absent. A generic hallucination check is scoped to the relevant quantitative or authority claims, not applied as an undifferentiated copied sentence across every task.

## Proposed scoring records

Future response scoring should store response/model/run/condition IDs, the selected dependency set, criterion decisions with supporting answer spans, error-presence decisions, adjudication status and evaluator identity/version. Required criteria are binary. A question-completeness score is distinct from its mean criterion score. Chain completeness is distinct from either, and must not treat shared nodes as independent responses.

The inherited MA-Bench weights remain metadata. Reusing their weighted full-brief aggregation after splitting, merging and adding questions would change the weighting implicitly and is not implemented.
