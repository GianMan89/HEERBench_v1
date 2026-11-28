# Construction method

## Scope and status

This version constructs an ABF question graph from the user-accepted MA-Bench re-audit v0.4. The original scenario text is unchanged. The accepted inclusion decision, adaptation case, polarity, reasoning tier, severity and original assessment wording remain separately recoverable for every source item. Construction does not convert the original brief-level score into an ABF score.

The accepted screening corpus contains 933 positive task candidates and 123 negative error checks. The other 201 original items remain baseline exclusions. The user reported reviewing the screening decisions with a logistics and military-planning colleague. That acceptance is the starting point of this construction. No independent item-level human annotations or agreement statistics were provided, and none are inferred. The newly written questions, answers, criteria and dependencies have not yet undergone military-expert validation.

## 1. Operational definition

**Ansprechen (A, Identify)** retrieves relevant supplied information while preserving whether it is an observed fact, report, planning assumption, order or existing assessment. An action already specified in an order is an A response when the question merely asks for that action.

**Beurteilen (B, Assess)** interprets the significance of the information for a bounded staff problem. It may establish a capability-demand mismatch, an implication, a trade, a scope limitation or a conditional risk. It must express an additional relationship rather than reproduce an assessment already stated in the packet.

**Folgern (F, Conclude)** derives a concrete consequence. Examples include a support or resource requirement, coordination task, revised schedule, contingent support arrangement, task-specific readiness prerequisite, comparison criterion or handover requirement. Merely restating a risk or announcing uncertainty is insufficient. An information requirement qualifies when it specifies what must be established and the decision or support commitment that depends on it.

This is the agreed operational definition for HEERBench. It is not presented as a verbatim doctrinal quotation or an assertion that an LLM internally reasons in three stages.

## 2. Source and item control

Every original item is retained verbatim in `provenance/rubric_items.jsonl`. The accepted v0.4 decisions are in `provenance/accepted_v04_items.json`. The complete construction ledger joins these records with conversion outcomes without overwriting the originals.

The 112 released source documents are the only factual scenario inputs used for reference construction. A rubric statement or a reference to an unreleased downstream product is not treated as a model-visible source. The evaluator references identify the released document by `case_id` and `doc_order`, a one-based inclusive line span, the supplied proposition and a checksum of the original text. `doc_number` and the original filename remain separate because source document numbers and packet order are not always the same.

No real-world information is silently added to repair a fictional scenario. Unspecified quantities, dates, capacities, authority decisions and technical limits remain unspecified. Ordinary comparisons and arithmetic may be derived explicitly. A new military consequence may be justified by the supplied premises without prescribing a unique unsourced solution.

## 3. Construction from the four accepted cases

| Accepted adaptation case | Construction rule |
|---|---|
| Explicit information | Construct an A question. Retain it only if it contributes to a complete chain. |
| Assessment | Construct a B task where an additional interpretation is required. Add source-supported A companions as needed. |
| Derived requirement | Construct an F task with a concrete consequence. Add A and B companions as needed. |
| Compound analytical requirement | Separate independently assessable functions. Preserve the mapping of each retained facet and identify omitted facets. |

An item is not mechanically expanded into three questions. Several original items can support one consolidated question, and a compound original can support several questions. Newly constructed companion questions are labelled explicitly. Some companions have no direct original rubric item because their role is to make a necessary premise independently observable. Their chain and ancestor lineage is recorded separately from direct item associations.

Where direct source inspection shows that an accepted purported assessment or requirement is already stated, the accepted case remains unchanged as provenance. The constructed question can instead be A, with a recorded construction-stage correction and a distinct B or F consequence if justified. This avoids making the assigned screening category override the actual measurement construct.

## 4. Neutral question formulation

Questions request an analytical product rather than ask whether a brief satisfies a rubric. They do not expose the original item number, pass condition or expected answer. Each is interpretable with the original packet and does not depend linguistically on an earlier response. Names and task boundaries may identify the problem without disclosing its resolution.

The same question wording is intended for independent and predecessor-supported conditions. The primary packet remains complete. Narrow explicit elicitation changes the measurement from producing a whole mission-analysis brief to satisfying a stated analytical task. Consequently, scores are complementary to original MA-Bench scores and are not directly interchangeable.

## 5. Reference and scoring construction

Each question has an exemplar answer, independently assessable content criteria, acceptable alternatives, local error conditions and source references. The exemplar is not an exact-match target. A response can satisfy the criteria with a different defensible formulation. Required qualifiers such as confidence, conditionality, scope and approval status are part of the assessed content.

The distinction between a relational criterion and a compound checklist matters. A comparison of three apportioned companies with two allocated companies is one relational proposition. Identifying the responsible command and reproducing a deadline are separately assessable factual propositions. Criteria should be split when one can be true while another is false. The review workbook provides criterion-level acceptance and revision fields for this reason.

A mandatory criterion is scored present or absent after accounting for acceptable alternatives and required qualifications. Local errors and inherited negative checks are recorded separately. No new military-importance weighting is inferred from the original rubric weights. A negative source item is not turned into a desired response. Its original pass condition describes the presence of an error, so the converted record uses an explicit error trigger.

The proposed initial scoring outputs are criterion completion per question, complete question success, and task-specific error incidence. A response containing a materially contradictory claim cannot receive credit for the contradicted criterion simply because it also contains the correct wording. Automated judging requires a separate validated protocol. No judge reliability estimate is claimed here.

## 6. Dependencies and complete chains

The scenario graph records observable information relationships between tasks. Each dependency contains a complete set of jointly required parents, a child, an explanation of their contribution and a qualitative counterfactual check. `ALL` means that the listed inputs jointly support the child judgement. It does not mean that each parent is independently sufficient.

A complete chain includes at least one A, one B and one F. A chain may contain several jointly relevant A inputs, more than one consequence, and additional A context required directly by F alongside B. Questions are reused across chains when the same analytical construct is involved. They are not duplicated merely to create disjoint triplets.

For an edge, the review question is whether changing or removing the specified information would require revising the downstream justification or consequence. This is a construction annotation, not an executed perturbation experiment. A shared topic, adjacent place in an order, or common terminology does not establish a dependency. Background facts without a substantive inferential role are excluded from parent sets.

Full source access means that a model can reconstruct a preceding result instead of adopting a supplied answer. The graph therefore does not establish an unavoidable information bottleneck or reveal internal reasoning. Later interventions can measure sensitivity to supplied preceding outputs, not prove use merely from their availability.

These distinctions are consistent with prior separation of constituent competence and composition performance in Press et al. (2023), and the emphasis on connected rather than merely co-occurring information in MuSiQue (Trivedi et al., 2022). Those papers do not validate the present military task definitions.

## 7. Consolidation and negative checks

Every accepted positive item receives a construction disposition. A retained item may be fully or partially reused. Partial coverage names the original content not carried into the questions. Items that cannot contribute to a defensible complete chain receive `EXCLUDED_NO_FULL_CHAIN` with a specific reason. This is separate from the 201 `BASELINE_EXCLUDED` records. No excluded source item is silently deleted.

Negative checks are attached only to questions where the corresponding mistake is substantively applicable. The link records an observable trigger and the reason for application. A check without an applicable question remains `ERROR_CHECK_NOT_APPLICABLE`. Its absence from a scoring opportunity does not imply that the underlying error is acceptable.

All retained questions have complete-chain membership. Every chain has valid sources, required stages and defined dependencies. Duplicate IDs, broken references, reversed-stage edges, unmapped positive provenance and incomplete disposition coverage are rejected by the structural validator. The full item ledger remains 1,257 rows.

## 8. Validation and limits

This is an AI-assisted candidate construction. Drafting, internal cross-checking of selected constructs and deterministic validation were performed in this workflow. Internal cross-checks are not independent military annotations. These checks establish consistency and traceability, not military or psychometric validity. The release status remains `constructed_pending_expert_validation` for all new questions and chains. Authored mappings are supplied as versioned annotations. Export and validation scripts reproduce their structure and presentation, not the semantic drafting process.

Expert review should assess the question, source passages, exemplar, criteria and dependencies together. A question may be sound individually while its chain is not. Conversely, a valid consequence can admit several military solutions, making an over-prescriptive reference unsuitable. Revisions that change a node require rechecking every chain in which it appears.

There is no model pilot, inter-rater reliability estimate, empirical perturbation result, difficulty calibration or representative-sampling claim in this version. The scenarios are a fixed inherited sample. Exclusion for chain suitability and addition of source-supported companions can alter the balance of task types. The method should be described as a constructed diagnostic corpus, not a random sample of all staff reasoning.

Question and chain counts are different analysis units. Shared nodes and common scenario documents create dependence. Later experiments should report unique-question performance, distinguish chain completeness from node scores and account for clustering by scenario and shared question structure. A simple binomial interval treating every chain membership as independent is inappropriate.

Documentation of composition, decisions, provenance and limits follows the motivation of Datasheets for Datasets (Gebru et al., 2021). This documentation is not a substitute for the planned military review.

## Sources

- MA-Bench supplied release v1.0.0, `provenance/MA_Bench_README.md` and the original JSONL records. Lafuente-Mercado et al. (2026), https://arxiv.org/abs/2610.19138.
- Press et al. (2023), *Measuring and Narrowing the Compositionality Gap in Language Models*, https://aclanthology.org/2023.findings-emnlp.378/.
- Trivedi et al. (2022), *MuSiQue: Multihop Questions via Single-hop Question Composition*, https://aclanthology.org/2022.tacl-1.31/.
- Gebru et al. (2021), *Datasheets for Datasets*, https://arxiv.org/abs/1803.09010.
