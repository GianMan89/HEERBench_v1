"""Graph-aware, paired analysis of a frozen HEERBench run plan.

The plan, not the set of successfully returned answers, defines denominators.
This module does not call a model and does not infer causal error propagation
from coincident errors. All functions accept JSON-compatible records.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math
import random
from statistics import mean, stdev
from typing import Any, Iterable, Mapping, Sequence


class AnalysisError(ValueError):
    """An analysis input violates the frozen-plan contract."""


CONDITIONS = ("E1", "E2", "E3")
GENERATION_STATUSES = {"succeeded", "failed", "blocked", "pending"}
JUDGE_STATUSES = {"succeeded", "failed", "unscorable", "pending"}


def _mean(values: Iterable[float | None]) -> float | None:
    valid = [float(v) for v in values if v is not None]
    return mean(valid) if valid else None


def _sd(values: Iterable[float | None]) -> float | None:
    valid = [float(v) for v in values if v is not None]
    return stdev(valid) if len(valid) >= 2 else None


def _quantile(values: Sequence[float], p: float) -> float:
    if not values:
        raise AnalysisError("A quantile requires observations.")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def _score_value(value: Any, key: str, request_id: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnalysisError(f"{request_id}: {key} must be numeric or null.")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise AnalysisError(f"{request_id}: {key} must be finite and in [0, 1].")
    return float(value)


def join_records(
    plan_records: Sequence[Mapping[str, Any]],
    result_records: Sequence[Mapping[str, Any]],
    questions: Sequence[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Join primary-judge records to every planned request, retaining failures.

    A result may contain a nested ``score`` dictionary or its score fields at
    top level. Results from explicitly non-primary judges are not pooled.
    Duplicate planned keys or primary result IDs fail rather than silently win.
    """
    if isinstance(questions, Mapping):
        qmap = dict(questions)
    else:
        qmap = {q["question_id"]: q for q in questions}
        if len(qmap) != len(questions):
            raise AnalysisError("Duplicate question IDs.")
    plans: dict[str, Mapping[str, Any]] = {}
    keys: set[tuple[Any, ...]] = set()
    for row in plan_records:
        required = ("request_id", "model_id", "condition", "repetition", "question_id")
        if any(k not in row for k in required):
            raise AnalysisError(f"Plan record lacks one of {required}.")
        rid = row["request_id"]
        if not isinstance(rid, str) or not rid or rid in plans:
            raise AnalysisError(f"Missing/duplicate request ID {rid!r}.")
        key = tuple(row[k] for k in required[1:])
        if key in keys:
            raise AnalysisError(f"Duplicate model/condition/repetition/question key {key}.")
        keys.add(key)
        if row["condition"] not in CONDITIONS:
            raise AnalysisError(f"Unknown condition {row['condition']}.")
        rep = row["repetition"]
        if isinstance(rep, bool) or not isinstance(rep, int) or rep < 0:
            raise AnalysisError("Repetition must be a nonnegative integer.")
        if row["question_id"] not in qmap:
            raise AnalysisError(f"Unknown planned question {row['question_id']}.")
        q = qmap[row["question_id"]]
        stage = q.get("abf_stage", q.get("stage"))
        if stage not in ("A", "B", "F"):
            raise AnalysisError(f"Invalid stage for {row['question_id']}.")
        if row.get("case_id", q["case_id"]) != q["case_id"]:
            raise AnalysisError(f"Incorrect scenario for {row['question_id']}.")
        if row.get("abf_stage", stage) != stage:
            raise AnalysisError(f"Incorrect stage for {row['question_id']}.")
        if stage == "A" and row["condition"] != "E1":
            raise AnalysisError("A is generated once in E1 and reused, not separately billed in E2/E3.")
        plans[rid] = row

    results: dict[str, Mapping[str, Any]] = {}
    ignored = 0
    for row in result_records:
        if row.get("judge_role", "primary") != "primary":
            ignored += 1
            continue
        rid = row.get("request_id", row.get("task_id"))
        if rid not in plans:
            raise AnalysisError(f"Result does not belong to this plan: {rid!r}.")
        if rid in results:
            raise AnalysisError(f"Duplicate primary result for {rid}.")
        p = plans[rid]
        for name in ("model_id", "condition", "repetition", "question_id"):
            if name in row and row[name] != p[name]:
                raise AnalysisError(f"Result/plan mismatch in {name} for {rid}.")
        results[rid] = row

    joined = []
    for rid, p in plans.items():
        result = dict(results.get(rid, {}))
        score = result.get("score", {})
        if score is None:
            score = {}
        if not isinstance(score, Mapping):
            raise AnalysisError(f"Score must be an object for {rid}.")
        score = {**result, **score}
        gen = result.get("generation_status", "pending")
        judge = result.get("judge_status", "pending")
        if gen not in GENERATION_STATUSES or judge not in JUDGE_STATUSES:
            raise AnalysisError(f"Unknown execution status for {rid}.")
        if gen != "succeeded" and judge == "succeeded":
            raise AnalysisError(f"Successful judging without a generated answer for {rid}.")
        q = qmap[p["question_id"]]
        raw_score = _score_value(score.get("criterion_score"), "criterion_score", rid)
        raw_error = _score_value(score.get("negative_error_rate"), "negative_error_rate", rid)
        full = score.get("fully_correct")
        has_error = score.get("has_error")
        if any(v is not None and not isinstance(v, bool) for v in (full, has_error)):
            raise AnalysisError(f"Correctness/error flags must be boolean or null for {rid}.")
        scoreable = (
            gen == "succeeded" and judge == "succeeded"
            and score.get("scoreable", True) is True
            and raw_score is not None and full is not None and has_error is not None
        )
        if scoreable and full and (raw_score != 1.0 or has_error):
            raise AnalysisError(f"Contradictory fully_correct flag for {rid}.")
        if scoreable and full is False and raw_score == 1.0 and has_error is False:
            raise AnalysisError(f"All criteria/no errors contradict fully_correct=False for {rid}.")
        joined.append({
            "request_id": rid,
            "model_id": p["model_id"],
            "condition": p["condition"],
            "repetition": p["repetition"],
            "question_id": p["question_id"],
            "case_id": q["case_id"],
            "abf_stage": q.get("abf_stage", q.get("stage")),
            "generation_status": gen,
            "judge_status": judge,
            "scoreable": scoreable,
            "criterion_score": raw_score if scoreable else None,
            "fully_correct": bool(full) if scoreable else None,
            "has_error": bool(has_error) if scoreable else None,
            "negative_error_rate": raw_error if scoreable else None,
            "criterion_score_lower_bound": raw_score if scoreable else 0.0,
            "fully_correct_lower_bound": float(bool(full)) if scoreable else 0.0,
            "n_required": score.get("n_required"),
            "n_required_satisfied": score.get("n_required_satisfied"),
            "n_inherited_errors_present": score.get("n_inherited_errors_present"),
            "n_local_errors_present": score.get("n_local_errors_present"),
        })
    return joined, ignored


def _group_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    complete = [x for x in rows if x["scoreable"]]
    return {
        "planned": total,
        "generated": sum(x["generation_status"] == "succeeded" for x in rows),
        "generation_failed": sum(x["generation_status"] == "failed" for x in rows),
        "generation_blocked": sum(x["generation_status"] == "blocked" for x in rows),
        "generation_pending": sum(x["generation_status"] == "pending" for x in rows),
        "judge_failed": sum(x["judge_status"] == "failed" for x in rows),
        "judge_unscorable": sum(x["judge_status"] == "unscorable" for x in rows),
        "scored": len(complete),
        "scoring_coverage": len(complete) / total if total else None,
        "criterion_score_observed": _mean(x["criterion_score"] for x in complete),
        "criterion_score_lower_bound": _mean(x["criterion_score_lower_bound"] for x in rows),
        "fully_correct_observed": _mean(float(x["fully_correct"]) for x in complete),
        "fully_correct_lower_bound": _mean(x["fully_correct_lower_bound"] for x in rows),
        "any_error_observed": _mean(float(x["has_error"]) for x in complete),
        "inherited_error_rate_observed": _mean(x["negative_error_rate"] for x in complete),
        "questions_with_inherited_error_rate": sum(x["negative_error_rate"] is not None for x in complete),
    }


def _scenario_run_scores(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        groups[(row["model_id"], row["condition"], row["case_id"], row["repetition"], row["abf_stage"])].append(row)
    out = []
    for (model, cond, case, rep, stage), rows in sorted(groups.items()):
        out.append({"model_id": model, "condition": cond, "case_id": case,
                    "repetition": rep, "stage": stage, **_group_summary(rows)})
    # B/F is an equal-stage average, not a criterion pool or a pooled A/B/F score.
    by_case: dict[tuple[Any, ...], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in out:
        if row["stage"] in ("B", "F"):
            by_case[(row["model_id"], row["condition"], row["case_id"], row["repetition"])][row["stage"]] = row
    metric_keys = ("scoring_coverage", "criterion_score_observed", "criterion_score_lower_bound",
                   "fully_correct_observed", "fully_correct_lower_bound", "any_error_observed",
                   "inherited_error_rate_observed")
    count_keys = ("planned", "generated", "generation_failed", "generation_blocked", "generation_pending",
                  "judge_failed", "judge_unscorable", "scored", "questions_with_inherited_error_rate")
    for (model, cond, case, rep), stages in sorted(by_case.items()):
        if set(stages) != {"B", "F"}:
            continue
        r = {"model_id": model, "condition": cond, "case_id": case, "repetition": rep, "stage": "BF"}
        for key in count_keys:
            r[key] = sum(x[key] for x in stages.values())
        for key in metric_keys:
            vv = [x[key] for x in stages.values()]
            # An unavailable stage cannot silently disappear from a BF mean.
            r[key] = mean(vv) if all(v is not None for v in vv) else None
        out.append(r)
    return out


def _macro_scores(scenario_rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in scenario_rows:
        groups[(row["model_id"], row["condition"], row["stage"], row["repetition"])].append(row)
    metrics = ("criterion_score_observed", "criterion_score_lower_bound", "fully_correct_observed",
               "fully_correct_lower_bound", "any_error_observed", "inherited_error_rate_observed", "scoring_coverage")
    runs = []
    for (model, condition, stage, rep), rows in sorted(groups.items()):
        r = {"model_id": model, "condition": condition, "stage": stage, "repetition": rep,
             "n_scenarios": len(rows), "planned": sum(x["planned"] for x in rows), "scored": sum(x["scored"] for x in rows)}
        for metric in metrics:
            r[metric] = _mean(x[metric] for x in rows)
            r[metric + "_n_scenarios"] = sum(x[metric] is not None for x in rows)
        runs.append(r)
    grouped_runs: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for r in runs:
        grouped_runs[(r["model_id"], r["condition"], r["stage"])].append(r)
    summary = []
    for (model, condition, stage), rr in sorted(grouped_runs.items()):
        row = {"model_id": model, "condition": condition, "stage": stage,
               "repetitions": len(rr), "planned": sum(x["planned"] for x in rr), "scored": sum(x["scored"] for x in rr)}
        for metric in metrics:
            row[metric] = _mean(x[metric] for x in rr)
            row[metric + "_run_sd"] = _sd(x[metric] for x in rr)
        summary.append(row)
    return runs, summary


def paired_cluster_bootstrap(
    scenario_repetition_values: Mapping[str, Mapping[int, float]],
    *, n_resamples: int = 10_000, seed: int = 20261127,
) -> dict[str, Any]:
    """Scenario-cluster bootstrap of already paired differences.

    Each resampled scenario brings its whole question graph. Its repetition
    blocks are sampled jointly for both conditions; nodes are never resampled
    as independent observations. The output is a descriptive benchmark
    interval, not a claim of representative military-scenario sampling.
    """
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int) or n_resamples < 0:
        raise AnalysisError("n_resamples must be a nonnegative integer.")
    values = [list(v.values()) for _, v in sorted(scenario_repetition_values.items()) if v]
    estimate = _mean(mean(v) for v in values)
    result = {"estimate": estimate, "ci_low": None, "ci_high": None,
              "n_scenarios": len(values), "n_scenario_repetitions": sum(map(len, values)),
              "bootstrap_resamples": n_resamples, "bootstrap_seed": seed,
              "interval_interpretation": "descriptive paired scenario/repetition cluster bootstrap"}
    if not values or n_resamples == 0 or (len(values) == 1 and len(values[0]) == 1):
        return result
    rng = random.Random(seed)
    ncase = len(values)
    means = []
    for _ in range(n_resamples):
        s = 0.0
        for _ in range(ncase):
            v = values[rng.randrange(ncase)]
            s += sum(v[rng.randrange(len(v))] for _ in range(len(v))) / len(v)
        means.append(s / ncase)
    result["ci_low"] = _quantile(means, 0.025)
    result["ci_high"] = _quantile(means, 0.975)
    return result


def _paired_effects(records: Sequence[Mapping[str, Any]], bootstrap: int, seed: int) -> list[dict[str, Any]]:
    ix = {(r["model_id"], r["condition"], r["repetition"], r["question_id"]): r for r in records}
    models = sorted({r["model_id"] for r in records})
    output = []
    for model in models:
        for left, right, label in (("E2", "E3", "oracle_minus_sequential"), ("E3", "E1", "sequential_minus_independent"), ("E2", "E1", "oracle_minus_independent")):
            for stage in ("B", "F", "BF"):
                target = [r for r in records if r["model_id"] == model and r["condition"] == left and r["abf_stage"] in (("B", "F") if stage == "BF" else (stage,))]
                matches = [(r, ix.get((model, right, r["repetition"], r["question_id"]))) for r in target]
                pairs = [(a, b) for a, b in matches if b is not None]
                if not pairs:
                    continue
                for metric, complete_only in (("criterion_score_lower_bound", False), ("criterion_score", True), ("fully_correct_lower_bound", False)):
                    blocks: dict[tuple[Any, ...], list[float]] = defaultdict(list)
                    for a, b in pairs:
                        if complete_only and not (a["scoreable"] and b["scoreable"]):
                            continue
                        blocks[(a["case_id"], a["repetition"], a["abf_stage"])].append(float(a[metric]) - float(b[metric]))
                    by_case_rep: dict[tuple[Any, ...], dict[str, float]] = defaultdict(dict)
                    for (case, rep, st), vv in blocks.items():
                        by_case_rep[(case, rep)][st] = mean(vv)
                    clustered: dict[str, dict[int, float]] = defaultdict(dict)
                    used = 0
                    for (case, rep), st in by_case_rep.items():
                        if stage == "BF" and set(st) != {"B", "F"}:
                            continue
                        clustered[case][rep] = mean(st.values())
                        used += sum(len(blocks[(case, rep, s)]) for s in st)
                    ci = paired_cluster_bootstrap(clustered, n_resamples=bootstrap, seed=seed)
                    output.append({"model_id": model, "contrast": label, "left_condition": left,
                                   "right_condition": right, "stage": stage, "metric": metric,
                                   "planned_pairs": len(pairs), "unmatched_left_requests": len(matches) - len(pairs),
                                   "used_pairs": used, "complete_case_only": complete_only, **ci})
    return output


def _chain_and_transition_scores(
    records: Sequence[Mapping[str, Any]], chains: Sequence[Mapping[str, Any]],
    dependencies: Sequence[Mapping[str, Any]], selected_chain_ids: Sequence[str] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    ix = {(r["model_id"], r["condition"], r["repetition"], r["question_id"]): r for r in records}
    stage = {r["question_id"]: r["abf_stage"] for r in records}
    models_reps = sorted({(r["model_id"], r["repetition"]) for r in records})
    cids = {c["chain_id"] for c in chains}
    if len(cids) != len(chains):
        raise AnalysisError("Duplicate chain IDs.")
    if len({d["dependency_id"] for d in dependencies}) != len(dependencies):
        raise AnalysisError("Duplicate dependency IDs.")
    if any(d.get("logic", "ALL") != "ALL" for d in dependencies):
        raise AnalysisError("Analysis requires conjunctive ALL parent sets.")
    if selected_chain_ids is not None and not set(selected_chain_ids) <= cids:
        raise AnalysisError("Unknown selected chain ID.")
    selected = [c for c in chains if selected_chain_ids is None or c["chain_id"] in selected_chain_ids]
    output = []
    def view(model: str, cond: str, rep: int, qi: str):
        return ix.get((model, "E1" if stage.get(qi) == "A" else cond, rep, qi))
    for model, rep in models_reps:
        for ch in selected:
            ids = list(dict.fromkeys(ch["a_ids"] + ch["b_ids"] + ch["f_ids"]))
            for condition in CONDITIONS:
                rr = [view(model, condition, rep, qi) for qi in ids]
                # Chains not selected for a model's plan are not failed trials.
                if any(r is None for r in rr):
                    if selected_chain_ids is not None:
                        raise AnalysisError(f"Selected chain {ch['chain_id']} lacks planned nodes for {model}/{condition}/{rep}.")
                    continue
                scored = all(r["scoreable"] for r in rr)
                label = {"E1": "independent_joint_task_success", "E2": "oracle_assisted_joint_task_success", "E3": "sequential_chain_success"}[condition]
                output.append({"model_id": model, "condition": condition, "repetition": rep,
                               "chain_id": ch["chain_id"], "case_id": ch["case_id"],
                               "metric_label": label, "unique_nodes": len(ids),
                               "scoreable": scored,
                               "success_observed": all(r["fully_correct"] for r in rr) if scored else None,
                               "success_lower_bound": int(scored and all(r["fully_correct"] for r in rr)),
                               "generation_incomplete_nodes": sum(r["generation_status"] != "succeeded" for r in rr),
                               "unscorable_nodes": sum(not r["scoreable"] for r in rr),
                               "a_response_source": "same-model same-repetition E1 response",
                               "oracle_caveat": "E2 downstream inputs are expert references, so this joint score is not an end-to-end generated chain" if condition == "E2" else ""})
    transitions = []
    for model, rep in models_reps:
        for edge in dependencies:
            child = view(model, "E3", rep, edge["child_id"])
            if child is None:
                continue
            parents = [view(model, "E3", rep, qi) for qi in edge["parent_ids"]]
            if any(p is None for p in parents):
                raise AnalysisError(f"E3 child {edge['child_id']} lacks a planned parent.")
            scoreable = child["scoreable"] and all(p["scoreable"] for p in parents)
            parent_correct = all(p["fully_correct"] for p in parents) if scoreable else None
            child_correct = child["fully_correct"] if scoreable else None
            if not scoreable:
                label = "unscorable"
            elif parent_correct and child_correct:
                label = "correct_to_correct"
            elif parent_correct:
                label = "correct_to_incorrect"
            elif child_correct:
                label = "incorrect_to_correct"
            else:
                label = "incorrect_to_incorrect_association"
            transitions.append({"model_id": model, "repetition": rep, "dependency_id": edge["dependency_id"],
                                "case_id": edge["case_id"], "child_id": edge["child_id"],
                                "child_stage": child["abf_stage"], "parent_count": len(parents),
                                "parent_set_logic": "ALL", "scoreable": scoreable,
                                "parent_set_correct": parent_correct, "child_correct": child_correct,
                                "transition": label, "interpretation": "association; no semantic propagation attribution inferred"})
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in output:
        groups[(row["model_id"], row["condition"], row["case_id"], row["repetition"], row["metric_label"])].append(row)
    summaries = []
    for (model, cond, case, rep, label), rr in sorted(groups.items()):
        summaries.append({"model_id": model, "condition": cond, "case_id": case, "repetition": rep,
                          "metric_label": label, "planned_chains": len(rr),
                          "scored_chains": sum(r["scoreable"] for r in rr),
                          "success_observed": _mean(float(r["success_observed"]) if r["success_observed"] is not None else None for r in rr),
                          "success_lower_bound": _mean(r["success_lower_bound"] for r in rr)})
    return output, transitions, summaries


def analyse(
    plan_records: Sequence[Mapping[str, Any]], result_records: Sequence[Mapping[str, Any]],
    questions: Sequence[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
    chains: Sequence[Mapping[str, Any]], dependencies: Sequence[Mapping[str, Any]],
    *, bootstrap: int = 10_000, seed: int = 20261127, is_mock: bool = False,
    selected_chain_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Analyse unique questions and matched complete chain views.

    ``result_records`` are primary-judge joined results, not raw completions or
    individual judge calls. A missing planned result remains pending.
    """
    records, ignored = join_records(plan_records, result_records, questions)
    scenario_rows = _scenario_run_scores(records)
    runs, summary = _macro_scores(scenario_rows)
    paired = _paired_effects(records, bootstrap, seed)
    chain_rows, transitions, chain_summary = _chain_and_transition_scores(records, chains, dependencies, selected_chain_ids)
    coverage_groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        coverage_groups[(row["model_id"], row["condition"])].append(row)
    coverage = [{"model_id": m, "condition": c, **_group_summary(rr)} for (m, c), rr in sorted(coverage_groups.items())]
    repetition_counts = {m: len({r["repetition"] for r in records if r["model_id"] == m}) for m in sorted({r["model_id"] for r in records})}
    warnings = [
        "Confidence intervals describe resampling of this fixed benchmark's scenarios and repetition blocks; the ten inherited scenarios are not a random sample of military staff work.",
        "Shared nodes and chains do not create independent observations. Node-level scores count each planned question once.",
        "Oracle-minus-sequential is the effect of replacing the entire preceding answer with an expert reference, including correctness, completeness, phrasing and length.",
        "Incorrect-to-incorrect transitions are associations. Attribution of error propagation requires a separate content-linked adjudication.",
        "Lower-bound scores assign zero to unscored planned requests and therefore mix availability with content success. Interpret them together with observed scores and coverage.",
        "A difference of two lower-bound scores is an availability-adjusted contrast, not a mathematical lower bound on the unknown complete-data contrast.",
    ]
    if is_mock:
        warnings.insert(0, "MOCK / PIPELINE TEST ONLY — these records are synthetic and are not empirical model performance.")
    if any(n < 2 for n in repetition_counts.values()):
        warnings.append("At least one model has only one repetition; run-to-run variability cannot be estimated for that model.")
    if any(not r["scoreable"] for r in records):
        warnings.append("Some planned outputs are unavailable or unscorable. Complete-case contrasts use a selected subset and must be read with lower-bound contrasts and coverage.")
    return {
        "metadata": {
            "analysis_version": "1.0", "is_mock": bool(is_mock),
            "label": "MOCK_PIPELINE_TEST_NOT_MODEL_RESULTS" if is_mock else "EMPIRICAL_RUN_ANALYSIS",
            "planned_generation_requests": len(plan_records),
            "observed_result_records": len(result_records) - ignored,
            "ignored_nonprimary_judge_records": ignored,
            "scenarios": sorted({r["case_id"] for r in records}),
            "models": sorted(repetition_counts), "repetitions_by_model": repetition_counts,
            "bootstrap_resamples": bootstrap, "bootstrap_seed": seed,
            "selected_chain_ids": list(selected_chain_ids) if selected_chain_ids is not None else None,
            "chain_selection_inferred": selected_chain_ids is None,
            "aggregation": "mean questions within stage/scenario/repetition, equal B/F stage mean, equal scenario mean, equal repetition mean",
            "corpus_reuse": "A is generated only in E1 and reused in chain views; reused A rows do not add generation or judge calls",
            "warnings": warnings,
        },
        "coverage": coverage,
        "question_scores": records,
        "scenario_run_scores": scenario_rows,
        "run_scores": runs,
        "stage_scores": summary,
        "paired_effects": paired,
        "chain_scores": chain_rows,
        "chain_scenario_run_scores": chain_summary,
        "transitions": transitions,
    }
