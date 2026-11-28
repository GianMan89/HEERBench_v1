"""Grouped judge-calibration sampling, blinded human worksheets and agreement metrics.

No human labels are generated or inferred. Development labels must not be reused as
held-out validation results. The caller retains private assignment metadata separately.
"""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import math
import pathlib
import random
from typing import Any, Iterable, Mapping, Sequence

from .judge import PROMPT_VERSION, build_judge_request

LABELS = {"positive": {"satisfied", "not_satisfied", "unscorable"},
          "local_error": {"error_present", "error_absent", "unscorable"},
          "inherited_error": {"error_present", "error_absent", "unscorable"}}


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def _text(record: Mapping[str, Any]) -> str:
    for key in ("text", "candidate_text", "response_text", "answer"):
        if isinstance(record.get(key), str):
            return record[key]
    response = record.get("response")
    if isinstance(response, Mapping) and isinstance(response.get("text"), str):
        return response["text"]
    raise ValueError("A generation record has no response text")


def eligible_answers(records: Iterable[Mapping[str, Any]], questions: Mapping[str, Mapping]) -> list[dict]:
    """Filter transport failures, mock records and shared aliases, never by correctness."""
    selected = []
    seen = set()
    for r in records:
        if r.get("generation_status", r.get("status")) != "succeeded":
            continue
        if r.get("mock", False) or r.get("is_mock", False) or r.get("provider") == "mock":
            continue
        if r.get("reused_from") or r.get("alias_of"):
            continue
        qid = r["question_id"]
        if qid not in questions:
            raise ValueError(f"Unknown question in generation record {qid}")
        task = r["task_id"]
        if task in seen:
            raise ValueError("Input contains duplicate generation task IDs; resolve attempts first")
        seen.add(task)
        text = _text(r)
        if "model_id" not in r or "condition" not in r:
            raise ValueError("Sampling needs model and condition metadata, hidden from reviewers later")
        q = questions[qid]
        selected.append({**r, "text": text, "case_id": q["case_id"],
                         "abf_stage": q["abf_stage"]})
    return sorted(selected, key=lambda r: r["task_id"])


def question_groups(questions: Mapping[str, Mapping], chains: Sequence[Mapping] = (),
                    group_by: str = "component") -> dict[str, str]:
    """Assign whole connected chain components, questions, or scenarios to one split."""
    if group_by == "question":
        return {q: q for q in questions}
    if group_by == "scenario":
        return {q: x["case_id"] for q, x in questions.items()}
    if group_by != "component":
        raise ValueError("group_by must be component, question or scenario")
    if not chains:
        raise ValueError("Component grouping requires the full accepted chain graph")
    parent = {q: q for q in questions}

    def find(q):
        while parent[q] != q:
            parent[q] = parent[parent[q]]
            q = parent[q]
        return q

    for chain in chains:
        ids = list(chain["a_ids"]) + list(chain["b_ids"]) + list(chain["f_ids"])
        if any(q not in questions for q in ids):
            raise ValueError("Chain grouping refers to an unknown question")
        if len({questions[q]["case_id"] for q in ids}) != 1:
            raise ValueError("A chain connects different scenarios")
        for q in ids[1:]:
            a, b = find(ids[0]), find(q)
            parent[max(a, b)] = min(a, b)
    components = collections.defaultdict(list)
    for q in questions:
        components[find(q)].append(q)
    result = {}
    for ids in components.values():
        label = "component-" + _fingerprint(sorted(ids))[:16]
        for q in ids:
            result[q] = label
    return result


def _balanced_pick(records: Sequence[dict], n: int, seed: int) -> list[dict]:
    """Greedy marginal balancing, with question diversity before repeated answers.

    Marginal balance is attempted for case, ABF stage, model and condition. It does
    not claim an impossible fully crossed design (A is generated once under E1).
    """
    if n < 0 or n > len(records):
        raise ValueError("Requested sample exceeds the eligible population")
    if not n:
        return []
    import numpy as np

    fields = ("case_id", "abf_stage", "model_id", "condition")
    rng = random.Random(seed)
    pool = sorted(records, key=lambda r: r["task_id"])
    rng.shuffle(pool)
    levels = {f: {value: i for i, value in enumerate(sorted({str(r[f]) for r in pool}))}
              for f in fields}
    codes = {f: np.asarray([levels[f][str(r[f])] for r in pool], dtype=np.int32) for f in fields}
    targets = {f: n / len(levels[f]) for f in fields}
    counts = {f: np.zeros(len(levels[f]), dtype=np.int64) for f in fields}
    qmap = {q: i for i, q in enumerate(sorted({r['question_id'] for r in pool}))}
    qcodes = np.asarray([qmap[r['question_id']] for r in pool], dtype=np.int32)
    used_questions = np.zeros(len(qmap), dtype=bool)
    available = np.ones(len(pool), dtype=bool)
    chosen = []
    for _ in range(n):
        # Same greedy objective and first-in-seeded-order tie break as the
        # dictionary implementation, evaluated in vector operations for full runs.
        scores = np.zeros(len(pool), dtype=float)
        compensation = np.zeros(len(pool), dtype=float)
        for f in fields:
            weights = np.maximum(-1.0, (targets[f] - counts[f]) / targets[f])
            values = weights[codes[f]]
            summed = scores + values
            # Compensated summation prevents the order of the four marginal
            # terms from breaking a mathematical tie (Python 3.12+ sum does
            # likewise). This is independent of Python's sum implementation.
            compensation += np.where(np.abs(scores) >= np.abs(values),
                                     (scores-summed)+values, (values-summed)+scores)
            scores = summed
        scores += compensation
        scores += (~used_questions[qcodes]) * 10
        scores[~available] = -np.inf
        index = int(np.argmax(scores))
        row = pool[index]
        chosen.append(row)
        used_questions[qcodes[index]] = True
        available[index] = False
        for f in fields:
            counts[f][codes[f][index]] += 1
    return chosen


def select_validation_sample(
    records: Iterable[Mapping[str, Any]],
    questions: Mapping[str, Mapping],
    chains: Sequence[Mapping] = (),
    *, n: int = 60, development_n: int = 10, seed: int = 20261127,
    group_by: str = "component",
) -> dict[str, Any]:
    """Select a reproducible development/held-out panel with no grouping leakage.

    Insufficient material is an error. For a smaller pilot, explicitly request smaller
    ``n`` and ``development_n``; no silent downsampling or inflated sample-size claim.
    """
    if not 0 < development_n < n:
        raise ValueError("Require 0 < development_n < n")
    eligible = eligible_answers(records, questions)
    if len(eligible) < n:
        raise ValueError(f"Need {n} distinct completed real answers; found {len(eligible)}")
    group_map = question_groups(questions, chains, group_by)
    by_group = collections.defaultdict(list)
    for r in eligible:
        by_group[group_map[r["question_id"]]].append(r)
    if len(by_group) < 2:
        raise ValueError("At least two independent grouping units are required for a held-out split")
    ids = sorted(by_group)
    target_pool = len(eligible) * development_n / n
    rng = random.Random(seed)
    best = None
    # Choose whole groups before selecting answers. Seek diversity and sufficient
    # remaining capacity; do not tune the split using judge or human scores.
    for _ in range(256):
        order = ids.copy()
        rng.shuffle(order)
        picked = []
        cap = 0
        for g in order:
            if len(eligible) - cap - len(by_group[g]) < n - development_n:
                continue
            picked.append(g)
            cap += len(by_group[g])
            if cap >= max(development_n, target_pool):
                break
        if cap < development_n:
            continue
        held = [g for g in ids if g not in picked]
        if sum(len(by_group[g]) for g in held) < n - development_n:
            continue
        dev_cases = {r["case_id"] for g in picked for r in by_group[g]}
        held_cases = {r["case_id"] for g in held for r in by_group[g]}
        score = (len(dev_cases) + len(held_cases), -abs(cap - target_pool), -len(picked))
        if best is None or score > best[0]:
            best = (score, set(picked))
    if best is None:
        raise ValueError("A leakage-free grouped split of the requested size is infeasible")
    dev_groups = best[1]
    dev_pool = [r for g in ids if g in dev_groups for r in by_group[g]]
    held_pool = [r for g in ids if g not in dev_groups for r in by_group[g]]
    panels = {"development": _balanced_pick(dev_pool, development_n, seed + 1),
              "heldout": _balanced_pick(held_pool, n - development_n, seed + 2)}
    assignment = []
    for split, panel in panels.items():
        for r in panel:
            assignment.append({"blind_answer_id": "HBJ-" + _fingerprint([seed, r["task_id"]])[:20],
                               "task_id": r["task_id"], "question_id": r["question_id"], "split": split,
                               "group_id": group_map[r["question_id"]], "case_id": r["case_id"],
                               "abf_stage": r["abf_stage"], "model_id": r["model_id"],
                               "condition": r["condition"], "repetition": r.get("repetition")})
    diagnostic = {}
    for split, panel in panels.items():
        diagnostic[split] = {"answers": len(panel), "empty_answers": sum(not r['text'].strip() for r in panel),
                             "unique_questions": len({r['question_id'] for r in panel}),
                             "groups": len({group_map[r['question_id']] for r in panel}),
                             "strata": {f: dict(sorted(collections.Counter(str(r[f]) for r in panel).items()))
                                        for f in ("case_id", "abf_stage", "model_id", "condition")}}
    return {"seed": seed, "group_by": group_by, "population_size": len(eligible),
            "population_task_hash": _fingerprint([r['task_id'] for r in eligible]),
            "development": panels["development"], "heldout": panels["heldout"],
            "assignment": assignment, "diagnostics": diagnostic,
            "notes": ["Marginal stratification, not full factorial coverage.",
                      "Development and heldout grouping units are disjoint.",
                      "Connected-component grouping avoids shared-chain leakage; scenario source text may still overlap between disconnected components.",
                      "Use scenario grouping to assess transfer to unseen scenario packets.",
                      "No scores or human annotations were used for selection."]}


def select_secondary_sample(records: Iterable[Mapping[str, Any]], questions: Mapping[str, Mapping],
                            *, fraction: float = 0.2, seed: int = 20261127) -> list[dict]:
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be in [0,1]")
    eligible = eligible_answers(records, questions)
    return _balanced_pick(eligible, math.ceil(fraction * len(eligible)), seed)


def _write_csv(path: pathlib.Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def human_review_protocol(reviewers: int) -> dict[str, Any]:
    """Describe the actual human reference without implying expert consensus."""
    if type(reviewers) is not int or reviewers not in {1, 2}:
        raise ValueError("human reviewers must be 1 or 2")
    return {
        "reviewers": reviewers,
        "reference_type": "single_expert" if reviewers == 1 else "two_expert_adjudicated",
        "reference_file": "reviewer_1.csv" if reviewers == 1 else "adjudicated.csv",
    }


def export_validation_package(
    selection: Mapping[str, Any], questions: Mapping[str, Mapping],
    source_records: Iterable[Mapping], inherited_error_checks: Iterable[Mapping],
    output_dir: str | pathlib.Path, *, reviewers: int = 1,
) -> dict[str, str]:
    """Create blank human worksheets and unscored, model-blinded answer packets.

    Single-expert labels are the reference directly. Two-expert mode preserves
    independent initial labels and a separately completed adjudication file.
    """
    protocol = human_review_protocol(reviewers)
    declared = selection.get("human_review")
    if declared is not None and (not isinstance(declared, Mapping)
            or type(declared.get("reviewers")) is not int or dict(declared) != protocol):
        raise ValueError("Requested human review mode differs from the selected panel protocol")
    human_files = (("reviewer_1.csv",) if reviewers == 1
                   else ("reviewer_1.csv", "reviewer_2.csv", "adjudicated.csv"))
    common_files = ("answers_blinded.jsonl", "answers_blinded.md", "assignment_PRIVATE.jsonl",
                    "selection.json", "development_answer_ids.json", "heldout_answer_ids.json", "README.md")
    dest = pathlib.Path(output_dir)
    # A pooled_run subdirectory may already exist. Never overwrite any existing
    # panel/annotation file, including a panel prepared under the other review mode.
    protected = set(human_files + common_files + ("reviewer_2.csv", "adjudicated.csv", "selection.private.json"))
    existing = [dest/n for n in protected if (dest/n).exists()]
    existing += list(dest.glob("reviewer_*.csv")) if dest.exists() else []
    if existing:
        raise FileExistsError("Refusing to overwrite human annotations or panel files; use a new directory")
    sr, er = collections.defaultdict(list), collections.defaultdict(list)
    for source in source_records:
        sr[source["question_id"]].append(source)
    for error in inherited_error_checks:
        er[error["question_id"]].append(error)
    records = {r["task_id"]: r for name in ("development", "heldout") for r in selection[name]}
    blinded, template, prose = [], [], [
        "# Blinded judge-validation answers\n",
        "Record human labels before inspecting automated verdicts or private model and condition metadata. "
        "Do not consult another reviewer's labels when independent review is used. "
        "Criterion decisions and linked answers are not independent statistical samples.\n",
    ]
    assignments = sorted(selection["assignment"], key=lambda x: x["blind_answer_id"])
    for a in assignments:
        q = questions[a["question_id"]]
        req = build_judge_request(q, _text(records[a["task_id"]]), sr[q["question_id"]], er[q["question_id"]])
        blind = {"blind_answer_id": a["blind_answer_id"], **req.payload}
        blinded.append(blind)
        for criterion in req.payload["positive_criteria"]:
            template.append({"blind_answer_id": a["blind_answer_id"], "item_id": criterion["criterion_id"],
                             "kind": "positive", "statement": criterion["text"], "verdict": "", "comment": ""})
        for error in req.payload["error_checks"]:
            template.append({"blind_answer_id": a["blind_answer_id"], "item_id": error["error_id"],
                             "kind": error["origin"] + "_error", "statement": error["trigger"], "verdict": "", "comment": ""})
        prose += [f"\n## {a['blind_answer_id']}\n", f"### Question\n\n{q['question']}\n", f"### Candidate answer\n\n{blind['candidate_answer']}\n", f"### Reference exemplar\n\n{q['expected_answer']}\n", "### Acceptable alternatives\n\n" + "\n".join("- " + x for x in q.get("acceptable_alternatives", [])) + "\n", "### Positive criteria\n\n" + "\n".join(f"- `{c['criterion_id']}` {c['text']} (required={c['required']})" for c in req.payload['positive_criteria']) + "\n", "### Error checks\n\n" + "\n".join(f"- `{e['error_id']}` {e['trigger']} Applicability: {e['applicability']}" for e in req.payload['error_checks']) + "\n", "### Sources\n"]
        for source in req.payload['sources']:
            prose += [f"**{source['source_link_id']} — {source['document_title']}, lines {source['line_start']}–{source['line_end']}**\n\n{source['source_excerpt']}\n"]
    fields = ["blind_answer_id", "item_id", "kind", "statement", "verdict", "comment"]
    dest.mkdir(parents=True, exist_ok=True)
    for name in human_files:
        _write_csv(dest/name, template, fields)
    (dest/'answers_blinded.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False)+'\n' for x in blinded), encoding='utf-8')
    (dest/'answers_blinded.md').write_text('\n'.join(prose), encoding='utf-8')
    (dest/'assignment_PRIVATE.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False)+'\n' for x in assignments), encoding='utf-8')
    metadata = {k: v for k,v in selection.items() if k not in {'development','heldout','assignment'}}
    metadata["human_review"] = protocol
    (dest/'selection.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    for split in ("development", "heldout"):
        ids = [a["blind_answer_id"] for a in assignments if a["split"] == split]
        (dest/(split + '_answer_ids.json')).write_text(json.dumps(
            {"split": split, "blind_answer_ids": ids}, indent=2) + '\n', encoding='utf-8')
    n_development = sum(a["split"] == "development" for a in assignments)
    n_heldout = sum(a["split"] == "heldout" for a in assignments)
    if reviewers == 1:
        instructions = """Complete `reviewer_1.csv` yourself before inspecting the automated judge's
verdicts or private model identities. This file is the human reference directly.
No second-reviewer file or adjudication is required or implied. Preserve the
initial labels and record any subsequent corrections explicitly.

This is a single-expert comparison. It cannot measure inter-human agreement or
establish independently adjudicated expert consensus. Disclose the evaluator's
military expertise and any role in constructing the questions or references.
"""
    else:
        instructions = """Give each expert the blinded answers and their own blank `reviewer_1.csv` or
`reviewer_2.csv`. Each expert labels independently before seeing the other's labels
or automated verdicts. Preserve both initial files. Afterwards resolve differences
and enter final labels in `adjudicated.csv`, the human reference for comparison.
"""
    (dest/'README.md').write_text(f"""# Judge calibration and held-out validation

This panel contains {len(assignments)} generated answers, with {n_development} for
development and {n_heldout} held out. The recorded human review mode uses
{reviewers} expert(s), with `{protocol['reference_file']}` as the reference.

{instructions}
Positive verdicts are `satisfied`, `not_satisfied`, `unscorable`. Error-check verdicts
are `error_present`, `error_absent`, `unscorable`. Enter a verdict for every required
row. Files contain no prefilled human or automated labels.

Use `development_answer_ids.json` to locate development rows without opening the
private assignment file. The analogous held-out list also contains blind IDs only.
`answers_blinded.md` supplies the questions, candidate answers, criteria and sources.
These are lists into the one canonical worksheet, not separate editable label sets.
The combined worksheet does not hide held-out labels automatically. Keep
`assignment_PRIVATE.jsonl` with the experiment coordinator because it identifies
the generating model, condition, repetition and sample assignments.

Use only development labels and development comparisons to refine judge
instructions. Freeze the prompt and judge configuration before examining held-out
agreement. Human labels may all be recorded beforehand while blinded to automated
verdicts. Use the development-only reporting option during prompt development.
Once the judge is frozen, complete the held-out assessment. Do not select a prompt
based on its held-out agreement and then describe that result as held out.

The default workflow uses one judge call per answer, so it does not estimate judge
repeatability. If additional calls are explicitly requested, report their actual
count without treating them as new independent answers. Report human–judge
agreement together with class-specific failure detection, coverage and abstention.
Criteria within an answer and linked questions are dependent observations. No
single threshold automatically declares a judge valid.

Sampling is independent of answer quality. Review `selection.json` to document the
achieved strata and group split, including absent or sparsely represented groups.
Single-expert labels and a small held-out sample limit the scope of the validation.
If a reference defect is discovered, document it and version affected benchmark
records before freezing the experimental corpus again. Do not silently change
references to make automated scores agree.
""", encoding='utf-8')
    return {name: str(dest/name) for name in human_files + common_files}


def load_label_csv(path: str | pathlib.Path, *, allow_blank: bool = False,
                   blind_answer_ids: set[str] | None = None) -> list[dict]:
    """Read selected human rows without validating another split's verdicts.

    Column structure is checked for the complete worksheet. When blind answer
    identifiers are supplied, only those rows are validated and returned.
    """
    with pathlib.Path(path).open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if any(not {"blind_answer_id", "item_id", "kind", "verdict"} <= set(r) for r in rows):
        raise ValueError("Annotation columns are missing")
    if blind_answer_ids is not None:
        rows = [r for r in rows if r["blind_answer_id"] in blind_answer_ids]
    seen = set()
    for r in rows:
        key = (r['blind_answer_id'], r['item_id'], r['kind'])
        if key in seen:
            raise ValueError("Duplicate annotation row")
        seen.add(key)
        if r['kind'] not in LABELS:
            raise ValueError("Unknown annotation kind")
        if not r['verdict'] and allow_blank:
            continue
        if r['verdict'] not in LABELS[r['kind']]:
            raise ValueError(f"Invalid or unfinished annotation at {key}")
    return rows


def _ratio(a: int | float, b: int | float) -> float | None:
    return a / b if b else None


def binary_agreement(reference: Sequence[bool | None], prediction: Sequence[bool | None]) -> dict:
    """True denotes criterion failure / error presence. None denotes unscorable.

    Conditional agreement is reported beside coverage, not substituted for it.
    Cohen's kappa is undefined for a degenerate one-class marginal, returned as null.
    """
    if len(reference) != len(prediction):
        raise ValueError("Reference and prediction lengths differ")
    if any(x is not None and type(x) is not bool for x in [*reference, *prediction]):
        raise ValueError("binary_agreement expects bool or None")
    pairs = [(a,b) for a,b in zip(reference,prediction) if a is not None and b is not None]
    tp = sum(a and b for a,b in pairs)
    fn = sum(a and not b for a,b in pairs)
    fp = sum(not a and b for a,b in pairs)
    tn = sum(not a and not b for a,b in pairs)
    n = len(pairs)
    observed = _ratio(tp+tn, n)
    expected = (((tp+fn)*(tp+fp) + (fp+tn)*(fn+tn)) / (n*n)) if n else None
    kappa = (observed-expected)/(1-expected) if expected is not None and expected < 1 else None
    reference_known = sum(x is not None for x in reference)
    abstained_failures = sum(a is True and b is None for a,b in zip(reference,prediction))
    return {"n_total": len(reference), "n_reference_scorable": reference_known, "n_paired_scorable": n,
            "n_reference_unscorable": len(reference)-reference_known,
            "n_prediction_unscorable_on_reference_scorable": reference_known-n,
            "coverage": _ratio(n, reference_known), "agreement": observed, "cohen_kappa": kappa,
            "true_positive": tp, "false_positive": fp, "false_negative": fn, "true_negative": tn,
            "failure_precision": _ratio(tp, tp+fp), "failure_recall": _ratio(tp, tp+fn),
            "sensitivity": _ratio(tp, tp+fn),
            "failure_f1": _ratio(2*tp, 2*tp+fp+fn), "specificity": _ratio(tn, tn+fp),
            "failure_recall_with_abstentions_as_misses": _ratio(tp, tp+fn+abstained_failures),
            "failure_f1_with_abstentions_as_misses": _ratio(2*tp, 2*tp+fp+fn+abstained_failures),
            "n_abstained_reference_failures": abstained_failures}


def compare_label_records(reference: Sequence[Mapping], prediction: Sequence[Mapping],
                          *, split_by_answer: Mapping[str, str] | None = None,
                          split: str = "heldout") -> dict[str, Any]:
    """Compare independent experts, or a judge against the declared human reference.

    Missing prediction rows count as abstentions and are reported explicitly.
    Extra prediction keys are rejected. Completed human reference labels are required.
    No majority or adjudication is inferred from two expert files.
    """
    def index(rows):
        result = {}
        for r in rows:
            if split_by_answer is not None:
                if r['blind_answer_id'] not in split_by_answer:
                    raise ValueError("Annotation answer is outside the selected panel")
                if split_by_answer[r['blind_answer_id']] != split:
                    continue
            k = (r['blind_answer_id'], r['item_id'], r['kind'])
            if k in result:
                raise ValueError("Duplicate annotation key")
            if r['kind'] not in LABELS or r['verdict'] not in LABELS[r['kind']]:
                raise ValueError("Invalid or incomplete label")
            result[k] = r
        return result
    ref, pred = index(reference), index(prediction)
    if set(pred)-set(ref):
        raise ValueError("Prediction includes keys outside the reference panel")
    if not ref:
        raise ValueError("No reference rows remain in the requested split")

    def event(r):
        if r is None or r['verdict'] == 'unscorable':
            return None
        return r['verdict'] in {'not_satisfied', 'error_present'}

    groups = {}
    for kind in sorted(LABELS):
        keys = sorted(k for k in ref if k[2] == kind)
        groups[kind] = binary_agreement([event(ref[k]) for k in keys], [event(pred.get(k)) for k in keys])
    return {"split": split if split_by_answer is not None else "unspecified",
            "n_answers": len({k[0] for k in ref}), "n_missing_prediction_rows": len(set(ref)-set(pred)),
            "by_kind": groups,
            "notes": ["Agreement conditions on paired scorable labels; inspect coverage and abstention-sensitive failure metrics.",
                      "Criteria within an answer and questions within a scenario are dependent. Counts are not independent validation cases.",
                      "The report does not impose an automatic acceptance threshold."]}


def judgement_label_rows(validated: Mapping[str, Any], request, blind_answer_id: str) -> list[dict]:
    """Convert one validated automated judgement into the human comparison format."""
    from .judge import validate_judgement
    result = validate_judgement(validated, request)
    origins = {x['error_id']: x['origin'] for x in request.payload['error_checks']}
    rows = [{"blind_answer_id": blind_answer_id, "item_id": x['criterion_id'], "kind": "positive", "verdict": x['verdict']}
            for x in result['criterion_results']]
    rows += [{"blind_answer_id": blind_answer_id, "item_id": x['error_id'], "kind": origins[x['error_id']]+'_error', "verdict": x['verdict']}
             for x in result['error_results']]
    return rows


def repeatability_report(runs: Sequence[Sequence[Mapping]],
                         expected_rows: Sequence[Mapping]) -> dict[str, Any]:
    """Describe repeated judging of the same answers, without increasing sample N.

    The expected rows may be blank human templates. A missing automated verdict is
    unavailable, never an agreement. Explicit semantic abstentions can agree but
    are reported separately from agreement on scorable verdicts.
    """
    if len(runs) < 2:
        raise ValueError("Repeatability requires at least two independent judge calls")
    key = lambda x: (x['blind_answer_id'], x['item_id'], x['kind'])
    expected = {key(x) for x in expected_rows}
    if len(expected) != len(expected_rows):
        raise ValueError("Duplicate expected annotation rows")
    indexes = []
    for run in runs:
        index = {}
        for row in run:
            k = key(row)
            if k not in expected or k in index:
                raise ValueError("Unknown or duplicate repeated-judge label")
            if row['kind'] not in LABELS or row['verdict'] not in LABELS[row['kind']]:
                raise ValueError("Invalid repeated-judge label")
            index[k] = row['verdict']
        indexes.append(index)
    summary = {}
    for kind in sorted(LABELS):
        keys = sorted(k for k in expected if k[2] == kind)
        complete = [k for k in keys if all(k in i for i in indexes)]
        scorable = [k for k in complete if all(i[k] != 'unscorable' for i in indexes)]
        same = lambda k: len({i[k] for i in indexes}) == 1
        summary[kind] = {
            'n_expected_items': len(keys), 'n_complete_items': len(complete),
            'n_all_repeats_scorable': len(scorable),
            'all_repeat_agreement_including_abstentions': _ratio(sum(same(k) for k in complete), len(complete)),
            'all_repeat_agreement_on_scorable_items': _ratio(sum(same(k) for k in scorable), len(scorable)),
            'n_items_with_missing_repeats': len(keys)-len(complete),
            'n_items_with_semantic_abstentions': len(complete)-len(scorable),
        }
    return {'repeats': len(runs), 'n_answers': len({k[0] for k in expected}),
            'by_kind': summary,
            'note': 'Repeat calls share answers and references; they are not independent validation cases.'}


def judge_configuration_digest(system_prompt: str, judge_config: Mapping,
                               corpus_digest: str) -> str:
    """Bind approval to instructions/deployment/corpus, independent of key paths or prices."""
    from .model_identity import model_spec
    return _fingerprint({'approval_schema_version': 2, 'system_prompt': system_prompt,
                         'judge_config': model_spec(judge_config), 'corpus_digest': corpus_digest,
                         'judge_protocol_version': PROMPT_VERSION})


def validate_judge_approval(approval: Mapping, *, system_prompt: str,
                          judge_config: Mapping, corpus_digest: str,
                          heldout_report: Mapping) -> dict:
    """Check a human-authored approval record; never infer approval from a score.

    This is a provenance check, not an automatic scientific acceptance threshold.
    The repository ships an explicitly unapproved template. Experiments for judge
    calibration can run without approval; publication claims require completed
    held-out review and an explicitly recorded reviewer decision.
    """
    if approval.get('decision') != 'approved':
        raise ValueError('Judge validation has no recorded human approval')
    for field in ('approved_by', 'approved_at_utc', 'decision_rationale', 'scope'):
        if not isinstance(approval.get(field), str) or not approval[field].strip():
            raise ValueError(f'Judge approval is missing {field}')
    expected = judge_configuration_digest(system_prompt, judge_config, corpus_digest)
    if approval.get('configuration_sha256') != expected:
        raise ValueError('Judge approval does not match the frozen judge/corpus configuration')
    if approval.get('heldout_report_sha256') != _fingerprint(heldout_report):
        raise ValueError('Judge approval does not match the supplied held-out validation report')
    if heldout_report.get('split') != 'heldout' or heldout_report.get('n_answers', 0) <= 0:
        raise ValueError('A genuine held-out report is required')
    report_configuration = heldout_report.get('judge_configuration_sha256')
    if report_configuration is not None and report_configuration != expected:
        raise ValueError('Held-out report does not match the frozen judge/corpus configuration')
    result = dict(approval)
    human = heldout_report.get('human_review')
    if human is not None:
        if not isinstance(human, dict) or human != human_review_protocol(human.get('reviewers')):
            raise ValueError('Held-out report has inconsistent human review metadata')
        if approval.get('human_review') is not None and approval['human_review'] != human:
            raise ValueError('Judge approval and held-out report disagree on the human review protocol')
        # The accepted report hash binds this field even for a generic approval
        # template created before the panel was selected.
        result['human_review'] = dict(human)
    elif approval.get('human_review') is not None:
        raise ValueError('Human review metadata is not supported by the held-out report')
    return result
