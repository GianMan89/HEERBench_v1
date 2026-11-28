"""Connect the frozen generation journal to blinded scoring and analysis."""
from __future__ import annotations

from dataclasses import asdict
from collections import defaultdict
import csv
import json
from pathlib import Path

from .judge import JudgeRequest, SYSTEM_PROMPT, build_judge_request
from .judge_validation import (select_secondary_sample, select_validation_sample,
    export_validation_package, validate_judge_approval, judgement_label_rows,
    load_label_csv, compare_label_records, repeatability_report, judge_configuration_digest,
    question_groups, human_review_protocol)
from .providers import GenerationRequest
from .util import canonical, digest, write_json, write_jsonl


def generation_records(runner):
    return [{**r["plan"], "task_id": r["id"], "generation_status": r["state"],
             "text": (r["result"] or {}).get("text", ""),
             "is_mock": runner.cfg["is_mock"],
             "provider": (r["result"] or {}).get("provider")}
            for r in runner.store.rows("generation")]


def _judge_system(runner):
    return runner.cfg.get("judge_system", SYSTEM_PROMPT)


def _validate_panel_selection(selection, records, dataset):
    """Preserve selected content, exact assignments and group separation."""
    current = {r['task_id']: r for r in records if r['generation_status'] == 'succeeded'}
    selected = {}
    split_for_task = {}
    for split in ('development', 'heldout'):
        if not selection.get(split):
            raise ValueError('A validation panel needs development and held-out answers')
        for row in selection[split]:
            task_id = row['task_id']
            if task_id in selected or task_id not in current:
                raise ValueError('Validation selection duplicates or references unavailable answers')
            for field in ('text', 'question_id', 'model_id', 'condition', 'repetition', 'case_id', 'abf_stage'):
                if row.get(field) != current[task_id].get(field):
                    raise ValueError('Validation answer or its metadata changed after selection')
            selected[task_id] = current[task_id]
            split_for_task[task_id] = split
    groups = question_groups(dataset.by_id, dataset.chains, selection['group_by'])
    assigned, blind_ids = set(), set()
    split_groups = {'development': set(), 'heldout': set()}
    for row in selection['assignment']:
        task = row['task_id']
        if task not in selected or task in assigned or row['blind_answer_id'] in blind_ids:
            raise ValueError('Validation assignment has duplicate or unknown identifiers')
        if row['split'] != split_for_task[task] or row['group_id'] != groups[selected[task]['question_id']]:
            raise ValueError('Validation assignment changed its split or grouping unit')
        for field in ('question_id', 'model_id', 'condition', 'repetition', 'case_id', 'abf_stage'):
            if row.get(field) != selected[task].get(field):
                raise ValueError('Validation assignment metadata does not match the selected answer')
        assigned.add(task)
        blind_ids.add(row['blind_answer_id'])
        split_groups[row['split']].add(row['group_id'])
    if assigned != set(selected) or split_groups['development'] & split_groups['heldout']:
        raise ValueError('Validation panel has incomplete assignments or development/held-out leakage')
    return selected


def prepare_judges(runner, *, validation_selection=None, validation_repetitions=1,
                   approval_file=None, heldout_report_file=None, model_ids=None, repetitions=None):
    cfg, ds = runner.cfg, runner.dataset
    records = generation_records(runner)
    from .runner import selected_record
    records = [r for r in records if selected_record(r, model_ids, repetitions)]
    pending = {"pending", "running", "submitted", "uncertain"}
    if any(r["generation_status"] in pending for r in records):
        raise ValueError("Complete the planned generation run before freezing the judge sample")
    primary_id = cfg["judge"]["primary"]
    primary = next(m for m in cfg["judge_models"] if m["id"] == primary_id)
    if (cfg["mode"] == "full" or cfg.get("calibration_only")) and validation_selection is None and not cfg["is_mock"]:
        if not approval_file and (runner.root / "judge_approval.json").exists():
            approval_file = runner.root / "judge_approval.json"
        if not heldout_report_file and (runner.root / "judge_validation_heldout.json").exists():
            heldout_report_file = runner.root / "judge_validation_heldout.json"
        if not approval_file or not heldout_report_file:
            raise ValueError("Full scoring requires the actual held-out judge report and recorded expert acceptance. Calibration and pilot scoring do not.")
        report = json.loads(Path(heldout_report_file).read_text())
        approval = validate_judge_approval(json.loads(Path(approval_file).read_text()), system_prompt=_judge_system(runner),
            judge_config=primary, corpus_digest=ds.fingerprint, heldout_report=report)
        write_json(runner.root / "judge_approval.json", approval)
        write_json(runner.root / "judge_validation_heldout.json", report)
    succeeded = [r for r in records if r["generation_status"] == "succeeded"]
    secondary_ids = set()
    if validation_selection:
        if type(validation_repetitions) is not int or validation_repetitions < 1:
            raise ValueError('Calibration repetitions must be a positive integer')
        selection = json.loads(Path(validation_selection).read_text())
        selected_ids = set(_validate_panel_selection(selection, records, ds))
        succeeded = [r for r in succeeded if r["task_id"] in selected_ids]
        frozen_selection = runner.root / "validation_selection.json"
        if frozen_selection.exists() and json.loads(frozen_selection.read_text()) != selection:
            raise ValueError("This run already has a different validation panel. Use a new judge fork.")
        write_json(frozen_selection, selection)
    elif not cfg["is_mock"]:
        groups = defaultdict(list)
        for record in records:
            unit = ((record["model_id"], record["repetition"])
                    if cfg["judge"].get("secondary_sampling_unit") == "model_repetition" else "all")
            groups[unit].append(record)
        secondary_ids = {r["task_id"] for group in groups.values()
                         for r in select_secondary_sample(group, ds.by_id,
                            fraction=cfg["judge"].get("secondary_fraction", 0.2), seed=cfg["seed"])}
    before = len(runner.store.rows("judge"))
    for r in succeeded:
        qid = r["question_id"]
        contract = build_judge_request(ds.by_id[qid], r["text"], ds.for_question(ds.sources, qid),
            ds.for_question(ds.errors, qid), ds.for_question(ds.local_errors, qid),
            system_prompt=_judge_system(runner))
        contract_blob = runner.store.put_blob(canonical(asdict(contract)))
        configs = [("calibration" if validation_selection else "primary", primary_id)]
        if r["task_id"] in secondary_ids:
            configs.append(("secondary", cfg["judge"]["secondary"]))
        for role, judge_id in configs:
            judge = next(m for m in cfg["judge_models"] if m["id"] == judge_id)
            repetitions = validation_repetitions if role == "calibration" else 1
            for jr in range(1, repetitions + 1):
                rid = "judge-" + digest([r["task_id"], judge_id, role, jr, contract.prompt_hash, judge])[:32]
                plan = {**{k: r[k] for k in ("model_id", "condition", "repetition", "question_id", "case_id", "abf_stage")},
                        "request_id": rid, "generation_request_id": r["task_id"],
                        "parent_request_ids": [], "judge_role": role, "judge_repetition": jr,
                        "executor_id": judge_id, "judge_contract_blob": contract_blob}
                # A short, explicit verdict for every criterion/check requires more than a one-number score.
                output_cap = min(judge.get("max_output_tokens", 16384),
                                 max(4096, 256 * (len(contract.criterion_ids) + len(contract.error_ids)) + 512))
                request = GenerationRequest(rid, judge["model"], contract.system, "",
                    canonical({"evaluation": contract.payload, "required_response_schema": contract.response_schema}),
                    max_output_tokens=output_cap)
                runner.store.add(plan, "judge")
                row = runner.store.get(rid)
                if row["state"] == "pending" and row["request"] is None:
                    runner.store.update(rid, "pending", request=runner.store.pack_request(request.to_dict()))
    previous_file = runner.root / "judge_sample.json"
    previous = json.loads(previous_file.read_text()) if previous_file.exists() else {}
    all_secondary = sorted(set(previous.get("secondary_answer_ids", [])) | secondary_ids)
    scored_ids = {r["plan"]["generation_request_id"] for r in runner.store.rows("judge")
                  if r["plan"]["judge_role"] == ("calibration" if validation_selection else "primary")}
    write_json(previous_file, {"primary_judge": primary_id,
        "secondary_judge": cfg["judge"].get("secondary"), "secondary_answer_ids": all_secondary,
        "secondary_fraction_requested": cfg["judge"].get("secondary_fraction", .2),
        "secondary_sampling_unit": cfg["judge"].get("secondary_sampling_unit", "global"),
        "primary_answer_count": len(scored_ids), "calibration": bool(validation_selection),
        "judge_prompt_sha256": digest(_judge_system(runner)),
        "validated_for_publication": (runner.root / "judge_approval.json").exists(),
        "is_mock": cfg["is_mock"]})
    return {"new_judge_requests": len(runner.store.rows("judge")) - before,
            "total_judge_requests": len(runner.store.rows("judge"))}


def prepare_validation(runner, outdir, n=60, development_n=10, group_by="component", *, reviewers=1):
    records = generation_records(runner)
    if any(r['generation_status'] in {'pending', 'running', 'submitted', 'uncertain'} for r in records):
        raise ValueError('Complete the planned generation run before freezing the validation panel')
    selection = select_validation_sample(records, runner.dataset.by_id,
        runner.dataset.chains, n=n, development_n=development_n,
        seed=runner.cfg["seed"], group_by=group_by)
    selection["human_review"] = human_review_protocol(reviewers)
    outdir = Path(outdir)
    paths = export_validation_package(selection, runner.dataset.by_id, runner.dataset.sources,
                                      runner.dataset.errors, outdir, reviewers=reviewers)
    write_json(outdir / "selection.private.json", selection)
    primary = next(m for m in runner.cfg["judge_models"] if m["id"] == runner.cfg["judge"]["primary"])
    write_json(outdir / "judge_approval.template.json", {
        "approval_schema_version": 2,
        "human_review": selection["human_review"],
        "decision": "not_yet_reviewed", "approved_by": "", "approved_at_utc": "", "scope": "",
        "decision_rationale": "", "configuration_sha256": judge_configuration_digest(
            _judge_system(runner), primary, runner.dataset.fingerprint), "heldout_report_sha256": ""})
    return paths


def _panel_human_review(selection):
    # Existing panels without this field used two independent experts.
    value = selection.get("human_review")
    if value is None:
        return human_review_protocol(2)
    if not isinstance(value, dict):
        raise ValueError("Invalid human review protocol")
    expected = human_review_protocol(value.get("reviewers"))
    if value != expected:
        raise ValueError("Human review metadata contradicts its declared reference")
    return expected


def _selected_human_labels(path, expected_rows, blind_ids):
    """Check the template's complete key set while interpreting only requested labels."""
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"blind_answer_id", "item_id", "kind", "verdict"}
    if any(not required <= set(row) for row in rows):
        raise ValueError("Annotation columns are missing")
    key = lambda row: (row["blind_answer_id"], row["item_id"], row["kind"])
    keys = [key(row) for row in rows]
    if len(keys) != len(set(keys)) or set(keys) != {key(row) for row in expected_rows}:
        raise ValueError(f"{Path(path).name} must retain every expected criterion/error-check row exactly once")
    return load_label_csv(path, blind_answer_ids=blind_ids)


def calibration_report(runner, paneldir, *, split="all"):
    """Report agreement with the declared human reference, without inventing raters.

    A development-only report neither interprets held-out verdicts nor emits an
    approval hash. Human CSV templates retain all panel rows, but only the chosen
    split must be labelled. Missing automated judgements remain missing coverage.
    """
    if split not in {"all", "development", "heldout"}:
        raise ValueError("split must be development, heldout or all")
    paneldir = Path(paneldir)
    selection = json.loads((paneldir / "selection.private.json").read_text())
    public_selection = paneldir / "selection.json"
    if public_selection.exists():
        public = json.loads(public_selection.read_text())
        if public.get("human_review") != selection.get("human_review"):
            raise ValueError("Public and private human review protocols differ")
    frozen_panel = runner.root / "validation_selection.json"
    if frozen_panel.exists() and json.loads(frozen_panel.read_text()) != selection:
        raise ValueError("Validation panel changed after judge preparation")
    human = _panel_human_review(selection)
    selected = _validate_panel_selection(selection, generation_records(runner), runner.dataset)
    assignment = {a["task_id"]: a for a in selection["assignment"]}
    split_map = {a["blind_answer_id"]: a["split"] for a in selection["assignment"]}
    labels_to_report = ("development", "heldout") if split == "all" else (split,)
    blind_ids = {key for key, value in split_map.items() if value in labels_to_report}
    by_repeat = {}
    expected_rows = []
    for task_id, record in selected.items():
        qid = record["question_id"]
        contract = build_judge_request(runner.dataset.by_id[qid], record["text"],
            runner.dataset.for_question(runner.dataset.sources, qid),
            runner.dataset.for_question(runner.dataset.errors, qid),
            runner.dataset.for_question(runner.dataset.local_errors, qid),
            system_prompt=_judge_system(runner))
        blind = assignment[task_id]["blind_answer_id"]
        expected_rows += [{"blind_answer_id": blind, "item_id": c["criterion_id"], "kind": "positive"}
                          for c in contract.payload["positive_criteria"]]
        expected_rows += [{"blind_answer_id": blind, "item_id": e["error_id"], "kind": e["origin"] + "_error"}
                          for e in contract.payload["error_checks"]]
    primary_id = runner.cfg["judge"]["primary"]
    primary = next(m for m in runner.cfg["judge_models"] if m["id"] == primary_id)
    seen = set()
    for row in runner.store.rows("judge"):
        p = row["plan"]
        if p["judge_role"] != "calibration" or p["generation_request_id"] not in assignment:
            continue
        blind = assignment[p["generation_request_id"]]["blind_answer_id"]
        if blind not in blind_ids:
            continue
        if p["executor_id"] != primary_id:
            raise ValueError("Calibration report mixes judge configurations")
        unit = (p["generation_request_id"], p["judge_repetition"])
        if unit in seen:
            raise ValueError("Duplicate calibration judgement for an answer and repetition")
        seen.add(unit)
        by_repeat.setdefault(p["judge_repetition"], [])
        contract = JudgeRequest(**json.loads(runner.store.get_blob(p["judge_contract_blob"])))
        if contract.system != _judge_system(runner):
            raise ValueError("Calibration report mixes judge prompts")
        expected_id = "judge-" + digest([p["generation_request_id"], primary_id,
            "calibration", p["judge_repetition"], contract.prompt_hash, primary])[:32]
        request = runner.store.unpack_request(row["request"]) if row["request"] else None
        if row["id"] != expected_id or not request or request.get("model") != primary["model"]:
            raise ValueError("Calibration report mixes judge configurations")
        if not row["result"] or "judgement" not in row["result"]:
            continue
        labels = judgement_label_rows(row["result"]["judgement"], contract, blind)
        by_repeat[p["judge_repetition"]].extend(labels)
    reference = _selected_human_labels(paneldir / human["reference_file"], expected_rows, blind_ids)
    human1 = human2 = None
    if human["reviewers"] == 2:
        human1 = _selected_human_labels(paneldir / "reviewer_1.csv", expected_rows, blind_ids)
        human2 = _selected_human_labels(paneldir / "reviewer_2.csv", expected_rows, blind_ids)
    selected_expected = [r for r in expected_rows if r["blind_answer_id"] in blind_ids]
    reports = {"human_review": human, "reported_split": split,
               "judge_repetitions": sorted(by_repeat),
               "expert_agreement": None, "repeatability": None,
               "limitations": ["Criteria within answers and linked questions are dependent observations.",
                               "Agreement with a small expert-labelled sample does not establish correctness on every benchmark output."]}
    if human["reviewers"] == 1:
        reports["expert_agreement_note"] = "Not measured. One human reviewer provides the reference; no independent expert consensus is claimed."
        reports["limitations"].append("Report the reviewer's involvement in constructing the benchmark and references; this can limit independence.")
    for label in labels_to_report:
        report = compare_label_records(reference, by_repeat.get(1, []), split_by_answer=split_map, split=label)
        ref_part = [r for r in reference if split_map[r["blind_answer_id"]] == label]
        report.update(human_review=human,
            reference_labels_sha256=digest([{k: r[k] for k in ("blind_answer_id", "item_id", "kind", "verdict")}
                                          for r in sorted(ref_part, key=lambda x: (x["blind_answer_id"], x["item_id"], x["kind"]))]),
            judge_configuration_sha256=judge_configuration_digest(_judge_system(runner), primary, runner.dataset.fingerprint),
            panel_sha256=digest(selection),
            judge_repetitions_planned=sorted(by_repeat),
            sample_diagnostics=selection.get("diagnostics", {}).get(label, {}))
        report["notes"] += (["Reference labels come from one military expert; inter-human agreement is not measured."]
                            if human["reviewers"] == 1 else ["Reference labels come from recorded two-expert adjudication."])
        reports[label] = report
        write_json(paneldir / ("judge_" + label + "_report.json"), report)
    if human["reviewers"] == 2:
        agreement_split = "heldout" if "heldout" in labels_to_report else "development"
        reports["expert_agreement"] = compare_label_records(human1, human2, split_by_answer=split_map, split=agreement_split)
    if len(by_repeat) >= 2:
        reports["repeatability"] = repeatability_report([by_repeat[k] for k in sorted(by_repeat)], selected_expected)
    else:
        reports["repeatability_note"] = "Not measured. Repeatability requires at least two independently requested judge calls on the same answers."
    summary_name = "calibration_report.json" if split == "all" else "calibration_" + split + "_report.json"
    write_json(paneldir / summary_name, reports)
    if "heldout" in reports:
        write_json(paneldir / "judge_approval.hashes.json", {
            "heldout_report_sha256": digest(reports["heldout"]),
            "human_review": human,
            "note": "Copy the digest into the approval template only after expert review. No approval is inferred."})
    return reports


def analysis_records(runner):
    primary = {}
    for r in runner.store.rows("judge"):
        if r["plan"]["judge_role"] != "primary":
            continue
        key = r["plan"]["generation_request_id"]
        if key in primary:
            raise ValueError("Multiple primary judgements for one answer. Resolve protocol versions explicitly.")
        primary[key] = r
    rows = []
    for g in runner.store.rows("generation"):
        j = primary.get(g["id"])
        state = g["state"] if g["state"] in {"succeeded", "failed", "blocked"} else "pending"
        jstate = j["state"] if j and j["state"] in {"succeeded", "failed", "unscorable"} else "pending"
        rows.append({**g["plan"], "generation_status": state, "judge_status": jstate,
                     "judge_role": "primary", "score": (j["result"] or {}).get("score", {}) if j else {}})
    return rows


def export_analysis(runner, outdir, bootstrap=10000, plots=False):
    from .analysis import analyse
    from .reporting import write_report
    report = analyse(runner.plan, analysis_records(runner), runner.dataset.questions,
                     runner.dataset.chains, runner.dataset.dependencies,
                     bootstrap=bootstrap, seed=runner.cfg["seed"], is_mock=runner.cfg["is_mock"],
                     selected_chain_ids=runner.cfg["selected_chain_ids"])
    approval_file = runner.root / "judge_approval.json"
    report["judge_validated_for_publication"] = False
    report["judge_validation_basis"] = None
    if approval_file.exists():
        primary = next(m for m in runner.cfg["judge_models"] if m["id"] == runner.cfg["judge"]["primary"])
        heldout_file = runner.root / "judge_validation_heldout.json"
        if not heldout_file.exists():
            raise ValueError("Judge approval is missing its accepted held-out report")
        approval = validate_judge_approval(json.loads(approval_file.read_text()),
            system_prompt=_judge_system(runner), judge_config=primary,
            corpus_digest=runner.dataset.fingerprint,
            heldout_report=json.loads(heldout_file.read_text()))
        report["judge_validated_for_publication"] = True
        report["judge_validation_basis"] = approval.get("human_review")
    paths = write_report(report, outdir, plots=plots)
    # A separate method ledger makes actual tokens, missing usage and all attempt counts inspectable.
    costs = []
    for row in runner.store.rows():
        result = row["result"] or {}
        if result:
            costs.append({"request_id": row["id"], "kind": row["kind"], "state": row["state"],
                          "model_id": row["plan"].get("executor_id", row["plan"]["model_id"]),
                          "attempts": row["attempts"], "usage": result.get("usage"), "cost": result.get("cost"),
                          "model_requested": result.get("model_requested"), "model_returned": result.get("model_returned"),
                          "execution_mode": result.get("execution_mode"), "latency_seconds": result.get("latency_seconds")})
    write_jsonl(Path(outdir) / "observed_usage_and_costs.jsonl", costs)
    from .cost_estimation import measured_charge_completeness
    charge_rows = runner.store.rows()
    if runner.cfg.get("reused_generation_only"):
        charge_rows = [r for r in charge_rows if r["kind"] == "judge"]
    charge_report = measured_charge_completeness(charge_rows)
    charge_report["historical_generation_cost_excluded"] = bool(runner.cfg.get("reused_generation_only"))
    charge_report["note"] = "Observed generation costs in a judge-development fork belong to the parent run, not new charges."
    write_json(Path(outdir) / "charge_completeness.json", charge_report)
    return paths
