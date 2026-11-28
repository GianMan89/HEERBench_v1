"""Read-only consolidation of compatible, incrementally collected experiments.

The frozen plan defines denominators. Run directories are never opened through
Runner or Store and are never changed by this module. A model/repetition/question
collision is an error, not an opportunity to choose a better answer or score.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
from typing import Any, Mapping

from .analysis import analyse
from .dataset import Dataset
from .judge import PROMPT_VERSION, SYSTEM_PROMPT
from .judge_validation import validate_judge_approval
from .model_identity import model_spec, model_signature
from .reporting import write_report
from .util import digest, read_jsonl, project_root, write_json, write_jsonl


class ConsolidationError(ValueError):
    """Run records cannot be combined without changing the measurement target."""


def _prompt(cfg: Mapping[str, Any], frozen: str, file_key: str, default: str | None = None) -> str:
    if frozen in cfg:
        value = cfg[frozen]
    elif cfg.get(file_key):
        value = Path(cfg[file_key]).read_text(encoding="utf-8")
    else:
        value = default
    if not isinstance(value, str) or not value.strip():
        raise ConsolidationError(f"Missing frozen {frozen}")
    return value


def compatibility_spec(cfg: Mapping[str, Any], dataset: Dataset | None = None) -> dict:
    """Return the scientific protocol identity, independent of selected model count.

    A frozen config is preferred. For planning, pass the already validated dataset
    to resolve the selected chains/questions before the first run is written.
    """
    chains = cfg.get("selected_chain_ids")
    questions = cfg.get("question_ids")
    fingerprint = cfg.get("dataset_fingerprint")
    if dataset is not None:
        if chains is None:
            chains = dataset.select_chains(cfg["mode"], cfg.get("chains_per_scenario", 2),
                                            cfg.get("seed", 20261127), cfg.get("chain_ids"))
        if questions is None:
            questions = sorted(set().union(*(dataset.chain_nodes(c) for c in chains)))
        fingerprint = fingerprint or dataset.fingerprint
    if not fingerprint or chains is None or questions is None:
        raise ConsolidationError("Compatibility requires the dataset fingerprint and complete selected task set")
    judge_policy = deepcopy(cfg.get("judge", {}))
    judge_models = {m["id"]: m for m in cfg.get("judge_models", [])}
    roles = {}
    for role in ("primary", "secondary"):
        model_id = judge_policy.pop(role, None)
        roles[role] = model_spec(judge_models[model_id]) if model_id in judge_models else {"unresolved_model_id": model_id}
    return {
        "compatibility_schema": "heerbench-cross-run-v1",
        "protocol_version": cfg.get("protocol_version", "heerbench-abf-e123-v1"),
        "mode": cfg.get("mode"),
        "dataset_fingerprint": fingerprint,
        "selected_chain_ids": sorted(chains),
        "question_ids": sorted(questions),
        "conditions": sorted(cfg.get("conditions", [])),
        "generator_system_sha256": digest(_prompt(cfg, "generator_system", "prompt_file")),
        "generation_settings": deepcopy(cfg.get("generation", {})),
        "judge_system_sha256": digest(_prompt(cfg, "judge_system", "judge_prompt_file", SYSTEM_PROMPT)),
        "judge_protocol_version": cfg.get("judge_protocol_version", PROMPT_VERSION),
        "scoring_version": cfg.get("scoring_version", "heerbench-atomic-scoring-v1"),
        "judge_models": roles,
        "judge_policy": judge_policy,
        "selection_seed": cfg.get("seed", 20261127),
    }


def compatibility_key(cfg: Mapping[str, Any], dataset: Dataset | None = None) -> str:
    return digest(compatibility_spec(cfg, dataset))


def _dataset_for(cfg: Mapping[str, Any], dataset: Dataset | None) -> Dataset:
    if dataset is None:
        path = Path(cfg["dataset"])
        if not path.exists():
            path = project_root() / "data/heerbench_v05"
        dataset = Dataset(path)
    dataset.validate()
    if dataset.fingerprint != cfg["dataset_fingerprint"]:
        raise ConsolidationError("Run dataset differs from the available corpus snapshot")
    return dataset


def _read_journal(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        raw = [dict(r) for r in connection.execute("SELECT * FROM tasks ORDER BY id")]
    except sqlite3.DatabaseError as exc:
        raise ConsolidationError(f"Cannot read the experiment journal {path.name}: {exc}") from exc
    finally:
        connection.close()
    for row in raw:
        for key in ("plan", "request", "result", "error"):
            row[key] = json.loads(row[key]) if row[key] is not None else None
    return raw


def _validate_approval(path: Path, cfg: Mapping[str, Any], dataset: Dataset) -> dict | None:
    approval_path, report_path = path / "judge_approval.json", path / "judge_validation_heldout.json"
    if not approval_path.exists():
        return None
    if not report_path.exists():
        raise ConsolidationError("Incomplete judge acceptance record in " + str(path))
    primary_id = cfg.get("judge", {}).get("primary")
    primary = next((m for m in cfg.get("judge_models", []) if m["id"] == primary_id), None)
    if primary is None:
        raise ConsolidationError("Judge approval has no matching frozen primary model")
    try:
        approval = validate_judge_approval(json.loads(approval_path.read_text()),
            system_prompt=_prompt(cfg, "judge_system", "judge_prompt_file", SYSTEM_PROMPT),
            judge_config=primary, corpus_digest=dataset.fingerprint,
            heldout_report=json.loads(report_path.read_text()))
    except ValueError as exc:
        raise ConsolidationError("Invalid judge acceptance in " + str(path) + ": " + str(exc)) from exc
    return approval


def _validate_plan_scope(plan: list[dict], cfg: Mapping[str, Any], dataset: Dataset) -> None:
    """Check that a frozen task set has not lost an entire planned condition/task."""
    qids = set(cfg["question_ids"])
    selected = cfg["selected_chain_ids"]
    if not selected or set(selected) - set(dataset.by_chain):
        raise ConsolidationError("Unknown or empty selected chain set")
    actual_qids = set().union(*(dataset.chain_nodes(c) for c in selected))
    if actual_qids != qids or len(qids) != len(cfg["question_ids"]):
        raise ConsolidationError("Selected questions do not equal the selected complete chain union")
    models = {m["id"] for m in cfg["models"]}
    if len(models) != len(cfg["models"]):
        raise ConsolidationError("Duplicate frozen model labels")
    expected = set()
    for model in models:
        reps = cfg.get("model_repetition_ids", {}).get(model,
            cfg.get("repetition_ids", list(range(1, cfg.get("repetitions", 1) + 1))))
        if not reps or len(set(reps)) != len(reps) or any(type(r) is not int or r < 1 for r in reps):
            raise ConsolidationError("Invalid explicit repetition identifiers in source configuration")
        expected.update((model, condition, rep, qid) for rep in reps for qid in qids
                        for condition in cfg["conditions"]
                        if condition == "E1" or dataset.by_id[qid]["abf_stage"] != "A")
    actual = [tuple(p[k] for k in ("model_id", "condition", "repetition", "question_id")) for p in plan]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ConsolidationError("Frozen plan does not cover exactly its declared models, repetitions, conditions and task set")
    for row in plan:
        q = dataset.by_id[row["question_id"]]
        parents = list(dataset.parents.get(row["question_id"], ())) if row["condition"] != "E1" else []
        if row.get("parent_question_ids", []) != parents:
            raise ConsolidationError("Plan parent questions differ from the validated dependency graph")
        if row["condition"] != "E3" and row.get("parent_request_ids", []):
            raise ConsolidationError("Only sequential questions may depend on generated parent requests")
        if row.get("case_id") != q["case_id"] or row.get("abf_stage") != q["abf_stage"]:
            raise ConsolidationError("Plan task metadata differs from the validated corpus")


def read_run_snapshot(path: str | Path, dataset: Dataset | None = None) -> dict:
    """Read one coherent journal snapshot without importing execution machinery.

    Different historical code hashes are recorded, not executed or silently
    relabelled as current code. Frozen configuration, plan and corpus hashes must
    still match. An unstarted plan is represented entirely as pending outcomes.
    """
    path = Path(path).resolve()
    cfg = json.loads((path / "config.frozen.json").read_text())
    manifest = json.loads((path / "manifest.json").read_text())
    plan = read_jsonl(path / "plan.jsonl")
    if digest(cfg) != manifest["config_sha256"]:
        raise ConsolidationError("Frozen configuration hash mismatch in " + str(path))
    if digest(plan) != manifest["plan_sha256"]:
        raise ConsolidationError("Frozen request-plan hash mismatch in " + str(path))
    dataset = _dataset_for(cfg, dataset)
    _validate_plan_scope(plan, cfg, dataset)
    rows = _read_journal(path / "journal.sqlite")
    pmap = {r["request_id"]: r for r in plan}
    if len(pmap) != len(plan):
        raise ConsolidationError("Duplicate request identifiers in source plan")
    generation = {}
    judges = []
    primary = {}
    judge_identities = defaultdict(set)
    for row in rows:
        if row["kind"] == "generation":
            if row["id"] not in pmap or row["plan"] != pmap[row["id"]]:
                raise ConsolidationError("Generation journal differs from its frozen source plan")
            if row["id"] in generation:
                raise ConsolidationError("Duplicate generation row in source journal")
            generation[row["id"]] = row
        elif row["kind"] == "judge":
            judges.append(row)
            parent = row["plan"].get("generation_request_id")
            if parent not in pmap:
                raise ConsolidationError("Judge row refers to an answer outside its source plan")
            for field in ("model_id", "condition", "repetition", "question_id"):
                if row["plan"].get(field) != pmap[parent][field]:
                    raise ConsolidationError("Judge metadata does not match its target generation")
            role = row["plan"].get("judge_role")
            if role in ("primary", "secondary"):
                if row["plan"].get("executor_id") != cfg.get("judge", {}).get(role):
                    raise ConsolidationError("Judge executor differs from its frozen role assignment")
                returned = (row.get("result") or {}).get("model_returned")
                if returned:
                    judge_identities[role].add(returned)
            if role == "primary":
                if parent in primary:
                    raise ConsolidationError("Multiple primary judgements for one generated answer")
                primary[parent] = row
        else:
            raise ConsolidationError("Unknown journal task kind")
    results = []
    identities = defaultdict(set)
    for p in plan:
        row = generation.get(p["request_id"])
        judge = primary.get(p["request_id"])
        gen_state = row["state"] if row and row["state"] in {"succeeded", "failed", "blocked"} else "pending"
        judge_state = judge["state"] if judge and judge["state"] in {"succeeded", "failed", "unscorable"} else "pending"
        results.append({**p, "generation_status": gen_state, "judge_status": judge_state,
                        "judge_role": "primary", "score": (judge["result"] or {}).get("score", {}) if judge else {}})
        if row and row["result"]:
            returned = row["result"].get("model_returned")
            if returned:
                identities[p["model_id"]].add(returned)
    endpoints_path = path / "resolved_endpoints.json"
    endpoints = json.loads(endpoints_path.read_text()) if endpoints_path.exists() else {}
    models = {m["id"]: m for m in cfg["models"]}
    if set(p["model_id"] for p in plan) - set(models):
        raise ConsolidationError("A source plan names a model outside its frozen configuration")
    approval = _validate_approval(path, cfg, dataset)
    validation_basis = deepcopy(approval.get("human_review")) if approval else None
    resolved_cfg = {**cfg, "judge_models": [model_spec(m, endpoints.get(m["id"])) | {"id": m["id"]}
                                            for m in cfg.get("judge_models", [])]}
    protocol = compatibility_spec(resolved_cfg, dataset)
    contains_mock = False
    for row in rows:
        result = row.get("result") or {}
        raw = result.get("raw_response") or {}
        contains_mock |= (result.get("provider") == "mock" or result.get("is_mock") is True
                          or (isinstance(raw, Mapping) and raw.get("is_mock") is True))
    # This fingerprint is for exact copied-run deduplication only. Updated timestamps
    # are excluded, while raw responses, identities, errors and scored results remain.
    observations = [{k: r[k] for k in ("id", "kind", "plan", "state", "attempts", "request", "result", "error")}
                    for r in rows]
    return {"path": path, "config": cfg, "manifest": manifest, "plan": plan,
            "dataset": dataset, "generation": list(generation.values()), "judges": judges,
            "results": results, "judge_validated": approval is not None,
            "judge_validation_basis": validation_basis,
            "compatibility": protocol, "compatibility_key": digest(protocol),
            "model_specs": {key: model_spec(model, endpoints.get(key)) for key, model in models.items()},
            "observed_model_identities": {k: sorted(v) for k, v in identities.items()},
            "observed_judge_identities": {k: sorted(v) for k, v in judge_identities.items()},
            "contains_mock_observations": contains_mock,
            "observation_fingerprint": digest(observations),
            "snapshot_fingerprint": digest({"config_sha256": manifest["config_sha256"],
                "plan_sha256": manifest["plan_sha256"], "observations": observations,
                "judge_validated": approval is not None,
                "judge_validation_basis": validation_basis})}


def _study_id(cfg: Mapping[str, Any]):
    if "study_id" in cfg:
        return cfg["study_id"]
    study = cfg.get("study")
    return study.get("id") if isinstance(study, Mapping) else study


def discover_run_paths(root: str | Path, output: str | Path | None = None) -> list[Path]:
    root = Path(root).resolve()
    out = Path(output).resolve() if output else None
    candidates = [root] if (root / "manifest.json").is_file() else []
    candidates.extend(p.parent for p in root.rglob("manifest.json"))
    return sorted({p.resolve() for p in candidates
                   if (p / "config.frozen.json").is_file() and (p / "plan.jsonl").is_file()
                   and not (out and (p == out or p.is_relative_to(out)))})


def _source_ledger(run: dict, root: Path) -> dict:
    cfg, manifest = run["config"], run["manifest"]
    p = run["path"]
    return {"run_id": manifest["run_id"],
            "source_directory": p.relative_to(root).as_posix() if p.is_relative_to(root) else str(p),
            "study_id": _study_id(cfg), "created_utc": manifest.get("created_utc"),
            "software_sha256": manifest.get("software_sha256"),
            "config_sha256": manifest["config_sha256"], "plan_sha256": manifest["plan_sha256"],
            "snapshot_fingerprint": run["snapshot_fingerprint"],
            "planned_requests": len(run["plan"]),
            "models": sorted({r["model_id"] for r in run["plan"]}),
            "repetition_ids_by_model": {m: sorted({r["repetition"] for r in run["plan"] if r["model_id"] == m})
                                       for m in sorted({r["model_id"] for r in run["plan"]})},
            "generation_states": dict(Counter(r["generation_status"] for r in run["results"])),
            "judge_states": dict(Counter(r["judge_status"] for r in run["results"])),
            "judge_validated_for_publication": run["judge_validated"],
            "judge_validation_basis": run["judge_validation_basis"],
            "observed_model_identities": run["observed_model_identities"],
            "observed_judge_identities": run["observed_judge_identities"]}


def _merge_group(runs: list[dict], root: Path, bootstrap: int) -> dict:
    exact, retained, duplicates = {}, [], []
    for run in runs:
        fp = run["snapshot_fingerprint"]
        if fp in exact:
            duplicates.append({"source_directory": str(run["path"]),
                               "same_physical_snapshot_as": str(exact[fp]["path"]),
                               "reason": "identical_frozen_plan_and_observations"})
        else:
            exact[fp] = run
            retained.append(run)
    model_specs, observed, logical, identifiers = {}, defaultdict(set), {}, {}
    observed_judges = defaultdict(set)
    plans, results, ledger = [], [], []
    for run in retained:
        for role, names in run["observed_judge_identities"].items():
            observed_judges[role].update(names)
            if len(observed_judges[role]) > 1:
                raise ConsolidationError(f"The {role} judge returned different version identifiers. Keep these scoring configurations separate.")
        for model, spec in run["model_specs"].items():
            if model in model_specs and model_specs[model] != spec:
                raise ConsolidationError(f"Model {model!r} has incompatible deployment or inference settings. Use a distinct model label/study.")
            model_specs[model] = spec
            observed[model].update(run["observed_model_identities"].get(model, []))
            if len(observed[model]) > 1:
                raise ConsolidationError(f"Model {model!r} returned different version identifiers across the pooled observations. Keep these versions separate.")
        for p in run["plan"]:
            key = tuple(p[k] for k in ("model_id", "condition", "repetition", "question_id"))
            if key in logical:
                raise ConsolidationError(f"Overlapping logical observation {key} in {run['path']} and {logical[key]}. Repetitions must have unique explicit IDs.")
            if p["request_id"] in identifiers:
                raise ConsolidationError("A generation request ID occurs in different physical run snapshots")
            logical[key], identifiers[p["request_id"]] = str(run["path"]), str(run["path"])
            plans.append(p)
        results.extend(run["results"])
        ledger.append(_source_ledger(run, root))
    first = retained[0]
    ds, cfg = first["dataset"], first["config"]
    # Parent identities are part of the logical run, not a permission to use another
    # model's output or a preceding repetition after incremental collection.
    pmap = {r["request_id"]: r for r in plans}
    for row in plans:
        expected_questions = list(row.get("parent_question_ids", []))
        parents = list(row.get("parent_request_ids", []))
        if row["condition"] == "E3":
            if len(parents) != len(expected_questions):
                raise ConsolidationError("Sequential parent request/question counts differ")
            for pid, qid in zip(parents, expected_questions):
                parent = pmap.get(pid)
                if parent is None or parent["question_id"] != qid or parent["model_id"] != row["model_id"] or parent["repetition"] != row["repetition"]:
                    raise ConsolidationError("Sequential dependency crosses an absent model or repetition")
                expected_condition = "E1" if ds.by_id[qid]["abf_stage"] == "A" else "E3"
                if parent["condition"] != expected_condition:
                    raise ConsolidationError("Sequential child uses the wrong parent condition")
    report = analyse(plans, results, ds.questions, ds.chains, ds.dependencies,
                     bootstrap=bootstrap, seed=cfg.get("seed", 20261127), is_mock=False,
                     selected_chain_ids=cfg["selected_chain_ids"])
    approved = all(r["judge_validated"] for r in retained)
    report["judge_validated_for_publication"] = approved
    report["judge_validation_basis"] = (deepcopy(retained[0]["judge_validation_basis"])
        if len(retained) == 1 else [
            {"run_id": row["run_id"], "source_directory": row["source_directory"],
             "calibration_accepted": row["judge_validated_for_publication"],
             "human_review": deepcopy(row["judge_validation_basis"])} for row in ledger])
    report["metadata"].update({"aggregation_kind": "compatible_incremental_runs",
        "compatibility_key": first["compatibility_key"], "source_runs": len(retained),
        "judge_validated_for_publication": approved,
        "study_ids": sorted({str(_study_id(r["config"])) for r in retained}),
        "source_software_hashes": sorted({r["manifest"].get("software_sha256", "unknown") for r in retained}),
        "observed_model_identities": {m: sorted(v) for m, v in observed.items()},
        "observed_judge_identities": {m: sorted(v) for m, v in observed_judges.items()}})
    if len(report["metadata"]["source_software_hashes"]) > 1:
        report["metadata"]["warnings"].append("Source runs used different code fingerprints. No old execution code was run during consolidation. Inspect source versions for changes outside the recorded protocol before publication.")
    if not approved:
        report["metadata"]["warnings"].append("At least one source run lacks matching accepted held-out judge validation. The combined report is provisional.")
    if len({r["study_id"] for r in ledger}) > 1:
        report["metadata"]["warnings"].append("Several study identifiers contribute nonoverlapping observations to this protocol-compatible report. Preserve their collection dates and provenance.")
    return {"key": first["compatibility_key"], "compatibility": first["compatibility"],
            "report": report, "plan": plans, "results": results, "ledger": ledger,
            "duplicates": duplicates, "model_specs": model_specs}


def consolidate_runs(root: str | Path, out: str | Path, *, all_runs: bool = True,
                     study: str | None = None, bootstrap: int = 10000, plots: bool = False) -> dict:
    """Write one report per compatible protocol, never pool mismatched settings.

    Mock runs and judge-development forks are excluded explicitly. Plans with no
    journal are retained as pending. The input directories remain read-only.
    """
    root, out = Path(root).resolve(), Path(out).resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    if bootstrap < 0:
        raise ValueError("bootstrap must be nonnegative")
    if not all_runs and study is None and not (root / "manifest.json").exists():
        raise ValueError("Select a study or use all_runs for a run collection")
    sources = discover_run_paths(root, out)
    if any(out == p or p.is_relative_to(out) for p in sources):
        raise ConsolidationError("Report output must not replace or contain source run directories")
    groups, excluded, cache = defaultdict(list), [], {}
    for path in sources:
        cfg = json.loads((path / "config.frozen.json").read_text())
        reason = None
        if cfg.get("is_mock") or cfg.get("mode") == "demo":
            reason = "mock_pipeline_run"
        elif cfg.get("reused_generation_only"):
            reason = "judge_development_fork_not_new_generation"
        elif study is not None and _study_id(cfg) != study:
            reason = "different_study"
        if reason:
            excluded.append({"source_directory": str(path), "reason": reason})
            continue
        fingerprint = cfg.get("dataset_fingerprint")
        run = read_run_snapshot(path, cache.get(fingerprint))
        cache[fingerprint] = run["dataset"]
        if run["contains_mock_observations"]:
            excluded.append({"source_directory": str(path), "reason": "mock_observation_in_journal"})
            continue
        groups[run["compatibility_key"]].append(run)
    if not groups:
        raise ConsolidationError("No eligible non-mock generation runs match this selection")
    # Validate every group before writing any result. A later collision cannot
    # leave an apparently complete index pointing to selectively retained groups.
    merged = [_merge_group(runs, root, bootstrap) for _, runs in sorted(groups.items())]
    index = {"schema_version": 1, "selection": {"all_runs": all_runs, "study": study},
             "source_root": str(root), "groups": [], "excluded_runs": excluded,
             "notes": ["Each group has identical dataset, task subset, prompt and judge settings.",
                       "Repetition IDs are preserved; planned failures and pending requests remain in denominators.",
                       "No scored answer is selected by quality, and no model/judge API is called."]}
    for group in merged:
        dest = out / ("protocol_" + group["key"][:16])
        paths = write_report(group["report"], dest, plots=plots)
        write_json(dest / "compatibility.json", group["compatibility"])
        write_json(dest / "model_configurations.json", group["model_specs"])
        write_json(dest / "source_run_ledger.json", group["ledger"])
        write_json(dest / "duplicate_snapshots.json", group["duplicates"])
        write_jsonl(dest / "merged_plan.jsonl", group["plan"])
        write_jsonl(dest / "merged_primary_results.jsonl", group["results"])
        meta = group["report"]["metadata"]
        index["groups"].append({"compatibility_key": group["key"], "mode": group["compatibility"]["mode"],
            "protocol_fields_differing_from_first_group": sorted(k for k, v in group["compatibility"].items()
                                                                 if merged[0]["compatibility"].get(k) != v),
            "directory": str(dest), "source_runs": len(group["ledger"]),
            "models": meta["models"], "repetitions_by_model": meta["repetitions_by_model"],
            "planned_requests": len(group["plan"]), "selected_chains": len(group["compatibility"]["selected_chain_ids"]),
            "judge_validated_for_publication": group["report"]["judge_validated_for_publication"],
            "judge_validation_basis": group["report"]["judge_validation_basis"],
            "duplicate_snapshots_excluded": len(group["duplicates"]), "files": paths})
    write_json(out / "index.json", index)
    lines = ["# Consolidated HEERBench reports", "",
             "Different protocol groups are reported separately. This index does not pool their scores.", "",
             "| Protocol | Scope | Models | Planned requests | Source runs | Accepted judge calibration |",
             "|---|---|---:|---:|---:|---|"]
    for group in index["groups"]:
        label = "protocol_" + group["compatibility_key"][:16]
        lines.append(f"| [{label}]({label}/report.md) | {group['mode']} | {len(group['models'])} | {group['planned_requests']} | {group['source_runs']} | {'All source runs' if group['judge_validated_for_publication'] else 'Provisional'} |")
    lines += ["", f"Explicitly excluded runs: {len(excluded)}. Reasons are recorded in index.json.", "",
              "The source-run ledger retains collection dates, frozen plan/configuration hashes, software versions, model identities, repetition IDs, coverage and the recorded human-reference basis for judge calibration. See each report for single-expert or two-expert reference scope; legacy records do not imply a reviewer count."]
    (out / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"index_json": str(out / "index.json"), "index_markdown": str(out / "index.md"),
            "protocol_groups": len(index["groups"]), "groups": index["groups"], "excluded_runs": excluded}
