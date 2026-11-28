"""Pool completed generation segments for a separate, immutable judge study.

Target responses retain their original identifiers, prompts and outcomes. They
are historical observations, not new model repetitions. No provider is called.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import tempfile

from .aggregation import (ConsolidationError, compatibility_key, model_signature,
                          read_run_snapshot)
from .credentials import reject_inline_credentials
from .judge_validation import judge_configuration_digest
from .store import Store
from .util import canonical, digest, software_fingerprint, write_json, write_jsonl, verified_blob_bytes


_TERMINAL = {"succeeded", "failed", "blocked", "unscorable"}
_COLUMNS = ("id", "kind", "plan", "state", "attempts", "request", "result", "error", "updated")
_JSON_COLUMNS = {"plan", "request", "result", "error"}


def _blob_ids(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key.endswith("_blob"):
                if not isinstance(item, str) or not re.fullmatch(r"[0-9a-f]{64}", item):
                    raise ConsolidationError("Invalid content-addressed blob identifier in source generation")
                yield item
            else:
                yield from _blob_ids(item)
    elif isinstance(value, list):
        for item in value:
            yield from _blob_ids(item)


def _resolved_config(run):
    cfg = deepcopy(run["config"])
    path = run["path"] / "resolved_endpoints.json"
    endpoints = json.loads(path.read_text()) if path.exists() else {}
    for model in cfg["models"] + cfg.get("judge_models", []):
        if model["id"] in endpoints:
            model["base_url"] = endpoints[model["id"]]
            model.pop("base_url_env", None)
    return cfg, endpoints


def _check_dependencies(plan, dataset):
    pmap = {row["request_id"]: row for row in plan}
    for row in plan:
        if row["condition"] != "E3":
            continue
        parents = row.get("parent_request_ids", [])
        questions = row.get("parent_question_ids", [])
        if len(parents) != len(questions):
            raise ConsolidationError("Sequential parent request/question counts differ")
        for pid, qid in zip(parents, questions):
            parent = pmap.get(pid)
            condition = "E1" if dataset.by_id[qid]["abf_stage"] == "A" else "E3"
            if parent is None or any(parent[field] != row[field] for field in ("model_id", "repetition")) \
                    or parent["question_id"] != qid or parent["condition"] != condition:
                raise ConsolidationError("Sequential dependency crosses an absent model, repetition or condition")


def prepare_study_calibration(segments: list[Path], out: Path) -> dict:
    """Create one calibration-only run from completed, compatible segments.

    All planned generation rows must exist and be terminal. Failed and blocked
    rows are copied unchanged, so pooling never improves the apparent completion
    rate by dropping failures. Overlapping logical observations or request IDs
    are rejected even if the source files are identical. No old judge output,
    validation selection, approval, credential file, or execution batch is copied.
    """
    paths = [Path(path).resolve() for path in segments]
    out = Path(out).resolve()
    if not paths:
        raise ConsolidationError("Select at least one completed pilot segment")
    if len(paths) != len(set(paths)):
        raise ConsolidationError("Duplicate source segment; each physical run may be supplied once")
    if any(out == path or out.is_relative_to(path) or path.is_relative_to(out) for path in paths):
        raise ConsolidationError("Calibration output must be outside and must not contain source runs")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ConsolidationError("Calibration output must be a new or empty directory")

    runs = [read_run_snapshot(path) for path in sorted(paths)]
    combined_models, signatures, identities = {}, {}, defaultdict(set)
    repetitions, logical, physical = defaultdict(set), set(), set()
    plans, records, source_ledger, response_ledger = [], [], [], []
    blob_paths, resolved_endpoints = {}, {}
    first_cfg = None
    expected_key = None

    for run in runs:
        cfg, endpoints = _resolved_config(run)
        reject_inline_credentials(cfg)
        if cfg.get("is_mock") or cfg.get("mode") == "demo" or run["contains_mock_observations"] \
                or any(m.get("kind") == "mock" for m in cfg["models"]):
            raise ConsolidationError("Calibration pooling requires real target generations, not mock runs")
        if cfg.get("reused_generation_only") or cfg.get("calibration_only"):
            raise ConsolidationError("Select original generation segments, not calibration or judge-reuse runs")
        key = compatibility_key(cfg, run["dataset"])
        if key != run["compatibility_key"]:
            raise ConsolidationError("Source endpoint resolution differs from its protocol snapshot")
        if expected_key is not None and key != expected_key:
            raise ConsolidationError("Selected segments have incompatible corpus, task, prompt or judge protocols")
        expected_key = key
        first_cfg = first_cfg or cfg

        generations = {record["id"]: record for record in run["generation"]}
        expected_ids = {row["request_id"] for row in run["plan"]}
        if set(generations) != expected_ids or any(r["state"] not in _TERMINAL for r in generations.values()):
            raise ConsolidationError("Finish the selected pilot segments first. Every planned generation must "
                                     "exist and be terminal before calibration pooling; missing, pending, "
                                     "running, submitted and uncertain requests cannot be omitted.")
        for record in generations.values():
            if record["state"] == "succeeded" and (not isinstance(record["result"], dict)
                    or not isinstance(record["result"].get("text"), str)
                    or not isinstance(record["request"], dict)):
                raise ConsolidationError("Successful target generation lacks its exact request or string answer")

        for model in cfg["models"]:
            model_id = model["id"]
            signature = model_signature(model)
            if model_id in signatures and signatures[model_id] != signature:
                raise ConsolidationError(f"Model {model_id!r} has conflicting deployment or inference settings")
            signatures[model_id] = signature
            combined_models.setdefault(model_id, deepcopy(model))
            identities[model_id].update(run["observed_model_identities"].get(model_id, []))
            if len(identities[model_id]) > 1:
                raise ConsolidationError(f"Model {model_id!r} returned conflicting version identities")
        for model_id, endpoint in endpoints.items():
            if model_id in resolved_endpoints and resolved_endpoints[model_id] != endpoint:
                raise ConsolidationError("A resolved model or judge endpoint differs between source segments")
            resolved_endpoints[model_id] = endpoint

        for plan in run["plan"]:
            observation = tuple(plan[field] for field in ("model_id", "repetition", "question_id", "condition"))
            if observation in logical:
                raise ConsolidationError("Duplicate logical model/repetition/question/condition observation")
            if plan["request_id"] in physical:
                raise ConsolidationError("Duplicate physical generation request identifier")
            logical.add(observation)
            physical.add(plan["request_id"])
            repetitions[plan["model_id"]].add(plan["repetition"])
            plans.append(deepcopy(plan))
            record = deepcopy(generations[plan["request_id"]])
            records.append(record)
            response_ledger.append({
                "request_id": record["id"], "source_run_id": run["manifest"]["run_id"],
                "source_run_path": str(run["path"]), "state": record["state"],
                "model_id": plan["model_id"], "repetition": plan["repetition"],
                "source_request_sha256": digest(record["request"]),
                "source_response_sha256": digest(record["result"]),
                "source_record_sha256": digest(record),
                "historical_usage": (record["result"] or {}).get("usage"),
                "historical_cost": (record["result"] or {}).get("cost"),
                "new_target_cost_usd": "0", "is_new_target_observation": False})

        for blob in sorted(set(_blob_ids(run["generation"]))):
            path = run["path"] / "blobs" / blob[:2] / (blob + ".txt")
            try:
                if not path.resolve().is_relative_to(run["path"]) or not path.is_file():
                    raise ValueError("Missing or outside source run")
                verified_blob_bytes(path, blob)
            except (OSError, ValueError) as exc:
                raise ConsolidationError(f"Source generation blob is missing, corrupted or outside its run: {blob} ({exc})") from exc
            blob_paths.setdefault(blob, path)
        source_ledger.append({
            "source_run_path": str(run["path"]), "source_run_id": run["manifest"]["run_id"],
            "source_study_id": cfg.get("study_id"),
            "source_config_sha256": run["manifest"]["config_sha256"],
            "source_plan_sha256": run["manifest"]["plan_sha256"],
            "source_software_sha256": run["manifest"].get("software_sha256"),
            "source_dataset_sha256": cfg["dataset_fingerprint"],
            "source_generation_sha256": digest(run["generation"]),
            "source_snapshot_fingerprint": run["snapshot_fingerprint"],
            "planned_requests": len(run["plan"]),
            "generation_states": dict(Counter(r["state"] for r in run["generation"])),
            "judge_records_not_copied": len(run["judges"]),
        })

    plans.sort(key=lambda p: (p["wave"], p["model_id"], p["repetition"], p["request_id"]))
    order = {p["request_id"]: i for i, p in enumerate(plans)}
    records.sort(key=lambda r: order[r["id"]])
    response_ledger.sort(key=lambda r: order[r["request_id"]])
    dataset = runs[0]["dataset"]
    _check_dependencies(plans, dataset)
    provenance = {
        "purpose": "study_calibration", "reused_generation_only": True,
        "source_protocol_key": expected_key, "source_segments": source_ledger,
        "new_target_generation_requests": 0, "new_target_generation_cost_usd": "0",
        "judge_results_copied": 0, "approvals_copied": 0,
        "target_generation_sha256": digest(records), "blob_ids_sha256": digest(sorted(blob_paths)),
    }
    cfg = deepcopy(first_cfg)
    cfg.update(models=[combined_models[key] for key in sorted(combined_models)],
               dataset=str(dataset.root), model_repetition_ids={key: sorted(value) for key, value in sorted(repetitions.items())},
               repetitions=max(len(value) for value in repetitions.values()),
               reused_generation_only=True, calibration_only=True, purpose="study_calibration",
               generation_reuse=provenance, is_mock=False)
    cfg.pop("repetition_ids", None)
    cfg.pop("study_segment_id", None)
    if compatibility_key(cfg, dataset) != expected_key:
        raise ConsolidationError("Calibration pooling would change the scientific protocol")
    fingerprint = digest(cfg)
    manifest = {
        "run_id": fingerprint[:20], "config_sha256": fingerprint, "plan_sha256": digest(plans),
        "dataset_sha256": dataset.fingerprint, "software_sha256": software_fingerprint(),
        "created_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
        "platform": platform.platform(), "is_mock": False, "calibration_only": True,
        "n_questions": len(cfg["question_ids"]), "n_chains": len(cfg["selected_chain_ids"]),
        "n_requests": len(plans), "requests_by_condition": dict(Counter(p["condition"] for p in plans)),
        "model_repetition_ids": cfg["model_repetition_ids"], "study_id": cfg.get("study_id"),
        "judge_prompt_sha256": digest(cfg["judge_system"]), **provenance,
        "scope": "Judge calibration on unchanged target responses pooled across completed compatible segments",
        "target_response_reuse_note": "Copied responses retain original repetition IDs and are not new target observations.",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=out.name + ".staging-", dir=out.parent))
    store = None
    try:
        write_json(temporary / "config.frozen.json", cfg)
        write_json(temporary / "manifest.json", manifest)
        write_jsonl(temporary / "plan.jsonl", plans)
        write_json(temporary / "resolved_endpoints.json", resolved_endpoints)
        for blob, source in blob_paths.items():
            target = temporary / "blobs" / blob[:2] / (blob + ".txt")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(verified_blob_bytes(source, blob))
            if hashlib.sha256(target.read_bytes()).hexdigest() != blob:
                raise ConsolidationError("Source blob changed while calibration snapshot was being copied")
        store = Store(temporary)
        store.db.executemany(
            "INSERT INTO tasks(" + ",".join(_COLUMNS) + ") VALUES(" + ",".join("?" for _ in _COLUMNS) + ")",
            [tuple(canonical(r[k]) if k in _JSON_COLUMNS and r[k] is not None else r[k]
                   for k in _COLUMNS) for r in records])
        store.db.commit()
        store.event(None, "generation_records_reused", {
            "purpose": "study_calibration", "source_runs": len(runs), "count": len(records),
            "new_target_generation_cost_usd": "0",
            "note": "Target costs attached to copied records were incurred in the source segments."})
        store.export()
        store.close()
        store = None
        write_json(temporary / "generation_reuse.json", {
            **provenance, "records_copied": len(records), "generation_blobs_copied": len(blob_paths),
            "generation_states": dict(Counter(r["state"] for r in records))})
        write_json(temporary / "source_segment_ledger.json", source_ledger)
        write_jsonl(temporary / "generation_reuse_ledger.jsonl", response_ledger)
        primary = next(m for m in cfg["judge_models"] if m["id"] == cfg["judge"]["primary"])
        write_json(temporary / "judge_approval.template.json", {
            "decision": "not_yet_reviewed", "approved_by": "", "approved_at_utc": "", "scope": "",
            "decision_rationale": "", "heldout_report_sha256": "",
            "configuration_sha256": judge_configuration_digest(cfg["judge_system"], primary, dataset.fingerprint),
            "note": "New calibration bundle. No previous judge scores, human panel assignments or approvals were inherited."})
        (temporary / "CALIBRATION_BUNDLE.md").write_text(
            "# Study calibration bundle\n\n"
            "Target answers and failed outcomes were copied unchanged from completed source segments. "
            "They retain their original request and repetition identifiers. All target token and cost "
            "records are historical. No new target generation was performed.\n\n"
            "Create one development/held-out panel from this combined pool before inspecting judge "
            "scores. Split shared questions/chains together across all models. Tune on development "
            "labels only. This bundle is excluded from ordinary result consolidation.\n\n"
            "No credential files, previous judge results or approvals were copied. Target generation "
            "and recovery are disabled. New judge calls remain separately logged and billed.\n",
            encoding="utf-8")
        if out.exists():
            out.rmdir()
        os.replace(temporary, out)
    except BaseException:
        if store is not None:
            store.close()
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {"run": str(out), "run_path": str(out), "run_id": manifest["run_id"],
            "source_runs": len(runs), "records_copied": len(records),
            "available_answers": sum(r["state"] == "succeeded" for r in records), "manifest": manifest}
