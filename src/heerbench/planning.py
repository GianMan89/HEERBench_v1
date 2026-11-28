from __future__ import annotations

from collections import Counter
import platform
from pathlib import Path
import sys
from datetime import datetime, timezone

from .dataset import Dataset
from .util import digest, write_json, write_jsonl, software_fingerprint


def freeze_config(cfg: dict, dataset: Dataset) -> dict:
    """Freeze scientific inputs without creating a run or reading credentials."""
    selected = dataset.select_chains(cfg["mode"], cfg["chains_per_scenario"], cfg["seed"], cfg.get("chain_ids"))
    qids = sorted(set().union(*(dataset.chain_nodes(c) for c in selected)))
    if cfg["mode"] == "full" and not cfg.get("chain_ids") and len(qids) != len(dataset.questions):
        raise ValueError("Full plan silently lost questions")
    return {**cfg, "selected_chain_ids": selected, "question_ids": qids,
            "dataset_fingerprint": dataset.fingerprint,
            "generator_system": cfg.get("generator_system") or Path(cfg["prompt_file"]).read_text(encoding="utf-8"),
            "judge_system": cfg.get("judge_system") or Path(cfg["judge_prompt_file"]).read_text(encoding="utf-8"),
            "is_mock": cfg["mode"] == "demo", "protocol_version": cfg.get("protocol_version", "heerbench-abf-v1")}


def model_repetitions(cfg: dict, model_id: str) -> list[int]:
    values = cfg.get("model_repetition_ids", {}).get(model_id,
                 cfg.get("repetition_ids", list(range(1, cfg["repetitions"] + 1))))
    if not isinstance(values, list) or not values or any(type(x) is not int or x < 1 for x in values):
        raise ValueError("Repetition identifiers must be nonempty positive integers")
    if len(values) != len(set(values)):
        raise ValueError("Duplicate repetition identifiers")
    return sorted(values)


def create_plan(cfg: dict, dataset: Dataset, outdir: str | Path) -> dict:
    outdir = Path(outdir)
    if outdir.exists() and any(outdir.iterdir()):
        raise ValueError("Run directory is not empty. Resume it or choose a new directory.")
    dataset.validate()
    frozen = freeze_config(cfg, dataset)
    selected, qids = frozen["selected_chain_ids"], frozen["question_ids"]
    fingerprint = digest(frozen)
    rows = []
    for model in frozen["models"]:
        for rep in model_repetitions(frozen, model["id"]):
            keys = {(qid, c): "hb-" + digest([fingerprint, model["id"], rep, qid, c])[:32]
                    for qid in qids for c in frozen["conditions"]
                    if c == "E1" or dataset.by_id[qid]["abf_stage"] != "A"}
            for (qid, condition), request_id in keys.items():
                q = dataset.by_id[qid]
                parents = list(dataset.parents.get(qid, ())) if condition != "E1" else []
                if set(parents) - set(qids):
                    raise ValueError("Selected chain subset is not dependency closed")
                parent_tasks = [keys[(p, "E1" if dataset.by_id[p]["abf_stage"] == "A" else "E3")]
                                for p in parents] if condition == "E3" else []
                rows.append({"request_id": request_id, "model_id": model["id"],
                             "repetition": rep, "condition": condition, "question_id": qid,
                             "case_id": q["case_id"], "abf_stage": q["abf_stage"],
                             "parent_question_ids": parents, "parent_request_ids": parent_tasks,
                             "wave": {"A": 0, "B": 1, "F": 2}[q["abf_stage"]],
                             "chain_ids": sorted(set(q["chain_ids"]) & set(selected))})
    rows.sort(key=lambda r: (r["wave"], digest([cfg["seed"], r["request_id"]])))
    manifest = {"run_id": fingerprint[:20], "config_sha256": fingerprint,
                "plan_sha256": digest(rows),
                "software_sha256": software_fingerprint(),
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "python": sys.version, "platform": platform.platform(),
                "is_mock": frozen["is_mock"], "n_questions": len(qids),
                "study_id": frozen.get("study_id"),
                "model_repetition_ids": {m["id"]: model_repetitions(frozen, m["id"]) for m in frozen["models"]},
                "n_chains": len(selected), "n_requests": len(rows),
                "requests_by_condition": dict(Counter(r["condition"] for r in rows)),
                "scope": "RQ1 and RQ2 only; no source-degradation or reasoning-mode ablations"}
    write_json(outdir / "config.frozen.json", frozen)
    write_json(outdir / "manifest.json", manifest)
    write_jsonl(outdir / "plan.jsonl", rows)
    return manifest

