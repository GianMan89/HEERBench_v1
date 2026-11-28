"""Incremental studies made of immutable, individually resumable run segments."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import copy
import json
from pathlib import Path
import re
import uuid

from .aggregation import compatibility_key, compatibility_spec, model_signature, read_run_snapshot
from .dataset import Dataset
from .planning import create_plan, freeze_config, model_repetitions
from .runner import Runner, load_run, preflight
from .locking import exclusive_file_lock
from .util import digest, read_jsonl, write_json


@contextmanager
def study_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    with exclusive_file_lock(
        root / "study.lock",
        "Another process is extending this study. Retry after it finishes planning.",
        error_type=ValueError,
    ):
        yield


def _name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", value) or value in {".", ".."}:
        raise ValueError("Study names must contain only letters, numbers, underscores, hyphens or periods")
    return value


def study_path(runs_root: str | Path, study: str) -> Path:
    return Path(runs_root).resolve() / _name(study)


def _segment_configs(root: Path, *, historical: bool = False):
    """Scan complete segments, recovering a plan committed before a registry write."""
    found = []
    for path in sorted((root / "segments").glob("*/manifest.json")):
        folder = path.parent
        if not (folder / "plan.jsonl").exists() or not (folder / "config.frozen.json").exists():
            continue
        if historical:
            snapshot = read_run_snapshot(folder)
            cfg, plan = snapshot["config"], snapshot["plan"]
        else:
            _, cfg, _, plan = load_run(folder)
        if cfg.get("study_id") != root.name:
            raise ValueError("A run segment has a different study identifier")
        found.append((folder, cfg, plan))
    return found


def prepare_study(cfg: dict, runs_root: str | Path, study: str) -> dict:
    """Ensure requested models have planned repetitions 1..N, without API calls.

    Existing segments and responses are never rewritten. Increasing repetitions
    creates only the missing model/repetition pairs. A changed deployment under
    an existing model label requires an explicitly new study or model label.
    """
    root = study_path(runs_root, study)
    dataset = Dataset(cfg["dataset"])
    dataset.validate()
    frozen = freeze_config({**copy.deepcopy(cfg), "study_id": study}, dataset)
    signature = compatibility_key(frozen)
    requested = {m["id"]: m for m in frozen["models"]}
    desired = {mid: set(model_repetitions(frozen, mid)) for mid in requested}
    with study_lock(root):
        index_file = root / "study.json"
        index = json.loads(index_file.read_text()) if index_file.exists() else None
        if index and index["protocol_sha256"] != signature:
            raise ValueError("This study uses different data, prompts, task selection or judge settings. Choose a new study name.")
        segments = _segment_configs(root)
        covered = {mid: set() for mid in requested}
        known_models = {}
        entries = []
        selected_paths = []
        for folder, existing, plan in segments:
            if compatibility_key(existing) != signature:
                raise ValueError("Incompatible run segment found inside a study")
            for m in existing["models"]:
                sig = model_signature(m)
                if m["id"] in known_models and known_models[m["id"]] != sig:
                    raise ValueError("A model label has multiple serving configurations inside the same study")
                known_models[m["id"]] = sig
                if m["id"] in requested and sig != model_signature(requested[m["id"]]):
                    raise ValueError("Model configuration changed for " + m["id"] + ". Use a new study/model label.")
            pair_map = {}
            for row in plan:
                pair_map.setdefault(row["model_id"], set()).add(row["repetition"])
            relevant = False
            for mid, reps in pair_map.items():
                if mid in covered:
                    if covered[mid] & reps:
                        raise ValueError("Duplicate model/repetition work is already registered")
                    covered[mid].update(reps)
                    relevant |= bool(desired[mid] & reps)
            if relevant:
                selected_paths.append(str(folder))
            entries.append({"path": folder.relative_to(root).as_posix(),
                            "model_repetition_ids": {k: sorted(v) for k, v in pair_map.items()},
                            "requests": len(plan)})
        missing = {mid: sorted(reps - covered[mid]) for mid, reps in desired.items() if reps - covered[mid]}
        new_path = None
        new_requests = 0
        if missing:
            new_cfg = copy.deepcopy(frozen)
            new_cfg["models"] = [requested[mid] for mid in missing]
            new_cfg["model_repetition_ids"] = missing
            new_cfg.pop("repetition_ids", None)
            new_path = root / "segments" / ("batch_" + uuid.uuid4().hex[:16])
            manifest = create_plan(new_cfg, dataset, new_path)
            new_requests = manifest["n_requests"]
            selected_paths.append(str(new_path))
            entries.append({"path": new_path.relative_to(root).as_posix(),
                            "model_repetition_ids": missing, "requests": new_requests})
        now = datetime.now(timezone.utc).isoformat()
        index = {"schema_version": 1, "study_id": study, "protocol_sha256": signature,
                 "protocol": compatibility_spec(frozen), "segments": entries,
                 "created_utc": index["created_utc"] if index else now, "updated_utc": now,
                 "model_signatures": {**known_models, **{k: model_signature(v) for k, v in requested.items()}}}
        write_json(index_file, index)
    return {"study": study, "study_path": str(root), "protocol_sha256": signature,
            "requested_models": list(requested), "requested_repetitions": {k: sorted(v) for k, v in desired.items()},
            "new_model_repetitions": missing, "new_generation_requests": new_requests,
            "new_segment": str(new_path) if new_path else None, "segments": selected_paths,
            "note": "Completed requests are retained. Repetitions are totals within this study, not extra reruns."}


def execute_study(selection: dict, *, kind: str = "generation", max_requests: int | None = None) -> dict:
    models = set(selection["requested_models"])
    repetitions = selection["requested_repetitions"]
    checks = []
    if kind == "generation":
        # Inspect all selected segments first so a missing key does not partly spend the planned batch.
        for path in selection["segments"]:
            _, cfg, _, _ = load_run(path)
            if not cfg["is_mock"]:
                check = preflight(path, check_credentials=True, model_ids=models)
                checks.append({"segment": path, **check})
        if any(not check["ready"] for check in checks):
            return {**selection, "dispatched": False, "preflight": checks}
    result = []
    remaining = max_requests
    for path in selection["segments"]:
        if remaining is not None and remaining <= 0:
            break
        runner = Runner(path)
        try:
            before = sum(r["attempts"] for r in runner.store.rows(kind))
            state = runner.step(kind, max_requests=remaining, model_ids=models, repetitions=repetitions)
            after = sum(r["attempts"] for r in runner.store.rows(kind))
            if remaining is not None:
                remaining -= after - before
            result.append({"segment": path, "status": state, "new_attempts": after - before})
        finally:
            runner.close()
    return {**selection, "dispatched": True, "segments_status": result}


def study_segments(runs_root: str | Path, study: str, models: list[str] | None = None) -> list[str]:
    root = study_path(runs_root, study)
    if not (root / "study.json").exists():
        raise ValueError("Unknown study " + study)
    selected = []
    # Discovery is read-only; execution still enforces the exact code hash.
    # Completed historical segments can be pooled for new judge calibration.
    for folder, cfg, _ in _segment_configs(root, historical=True):
        if models is None or set(models) & {m["id"] for m in cfg["models"]}:
            selected.append(str(folder))
    return selected
