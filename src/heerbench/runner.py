from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
from decimal import Decimal
import json
import os
from pathlib import Path
import time
from datetime import date

from .config import resolve_endpoint
from .credentials import CredentialError, validate_credentials
from .costs import cost_from_usage, conservative_request_reservation, estimate_request_cost
from .dataset import Dataset
from .prompts import build_generation_request
from .providers import BatchHandle, GenerationRequest, GenerationResponse, ProviderError, TokenUsage, build_provider
from .store import Store
from .util import canonical, digest, estimated_tokens, project_root, read_jsonl, token_upper_bound, write_json, software_fingerprint


def load_run(root: str | Path):
    root = Path(root).resolve()
    cfg = json.loads((root / "config.frozen.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    if digest(cfg) != manifest["config_sha256"]:
        raise ValueError("Frozen configuration was edited. Create a new run for protocol changes.")
    if manifest.get("software_sha256") != software_fingerprint():
        raise ValueError("Experiment code changed after planning. Use the original code revision to resume.")
    path = Path(cfg["dataset"])
    if not path.exists():
        path = project_root() / "data/heerbench_v05"
    dataset = Dataset(path)
    dataset.validate()
    if dataset.fingerprint != cfg["dataset_fingerprint"]:
        raise ValueError("Run dataset differs from the planned dataset")
    plan = read_jsonl(root / "plan.jsonl")
    if digest(plan) != manifest["plan_sha256"]:
        raise ValueError("Frozen request plan was edited")
    return root, cfg, dataset, plan


def context_check(request: GenerationRequest, model: dict) -> dict:
    text = request.system + "\n" + request.prefix + "\n" + request.task
    method = "utf8_bytes_divided_by_3_plus_25_percent_margin_and_1024_framing_tokens"
    estimate = int(estimated_tokens(text) * 1.25) + 1024
    if model.get("tokenizer_file"):
        from tokenizers import Tokenizer
        import hashlib
        path = Path(model["tokenizer_file"])
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if sha != model.get("tokenizer_sha256"):
            raise ValueError("Tokenizer asset changed or has no frozen sha256")
        estimate = len(Tokenizer.from_file(str(path)).encode(text, add_special_tokens=False).ids) + 1024
        method = "local_tokenizer_plus_1024_message_framing_tokens"
    bound = token_upper_bound(text) + 1024
    limit = model.get("max_context_tokens")
    input_limit = model.get("max_input_tokens", limit)
    if not limit or not input_limit:
        return {"fits_estimate": False, "reason": "Unverified context capacity", "method": method,
                "input_estimate": estimate, "byte_upper_bound": bound}
    return {"fits_estimate": estimate <= input_limit and estimate + request.max_output_tokens <= limit,
            "fits_byte_bound": bound <= input_limit and bound + request.max_output_tokens <= limit,
            "input_estimate": estimate, "byte_upper_bound": bound, "method": method,
            "max_context_tokens": limit, "max_input_tokens": input_limit,
            "output_reservation": request.max_output_tokens,
            "note": "The exact complete prompt is sent. Provider context errors remain visible; no truncation."}


def effective_transport(cfg: dict, model: dict) -> str:
    requested = cfg["execution"]["transport"]
    native = model.get("kind") in {"openai_responses", "anthropic", "gemini"}
    return "batch" if requested == "batch" and native else "standard"


def price_is_current(model: dict) -> bool:
    expiry = model.get("pricing_valid_until")
    return not expiry or date.today().isoformat() <= str(expiry)


def selected_record(record, model_ids=None, repetitions=None):
    row = record.get("plan", record)
    return ((model_ids is None or row["model_id"] in model_ids) and
            (repetitions is None or row["repetition"] in repetitions.get(row["model_id"], [])))


def preflight(root: str | Path, check_credentials: bool = False, model_ids=None) -> dict:
    root, cfg, dataset, plan = load_run(root)
    reports = []
    for model in cfg["models"]:
        if model_ids is not None and model["id"] not in model_ids:
            continue
        problems, warnings = [], []
        if model.get("enabled") is False:
            problems.append(model.get("provision_note", "Model is not enabled"))
        if model.get("provider_max_output_tokens") and model["max_output_tokens"] > model["provider_max_output_tokens"]:
            problems.append("Configured output budget exceeds the verified provider output ceiling")
        try:
            resolved = resolve_endpoint(model)
        except ValueError as exc:
            resolved = model
            problems.append(str(exc))
        if check_credentials and not cfg["is_mock"]:
            try:
                validate_credentials(resolved)
            except CredentialError as exc:
                problems.append(str(exc))
        if not price_is_current(model):
            warnings.append("The saved rate card has expired. Cost estimates are unavailable until prices are refreshed.")
            if cfg["execution"].get("max_cost_usd") is not None:
                problems.append("An expired price card cannot enforce a cost cap")
        contexts = []
        seen = set()
        for row in plan:
            unit = (row["question_id"], row["condition"])
            if row["model_id"] != model["id"] or unit in seen:
                continue
            seen.add(unit)
            # Reserve actual oracle answers for preflight. Actual sequential outputs checked again before every call.
            parents = {p: dataset.by_id[p]["expected_answer"] for p in row["parent_question_ids"]}
            req = build_generation_request(row, cfg, dataset, parents)
            check = context_check(req, model)
            contexts.append(check)
            if not check["fits_estimate"]:
                problems.append(f"Estimated full prompt exceeds verified capacity: {row['question_id']} {row['condition']}")
        if any(not c.get("fits_byte_bound") for c in contexts):
            warnings.append("Some prompts fit only an estimate. Use a frozen local tokenizer for tighter admission checks.")
        if effective_transport(cfg, model) == "standard" and cfg["execution"]["transport"] == "batch":
            warnings.append("This endpoint uses standard requests and standard pricing; native batch is unavailable.")
        if model.get("weight_revision") in {None, "unknown", "provider_managed", "provider_managed_preview"}:
            warnings.append("Weight revision is not independently pinned; retain returned identities and run dates.")
        reports.append({"model_id": model["id"], "ready": not problems,
                        "problems": sorted(set(problems)), "warnings": warnings,
                        "effective_transport": effective_transport(cfg, model),
                        "max_input_estimate": max((c["input_estimate"] for c in contexts), default=0),
                        "max_byte_upper_bound": max((c["byte_upper_bound"] for c in contexts), default=0)})
    result = {"ready": all(m["ready"] for m in reports), "models": reports,
              "network_requests": 0, "credentials_checked": check_credentials}
    write_json(root / "preflight.json", result)
    return result


def estimate_run(root: str | Path, expected_output_tokens: int = 1000) -> dict:
    from .cost_estimation import estimate_generation_request_cost

    if type(expected_output_tokens) is not int or expected_output_tokens < 1:
        raise ValueError("Expected output tokens must be a positive integer")
    root, cfg, dataset, plan = load_run(root)
    reports = []
    for model in cfg["models"]:
        total, upper = Decimal(0), Decimal(0)
        complete, upper_complete = price_is_current(model), price_is_current(model)
        reps = defaultdict(Decimal)
        n = 0
        for row in plan:
            if row["model_id"] != model["id"]:
                continue
            parents = {p: dataset.by_id[p]["expected_answer"] for p in row["parent_question_ids"]}
            req = build_generation_request(row, cfg, dataset, parents)
            estimate = estimated_tokens(req.system + req.prefix + req.task) + 256
            # Prior generated answers do not yet exist. Reserve output caps for every direct parent.
            bound = token_upper_bound(req.system + req.prefix + req.task) + 1024
            if row["condition"] == "E3":
                bound += len(row["parent_question_ids"]) * model["max_output_tokens"] * 4
            mode = effective_transport(cfg, model)
            cost = estimate_generation_request_cost(req, model, mode, estimate, expected_output_tokens)
            reserve = conservative_request_reservation(bound, req.max_output_tokens,
                                                       model.get("pricing", {}), mode)
            n += 1
            if cost.complete:
                total += Decimal(cost.total_usd)
                reps[row["repetition"]] += Decimal(cost.total_usd)
            else:
                complete = False
            if reserve.complete:
                upper += Decimal(reserve.total_usd)
            else:
                upper_complete = False
        reports.append({"model_id": model["id"], "requests": n, "transport": effective_transport(cfg, model),
                        "generation_estimate_usd": str(total) if complete else None,
                        "generation_estimate_per_repetition_usd": {str(k): str(v) for k, v in reps.items()} if complete else None,
                        "conservative_reservation_usd": str(upper) if upper_complete else None,
                        "pricing_complete": complete})
    result = {"models": reports, "expected_output_tokens_per_answer": expected_output_tokens,
              "assumptions": ["Estimated input tokens, no cache hits, native batch only where supported.",
                              "Every configured cache breakpoint is assumed to miss and write its prefix at the stated TTL rate.",
                              "References substitute for unknown generated parents in the planning estimate.",
                              "Output includes billed reasoning where the provider reports it.",
                              "Judge costs have a separate offline forecast and are recalculated from observed usage after scoring.",
                              "Pilot measurements supersede these estimates. Tax, hosting, storage and failed-attempt charges may be additional."],
              "generation_total_usd": str(sum(Decimal(x["generation_estimate_usd"]) for x in reports))
                  if all(x["pricing_complete"] for x in reports) else None}
    write_json(root / "cost_estimate.json", result)
    return result


class Runner:
    def __init__(self, root: str | Path, *, provider_factory=build_provider):
        self.root, self.cfg, self.dataset, self.plan = load_run(root)
        self.store = Store(self.root)
        for row in self.plan:
            self.store.add(row)
        self.factory = provider_factory
        self.providers = {}

    def close(self):
        for p in self.providers.values():
            p.close()
        self.store.close()

    def model(self, row: dict) -> dict:
        models = self.cfg["models"] if not row.get("judge_role") else self.cfg["judge_models"]
        key = row.get("executor_id", row["model_id"])
        return next(m for m in models if m["id"] == key)

    def provider(self, row: dict):
        model = self.model(row)
        key = model["id"]
        if key not in self.providers:
            resolved = resolve_endpoint(model)
            # Endpoint locations are not secrets. Freeze them at first execution.
            path = self.root / "resolved_endpoints.json"
            endpoints = json.loads(path.read_text()) if path.exists() else {}
            location = resolved.get("base_url")
            if key in endpoints and endpoints[key] != location:
                raise ValueError("Endpoint changed within a frozen run. Create a new run.")
            endpoints[key] = location
            write_json(path, endpoints)
            self.providers[key] = self.factory(resolved)
        return self.providers[key]

    def request(self, record: dict) -> GenerationRequest:
        if record["request"]:
            return GenerationRequest(**self.store.unpack_request(record["request"]))
        row = record["plan"]
        answers = {}
        for pid, qid in zip(row["parent_request_ids"], row["parent_question_ids"]):
            parent = self.store.get(pid)
            if parent["state"] != "succeeded":
                raise ValueError("Parent request has not succeeded")
            answers[qid] = parent["result"]["text"]
        return build_generation_request(row, self.cfg, self.dataset, answers)

    def ready(self, kind: str, model_ids=None, repetitions=None) -> list[dict]:
        result = []
        for r in self.store.rows(kind):
            if not selected_record(r, model_ids, repetitions):
                continue
            if r["state"] != "pending":
                continue
            parents = [self.store.get(p) for p in r["plan"].get("parent_request_ids", [])]
            if any(p is None for p in parents):
                raise ValueError("Unknown parent request in frozen plan")
            if any(p["state"] in {"failed", "blocked", "unscorable"} for p in parents):
                self.store.update(r["id"], "blocked", error={"kind": "upstream_transport_failure"})
            elif all(p["state"] == "succeeded" for p in parents):
                result.append(r)
        return result

    def _reservation(self, record: dict, request: GenerationRequest, mode: str):
        model = self.model(record["plan"])
        check = context_check(request, model)
        if not check["fits_estimate"]:
            raise ValueError("Complete prompt does not fit configured model context: " + request.request_id)
        reserve = conservative_request_reservation(check["byte_upper_bound"], request.max_output_tokens,
                                                    model.get("pricing", {}), mode).to_dict()
        if not price_is_current(model):
            reserve.update(complete=False, total_usd=None, note="Saved rate card expired")
        return check, reserve

    def _check_budget(self, reserves: list[dict]):
        cap = self.cfg["execution"].get("max_cost_usd")
        if cap is not None:
            if not all(r["complete"] for r in reserves):
                raise ValueError("Cannot enforce a cost cap with missing token prices")
            committed = Decimal(0)
            for e in self.store.db.execute("SELECT detail FROM events WHERE event='cost_reserved'"):
                value = json.loads(e[0])
                if value["complete"]:
                    committed += Decimal(value["total_usd"])
            if committed + sum(Decimal(r["total_usd"]) for r in reserves) > Decimal(str(cap)):
                raise ValueError("Conservative cumulative reservation reaches max_cost_usd. The journal is safe to resume.")

    def _reserve(self, record: dict, request: GenerationRequest, mode: str):
        check, reserve = self._reservation(record, request, mode)
        self._check_budget([reserve])
        self.store.event(record["id"], "context_check", check)
        self.store.event(record["id"], "cost_reserved", reserve)

    def _finish(self, record: dict, response: GenerationResponse):
        result = response.to_dict()
        result["output_flags"] = {
            "empty_answer": not bool(response.text.strip()),
            "output_limit_reached": response.finish_reason in {"length", "max_tokens", "MAX_TOKENS", "incomplete"},
            "refusal_or_safety_stop": response.finish_reason in {"content_filter", "SAFETY", "refusal"},
            "note": "Returned answers are retained and scored once; these flags do not trigger content-based retries."}
        model = self.model(record["plan"])
        result["cost"] = cost_from_usage(response.usage, model.get("pricing", {}), response.execution_mode).to_dict()
        if not price_is_current(model):
            result["cost"].update(complete=False, total_usd=None,
                                  note="Saved rate card expired before this response; reconcile against the provider invoice")
        state = "succeeded"
        if record["kind"] == "judge":
            from .judge import JudgeRequest, validate_judgement, score_judgement, JudgeValidationError
            contract = JudgeRequest(**json.loads(self.store.get_blob(record["plan"]["judge_contract_blob"])))
            try:
                judgement = validate_judgement(response.text, contract)
                result["judgement"] = judgement
                result["score"] = score_judgement(judgement, contract)
                if not result["score"]["scoreable"]:
                    state = "unscorable"
            except JudgeValidationError as exc:
                self.store.update(record["id"], "failed", result=result,
                                  error={"kind": "judge_protocol_error", "message": str(exc)})
                return
        self.store.update(record["id"], state, result=result)

    def _failure(self, record: dict, exc: ProviderError):
        current = self.store.get(record["id"])
        state = "uncertain" if exc.submission_uncertain else "failed"
        if exc.retryable and not exc.submission_uncertain and current["attempts"] < self.cfg["execution"]["max_attempts"]:
            state = "pending"
        self.store.update(record["id"], state, error=exc.to_dict())
        return state

    def step(self, kind: str = "generation", max_requests: int | None = None, *, model_ids=None, repetitions=None) -> dict:
        """Resume once: collect completed batches, then dispatch dependency-ready work.

        Native batches return immediately. Invoke again after completion for the next wave.
        Standard requests run to the requested limit or completion. No hidden background process.
        """
        with self.store.lock():
            if kind == "generation" and self.cfg.get("reused_generation_only"):
                raise ValueError("This is a judge-development fork with fixed target responses. Target generation is disabled.")
            if not self.cfg["is_mock"] and kind == "generation":
                check = preflight(self.root, check_credentials=True, model_ids=model_ids)
                if not check["ready"]:
                    raise ValueError("Preflight failed before any dispatch. Inspect preflight.json and create a corrected run.")
            if not self.cfg["is_mock"] and kind == "judge":
                active = {r["plan"]["executor_id"] for r in self.store.rows("judge")
                          if r["state"] in {"pending", "submitted", "running"} and selected_record(r, model_ids, repetitions)}
                for entry in self.cfg["judge_models"]:
                    if entry["id"] not in active:
                        continue
                    model = resolve_endpoint(entry)
                    if model.get("enabled") is False:
                        raise ValueError("A selected judge is disabled: " + model["id"])
                    validate_credentials(model)
                    if not price_is_current(model) and self.cfg["execution"].get("max_cost_usd") is not None:
                        raise ValueError("Expired judge rate card cannot enforce the cost cap")
            for record in self.store.rows(kind):
                if record["state"] == "running" and selected_record(record, model_ids, repetitions):
                    self.store.update(record["id"], "uncertain", error={"kind": "interrupted_request",
                        "message": "Provider may have completed the call. Inspect logs before explicit recovery."})
            self._collect(kind, model_ids, repetitions)
            count = 0
            while max_requests is None or count < max_requests:
                ready = self.ready(kind, model_ids, repetitions)
                if not ready:
                    break
                remaining = (max_requests - count) if max_requests is not None else len(ready)
                record = ready[0]
                mode = effective_transport(self.cfg, self.model(record["plan"]))
                if self.cfg["is_mock"]:
                    for current in ready[:remaining]:
                        self._mock(current)
                        count += 1
                elif mode == "batch":
                    key = self.model(record["plan"])["id"]
                    same = [r for r in ready if self.model(r["plan"])["id"] == key][:remaining]
                    used = self._submit(same, kind)
                    count += used
                else:
                    standard = [r for r in ready if effective_transport(self.cfg, self.model(r["plan"])) == "standard"]
                    for record in standard[:remaining]:
                        req = self.request(record)
                        self._reserve(record, req, mode)
                        self.store.update(record["id"], "running", request=self.store.pack_request(req.to_dict()), attempt=True)
                        try:
                            self._finish(record, self.provider(record["plan"]).generate(req))
                        except ProviderError as exc:
                            state = self._failure(record, exc)
                            if state == "pending":
                                time.sleep(min(60, max(exc.retry_after or 0, 2 ** self.store.get(record["id"])["attempts"])))
                        count += 1
            self.ready(kind, model_ids, repetitions)  # propagate a final failed parent without generating a placeholder
            self.store.export()
            return self.status()

    def _mock(self, record):
        req = self.request(record)
        text = "OFFLINE PIPELINE CHECK. No language model answered this question."
        if record["kind"] == "judge":
            from .judge import JudgeRequest, mock_judgement
            contract = JudgeRequest(**json.loads(self.store.get_blob(record["plan"]["judge_contract_blob"])))
            text = canonical(mock_judgement(contract))
        self.store.update(record["id"], "running", request=self.store.pack_request(req.to_dict()), attempt=True)
        response = GenerationResponse(req.request_id, text, req.model, req.model, "mock", "mock",
                                      TokenUsage(input_tokens=0, output_tokens=0), {"is_mock": True},
                                      0.0, digest(req.to_dict()), "standard")
        self._finish(record, response)

    def _submit(self, records, kind):
        provider = self.provider(records[0]["plan"])
        chosen, requests = [], []
        for record in records[:self.cfg["execution"]["batch_max_requests"]]:
            req = self.request(record)
            if provider.estimate_batch_bytes(requests + [req]) > self.cfg["execution"]["batch_max_bytes"]:
                if not requests:
                    raise ValueError("One complete request exceeds configured batch payload size")
                break
            chosen.append(record)
            requests.append(req)
        local_id = "submission-" + digest([r.request_id for r in requests] + [self.store.get(r.request_id)["attempts"] for r in requests])[:24]
        reservations = [self._reservation(record, req, "batch") for record, req in zip(chosen, requests)]
        self._check_budget([r for _, r in reservations])
        # All requests are admitted before any reservation is committed.
        from .store import now
        events = []
        for record, (check, reserve) in zip(chosen, reservations):
            events += [(now(), record["id"], "context_check", canonical(check)),
                       (now(), record["id"], "cost_reserved", canonical(reserve))]
        with self.store.db:
            self.store.db.executemany("INSERT INTO events(time,task_id,event,detail) VALUES(?,?,?,?)", events)
        self.store.batch(local_id, self.model(chosen[0]["plan"])["id"], kind, "submitting",
                         [r.request_id for r in requests])
        for record, req in zip(chosen, requests):
            self.store.update(record["id"], "running", request=self.store.pack_request(req.to_dict()), attempt=True)
        try:
            handle = provider.submit_batch(requests, label=local_id)
            self.store.batch(local_id, self.model(chosen[0]["plan"])["id"], kind, "submitted", handle.request_ids, handle.to_dict())
            for record in chosen:
                self.store.update(record["id"], "submitted")
        except ProviderError as exc:
            self.store.batch(local_id, self.model(chosen[0]["plan"])["id"], kind,
                             "uncertain" if exc.submission_uncertain else "failed", [r.request_id for r in requests])
            for record in chosen:
                self._failure(record, exc)
        return len(chosen)

    def _collect(self, kind, model_ids=None, repetitions=None):
        if self.cfg["is_mock"]:
            return
        for batch in self.store.batches(kind):
            if batch["state"] not in {"submitted", "collecting"}:
                continue
            records = [self.store.get(i) for i in batch["task_ids"]]
            if not any(selected_record(r, model_ids, repetitions) for r in records):
                continue
            provider = self.provider(records[0]["plan"])
            handle = BatchHandle.from_dict(batch["handle"])
            status = provider.poll_batch(handle)
            self.store.event(None, "batch_poll", status.to_dict())
            if not status.terminal:
                continue
            self.store.batch(batch["id"], batch["model_id"], kind, "collecting", batch["task_ids"], handle.to_dict())
            output = provider.collect_batch(handle, {r["id"]: self.request(r) for r in records})
            for item in output:
                record = self.store.get(item["request_id"])
                if record["state"] in {"succeeded", "unscorable", "failed"}:
                    continue
                if item["status"] == "succeeded":
                    self._finish(record, item["response"])
                else:
                    self.store.update(record["id"], "failed", error=item["error"])
            self.store.batch(batch["id"], batch["model_id"], kind, "collected", batch["task_ids"], handle.to_dict())

    def recover(self, ids: list[str], reason: str):
        if not reason.strip():
            raise ValueError("A recovery reason is required for the publication audit trail")
        with self.store.lock():
            for task_id in ids:
                row = self.store.get(task_id)
                if row and row["kind"] == "generation" and self.cfg.get("reused_generation_only"):
                    raise ValueError("A judge-development fork cannot regenerate its fixed target answers")
                if not row or row["state"] not in {"uncertain", "failed", "blocked"}:
                    raise ValueError("Only uncertain, failed or blocked requests can be explicitly recovered")
                if row["result"] and row["result"].get("text"):
                    raise ValueError("A returned answer cannot be regenerated selectively. Review the retained response.")
                self.store.event(task_id, "manual_recovery", {"reason": reason, "previous_error": row["error"],
                    "warning": "An uncertain prior attempt may have incurred charges."})
                self.store.update(task_id, "pending")

    def adopt_batch(self, submission_id: str, provider_batch_id: str, reason: str):
        """Attach a provider-confirmed job after an interrupted submission, without resending it."""
        if not reason.strip() or not provider_batch_id.strip():
            raise ValueError("A provider job ID and recovery reason are required")
        with self.store.lock():
            batches = self.store.batches("generation") + self.store.batches("judge")
            batch = next((b for b in batches if b["id"] == submission_id), None)
            if not batch or batch["state"] not in {"submitting", "uncertain"}:
                raise ValueError("No unresolved local submission with that ID")
            records = [self.store.get(i) for i in batch["task_ids"]]
            if any(r["result"] for r in records):
                raise ValueError("Cannot adopt a job over existing completed responses")
            provider = self.provider(records[0]["plan"])
            requests = [self.request(r) for r in records]
            custom, hashes = provider._batch_maps(requests)
            handle = BatchHandle(provider.kind, provider_batch_id, batch["task_ids"], custom, hashes)
            # This read verifies that the authenticated account can see the claimed job.
            provider.poll_batch(handle)
            self.store.event(None, "manual_batch_adoption", {"submission_id": submission_id,
                "provider_batch_id": provider_batch_id, "reason": reason,
                "note": "Result custom IDs and request payload hashes are checked again on collection."})
            self.store.batch(submission_id, batch["model_id"], batch["kind"], "submitted", batch["task_ids"], handle.to_dict())
            for record in records:
                self.store.update(record["id"], "submitted")

    def status(self):
        result = {kind: dict(Counter(r["state"] for r in self.store.rows(kind))) for kind in ("generation", "judge")}
        result["batches"] = [{"local_submission_id": b["id"], "state": b["state"],
                              "provider_batch_id": (b["handle"] or {}).get("batch_id")}
                             for kind in ("generation", "judge") for b in self.store.batches(kind)]
        return result
