from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from .util import digest, read_jsonl


class Dataset:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.questions = read_jsonl(self.root / "reference/questions.jsonl")
        self.chains = read_jsonl(self.root / "reference/chains.jsonl")
        self.dependencies = read_jsonl(self.root / "reference/dependencies.jsonl")
        self.sources = read_jsonl(self.root / "reference/sources.jsonl")
        self.errors = read_jsonl(self.root / "reference/error_checks.jsonl")
        self.local_errors = read_jsonl(self.root / "reference/local_error_conditions.jsonl")
        self.by_id = {q["question_id"]: q for q in self.questions}
        self.by_chain = {c["chain_id"]: c for c in self.chains}
        self.parents = {}
        for d in self.dependencies:
            child, parents = d["child_id"], tuple(sorted(d["parent_ids"]))
            if child in self.parents and self.parents[child] != parents:
                raise ValueError(f"Alternative parent sets require separate contexts, unsupported: {child}")
            self.parents[child] = parents
        self.packet_paths = {c: self.root / "model_input/packets" / (c + ".md")
                             for c in {q["case_id"] for q in self.questions}}
        self.packets = {c: p.read_text(encoding="utf-8") for c, p in self.packet_paths.items()}
        self.acceptance = json.loads((self.root / "acceptance.json").read_text())

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256((self.root / "MANIFEST.sha256").read_bytes()).hexdigest()

    def for_question(self, records, question_id):
        return [r for r in records if r["question_id"] == question_id]

    def chain_nodes(self, chain_id: str) -> set[str]:
        c = self.by_chain[chain_id]
        return set(c["a_ids"] + c["b_ids"] + c["f_ids"])

    def validate(self) -> dict:
        mismatches = []
        for line in (self.root / "MANIFEST.sha256").read_text().splitlines():
            expected, name = line.split("  ", 1)
            path = (self.root / name).resolve()
            if not path.is_relative_to(self.root):
                raise ValueError("Unsafe manifest path")
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                mismatches.append(name)
        if mismatches:
            raise ValueError("Dataset integrity check failed: " + ", ".join(mismatches))
        if len(self.by_id) != len(self.questions) or len(self.by_chain) != len(self.chains):
            raise ValueError("Duplicate dataset identifiers")
        ranks = {"A": 0, "B": 1, "F": 2}
        membership = defaultdict(set)
        for chain in self.chains:
            nodes = self.chain_nodes(chain["chain_id"])
            for stage, key in (("A", "a_ids"), ("B", "b_ids"), ("F", "f_ids")):
                if not chain[key]:
                    raise ValueError("Incomplete chain " + chain["chain_id"])
                for qid in chain[key]:
                    q = self.by_id[qid]
                    if q["case_id"] != chain["case_id"] or q["abf_stage"] != stage:
                        raise ValueError("Chain stage/scenario mismatch")
                    membership[qid].add(chain["chain_id"])
                    if not set(self.parents.get(qid, ())).issubset(nodes):
                        raise ValueError("A chain omits a jointly required parent")
        for q in self.questions:
            qid = q["question_id"]
            if qid not in membership or not q["criteria"] or not q["source_refs"]:
                raise ValueError("Question has no complete chain, criteria or sources: " + qid)
            if q["abf_stage"] != "A" and qid not in self.parents:
                raise ValueError("Missing parent set: " + qid)
            for parent in self.parents.get(qid, ()):
                p = self.by_id[parent]
                if p["case_id"] != q["case_id"] or ranks[p["abf_stage"]] >= ranks[q["abf_stage"]]:
                    raise ValueError("Dependency is cyclic or crosses a scenario")
        return {"valid": True, "questions": len(self.questions), "chains": len(self.chains),
                "dependency_sets": len(self.parents),
                "stages": dict(Counter(q["abf_stage"] for q in self.questions)),
                "scenarios": len(self.packets), "fingerprint": self.fingerprint,
                "acceptance": self.acceptance}

    def select_chains(self, mode: str, per_scenario: int = 2, seed: int = 20261127,
                      explicit: list[str] | None = None) -> list[str]:
        if explicit is not None:
            if not explicit or len(explicit) != len(set(explicit)):
                raise ValueError("Explicit chain IDs must be nonempty and unique")
            if set(explicit) - set(self.by_chain):
                raise ValueError("Unknown explicit chain ID")
            return sorted(explicit)
        if mode == "full":
            return sorted(self.by_chain)
        if mode not in {"pilot", "demo"} or per_scenario < 1:
            raise ValueError("Invalid chain selection")
        by_case = defaultdict(list)
        for c in self.chains:
            by_case[c["case_id"]].append(c["chain_id"])
        selected = []
        for case, candidates in sorted(by_case.items()):
            # Seeded hash sampling is reproducible across Python versions.
            selected += sorted(candidates, key=lambda x: digest([seed, case, x]))[:per_scenario]
        return sorted(selected)

