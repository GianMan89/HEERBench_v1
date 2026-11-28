from __future__ import annotations

import json
from .providers import GenerationRequest
from .util import canonical, digest


def build_generation_request(row: dict, cfg: dict, dataset, parent_answers: dict[str, str]) -> GenerationRequest:
    model = next(m for m in cfg["models"] if m["id"] == row["model_id"])
    q = dataset.by_id[row["question_id"]]
    preceding = []
    if row["condition"] != "E1":
        for p in row["parent_question_ids"]:
            text = dataset.by_id[p]["expected_answer"] if row["condition"] == "E2" else parent_answers[p]
            preceding.append({"question_id": p, "question": dataset.by_id[p]["question"], "answer": text})
    # Identical labels for oracle and generated answers avoid an authority-label confound.
    task = {"current_question": q["question"], "question_id": q["question_id"],
            "preceding_analytical_outputs": preceding}
    return GenerationRequest(
        request_id=row["request_id"], model=model["model"], system=cfg["generator_system"],
        prefix="<scenario_documents>\n" + dataset.packets[q["case_id"]] + "\n</scenario_documents>",
        task=canonical(task), max_output_tokens=model["max_output_tokens"],
        temperature=model.get("temperature"), top_p=model.get("top_p"),
        # Provider seed is optional. The selection seed is not silently treated as a decoding seed.
        seed=model.get("generation_seed"), parameters={})

