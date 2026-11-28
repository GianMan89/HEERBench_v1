"""Export auditable experiment analysis as JSON, CSV, Markdown and optional SVG."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def _csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    if not records:
        path.write_text("", encoding="utf-8")
        return
    keys = list(dict.fromkeys(k for row in records for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in records:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def _text(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _pct(value: Any) -> str:
    return "—" if value is None else f"{float(value) * 100:.1f}%"


def _pp(value: Any) -> str:
    return "—" if value is None else f"{float(value) * 100:+.2f}"


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    return "| " + " | ".join(headers) + " |\n| " + " | ".join("---" for _ in headers) + " |\n" + "\n".join("| " + " | ".join(_text(v) for v in row) + " |" for row in rows) + "\n"


def _human_reference_description(basis: Any) -> str:
    """Describe recorded review metadata without inferring missing expert counts."""
    if not isinstance(basis, Mapping) or not basis:
        return "Human-reference design not recorded; no reviewer count inferred"
    if basis.get("reviewers") == 1 and basis.get("reference_type") == "single_expert":
        return "One military expert, single-expert reference; no inter-human agreement or adjudicated consensus"
    if basis.get("reviewers") == 2 and basis.get("reference_type") == "two_expert_adjudicated":
        return "Two experts, adjudicated reference; independence from corpus construction is not established by this metadata"
    return "Recorded human-reference metadata require inspection; no reviewer count inferred"


def markdown_report(report: Mapping[str, Any]) -> str:
    meta = report["metadata"]
    heading = "MOCK — pipeline validation only" if meta["is_mock"] else "HEERBench experiment analysis"
    out = [f"# {heading}\n"]
    if meta["is_mock"]:
        out += ["**Synthetic responses and scores test the software. They are not measurements of any real model and must not be reported as experimental findings.**\n"]
    elif report.get("judge_validated_for_publication", meta.get("judge_validated_for_publication")) is False:
        out += ["**JUDGE CALIBRATION NOT YET ACCEPTED. These are provisional pilot or diagnostic scores. At least one contributing run has no recorded acceptance of a held-out expert assessment for its frozen judge. Do not present the scores as having accepted judge calibration.**\n"]
    elif report.get("judge_validated_for_publication", meta.get("judge_validated_for_publication")) is True:
        out += ["The frozen judge configuration has a recorded acceptance of its held-out calibration assessment. Consult the original records for achieved agreement, failure detection and scope. Accepted calibration records the expert decision and does not certify scientific adequacy. This status does not assert a human performance baseline.\n"]
    basis = report.get("judge_validation_basis", meta.get("judge_validation_basis"))
    if not meta["is_mock"] and isinstance(basis, list):
        out += ["## Judge calibration basis\n",
                "Each source retains its recorded human reference. Reviewer counts are not added across runs and the same expert may have assessed several runs.\n",
                _table(["Source run", "Calibration acceptance", "Human reference"], [
                    [row.get("source_directory", row.get("run_id", "Unspecified")),
                     "Recorded" if row.get("calibration_accepted") else "Not recorded",
                     _human_reference_description(row.get("human_review"))]
                    for row in basis])]
    elif not meta["is_mock"] and (basis is not None or
            report.get("judge_validated_for_publication", meta.get("judge_validated_for_publication")) is True):
        out += ["**Human reference.** " + _human_reference_description(basis) + ".\n"]
    out += [
        f"The frozen plan contains **{meta['planned_generation_requests']:,} generation requests** across {len(meta['models'])} model configurations and {len(meta['scenarios'])} scenarios. The generation count excludes judge calls and does not duplicate shared A responses.\n",
        "All three conditions provide the full unchanged scenario packet. E1 supplies no preceding answers, E2 supplies the accepted reference answers for the defined parents, and E3 supplies the same model's generated parent answers from the same repetition. A is generated once and shared. The table entries below are percentages, while condition differences are percentage points.\n",
        "## Coverage\n",
        "Execution failures, blocked descendants and unresolved judging remain in the planned denominators. The observed score uses fully scoreable responses. The lower-bound score assigns zero to every unscored planned response, combining content performance and availability. Neither view should be reported alone when coverage is incomplete.\n",
    ]
    out.append(_table(["Model", "Condition", "Planned", "Generated", "Scored", "Scoring coverage", "Generation failed", "Blocked", "Judge failed", "Judge unscorable"], [
        [r["model_id"], r["condition"], r["planned"], r["generated"], r["scored"], _pct(r["scoring_coverage"]), r["generation_failed"], r["generation_blocked"], r["judge_failed"], r["judge_unscorable"]]
        for r in report["coverage"]]))
    out += ["\n## Scenario-macro question performance\n",
            "Question criteria are averaged within each stage, scenario and repetition. Scenarios and repetitions then receive equal weight. BF is an equal-weight average of B and F, excluding A, so the condition comparison has the same target. SD is the standard deviation across complete repetition-level scenario means, not a confidence interval.\n"]
    out.append(_table(["Model", "Condition", "Stage", "Observed criterion score", "Lower-bound criterion score", "Run SD", "Fully correct, lower bound", "Any material error, observed"], [
        [r["model_id"], r["condition"], r["stage"], _pct(r["criterion_score_observed"]), _pct(r["criterion_score_lower_bound"]), _pct(r["criterion_score_lower_bound_run_sd"]), _pct(r["fully_correct_lower_bound"]), _pct(r["any_error_observed"])]
        for r in report["stage_scores"]]))
    out += ["\n## Paired condition differences\n",
            "The table shows the availability-adjusted difference of lower-bound criterion scores for BF in percentage points. This difference is not itself a bound on the unknown complete-data effect. The full CSV additionally reports B and F, complete-case criterion differences and fully-correct differences. Positive oracle-minus-sequential values favour expert preceding answers. The contrast changes the complete preceding answer, including phrasing and completeness, and does not identify an error-only causal effect.\n"]
    rows = [r for r in report["paired_effects"] if r["stage"] == "BF" and r["metric"] == "criterion_score_lower_bound"]
    out.append(_table(["Model", "Contrast", "Difference, pp", "95% bootstrap interval, pp", "Paired responses", "Scenarios"], [
        [r["model_id"], r["contrast"], _pp(r["estimate"]), "—" if r["ci_low"] is None else f"[{_pp(r['ci_low'])}, {_pp(r['ci_high'])}]", r["used_pairs"], r["n_scenarios"]] for r in rows]))
    out += ["\n## Complete chains and transitions\n",
            "`chain_scores.csv` contains one matched record per selected chain, model, repetition and condition. Only E3 is a generated end-to-end chain. E1 records joint success of independently elicited tasks. E2 records joint task success with expert-parent assistance, even when the model's own preceding response was wrong. Shared nodes are reused rather than counted as extra generations.\n",
            "`transitions.csv` records whether all required parents and the child are correct. Incorrect-to-incorrect is labelled an association. It is not automatically attributed to propagation of the same error. Unscorable parent or child sets are retained separately.\n",
            "## Statistical interpretation\n",
            f"Paired intervals use {meta['bootstrap_resamples']:,} scenario-cluster bootstrap draws with seed {meta['bootstrap_seed']}. A scenario contributes its complete graph, and repetition blocks are resampled jointly for the compared conditions. Questions and chain memberships are not resampled as independent units.\n",
    ]
    out += ["\n".join("- " + w for w in meta["warnings"]) + "\n"]
    out += ["\n## Files\n",
            "The JSON report retains the complete metadata and all tables. CSV files expose question scores, execution coverage, scenario/repetition means, stage summaries, paired contrasts, chain outcomes and transitions. No original MA-Bench severity weighting is applied.\n"]
    return "\n".join(out)


def write_report(report: Mapping[str, Any], output_dir: str | Path, *, plots: bool = False) -> dict[str, str]:
    """Write all analysis tables. ``plots=True`` requires the plots extra."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    p = out / "analysis.json"
    p.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    files["analysis_json"] = str(p)
    for name, rows in report.items():
        if name != "metadata" and isinstance(rows, list):
            p = out / f"{name}.csv"
            _csv(p, rows)
            files[name] = str(p)
    p = out / "report.md"
    p.write_text(markdown_report(report), encoding="utf-8")
    files["report_markdown"] = str(p)
    if plots:
        files.update(plot_report(report, out))
    return files


def plot_report(report: Mapping[str, Any], output_dir: str | Path) -> dict[str, str]:
    """Generate publication-exportable SVGs without inventing absent intervals."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Plotting requires matplotlib. Install the repository's plots extra.") from exc
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = report["metadata"]
    title_prefix = "MOCK PIPELINE TEST — " if meta["is_mock"] else ""
    if not meta["is_mock"] and report.get("judge_validated_for_publication", meta.get("judge_validated_for_publication")) is False:
        title_prefix = "PROVISIONAL / UNVALIDATED JUDGE — "
    models = meta["models"]
    colours = {"E1": "#446E97", "E2": "#38856C", "E3": "#B37734"}
    labels = {"E1": "Independent", "E2": "Expert parents", "E3": "Generated parents"}
    lookup = {(r["model_id"], r["condition"], r["stage"]): r for r in report["stage_scores"]}
    fig, axes = plt.subplots(1, 3, figsize=(max(12, len(models) * 1.8), 4.8), sharey=True)
    for ax, stage in zip(axes, ("A", "B", "F")):
        conditions = ("E1",) if stage == "A" else ("E1", "E2", "E3")
        width = 0.8 / len(conditions)
        for k, condition in enumerate(conditions):
            x, y, sd = [], [], []
            for i, model in enumerate(models):
                r = lookup.get((model, condition, stage))
                if r is None or r["criterion_score_lower_bound"] is None:
                    continue
                x.append(i - 0.4 + width * (k + 0.5))
                y.append(r["criterion_score_lower_bound"] * 100)
                sd.append((r["criterion_score_lower_bound_run_sd"] or 0) * 100)
            ax.bar(x, y, width=width * 0.94, yerr=sd, capsize=2, color=colours[condition], label=labels[condition])
        ax.set_title(stage)
        ax.set_xticks(range(len(models)), models, rotation=35, ha="right")
        ax.set_ylim(0, 105)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#D9DFE5", linewidth=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Scenario-macro criterion score, lower bound (%)")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle(title_prefix + "ABF task performance")
    fig.text(0.5, 0.015, "Bars use planned denominators. Error bars are ±1 run SD where repeated runs exist. Inspect coverage alongside performance.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    p = out / "stage_performance.svg"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)

    contrasts = ("oracle_minus_sequential", "sequential_minus_independent")
    rows = [r for r in report["paired_effects"] if r["stage"] == "BF" and r["metric"] == "criterion_score_lower_bound" and r["contrast"] in contrasts]
    rows = sorted(rows, key=lambda r: (r["contrast"], r["model_id"]))
    fig, ax = plt.subplots(figsize=(8.5, max(3, len(rows) * 0.4)))
    for i, row in enumerate(rows):
        if row["estimate"] is None:
            continue
        xx = row["estimate"] * 100
        if row["ci_low"] is not None:
            ax.plot([row["ci_low"] * 100, row["ci_high"] * 100], [i, i], color="#446E97", lw=1.5)
        ax.plot(xx, i, "o", color="#244A70")
    ax.set_yticks(range(len(rows)), [f"{r['model_id']}  {r['left_condition']} − {r['right_condition']}" for r in rows])
    ax.invert_yaxis()
    ax.axvline(0, color="#777777", ls="--", lw=0.8)
    ax.set_xlabel("Paired BF criterion-score difference (percentage points)")
    ax.set_title(title_prefix + "Condition contrasts with descriptive 95% bootstrap intervals")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p2 = out / "paired_conditions.svg"
    fig.savefig(p2, bbox_inches="tight")
    plt.close(fig)
    return {"stage_plot": str(p), "paired_plot": str(p2)}
