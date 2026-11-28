---
license: cc-by-4.0
task_categories: [text-generation]
language: [en]
tags: [military-planning, mission-analysis, joint-planning-process, rubric-evaluation, long-context]
pretty_name: MA-Bench
size_categories: [1K<n<10K]
configs:
  - config_name: cases
    data_files: data/cases.jsonl
  - config_name: documents
    data_files: data/documents.jsonl
  - config_name: rubric_items
    data_files: data/rubric_items.jsonl
  - config_name: briefs
    data_files: runs/briefs.jsonl
  - config_name: verdicts
    data_files: runs/verdicts.jsonl
  - config_name: expert_study
    data_files: runs/expert_study.jsonl
  - config_name: expert_study_briefs
    data_files: runs/expert_study_briefs.jsonl
---

# MA-Bench: Measuring Operational Planning in Language Models

Paper: [arXiv:2610.19138](https://arxiv.org/abs/2610.19138)

If you use MA-Bench, please cite:

```bibtex
@misc{lafuentemercado2026mabench,
  title         = {{MA-Bench}: Measuring Operational Planning in Language Models},
  author        = {Lafuente-Mercado, Rodney and Sincero, Robert and Sage, Phil and Xue, Yuan},
  year          = {2026},
  eprint        = {2610.19138},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi           = {10.48550/arXiv.2610.19138},
  url           = {https://arxiv.org/abs/2610.19138}
}
```

MA-Bench evaluates one step of the Joint Planning Process (JP 5-0): mission analysis. Each of the 10 scenarios pairs a
full synthetic staff packet (10 to 12 documents,
35,649 to 242,658 characters) with an expert-authored rubric of
74 to 170 atomic, binary items (1,257 items in total).
All scenarios are original and unpublished; no packet or rubric text exists elsewhere. This is version 1.0.0, with rubrics frozen on 2026-09-14.

## Files

| file | rows | one row per |
|---|---:|---|
| `data/cases.jsonl` | 10 | scenario: `case_id`, `operation`, `scenario`, the three judge header fields (`planning_level`, `document_type`, `difficulty`), counts |
| `data/documents.jsonl` | 112 | model-facing document: `case_id`, `doc_number`, `doc_order`, `title`, `source_file`, `text` |
| `data/rubric_items.jsonl` | 1,257 | rubric item: `item_id`, `item_uid`, `step`, `step_name`, `tier`, `severity`, `polarity`, `weight`, `question`, `pass_condition`, `fail_condition`, `source_fact`, `artifact_anchor`, `source_documents` |
| `runs/briefs.jsonl` | 1,560 | generated brief: `brief_id`, `condition`, `case_id`, `scenario`, `model`, `model_slug`, `seed`, `text` (as graded) |
| `runs/verdicts.jsonl` | 2,796 | judge verdict on a brief: `brief_id`, `judge`, `case_id`, `failed_items`, `failures` (each with `evidence` and `rationale`), `unknown_ids`, `overall_rationale`, `judge_model`, `served_model` |
| `runs/expert_study.jsonl` | 190 | expert item verdict (see Expert study) |
| `runs/expert_study_briefs.jsonl` | 10 | brief the experts graded: `brief`, `scenario`, `model`, `text` |

- `packets/<case>_<operation>/<document>.md`: each scenario's documents, one file per document, exactly as models
  receive them (markdown converted from the authored Word or markdown files; images omitted).
- `rubrics/<case>_<operation>.csv`: each scenario's rubric, one row per item: step, tier, severity and weight, the
  question with its pass and fail conditions, and the source fact and documents it rests on. UTF-8; opens in Excel.
- `scoring/`: the task prompt, the frozen judge prompt and its input builder (`judge_input.py`), the severity weights,
  and the reference scorer (`score.py`).
- `MANIFEST.json`: the sha256 of every file.

## Scenarios

| case | code | operation | documents | packet characters | rubric items |
|---|---|---|---:|---:|---:|
| C1 | `SF-FULL` | Silent Fathom | 12 | 134,337 | 87 |
| C2 | `JA-FULL` | Jagged Anvil | 11 | 70,196 | 147 |
| C3 | `CT-FULL` | Coral Tempest | 11 | 35,649 | 130 |
| C4 | `ST-FULL` | Sea Tiger | 12 | 181,678 | 170 |
| C5 | `GG-FULL` | Gulf Guardian | 12 | 80,341 | 165 |
| C6 | `PH-FULL` | Pacific Harrier | 10 | 229,187 | 125 |
| C7 | `SS-FULL` | Southern Shield | 12 | 242,658 | 163 |
| C8 | `SM-FULL` | Steadfast Mariner | 11 | 156,729 | 111 |
| C9 | `RH-FULL` | Resolute Herald | 11 | 151,265 | 85 |
| C10 | `OR-FULL` | Onyx Rain | 10 | 93,897 | 74 |

The code is how brief ids name each scenario.

## Model input

A model receives the task prompt (`scoring/task_prompt.md`, with the operation name filled in and the length-cap
paragraph appended), a blank line, `# SOURCE DOCUMENTS`, and then the packet: the scenario's documents in
`doc_order`, assembled exactly as

```python
packet = f"# {operation} — Mission Analysis Source Packet\n" + "".join(
    f"\n\n## Document {doc_number} — {title} (source file: {source_file})\n\n{text}\n" for each document)
```

## Rubric

Each item is an atomic, binary question about the brief, with its pass and fail conditions and the source fact and
documents it rests on. Every item has a mission analysis step (S1 to S12, named in `step_name`), a tier and a severity.

- Tier: 1 = stated in one document (400 items), 2 = synthesis across facts (406), 3 = planning judgment (451).
- Severity: CI, critically important (484); I, important (536); SI, slightly important (80);
  D, detrimental (82); CD, critically detrimental (75). D and CD items name a wrong move, and their
  pass condition is the presence of that move.

## Judging and scoring

The system under test receives the task prompt (`scoring/task_prompt.md`, operation name filled in, cap paragraph appended) followed by the packet (see `documents.jsonl`) and returns one brief, truncated to 3,000 words before grading. The cap is part of the protocol: it equalizes the length budget across models. A reasoning-model judge grades the brief
against every rubric item in one call, returning only failed items with evidence; the judge never sees the packet. Items are binary and
atomic; absence is failure. Negative items (severity D, CD) describe a mistake and count against the brief when the mistake is present.

Severity weights: CI +9, I +5, SI +2, D -5, CD -9. Weighted score = (positive weight earned + negative weight incurred) / total positive weight, not clamped.

## Generated briefs

`runs/briefs.jsonl` holds 1,560 briefs, each exactly as graded (in capped conditions, text beyond the 3,000th word is
already removed). `brief_id` is `<condition>/<scenario code>__<model_slug>__s<seed>`.

| condition | briefs | settings | models |
|---|---:|---|---|
| `benchmark_capped` | 540 | 3,000-word cap, medium reasoning effort where the provider exposes it | Claude Fable 5.1, Claude Opus 5, DeepSeek-V4-Pro, GLM-5.3, GPT-5.6 Luna, GPT-5.6 Sol, GPT-6 Astra, Gemini 3.1 Pro, Gemma 4 31B, Grok 4.6, Kimi K3, MiniMax M3, Mistral Medium 3.5, Muse Spark 1.3, Nemotron 3 Ultra, Qwen3.8-Max |
| `ladder_capped` | 120 | the same settings; the capability ladder | Claude Sonnet 5, DeepSeek-V4-Flash, GPT-5.4, Gemini 3.8 Flash |
| `effort_low` | 150 | 3,000-word cap, low reasoning effort | Claude Sonnet 5, GPT-5.4, GPT-5.6 Sol, GPT-6 Astra, Gemini 3.8 Flash |
| `effort_high` | 150 | 3,000-word cap, high reasoning effort | Claude Sonnet 5, GPT-5.4, GPT-5.6 Sol, GPT-6 Astra, Gemini 3.8 Flash |
| `uncapped` | 600 | no length cap, medium reasoning effort | Claude Fable 5.1, Claude Opus 5, Claude Sonnet 5, DeepSeek-V4-Flash, DeepSeek-V4-Pro, GLM-5.3, GPT-5.4, GPT-5.6 Luna, GPT-5.6 Sol, GPT-6 Astra, Gemini 3.1 Pro, Gemini 3.8 Flash, Gemma 4 31B, Grok 4.6, Kimi K3, MiniMax M3, Mistral Medium 3.5, Muse Spark 1.3, Nemotron 3 Ultra, Qwen3.8-Max |

### Models

| slug | model |
|---|---|
| `anthropic_claude_fable_5_1` | Claude Fable 5.1 |
| `anthropic_claude_opus_5` | Claude Opus 5 |
| `anthropic_claude_sonnet_5` | Claude Sonnet 5 |
| `azure_ai_deepseek_v4_flash` | DeepSeek-V4-Flash |
| `azure_ai_deepseek_v4_pro` | DeepSeek-V4-Pro |
| `gemini_gemini_3_1_pro_preview` | Gemini 3.1 Pro |
| `gemini_gemini_3_8_flash` | Gemini 3.8 Flash |
| `google_gemma_4_31b_it` | Gemma 4 31B |
| `fireworks_ai_glm_5p3` | GLM-5.3 |
| `openai_gpt_5_4` | GPT-5.4 |
| `openai_gpt_5_6_luna` | GPT-5.6 Luna |
| `openai_gpt_5_6_sol` | GPT-5.6 Sol |
| `openai_gpt_6_astra` | GPT-6 Astra |
| `xai_grok_4_6` | Grok 4.6 |
| `fireworks_ai_kimi_k3` | Kimi K3 |
| `fireworks_ai_minimax_m3` | MiniMax M3 |
| `mistral_mistral_medium_3_5` | Mistral Medium 3.5 |
| `meta_ai_muse_spark_1_3` | Muse Spark 1.3 |
| `fireworks_ai_nemotron_3_ultra_nvfp4` | Nemotron 3 Ultra |
| `fireworks_ai_qwen3p8_max` | Qwen3.8-Max |

## Verdicts

`runs/verdicts.jsonl` holds each judge's verdict on a brief. All judges used the same frozen judge prompt (`scoring/judge_prompt.md`):

- `gpt-5.6`, at high reasoning effort: every brief (1,560).
- `gemini-3.1-pro`: every brief outside `uncapped`, and the `uncapped` briefs of GPT-6 Astra and GPT-5.4
  (1,020).
- `claude-opus-5`: 216 `benchmark_capped` briefs (scenarios C1, C2, C3, C5, C6 and C8, ten models,
  every seed).

A verdict lists only the items the brief fails (`failures`, each with quoted evidence and a rationale); every rubric
item not listed passes. A positive item (severity CI, I, SI) earns its weight when it passes. For a negative item (D,
CD) the pass condition is the presence of the wrong move it names, so a listed negative item is a mistake the brief
avoided and an unlisted one is a mistake it made, charged its (negative) weight. `scoring/score.py` turns a verdict
into a brief's weighted score. A model's score is the mean over scenarios of its mean brief score. `python scoring/score.py data/rubric_items.jsonl <verdicts>` scores a JSONL of verdicts for one judge. In
some verdicts, `condition`, `brief_id` and `model_slug` were renamed to match this layout; the judge's output is
unchanged.

## Expert study

Four JPME-credentialed planners, de-identified as `expert_1` to `expert_4`, independently graded the same 190 item
verdicts: nineteen rubric items in each of ten briefs, sampled across every tier and severity. They saw each brief
and rubric item as the judge does, without the packet and without the model's name. Each row gives the brief (`B01`
to `B10`), its scenario code and model, the rubric item as the experts saw it (id, step, tier, severity, question,
pass and fail conditions), each expert's verdict, the expert majority (`Pass`, `Fail`, or `tie` for a two-two split),
and the production judge's verdict on the same item (gpt-5.6 at high reasoning effort, with the judge prompt that was
then frozen) with its note.

The briefs were sampled before the packets were frozen on 2026-09-08, so 53 of the 190 items carry rubric text that
later changed; the text given is what the experts graded, and the judge graded the same text. `runs/expert_study.jsonl` has one row per item verdict and `runs/expert_study_briefs.jsonl` the ten briefs.

## Provenance

Rubrics frozen 2026-09-14 from the ops team's expert-reviewed final sheet (1,156 delivered, 100 gap-fill items and 1 item restored after the expert grading study; 3 earlier items withdrawn). Relative to the 2026-09-08 freeze, 5 items had their question or pass/fail text revised, 7 gained source-document references, and 1 item was restored; the document sets are unchanged. Rubrics were authored against the packet plus downstream products of a completed mission analysis (component orders, CCIRs, event
templates, and for some scenarios an OSINT bundle) that the model never receives; those documents are not part of this release.
