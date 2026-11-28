# Judge calibration across separately executed pilot models

`prepare_study_calibration` creates a separate calibration run from compatible,
completed generation segments. This supports the practical case in which each
model was run separately and a calibration panel should draw on the combined
model pool. The combined pool contains the original answers
from all selected segments. No new target answers are generated.

## Workflow

1. Complete the selected pilot segments. Every planned generation must have a
   terminal outcome. A request may have succeeded, failed, been blocked by a
   failed dependency, or been recorded as unscorable. Resolve unfinished or
   ambiguous requests in their original run before pooling.
2. Pool the original generation runs into a new calibration directory.
3. Select one development and held-out panel from the combined pool. With the
   default settings, the panel contains 10 development answers and 50 held-out
   answers. The selection must have enough completed answers and independent
   grouping units. The code does not silently reduce the requested panel size.
4. The single military expert labels `reviewer_1.csv` before seeing automated
   verdicts, using the blinded answer packet. Tune judge instructions from the
   development comparison only. Keep held-out comparison results separate until
   the judge configuration is frozen. No second reviewer or adjudication file
   is required in the default mode.
5. Run one judge call per selected fixed answer by default. Optional repeated
   scoring calls are not additional target-model repetitions. Record the
   expert acceptance decision against the held-out report and exact judge
   configuration before using it for publication scoring.

The Python API is available independently of the command-line study workflow.

```python
from pathlib import Path

from heerbench.evaluation import prepare_validation
from heerbench.runner import Runner
from heerbench.study_calibration import prepare_study_calibration

result = prepare_study_calibration(
    [Path("runs/pilot_model_one"), Path("runs/pilot_model_two")],
    Path("runs/pilot_judge_calibration"),
)
runner = Runner(result["run_path"])
try:
    prepare_validation(
        runner,
        Path("calibration/panel"),
        n=60,
        development_n=10,
        reviewers=1,
        group_by="component",
    )
finally:
    runner.close()
```

Creating the bundle and exporting the expert panel are offline operations.
Later judge calls use their own API credentials and incur their own charges.

## Compatibility and preservation rules

The source configuration, frozen plan and corpus fingerprint are checked before
any output is written. All sources must have the same scientific protocol,
including scenario/task selection, generator instructions, conditions,
generation settings, judge instructions and judge policy. Selecting different
target models does not change the protocol key.

For a target model appearing in more than one segment, deployment and inference
settings must match. Conflicting precision, provider routing, output budgets or
returned model versions are rejected. Repetition identifiers remain global and
are never renumbered. For example, model repetitions 2 and 6 remain 2 and 6.

A repeated logical observation is rejected even if it has a different physical
request ID. The logical unit is the combination of target model, repetition,
question and condition. Supplying the same run twice also fails. The method
never chooses between duplicate answers using their quality or judge score.

The complete frozen generation plans are combined. Failed, blocked and
unscorable rows remain in both the plan and journal with their original states.
Missing or unfinished rows cause an error rather than being dropped, converted
to synthetic failures, or replaced with invented answers. Only successful
responses are eligible for an answer-scoring panel. The full source completion
counts remain available in the provenance ledger.

Sequential dependencies keep their original request IDs. The pool verifies
that each parent belongs to the same model and repetition and has the expected
condition. Shared questions across models remain separate observations, while
development/held-out grouping keeps all responses from a shared question or
connected chain component on the same side of the split. Component grouping
still allows different components from the same scenario in both splits. Use
scenario grouping when the validation target is transfer to unseen scenario
packets.

## Contents of the calibration run

The bundle includes the combined frozen configuration and request plan, a new
manifest, a generation-only journal, and all content-addressed blobs referenced
by the copied generation records. Successful responses, raw provider outputs,
exact logged request content, attempts, errors and original timestamps are
preserved. Blob hashes are checked before and after copying.

The manifest freezes the current software used to operate the calibration run.
It separately records each source run's original software hash. This does not
relabel old target generations as having been produced by the current code.
Inspect differing source revisions when assessing experimental comparability.

The following ledgers document reuse.

| File | Contents |
|---|---|
| `source_segment_ledger.json` | Source paths and run IDs, original configuration/plan/software/corpus hashes, generation counts and states |
| `generation_reuse_ledger.jsonl` | Original response IDs and repetitions, source-record/request/response hashes, historical usage and costs |
| `generation_reuse.json` | Combined reuse summary, copied record/blob counts and zero new target-generation cost |

Original source directories are opened for reading and remain unchanged. The
bundle does not copy credential files, old judge tasks or scores, previous
panel selections, expert labels, validation approvals, or executable batch
handles. A new unapproved judge-acceptance template is provided. Credential file
paths in configuration are references, not credential contents, and are not
opened while pooling.

The configuration sets `reused_generation_only: true`,
`calibration_only: true` and `purpose: study_calibration`. Target generation and
generation recovery are disabled. Ordinary result consolidation excludes this
run so the copied answers cannot increase the reported number of model trials.
Original token and cost fields remain attached to their observations but are
historical costs. New judge requests have their own records and billing.

## Failure handling

Pooling rejects incompatible protocols, conflicting deployments, duplicate
observations, mock data, previously reused/calibration runs, unfinished source
generations, corrupted blobs and unsafe or occupied output locations. Output is
built in a temporary sibling directory and installed only after validation and
copying succeed. Existing populated directories and original runs are never
overwritten.

The software does not guarantee that 60 answers or one expert provide sufficient
judge validation for every intended claim. The number of independent questions,
chains, scenarios, models and error categories also matters. Increasing answer
count by adding model responses to shared questions does not create additional
independent scenarios.

The default reports agreement with the single-expert reference, not expert
consensus. They cannot measure inter-human agreement or repeated-judge stability
from one scoring call. The expert's role in corpus construction must be disclosed.
To use two independent experts, pass `--reviewers 2` to `calibration-prepare` or
`reviewers=2` to `prepare_validation`; complete both worksheets and the adjudicated
reference. Sample sizes and judge repetitions are separately configurable.
