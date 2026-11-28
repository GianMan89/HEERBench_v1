# Release verification for v0.3.0

Verification used Python 3.12.14 on Linux. No paid model calls were made.

## Completed checks

- **289 automated tests passed** in the final full suite. Tests cover corpus
  integrity, exact E1/E2/E3 inputs and parent links, graph-aware analysis,
  file-only credentials, provider contracts, native batch dependency waves,
  uncertain submissions and recovery, conservative cost admission, structured
  judging, expert-panel integrity, immutable study extensions, selected-model
  execution, safe calibration pooling, read-only compatible reporting and
  credential-free release packaging.
- The construction validator checked **1,257 original rubric items, 112 documents,
  488 questions, 2,235 criteria, 150 chains, 313 dependency sets, 1,219 source
  links and 492 inherited error-check links** with no errors.
- All **53 corpus files** in the v0.2 repository were compared byte for byte
  with this release and remain unchanged, including the separate acceptance
  overlay. No new military labels, references or scenario content were invented.
- The current offline CLI demonstration completed **88 mock generations and
  88 explicitly unscorable mock judgements**, without contacting providers.
- Pilot planning contains **20 complete chains, 84 unique questions and 336
  target requests** for two models and one repetition. Ordinary judging forecasts
  336 primary and 68 secondary calls.
- Default full planning contains **36,762 target requests** for eleven models
  and three repetitions, 36,762 primary calls and 7,359 secondary calls. The
  secondary count uses ceiling rounding separately for each model/repetition.
- Optional extended planning contains **17,824 target requests** for sixteen
  selectable entries and one repetition. Gemma is shared with the default group,
  yielding **26 unique targets** across the two pools.
- Complete-prompt preflight estimates fit all current default and extended
  endpoints. The maximum planning input estimate including the admission margin
  is 103,139 tokens. This is not a vendor-tokenizer certificate. Actual E3 outputs
  are checked again before dispatch; context failures are retained without
  truncating scenario information.
- Incrementality tests exercise model A/repetition 1, an unchanged resume,
  addition of model B, then expansion of A to repetitions 2 and 3. Earlier
  configurations, request plans and responses remain unchanged. Selecting A
  neither dispatches B nor requires B's key.
- Calibration tests cover the new default of **60 actual-answer selections,
  10 development and 50 held out, one expert and one judge call per answer**.
  They verify the single blank worksheet, exact criterion coverage, component
  separation, development-only reporting without held-out verdict disclosure,
  absent-error metrics, missing automated labels and immutable selected content.
  The optional two-expert, adjudicated 200-answer / 3-call design remains tested.
  These tests use temporary synthetic provider fixtures and are software checks,
  not actual judge-validation findings.
- Calibration reports reject changed model versions, configuration or submitted
  model identities under an existing judge ID. Accepted calibration is bound to
  the held-out report, recorded expert reference type, judge and corpus.
  Single-source and consolidated reports retain that reference type without
  inferring additional reviewers or judge repeatability.
- Refreshed planning examples preserve the preceding release's generation and
  ordinary judging totals. The separately reported default calibration forecast
  now uses **60 calls rather than 600**, with configurable counts for other designs.
- Packaging tests prove that sentinel credentials are not read or included,
  local keys remain untouched, symlinks and local experiment directories are
  excluded, and the four distributed key files are empty.

## What these checks establish

The tests establish software behaviour against frozen data and documented
provider contracts. Public catalogue metadata establishes that model identifiers
and pinned routes were listed when checked. It does not establish live account
access, quota, endpoint behaviour, judge validity, model accuracy or military
operational effectiveness. No API credentials were used for model calls, and no
real human-labelled validation panel was produced during this release.

Planning cost estimates are included in `examples/planned_runs/`. They use real
prompt material and stated token-length assumptions, not observed invoices.
Model-managed weight changes cannot be ruled out by stable names alone. Real
pilot measurements and held-out expert judge validation remain empirical steps.
