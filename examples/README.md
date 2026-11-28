# Example artifacts

`offline_demo/` contains output from the explicitly labelled, zero-network mock
pipeline. Its 88 generation placeholders and 88 unscorable judge outputs are
software checks, not observations about a language model. Coverage lower bounds
of zero are mechanical placeholders and must not appear as empirical results.

`planned_runs/` contains planning, offline preflight and cost-forecast examples
for the distributed pilot and main configurations. No target-model generation
took place for these plans. Provider identities, context limits and prices refer
to the catalogue snapshot. Workspace paths are illustrative. Create fresh plans
on your own machine instead of attempting to resume these manifest excerpts.

The estimates assume 1,000 billed output tokens per generated answer and 1,200
per judge response, complete source packets and reference substitutes for unknown
sequential answers. They include configured cache-write charges and assume no
cache hits. Inputs are estimated from UTF-8 length, not counted by a vendor
model-specific tokenizer. Long-context rate thresholds are applied to those
estimates. Actual tokenisation or longer generated parents can cross a threshold.

| Configuration | Target generation | Ordinary judging | Total |
|---|---:|---:|---:|
| Two-model pilot, one repetition | $1.45 | $4.70 | $6.15 |
| Eleven-model full corpus, one repetition | $92.03 | $86.03 | $178.06 |
| Eleven-model full corpus, three repetitions | $276.09 | $258.08 | $534.17 |

All amounts are USD planning estimates. Pilot uses standard transport. Main uses
native batch where supported and standard router requests otherwise. The separate default
calibration panel now contains 60 answers judged once, giving 60 judge calls
instead of the previous 600. Its forecast is separate from ordinary scoring and
is not repeated for each target model. The exact prompt mix depends on panel
selection. Single-expert annotation is a manual activity, not an API charge.

These totals exclude taxes, storage, unknown charges for failed/uncertain attempts
and any future price changes. Measured pilot usage should replace the token
assumptions before committing to the full experiment. They are not guaranteed
spending caps. `cost_estimate.json` preserves unrounded values and all assumptions.

`planned_runs/extended/` covers the optional sixteen-entry pool.
`planned_runs/costs_per_model_repetition.csv` and the companion JSON give
generation, ordinary judging and total estimates for each of the twenty-six
distinct targets, under the main transport policy. Optional frontier entries
are never executed by default.
