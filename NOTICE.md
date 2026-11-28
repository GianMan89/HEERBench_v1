# Provenance notice

This repository packages HEERBench ABF corpus version 0.5, constructed from the
MA-Bench release supplied by the project authors. It preserves all 1,257 original
rubric records and 112 original documents in its provenance directory. The approved
construction archive SHA-256 is
`3d8636cefe60925290c45a3d3345589e9609c9e28ad1ceea115e651322b75a92`.

All original `reference/`, `model_input/`, `provenance/`, `method/` and `validation/`
files are retained. The construction validator is in `scripts/` within the dataset.
`acceptance.json` is a new overlay recording the user's subsequent statement of
joint expert acceptance. It supplies neither retrospective reviewer labels nor an
inter-rater agreement statistic.

The experiment software was prepared with AI assistance and reviewed through
offline contract and integration tests. Its deterministic construction checks do
not substitute for the supplied military expert acceptance or the separate,
still-required empirical validation of an LLM judge. Mock examples are software
checks and are labelled throughout.

No credentials, human annotation decisions or real model experiment outcomes are
included. Provider rates and endpoint availability are dated configuration data,
with source URLs, not promises of future availability.
