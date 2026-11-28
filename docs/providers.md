# Providers, routing, batching and costs

## Contract and reproducibility

The adapters use HTTP directly through `httpx`. A `GenerationRequest` carries
`request_id`, `model`, `system`, the stable scenario `prefix`, the variable `task`,
an output budget and explicitly configured sampling parameters. `parameters` is
provider-specific. Reserved fields cannot replace the prompt, model, tool policy
or number of generated candidates. Each call generates a fresh answer. There is
no completion cache and no silent model replacement.

`GenerationResponse` retains the requested and returned models, final text,
termination reason, normalized token counts, full provider response, payload hash,
execution mode and synchronous latency. Provider reasoning text is retained only
in the raw response when the API returns it, not concatenated into the scored
answer. This does not provide access to unreturned internal reasoning.

`accepted_response_models` is an explicit allowlist, with exact requested-model
matching as its default. If a pilot returns a dated version, inspect it and freeze
that version in the experiment configuration. Replacing an alias or changing the
allowlist after a run starts creates a different experiment configuration.
Routing providers must additionally use one pinned backend and disable fallback.
Provider errors retain the response body for diagnostic and usage accounting.

## Current model availability

`configs/provider_catalog.yaml` records the API identifiers, source URLs, serving
profiles and dated price cards. The v0.2 default pool contains eleven target models,
while `extended` contains the optional frontier comparison pool. Models are selected
per run rather than all submitted automatically. See [the model catalogue](model_catalog.md)
for the exact identities, architecture metadata, availability and current source links.

The default target pool is GPT-6 Luna, Claude Haiku 5.5, Gemini 3.5 Flash-Lite,
Qwen3.5-9B, Qwen3.6-27B, Qwen3.6-35B-A3B, Gemma 4 31B, DeepSeek V4 Flash,
Mistral Small 4, Llama 4 Maverick and Nemotron 3.5 Lightning. Grok is excluded from
the default pool and is available only as an explicit optional selection. GPT-6.1
Terra and Claude Sonnet 5.5 are the primary and secondary judge profiles.

The two-model pilot uses Qwen3.5-9B and Gemma 4 31B. Their router backends and
precision are pinned in the catalogue. In particular, the v0.2 Gemma profile uses
CoreWeave FP4 and must not be pooled silently with the earlier DeepInfra FP8 profile.
Precision is part of the evaluated system. Undisclosed weight revision or precision
remains unknown rather than inferred from a family name.

The `legacy` group preserves earlier profiles, including disabled endpoints and
configuration provenance. Those entries are not default recommendations. A listed
endpoint is not proof of account access, quota or a successful completion. No
availability check in this release substitutes for a small live pilot. Source
snapshots and routing documentation are linked from the catalogue. Do not truncate
long scenarios or silently select another checkpoint to make an endpoint succeed.

Context admission requires the complete system prompt, source packet, question,
predecessor answers and output allocation. A characters-per-token estimate is not
an exact tokenizer count. Use a versioned local tokenizer or a provider token
counter where available. Retain the counting method and observed input counts.
An API context rejection is a failed execution, never permission to remove sources.

## Prefix caching

The scenario prefix stays byte-identical across comparable requests. Task and
predecessor text follow it. Caching changes execution cost and latency, not the
logical information supplied. Native cache hits are best effort. Report the
observed token categories, and do not assume a fixed hit percentage for forecasts.

| Adapter | Implemented cache option | Accounting |
| --- | --- | --- |
| OpenAI Responses | Explicit stable-prefix breakpoint, implicit mode or explicit mode without breakpoints | Separate read, 30-minute write and 12-hour write tokens |
| Anthropic Messages | Explicit breakpoint on the scenario block, 5-minute or 1-hour TTL | Ordinary input, read and write counts are disjoint |
| Gemini generateContent | Provider-native implicit prefix caching | Cached prompt tokens are included in total prompt tokens |
| Compatible chat | Stable-prefix ordering, without vendor-specific cache directives | Provider-reported usage only |

The configured OpenAI price cards can expose charged cache writes. The code uses
the selected model's exact input, cache-read and lifetime-specific write rates.
Rates and thresholds are not borrowed from a different model family. An absent
lifetime split produces an incomplete cost estimate.
[OpenAI caching documentation](https://developers.openai.com/api/docs/guides/prompt-caching)

Anthropic permits prompt caching with Message Batches, but cache hits are not
guaranteed by batch submission. The short and long lifetime categories remain
separate in normalized usage.
[Anthropic batch guide](https://platform.claude.com/docs/en/build-with-claude/batch-processing)
and [cache guide](https://platform.claude.com/docs/en/build-with-claude/prompt-caching).

This release does not create Gemini explicit cache-storage resources. Consequently,
it does not require cache-resource renewal, expiry recovery or storage cleanup.
The separate storage rate is documented in the catalog for future extensions.
[Gemini caching](https://aistudio.google.com/docs/caching).

## Native batches

`submit_batch(requests, label)` returns a serializable `BatchHandle`.
`poll_batch(handle)` returns `BatchStatus` with a `terminal` property.
`collect_batch(handle, requests_by_id)` returns a list of dictionaries containing
`request_id`, `status`, `response` and `error`. Only successful native batch
responses receive `execution_mode="batch"`.

| Provider | Implemented stages |
| --- | --- |
| OpenAI | Upload JSONL to Files, create `/batches` for `/v1/responses`, poll, retrieve both output and error files |
| Anthropic | Create `/messages/batches`, poll and retrieve JSONL results |
| Gemini | Submit inline `batchGenerateContent`, poll long-running operation and collect inline results |

OpenAI and Anthropic records are joined through hashed custom IDs, not file order.
Gemini metadata keys are used when returned. Otherwise its documented inline
ordering guarantee is used only with an exact result count. Duplicate IDs,
unexpected IDs and changed request payloads fail validation. Missing results are
explicit failures. They are not automatically regenerated.

Sources [OpenAI Batch](https://developers.openai.com/api/docs/guides/batch),
[Anthropic batch API](https://platform.claude.com/docs/en/api/messages/batches/create),
[Gemini Batch](https://aistudio.google.com/docs/batch-api),
[Gemini batch response reference](https://aistudio.google.com/api-reference/batch-api).

Gemini inline batches have a 20 MB request limit. `estimate_batch_bytes` exposes
the payload size to the runner. Batch requests can only include ready nodes in
the ABF graph. Generation of later stages waits for the preceding outputs that
their condition requires. Parallel synchronous calls receive no native batch
discount. Compatible chat endpoints intentionally reject native batch submission.

## Recovery and billing

The HTTP layer performs one attempt. It classifies transient status codes and
returns numeric `Retry-After` when available. The runner decides retry policy and
logs attempts. A timeout during generation or batch submission, or a server failure after submission
may have created a paid job. Such exceptions set `submission_uncertain=True` and
must not be retried blindly. Save the submission intent before the HTTP call and
the returned handle immediately after it. Polling and collection can resume from
the saved handle. A failed parsing or identity check retains the original response.

Each price card lists `standard` and `batch` rates separately. `cost_from_usage`
uses `Decimal` and returns an observed-token list-price calculation, which is not
an invoice. Missing rates or token categories return `total_usd=null` and explicit
unresolved fields. The known subtotal does not include unassignable input usage.
Reasoning tokens are included once in billed output. In particular, Gemini
reports candidate and thinking token counts separately, while the selected other
APIs include thinking in their output total.
[Gemini token fields](https://aistudio.google.com/api-reference/generate-content).

The Gemini 3.1 Pro batch cache-read rate remains the same as its standard read
rate. A universal 50 percent multiplier would therefore misprice cached batch
requests. Context-dependent price tiers use the complete prompt length.
[Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing),
[OpenAI pricing](https://platform.openai.com/pricing),
[Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing).

`conservative_request_reservation` reserves input at the highest configured
input/write rate and output at its cap. It is conservative only conditional on
the supplied input-token bound and current price card. Regional surcharges,
external tools, taxes and other account usage require separate reconciliation.
The implemented requests enable no external tools.

Offline forecasting uses each actual source packet and, for judging, the actual
question, source passages, reference, alternatives, criterion and error checks,
system instructions and response schema. Candidate answers are length surrogates
only. No cache hits are assumed. A generation prefix configured for explicit
caching is charged as a new write on every request in the estimate. Current judge
requests have no separate prefix breakpoint, so explicit caching adds no write
fee to those requests. Actual usage may differ and replaces these estimates.

`forecast_judges` includes primary scoring of every planned answer. Under the
v0.2 sampling policy, the secondary-call count is the sum of
`ceil(secondary_fraction * answers_in_group)` across target-model/global-repetition
groups. This preserves the selected panel when other models or repetitions are
added later. The default secondary fraction is 20%. The actual prompt mix cannot
be known until successful answers exist, so the forecast uses the planned mean
and includes a minimum/maximum prompt-cost envelope. A separate optional
60-call estimate covers 60 validation answers with one primary-judge call each.
Use `estimate --calibration-answers` and `--calibration-repetitions` for another
panel design. This estimate is not included in the ordinary scoring total and
does not imply that a validation panel has already been collected or divided
into valid groups. Output-token assumptions include billed reasoning.

`measured_charge_completeness` reports observed list-price subtotals alongside
requests with unknown usage, missing rates or unquantified earlier attempts.
Summing successful response costs alone does not establish the full charge when
failed, uncertain or repeated submissions exist. The final provider invoice is
the authority for actual expenditure.

## Credentials and offline tests

Place each required raw API key in its configured local file, such as
`secrets/openai.key`, `secrets/anthropic.key`, `secrets/google.key` or
`secrets/openrouter.key`. The supplied files are blank and Git ignores their contents.
Active profiles read keys only from files and do not fall back to credential
environment variables. Keys are read only for an explicitly requested credential
check or network operation. Values are never stored in request payloads, frozen
manifests or exported reports. See [credential handling](credentials.md) for the
complete file contract. HTTP proxy variables are honoured. If the environment uses
a SOCKS proxy, install `httpx[socks]`.

The contract suite uses `httpx.MockTransport`, exercises real serialization and
batch response decoding, and performs no network or paid calls. Passing it verifies
the implementation against documented contracts. A small paid pilot still has to
confirm account access, current model versions, token limits and judge validity.

## Provisioning optional legacy checkpoints

`configs/local_endpoint.example.yaml` preserves the original Qwen3-30B-A3B and
Qwen3-235B-A22B identities for a provisioned OpenAI-compatible endpoint. It stays
disabled and has unknown prices until the deployment is verified. Configure the
actual base URL, served model identifier, exact weight revision, precision, chat
template, context extension and maximum context. Record the server version and
hardware in the run metadata. A local service can use a base URL such as
`http://localhost:8000/v1`, with its generation route at `/chat/completions`.
No server is deployed or started by this repository.

Dedicated GPU hosting is charged by deployment time rather than necessarily by
tokens. Keep infrastructure charges in a separate ledger. Do not replace unknown
API or hosting costs with zero. Fireworks lists the original 235B checkpoint for
on-demand deployment, but does not currently advertise it as serverless.
[Fireworks checkpoint page](https://fireworks.ai/models/fireworks/qwen3-235b-a22b).
