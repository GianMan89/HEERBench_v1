# Model catalogue and provider verification

This release records public provider documentation and public endpoint metadata retrieved on **28 November 2026 UTC**. Listed availability means the exact model and route were found in those sources. It does not establish account access, quota, or successful completion of a paid request. No paid model calls were made when preparing the catalogue.

The experiment chooses models at run creation. `default` (also `defaults`) is the agreed 11-model affordable pool. `pilot` contains Qwen3.5-9B and Gemma 4 31B. `extended` (also `frontier`) contains the 16 user-requested comparison models. Gemma occurs in both pools but has one catalogue identity. The union contains **26 unique target models**. Judges are configured separately. Grok appears only in the optional extended pool.

## Default target pool

Rates below are standard, uncached USD per million input / output tokens at the first pricing tier. Source URLs are attached to every entry in `configs/provider_catalog.yaml`. Router records are preserved in `docs/provider_sources/`.

| Model / selector | Exact API ID | Pinned service | Input cap | Input / output per 1M |
|---|---|---|---:|---:|
| GPT-6 Luna / `gpt6_luna` | `gpt-6-luna` | OpenAI Responses | 1,050,000 | $0.10 / $0.50 |
| Claude Haiku 5.5 / `claude_haiku55` | `claude-haiku-5-5` | Anthropic Messages | 1,000,000 | $0.10 / $0.50 |
| Gemini 3.5 Flash-Lite / `gemini35_flash_lite` | `gemini-3.5-flash-lite` | Gemini Developer API | 1,048,576 | $0.30 / $2.50 |
| Qwen3.5-9B / `qwen35_9b` | `qwen/qwen3.5-9b` | primeintellect/bf16 | 262,144 | $0.085000000 / $0.125000000 |
| Qwen3.6-27B / `qwen36_27b` | `qwen/qwen3.6-27b` | chutes/fp8 | 262,144 | $0.25000000 / $2.000000 |
| Qwen3.6-35B-A3B / `qwen36_35b_a3b` | `qwen/qwen3.6-35b-a3b` | coreweave/fp8 | 262,144 | $0.1000000 / $0.89000000 |
| Gemma 4 31B / `gemma4_31b` | `google/gemma-4-31b-it` | coreweave/fp4 | 262,144 | $0.1000000 / $0.34000000 |
| DeepSeek V4 Flash / `deepseek_v4_flash` | `deepseek/deepseek-v4-flash` | deepinfra/fp8 | 1,048,576 | $0.09000000 / $0.18000000 |
| Mistral Small 4 / `mistral_small4` | `mistralai/mistral-small-2603` | mistral/zdr | 262,144 | $0.15000000 / $0.6000000 |
| Llama 4 Maverick / `llama4_maverick` | `meta-llama/llama-4-maverick` | parasail/fp8 | 524,288 | $0.35000000 / $1.000000 |
| NVIDIA Nemotron 3.5 Lightning / `nemotron35_lightning` | `nvidia/nemotron-3.5-lightning` | deepinfra/bf16 | 262,144 | $0.06000000 / $0.16000000 |

Haiku 5.5 charges **$0.50 / $2.50** when total input exceeds 100,000 tokens. The smaller advertised **$0.10 / $0.50** rate must not be applied to those requests. The catalogue also specifies the higher cache-read and cache-write rates. GPT-6 Luna's rate increases above 272,000 input tokens. The threshold applies to the complete request, including the packet and preceding answers. Native token counts matter because tokenisation differs across providers.

The inexpensive Nemotron 3.5 Lightning endpoint chosen here is DeepInfra BF16 with a **262,144-token** context, rather than an unqualified 1M model headline. Its rate is $0.06 / $0.16. The single pinned backend is the unit of configuration. Qwen3.5-9B uses PrimeIntellect BF16, Qwen3.6-27B Chutes FP8, and Qwen3.6-35B-A3B CoreWeave FP8. Gemma uses CoreWeave FP4. Precision is reported with results and is not treated as a controlled variable across different architectures.

## Extended target pool

This pool preserves the requested model generations even when newer generations now exist. Selection of an older model never silently selects its successor.

| Model / selector | Exact API ID | Pinned service | Input cap | Input / output per 1M |
|---|---|---|---:|---:|
| Muse Spark 1.3 / `muse_spark13` | `meta/muse-spark-1.3` | meta | 1,048,576 | $1.25000000 / $4.25000000 |
| Grok 4.6 / `grok46` | `x-ai/grok-4.6` | xai/zdr | 500,000 | $2.000000 / $6.000000 |
| Claude Opus 5 / `claude_opus5` | `claude-opus-5` | Anthropic Messages | 1,000,000 | $5 / $25 |
| GLM-5.3 / `glm53` | `z-ai/glm-5.3` | z-ai/fp8 | 1,048,576 | $1.4000000 / $4.4000000 |
| GPT-6 Astra / `gpt6_astra` | `gpt-6-astra` | OpenAI Responses | 1,050,000 | $10 / $50 |
| Kimi K3 / `kimi_k3` | `moonshotai/kimi-k3` | moonshotai | 1,048,576 | $3.000000 / $15.000000 |
| GPT-5.6 Sol / `gpt56_sol` | `gpt-5.6-sol` | OpenAI Responses | 1,050,000 | $4 / $20 |
| GPT-5.6 Luna / `gpt56_luna` | `gpt-5.6-luna` | OpenAI Responses | 1,050,000 | $0.20 / $1.20 |
| NVIDIA Nemotron 3 Ultra / `nemotron3_ultra` | `nvidia/nemotron-3-ultra-550b-a55b` | coreweave/bf16 | 262,144 | $0.5000000 / $2.15000000 |
| MiniMax M3 / `minimax_m3` | `minimax/minimax-m3` | deepinfra/fp8 | 524,288 | $0.28000000 / $1.1000000 |
| Claude Fable 5.1 / `claude_fable51` | `claude-fable-5-1` | Anthropic Messages | 1,000,000 | $10 / $50 |
| Qwen3.8-Max / `qwen38_max` | `qwen/qwen3.8-max-0902` | alibaba | 983,616 | $2.000000 / $6.000000 |
| DeepSeek-V4-Pro / `deepseek_v4_pro` | `deepseek/deepseek-v4-pro` | siliconflow/fp8 | 1,048,576 | $0.9000000 / $1.7000000 |
| Gemini 3.1 Pro / `gemini31_pro` | `gemini-3.1-pro-preview` | Gemini Developer API | 1,048,576 | $2 / $12 |
| Mistral Medium 3.5 / `mistral_medium35` | `mistralai/mistral-medium-3-5` | mistral/zdr | 262,144 | $1.5000000 / $7.5000000 |
| Gemma 4 31B / `gemma4_31b` | `google/gemma-4-31b-it` | coreweave/fp4 | 262,144 | $0.1000000 / $0.34000000 |

Three details affect interpretation and budgeting.

- **Qwen3.8-Max** resolves in current provider metadata to the `qwen/qwen3.8-max-0902` snapshot. The exact resolved ID is pinned and must be reported with the experimental results.
- **Kimi K3** uses the model-author Moonshot route at **$3 / $15**. Earlier headline estimates of $1.70 / $8.50 do not describe this pinned route. Cheaper router backends can be added as separately named, verified configurations, with their precision and prices recorded.
- **GPT-5.6 Sol** has a published promotional rate ending **30 November 2026**. `pricing_valid_until` records that deadline. A later run needs an updated rate card. The post-promotion rate was not inferred. Grok's router pricing also rises at the 200,000-token threshold, represented explicitly in the tier table.

Gemini 3.1 Pro is the explicitly named **preview** endpoint `gemini-3.1-pro-preview`. The result metadata must preserve any returned modelVersion. An unknown returned modelVersion needs explicit review and configuration, rather than a wildcard acceptance.

## Judges

| Role | Catalogue selector | API ID | Standard input / output | Native batch input / output |
|---|---|---|---:|---:|
| Primary | `gpt61_terra` | `gpt-6.1-terra` | $1 / $5 | $0.50 / $2.50 |
| Secondary | `claude_sonnet55` | `claude-sonnet-5-5` | $2 / $10 | $1 / $5 |

The primary uses fixed medium reasoning effort. The secondary uses adaptive thinking with fixed high effort. These are model-specific settings, not claims of equal computation. The judges still require expert calibration. Price or generation capability does not establish scoring validity. GPT-6.2 Sol remains a legacy judge profile for prior configurations.

## Context, parameters and cost rules

Every entry distinguishes the provider output ceiling from the experiment's smaller output allowance. A request must satisfy the input cap and the combined context reservation. Packets must never be truncated to make a model fit. Gemini documents separate input and output limits, so the runner's combined check stores their sum and separately enforces its 1,048,576-token input cap. The sum is not advertised as an input window.

Native OpenAI configurations use explicit 30-minute prompt caching. Native Anthropic configurations use explicit 5-minute caching. Cache writes are priced separately from cache reads. Gemini uses implicit caching so the harness creates no billed cache-storage objects. Router caching is recorded from returned usage, with zero cache reuse assumed in the default cost forecast. A cache hit is never presumed simply because two prompts share a prefix.

Native batch prices are listed only for OpenAI, Anthropic and Gemini. The OpenRouter adapter executes standard requests even where the backend may offer a separate batch product. No generic 50% router discount is applied. Gemini 3.5 Flash-Lite's batch cache rate is $0.015 per million, while the older Gemini 3.1 Pro pricing has its own distinct table. Costs must be selected from the corresponding model, mode and threshold.

For every OpenRouter profile, `provider.only` specifies one endpoint tag, `allow_fallbacks` is false, and `require_parameters` is true. Known precision is pinned. Returned model and backend names are checked. Host weight hashes remain unknown where the provider does not publish them. Model access can change without a new local release, which is why the pilot and logged response metadata remain necessary.

Most router profiles retain their model's documented default reasoning behaviour. The native settings explicitly pin their advertised defaults where supported. The exact supplied configuration is frozen per run. There are no reasoning-mode ablations in this protocol. Unspecified vendor internals must not be interpreted as identical compute budgets.

The pool includes dense models, mixtures of experts and hybrid architectures. Architecture and parameter counts are descriptive, provider-reported covariates. Differences in training, tokenisers, precision, post-training and inference policy prevent attributing cross-model score differences to architecture alone. Unknown counts remain null. Mistral Small 4's sources give different active-parameter accounting conventions, so the catalogue records the discrepancy rather than pretending they are identical.

## Credentials and legacy profiles

The default and extended pools need four local key files.

- `secrets/openai.key`
- `secrets/anthropic.key`
- `secrets/google.key`
- `secrets/openrouter.key`

Only the selected model and judges need their relevant files. Each contains the raw API key and is ignored by Git. No active catalogue profile reads a credential from an environment variable. The old direct SiliconFlow profile uses the separate `secrets/siliconflow.key` file if explicitly selected.

The `legacy` group preserves prior model selectors, including disabled entries whose original endpoints could not serve the corpus. It is not an additional recommended evaluation pool. The untouched v0.1 catalogue is archived as `docs/provider_sources/catalog_v01_original.yaml` for provenance. Gemma's v0.2 serving profile changes from DeepInfra FP8 to CoreWeave FP4, so its old and new outputs must not be silently resumed or pooled as the same configuration. Frozen run manifests identify the exact profile used.

## Primary sources

- [OpenAI model catalogue](https://developers.openai.com/api/docs/models), individual model pages linked from each profile, [prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching) and [batch API](https://developers.openai.com/api/docs/guides/batch).
- [Anthropic model catalogue](https://platform.claude.com/docs/en/models/overview), [pricing](https://platform.claude.com/docs/en/about-claude/pricing) and [effort configuration](https://platform.claude.com/docs/en/build-with-claude/effort).
- [Gemini 3.5 Flash-Lite](https://aistudio.google.com/docs/models/gemini-3.5-flash-lite), [Gemini 3.1 Pro Preview](https://aistudio.google.com/docs/models/gemini-3.1-pro-preview) and [thinking configuration](https://ai.google.dev/gemini-api/docs/thinking).
- [OpenRouter endpoint metadata API](https://openrouter.ai/api/v1/models/qwen/qwen3.5-9b/endpoints). Every archived response records its exact URL and retrieval timestamp. Endpoint snapshots are public metadata, not paid completion records.
- [Qwen3.5-9B model card](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/README.md), [Qwen architecture documentation](https://huggingface.co/docs/transformers/en/model_doc/qwen3_5), [Mistral Small 4 model documentation](https://docs.mistral.ai/models/mistral-small-4) and [its release announcement](https://mistral.ai/news/mistral-small-4/).
