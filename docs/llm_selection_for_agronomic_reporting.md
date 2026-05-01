# Open-Weight LLM Selection For Agronomic Reporting

## Scope

The LLM block should not reconstruct cotton bolls, estimate diameter, or compute volume. Those claims must come from calibrated geometry, multi-view association, and morphology extraction. The LLM block is useful after measurement, where it converts structured outputs into an agronomist-facing report and can be evaluated for schema validity, factual grounding, latency, and expert alignment.

## Recommended Model Set

Use three models:

| Role | Model | Why it belongs | Main risk |
|---|---|---|---|
| Primary reasoning model | `Qwen/Qwen3-32B` | Apache-2.0, strong reasoning mode, tool/agent support, long-context option, good fit for structured reasoning over morphology tables. | Knowledge hallucination risk if allowed to free-form outside supplied measurements. |
| Multimodal/local VLM baseline | `google/gemma-3-27b-it` | Open-weight VLM with image+text input, 128K context, useful for qualitative frame-level interpretation and figure-caption assistance. | Gemma license is not Apache/OSI open source; treat as open-weight, not strictly open-source. |
| Efficient Apache VLM/agent baseline | `mistralai/Mistral-Small-3.1-24B-Instruct-2503` | Apache-2.0, vision, JSON/function-calling orientation, 128K context, smaller than Gemma/Qwen for latency-cost comparison. | May be less strong than Qwen for deeper reasoning; should be evaluated, not assumed. |

Optional fourth baseline: `meta-llama/Llama-3.3-70B-Instruct`. It is a strong text-only 70B model with 128K context, but the license is custom rather than Apache, and the model is heavier. Use it only if compute is available and if the paper needs a familiar large-model baseline.

## Why These Models Fit This Project

The project needs structured agronomic interpretation, not generic chat. The LLM receives a JSON/table like:

```json
{
  "plot_id": "P205",
  "phase": "post",
  "boll_count": 3307,
  "mean_diameter_mm": 31.2,
  "mean_volume_mm3": 15910,
  "visibility": 0.74,
  "occlusion": 0.21,
  "pre_post_delta": {
    "count_visible_pct": 18.4,
    "volume_visible_pct": 22.1
  }
}
```

The LLM must return a fixed schema:

```json
{
  "recommendation": "...",
  "evidence": ["..."],
  "uncertainties": ["..."],
  "do_not_claim": ["..."],
  "needs_human_review": true
}
```

This makes the LLM output auditable. It also lets the paper compare models without depending on subjective prose quality.

## Evaluation Axes

Use these model-comparison metrics:

| Metric | Definition | Direction |
|---|---|---|
| Schema validity | Percent of outputs that parse and match the required JSON schema. | Higher is better |
| Measurement faithfulness | Percent of claims directly supported by provided numeric morphology fields. | Higher is better |
| Hallucination rate | Unsupported agronomic claims, invented disease/stress/yield claims, or invented field conditions. | Lower is better |
| Expert alignment | Agreement with agronomist rubric on recommendation usefulness and caution. | Higher is better |
| Latency | Wall-clock generation time at fixed prompt/output budget. | Lower is better |
| Token cost | Input/output token count at fixed task set. | Lower is better |

## Paper Framing

The safest paper claim is:

> We evaluate open-weight LLMs only as a post-measurement decision-support layer. The model is given structured morphology outputs and is not allowed to infer unseen biological states. This isolates geometric phenotyping accuracy from language-generation behavior and allows LLM reliability to be evaluated through schema validity, faithfulness, latency, and expert alignment.

## Evidence Notes

- Qwen3-32B is Apache-2.0, has thinking and non-thinking modes, supports 32K native context and 131K context with YaRN, and emphasizes reasoning, tool integration, and multilingual instruction following.
- Gemma 3 27B is a multimodal open-weight model with text and image input, 128K input context for the 27B size, and explicit caution that factual accuracy and domain limitations must be monitored.
- Mistral Small 3.1 24B is Apache-2.0, supports vision, JSON/function calling, strong system-prompt following, and 128K context.
- Llama 3.3 70B is a strong text-only baseline with 128K context and strong reported reasoning benchmarks, but uses a custom Llama license.
- Recent agriculture benchmarks such as AgriBench, AgMMU, AgroBench, and AgriChat show that agriculture remains a difficult domain for general VLMs/LLMs, especially fine-grained perception, factual grounding, and expert-level decision support. That supports our choice to use LLMs only after the measured geometry has already been computed.
- Reddit/LocalLLaMA discussions are useful for operational signs such as local latency, quantization, and hallucination anecdotes, but they should not be cited as scientific evidence in the paper. X searches did not return reliable indexed evidence for the target agriculture/model queries during this pass.

## Sources

- Qwen/Qwen3-32B model card: https://huggingface.co/Qwen/Qwen3-32B
- google/gemma-3-27b-it model card: https://huggingface.co/google/gemma-3-27b-it
- mistralai/Mistral-Small-3.1-24B-Instruct-2503 model card: https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503
- meta-llama/Llama-3.3-70B-Instruct model card: https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct
- AgriBench: https://arxiv.org/abs/2412.00465
- AgMMU: https://arxiv.org/abs/2504.10568
- AgroBench: https://arxiv.org/abs/2507.20519
- AgriChat: https://arxiv.org/abs/2603.16934
