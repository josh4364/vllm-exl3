![vllm-exl3 — EXL3 quantization plugin for routed MoE serving](assets/header.png)

# vllm-exl3

[![Follow on X](https://img.shields.io/badge/Follow-%40ViC305-black?logo=x)](https://x.com/ViC305) [![Follow on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Follow-vcruz305-yellow)](https://huggingface.co/vcruz305)

An out-of-tree vLLM plugin that registers `--quantization exl3`, serving
EXL3 (ExLlamaV3 trellis, MCG codebook) quantized packs — routed MoE experts
run packed through `exllamav3_ext` kernels, never dequantized to a dense
format at load.

If you use this plugin, please credit **vcruz305**.

## Scope (read this first)

This is **not** a plugin for stock vLLM. Upstream vLLM declined EXL3 support
([vllm-project/vllm#19896](https://github.com/vllm-project/vllm/issues/19896)),
and this plugin targets vLLM **fork lineages** that provide the
`RoutedExperts` fused-MoE layer family (the DGX Spark GLM/DeepSeek serving
forks). It also requires
[exllamav3](https://github.com/turboderp-org/exllamav3) with its compiled
`exllamav3_ext` kernels for your GPU arch.

## Supported architectures

| Architecture | Status | Reference pack |
|---|---|---|
| `Glm5Next` (GLM-5.3-Flash) | serving-proven | GLM-5.3-Flash EXL3 K2 / K2K3-mix |
| `DeepseekV4` (DeepSeek-V4-Flash) | serving-proven, needs fork-side loader adjustments for vision/MTP checkpoint extras | DSV4-Flash-Vision EXL3 MixedK |

## Config contract

The pack's `config.json` must declare the quantization; without it, vLLM
silently resolves whatever the base model's config claims and the load is
wrong by construction:

```json
"quantization_config": {
  "quant_method": "exl3",
  "bits": 2,
  "codebook": "mcg",
  "layer_bits": {"3": 3, "13": 3},
  "non_routed_quantization": {"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [128, 128]}
}
```

- `bits` — default bits-per-weight for routed experts.
- `layer_bits` *(optional)* — per-layer override map for mixed-bitrate
  (MixedK) packs; keys are layer indices as strings.
- `non_routed_quantization` *(optional)* — for packs whose non-routed
  weights stay in the official source format (e.g. DeepSeek block-FP8),
  the declared quant method handles those layers; the exl3 method composes
  with it instead of forcing them unquantized. Omit for packs whose
  non-routed weights are native BF16 (e.g. GLM-5.3-Flash).

## Install

Prebuilt wheels ship alongside the runtime wheels on Hugging Face for fast
one-shot installs — see
[vcruz305/GLM-5.3-Flash-EXL3-K2-spark-vllm](https://huggingface.co/vcruz305/GLM-5.3-Flash-EXL3-K2-spark-vllm)
and the recipe repos below. Or install straight from a GitHub release:

```bash
pip install https://github.com/vcruz305/vllm-exl3/releases/download/v0.2.0/vllm_exl3-0.2.0-py3-none-any.whl
```

The old `glm53_exl3_plugin` import path still works via a deprecated shim
and will be removed in a future release.

## Recipes

- [GLM-5.3-Flash EXL3 K2 on one DGX Spark](https://github.com/vcruz305/GLM-5.3-Flash-EXL3-K2-DGX-Spark-recipe)
- [GLM-5.3-Flash EXL3 K2/K3 mix on one DGX Spark](https://github.com/vcruz305/GLM-5.3-Flash-EXL3-K2K3-mix-DGX-Spark-recipe)

## License

Apache-2.0. Redistribution must retain the [NOTICE](NOTICE) file — see
`LICENSE` §4(d).

## Roadmap

- **0.3.0 — stock vLLM support (DeepSeek-V4 first).** Compat layer for stock's
  `FusedMoE`/`FusedMoEMethodBase` alongside the fork's `RoutedExperts`, quant
  registration via `register_quantization_config`, non-routed delegation to
  stock `fp8`. GLM-5.3 remains fork-only until the architecture exists upstream.
- Fat-expert prefill acceleration (sorted/batched expert dispatch) for extreme
  contexts.
