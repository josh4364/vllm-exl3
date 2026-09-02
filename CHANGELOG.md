# Changelog

## 0.2.3

- Dense EXL3 for non-routed linears (`quantization_config.non_routed_exl3`): per-module `layers` map with `bits` and `bf16_shards`, `mul1` codebook alongside `mcg`, mixed EXL3/BF16 shards inside one merged linear, stale BF16 `.weight` tensors discarded with a shape check. `tools/dense_overlay.py` assembles an overlay pack from an existing EXL3 checkpoint (it moves tensors, it does not quantize).
- `quantization_config.non_routed_dtype_policy: "bf16_as_stored"`: dense linears go to vLLM's unquantized method instead of the `non_routed_quantization` delegate, which still serves source-format MTP experts. Fixes silent empty output when BF16 dense weights met an fp8 delegate (DeepSeek-V4-Flash on stock vLLM 0.28).
- Fix: `mul1` codebook marker constant (0x83DCD12D as signed int32 is -2082680531).

## 0.2.2

- Mixed-format packs: `quantization_config.mtp_experts: "source"` routes MTP/draft-block routed experts through the declared `non_routed_quantization` method (e.g. MXFP4) instead of EXL3, enabling MTP speculative serving for packs that keep drafter experts in the source format. Default (`"exl3"`) is unchanged.

## 0.2.1

- Fix: the `glm53_exl3_plugin` compatibility shim now provides a real `glm53_exl3_plugin.exl3` submodule (submodule imports bypassed the lazy alias).

## 0.2.0

First standalone release. Renamed from `glm53_exl3_plugin` 0.1.1 — identical
behavior plus the package rename; the old import path remains as a
deprecated shim.

## 0.1.1 (as glm53_exl3_plugin, shipped in the GLM recipe)

- Raise the fused-MoE per-expert row cap (`TEMP_ROWS_FUSED` 128 → 2048),
  fixing the >163k-token prefill stall where fat experts fell back to a
  slow per-expert reconstruction path.
- Delegate non-routed layers to a pack-declared source-format quant method
  (`quantization_config.non_routed_quantization`).

## 0.1.0 (as glm53_exl3_plugin)

Initial in-recipe release: EXL3/MCG routed-expert quantization method for
vLLM fork runtimes, per-layer `layer_bits` support.
