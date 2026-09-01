# Changelog

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
