# Acknowledgments

- **[ExLlamaV3](https://github.com/turboderp-org/exllamav3)** (turboderp) —
  the EXL3 trellis/MCG format and the compiled kernels this plugin drives.
- **[vLLM](https://github.com/vllm-project/vllm)** — the serving engine and
  the plugin/quantization interfaces this package registers into.
- **[MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks)**
  (MIT-licensed) — idea and benchmark credits; no code was copied:
  - fat-expert prefill batching approach and measurements
    ([PR #77](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/pull/77)
    by @plotarmordev);
  - indexer prefill-workspace right-sizing
    ([PR #86](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/pull/86)
    by @nood-co1);
  - `TEMP_ROWS` / batched-tokens cold-prefill ladder data
    ([PR #40](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/pull/40)
    by @MiaAI-Lab);
  - K-pool tail slot-mapping clamp
    ([PR #50](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/pull/50)
    by @MiaAI-Lab);
  - CUDA-graph KV-deduction opt-out measurements
    ([PR #25](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/pull/25)
    by @im0xMagnus).
