#include <torch/extension.h>

#include <array>
#include <cstdint>
#include <vector>

#include "exl3_dequant.cuh"

namespace {

at::Tensor dequant_cpu(const at::Tensor& trellis, const at::Tensor& suh,
                       const at::Tensor& svh, int64_t bits, bool mcg) {
    const auto packed = trellis.contiguous();
    const auto su = suh.to(at::kCPU).contiguous();
    const auto sv = svh.to(at::kCPU).contiguous();
    const int64_t rows = packed.size(0) * 16;
    const int64_t cols = packed.size(1) * 16;
    TORCH_CHECK(rows % 128 == 0 && cols % 128 == 0,
                "trellis dimensions must reconstruct whole 128-element Hadamard blocks");
    auto out = at::empty({rows, cols}, packed.options().dtype(at::kHalf));
    auto* dst = out.data_ptr<c10::Half>();
    const auto* p = packed.data_ptr<int16_t>();
    const auto* sup = su.data_ptr<c10::Half>();
    const auto* svp = sv.data_ptr<c10::Half>();
    const std::size_t words = static_cast<std::size_t>(16 * bits);
    std::vector<float> matrix(static_cast<std::size_t>(rows * cols), 0.0f);

    for (int64_t kt = 0; kt < packed.size(0); ++kt) {
        for (int64_t nt = 0; nt < packed.size(1); ++nt) {
            const auto* tile = reinterpret_cast<const std::uint16_t*>(
                p + (kt * packed.size(1) + nt) * words);
            for (int i = 0; i < 256; ++i) {
                const int64_t r = kt * 16 + i / 16;
                const int64_t c = nt * 16 + i % 16;
                matrix[static_cast<std::size_t>(r * cols + c)] =
                    vllm_exl3::decode_weight(tile, words, i, static_cast<int>(bits), mcg);
            }
        }
    }

    constexpr float norm = 0.08838834764831845f;
    std::array<float, 128> line{};
    for (int64_t r0 = 0; r0 < rows; r0 += 128) {
        for (int64_t c0 = 0; c0 < cols; c0 += 128) {
            for (int r = 0; r < 128; ++r) {
                for (int c = 0; c < 128; ++c) line[c] = matrix[(r0 + r) * cols + c0 + c];
                vllm_exl3::hadamard(line.data(), 128);
                for (int c = 0; c < 128; ++c) matrix[(r0 + r) * cols + c0 + c] = line[c] * norm;
            }
            for (int c = 0; c < 128; ++c) {
                for (int r = 0; r < 128; ++r) line[r] = matrix[(r0 + r) * cols + c0 + c];
                vllm_exl3::hadamard(line.data(), 128);
                for (int r = 0; r < 128; ++r) {
                    const float value = line[r] * norm * static_cast<float>(sup[r0 + r]) *
                                         static_cast<float>(svp[c0 + c]);
                    dst[(r0 + r) * cols + c0 + c] = c10::Half(value);
                }
            }
        }
    }
    return out;
}

}  // namespace

at::Tensor dequant_trellis(const at::Tensor& trellis, const at::Tensor& suh,
                           const at::Tensor& svh, int64_t bits, bool mcg) {
    TORCH_CHECK(bits == 2 || bits == 3 || bits == 4 || bits == 8,
                "K must be one of 2, 3, 4, or 8");
    TORCH_CHECK(trellis.dim() == 3 && trellis.scalar_type() == at::kShort,
                "trellis must be int16 [rows/16, cols/16, 16*K]");
    TORCH_CHECK(trellis.size(2) == 16 * bits, "trellis last dimension must be 16*K");
    TORCH_CHECK(trellis.size(0) > 0 && trellis.size(1) > 0, "trellis dimensions must be positive");
    TORCH_CHECK(suh.scalar_type() == at::kHalf && svh.scalar_type() == at::kHalf,
                "suh and svh must be float16");
    TORCH_CHECK(suh.numel() >= trellis.size(0) * 16 && svh.numel() >= trellis.size(1) * 16,
                "suh/svh are too short");
    const auto device = trellis.device();
    auto result = dequant_cpu(trellis.to(at::kCPU), suh, svh, bits, mcg);
    return device.is_cpu() ? result : result.to(device);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dequant_trellis", &dequant_trellis,
          "Decode an EXL3 trellis tensor into an fp16 weight matrix");
}
