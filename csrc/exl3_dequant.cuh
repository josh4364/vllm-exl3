#pragma once

#include <cstdint>
#include <cstring>
#include <cmath>

namespace vllm_exl3 {

constexpr std::uint32_t MCG_MULTIPLIER = 0xCBAC1FEDu;
constexpr std::uint32_t MUL1_MULTIPLIER = 0x83DCD12Du;

inline std::uint32_t lop3_6a(std::uint32_t a, std::uint32_t b,
                             std::uint32_t c) {
    std::uint32_t out = 0;
    for (unsigned bit = 0; bit < 32; ++bit) {
        // PTX LOP3 indexes the LUT as a:b:c (a is the most-significant bit).
        const unsigned index = (((a >> bit) & 1u) << 2u) |
                               (((b >> bit) & 1u) << 1u) |
                               ((c >> bit) & 1u);
        if ((0x6Au >> index) & 1u) out |= 1u << bit;
    }
    return out;
}

inline float half_bits_to_float(std::uint16_t bits) {
    const std::uint32_t sign = (bits >> 15) & 1u;
    const std::uint32_t exp = (bits >> 10) & 0x1Fu;
    const std::uint32_t mant = bits & 0x3FFu;
    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        const float value = std::ldexp(static_cast<float>(mant), -24);
        return sign ? -value : value;
    }
    if (exp == 31) return mant ? NAN : (sign ? -INFINITY : INFINITY);
    const float value = std::ldexp(1.0f + static_cast<float>(mant) / 1024.0f,
                                   static_cast<int>(exp) - 15);
    return sign ? -value : value;
}

inline std::uint16_t float_to_half_bits(float value) {
    std::uint32_t raw;
    std::memcpy(&raw, &value, sizeof(raw));
    const std::uint32_t sign = (raw >> 16) & 0x8000u;
    const int exp = static_cast<int>((raw >> 23) & 0xFFu) - 127;
    const std::uint32_t mant = raw & 0x7FFFFFu;
    if (exp > 15) return static_cast<std::uint16_t>(sign | 0x7C00u);
    if (exp < -14) {
        if (exp < -24) return static_cast<std::uint16_t>(sign);
        const std::uint32_t m = (mant | 0x800000u) >> (-exp - 1);
        return static_cast<std::uint16_t>(sign | (m >> 13));
    }
    return static_cast<std::uint16_t>(sign |
        (static_cast<std::uint32_t>(exp + 15) << 10) | (mant >> 13));
}

inline float decode_mcg(std::uint16_t window) {
    std::uint32_t x = static_cast<std::uint32_t>(window) * MCG_MULTIPLIER;
    x = lop3_6a(x, 0x8fff8fffu, 0x3b603b60u);
    return half_bits_to_float(static_cast<std::uint16_t>(x)) +
           half_bits_to_float(static_cast<std::uint16_t>(x >> 16));
}

inline float decode_mul1(std::uint16_t window) {
    const std::uint32_t x = static_cast<std::uint32_t>(window) * MUL1_MULTIPLIER;
    const std::uint32_t sum = 0x6400u + (x & 0xFFu) + ((x >> 8) & 0xFFu) +
                              ((x >> 16) & 0xFFu) + ((x >> 24) & 0xFFu);
    return half_bits_to_float(static_cast<std::uint16_t>(sum)) *
           half_bits_to_float(0x1eeeu) + half_bits_to_float(0xc931u);
}

inline std::uint16_t read_window(const std::uint16_t* packed, std::size_t words,
                                 std::size_t bit) {
    bit %= words * 16;
    const std::size_t word = bit / 16;
    const unsigned shift = static_cast<unsigned>(bit & 15u);
    const std::uint32_t a = packed[word];
    const std::uint32_t b = packed[(word + 1) % words];
    return static_cast<std::uint16_t>((a >> shift) | (b << ((16 - shift) & 15)));
}

inline float decode_weight(const std::uint16_t* packed, std::size_t words,
                           int index, int bits, bool mcg) {
    const std::size_t start = (static_cast<std::size_t>(index) + 257u) * bits - 16u;
    const std::uint16_t window = read_window(packed, words, start);
    return mcg ? decode_mcg(window) : decode_mul1(window);
}

inline void hadamard(float* values, int n) {
    for (int stride = 1; stride < n; stride <<= 1) {
        for (int base = 0; base < n; base += stride << 1) {
            for (int i = 0; i < stride; ++i) {
                const float a = values[base + i];
                const float b = values[base + stride + i];
                values[base + i] = a + b;
                values[base + stride + i] = a - b;
            }
        }
    }
}

#if defined(__CUDACC__)
// Device equivalents used by the CUDA kernel variants.  Keeping this primitive
// header-only lets future GEMV kernels share exactly the same trellis window
// extraction and MCG codebook as the extension harness.
__device__ __forceinline__ std::uint32_t device_lop3_6a(std::uint32_t a,
                                                        std::uint32_t b,
                                                        std::uint32_t c) {
    std::uint32_t out = 0;
    #pragma unroll
    for (unsigned bit = 0; bit < 32; ++bit) {
        const unsigned index = (((a >> bit) & 1u) << 2u) |
                               (((b >> bit) & 1u) << 1u) |
                               ((c >> bit) & 1u);
        if ((0x6Au >> index) & 1u) out |= 1u << bit;
    }
    return out;
}

__device__ __forceinline__ std::uint16_t device_read_window(
    const std::uint16_t* packed, int words, std::size_t bit) {
    bit %= static_cast<std::size_t>(words * 16);
    const int word = static_cast<int>(bit / 16);
    const unsigned shift = static_cast<unsigned>(bit & 15u);
    const std::uint32_t a = packed[word];
    const std::uint32_t b = packed[(word + 1) % words];
    return static_cast<std::uint16_t>((a >> shift) | (b << ((16 - shift) & 15)));
}

template <bool MCG>
__device__ __forceinline__ float dequant_weight_device(
    const std::uint16_t* packed, int words, int index, int bits) {
    const std::size_t start = (static_cast<std::size_t>(index) + 257u) * bits - 16u;
    const std::uint32_t window = device_read_window(packed, words, start);
    if constexpr (MCG) {
        std::uint32_t x = window * MCG_MULTIPLIER;
        x = device_lop3_6a(x, 0x8fff8fffu, 0x3b603b60u);
        return __half2float(__ushort_as_half(static_cast<std::uint16_t>(x))) +
               __half2float(__ushort_as_half(static_cast<std::uint16_t>(x >> 16)));
    }
    const std::uint32_t x = window * MUL1_MULTIPLIER;
    const int sum = 0x6400 + static_cast<int>(x & 0xffu) +
                    static_cast<int>((x >> 8) & 0xffu) +
                    static_cast<int>((x >> 16) & 0xffu) +
                    static_cast<int>((x >> 24) & 0xffu);
    return __half2float(__ushort_as_half(static_cast<std::uint16_t>(sum))) *
           __half2float(__ushort_as_half(0x1eeeu)) +
                        __half2float(__ushort_as_half(0xc931u));
}

__device__ __forceinline__ void hadamard_device(float* values, int n) {
    for (int stride = 1; stride < n; stride <<= 1) {
        for (int base = 0; base < n; base += stride << 1) {
            for (int i = 0; i < stride; ++i) {
                const float a = values[base + i];
                const float b = values[base + stride + i];
                values[base + i] = a + b;
                values[base + stride + i] = a - b;
            }
        }
    }
}
#endif

}  // namespace vllm_exl3
