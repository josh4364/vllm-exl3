#include <cuda_fp16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
#include <cstdlib>

#include "util.h"
#include "util.cuh"
#include "quant/exl3_gemv_kernel.cuh"

namespace cg = cooperative_groups;

// Tile configurations.  CFG 0/1 mirror exl3_gemv_kernel.  CFG 2/3 issue every
// load of a work item up front (PF covers the whole k range of a warp for
// hidden=4096 / inter=2048 at 16 k-splits), which matters on GB10's
// high-latency LPDDR5X: with PF=4 each 32 KB item costs several serialized
// memory round trips.
template <int CFG> struct P2bCfg;
// MINB = minimum resident blocks per SM requested through __launch_bounds__
// (caps registers; measured occupancy was 50% at 78 regs for CFG 1).
// WI = warp-per-item: each warp streams one (pair, column group) over the whole
// K range, so there is no cross-warp k-split, no shared-memory reduction and no
// __syncthreads per item (ncu on GB10 showed `barrier` as the top stall of the
// k-split design). WK is then simply the number of warps per block.
template <> struct P2bCfg<0> { static constexpr int WK = 16, WNT = 2, PF = 4,  FOLD = 4, MINB = 2; static constexpr bool WI = false; };
template <> struct P2bCfg<1> { static constexpr int WK = 8,  WNT = 4, PF = 2,  FOLD = 2, MINB = 3; static constexpr bool WI = false; };
template <> struct P2bCfg<2> { static constexpr int WK = 16, WNT = 4, PF = 16, FOLD = 4, MINB = 2; static constexpr bool WI = false; };
template <> struct P2bCfg<3> { static constexpr int WK = 16, WNT = 2, PF = 16, FOLD = 4, MINB = 3; static constexpr bool WI = false; };
template <> struct P2bCfg<4> { static constexpr int WK = 8,  WNT = 4, PF = 4,  FOLD = 4, MINB = 4; static constexpr bool WI = false; };
template <> struct P2bCfg<5> { static constexpr int WK = 8,  WNT = 4, PF = 2,  FOLD = 2, MINB = 4; static constexpr bool WI = false; };
template <> struct P2bCfg<6> { static constexpr int WK = 4,  WNT = 4, PF = 4,  FOLD = 4, MINB = 8; static constexpr bool WI = false; };
template <> struct P2bCfg<7> { static constexpr int WK = 8,  WNT = 4, PF = 8,  FOLD = 4, MINB = 3; static constexpr bool WI = true; };
template <> struct P2bCfg<8> { static constexpr int WK = 8,  WNT = 2, PF = 8,  FOLD = 4, MINB = 3; static constexpr bool WI = true; };
template <> struct P2bCfg<9> { static constexpr int WK = 8,  WNT = 4, PF = 4,  FOLD = 4, MINB = 3; static constexpr bool WI = true; };
constexpr int P2B_NUM_CFG = 10;
constexpr int P2B_MAX_COLS = 64;
constexpr int P2B_MAX_PAIRS = 64;

template <int bits, int cb, int CFG>
__device__ __forceinline__ void run_gemv_tile(
    const uint32_t* __restrict__ B32,
    const half2* __restrict__ A2,
    half* __restrict__ C,
    int kslices,
    int size_k,
    int group,
    int ntiles,
    int warp,
    int lane,
    float* sh_red)   // [WK][COLS]
{
    constexpr int WK = P2bCfg<CFG>::WK;
    constexpr int WNT = P2bCfg<CFG>::WNT;
    constexpr int PF = P2bCfg<CFG>::PF;
    constexpr int FOLD = P2bCfg<CFG>::FOLD;
    constexpr int THREADS = WK * 32;
    constexpr int COLS = WNT * 16;
    static_assert(COLS <= P2B_MAX_COLS, "reduction buffer too small");
    static_assert(PF % FOLD == 0, "FOLD must divide PF");
    constexpr int TWORDS = 8 * bits;
    constexpr int LOADS = bits == 2 ? WNT / 2 : WNT;
    constexpr int LSTRIDE = bits == 3 ? 24 : 32;

    const int chunk = CEIL_DIVIDE(kslices, WK);
    const int ks0 = warp * chunk;
    const int myn = max(0, min(chunk, kslices - ks0));
    const size_t slice_stride = (size_t) ntiles * TWORDS;

    const size_t a_row0 = 0;
    const bool r0_ok = lane < 4;
    const half2 hzero = __half2half2(__ushort_as_half(0));

    int x_src_a = 0, x_src_b = 0, x_s2 = 0;
    if constexpr (bits == 2) {
        int i1 = lane >> 1;
        x_src_b = i1;
        x_src_a = (i1 + 15) & 15;
    } else if constexpr (bits == 3) {
        int t_offset = lane << 3;
        int b1 = (t_offset + 257) * 3;
        int b2 = b1 + 21;
        int i0 = (b1 - 16) / 32;
        int i2 = (b2 - 1) / 32;
        x_s2 = (i2 + 1) * 32 - b2;
        x_src_a = i0 % 24;
        x_src_b = i2 % 24;
    }

    const uint32_t* bp = B32 + (size_t) ks0 * slice_stride + group * WNT * TWORDS + lane;

    auto ld_b = [&] (int i, int l) -> uint32_t {
        if constexpr (bits == 3)
            return lane < 24 ? __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE) : 0;
        else
            return __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE);
    };

    uint32_t pf[PF][LOADS];
    #pragma unroll
    for (int d = 0; d < PF; ++d)
        if (d < myn)
            #pragma unroll
            for (int l = 0; l < LOADS; ++l)
                pf[d][l] = ld_b(d, l);

    FragC_h ch[WNT][2] = {};
    float2 acc0[WNT][2] = {};

    for (int ib = 0; ib < myn; ib += PF) {
        #pragma unroll
        for (int d = 0; d < PF; ++d) {
            const int i = ib + d;
            if (i >= myn) break;

            uint32_t bw[LOADS];
            #pragma unroll
            for (int l = 0; l < LOADS; ++l)
                bw[l] = pf[d][l];

            if (i + PF < myn) {
                #pragma unroll
                for (int l = 0; l < LOADS; ++l)
                    pf[d][l] = ld_b(i + PF, l);
            }

            const size_t a_col = (size_t) (ks0 + i) * 8 + (lane & 3);
            FragB a01, a23;
            a01[0] = r0_ok ? A2[a_row0 + a_col] : hzero;
            a23[0] = r0_ok ? A2[a_row0 + a_col + 4] : hzero;
            a01[1] = hzero;
            a23[1] = hzero;

            #pragma unroll
            for (int t = 0; t < WNT; ++t) {
                FragB f0, f1;
                if constexpr (bits == 4) {
                    uint32_t aw = __shfl_sync(0xffffffffu, bw[t], (lane + 31) & 31);
                    exl3_gemv_ns::dq8_regs_4bits<cb>(aw, bw[t], f0, f1);
                } else if constexpr (bits == 2) {
                    const uint32_t w = bw[t >> 1];
                    const int base = (t & 1) << 4;
                    uint32_t bwv = __shfl_sync(0xffffffffu, w, base + x_src_b);
                    uint32_t awv = __shfl_sync(0xffffffffu, w, base + x_src_a);
                    exl3_gemv_ns::dq8_regs_2bits<cb>(awv, bwv, lane << 3, f0, f1);
                } else {
                    uint32_t awv = __shfl_sync(0xffffffffu, bw[t], x_src_a);
                    uint32_t bwv = __shfl_sync(0xffffffffu, bw[t], x_src_b);
                    exl3_gemv_ns::dq8_regs_3bits<cb>(awv, bwv, x_s2, f0, f1);
                }

                exl3_gemv_ns::mma_ab_h(a01, a23, f0, ch[t][0]);
                exl3_gemv_ns::mma_ab_h(a01, a23, f1, ch[t][1]);
            }

            if ((d + 1) % FOLD == 0 || i + 1 == myn) {
                #pragma unroll
                for (int t = 0; t < WNT; ++t)
                    #pragma unroll
                    for (int f = 0; f < 2; ++f) {
                        acc0[t][f].x += __low2float(ch[t][f][0]);
                        acc0[t][f].y += __high2float(ch[t][f][0]);
                        ch[t][f][0] = hzero;
                    }
            }
        }
    }

    // Warp reduction
    if (lane < 4) {
        #pragma unroll
        for (int t = 0; t < WNT; ++t) {
            #pragma unroll
            for (int f = 0; f < 2; ++f) {
                const int col = t * 16 + f * 8 + (lane & 3) * 2;
                sh_red[warp * COLS + col + 0] = acc0[t][f].x;
                sh_red[warp * COLS + col + 1] = acc0[t][f].y;
            }
        }
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < COLS; idx += THREADS) {
        float sum = 0.0f;
        #pragma unroll
        for (int j = 0; j < WK; ++j)
            sum += sh_red[j * COLS + idx];
        const int col = group * COLS + idx;
        C[col] = __float2half_rn(sum);
    }
    __syncthreads();
}

// Warp-per-item variant of run_gemv_tile: one warp streams the whole K range of
// one (pair, column group); no shared memory, no block barrier.
template <int bits, int cb, int CFG>
__device__ __forceinline__ void run_gemv_warp(
    const uint32_t* __restrict__ B32,
    const half2* __restrict__ A2,
    half* __restrict__ C,
    int kslices,
    int group,
    int ntiles,
    int lane)
{
    constexpr int WNT = P2bCfg<CFG>::WNT;
    constexpr int PF = P2bCfg<CFG>::PF;
    constexpr int FOLD = P2bCfg<CFG>::FOLD;
    constexpr int COLS = WNT * 16;
    static_assert(PF % FOLD == 0, "FOLD must divide PF");
    constexpr int TWORDS = 8 * bits;
    constexpr int LOADS = bits == 2 ? WNT / 2 : WNT;
    constexpr int LSTRIDE = bits == 3 ? 24 : 32;

    const int ks0 = 0;
    const int myn = kslices;
    const size_t slice_stride = (size_t) ntiles * TWORDS;

    const size_t a_row0 = 0;
    const bool r0_ok = lane < 4;
    const half2 hzero = __half2half2(__ushort_as_half(0));

    int x_src_a = 0, x_src_b = 0, x_s2 = 0;
    if constexpr (bits == 2) {
        int i1 = lane >> 1;
        x_src_b = i1;
        x_src_a = (i1 + 15) & 15;
    } else if constexpr (bits == 3) {
        int t_offset = lane << 3;
        int b1 = (t_offset + 257) * 3;
        int b2 = b1 + 21;
        int i0 = (b1 - 16) / 32;
        int i2 = (b2 - 1) / 32;
        x_s2 = (i2 + 1) * 32 - b2;
        x_src_a = i0 % 24;
        x_src_b = i2 % 24;
    }

    const uint32_t* bp = B32 + (size_t) ks0 * slice_stride + group * WNT * TWORDS + lane;

    auto ld_b = [&] (int i, int l) -> uint32_t {
        if constexpr (bits == 3)
            return lane < 24 ? __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE) : 0;
        else
            return __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE);
    };

    uint32_t pf[PF][LOADS];
    #pragma unroll
    for (int d = 0; d < PF; ++d)
        if (d < myn)
            #pragma unroll
            for (int l = 0; l < LOADS; ++l)
                pf[d][l] = ld_b(d, l);

    FragC_h ch[WNT][2] = {};
    float2 acc0[WNT][2] = {};

    for (int ib = 0; ib < myn; ib += PF) {
        #pragma unroll
        for (int d = 0; d < PF; ++d) {
            const int i = ib + d;
            if (i >= myn) break;

            uint32_t bw[LOADS];
            #pragma unroll
            for (int l = 0; l < LOADS; ++l)
                bw[l] = pf[d][l];

            if (i + PF < myn) {
                #pragma unroll
                for (int l = 0; l < LOADS; ++l)
                    pf[d][l] = ld_b(i + PF, l);
            }

            const size_t a_col = (size_t) (ks0 + i) * 8 + (lane & 3);
            FragB a01, a23;
            a01[0] = r0_ok ? A2[a_row0 + a_col] : hzero;
            a23[0] = r0_ok ? A2[a_row0 + a_col + 4] : hzero;
            a01[1] = hzero;
            a23[1] = hzero;

            #pragma unroll
            for (int t = 0; t < WNT; ++t) {
                FragB f0, f1;
                if constexpr (bits == 4) {
                    uint32_t aw = __shfl_sync(0xffffffffu, bw[t], (lane + 31) & 31);
                    exl3_gemv_ns::dq8_regs_4bits<cb>(aw, bw[t], f0, f1);
                } else if constexpr (bits == 2) {
                    const uint32_t w = bw[t >> 1];
                    const int base = (t & 1) << 4;
                    uint32_t bwv = __shfl_sync(0xffffffffu, w, base + x_src_b);
                    uint32_t awv = __shfl_sync(0xffffffffu, w, base + x_src_a);
                    exl3_gemv_ns::dq8_regs_2bits<cb>(awv, bwv, lane << 3, f0, f1);
                } else {
                    uint32_t awv = __shfl_sync(0xffffffffu, bw[t], x_src_a);
                    uint32_t bwv = __shfl_sync(0xffffffffu, bw[t], x_src_b);
                    exl3_gemv_ns::dq8_regs_3bits<cb>(awv, bwv, x_s2, f0, f1);
                }

                exl3_gemv_ns::mma_ab_h(a01, a23, f0, ch[t][0]);
                exl3_gemv_ns::mma_ab_h(a01, a23, f1, ch[t][1]);
            }

            if ((d + 1) % FOLD == 0 || i + 1 == myn) {
                #pragma unroll
                for (int t = 0; t < WNT; ++t)
                    #pragma unroll
                    for (int f = 0; f < 2; ++f) {
                        acc0[t][f].x += __low2float(ch[t][f][0]);
                        acc0[t][f].y += __high2float(ch[t][f][0]);
                        ch[t][f][0] = hzero;
                    }
            }
        }
    }

    // Lanes 0..3 hold row 0: cols t*16 + f*8 + 2*(lane&3) (+1). No cross-warp
    // reduction: this warp covered the whole K range.
    if (lane < 4) {
        #pragma unroll
        for (int t = 0; t < WNT; ++t) {
            #pragma unroll
            for (int f = 0; f < 2; ++f) {
                const int col = group * COLS + t * 16 + f * 8 + (lane & 3) * 2;
                C[col + 0] = __float2half_rn(acc0[t][f].x);
                C[col + 1] = __float2half_rn(acc0[t][f].y);
            }
        }
    }
}

template <int BITS, int CFG>
__global__ __launch_bounds__(P2bCfg<CFG>::WK * 32, P2bCfg<CFG>::MINB)
void p2b_moe_batched_kernel(
    const half* __restrict__ x,
    const int64_t* __restrict__ gt_ptrs,
    const int64_t* __restrict__ gu_ptrs,
    const int64_t* __restrict__ gv_ptrs,
    const int64_t* __restrict__ ut_ptrs,
    const int64_t* __restrict__ uu_ptrs,
    const int64_t* __restrict__ uv_ptrs,
    const int64_t* __restrict__ dt_ptrs,
    const int64_t* __restrict__ du_ptrs,
    const int64_t* __restrict__ dv_ptrs,
    const int32_t* __restrict__ ids,
    const half* __restrict__ rw,
    half* __restrict__ gate,
    half* __restrict__ up,
    half* __restrict__ down,
    half* __restrict__ out,
    half* __restrict__ had_gate,
    half* __restrict__ had_up,
    half* __restrict__ had_down,
    float* __restrict__ accum,
    int experts,
    int m,
    int topk,
    int hidden,
    int inter)
{
    // `experts` counts (row, slot) pairs across the whole decode batch; pair e
    // belongs to input/output row e / topk.
    auto grid = cg::this_grid();
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;

    constexpr int COLS = P2bCfg<CFG>::WNT * 16;
    const int ntiles_gate = inter / 16;
    const int kslices_gate = hidden / 16;
    const int num_groups_gate = inter / COLS;

    const int ntiles_down = hidden / 16;
    const int kslices_down = inter / 16;
    const int num_groups_down = hidden / COLS;

    __shared__ float sh_red[P2bCfg<CFG>::WK * P2B_MAX_COLS];

    // Resolve the (pair -> expert -> tensor) pointer chase once per block so
    // the streaming loop does not start each work item with two dependent
    // global loads.
    __shared__ const half* s_gu[P2B_MAX_PAIRS];
    __shared__ const half* s_gv[P2B_MAX_PAIRS];
    __shared__ const half* s_uu[P2B_MAX_PAIRS];
    __shared__ const half* s_uv[P2B_MAX_PAIRS];
    __shared__ const half* s_du[P2B_MAX_PAIRS];
    __shared__ const half* s_dv[P2B_MAX_PAIRS];
    __shared__ const uint32_t* s_gt[P2B_MAX_PAIRS];
    __shared__ const uint32_t* s_ut[P2B_MAX_PAIRS];
    __shared__ const uint32_t* s_dt[P2B_MAX_PAIRS];
    for (int e = threadIdx.x; e < experts; e += blockDim.x) {
        const int src = ids[e];
        s_gt[e] = reinterpret_cast<const uint32_t*>(gt_ptrs[src]);
        s_gu[e] = reinterpret_cast<const half*>(gu_ptrs[src]);
        s_gv[e] = reinterpret_cast<const half*>(gv_ptrs[src]);
        s_ut[e] = reinterpret_cast<const uint32_t*>(ut_ptrs[src]);
        s_uu[e] = reinterpret_cast<const half*>(uu_ptrs[src]);
        s_uv[e] = reinterpret_cast<const half*>(uv_ptrs[src]);
        s_dt[e] = reinterpret_cast<const uint32_t*>(dt_ptrs[src]);
        s_du[e] = reinterpret_cast<const half*>(du_ptrs[src]);
        s_dv[e] = reinterpret_cast<const half*>(dv_ptrs[src]);
    }

    // Zero accum
    for (int j = tid; j < m * hidden; j += total_threads)
        accum[j] = 0.0f;
    __syncthreads();

    // Phase 1: Input Hadamard for Gate and Up across all active experts
    {
        int warps_per_exp = hidden / 128;
        int total_warps = experts * warps_per_exp;
        int this_warp = warp + (blockDim.x / 32) * blockIdx.x;
        int grid_warps = gridDim.x * (blockDim.x / 32);

        for (; this_warp < total_warps; this_warp += grid_warps) {
            int e = this_warp / warps_per_exp;
            int w = this_warp % warps_per_exp;
            const half* gu_e = s_gu[e];
            const half* uu_e = s_uu[e];
            half* hg_e = had_gate + e * hidden;
            half* hu_e = had_up + e * hidden;
            const half* x_e = x + (size_t) (e / topk) * hidden;

            had_hf_r_128_inner<true, false>(x_e + w * 128, hg_e + w * 128, gu_e + (w * 128) % hidden, 0.088388347648f);
            had_hf_r_128_inner<true, false>(x_e + w * 128, hu_e + w * 128, uu_e + (w * 128) % hidden, 0.088388347648f);
        }
        grid.sync();
    }

    // Phase 2: Batched Gate & Up GEMV across all active experts
    {
        int total_work = 2 * experts * num_groups_gate;
        if constexpr (P2bCfg<CFG>::WI) {
            const int gwarp = blockIdx.x * (blockDim.x / 32) + warp;
            const int gwarps = gridDim.x * (blockDim.x / 32);
            for (int item = gwarp; item < total_work; item += gwarps) {
                int is_up = item & 1;
                int rem = item >> 1;
                int e = rem / num_groups_gate;
                int group = rem % num_groups_gate;
                const uint32_t* B32 = is_up ? s_ut[e] : s_gt[e];
                const half2* A2 = reinterpret_cast<const half2*>((is_up ? had_up : had_gate) + e * hidden);
                half* C = (is_up ? up : gate) + e * inter;
                run_gemv_warp<BITS, 1, CFG>(B32, A2, C, kslices_gate, group, ntiles_gate, lane);
            }
        } else
        for (int item = blockIdx.x; item < total_work; item += gridDim.x) {
            int is_up = item & 1;
            int rem = item >> 1;
            int e = rem / num_groups_gate;
            int group = rem % num_groups_gate;

            const uint32_t* B32 = is_up ? s_ut[e] : s_gt[e];
            const half2* A2 = reinterpret_cast<const half2*>((is_up ? had_up : had_gate) + e * hidden);
            half* C = (is_up ? up : gate) + e * inter;

            run_gemv_tile<BITS, 1, CFG>(B32, A2, C, kslices_gate, hidden, group, ntiles_gate, warp, lane, sh_red);
        }
        grid.sync();
    }

    // Epilogue Hadamard on Gate and Up
    {
        int warps_per_exp = inter / 128;
        int total_warps = experts * warps_per_exp;
        int this_warp = warp + (blockDim.x / 32) * blockIdx.x;
        int grid_warps = gridDim.x * (blockDim.x / 32);

        for (; this_warp < total_warps; this_warp += grid_warps) {
            int e = this_warp / warps_per_exp;
            int w = this_warp % warps_per_exp;
            const half* gv_e = s_gv[e];
            const half* uv_e = s_uv[e];
            half* gp_e = gate + e * inter;
            half* up_e = up + e * inter;

            had_hf_r_128_inner<false, true>(gp_e + w * 128, gp_e + w * 128, gv_e + (w * 128) % inter, 0.088388347648f);
            had_hf_r_128_inner<false, true>(up_e + w * 128, up_e + w * 128, uv_e + (w * 128) % inter, 0.088388347648f);
        }
        grid.sync();
    }

    // Phase 3: SwiGLU activation + Down input Hadamard across all active experts
    {
        // First compute SwiGLU: had_down[e, j] = silu(gate[e, j]) * up[e, j]
        int total_elements = experts * inter;
        for (int j = tid; j < total_elements; j += total_threads) {
            float g = __half2float(gate[j]);
            float u = __half2float(up[j]);
            float s = g / (1.0f + expf(-g));
            had_down[j] = __float2half(s * u);
        }
        grid.sync();

        // Down input Hadamard on had_down
        int warps_per_exp = inter / 128;
        int total_warps = experts * warps_per_exp;
        int this_warp = warp + (blockDim.x / 32) * blockIdx.x;
        int grid_warps = gridDim.x * (blockDim.x / 32);

        for (; this_warp < total_warps; this_warp += grid_warps) {
            int e = this_warp / warps_per_exp;
            int w = this_warp % warps_per_exp;
            const half* du_e = s_du[e];
            half* hd_e = had_down + e * inter;

            had_hf_r_128_inner<true, false>(hd_e + w * 128, hd_e + w * 128, du_e + (w * 128) % inter, 0.088388347648f);
        }
        grid.sync();
    }

    // Phase 4: Batched Down GEMV across all active experts
    {
        int total_work = experts * num_groups_down;
        if constexpr (P2bCfg<CFG>::WI) {
            const int gwarp = blockIdx.x * (blockDim.x / 32) + warp;
            const int gwarps = gridDim.x * (blockDim.x / 32);
            for (int item = gwarp; item < total_work; item += gwarps) {
                int e = item / num_groups_down;
                int group = item % num_groups_down;
                const uint32_t* B32 = s_dt[e];
                const half2* A2 = reinterpret_cast<const half2*>(had_down + e * inter);
                half* C = down + e * hidden;
                run_gemv_warp<BITS, 1, CFG>(B32, A2, C, kslices_down, group, ntiles_down, lane);
            }
        } else
        for (int item = blockIdx.x; item < total_work; item += gridDim.x) {
            int e = item / num_groups_down;
            int group = item % num_groups_down;

            const uint32_t* B32 = s_dt[e];
            const half2* A2 = reinterpret_cast<const half2*>(had_down + e * inter);
            half* C = down + e * hidden;

            run_gemv_tile<BITS, 1, CFG>(B32, A2, C, kslices_down, inter, group, ntiles_down, warp, lane, sh_red);
        }
        grid.sync();
    }

    // Down output Hadamard and atomic accumulation into accum
    {
        int warps_per_exp = hidden / 128;
        int total_warps = experts * warps_per_exp;
        int this_warp = warp + (blockDim.x / 32) * blockIdx.x;
        int grid_warps = gridDim.x * (blockDim.x / 32);

        for (; this_warp < total_warps; this_warp += grid_warps) {
            int e = this_warp / warps_per_exp;
            int w = this_warp % warps_per_exp;
            const half* dv_e = s_dv[e];
            half* dp_e = down + e * hidden;

            had_hf_r_128_inner<false, true>(dp_e + w * 128, dp_e + w * 128, dv_e + (w * 128) % hidden, 0.088388347648f);
        }
        grid.sync();

        // Weighted reduction into accum
        int total_elements = experts * hidden;
        for (int j = tid; j < total_elements; j += total_threads) {
            int e = j / hidden;
            int col = j % hidden;
            float w = __half2float(rw[e]);
            atomicAdd(accum + (size_t) (e / topk) * hidden + col, w * __half2float(down[j]));
        }
        grid.sync();
    }

    // Write back to out
    for (int j = tid; j < m * hidden; j += total_threads) {
        out[j] = __float2half(accum[j]);
    }
}

template <int BITS, int CFG>
static void launch_moe_batched(
    const at::Tensor& x, const at::Tensor& gt, const at::Tensor& gu,
    const at::Tensor& gv, const at::Tensor& ut, const at::Tensor& uu,
    const at::Tensor& uv, const at::Tensor& dt, const at::Tensor& du,
    const at::Tensor& dv, const at::Tensor& ids, const at::Tensor& rw,
    at::Tensor& out, at::Tensor& gate, at::Tensor& up, at::Tensor& down,
    at::Tensor& had_gate, at::Tensor& had_up, at::Tensor& had_down,
    at::Tensor& accum, int e, int m, int topk, int hidden, int inter)
{
    constexpr int THREADS = P2bCfg<CFG>::WK * 32;
    int dev = 0, sms = 0, resident = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    void* kernel = (void*) p2b_moe_batched_kernel<BITS, CFG>;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, kernel, THREADS, 0);
    const int grid = std::max(1, resident * sms);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const half* xp = reinterpret_cast<const half*>(x.data_ptr<c10::Half>());
    const int64_t* gtp = gt.data_ptr<int64_t>();
    const int64_t* gup = gu.data_ptr<int64_t>();
    const int64_t* gvp = gv.data_ptr<int64_t>();
    const int64_t* utp = ut.data_ptr<int64_t>();
    const int64_t* uup = uu.data_ptr<int64_t>();
    const int64_t* uvp = uv.data_ptr<int64_t>();
    const int64_t* dtp = dt.data_ptr<int64_t>();
    const int64_t* dup = du.data_ptr<int64_t>();
    const int64_t* dvp = dv.data_ptr<int64_t>();
    const int32_t* idp = ids.data_ptr<int32_t>();
    const half* rwp = reinterpret_cast<const half*>(rw.data_ptr<c10::Half>());

    half* gp = reinterpret_cast<half*>(gate.data_ptr<c10::Half>());
    half* up_p = reinterpret_cast<half*>(up.data_ptr<c10::Half>());
    half* dp = reinterpret_cast<half*>(down.data_ptr<c10::Half>());
    half* op = reinterpret_cast<half*>(out.data_ptr<c10::Half>());
    half* hg_p = reinterpret_cast<half*>(had_gate.data_ptr<c10::Half>());
    half* hu_p = reinterpret_cast<half*>(had_up.data_ptr<c10::Half>());
    half* hd_p = reinterpret_cast<half*>(had_down.data_ptr<c10::Half>());
    float* accp = accum.data_ptr<float>();

    void* args[] = {
        (void*)&xp, (void*)&gtp, (void*)&gup, (void*)&gvp,
        (void*)&utp, (void*)&uup, (void*)&uvp,
        (void*)&dtp, (void*)&dup, (void*)&dvp,
        (void*)&idp, (void*)&rwp,
        (void*)&gp, (void*)&up_p, (void*)&dp, (void*)&op,
        (void*)&hg_p, (void*)&hu_p, (void*)&hd_p, (void*)&accp,
        (void*)&e, (void*)&m, (void*)&topk, (void*)&hidden, (void*)&inter
    };

    cuda_check(cudaLaunchCooperativeKernel(kernel, dim3(grid), dim3(THREADS), args, 0, stream));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

at::Tensor p2b_fused_moe_cuda(const at::Tensor& x, at::Tensor& out,
    const at::Tensor& gt, const at::Tensor& gu, const at::Tensor& gv,
    const at::Tensor& ut, const at::Tensor& uu, const at::Tensor& uv,
    const at::Tensor& dt, const at::Tensor& du, const at::Tensor& dv,
    const at::Tensor& ids, const at::Tensor& rw, int64_t kg, int64_t ku,
    int64_t kd, bool mcg) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf, "fused MoE requires CUDA fp16 input");
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kHalf, "fused MoE output must be CUDA fp16");
    TORCH_CHECK(mcg && kg == ku && ku == kd && (kg == 2 || kg == 3 || kg == 4), "unsupported fused MoE K");
    const int e = static_cast<int>(ids.numel());
    const int m = static_cast<int>(x.numel() / x.size(-1));
    constexpr int hidden = 4096, inter = 2048;
    TORCH_CHECK(m >= 1 && e >= m && e % m == 0,
                "fused MoE expects ids/weights flattened as (rows, topk); got ",
                e, " pairs for ", m, " rows");
    TORCH_CHECK(rw.numel() == e, "routing weights must match expert ids");
    TORCH_CHECK(x.size(-1) == hidden, "fused MoE is specialized to hidden=4096");
    const int topk = e / m;
    TORCH_CHECK(e <= P2B_MAX_PAIRS, "fused MoE supports at most ", P2B_MAX_PAIRS, " (row, slot) pairs per launch");

    // Per-(row, slot) pair scratch.
    auto gate = at::empty({e, inter}, x.options());
    auto up = at::empty({e, inter}, x.options());
    auto down = at::empty({e, hidden}, x.options());
    auto had_gate = at::empty({e, hidden}, x.options());
    auto had_up = at::empty({e, hidden}, x.options());
    auto had_down = at::empty({e, inter}, x.options());
    auto accum = at::zeros({m, hidden}, x.options().dtype(at::kFloat));

    // Tile configuration: default CFG 1 (best measured on GB10, 2026-09-04: 91 GB/s vs 84 for CFG 0);
    // VLLM_EXL3_P2B_CFG overrides (0-3) for benchmarking.
    static const int cfg = [] {
        const char* v = std::getenv("VLLM_EXL3_P2B_CFG");
        int c = v ? std::atoi(v) : 1;
        return (c >= 0 && c < P2B_NUM_CFG) ? c : 1;
    }();
    #define P2B_ARGS x, gt, gu, gv, ut, uu, uv, dt, du, dv, ids, rw, out, gate, up, down, had_gate, had_up, had_down, accum, e, m, topk, hidden, inter
    #define P2B_DISPATCH_K(K_) \
        switch (cfg) { \
            case 0: launch_moe_batched<K_, 0>(P2B_ARGS); break; \
            case 1: launch_moe_batched<K_, 1>(P2B_ARGS); break; \
            case 3: launch_moe_batched<K_, 3>(P2B_ARGS); break; \
            case 2: launch_moe_batched<K_, 2>(P2B_ARGS); break; \
            case 4: launch_moe_batched<K_, 4>(P2B_ARGS); break; \
            case 5: launch_moe_batched<K_, 5>(P2B_ARGS); break; \
            case 6: launch_moe_batched<K_, 6>(P2B_ARGS); break; \
            case 7: launch_moe_batched<K_, 7>(P2B_ARGS); break; \
            case 8: launch_moe_batched<K_, 8>(P2B_ARGS); break; \
            case 9: launch_moe_batched<K_, 9>(P2B_ARGS); break; \
            default: launch_moe_batched<K_, 1>(P2B_ARGS); break; \
        }
    if (kg == 2) { P2B_DISPATCH_K(2) }
    else if (kg == 3) { P2B_DISPATCH_K(3) }
    else if (kg == 4) { P2B_DISPATCH_K(4) }
    #undef P2B_DISPATCH_K
    #undef P2B_ARGS

    return out;
}
