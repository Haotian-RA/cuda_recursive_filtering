#ifndef STCR_KERNELS_CUH
#define STCR_KERNELS_CUH

// STCR (Single-stage Tridiagonal Cyclic Reduction) kernels for cascaded
// biquad IIR filtering.
//
// REQUIRES the following macros to be defined before inclusion:
//   N_SECTIONS   — number of cascaded biquads
//   BLOCK_SIZE   — threads per TB (32 or 64)
//   N_BLOCKS     — register-resident samples per thread
//   N_TB_PER_SM  — occupancy target for __launch_bounds__
//
// Exposes:
//   KERNEL_FUNC — one of six kernels, selected by BLOCK_SIZE, N_BLOCKS and
//   the STCR_HANDUNROLLED flag:
//     STCR_32_LOOP, STCR_64_LOOP        — generic for-loop back substitution
//     STCR_32_UNROLL_32 / _64 / _128    — literal-index hand-unrolled forms
//     STCR_64_UNROLL_64                 — literal-index hand-unrolled form
//   Loop and unrolled forms execute a bit-identical operation sequence
//   (validated in stcr_emulator.py); the unrolled forms are the spill-proof
//   fallback for compilers that mishandle the fully-unrolled loop.
//   setup_kernel_coefficients(sos) — copies SOS coefficients to constant memory
//
// Launch-versioned status flags (same protocol as ph_kernels.cuh):
//   Each kernel launch passes a monotonically increasing `launch` index
//   (0, 1, 2, ...). Chunk status values are launch-relative:
//     part_flag = 2*launch + 1   (partcarry published)
//     full_flag = 2*launch + 2   (fullcarry published)
//   Any value < part_flag (including every flag left by an earlier launch)
//   reads as "not ready", so the status array never needs to be reset
//   between launches. It must be zeroed ONCE after allocation (launch 0
//   expects values < 1), e.g. with a single cudaMemset at setup time.
//
//   Correctness assumptions (both hold in the provided test drivers):
//     1. Launches sharing a status buffer are serialized (same stream).
//     2. Every launch uses the same grid size and `launch` increments by
//        exactly 1 per launch, so chunk_id = ticket - launch*gridDim.x.

#include "iir_utils.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <array>
#include <cassert>

static const int order = 2;
static const int warp_size = 32;

#define N_BLOCKS_LOG2   __builtin_ctz(N_BLOCKS)
#define HALF_N_BLOCKS   (N_BLOCKS >> 1)
#define CHUNK_SIZE      (BLOCK_SIZE * N_BLOCKS)

// Thread-block dimensions for the <<<>>> launch (STCR uses 1D blocks).
#define KERNEL_TB_DIM BLOCK_SIZE

static __device__ unsigned int counter = 0;

// Per-section coefficients
static __constant__ T xi1[N_SECTIONS], xi2[N_SECTIONS], yi1[N_SECTIONS], yi2[N_SECTIONS];
static __constant__ T b1[N_SECTIONS], b2[N_SECTIONS], a1[N_SECTIONS], a2[N_SECTIONS];

// CR factors per section
static __constant__ T e[N_SECTIONS][N_BLOCKS_LOG2 + 1], f[N_SECTIONS][N_BLOCKS_LOG2 + 1];
static __constant__ T fde[N_SECTIONS][N_BLOCKS_LOG2 + 1];
static __constant__ T h0[N_SECTIONS][BLOCK_SIZE];
static __constant__ T C_cross[N_SECTIONS][BLOCK_SIZE][4];

static __device__ T hb2[N_SECTIONS][BLOCK_SIZE], hb1[N_SECTIONS][BLOCK_SIZE];
static __device__ T he2[N_SECTIONS][BLOCK_SIZE], he1[N_SECTIONS][BLOCK_SIZE];

static __constant__ T cr_h[N_SECTIONS][N_BLOCKS_LOG2 + 1], cr_g[N_SECTIONS][N_BLOCKS_LOG2 + 1];
static __constant__ T cr_p[N_SECTIONS][N_BLOCKS_LOG2 + 1], cr_q[N_SECTIONS][N_BLOCKS_LOG2 + 1];
static __constant__ T cr_d[N_SECTIONS][N_BLOCKS_LOG2 + 1], cr_c[N_SECTIONS][N_BLOCKS_LOG2 + 1];


// ==========================================================================
// Host-side helpers
// ==========================================================================
inline void impulse_response(T h_a1, T h_a2, T h_b1, T h_b2, T* h_h1, T* h_h2) {
    T h0[N_BLOCKS + 1];

    h0[0] = 1.0f;
    h0[1] = h_a1;

    for (int n = 2; n < N_BLOCKS + 1; n++)
        h0[n] = h_a1 * h0[n-1] + h_a2 * h0[n-2];

    for (int n = 0; n < N_BLOCKS; n++) {
        h_h2[n] = h_a2 * h0[n];
        h_h1[n] = h0[n + 1];
    }
}

inline void gaussian_elimination_factors(T h_a1, T h_a2, T* h_f, T* h_e, T* h_fde,
        T* h_h, T* h_g, T* h_p, T* h_q, T* h_d, T* h_c, T h_b1, T h_b2, T* h_dc) {

    h_f[0] = - h_a2;
    h_e[0] = - h_a1;
    h_fde[0] = h_f[0]/h_e[0];
    h_h[0] = h_f[0];
    h_g[0] = h_e[0];
    h_p[1] = h_f[0];
    h_q[1] = h_e[0];

    for (int n = 1; n < N_BLOCKS_LOG2 + 1; n++) {
        h_f[n] = h_f[n-1]*h_f[n-1];
        h_e[n] = 2*h_f[n-1] - h_e[n-1]*h_e[n-1];
        h_fde[n] = h_f[n]/h_e[n];
        h_h[n] = -h_e[n-1]*h_h[n-1];
        h_g[n] = h_f[n-1] - h_e[n-1]*h_g[n-1];
        h_d[n] = -h_f[n-1]*h_f[n-1]/h_e[n-1];
        h_c[n] = h_e[n-1] - h_f[n-1]/h_e[n-1];

        if (n > 1) {
            h_q[n] = -h_e[n-1]*h_q[n-1];
            h_p[n] = h_f[n-1] - h_e[n-1]*h_p[n-1];
        }
    }

    h_dc[0] = h_f[0]*h_b1 - h_e[0]*h_b2;
    h_dc[1] = h_b1 - h_e[0];
    h_dc[2] = h_b2*h_f[0];
    h_dc[3] = h_b2 + h_f[0] - h_b1*h_e[0];
    h_dc[4] = h_b2 - h_e[0]*h_b1;
    h_dc[5] = -h_e[0]*h_b2;
}

inline void block_filtering_factors(T* h_f, T* h_e, T* h_h, T* h_g, T* h_p, T* h_q,
    T* h_h0, T* h_hb2, T* h_hb1, T* h_he2, T* h_he1) {

    h_h0[0] = 1;
    h_h0[1] = -h_e[N_BLOCKS_LOG2];

    for (int l = 2; l < BLOCK_SIZE; l++)
        h_h0[l] = -h_e[N_BLOCKS_LOG2] * h_h0[l-1] - h_f[N_BLOCKS_LOG2] * h_h0[l-2];

    T tmp;
    for (int l = 0; l < BLOCK_SIZE; l++) {
        h_hb2[l] = h_h[N_BLOCKS_LOG2] * h_h0[l];
        h_he1[l] = h_q[N_BLOCKS_LOG2] * h_h0[l];
        if (l == 0)
            tmp = 0.0;
        else
            tmp = h_h0[l-1];
        h_hb1[l] = h_f[N_BLOCKS_LOG2] * tmp + h_g[N_BLOCKS_LOG2] * h_h0[l];
        h_he2[l] = h_f[N_BLOCKS_LOG2] * tmp + h_p[N_BLOCKS_LOG2] * h_h0[l];
    }
}

inline void C_power(const T* C_in, int n, T* result) {
    result[0] = 1.0f; result[1] = 0.0f;
    result[2] = 0.0f; result[3] = 1.0f;

    if (n == 0) return;
    if (n == 1) {
        result[0] = C_in[0]; result[1] = C_in[1];
        result[2] = C_in[2]; result[3] = C_in[3];
        return;
    }

    T base[4] = {C_in[0], C_in[1], C_in[2], C_in[3]};

    while (n > 0) {
        if (n & 1) {
            T temp[4];
            temp[0] = result[0] * base[0] + result[1] * base[2];
            temp[1] = result[0] * base[1] + result[1] * base[3];
            temp[2] = result[2] * base[0] + result[3] * base[2];
            temp[3] = result[2] * base[1] + result[3] * base[3];

            result[0] = temp[0]; result[1] = temp[1];
            result[2] = temp[2]; result[3] = temp[3];
        }

        T temp[4];
        temp[0] = base[0] * base[0] + base[1] * base[2];
        temp[1] = base[0] * base[1] + base[1] * base[3];
        temp[2] = base[2] * base[0] + base[3] * base[2];
        temp[3] = base[2] * base[1] + base[3] * base[3];

        base[0] = temp[0]; base[1] = temp[1];
        base[2] = temp[2]; base[3] = temp[3];

        n >>= 1;
    }
}


// ==========================================================================
// Coefficient setup: convert SOS -> constant memory
// ==========================================================================
inline void setup_kernel_coefficients(const std::vector<std::array<T, 6>>& sos) {
    T h_b1[N_SECTIONS], h_b2[N_SECTIONS];
    T h_a1[N_SECTIONS], h_a2[N_SECTIONS];
    T h_xi1[N_SECTIONS], h_xi2[N_SECTIONS];
    T h_yi1[N_SECTIONS], h_yi2[N_SECTIONS];

    for (int sec = 0; sec < N_SECTIONS; sec++) {
        h_b1[sec] = sos[sec][1];
        h_b2[sec] = sos[sec][2];
        h_a1[sec] = -sos[sec][4];
        h_a2[sec] = -sos[sec][5];
        h_xi1[sec] = 0.0f;
        h_xi2[sec] = 0.0f;
        h_yi1[sec] = 0.0f;
        h_yi2[sec] = 0.0f;
    }

    // STCR-specific precomputation
    T h_e[N_SECTIONS][N_BLOCKS_LOG2 + 1], h_f[N_SECTIONS][N_BLOCKS_LOG2 + 1];
    T h_fde[N_SECTIONS][N_BLOCKS_LOG2 + 1];
    T h_h[N_SECTIONS][N_BLOCKS_LOG2 + 1], h_g[N_SECTIONS][N_BLOCKS_LOG2 + 1];
    T h_p[N_SECTIONS][N_BLOCKS_LOG2 + 1], h_q[N_SECTIONS][N_BLOCKS_LOG2 + 1];
    T h_d[N_SECTIONS][N_BLOCKS_LOG2 + 1], h_c[N_SECTIONS][N_BLOCKS_LOG2 + 1];
    T h_dc[N_SECTIONS][6];

    T h_h0[N_SECTIONS][BLOCK_SIZE];
    T h_hb2[N_SECTIONS][BLOCK_SIZE], h_hb1[N_SECTIONS][BLOCK_SIZE];
    T h_he2[N_SECTIONS][BLOCK_SIZE], h_he1[N_SECTIONS][BLOCK_SIZE];

    T h_C_cross[N_SECTIONS][BLOCK_SIZE][4];
    T h_h1[N_SECTIONS][N_BLOCKS], h_h2[N_SECTIONS][N_BLOCKS];

    for (int sec = 0; sec < N_SECTIONS; sec++) {
        gaussian_elimination_factors(h_a1[sec], h_a2[sec],
            h_f[sec], h_e[sec], h_fde[sec],
            h_h[sec], h_g[sec], h_p[sec], h_q[sec],
            h_d[sec], h_c[sec], h_b1[sec], h_b2[sec], h_dc[sec]);

        block_filtering_factors(h_f[sec], h_e[sec], h_h[sec], h_g[sec], h_p[sec], h_q[sec],
            h_h0[sec], h_hb2[sec], h_hb1[sec], h_he2[sec], h_he1[sec]);

        impulse_response(h_a1[sec], h_a2[sec], h_b1[sec], h_b2[sec],
                         h_h1[sec], h_h2[sec]);

        T h_C[4] = {h_h2[sec][N_BLOCKS - 2], h_h1[sec][N_BLOCKS - 2],
                    h_h2[sec][N_BLOCKS - 1], h_h1[sec][N_BLOCKS - 1]};

        for (int n = 0; n < BLOCK_SIZE; n++)
            C_power(h_C, n + 1, h_C_cross[sec][n]);
    }

    // Copy basic coefficients
    assert(cudaSuccess == cudaMemcpyToSymbol(b1,  h_b1,  N_SECTIONS * sizeof(T)));
    assert(cudaSuccess == cudaMemcpyToSymbol(b2,  h_b2,  N_SECTIONS * sizeof(T)));
    assert(cudaSuccess == cudaMemcpyToSymbol(a1,  h_a1,  N_SECTIONS * sizeof(T)));
    assert(cudaSuccess == cudaMemcpyToSymbol(a2,  h_a2,  N_SECTIONS * sizeof(T)));
    assert(cudaSuccess == cudaMemcpyToSymbol(xi1, h_xi1, N_SECTIONS * sizeof(T)));
    assert(cudaSuccess == cudaMemcpyToSymbol(xi2, h_xi2, N_SECTIONS * sizeof(T)));
    assert(cudaSuccess == cudaMemcpyToSymbol(yi1, h_yi1, N_SECTIONS * sizeof(T)));
    assert(cudaSuccess == cudaMemcpyToSymbol(yi2, h_yi2, N_SECTIONS * sizeof(T)));

    // Copy CR factors
    assert(cudaSuccess == cudaMemcpyToSymbol(e,   h_e,   N_SECTIONS * (N_BLOCKS_LOG2 + 1) * sizeof(T)));
    assert(cudaSuccess == cudaMemcpyToSymbol(f,   h_f,   N_SECTIONS * (N_BLOCKS_LOG2 + 1) * sizeof(T)));
    assert(cudaSuccess == cudaMemcpyToSymbol(fde, h_fde, N_SECTIONS * (N_BLOCKS_LOG2 + 1) * sizeof(T)));
    assert(cudaSuccess == cudaMemcpyToSymbol(cr_h, h_h,  N_SECTIONS * (N_BLOCKS_LOG2 + 1) * sizeof(T)));
    assert(cudaSuccess == cudaMemcpyToSymbol(cr_g, h_g,  N_SECTIONS * (N_BLOCKS_LOG2 + 1) * sizeof(T)));
    assert(cudaSuccess == cudaMemcpyToSymbol(cr_p, h_p,  N_SECTIONS * (N_BLOCKS_LOG2 + 1) * sizeof(T)));
    assert(cudaSuccess == cudaMemcpyToSymbol(cr_q, h_q,  N_SECTIONS * (N_BLOCKS_LOG2 + 1) * sizeof(T)));
    assert(cudaSuccess == cudaMemcpyToSymbol(cr_d, h_d,  N_SECTIONS * (N_BLOCKS_LOG2 + 1) * sizeof(T)));
    assert(cudaSuccess == cudaMemcpyToSymbol(cr_c, h_c,  N_SECTIONS * (N_BLOCKS_LOG2 + 1) * sizeof(T)));

    // Copy block filtering factors
    assert(cudaSuccess == cudaMemcpyToSymbol(h0,  h_h0,  N_SECTIONS * BLOCK_SIZE * sizeof(T)));
    assert(cudaSuccess == cudaMemcpyToSymbol(hb2, h_hb2, N_SECTIONS * BLOCK_SIZE * sizeof(T)));
    assert(cudaSuccess == cudaMemcpyToSymbol(hb1, h_hb1, N_SECTIONS * BLOCK_SIZE * sizeof(T)));
    assert(cudaSuccess == cudaMemcpyToSymbol(he2, h_he2, N_SECTIONS * BLOCK_SIZE * sizeof(T)));
    assert(cudaSuccess == cudaMemcpyToSymbol(he1, h_he1, N_SECTIONS * BLOCK_SIZE * sizeof(T)));

    // Copy C_cross for lookback
    assert(cudaSuccess == cudaMemcpyToSymbol(C_cross, h_C_cross, N_SECTIONS * BLOCK_SIZE * 4 * sizeof(T)));
}


// ==========================================================================
// STCR kernels — verbatim from the original main_STCR.cu
// ==========================================================================

static __global__ __launch_bounds__(BLOCK_SIZE, N_TB_PER_SM)
void STCR_32_LOOP(const T* const __restrict__ input,
        T* const __restrict__ output,
        volatile int* const __restrict__ status,
        volatile T* const __restrict__ partcarry,
        volatile T* const __restrict__ fullcarry,
        const int launch)
{
    __shared__ T in[BLOCK_SIZE][N_BLOCKS + 1];
    __shared__ int cid;
    __shared__ T spartc[CHUNK_SIZE / warp_size * order];

    const int tid = threadIdx.x;
    const int lane = tid;

    // Launch-relative status values (see header comment).
    const int part_flag = 2 * launch + 1;
    const int full_flag = part_flag + 1;

    if (tid == 0) {
        cid = (int)(atomicAdd(&counter, 1u) - (unsigned int)launch * gridDim.x);
    }

    __syncwarp();

    const int chunk_id = cid;
    const int chunk_start = chunk_id * CHUNK_SIZE;

    for (int i = tid; i < CHUNK_SIZE; i += BLOCK_SIZE) {
        int global_idx = chunk_start + i;
        int tx = i / N_BLOCKS;
        int bx = i % N_BLOCKS;
        in[tx][bx] = input[global_idx];
    }

    T in_reg[N_BLOCKS];
    T _xc, _xi1, _xi2, _yc, _yi1, _yi2;

    __syncwarp();

    for (int sec = 0; sec < N_SECTIONS; sec++) {

        // Phase 1: FIR + CR
        if (tid == 0) {
            _yi2 = 0;
            _yi1 = 0;
            if (chunk_id == 0) {
                _xi2 = xi2[sec];
                _xi1 = xi1[sec];
            } else if (sec == 0) {
                _xi2 = input[chunk_start - 2];
                _xi1 = input[chunk_start - 1];
            }
        } else {
            if (sec == 0) {
                _xi2 = in[tid - 1][N_BLOCKS - 2];
                _xi1 = in[tid - 1][N_BLOCKS - 1];
                _yi2 = in[tid - 1][N_BLOCKS - 4];
                _yi1 = in[tid - 1][N_BLOCKS - 3];
            }

            _yi2 = fmaf(b1[sec], _yi1, fmaf(b2[sec], _yi2, _xi2));
            _yi1 = fmaf(b1[sec], _xi2, fmaf(b2[sec], _yi1, _xi1));
        }

        #pragma unroll
        for (int n = 0; n < N_BLOCKS; n++) {

            if (sec == 0)
                _xc = in[tid][n];
            else
                _xc = in_reg[n];

            // FIR
            _yc = fmaf(b1[sec], _xi1, fmaf(b2[sec], _xi2, _xc));

            // First round CR (double sided)
            in_reg[n] = fmaf(-e[sec][0], _yi1, fmaf(f[sec][0], _yi2, _yc));

            _xi2 = _xi1;
            _xi1 = _xc;
            _yi2 = _yi1;
            _yi1 = _yc;
        }

        // Remaining CR rounds
        #pragma unroll
        for (int ro = 1; ro < N_BLOCKS_LOG2; ro++) {

            const int step = 2 << ro;
            const int sub = 1 << ro;

            #pragma unroll
            for (int i = 0; i < 2; i++) {
                const int off = sub - 2 + i;

                _xi2 = __shfl_up_sync(0xffffffff, in_reg[N_BLOCKS - 2 + i], 1);
                if (tid == 0) _xi2 = 0;

                #pragma unroll
                for (int n = 0; n < N_BLOCKS; n += step) {
                    in_reg[n + off] = fmaf(-fde[sec][ro], _xi2, in_reg[n + off]);

                    _xi2 = in_reg[n + off + sub];
                    in_reg[n + off + sub] = fmaf(-e[sec][ro], in_reg[n + off], _xi2);
                }
            }
        }

        // Block filtering
        _yi2 = h0[sec][0] * in_reg[N_BLOCKS - 2];
        _yi1 = h0[sec][0] * in_reg[N_BLOCKS - 1];

        #pragma unroll
        for (int n = 1; n < BLOCK_SIZE; n++) {

            _xc = __shfl_up_sync(0xffffffff, in_reg[N_BLOCKS - 2], n);
            _yc = __shfl_up_sync(0xffffffff, in_reg[N_BLOCKS - 1], n);

            if (tid < n) {
                _xc = 0;
                _yc = 0;
            }

            _yi2 = fmaf(h0[sec][n], _xc, _yi2);
            _yi1 = fmaf(h0[sec][n], _yc, _yi1);
        }

        // Phase 2: Lookback
        if (lane == warp_size - 1) {
            partcarry[chunk_id * order * N_SECTIONS + sec * order] = _yi2;
            partcarry[chunk_id * order * N_SECTIONS + sec * order + 1] = _yi1;
        }

        T _h2_e, _h1_e, _h2_o, _h1_o;

        __syncwarp();
        __threadfence();
        if (tid == BLOCK_SIZE - 1)
            status[chunk_id * N_SECTIONS + sec] = part_flag;

        int flag = part_flag;
        bool no_zeros, has_status_2, reached_origin;

        do {
            if (chunk_id > lane) {
                flag = status[(chunk_id - 1 - lane) * N_SECTIONS + sec];
            }
            no_zeros = !__any_sync(0xffffffff, flag < part_flag);
            has_status_2 = !__all_sync(0xffffffff, flag != full_flag) && no_zeros;
            reached_origin = (chunk_id < warp_size) && no_zeros;
        } while (!(has_status_2 || reached_origin));

        __threadfence();

        int mask = __ballot_sync(0xffffffff, flag == full_flag);

        T X0, X1;
        int start_chunk;

        if (mask == 0) {
            X0 = yi2[sec];
            X1 = yi1[sec];
            start_chunk = 0;
        } else {
            const int pos = __ffs(mask) - 1;
            const int full_chunk = chunk_id - 1 - pos;
            start_chunk = full_chunk + 1;

            T fc;
            if (lane < order) {
                fc = fullcarry[full_chunk * order * N_SECTIONS + sec * order + lane];
            }
            X0 = __shfl_sync(0xffffffff, fc, 0);
            X1 = __shfl_sync(0xffffffff, fc, 1);
        }

        const int num_partcarries = chunk_id - start_chunk;

        if (num_partcarries > 0) {
            for (int i = (start_chunk * order + lane); i < (chunk_id * order); i += warp_size) {
                spartc[i - start_chunk * order] = partcarry[(i / order) * order * N_SECTIONS + sec * order + (i % order)];
            }
            __syncwarp();

            _h2_e = C_cross[sec][BLOCK_SIZE - 1][0];
            _h1_e = C_cross[sec][BLOCK_SIZE - 1][1];
            _h2_o = C_cross[sec][BLOCK_SIZE - 1][2];
            _h1_o = C_cross[sec][BLOCK_SIZE - 1][3];

            for (int i = 0; i < num_partcarries; i++) {
                const T p0 = spartc[i * order];
                const T p1 = spartc[i * order + 1];
                const T h0_val = fmaf(_h1_e, X1, fmaf(_h2_e, X0, p0));
                const T h1_val = fmaf(_h1_o, X1, fmaf(_h2_o, X0, p1));
                X0 = h0_val;
                X1 = h1_val;
            }
        }

        // phase 3: back substitution
        in_reg[N_BLOCKS - 2] = fmaf(-he1[sec][tid], X1, fmaf(-he2[sec][tid], X0, _yi2));
        in_reg[N_BLOCKS - 1] = fmaf(-hb1[sec][tid], X1, fmaf(-hb2[sec][tid], X0, _yi1));

        if (tid == BLOCK_SIZE - 1) {
            fullcarry[chunk_id * order * N_SECTIONS + sec * order] = in_reg[N_BLOCKS - 2];
            fullcarry[chunk_id * order * N_SECTIONS + sec * order + 1] = in_reg[N_BLOCKS - 1];
        }
        __syncwarp();
        __threadfence();
        if (tid == BLOCK_SIZE - 1) {
            status[chunk_id * N_SECTIONS + sec] = full_flag;
        }

        T _yi2_e = __shfl_up_sync(0xffffffff, in_reg[N_BLOCKS - 2], 2);
        T _yi2_o = __shfl_up_sync(0xffffffff, in_reg[N_BLOCKS - 1], 1);
        T _yi1_e = __shfl_up_sync(0xffffffff, in_reg[N_BLOCKS - 2], 1);
        T _yi1_o = __shfl_up_sync(0xffffffff, in_reg[N_BLOCKS - 1], 2);

        if (tid == 0){
            _h2_e = cr_p[sec][N_BLOCKS_LOG2 - 1];
            _h2_o = cr_h[sec][N_BLOCKS_LOG2 - 1];
            _h1_e = cr_q[sec][N_BLOCKS_LOG2 - 1];
            _h1_o = cr_g[sec][N_BLOCKS_LOG2 - 1];
            _yi2_e = X0;
            _yi2_o = X0;
            _yi1_e = X1;
            _yi1_o = X1;
            _xi2 = X0;
            _xi1 = X1;
        } else {
            _h2_e = cr_d[sec][N_BLOCKS_LOG2];
            _h2_o = cr_c[sec][N_BLOCKS_LOG2];
            _h1_e = cr_c[sec][N_BLOCKS_LOG2];
            _h1_o = cr_d[sec][N_BLOCKS_LOG2];
            if (tid == 1) {
                _yi2_e = X0;
                _yi1_o = X1;
            }
            _xi2 = _yi1_e;
            _xi1 = _yi2_o;
        }

        in_reg[HALF_N_BLOCKS - 2] = fmaf(-_h1_e, _yi1_e, fmaf(-_h2_e, _yi2_e, in_reg[HALF_N_BLOCKS - 2]));
        in_reg[HALF_N_BLOCKS - 1] = fmaf(-_h1_o, _yi1_o, fmaf(-_h2_o, _yi2_o, in_reg[HALF_N_BLOCKS - 1]));

        T _zi2_e, _zi1_e, _zi2_o, _zi1_o;

        // Generic back-substitution rounds (for-loop form). All loop bounds and
        // indices are compile-time constants after full unrolling, so in_reg[]
        // stays register-resident. Bit-identical operation sequence to the
        // hand-unrolled variants (validated in stcr_emulator.py).
        #pragma unroll
        for (int r = N_BLOCKS_LOG2 - 2; r > 0; r--) {

            const int stride = 2 << r;
            const int sub = stride >> 1;

            // n = 0
            _yi2_e = __shfl_up_sync(0xffffffff, in_reg[N_BLOCKS - stride - 2], 1);
            _yi1_o = __shfl_up_sync(0xffffffff, in_reg[N_BLOCKS - stride - 1], 1);
            if (tid == 0) {
                _h2_o = cr_h[sec][r];
                _h1_o = cr_g[sec][r];
                _h2_e = cr_p[sec][r];
                _h1_e = cr_q[sec][r];
                _yi2_e = X0;
                _yi1_o = X1;
            } else {
                _h2_o = cr_c[sec][r + 1];
                _h1_o = cr_d[sec][r + 1];
                _h2_e = _h1_o;
                _h1_e = _h2_o;
            }
            _zi2_o = _yi2_o;
            _zi1_o = _yi1_o;
            _zi2_e = _yi2_e;
            _zi1_e = _yi1_e;

            in_reg[sub - 2] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[sub - 2]));
            in_reg[sub - 1] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[sub - 1]));

            // n = 1
            _h2_o = cr_d[sec][r + 1];
            _h1_o = cr_c[sec][r + 1];
            _h2_e = _h2_o;
            _h1_e = _h1_o;
            _zi2_o = _xi1;
            _zi1_o = in_reg[stride - 1];
            _zi2_e = _xi2;
            _zi1_e = in_reg[stride - 2];

            in_reg[stride + sub - 2] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[stride + sub - 2]));
            in_reg[stride + sub - 1] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[stride + sub - 1]));

            // n = 2 .. P-1
            #pragma unroll
            for (int n = 2; n < (HALF_N_BLOCKS >> r); n++) {
                _zi2_o = _zi1_o;
                _zi1_o = in_reg[stride * n - 1];
                _zi2_e = _zi1_e;
                _zi1_e = in_reg[stride * n - 2];

                in_reg[stride * n + sub - 2] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[stride * n + sub - 2]));
                in_reg[stride * n + sub - 1] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[stride * n + sub - 1]));
            }
        }

        if (sec == N_SECTIONS - 1) {
            #pragma unroll
            for (int n = 0; n < N_BLOCKS; n++)
                in[tid][n] = in_reg[n];
        } else {
            _yi2 = __shfl_up_sync(0xffffffff, in_reg[N_BLOCKS - 4], 1);
            _yi1 = __shfl_up_sync(0xffffffff, in_reg[N_BLOCKS - 3], 1);
        }
    }

    __syncwarp();

    for (int i = tid; i < CHUNK_SIZE; i += BLOCK_SIZE) {
        int global_idx = chunk_start + i;
        int tx = i / N_BLOCKS;
        int bx = i % N_BLOCKS;
        output[global_idx] = in[tx][bx];
    }
}


#if N_BLOCKS == 32
static __global__ __launch_bounds__(BLOCK_SIZE, N_TB_PER_SM)
void STCR_32_UNROLL_32(const T* const __restrict__ input,
        T* const __restrict__ output,
        volatile int* const __restrict__ status,
        volatile T* const __restrict__ partcarry,
        volatile T* const __restrict__ fullcarry,
        const int launch)
{
    __shared__ T in[BLOCK_SIZE][N_BLOCKS + 1];
    __shared__ int cid;
    __shared__ T spartc[CHUNK_SIZE / warp_size * order];

    const int tid = threadIdx.x;
    const int lane = tid;

    // Launch-relative status values (see header comment).
    const int part_flag = 2 * launch + 1;
    const int full_flag = part_flag + 1;

    if (tid == 0) {
        cid = (int)(atomicAdd(&counter, 1u) - (unsigned int)launch * gridDim.x);
    }

    __syncwarp();

    const int chunk_id = cid;
    const int chunk_start = chunk_id * CHUNK_SIZE;

    for (int i = tid; i < CHUNK_SIZE; i += BLOCK_SIZE) {
        int global_idx = chunk_start + i;
        int tx = i / N_BLOCKS;
        int bx = i % N_BLOCKS;
        in[tx][bx] = input[global_idx];
    }

    T in_reg[N_BLOCKS];
    T _xc, _xi1, _xi2, _yc, _yi1, _yi2;

    __syncwarp();

    for (int sec = 0; sec < N_SECTIONS; sec++) {

        // Phase 1: FIR + CR
        if (tid == 0) {
            _yi2 = 0;
            _yi1 = 0;
            if (chunk_id == 0) {
                _xi2 = xi2[sec];
                _xi1 = xi1[sec];
            } else if (sec == 0) {
                _xi2 = input[chunk_start - 2];
                _xi1 = input[chunk_start - 1];
            }
        } else {
            if (sec == 0) {
                _xi2 = in[tid - 1][N_BLOCKS - 2];
                _xi1 = in[tid - 1][N_BLOCKS - 1];
                _yi2 = in[tid - 1][N_BLOCKS - 4];
                _yi1 = in[tid - 1][N_BLOCKS - 3];
            }

            _yi2 = fmaf(b1[sec], _yi1, fmaf(b2[sec], _yi2, _xi2));
            _yi1 = fmaf(b1[sec], _xi2, fmaf(b2[sec], _yi1, _xi1));
        }

        #pragma unroll
        for (int n = 0; n < N_BLOCKS; n++) {

            if (sec == 0)
                _xc = in[tid][n];
            else
                _xc = in_reg[n];

            // FIR
            _yc = fmaf(b1[sec], _xi1, fmaf(b2[sec], _xi2, _xc));

            // First round CR (double sided)
            in_reg[n] = fmaf(-e[sec][0], _yi1, fmaf(f[sec][0], _yi2, _yc));

            _xi2 = _xi1;
            _xi1 = _xc;
            _yi2 = _yi1;
            _yi1 = _yc;
        }

        // Remaining CR rounds
        #pragma unroll
        for (int ro = 1; ro < N_BLOCKS_LOG2; ro++) {

            const int step = 2 << ro;
            const int sub = 1 << ro;

            #pragma unroll
            for (int i = 0; i < 2; i++) {
                const int off = sub - 2 + i;

                _xi2 = __shfl_up_sync(0xffffffff, in_reg[N_BLOCKS - 2 + i], 1);
                if (tid == 0) _xi2 = 0;

                #pragma unroll
                for (int n = 0; n < N_BLOCKS; n += step) {
                    in_reg[n + off] = fmaf(-fde[sec][ro], _xi2, in_reg[n + off]);

                    _xi2 = in_reg[n + off + sub];
                    in_reg[n + off + sub] = fmaf(-e[sec][ro], in_reg[n + off], _xi2);
                }
            }
        }

        // Block filtering
        _yi2 = h0[sec][0] * in_reg[N_BLOCKS - 2];
        _yi1 = h0[sec][0] * in_reg[N_BLOCKS - 1];

        #pragma unroll
        for (int n = 1; n < BLOCK_SIZE; n++) {

            _xc = __shfl_up_sync(0xffffffff, in_reg[N_BLOCKS - 2], n);
            _yc = __shfl_up_sync(0xffffffff, in_reg[N_BLOCKS - 1], n);

            if (tid < n) {
                _xc = 0;
                _yc = 0;
            }

            _yi2 = fmaf(h0[sec][n], _xc, _yi2);
            _yi1 = fmaf(h0[sec][n], _yc, _yi1);
        }

        // Phase 2: Lookback
        if (lane == warp_size - 1) {
            partcarry[chunk_id * order * N_SECTIONS + sec * order] = _yi2;
            partcarry[chunk_id * order * N_SECTIONS + sec * order + 1] = _yi1;
        }

        T _h2_e, _h1_e, _h2_o, _h1_o;

        __syncwarp();
        __threadfence();
        if (tid == BLOCK_SIZE - 1)
            status[chunk_id * N_SECTIONS + sec] = part_flag;

        int flag = part_flag;
        bool no_zeros, has_status_2, reached_origin;

        do {
            if (chunk_id > lane) {
                flag = status[(chunk_id - 1 - lane) * N_SECTIONS + sec];
            }
            no_zeros = !__any_sync(0xffffffff, flag < part_flag);
            has_status_2 = !__all_sync(0xffffffff, flag != full_flag) && no_zeros;
            reached_origin = (chunk_id < warp_size) && no_zeros;
        } while (!(has_status_2 || reached_origin));

        __threadfence();

        int mask = __ballot_sync(0xffffffff, flag == full_flag);

        T X0, X1;
        int start_chunk;

        if (mask == 0) {
            X0 = yi2[sec];
            X1 = yi1[sec];
            start_chunk = 0;
        } else {
            const int pos = __ffs(mask) - 1;
            const int full_chunk = chunk_id - 1 - pos;
            start_chunk = full_chunk + 1;

            T fc;
            if (lane < order) {
                fc = fullcarry[full_chunk * order * N_SECTIONS + sec * order + lane];
            }
            X0 = __shfl_sync(0xffffffff, fc, 0);
            X1 = __shfl_sync(0xffffffff, fc, 1);
        }

        const int num_partcarries = chunk_id - start_chunk;

        if (num_partcarries > 0) {
            for (int i = (start_chunk * order + lane); i < (chunk_id * order); i += warp_size) {
                spartc[i - start_chunk * order] = partcarry[(i / order) * order * N_SECTIONS + sec * order + (i % order)];
            }
            __syncwarp();

            _h2_e = C_cross[sec][BLOCK_SIZE - 1][0];
            _h1_e = C_cross[sec][BLOCK_SIZE - 1][1];
            _h2_o = C_cross[sec][BLOCK_SIZE - 1][2];
            _h1_o = C_cross[sec][BLOCK_SIZE - 1][3];

            for (int i = 0; i < num_partcarries; i++) {
                const T p0 = spartc[i * order];
                const T p1 = spartc[i * order + 1];
                const T h0_val = fmaf(_h1_e, X1, fmaf(_h2_e, X0, p0));
                const T h1_val = fmaf(_h1_o, X1, fmaf(_h2_o, X0, p1));
                X0 = h0_val;
                X1 = h1_val;
            }
        }

        // phase 3: back substitution
        in_reg[N_BLOCKS - 2] = fmaf(-he1[sec][tid], X1, fmaf(-he2[sec][tid], X0, _yi2));
        in_reg[N_BLOCKS - 1] = fmaf(-hb1[sec][tid], X1, fmaf(-hb2[sec][tid], X0, _yi1));

        if (tid == BLOCK_SIZE - 1) {
            fullcarry[chunk_id * order * N_SECTIONS + sec * order] = in_reg[N_BLOCKS - 2];
            fullcarry[chunk_id * order * N_SECTIONS + sec * order + 1] = in_reg[N_BLOCKS - 1];
        }
        __syncwarp();
        __threadfence();
        if (tid == BLOCK_SIZE - 1) {
            status[chunk_id * N_SECTIONS + sec] = full_flag;
        }

        T _yi2_e = __shfl_up_sync(0xffffffff, in_reg[N_BLOCKS - 2], 2);
        T _yi2_o = __shfl_up_sync(0xffffffff, in_reg[N_BLOCKS - 1], 1);
        T _yi1_e = __shfl_up_sync(0xffffffff, in_reg[N_BLOCKS - 2], 1);
        T _yi1_o = __shfl_up_sync(0xffffffff, in_reg[N_BLOCKS - 1], 2);

        if (tid == 0){
            _h2_e = cr_p[sec][N_BLOCKS_LOG2 - 1];
            _h2_o = cr_h[sec][N_BLOCKS_LOG2 - 1];
            _h1_e = cr_q[sec][N_BLOCKS_LOG2 - 1];
            _h1_o = cr_g[sec][N_BLOCKS_LOG2 - 1];
            _yi2_e = X0;
            _yi2_o = X0;
            _yi1_e = X1;
            _yi1_o = X1;
            _xi2 = X0;
            _xi1 = X1;
        } else {
            _h2_e = cr_d[sec][N_BLOCKS_LOG2];
            _h2_o = cr_c[sec][N_BLOCKS_LOG2];
            _h1_e = cr_c[sec][N_BLOCKS_LOG2];
            _h1_o = cr_d[sec][N_BLOCKS_LOG2];
            if (tid == 1) {
                _yi2_e = X0;
                _yi1_o = X1;
            }
            _xi2 = _yi1_e;
            _xi1 = _yi2_o;
        }

        in_reg[HALF_N_BLOCKS - 2] = fmaf(-_h1_e, _yi1_e, fmaf(-_h2_e, _yi2_e, in_reg[HALF_N_BLOCKS - 2]));
        in_reg[HALF_N_BLOCKS - 1] = fmaf(-_h1_o, _yi1_o, fmaf(-_h2_o, _yi2_o, in_reg[HALF_N_BLOCKS - 1]));

        T _zi2_e, _zi1_e, _zi2_o, _zi1_o;


        //=============================================================================
        // ro = 3: P=2, step=16, sub=8
        //=============================================================================

        // n = 0
        _yi2_e = __shfl_up_sync(0xffffffff, in_reg[14], 1);
        _yi1_o = __shfl_up_sync(0xffffffff, in_reg[15], 1);
        if (tid == 0) {
            _h2_o = cr_h[sec][3];
            _h1_o = cr_g[sec][3];
            _h2_e = cr_p[sec][3];
            _h1_e = cr_q[sec][3];
            _yi2_e = X0;
            _yi1_o = X1;
        } else {
            _h2_o = cr_c[sec][4];
            _h1_o = cr_d[sec][4];
            _h2_e = _h1_o;
            _h1_e = _h2_o;
        }
        _zi2_o = _yi2_o;
        _zi1_o = _yi1_o;
        _zi2_e = _yi2_e;
        _zi1_e = _yi1_e;

        in_reg[6] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[6]));
        in_reg[7] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[7]));

        // n = 1
        _h2_o = cr_d[sec][4];
        _h1_o = cr_c[sec][4];
        _h2_e = _h2_o;
        _h1_e = _h1_o;
        _zi2_o = _xi1;
        _zi1_o = in_reg[15];
        _zi2_e = _xi2;
        _zi1_e = in_reg[14];

        in_reg[22] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[22]));
        in_reg[23] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[23]));

        //=============================================================================
        // ro = 2: P=4, step=8, sub=4
        //=============================================================================

        // n = 0
        _yi2_e = __shfl_up_sync(0xffffffff, in_reg[22], 1);
        _yi1_o = __shfl_up_sync(0xffffffff, in_reg[23], 1);
        if (tid == 0) {
            _h2_o = cr_h[sec][2];
            _h1_o = cr_g[sec][2];
            _h2_e = cr_p[sec][2];
            _h1_e = cr_q[sec][2];
            _yi2_e = X0;
            _yi1_o = X1;
        } else {
            _h2_o = cr_c[sec][3];
            _h1_o = cr_d[sec][3];
            _h2_e = _h1_o;
            _h1_e = _h2_o;
        }
        _zi2_o = _yi2_o;
        _zi1_o = _yi1_o;
        _zi2_e = _yi2_e;
        _zi1_e = _yi1_e;

        in_reg[2] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[2]));
        in_reg[3] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[3]));

        // n = 1
        _h2_o = cr_d[sec][3];
        _h1_o = cr_c[sec][3];
        _h2_e = _h2_o;
        _h1_e = _h1_o;
        _zi2_o = _xi1;
        _zi1_o = in_reg[7];
        _zi2_e = _xi2;
        _zi1_e = in_reg[6];

        in_reg[10] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[10]));
        in_reg[11] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[11]));

        // n = 2
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[15];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[14];

        in_reg[18] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[18]));
        in_reg[19] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[19]));

        // n = 3
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[23];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[22];

        in_reg[26] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[26]));
        in_reg[27] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[27]));

        //=============================================================================
        // ro = 1: P=8, step=4, sub=2
        //=============================================================================

        // n = 0
        _yi2_e = __shfl_up_sync(0xffffffff, in_reg[26], 1);
        _yi1_o = __shfl_up_sync(0xffffffff, in_reg[27], 1);
        if (tid == 0) {
            _h2_o = cr_h[sec][1];
            _h1_o = cr_g[sec][1];
            _h2_e = cr_p[sec][1];
            _h1_e = cr_q[sec][1];
            _yi2_e = X0;
            _yi1_o = X1;
        } else {
            _h2_o = cr_c[sec][2];
            _h1_o = cr_d[sec][2];
            _h2_e = _h1_o;
            _h1_e = _h2_o;
        }
        _zi2_o = _yi2_o;
        _zi1_o = _yi1_o;
        _zi2_e = _yi2_e;
        _zi1_e = _yi1_e;

        in_reg[0] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[0]));
        in_reg[1] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[1]));

        // n = 1
        _h2_o = cr_d[sec][2];
        _h1_o = cr_c[sec][2];
        _h2_e = _h2_o;
        _h1_e = _h1_o;
        _zi2_o = _xi1;
        _zi1_o = in_reg[3];
        _zi2_e = _xi2;
        _zi1_e = in_reg[2];

        in_reg[4] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[4]));
        in_reg[5] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[5]));

        // n = 2
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[7];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[6];

        in_reg[8] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[8]));
        in_reg[9] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[9]));

        // n = 3
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[11];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[10];

        in_reg[12] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[12]));
        in_reg[13] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[13]));

        // n = 4
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[15];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[14];

        in_reg[16] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[16]));
        in_reg[17] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[17]));

        // n = 5
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[19];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[18];

        in_reg[20] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[20]));
        in_reg[21] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[21]));

        // n = 6
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[23];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[22];

        in_reg[24] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[24]));
        in_reg[25] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[25]));

        // n = 7
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[27];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[26];

        in_reg[28] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[28]));
        in_reg[29] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[29]));

        //=============================================================================
        // Final output
        //=============================================================================


        if (sec == N_SECTIONS - 1) {
            #pragma unroll
            for (int n = 0; n < N_BLOCKS; n++)
                in[tid][n] = in_reg[n];
        } else {
            _yi2 = __shfl_up_sync(0xffffffff, in_reg[N_BLOCKS - 4], 1);
            _yi1 = __shfl_up_sync(0xffffffff, in_reg[N_BLOCKS - 3], 1);
        }
    }

    __syncwarp();

    for (int i = tid; i < CHUNK_SIZE; i += BLOCK_SIZE) {
        int global_idx = chunk_start + i;
        int tx = i / N_BLOCKS;
        int bx = i % N_BLOCKS;
        output[global_idx] = in[tx][bx];
    }
}


#endif  // N_BLOCKS == 32

#if N_BLOCKS == 64
static __global__ __launch_bounds__(BLOCK_SIZE, N_TB_PER_SM)
void STCR_32_UNROLL_64(const T* const __restrict__ input,
        T* const __restrict__ output,
        volatile int* const __restrict__ status,
        volatile T* const __restrict__ partcarry,
        volatile T* const __restrict__ fullcarry,
        const int launch)
{
    __shared__ T in[BLOCK_SIZE][N_BLOCKS + 1];
    __shared__ int cid;
    __shared__ T spartc[CHUNK_SIZE / warp_size * order];

    const int tid = threadIdx.x;
    const int lane = tid;

    // Launch-relative status values (see header comment).
    const int part_flag = 2 * launch + 1;
    const int full_flag = part_flag + 1;

    if (tid == 0) {
        cid = (int)(atomicAdd(&counter, 1u) - (unsigned int)launch * gridDim.x);
    }

    __syncwarp();

    const int chunk_id = cid;
    const int chunk_start = chunk_id * CHUNK_SIZE;

    for (int i = tid; i < CHUNK_SIZE; i += BLOCK_SIZE) {
        int global_idx = chunk_start + i;
        int tx = i / N_BLOCKS;
        int bx = i % N_BLOCKS;
        in[tx][bx] = input[global_idx];
    }

    T in_reg[N_BLOCKS];
    T _xc, _xi1, _xi2, _yc, _yi1, _yi2;

    __syncwarp();

    for (int sec = 0; sec < N_SECTIONS; sec++) {

        // Phase 1: FIR + CR
        if (tid == 0) {
            _yi2 = 0;
            _yi1 = 0;
            if (chunk_id == 0) {
                _xi2 = xi2[sec];
                _xi1 = xi1[sec];
            } else if (sec == 0) {
                _xi2 = input[chunk_start - 2];
                _xi1 = input[chunk_start - 1];
            }
        } else {
            if (sec == 0) {
                _xi2 = in[tid - 1][N_BLOCKS - 2];
                _xi1 = in[tid - 1][N_BLOCKS - 1];
                _yi2 = in[tid - 1][N_BLOCKS - 4];
                _yi1 = in[tid - 1][N_BLOCKS - 3];
            }

            _yi2 = fmaf(b1[sec], _yi1, fmaf(b2[sec], _yi2, _xi2));
            _yi1 = fmaf(b1[sec], _xi2, fmaf(b2[sec], _yi1, _xi1));
        }

        #pragma unroll
        for (int n = 0; n < N_BLOCKS; n++) {

            if (sec == 0)
                _xc = in[tid][n];
            else
                _xc = in_reg[n];

            // FIR
            _yc = fmaf(b1[sec], _xi1, fmaf(b2[sec], _xi2, _xc));

            // First round CR (double sided)
            in_reg[n] = fmaf(-e[sec][0], _yi1, fmaf(f[sec][0], _yi2, _yc));

            _xi2 = _xi1;
            _xi1 = _xc;
            _yi2 = _yi1;
            _yi1 = _yc;
        }

        // Remaining CR rounds
        #pragma unroll
        for (int ro = 1; ro < N_BLOCKS_LOG2; ro++) {

            const int step = 2 << ro;
            const int sub = 1 << ro;

            #pragma unroll
            for (int i = 0; i < 2; i++) {
                const int off = sub - 2 + i;

                _xi2 = __shfl_up_sync(0xffffffff, in_reg[N_BLOCKS - 2 + i], 1);
                if (tid == 0) _xi2 = 0;

                #pragma unroll
                for (int n = 0; n < N_BLOCKS; n += step) {
                    in_reg[n + off] = fmaf(-fde[sec][ro], _xi2, in_reg[n + off]);

                    _xi2 = in_reg[n + off + sub];
                    in_reg[n + off + sub] = fmaf(-e[sec][ro], in_reg[n + off], _xi2);
                }
            }
        }

        // Block filtering
        _yi2 = h0[sec][0] * in_reg[N_BLOCKS - 2];
        _yi1 = h0[sec][0] * in_reg[N_BLOCKS - 1];

        #pragma unroll
        for (int n = 1; n < BLOCK_SIZE; n++) {

            _xc = __shfl_up_sync(0xffffffff, in_reg[N_BLOCKS - 2], n);
            _yc = __shfl_up_sync(0xffffffff, in_reg[N_BLOCKS - 1], n);

            if (tid < n) {
                _xc = 0;
                _yc = 0;
            }

            _yi2 = fmaf(h0[sec][n], _xc, _yi2);
            _yi1 = fmaf(h0[sec][n], _yc, _yi1);
        }

        // Phase 2: Lookback
        if (lane == warp_size - 1) {
            partcarry[chunk_id * order * N_SECTIONS + sec * order] = _yi2;
            partcarry[chunk_id * order * N_SECTIONS + sec * order + 1] = _yi1;
        }

        T _h2_e, _h1_e, _h2_o, _h1_o;

        __syncwarp();
        __threadfence();
        if (tid == BLOCK_SIZE - 1)
            status[chunk_id * N_SECTIONS + sec] = part_flag;

        int flag = part_flag;
        bool no_zeros, has_status_2, reached_origin;

        do {
            if (chunk_id > lane) {
                flag = status[(chunk_id - 1 - lane) * N_SECTIONS + sec];
            }
            no_zeros = !__any_sync(0xffffffff, flag < part_flag);
            has_status_2 = !__all_sync(0xffffffff, flag != full_flag) && no_zeros;
            reached_origin = (chunk_id < warp_size) && no_zeros;
        } while (!(has_status_2 || reached_origin));

        __threadfence();

        int mask = __ballot_sync(0xffffffff, flag == full_flag);

        T X0, X1;
        int start_chunk;

        if (mask == 0) {
            X0 = yi2[sec];
            X1 = yi1[sec];
            start_chunk = 0;
        } else {
            const int pos = __ffs(mask) - 1;
            const int full_chunk = chunk_id - 1 - pos;
            start_chunk = full_chunk + 1;

            T fc;
            if (lane < order) {
                fc = fullcarry[full_chunk * order * N_SECTIONS + sec * order + lane];
            }
            X0 = __shfl_sync(0xffffffff, fc, 0);
            X1 = __shfl_sync(0xffffffff, fc, 1);
        }

        const int num_partcarries = chunk_id - start_chunk;

        if (num_partcarries > 0) {
            for (int i = (start_chunk * order + lane); i < (chunk_id * order); i += warp_size) {
                spartc[i - start_chunk * order] = partcarry[(i / order) * order * N_SECTIONS + sec * order + (i % order)];
            }
            __syncwarp();

            _h2_e = C_cross[sec][BLOCK_SIZE - 1][0];
            _h1_e = C_cross[sec][BLOCK_SIZE - 1][1];
            _h2_o = C_cross[sec][BLOCK_SIZE - 1][2];
            _h1_o = C_cross[sec][BLOCK_SIZE - 1][3];

            for (int i = 0; i < num_partcarries; i++) {
                const T p0 = spartc[i * order];
                const T p1 = spartc[i * order + 1];
                const T h0_val = fmaf(_h1_e, X1, fmaf(_h2_e, X0, p0));
                const T h1_val = fmaf(_h1_o, X1, fmaf(_h2_o, X0, p1));
                X0 = h0_val;
                X1 = h1_val;
            }
        }

        // phase 3: back substitution
        in_reg[N_BLOCKS - 2] = fmaf(-he1[sec][tid], X1, fmaf(-he2[sec][tid], X0, _yi2));
        in_reg[N_BLOCKS - 1] = fmaf(-hb1[sec][tid], X1, fmaf(-hb2[sec][tid], X0, _yi1));

        if (tid == BLOCK_SIZE - 1) {
            fullcarry[chunk_id * order * N_SECTIONS + sec * order] = in_reg[N_BLOCKS - 2];
            fullcarry[chunk_id * order * N_SECTIONS + sec * order + 1] = in_reg[N_BLOCKS - 1];
        }
        __syncwarp();
        __threadfence();
        if (tid == BLOCK_SIZE - 1) {
            status[chunk_id * N_SECTIONS + sec] = full_flag;
        }

        T _yi2_e = __shfl_up_sync(0xffffffff, in_reg[N_BLOCKS - 2], 2);
        T _yi2_o = __shfl_up_sync(0xffffffff, in_reg[N_BLOCKS - 1], 1);
        T _yi1_e = __shfl_up_sync(0xffffffff, in_reg[N_BLOCKS - 2], 1);
        T _yi1_o = __shfl_up_sync(0xffffffff, in_reg[N_BLOCKS - 1], 2);

        if (tid == 0){
            _h2_e = cr_p[sec][N_BLOCKS_LOG2 - 1];
            _h2_o = cr_h[sec][N_BLOCKS_LOG2 - 1];
            _h1_e = cr_q[sec][N_BLOCKS_LOG2 - 1];
            _h1_o = cr_g[sec][N_BLOCKS_LOG2 - 1];
            _yi2_e = X0;
            _yi2_o = X0;
            _yi1_e = X1;
            _yi1_o = X1;
            _xi2 = X0;
            _xi1 = X1;
        } else {
            _h2_e = cr_d[sec][N_BLOCKS_LOG2];
            _h2_o = cr_c[sec][N_BLOCKS_LOG2];
            _h1_e = cr_c[sec][N_BLOCKS_LOG2];
            _h1_o = cr_d[sec][N_BLOCKS_LOG2];
            if (tid == 1) {
                _yi2_e = X0;
                _yi1_o = X1;
            }
            _xi2 = _yi1_e;
            _xi1 = _yi2_o;
        }

        in_reg[HALF_N_BLOCKS - 2] = fmaf(-_h1_e, _yi1_e, fmaf(-_h2_e, _yi2_e, in_reg[HALF_N_BLOCKS - 2]));
        in_reg[HALF_N_BLOCKS - 1] = fmaf(-_h1_o, _yi1_o, fmaf(-_h2_o, _yi2_o, in_reg[HALF_N_BLOCKS - 1]));

        T _zi2_e, _zi1_e, _zi2_o, _zi1_o;

        //=============================================================================
        // ro = 4: P=2, step=32, sub=16
        //=============================================================================

        // n = 0
        _yi2_e = __shfl_up_sync(0xffffffff, in_reg[30], 1);
        _yi1_o = __shfl_up_sync(0xffffffff, in_reg[31], 1);
        if (tid == 0) {
            _h2_o = cr_h[sec][4];
            _h1_o = cr_g[sec][4];
            _h2_e = cr_p[sec][4];
            _h1_e = cr_q[sec][4];
            _yi2_e = X0;
            _yi1_o = X1;
        } else {
            _h2_o = cr_c[sec][5];
            _h1_o = cr_d[sec][5];
            _h2_e = _h1_o;
            _h1_e = _h2_o;
        }
        _zi2_o = _yi2_o;
        _zi1_o = _yi1_o;
        _zi2_e = _yi2_e;
        _zi1_e = _yi1_e;

        in_reg[14] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[14]));
        in_reg[15] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[15]));

        // n = 1
        _h2_o = cr_d[sec][5];
        _h1_o = cr_c[sec][5];
        _h2_e = _h2_o;
        _h1_e = _h1_o;
        _zi2_o = _xi1;
        _zi1_o = in_reg[31];
        _zi2_e = _xi2;
        _zi1_e = in_reg[30];

        in_reg[46] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[46]));
        in_reg[47] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[47]));

        //=============================================================================
        // ro = 3: P=4, step=16, sub=8
        //=============================================================================

        // n = 0
        _yi2_e = __shfl_up_sync(0xffffffff, in_reg[46], 1);
        _yi1_o = __shfl_up_sync(0xffffffff, in_reg[47], 1);
        if (tid == 0) {
            _h2_o = cr_h[sec][3];
            _h1_o = cr_g[sec][3];
            _h2_e = cr_p[sec][3];
            _h1_e = cr_q[sec][3];
            _yi2_e = X0;
            _yi1_o = X1;
        } else {
            _h2_o = cr_c[sec][4];
            _h1_o = cr_d[sec][4];
            _h2_e = _h1_o;
            _h1_e = _h2_o;
        }
        _zi2_o = _yi2_o;
        _zi1_o = _yi1_o;
        _zi2_e = _yi2_e;
        _zi1_e = _yi1_e;

        in_reg[6] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[6]));
        in_reg[7] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[7]));

        // n = 1
        _h2_o = cr_d[sec][4];
        _h1_o = cr_c[sec][4];
        _h2_e = _h2_o;
        _h1_e = _h1_o;
        _zi2_o = _xi1;
        _zi1_o = in_reg[15];
        _zi2_e = _xi2;
        _zi1_e = in_reg[14];

        in_reg[22] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[22]));
        in_reg[23] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[23]));
        // n = 2
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[31];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[30];

        in_reg[38] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[38]));
        in_reg[39] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[39]));
        // n = 3
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[47];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[46];

        in_reg[54] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[54]));
        in_reg[55] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[55]));

        //=============================================================================
        // ro = 2: P=8, step=8, sub=4
        //=============================================================================

        // n = 0
        _yi2_e = __shfl_up_sync(0xffffffff, in_reg[54], 1);
        _yi1_o = __shfl_up_sync(0xffffffff, in_reg[55], 1);
        if (tid == 0) {
            _h2_o = cr_h[sec][2];
            _h1_o = cr_g[sec][2];
            _h2_e = cr_p[sec][2];
            _h1_e = cr_q[sec][2];
            _yi2_e = X0;
            _yi1_o = X1;
        } else {
            _h2_o = cr_c[sec][3];
            _h1_o = cr_d[sec][3];
            _h2_e = _h1_o;
            _h1_e = _h2_o;
        }
        _zi2_o = _yi2_o;
        _zi1_o = _yi1_o;
        _zi2_e = _yi2_e;
        _zi1_e = _yi1_e;

        in_reg[2] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[2]));
        in_reg[3] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[3]));

        // n = 1
        _h2_o = cr_d[sec][3];
        _h1_o = cr_c[sec][3];
        _h2_e = _h2_o;
        _h1_e = _h1_o;
        _zi2_o = _xi1;
        _zi1_o = in_reg[7];
        _zi2_e = _xi2;
        _zi1_e = in_reg[6];

        in_reg[10] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[10]));
        in_reg[11] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[11]));
        // n = 2
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[15];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[14];

        in_reg[18] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[18]));
        in_reg[19] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[19]));
        // n = 3
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[23];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[22];

        in_reg[26] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[26]));
        in_reg[27] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[27]));
        // n = 4
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[31];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[30];

        in_reg[34] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[34]));
        in_reg[35] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[35]));
        // n = 5
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[39];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[38];

        in_reg[42] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[42]));
        in_reg[43] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[43]));
        // n = 6
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[47];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[46];

        in_reg[50] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[50]));
        in_reg[51] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[51]));
        // n = 7
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[55];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[54];

        in_reg[58] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[58]));
        in_reg[59] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[59]));

        //=============================================================================
        // ro = 1: P=16, step=4, sub=2
        //=============================================================================

        // n = 0
        _yi2_e = __shfl_up_sync(0xffffffff, in_reg[58], 1);
        _yi1_o = __shfl_up_sync(0xffffffff, in_reg[59], 1);
        if (tid == 0) {
            _h2_o = cr_h[sec][1];
            _h1_o = cr_g[sec][1];
            _h2_e = cr_p[sec][1];
            _h1_e = cr_q[sec][1];
            _yi2_e = X0;
            _yi1_o = X1;
        } else {
            _h2_o = cr_c[sec][2];
            _h1_o = cr_d[sec][2];
            _h2_e = _h1_o;
            _h1_e = _h2_o;
        }
        _zi2_o = _yi2_o;
        _zi1_o = _yi1_o;
        _zi2_e = _yi2_e;
        _zi1_e = _yi1_e;

        in_reg[0] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[0]));
        in_reg[1] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[1]));

        // n = 1
        _h2_o = cr_d[sec][2];
        _h1_o = cr_c[sec][2];
        _h2_e = _h2_o;
        _h1_e = _h1_o;
        _zi2_o = _xi1;
        _zi1_o = in_reg[3];
        _zi2_e = _xi2;
        _zi1_e = in_reg[2];

        in_reg[4] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[4]));
        in_reg[5] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[5]));
        // n = 2
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[7];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[6];

        in_reg[8] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[8]));
        in_reg[9] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[9]));
        // n = 3
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[11];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[10];

        in_reg[12] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[12]));
        in_reg[13] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[13]));
        // n = 4
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[15];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[14];

        in_reg[16] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[16]));
        in_reg[17] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[17]));
        // n = 5
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[19];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[18];

        in_reg[20] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[20]));
        in_reg[21] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[21]));
        // n = 6
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[23];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[22];

        in_reg[24] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[24]));
        in_reg[25] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[25]));
        // n = 7
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[27];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[26];

        in_reg[28] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[28]));
        in_reg[29] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[29]));
        // n = 8
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[31];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[30];

        in_reg[32] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[32]));
        in_reg[33] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[33]));
        // n = 9
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[35];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[34];

        in_reg[36] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[36]));
        in_reg[37] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[37]));
        // n = 10
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[39];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[38];

        in_reg[40] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[40]));
        in_reg[41] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[41]));
        // n = 11
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[43];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[42];

        in_reg[44] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[44]));
        in_reg[45] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[45]));
        // n = 12
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[47];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[46];

        in_reg[48] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[48]));
        in_reg[49] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[49]));
        // n = 13
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[51];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[50];

        in_reg[52] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[52]));
        in_reg[53] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[53]));
        // n = 14
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[55];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[54];

        in_reg[56] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[56]));
        in_reg[57] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[57]));
        // n = 15
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[59];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[58];

        in_reg[60] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[60]));
        in_reg[61] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[61]));

        if (sec == N_SECTIONS - 1) {
            #pragma unroll
            for (int n = 0; n < N_BLOCKS; n++)
                in[tid][n] = in_reg[n];
        } else {
            _yi2 = __shfl_up_sync(0xffffffff, in_reg[N_BLOCKS - 4], 1);
            _yi1 = __shfl_up_sync(0xffffffff, in_reg[N_BLOCKS - 3], 1);
        }
    }

    __syncwarp();

    for (int i = tid; i < CHUNK_SIZE; i += BLOCK_SIZE) {
        int global_idx = chunk_start + i;
        int tx = i / N_BLOCKS;
        int bx = i % N_BLOCKS;
        output[global_idx] = in[tx][bx];
    }
}


#endif  // N_BLOCKS == 64

#if N_BLOCKS == 128
static __global__ __launch_bounds__(BLOCK_SIZE, N_TB_PER_SM)
void STCR_32_UNROLL_128(const T* const __restrict__ input,
        T* const __restrict__ output,
        volatile int* const __restrict__ status,
        volatile T* const __restrict__ partcarry,
        volatile T* const __restrict__ fullcarry,
        const int launch)
{
    __shared__ T in[BLOCK_SIZE][N_BLOCKS + 1];
    __shared__ int cid;
    __shared__ T spartc[CHUNK_SIZE / warp_size * order];

    const int tid = threadIdx.x;
    const int lane = tid;

    // Launch-relative status values (see header comment).
    const int part_flag = 2 * launch + 1;
    const int full_flag = part_flag + 1;

    if (tid == 0) {
        cid = (int)(atomicAdd(&counter, 1u) - (unsigned int)launch * gridDim.x);
    }

    __syncwarp();

    const int chunk_id = cid;
    const int chunk_start = chunk_id * CHUNK_SIZE;

    for (int i = tid; i < CHUNK_SIZE; i += BLOCK_SIZE) {
        int global_idx = chunk_start + i;
        int tx = i / N_BLOCKS;
        int bx = i % N_BLOCKS;
        in[tx][bx] = input[global_idx];
    }

    T in_reg[N_BLOCKS];
    T _xc, _xi1, _xi2, _yc, _yi1, _yi2;

    __syncwarp();

    for (int sec = 0; sec < N_SECTIONS; sec++) {

        // Phase 1: FIR + CR
        if (tid == 0) {
            _yi2 = 0;
            _yi1 = 0;
            if (chunk_id == 0) {
                _xi2 = xi2[sec];
                _xi1 = xi1[sec];
            } else if (sec == 0) {
                _xi2 = input[chunk_start - 2];
                _xi1 = input[chunk_start - 1];
            }
        } else {
            if (sec == 0) {
                _xi2 = in[tid - 1][N_BLOCKS - 2];
                _xi1 = in[tid - 1][N_BLOCKS - 1];
                _yi2 = in[tid - 1][N_BLOCKS - 4];
                _yi1 = in[tid - 1][N_BLOCKS - 3];
            }

            _yi2 = fmaf(b1[sec], _yi1, fmaf(b2[sec], _yi2, _xi2));
            _yi1 = fmaf(b1[sec], _xi2, fmaf(b2[sec], _yi1, _xi1));
        }

        #pragma unroll
        for (int n = 0; n < N_BLOCKS; n++) {

            if (sec == 0)
                _xc = in[tid][n];
            else
                _xc = in_reg[n];

            // FIR
            _yc = fmaf(b1[sec], _xi1, fmaf(b2[sec], _xi2, _xc));

            // First round CR (double sided)
            in_reg[n] = fmaf(-e[sec][0], _yi1, fmaf(f[sec][0], _yi2, _yc));

            _xi2 = _xi1;
            _xi1 = _xc;
            _yi2 = _yi1;
            _yi1 = _yc;
        }

        // Remaining CR rounds
        #pragma unroll
        for (int ro = 1; ro < N_BLOCKS_LOG2; ro++) {

            const int step = 2 << ro;
            const int sub = 1 << ro;

            #pragma unroll
            for (int i = 0; i < 2; i++) {
                const int off = sub - 2 + i;

                _xi2 = __shfl_up_sync(0xffffffff, in_reg[N_BLOCKS - 2 + i], 1);
                if (tid == 0) _xi2 = 0;

                #pragma unroll
                for (int n = 0; n < N_BLOCKS; n += step) {
                    in_reg[n + off] = fmaf(-fde[sec][ro], _xi2, in_reg[n + off]);

                    _xi2 = in_reg[n + off + sub];
                    in_reg[n + off + sub] = fmaf(-e[sec][ro], in_reg[n + off], _xi2);
                }
            }
        }

        // Block filtering
        _yi2 = h0[sec][0] * in_reg[N_BLOCKS - 2];
        _yi1 = h0[sec][0] * in_reg[N_BLOCKS - 1];

        #pragma unroll
        for (int n = 1; n < BLOCK_SIZE; n++) {

            _xc = __shfl_up_sync(0xffffffff, in_reg[N_BLOCKS - 2], n);
            _yc = __shfl_up_sync(0xffffffff, in_reg[N_BLOCKS - 1], n);

            if (tid < n) {
                _xc = 0;
                _yc = 0;
            }

            _yi2 = fmaf(h0[sec][n], _xc, _yi2);
            _yi1 = fmaf(h0[sec][n], _yc, _yi1);
        }

        // Phase 2: Lookback
        if (lane == warp_size - 1) {
            partcarry[chunk_id * order * N_SECTIONS + sec * order] = _yi2;
            partcarry[chunk_id * order * N_SECTIONS + sec * order + 1] = _yi1;
        }

        T _h2_e, _h1_e, _h2_o, _h1_o;

        __syncwarp();
        __threadfence();
        if (tid == BLOCK_SIZE - 1)
            status[chunk_id * N_SECTIONS + sec] = part_flag;

        int flag = part_flag;
        bool no_zeros, has_status_2, reached_origin;

        do {
            if (chunk_id > lane) {
                flag = status[(chunk_id - 1 - lane) * N_SECTIONS + sec];
            }
            no_zeros = !__any_sync(0xffffffff, flag < part_flag);
            has_status_2 = !__all_sync(0xffffffff, flag != full_flag) && no_zeros;
            reached_origin = (chunk_id < warp_size) && no_zeros;
        } while (!(has_status_2 || reached_origin));

        __threadfence();

        int mask = __ballot_sync(0xffffffff, flag == full_flag);

        T X0, X1;
        int start_chunk;

        if (mask == 0) {
            X0 = yi2[sec];
            X1 = yi1[sec];
            start_chunk = 0;
        } else {
            const int pos = __ffs(mask) - 1;
            const int full_chunk = chunk_id - 1 - pos;
            start_chunk = full_chunk + 1;

            T fc;
            if (lane < order) {
                fc = fullcarry[full_chunk * order * N_SECTIONS + sec * order + lane];
            }
            X0 = __shfl_sync(0xffffffff, fc, 0);
            X1 = __shfl_sync(0xffffffff, fc, 1);
        }

        const int num_partcarries = chunk_id - start_chunk;

        if (num_partcarries > 0) {
            for (int i = (start_chunk * order + lane); i < (chunk_id * order); i += warp_size) {
                spartc[i - start_chunk * order] = partcarry[(i / order) * order * N_SECTIONS + sec * order + (i % order)];
            }
            __syncwarp();

            _h2_e = C_cross[sec][BLOCK_SIZE - 1][0];
            _h1_e = C_cross[sec][BLOCK_SIZE - 1][1];
            _h2_o = C_cross[sec][BLOCK_SIZE - 1][2];
            _h1_o = C_cross[sec][BLOCK_SIZE - 1][3];

            for (int i = 0; i < num_partcarries; i++) {
                const T p0 = spartc[i * order];
                const T p1 = spartc[i * order + 1];
                const T h0_val = fmaf(_h1_e, X1, fmaf(_h2_e, X0, p0));
                const T h1_val = fmaf(_h1_o, X1, fmaf(_h2_o, X0, p1));
                X0 = h0_val;
                X1 = h1_val;
            }
        }

        // phase 3: back substitution
        in_reg[N_BLOCKS - 2] = fmaf(-he1[sec][tid], X1, fmaf(-he2[sec][tid], X0, _yi2));
        in_reg[N_BLOCKS - 1] = fmaf(-hb1[sec][tid], X1, fmaf(-hb2[sec][tid], X0, _yi1));

        if (tid == BLOCK_SIZE - 1) {
            fullcarry[chunk_id * order * N_SECTIONS + sec * order] = in_reg[N_BLOCKS - 2];
            fullcarry[chunk_id * order * N_SECTIONS + sec * order + 1] = in_reg[N_BLOCKS - 1];
        }
        __syncwarp();
        __threadfence();
        if (tid == BLOCK_SIZE - 1) {
            status[chunk_id * N_SECTIONS + sec] = full_flag;
        }

        T _yi2_e = __shfl_up_sync(0xffffffff, in_reg[N_BLOCKS - 2], 2);
        T _yi2_o = __shfl_up_sync(0xffffffff, in_reg[N_BLOCKS - 1], 1);
        T _yi1_e = __shfl_up_sync(0xffffffff, in_reg[N_BLOCKS - 2], 1);
        T _yi1_o = __shfl_up_sync(0xffffffff, in_reg[N_BLOCKS - 1], 2);

        if (tid == 0){
            _h2_e = cr_p[sec][N_BLOCKS_LOG2 - 1];
            _h2_o = cr_h[sec][N_BLOCKS_LOG2 - 1];
            _h1_e = cr_q[sec][N_BLOCKS_LOG2 - 1];
            _h1_o = cr_g[sec][N_BLOCKS_LOG2 - 1];
            _yi2_e = X0;
            _yi2_o = X0;
            _yi1_e = X1;
            _yi1_o = X1;
            _xi2 = X0;
            _xi1 = X1;
        } else {
            _h2_e = cr_d[sec][N_BLOCKS_LOG2];
            _h2_o = cr_c[sec][N_BLOCKS_LOG2];
            _h1_e = cr_c[sec][N_BLOCKS_LOG2];
            _h1_o = cr_d[sec][N_BLOCKS_LOG2];
            if (tid == 1) {
                _yi2_e = X0;
                _yi1_o = X1;
            }
            _xi2 = _yi1_e;
            _xi1 = _yi2_o;
        }

        in_reg[HALF_N_BLOCKS - 2] = fmaf(-_h1_e, _yi1_e, fmaf(-_h2_e, _yi2_e, in_reg[HALF_N_BLOCKS - 2]));
        in_reg[HALF_N_BLOCKS - 1] = fmaf(-_h1_o, _yi1_o, fmaf(-_h2_o, _yi2_o, in_reg[HALF_N_BLOCKS - 1]));

        T _zi2_e, _zi1_e, _zi2_o, _zi1_o;

        //=============================================================================
        // ro = 5: P=2, step=64, sub=32
        //=============================================================================

        // n = 0
        _yi2_e = __shfl_up_sync(0xffffffff, in_reg[62], 1);
        _yi1_o = __shfl_up_sync(0xffffffff, in_reg[63], 1);
        if (tid == 0) {
            _h2_o = cr_h[sec][5];
            _h1_o = cr_g[sec][5];
            _h2_e = cr_p[sec][5];
            _h1_e = cr_q[sec][5];
            _yi2_e = X0;
            _yi1_o = X1;
        } else {
            _h2_o = cr_c[sec][6];
            _h1_o = cr_d[sec][6];
            _h2_e = _h1_o;
            _h1_e = _h2_o;
        }
        _zi2_o = _yi2_o;
        _zi1_o = _yi1_o;
        _zi2_e = _yi2_e;
        _zi1_e = _yi1_e;

        in_reg[30] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[30]));
        in_reg[31] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[31]));

        // n = 1
        _h2_o = cr_d[sec][6];
        _h1_o = cr_c[sec][6];
        _h2_e = _h2_o;
        _h1_e = _h1_o;
        _zi2_o = _xi1;
        _zi1_o = in_reg[63];
        _zi2_e = _xi2;
        _zi1_e = in_reg[62];

        in_reg[94] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[94]));
        in_reg[95] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[95]));

        //=============================================================================
        // ro = 4: P=4, step=32, sub=16
        //=============================================================================

        // n = 0
        _yi2_e = __shfl_up_sync(0xffffffff, in_reg[94], 1);
        _yi1_o = __shfl_up_sync(0xffffffff, in_reg[95], 1);
        if (tid == 0) {
            _h2_o = cr_h[sec][4];
            _h1_o = cr_g[sec][4];
            _h2_e = cr_p[sec][4];
            _h1_e = cr_q[sec][4];
            _yi2_e = X0;
            _yi1_o = X1;
        } else {
            _h2_o = cr_c[sec][5];
            _h1_o = cr_d[sec][5];
            _h2_e = _h1_o;
            _h1_e = _h2_o;
        }
        _zi2_o = _yi2_o;
        _zi1_o = _yi1_o;
        _zi2_e = _yi2_e;
        _zi1_e = _yi1_e;

        in_reg[14] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[14]));
        in_reg[15] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[15]));

        // n = 1
        _h2_o = cr_d[sec][5];
        _h1_o = cr_c[sec][5];
        _h2_e = _h2_o;
        _h1_e = _h1_o;
        _zi2_o = _xi1;
        _zi1_o = in_reg[31];
        _zi2_e = _xi2;
        _zi1_e = in_reg[30];

        in_reg[46] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[46]));
        in_reg[47] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[47]));
        // n = 2
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[63];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[62];

        in_reg[78] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[78]));
        in_reg[79] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[79]));
        // n = 3
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[95];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[94];

        in_reg[110] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[110]));
        in_reg[111] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[111]));

        //=============================================================================
        // ro = 3: P=8, step=16, sub=8
        //=============================================================================

        // n = 0
        _yi2_e = __shfl_up_sync(0xffffffff, in_reg[110], 1);
        _yi1_o = __shfl_up_sync(0xffffffff, in_reg[111], 1);
        if (tid == 0) {
            _h2_o = cr_h[sec][3];
            _h1_o = cr_g[sec][3];
            _h2_e = cr_p[sec][3];
            _h1_e = cr_q[sec][3];
            _yi2_e = X0;
            _yi1_o = X1;
        } else {
            _h2_o = cr_c[sec][4];
            _h1_o = cr_d[sec][4];
            _h2_e = _h1_o;
            _h1_e = _h2_o;
        }
        _zi2_o = _yi2_o;
        _zi1_o = _yi1_o;
        _zi2_e = _yi2_e;
        _zi1_e = _yi1_e;

        in_reg[6] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[6]));
        in_reg[7] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[7]));

        // n = 1
        _h2_o = cr_d[sec][4];
        _h1_o = cr_c[sec][4];
        _h2_e = _h2_o;
        _h1_e = _h1_o;
        _zi2_o = _xi1;
        _zi1_o = in_reg[15];
        _zi2_e = _xi2;
        _zi1_e = in_reg[14];

        in_reg[22] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[22]));
        in_reg[23] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[23]));
        // n = 2
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[31];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[30];

        in_reg[38] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[38]));
        in_reg[39] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[39]));
        // n = 3
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[47];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[46];

        in_reg[54] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[54]));
        in_reg[55] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[55]));
        // n = 4
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[63];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[62];

        in_reg[70] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[70]));
        in_reg[71] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[71]));
        // n = 5
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[79];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[78];

        in_reg[86] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[86]));
        in_reg[87] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[87]));
        // n = 6
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[95];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[94];

        in_reg[102] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[102]));
        in_reg[103] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[103]));
        // n = 7
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[111];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[110];

        in_reg[118] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[118]));
        in_reg[119] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[119]));

        //=============================================================================
        // ro = 2: P=16, step=8, sub=4
        //=============================================================================

        // n = 0
        _yi2_e = __shfl_up_sync(0xffffffff, in_reg[118], 1);
        _yi1_o = __shfl_up_sync(0xffffffff, in_reg[119], 1);
        if (tid == 0) {
            _h2_o = cr_h[sec][2];
            _h1_o = cr_g[sec][2];
            _h2_e = cr_p[sec][2];
            _h1_e = cr_q[sec][2];
            _yi2_e = X0;
            _yi1_o = X1;
        } else {
            _h2_o = cr_c[sec][3];
            _h1_o = cr_d[sec][3];
            _h2_e = _h1_o;
            _h1_e = _h2_o;
        }
        _zi2_o = _yi2_o;
        _zi1_o = _yi1_o;
        _zi2_e = _yi2_e;
        _zi1_e = _yi1_e;

        in_reg[2] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[2]));
        in_reg[3] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[3]));

        // n = 1
        _h2_o = cr_d[sec][3];
        _h1_o = cr_c[sec][3];
        _h2_e = _h2_o;
        _h1_e = _h1_o;
        _zi2_o = _xi1;
        _zi1_o = in_reg[7];
        _zi2_e = _xi2;
        _zi1_e = in_reg[6];

        in_reg[10] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[10]));
        in_reg[11] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[11]));
        // n = 2
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[15];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[14];

        in_reg[18] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[18]));
        in_reg[19] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[19]));
        // n = 3
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[23];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[22];

        in_reg[26] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[26]));
        in_reg[27] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[27]));
        // n = 4
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[31];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[30];

        in_reg[34] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[34]));
        in_reg[35] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[35]));
        // n = 5
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[39];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[38];

        in_reg[42] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[42]));
        in_reg[43] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[43]));
        // n = 6
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[47];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[46];

        in_reg[50] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[50]));
        in_reg[51] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[51]));
        // n = 7
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[55];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[54];

        in_reg[58] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[58]));
        in_reg[59] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[59]));
        // n = 8
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[63];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[62];

        in_reg[66] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[66]));
        in_reg[67] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[67]));
        // n = 9
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[71];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[70];

        in_reg[74] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[74]));
        in_reg[75] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[75]));
        // n = 10
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[79];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[78];

        in_reg[82] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[82]));
        in_reg[83] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[83]));
        // n = 11
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[87];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[86];

        in_reg[90] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[90]));
        in_reg[91] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[91]));
        // n = 12
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[95];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[94];

        in_reg[98] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[98]));
        in_reg[99] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[99]));
        // n = 13
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[103];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[102];

        in_reg[106] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[106]));
        in_reg[107] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[107]));
        // n = 14
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[111];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[110];

        in_reg[114] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[114]));
        in_reg[115] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[115]));
        // n = 15
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[119];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[118];

        in_reg[122] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[122]));
        in_reg[123] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[123]));

        //=============================================================================
        // ro = 1: P=32, step=4, sub=2
        //=============================================================================

        // n = 0
        _yi2_e = __shfl_up_sync(0xffffffff, in_reg[122], 1);
        _yi1_o = __shfl_up_sync(0xffffffff, in_reg[123], 1);
        if (tid == 0) {
            _h2_o = cr_h[sec][1];
            _h1_o = cr_g[sec][1];
            _h2_e = cr_p[sec][1];
            _h1_e = cr_q[sec][1];
            _yi2_e = X0;
            _yi1_o = X1;
        } else {
            _h2_o = cr_c[sec][2];
            _h1_o = cr_d[sec][2];
            _h2_e = _h1_o;
            _h1_e = _h2_o;
        }
        _zi2_o = _yi2_o;
        _zi1_o = _yi1_o;
        _zi2_e = _yi2_e;
        _zi1_e = _yi1_e;

        in_reg[0] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[0]));
        in_reg[1] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[1]));

        // n = 1
        _h2_o = cr_d[sec][2];
        _h1_o = cr_c[sec][2];
        _h2_e = _h2_o;
        _h1_e = _h1_o;
        _zi2_o = _xi1;
        _zi1_o = in_reg[3];
        _zi2_e = _xi2;
        _zi1_e = in_reg[2];

        in_reg[4] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[4]));
        in_reg[5] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[5]));
        // n = 2
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[7];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[6];

        in_reg[8] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[8]));
        in_reg[9] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[9]));
        // n = 3
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[11];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[10];

        in_reg[12] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[12]));
        in_reg[13] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[13]));
        // n = 4
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[15];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[14];

        in_reg[16] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[16]));
        in_reg[17] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[17]));
        // n = 5
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[19];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[18];

        in_reg[20] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[20]));
        in_reg[21] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[21]));
        // n = 6
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[23];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[22];

        in_reg[24] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[24]));
        in_reg[25] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[25]));
        // n = 7
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[27];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[26];

        in_reg[28] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[28]));
        in_reg[29] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[29]));
        // n = 8
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[31];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[30];

        in_reg[32] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[32]));
        in_reg[33] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[33]));
        // n = 9
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[35];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[34];

        in_reg[36] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[36]));
        in_reg[37] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[37]));
        // n = 10
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[39];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[38];

        in_reg[40] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[40]));
        in_reg[41] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[41]));
        // n = 11
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[43];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[42];

        in_reg[44] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[44]));
        in_reg[45] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[45]));
        // n = 12
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[47];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[46];

        in_reg[48] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[48]));
        in_reg[49] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[49]));
        // n = 13
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[51];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[50];

        in_reg[52] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[52]));
        in_reg[53] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[53]));
        // n = 14
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[55];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[54];

        in_reg[56] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[56]));
        in_reg[57] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[57]));
        // n = 15
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[59];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[58];

        in_reg[60] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[60]));
        in_reg[61] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[61]));
        // n = 16
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[63];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[62];

        in_reg[64] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[64]));
        in_reg[65] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[65]));
        // n = 17
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[67];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[66];

        in_reg[68] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[68]));
        in_reg[69] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[69]));
        // n = 18
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[71];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[70];

        in_reg[72] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[72]));
        in_reg[73] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[73]));
        // n = 19
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[75];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[74];

        in_reg[76] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[76]));
        in_reg[77] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[77]));
        // n = 20
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[79];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[78];

        in_reg[80] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[80]));
        in_reg[81] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[81]));
        // n = 21
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[83];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[82];

        in_reg[84] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[84]));
        in_reg[85] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[85]));
        // n = 22
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[87];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[86];

        in_reg[88] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[88]));
        in_reg[89] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[89]));
        // n = 23
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[91];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[90];

        in_reg[92] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[92]));
        in_reg[93] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[93]));
        // n = 24
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[95];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[94];

        in_reg[96] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[96]));
        in_reg[97] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[97]));
        // n = 25
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[99];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[98];

        in_reg[100] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[100]));
        in_reg[101] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[101]));
        // n = 26
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[103];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[102];

        in_reg[104] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[104]));
        in_reg[105] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[105]));
        // n = 27
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[107];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[106];

        in_reg[108] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[108]));
        in_reg[109] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[109]));
        // n = 28
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[111];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[110];

        in_reg[112] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[112]));
        in_reg[113] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[113]));
        // n = 29
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[115];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[114];

        in_reg[116] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[116]));
        in_reg[117] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[117]));
        // n = 30
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[119];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[118];

        in_reg[120] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[120]));
        in_reg[121] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[121]));
        // n = 31
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[123];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[122];

        in_reg[124] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[124]));
        in_reg[125] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[125]));

        if (sec == N_SECTIONS - 1) {
            #pragma unroll
            for (int n = 0; n < N_BLOCKS; n++)
                in[tid][n] = in_reg[n];
        } else {
            _yi2 = __shfl_up_sync(0xffffffff, in_reg[N_BLOCKS - 4], 1);
            _yi1 = __shfl_up_sync(0xffffffff, in_reg[N_BLOCKS - 3], 1);
        }
    }

    __syncwarp();

    for (int i = tid; i < CHUNK_SIZE; i += BLOCK_SIZE) {
        int global_idx = chunk_start + i;
        int tx = i / N_BLOCKS;
        int bx = i % N_BLOCKS;
        output[global_idx] = in[tx][bx];
    }
}


#endif  // N_BLOCKS == 128

static __global__ __launch_bounds__(BLOCK_SIZE, N_TB_PER_SM)
void STCR_64_LOOP(const T* const __restrict__ input,
        T* const __restrict__ output,
        volatile int* const __restrict__ status,
        volatile T* const __restrict__ partcarry,
        volatile T* const __restrict__ fullcarry,
        const int launch)
{
    __shared__ T in[BLOCK_SIZE][N_BLOCKS + 1];
    __shared__ int cid;
    __shared__ T sfullc[order];

    const int tid = threadIdx.x;
    const int warp = tid / warp_size;
    const int lane = tid % warp_size;

    // Launch-relative status values (see header comment).
    const int part_flag = 2 * launch + 1;
    const int full_flag = part_flag + 1;

    if (tid == 0) {
        cid = (int)(atomicAdd(&counter, 1u) - (unsigned int)launch * gridDim.x);
    }

    __syncthreads();

    const int chunk_id = cid;
    const int chunk_start = chunk_id * CHUNK_SIZE;

    for (int i = tid; i < CHUNK_SIZE; i += BLOCK_SIZE) {
        int global_idx = chunk_start + i;
        int tx = i / N_BLOCKS;
        int bx = i % N_BLOCKS;
        in[tx][bx] = input[global_idx];
    }

    T in_reg[N_BLOCKS];
    T _xc, _xi1, _xi2, _yc, _yi1, _yi2;

    __syncthreads();

    for (int sec = 0; sec < N_SECTIONS; sec++) {

        if (tid == 0) {
            _yi2 = 0;
            _yi1 = 0;
            if (chunk_id == 0) {
                _xi2 = xi2[sec];
                _xi1 = xi1[sec];
            } else if (sec == 0) {
                _xi2 = input[chunk_start - 2];
                _xi1 = input[chunk_start - 1];
            }
        } else {
            if (sec == 0) {
                _xi2 = in[tid - 1][N_BLOCKS - 2];
                _xi1 = in[tid - 1][N_BLOCKS - 1];
            }
            _yi2 = in[tid - 1][N_BLOCKS - 4];
            _yi1 = in[tid - 1][N_BLOCKS - 3];

            _yi2 = fmaf(b1[sec], _yi1, fmaf(b2[sec], _yi2, _xi2));
            _yi1 = fmaf(b1[sec], _xi2, fmaf(b2[sec], _yi1, _xi1));
        }

        #pragma unroll
        for (int n = 0; n < N_BLOCKS; n++) {

            if (sec == 0)
                _xc = in[tid][n];
            else
                _xc = in_reg[n];

            // FIR
            _yc = fmaf(b1[sec], _xi1, fmaf(b2[sec], _xi2, _xc));

            // First round CR (double sided)
            in_reg[n] = fmaf(-e[sec][0], _yi1, fmaf(f[sec][0], _yi2, _yc));

            if (n > N_BLOCKS - 3)
                in[tid][n] = in_reg[n];

            _xi2 = _xi1;
            _xi1 = _xc;
            _yi2 = _yi1;
            _yi1 = _yc;
        }

        __syncthreads();

        // Remaining CR rounds
        #pragma unroll
        for (int ro = 1; ro < N_BLOCKS_LOG2; ro++) {

            const int step = 2 << ro;
            const int sub = 1 << ro;

            #pragma unroll
            for (int i = 0; i < 2; i++) {
                const int off = sub - 2 + i;

                if (tid == 0)
                    _xi2 = 0;
                else
                    _xi2 = in[tid - 1][N_BLOCKS - 2 + i];

                __syncthreads();

                #pragma unroll
                for (int n = 0; n < N_BLOCKS; n += step) {
                    in_reg[n + off] = fmaf(-fde[sec][ro], _xi2, in_reg[n + off]);

                    _xi2 = in_reg[n + off + sub];
                    in_reg[n + off + sub] = fmaf(-e[sec][ro], in_reg[n + off], _xi2);
                }

                in[tid][N_BLOCKS - 2 + i] = in_reg[N_BLOCKS - 2 + i];

                __syncthreads();
            }
        }

        // Block filtering
        _yi2 = h0[sec][0] * in_reg[N_BLOCKS - 2];
        _yi1 = h0[sec][0] * in_reg[N_BLOCKS - 1];

        #pragma unroll
        for (int n = 1; n < BLOCK_SIZE; n++) {

            if (tid >= n) {
                _xc = in[tid - n][N_BLOCKS - 2];
                _yc = in[tid - n][N_BLOCKS - 1];
            } else {
                _xc = 0;
                _yc = 0;
            }
            _yi2 = fmaf(h0[sec][n], _xc, _yi2);
            _yi1 = fmaf(h0[sec][n], _yc, _yi1);
        }

        T _h2_e, _h1_e, _h2_o, _h1_o;
        T X0, X1;

        // phase 2 - Lookback
        if (warp == 1) {

            if (lane == warp_size - 1) {
                partcarry[chunk_id * order * N_SECTIONS + sec * order] = _yi2;
                partcarry[chunk_id * order * N_SECTIONS + sec * order + 1] = _yi1;
            }

            __syncwarp();
            __threadfence();
            if (tid == BLOCK_SIZE - 1)
                status[chunk_id * N_SECTIONS + sec] = part_flag;

            int flag = part_flag;
            bool no_zeros, has_status_2, reached_origin;

            do {
                if (chunk_id > lane) {
                    flag = status[(chunk_id - 1 - lane) * N_SECTIONS + sec];
                }
                no_zeros = !__any_sync(0xffffffff, flag < part_flag);
                has_status_2 = !__all_sync(0xffffffff, flag != full_flag) && no_zeros;
                reached_origin = (chunk_id < warp_size) && no_zeros;
            } while (!(has_status_2 || reached_origin));

            __threadfence();

            int mask = __ballot_sync(0xffffffff, flag == full_flag);

            int start_chunk;

            if (mask == 0) {
                X0 = yi2[sec];
                X1 = yi1[sec];
                start_chunk = 0;
            } else {
                const int pos = __ffs(mask) - 1;
                const int full_chunk = chunk_id - 1 - pos;
                start_chunk = full_chunk + 1;

                T fc;
                if (lane < order) {
                    fc = fullcarry[full_chunk * order * N_SECTIONS + sec * order + lane];
                }
                X0 = __shfl_sync(0xffffffff, fc, 0);
                X1 = __shfl_sync(0xffffffff, fc, 1);
            }

            const int num_partcarries = chunk_id - start_chunk;

            if (num_partcarries > 0) {
                for (int i = (start_chunk * order + lane); i < (chunk_id * order); i += warp_size) {
                    in[i - start_chunk * order][0] = partcarry[(i / order) * order * N_SECTIONS + sec * order + (i % order)];
                }
                __syncwarp();

                if (lane == 0) {
                    _h2_e = C_cross[sec][BLOCK_SIZE - 1][0];
                    _h1_e = C_cross[sec][BLOCK_SIZE - 1][1];
                    _h2_o = C_cross[sec][BLOCK_SIZE - 1][2];
                    _h1_o = C_cross[sec][BLOCK_SIZE - 1][3];

                    for (int i = 0; i < num_partcarries; i++) {
                        const T p0 = in[i * order][0];
                        const T p1 = in[i * order + 1][0];
                        const T h0_val = fmaf(_h1_e, X1, fmaf(_h2_e, X0, p0));
                        const T h1_val = fmaf(_h1_o, X1, fmaf(_h2_o, X0, p1));
                        X0 = h0_val;
                        X1 = h1_val;
                    }
                }
            }

            if (lane == 0) {
                sfullc[0] = X0;
                sfullc[1] = X1;
            }
        }

        __syncthreads();
        X0 = sfullc[0];
        X1 = sfullc[1];

        // phase 3: back substitution
        in_reg[N_BLOCKS - 2] = fmaf(-he1[sec][tid], X1, fmaf(-he2[sec][tid], X0, _yi2));
        in_reg[N_BLOCKS - 1] = fmaf(-hb1[sec][tid], X1, fmaf(-hb2[sec][tid], X0, _yi1));

        if (tid == BLOCK_SIZE - 1) {
            fullcarry[chunk_id * order * N_SECTIONS + sec * order] = in_reg[N_BLOCKS - 2];
            fullcarry[chunk_id * order * N_SECTIONS + sec * order + 1] = in_reg[N_BLOCKS - 1];
        }
        __syncwarp();
        __threadfence();
        if (tid == BLOCK_SIZE - 1) {
            status[chunk_id * N_SECTIONS + sec] = full_flag;
        }

        in[tid][N_BLOCKS - 2] = in_reg[N_BLOCKS - 2];
        in[tid][N_BLOCKS - 1] = in_reg[N_BLOCKS - 1];

        __syncthreads();

        T _yi2_o, _yi1_o, _yi2_e, _yi1_e;

        if (tid == 0){
            _h2_o = cr_h[sec][N_BLOCKS_LOG2 - 1];
            _h1_o = cr_g[sec][N_BLOCKS_LOG2 - 1];
            _h2_e = cr_p[sec][N_BLOCKS_LOG2 - 1];
            _h1_e = cr_q[sec][N_BLOCKS_LOG2 - 1];
            _yi2_o = X0;
            _yi1_e = X1;
            _yi2_e = X0;
            _yi1_o = X1;
            _xi2 = X0;
            _xi1 = X1;
        } else {
            _h2_o = cr_c[sec][N_BLOCKS_LOG2];
            _h1_o = cr_d[sec][N_BLOCKS_LOG2];
            _h2_e = cr_d[sec][N_BLOCKS_LOG2];
            _h1_e = cr_c[sec][N_BLOCKS_LOG2];
            _yi1_e = in[tid - 1][N_BLOCKS - 2];
            _yi2_o = in[tid - 1][N_BLOCKS - 1];
            if (tid == 1) {
                _yi2_e = X0;
                _yi1_o = X1;
            } else {
                _yi2_e = in[tid - 2][N_BLOCKS - 2];
                _yi1_o = in[tid - 2][N_BLOCKS - 1];
            }
            _xi2 = _yi1_e;
            _xi1 = _yi2_o;
        }

        in_reg[HALF_N_BLOCKS - 2] = fmaf(-_h1_e, _yi1_e, fmaf(-_h2_e, _yi2_e, in_reg[HALF_N_BLOCKS - 2]));
        in_reg[HALF_N_BLOCKS - 1] = fmaf(-_h1_o, _yi1_o, fmaf(-_h2_o, _yi2_o, in_reg[HALF_N_BLOCKS - 1]));

        in[tid][HALF_N_BLOCKS - 2] = in_reg[HALF_N_BLOCKS - 2];
        in[tid][HALF_N_BLOCKS - 1] = in_reg[HALF_N_BLOCKS - 1];

        __syncthreads();

        T _zi2_e, _zi1_e, _zi2_o, _zi1_o;

        // Generic back-substitution rounds (for-loop form, shared-memory
        // neighbor exchange for the two-warp thread block). Loop bounds and
        // indices are compile-time constants after full unrolling. Each round
        // publishes the pair of columns the next round (or the next section's
        // phase 1) reads, then synchronizes — same choreography as the
        // hand-unrolled variant.
        #pragma unroll
        for (int r = N_BLOCKS_LOG2 - 2; r > 0; r--) {

            const int stride = 2 << r;
            const int sub = stride >> 1;

            // n = 0
            if (tid == 0) {
                _h2_o = cr_h[sec][r];
                _h1_o = cr_g[sec][r];
                _h2_e = cr_p[sec][r];
                _h1_e = cr_q[sec][r];
                _yi2_e = X0;
                _yi1_o = X1;
            } else {
                _h2_o = cr_c[sec][r + 1];
                _h1_o = cr_d[sec][r + 1];
                _h2_e = _h1_o;
                _h1_e = _h2_o;
                _yi1_o = in[tid - 1][N_BLOCKS - stride - 1];
                _yi2_e = in[tid - 1][N_BLOCKS - stride - 2];
            }
            _zi2_o = _yi2_o;
            _zi1_o = _yi1_o;
            _zi2_e = _yi2_e;
            _zi1_e = _yi1_e;

            in_reg[sub - 2] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[sub - 2]));
            in_reg[sub - 1] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[sub - 1]));

            if (sec == N_SECTIONS - 1) {
                in[tid][sub - 2] = in_reg[sub - 2];
                in[tid][sub - 1] = in_reg[sub - 1];
            }

            // n = 1
            _h2_o = cr_d[sec][r + 1];
            _h1_o = cr_c[sec][r + 1];
            _h2_e = _h2_o;
            _h1_e = _h1_o;
            _zi2_o = _xi1;
            _zi1_o = in_reg[stride - 1];
            _zi2_e = _xi2;
            _zi1_e = in_reg[stride - 2];

            in_reg[stride + sub - 2] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[stride + sub - 2]));
            in_reg[stride + sub - 1] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[stride + sub - 1]));

            if (sec == N_SECTIONS - 1) {
                in[tid][stride + sub - 2] = in_reg[stride + sub - 2];
                in[tid][stride + sub - 1] = in_reg[stride + sub - 1];
            }

            // n = 2 .. P-1
            #pragma unroll
            for (int n = 2; n < (HALF_N_BLOCKS >> r); n++) {
                _zi2_o = _zi1_o;
                _zi1_o = in_reg[stride * n - 1];
                _zi2_e = _zi1_e;
                _zi1_e = in_reg[stride * n - 2];

                in_reg[stride * n + sub - 2] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[stride * n + sub - 2]));
                in_reg[stride * n + sub - 1] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[stride * n + sub - 1]));

                if (sec == N_SECTIONS - 1) {
                    in[tid][stride * n + sub - 2] = in_reg[stride * n + sub - 2];
                    in[tid][stride * n + sub - 1] = in_reg[stride * n + sub - 1];
                }
            }

            // Publish the pair of columns the next round (or the next
            // section's phase 1 boundary read) needs from this thread.
            in[tid][N_BLOCKS - sub - 2] = in_reg[N_BLOCKS - sub - 2];
            in[tid][N_BLOCKS - sub - 1] = in_reg[N_BLOCKS - sub - 1];

            __syncthreads();
        }

    }

    for (int i = tid; i < CHUNK_SIZE; i += BLOCK_SIZE) {
        int global_idx = chunk_start + i;
        int tx = i / N_BLOCKS;
        int bx = i % N_BLOCKS;
        output[global_idx] = in[tx][bx];
    }
}


#if N_BLOCKS == 64
static __global__ __launch_bounds__(BLOCK_SIZE, N_TB_PER_SM)
void STCR_64_UNROLL_64(const T* const __restrict__ input,
        T* const __restrict__ output,
        volatile int* const __restrict__ status,
        volatile T* const __restrict__ partcarry,
        volatile T* const __restrict__ fullcarry,
        const int launch)
{
    __shared__ T in[BLOCK_SIZE][N_BLOCKS + 1];
    __shared__ int cid;
    __shared__ T sfullc[order];

    const int tid = threadIdx.x;
    const int warp = tid / warp_size;
    const int lane = tid % warp_size;

    // Launch-relative status values (see header comment).
    const int part_flag = 2 * launch + 1;
    const int full_flag = part_flag + 1;

    if (tid == 0) {
        cid = (int)(atomicAdd(&counter, 1u) - (unsigned int)launch * gridDim.x);
    }

    __syncthreads();

    const int chunk_id = cid;
    const int chunk_start = chunk_id * CHUNK_SIZE;

    for (int i = tid; i < CHUNK_SIZE; i += BLOCK_SIZE) {
        int global_idx = chunk_start + i;
        int tx = i / N_BLOCKS;
        int bx = i % N_BLOCKS;
        in[tx][bx] = input[global_idx];
    }

    T in_reg[N_BLOCKS];
    T _xc, _xi1, _xi2, _yc, _yi1, _yi2;

    __syncthreads();

    for (int sec = 0; sec < N_SECTIONS; sec++) {

        if (tid == 0) {
            _yi2 = 0;
            _yi1 = 0;
            if (chunk_id == 0) {
                _xi2 = xi2[sec];
                _xi1 = xi1[sec];
            } else if (sec == 0) {
                _xi2 = input[chunk_start - 2];
                _xi1 = input[chunk_start - 1];
            }
        } else {
            if (sec == 0) {
                _xi2 = in[tid - 1][N_BLOCKS - 2];
                _xi1 = in[tid - 1][N_BLOCKS - 1];
            }
            _yi2 = in[tid - 1][N_BLOCKS - 4];
            _yi1 = in[tid - 1][N_BLOCKS - 3];

            _yi2 = fmaf(b1[sec], _yi1, fmaf(b2[sec], _yi2, _xi2));
            _yi1 = fmaf(b1[sec], _xi2, fmaf(b2[sec], _yi1, _xi1));
        }

        #pragma unroll
        for (int n = 0; n < N_BLOCKS; n++) {

            if (sec == 0)
                _xc = in[tid][n];
            else
                _xc = in_reg[n];

            // FIR
            _yc = fmaf(b1[sec], _xi1, fmaf(b2[sec], _xi2, _xc));

            // First round CR (double sided)
            in_reg[n] = fmaf(-e[sec][0], _yi1, fmaf(f[sec][0], _yi2, _yc));

            if (n > N_BLOCKS - 3)
                in[tid][n] = in_reg[n];

            _xi2 = _xi1;
            _xi1 = _xc;
            _yi2 = _yi1;
            _yi1 = _yc;
        }

        __syncthreads();

        // Remaining CR rounds
        #pragma unroll
        for (int ro = 1; ro < N_BLOCKS_LOG2; ro++) {

            const int step = 2 << ro;
            const int sub = 1 << ro;

            #pragma unroll
            for (int i = 0; i < 2; i++) {
                const int off = sub - 2 + i;

                if (tid == 0)
                    _xi2 = 0;
                else
                    _xi2 = in[tid - 1][N_BLOCKS - 2 + i];

                __syncthreads();

                #pragma unroll
                for (int n = 0; n < N_BLOCKS; n += step) {
                    in_reg[n + off] = fmaf(-fde[sec][ro], _xi2, in_reg[n + off]);

                    _xi2 = in_reg[n + off + sub];
                    in_reg[n + off + sub] = fmaf(-e[sec][ro], in_reg[n + off], _xi2);
                }

                in[tid][N_BLOCKS - 2 + i] = in_reg[N_BLOCKS - 2 + i];

                __syncthreads();
            }
        }

        // Block filtering
        _yi2 = h0[sec][0] * in_reg[N_BLOCKS - 2];
        _yi1 = h0[sec][0] * in_reg[N_BLOCKS - 1];

        #pragma unroll
        for (int n = 1; n < BLOCK_SIZE; n++) {

            if (tid >= n) {
                _xc = in[tid - n][N_BLOCKS - 2];
                _yc = in[tid - n][N_BLOCKS - 1];
            } else {
                _xc = 0;
                _yc = 0;
            }
            _yi2 = fmaf(h0[sec][n], _xc, _yi2);
            _yi1 = fmaf(h0[sec][n], _yc, _yi1);
        }

        T _h2_e, _h1_e, _h2_o, _h1_o;
        T X0, X1;

        // phase 2 - Lookback
        if (warp == 1) {

            if (lane == warp_size - 1) {
                partcarry[chunk_id * order * N_SECTIONS + sec * order] = _yi2;
                partcarry[chunk_id * order * N_SECTIONS + sec * order + 1] = _yi1;
            }

            __syncwarp();
            __threadfence();
            if (tid == BLOCK_SIZE - 1)
                status[chunk_id * N_SECTIONS + sec] = part_flag;

            int flag = part_flag;
            bool no_zeros, has_status_2, reached_origin;

            do {
                if (chunk_id > lane) {
                    flag = status[(chunk_id - 1 - lane) * N_SECTIONS + sec];
                }
                no_zeros = !__any_sync(0xffffffff, flag < part_flag);
                has_status_2 = !__all_sync(0xffffffff, flag != full_flag) && no_zeros;
                reached_origin = (chunk_id < warp_size) && no_zeros;
            } while (!(has_status_2 || reached_origin));

            __threadfence();

            int mask = __ballot_sync(0xffffffff, flag == full_flag);

            int start_chunk;

            if (mask == 0) {
                X0 = yi2[sec];
                X1 = yi1[sec];
                start_chunk = 0;
            } else {
                const int pos = __ffs(mask) - 1;
                const int full_chunk = chunk_id - 1 - pos;
                start_chunk = full_chunk + 1;

                T fc;
                if (lane < order) {
                    fc = fullcarry[full_chunk * order * N_SECTIONS + sec * order + lane];
                }
                X0 = __shfl_sync(0xffffffff, fc, 0);
                X1 = __shfl_sync(0xffffffff, fc, 1);
            }

            const int num_partcarries = chunk_id - start_chunk;

            if (num_partcarries > 0) {
                for (int i = (start_chunk * order + lane); i < (chunk_id * order); i += warp_size) {
                    in[i - start_chunk * order][0] = partcarry[(i / order) * order * N_SECTIONS + sec * order + (i % order)];
                }
                __syncwarp();

                if (lane == 0) {
                    _h2_e = C_cross[sec][BLOCK_SIZE - 1][0];
                    _h1_e = C_cross[sec][BLOCK_SIZE - 1][1];
                    _h2_o = C_cross[sec][BLOCK_SIZE - 1][2];
                    _h1_o = C_cross[sec][BLOCK_SIZE - 1][3];

                    for (int i = 0; i < num_partcarries; i++) {
                        const T p0 = in[i * order][0];
                        const T p1 = in[i * order + 1][0];
                        const T h0_val = fmaf(_h1_e, X1, fmaf(_h2_e, X0, p0));
                        const T h1_val = fmaf(_h1_o, X1, fmaf(_h2_o, X0, p1));
                        X0 = h0_val;
                        X1 = h1_val;
                    }
                }
            }

            if (lane == 0) {
                sfullc[0] = X0;
                sfullc[1] = X1;
            }
        }

        __syncthreads();
        X0 = sfullc[0];
        X1 = sfullc[1];

        // phase 3: back substitution
        in_reg[N_BLOCKS - 2] = fmaf(-he1[sec][tid], X1, fmaf(-he2[sec][tid], X0, _yi2));
        in_reg[N_BLOCKS - 1] = fmaf(-hb1[sec][tid], X1, fmaf(-hb2[sec][tid], X0, _yi1));

        if (tid == BLOCK_SIZE - 1) {
            fullcarry[chunk_id * order * N_SECTIONS + sec * order] = in_reg[N_BLOCKS - 2];
            fullcarry[chunk_id * order * N_SECTIONS + sec * order + 1] = in_reg[N_BLOCKS - 1];
        }
        __syncwarp();
        __threadfence();
        if (tid == BLOCK_SIZE - 1) {
            status[chunk_id * N_SECTIONS + sec] = full_flag;
        }

        in[tid][N_BLOCKS - 2] = in_reg[N_BLOCKS - 2];
        in[tid][N_BLOCKS - 1] = in_reg[N_BLOCKS - 1];

        __syncthreads();

        T _yi2_o, _yi1_o, _yi2_e, _yi1_e;

        if (tid == 0){
            _h2_o = cr_h[sec][N_BLOCKS_LOG2 - 1];
            _h1_o = cr_g[sec][N_BLOCKS_LOG2 - 1];
            _h2_e = cr_p[sec][N_BLOCKS_LOG2 - 1];
            _h1_e = cr_q[sec][N_BLOCKS_LOG2 - 1];
            _yi2_o = X0;
            _yi1_e = X1;
            _yi2_e = X0;
            _yi1_o = X1;
            _xi2 = X0;
            _xi1 = X1;
        } else {
            _h2_o = cr_c[sec][N_BLOCKS_LOG2];
            _h1_o = cr_d[sec][N_BLOCKS_LOG2];
            _h2_e = cr_d[sec][N_BLOCKS_LOG2];
            _h1_e = cr_c[sec][N_BLOCKS_LOG2];
            _yi1_e = in[tid - 1][N_BLOCKS - 2];
            _yi2_o = in[tid - 1][N_BLOCKS - 1];
            if (tid == 1) {
                _yi2_e = X0;
                _yi1_o = X1;
            } else {
                _yi2_e = in[tid - 2][N_BLOCKS - 2];
                _yi1_o = in[tid - 2][N_BLOCKS - 1];
            }
            _xi2 = _yi1_e;
            _xi1 = _yi2_o;
        }

        in_reg[HALF_N_BLOCKS - 2] = fmaf(-_h1_e, _yi1_e, fmaf(-_h2_e, _yi2_e, in_reg[HALF_N_BLOCKS - 2]));
        in_reg[HALF_N_BLOCKS - 1] = fmaf(-_h1_o, _yi1_o, fmaf(-_h2_o, _yi2_o, in_reg[HALF_N_BLOCKS - 1]));

        in[tid][HALF_N_BLOCKS - 2] = in_reg[HALF_N_BLOCKS - 2];
        in[tid][HALF_N_BLOCKS - 1] = in_reg[HALF_N_BLOCKS - 1];

        __syncthreads();

        T _zi2_e, _zi1_e, _zi2_o, _zi1_o;


        //=============================================================================
        // ro = 4: P=2, step=32, sub=16
        //=============================================================================

        // n = 0
        if (tid == 0) {
            _h2_o = cr_h[sec][4];
            _h1_o = cr_g[sec][4];
            _h2_e = cr_p[sec][4];
            _h1_e = cr_q[sec][4];
            _yi2_e = X0;
            _yi1_o = X1;
        } else {
            _h2_o = cr_c[sec][5];
            _h1_o = cr_d[sec][5];
            _h2_e = _h1_o;
            _h1_e = _h2_o;
            _yi1_o = in[tid - 1][31];
            _yi2_e = in[tid - 1][30];
        }

        _zi2_o = _yi2_o;
        _zi1_o = _yi1_o;
        _zi2_e = _yi2_e;
        _zi1_e = _yi1_e;

        in_reg[14] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[14]));
        in_reg[15] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[15]));

        if (sec == N_SECTIONS - 1) {
            in[tid][14] = in_reg[14];
            in[tid][15] = in_reg[15];
        }

        // n = 1
        _h2_o = cr_d[sec][5];
        _h1_o = cr_c[sec][5];
        _h2_e = _h2_o;
        _h1_e = _h1_o;
        _zi2_o = _xi1;
        _zi1_o = in_reg[31];
        _zi2_e = _xi2;
        _zi1_e = in_reg[30];

        in_reg[46] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[46]));
        in_reg[47] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[47]));

        if (sec == N_SECTIONS - 1) {
            in[tid][46] = in_reg[46];
            in[tid][47] = in_reg[47];
        }

        if (sec < N_SECTIONS - 1) {
            in[tid][46] = in_reg[46];
            in[tid][47] = in_reg[47];
        }

        __syncthreads();

        //=============================================================================
        // ro = 3: P=4, step=16, sub=8
        //=============================================================================

        // n = 0
        if (tid == 0) {
            _h2_o = cr_h[sec][3];
            _h1_o = cr_g[sec][3];
            _h2_e = cr_p[sec][3];
            _h1_e = cr_q[sec][3];
            _yi2_e = X0;
            _yi1_o = X1;
        } else {
            _h2_o = cr_c[sec][4];
            _h1_o = cr_d[sec][4];
            _h2_e = _h1_o;
            _h1_e = _h2_o;
            _yi1_o = in[tid - 1][47];
            _yi2_e = in[tid - 1][46];
        }

        _zi2_o = _yi2_o;
        _zi1_o = _yi1_o;
        _zi2_e = _yi2_e;
        _zi1_e = _yi1_e;

        in_reg[6] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[6]));
        in_reg[7] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[7]));

        if (sec == N_SECTIONS - 1) {
            in[tid][6] = in_reg[6];
            in[tid][7] = in_reg[7];
        }

        // n = 1
        _h2_o = cr_d[sec][4];
        _h1_o = cr_c[sec][4];
        _h2_e = _h2_o;
        _h1_e = _h1_o;
        _zi2_o = _xi1;
        _zi1_o = in_reg[15];
        _zi2_e = _xi2;
        _zi1_e = in_reg[14];

        in_reg[22] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[22]));
        in_reg[23] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[23]));

        if (sec == N_SECTIONS - 1) {
            in[tid][22] = in_reg[22];
            in[tid][23] = in_reg[23];
        }

        // n = 2
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[31];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[30];

        in_reg[38] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[38]));
        in_reg[39] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[39]));

        if (sec == N_SECTIONS - 1) {
            in[tid][38] = in_reg[38];
            in[tid][39] = in_reg[39];
        }

        // n = 3
        _zi2_o = _zi1_o;
        _zi1_o = in_reg[47];
        _zi2_e = _zi1_e;
        _zi1_e = in_reg[46];

        in_reg[54] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[54]));
        in_reg[55] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[55]));

        if (sec == N_SECTIONS - 1) {
            in[tid][54] = in_reg[54];
            in[tid][55] = in_reg[55];
        }

        if (sec < N_SECTIONS - 1) {
            in[tid][54] = in_reg[54];
            in[tid][55] = in_reg[55];
        }

        __syncthreads();

        //=============================================================================
        // ro = 2: P=8, step=8, sub=4
        //=============================================================================

        // n = 0
        if (tid == 0) {
            _h2_o = cr_h[sec][2];
            _h1_o = cr_g[sec][2];
            _h2_e = cr_p[sec][2];
            _h1_e = cr_q[sec][2];
            _yi2_e = X0;
            _yi1_o = X1;
        } else {
            _h2_o = cr_c[sec][3];
            _h1_o = cr_d[sec][3];
            _h2_e = _h1_o;
            _h1_e = _h2_o;
            _yi1_o = in[tid - 1][55];
            _yi2_e = in[tid - 1][54];
        }

        _zi2_o = _yi2_o;
        _zi1_o = _yi1_o;
        _zi2_e = _yi2_e;
        _zi1_e = _yi1_e;

        in_reg[2] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[2]));
        in_reg[3] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[3]));

        if (sec == N_SECTIONS - 1) { in[tid][2] = in_reg[2]; in[tid][3] = in_reg[3]; }

        // n = 1
        _h2_o = cr_d[sec][3]; _h1_o = cr_c[sec][3]; _h2_e = _h2_o; _h1_e = _h1_o;
        _zi2_o = _xi1; _zi1_o = in_reg[7]; _zi2_e = _xi2; _zi1_e = in_reg[6];
        in_reg[10] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[10]));
        in_reg[11] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[11]));
        if (sec == N_SECTIONS - 1) { in[tid][10] = in_reg[10]; in[tid][11] = in_reg[11]; }

        // n = 2
        _zi2_o = _zi1_o; _zi1_o = in_reg[15]; _zi2_e = _zi1_e; _zi1_e = in_reg[14];
        in_reg[18] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[18]));
        in_reg[19] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[19]));
        if (sec == N_SECTIONS - 1) { in[tid][18] = in_reg[18]; in[tid][19] = in_reg[19]; }

        // n = 3
        _zi2_o = _zi1_o; _zi1_o = in_reg[23]; _zi2_e = _zi1_e; _zi1_e = in_reg[22];
        in_reg[26] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[26]));
        in_reg[27] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[27]));
        if (sec == N_SECTIONS - 1) { in[tid][26] = in_reg[26]; in[tid][27] = in_reg[27]; }

        // n = 4
        _zi2_o = _zi1_o; _zi1_o = in_reg[31]; _zi2_e = _zi1_e; _zi1_e = in_reg[30];
        in_reg[34] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[34]));
        in_reg[35] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[35]));
        if (sec == N_SECTIONS - 1) { in[tid][34] = in_reg[34]; in[tid][35] = in_reg[35]; }

        // n = 5
        _zi2_o = _zi1_o; _zi1_o = in_reg[39]; _zi2_e = _zi1_e; _zi1_e = in_reg[38];
        in_reg[42] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[42]));
        in_reg[43] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[43]));
        if (sec == N_SECTIONS - 1) { in[tid][42] = in_reg[42]; in[tid][43] = in_reg[43]; }

        // n = 6
        _zi2_o = _zi1_o; _zi1_o = in_reg[47]; _zi2_e = _zi1_e; _zi1_e = in_reg[46];
        in_reg[50] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[50]));
        in_reg[51] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[51]));
        if (sec == N_SECTIONS - 1) { in[tid][50] = in_reg[50]; in[tid][51] = in_reg[51]; }

        // n = 7
        _zi2_o = _zi1_o; _zi1_o = in_reg[55]; _zi2_e = _zi1_e; _zi1_e = in_reg[54];
        in_reg[58] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[58]));
        in_reg[59] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[59]));
        if (sec == N_SECTIONS - 1) { in[tid][58] = in_reg[58]; in[tid][59] = in_reg[59]; }

        if (sec < N_SECTIONS - 1) {
            in[tid][58] = in_reg[58];
            in[tid][59] = in_reg[59];
        }

        __syncthreads();

        //=============================================================================
        // ro = 1: P=16, step=4, sub=2
        //=============================================================================

        // n = 0
        if (tid == 0) {
            _h2_o = cr_h[sec][1]; _h1_o = cr_g[sec][1];
            _h2_e = cr_p[sec][1]; _h1_e = cr_q[sec][1];
            _yi2_e = X0; _yi1_o = X1;
        } else {
            _h2_o = cr_c[sec][2]; _h1_o = cr_d[sec][2];
            _h2_e = _h1_o; _h1_e = _h2_o;
            _yi1_o = in[tid - 1][59]; _yi2_e = in[tid - 1][58];
        }
        _zi2_o = _yi2_o; _zi1_o = _yi1_o; _zi2_e = _yi2_e; _zi1_e = _yi1_e;

        in_reg[0] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[0]));
        in_reg[1] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[1]));
        if (sec == N_SECTIONS - 1) { in[tid][0] = in_reg[0]; in[tid][1] = in_reg[1]; }

        // n = 1
        _h2_o = cr_d[sec][2]; _h1_o = cr_c[sec][2]; _h2_e = _h2_o; _h1_e = _h1_o;
        _zi2_o = _xi1; _zi1_o = in_reg[3]; _zi2_e = _xi2; _zi1_e = in_reg[2];
        in_reg[4] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[4]));
        in_reg[5] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[5]));
        if (sec == N_SECTIONS - 1) { in[tid][4] = in_reg[4]; in[tid][5] = in_reg[5]; }

        // n = 2
        _zi2_o = _zi1_o; _zi1_o = in_reg[7]; _zi2_e = _zi1_e; _zi1_e = in_reg[6];
        in_reg[8] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[8]));
        in_reg[9] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[9]));
        if (sec == N_SECTIONS - 1) { in[tid][8] = in_reg[8]; in[tid][9] = in_reg[9]; }

        // n = 3
        _zi2_o = _zi1_o; _zi1_o = in_reg[11]; _zi2_e = _zi1_e; _zi1_e = in_reg[10];
        in_reg[12] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[12]));
        in_reg[13] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[13]));
        if (sec == N_SECTIONS - 1) { in[tid][12] = in_reg[12]; in[tid][13] = in_reg[13]; }

        // n = 4
        _zi2_o = _zi1_o; _zi1_o = in_reg[15]; _zi2_e = _zi1_e; _zi1_e = in_reg[14];
        in_reg[16] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[16]));
        in_reg[17] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[17]));
        if (sec == N_SECTIONS - 1) { in[tid][16] = in_reg[16]; in[tid][17] = in_reg[17]; }

        // n = 5
        _zi2_o = _zi1_o; _zi1_o = in_reg[19]; _zi2_e = _zi1_e; _zi1_e = in_reg[18];
        in_reg[20] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[20]));
        in_reg[21] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[21]));
        if (sec == N_SECTIONS - 1) { in[tid][20] = in_reg[20]; in[tid][21] = in_reg[21]; }

        // n = 6
        _zi2_o = _zi1_o; _zi1_o = in_reg[23]; _zi2_e = _zi1_e; _zi1_e = in_reg[22];
        in_reg[24] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[24]));
        in_reg[25] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[25]));
        if (sec == N_SECTIONS - 1) { in[tid][24] = in_reg[24]; in[tid][25] = in_reg[25]; }

        // n = 7
        _zi2_o = _zi1_o; _zi1_o = in_reg[27]; _zi2_e = _zi1_e; _zi1_e = in_reg[26];
        in_reg[28] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[28]));
        in_reg[29] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[29]));
        if (sec == N_SECTIONS - 1) { in[tid][28] = in_reg[28]; in[tid][29] = in_reg[29]; }

        // n = 8
        _zi2_o = _zi1_o; _zi1_o = in_reg[31]; _zi2_e = _zi1_e; _zi1_e = in_reg[30];
        in_reg[32] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[32]));
        in_reg[33] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[33]));
        if (sec == N_SECTIONS - 1) { in[tid][32] = in_reg[32]; in[tid][33] = in_reg[33]; }

        // n = 9
        _zi2_o = _zi1_o; _zi1_o = in_reg[35]; _zi2_e = _zi1_e; _zi1_e = in_reg[34];
        in_reg[36] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[36]));
        in_reg[37] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[37]));
        if (sec == N_SECTIONS - 1) { in[tid][36] = in_reg[36]; in[tid][37] = in_reg[37]; }

        // n = 10
        _zi2_o = _zi1_o; _zi1_o = in_reg[39]; _zi2_e = _zi1_e; _zi1_e = in_reg[38];
        in_reg[40] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[40]));
        in_reg[41] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[41]));
        if (sec == N_SECTIONS - 1) { in[tid][40] = in_reg[40]; in[tid][41] = in_reg[41]; }

        // n = 11
        _zi2_o = _zi1_o; _zi1_o = in_reg[43]; _zi2_e = _zi1_e; _zi1_e = in_reg[42];
        in_reg[44] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[44]));
        in_reg[45] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[45]));
        if (sec == N_SECTIONS - 1) { in[tid][44] = in_reg[44]; in[tid][45] = in_reg[45]; }

        // n = 12
        _zi2_o = _zi1_o; _zi1_o = in_reg[47]; _zi2_e = _zi1_e; _zi1_e = in_reg[46];
        in_reg[48] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[48]));
        in_reg[49] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[49]));
        if (sec == N_SECTIONS - 1) { in[tid][48] = in_reg[48]; in[tid][49] = in_reg[49]; }

        // n = 13
        _zi2_o = _zi1_o; _zi1_o = in_reg[51]; _zi2_e = _zi1_e; _zi1_e = in_reg[50];
        in_reg[52] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[52]));
        in_reg[53] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[53]));
        if (sec == N_SECTIONS - 1) { in[tid][52] = in_reg[52]; in[tid][53] = in_reg[53]; }

        // n = 14
        _zi2_o = _zi1_o; _zi1_o = in_reg[55]; _zi2_e = _zi1_e; _zi1_e = in_reg[54];
        in_reg[56] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[56]));
        in_reg[57] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[57]));
        if (sec == N_SECTIONS - 1) { in[tid][56] = in_reg[56]; in[tid][57] = in_reg[57]; }

        // n = 15
        _zi2_o = _zi1_o; _zi1_o = in_reg[59]; _zi2_e = _zi1_e; _zi1_e = in_reg[58];
        in_reg[60] = fmaf(-_h1_e, _zi1_e, fmaf(-_h2_e, _zi2_e, in_reg[60]));
        in_reg[61] = fmaf(-_h1_o, _zi1_o, fmaf(-_h2_o, _zi2_o, in_reg[61]));

        in[tid][60] = in_reg[60];
        in[tid][61] = in_reg[61];

        __syncthreads();

    }

    for (int i = tid; i < CHUNK_SIZE; i += BLOCK_SIZE) {
        int global_idx = chunk_start + i;
        int tx = i / N_BLOCKS;
        int bx = i % N_BLOCKS;
        output[global_idx] = in[tx][bx];
    }
}


#endif  // N_BLOCKS == 64

// ==========================================================================
// Kernel selection macro
// ==========================================================================
#ifdef STCR_HANDUNROLLED
    #if BLOCK_SIZE == 32 && N_BLOCKS == 32
        #define KERNEL_FUNC STCR_32_UNROLL_32
    #elif BLOCK_SIZE == 32 && N_BLOCKS == 64
        #define KERNEL_FUNC STCR_32_UNROLL_64
    #elif BLOCK_SIZE == 32 && N_BLOCKS == 128
        #define KERNEL_FUNC STCR_32_UNROLL_128
    #elif BLOCK_SIZE == 64 && N_BLOCKS == 64
        #define KERNEL_FUNC STCR_64_UNROLL_64
    #else
        #error "STCR_HANDUNROLLED supports (BLOCK_SIZE, N_BLOCKS) = (32,32), (32,64), (32,128), (64,64) only"
    #endif
#else
    #if BLOCK_SIZE == 32
        #define KERNEL_FUNC STCR_32_LOOP
    #elif BLOCK_SIZE == 64
        #define KERNEL_FUNC STCR_64_LOOP
    #else
        #error "STCR kernels only support BLOCK_SIZE 32 or 64"
    #endif
#endif

#endif // STCR_KERNELS_CUH
