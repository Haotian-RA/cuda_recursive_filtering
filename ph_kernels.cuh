#ifndef PH_KERNELS_CUH
#define PH_KERNELS_CUH

// PH factorization kernels for cascaded biquad IIR filtering.
//
// REQUIRES the following macros to be defined before inclusion:
//   N_SECTIONS   — number of cascaded biquads
//   BLOCK_SIZE   — threads per TB (32 or 64)
//   N_BLOCKS     — register-resident samples per thread
//   N_TB_PER_SM  — occupancy target for __launch_bounds__
//
// Exposes:
//   KERNEL_FUNC              — selected PH_32 or PH_64 based on BLOCK_SIZE
//   setup_kernel_coefficients(sos) — copies SOS coefficients to constant memory
//
// Launch-versioned status flags:
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

// Thread-block dimensions for the <<<>>> launch (PH uses 1D blocks).
#define KERNEL_TB_DIM BLOCK_SIZE

#include "iir_utils.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <array>
#include <cassert>

static const int order = 2;
static const int warp_size = 32;

static __device__ unsigned int counter = 0;

static __constant__ T xi1[N_SECTIONS], xi2[N_SECTIONS], yi1[N_SECTIONS], yi2[N_SECTIONS];
static __constant__ T b1[N_SECTIONS], b2[N_SECTIONS], a1[N_SECTIONS], a2[N_SECTIONS];
static __constant__ T C_cross[N_SECTIONS][BLOCK_SIZE][4];
static __device__   T C_inner[N_SECTIONS][BLOCK_SIZE * 4];
static __constant__ T h1[N_SECTIONS][N_BLOCKS], h2[N_SECTIONS][N_BLOCKS];


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

inline void C_power2(const T* C_in, int n, T* result, int stride) {
    result[0 * stride] = 1.0f; result[1 * stride] = 0.0f;
    result[2 * stride] = 0.0f; result[3 * stride] = 1.0f;

    if (n == 0) return;
    if (n == 1) {
        result[0 * stride] = C_in[0]; result[1 * stride] = C_in[1];
        result[2 * stride] = C_in[2]; result[3 * stride] = C_in[3];
        return;
    }

    T base[4] = {C_in[0], C_in[1], C_in[2], C_in[3]};

    while (n > 0) {
        if (n & 1) {
            T temp[4];
            temp[0] = result[0 * stride] * base[0] + result[1 * stride] * base[2];
            temp[1] = result[0 * stride] * base[1] + result[1 * stride] * base[3];
            temp[2] = result[2 * stride] * base[0] + result[3 * stride] * base[2];
            temp[3] = result[2 * stride] * base[1] + result[3 * stride] * base[3];

            result[0 * stride] = temp[0]; result[1 * stride] = temp[1];
            result[2 * stride] = temp[2]; result[3 * stride] = temp[3];
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
// Called from test drivers after tf2sos.
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

    T h_h1[N_SECTIONS][N_BLOCKS], h_h2[N_SECTIONS][N_BLOCKS];
    T h_C_cross[N_SECTIONS][BLOCK_SIZE][4];
    T h_C_inner[N_SECTIONS][4 * BLOCK_SIZE];

    for (int sec = 0; sec < N_SECTIONS; sec++) {
        impulse_response(h_a1[sec], h_a2[sec], h_b1[sec], h_b2[sec],
                         h_h1[sec], h_h2[sec]);

        T h_C[4] = {h_h2[sec][N_BLOCKS - 2], h_h1[sec][N_BLOCKS - 2],
                    h_h2[sec][N_BLOCKS - 1], h_h1[sec][N_BLOCKS - 1]};

        for (int n = 0; n < BLOCK_SIZE; n++)
            C_power(h_C, n + 1, h_C_cross[sec][n]);

        for (int n = 0; n < BLOCK_SIZE; n++)
            C_power2(h_C, n + 1, &h_C_inner[sec][n], BLOCK_SIZE);
    }

    assert(cudaSuccess == cudaMemcpyToSymbol(b1,  h_b1,  N_SECTIONS * sizeof(T)));
    assert(cudaSuccess == cudaMemcpyToSymbol(b2,  h_b2,  N_SECTIONS * sizeof(T)));
    assert(cudaSuccess == cudaMemcpyToSymbol(a1,  h_a1,  N_SECTIONS * sizeof(T)));
    assert(cudaSuccess == cudaMemcpyToSymbol(a2,  h_a2,  N_SECTIONS * sizeof(T)));
    assert(cudaSuccess == cudaMemcpyToSymbol(xi1, h_xi1, N_SECTIONS * sizeof(T)));
    assert(cudaSuccess == cudaMemcpyToSymbol(xi2, h_xi2, N_SECTIONS * sizeof(T)));
    assert(cudaSuccess == cudaMemcpyToSymbol(yi1, h_yi1, N_SECTIONS * sizeof(T)));
    assert(cudaSuccess == cudaMemcpyToSymbol(yi2, h_yi2, N_SECTIONS * sizeof(T)));
    assert(cudaSuccess == cudaMemcpyToSymbol(C_cross, h_C_cross,
                                              N_SECTIONS * BLOCK_SIZE * 4 * sizeof(T)));
    assert(cudaSuccess == cudaMemcpyToSymbol(h1, h_h1, N_SECTIONS * N_BLOCKS * sizeof(T)));
    assert(cudaSuccess == cudaMemcpyToSymbol(h2, h_h2, N_SECTIONS * N_BLOCKS * sizeof(T)));
    assert(cudaSuccess == cudaMemcpyToSymbol(C_inner, h_C_inner,
                                              N_SECTIONS * BLOCK_SIZE * 4 * sizeof(T)));
}


// ==========================================================================
// PH kernels — verbatim from the original main_PH.cu
// ==========================================================================
#define CHUNK_SIZE (BLOCK_SIZE * N_BLOCKS)

static __global__ __launch_bounds__(BLOCK_SIZE, N_TB_PER_SM)
void PH_32(const T* const __restrict__ input,
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

        if (tid == 0) {
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
        }

        #pragma unroll
        for (int n = 0; n < N_BLOCKS; n++){

            if (sec == 0)
                _xc = in[tid][n];
            else
                _xc = in_reg[n];

            _yc = fmaf(b1[sec], _xi1, fmaf(b2[sec], _xi2, _xc));

            if (n == 1)
                _yc = fmaf(a1[sec], _yi1, _yc);
            else if (n > 1)
                _yc = fmaf(a1[sec], _yi1, fmaf(a2[sec], _yi2, _yc));

            if (n < N_BLOCKS - 2)
                in_reg[n] = _yc;

            _xi2 = _xi1;
            _xi1 = _xc;
            _yi2 = _yi1;
            _yi1 = _yc;
        }

        int cond;
        T spc2, spc1;
        T help0, help1, help2, help3;

        help0 = C_cross[sec][0][0];
        help1 = C_cross[sec][0][1];
        help2 = C_cross[sec][0][2];
        help3 = C_cross[sec][0][3];

        cond = ((lane & 1) != 0);
        spc2 = __shfl_sync(0xffffffff, _yi2, 0, 2);
        spc1 = __shfl_sync(0xffffffff, _yi1, 0, 2);
        if (cond) {
            _yi2 = fmaf(help1, spc1, fmaf(help0, spc2, _yi2));
            _yi1 = fmaf(help3, spc1, fmaf(help2, spc2, _yi1));
        }

        if (lane < 2) {
            T* Clocal = (T*)C_cross[sec][lane];
            help0 = Clocal[0];
            help1 = Clocal[1];
            help2 = Clocal[2];
            help3 = Clocal[3];
        }
        help0 = __shfl_sync(0xffffffff, help0, lane % 2);
        help1 = __shfl_sync(0xffffffff, help1, lane % 2);
        help2 = __shfl_sync(0xffffffff, help2, lane % 2);
        help3 = __shfl_sync(0xffffffff, help3, lane % 2);

        cond = ((lane & 2) != 0);
        spc2 = __shfl_sync(0xffffffff, _yi2, 1, 4);
        spc1 = __shfl_sync(0xffffffff, _yi1, 1, 4);
        if (cond) {
            _yi2 = fmaf(help1, spc1, fmaf(help0, spc2, _yi2));
            _yi1 = fmaf(help3, spc1, fmaf(help2, spc2, _yi1));
        }

        if (lane < 4) {
            help0 = C_cross[sec][lane][0];
            help1 = C_cross[sec][lane][1];
            help2 = C_cross[sec][lane][2];
            help3 = C_cross[sec][lane][3];
        }
        help0 = __shfl_sync(0xffffffff, help0, lane % 4);
        help1 = __shfl_sync(0xffffffff, help1, lane % 4);
        help2 = __shfl_sync(0xffffffff, help2, lane % 4);
        help3 = __shfl_sync(0xffffffff, help3, lane % 4);

        cond = ((lane & 4) != 0);
        spc2 = __shfl_sync(0xffffffff, _yi2, 3, 8);
        spc1 = __shfl_sync(0xffffffff, _yi1, 3, 8);
        if (cond) {
            _yi2 = fmaf(help1, spc1, fmaf(help0, spc2, _yi2));
            _yi1 = fmaf(help3, spc1, fmaf(help2, spc2, _yi1));
        }


        if (lane < 8) {
            help0 = C_cross[sec][lane][0];
            help1 = C_cross[sec][lane][1];
            help2 = C_cross[sec][lane][2];
            help3 = C_cross[sec][lane][3];
        }
        help0 = __shfl_sync(0xffffffff, help0, lane % 8);
        help1 = __shfl_sync(0xffffffff, help1, lane % 8);
        help2 = __shfl_sync(0xffffffff, help2, lane % 8);
        help3 = __shfl_sync(0xffffffff, help3, lane % 8);

        cond = ((lane & 8) != 0);
        spc2 = __shfl_sync(0xffffffff, _yi2, 7, 16);
        spc1 = __shfl_sync(0xffffffff, _yi1, 7, 16);
        if (cond) {
            _yi2 = fmaf(help1, spc1, fmaf(help0, spc2, _yi2));
            _yi1 = fmaf(help3, spc1, fmaf(help2, spc2, _yi1));
        }

        if (lane < 16) {
            help0 = C_cross[sec][lane][0];
            help1 = C_cross[sec][lane][1];
            help2 = C_cross[sec][lane][2];
            help3 = C_cross[sec][lane][3];
        }
        help0 = __shfl_sync(0xffffffff, help0, lane % 16);
        help1 = __shfl_sync(0xffffffff, help1, lane % 16);
        help2 = __shfl_sync(0xffffffff, help2, lane % 16);
        help3 = __shfl_sync(0xffffffff, help3, lane % 16);

        cond = ((lane & 16) != 0);
        spc2 = __shfl_sync(0xffffffff, _yi2, 15, 32);
        spc1 = __shfl_sync(0xffffffff, _yi1, 15, 32);
        if (cond) {
            _yi2 = fmaf(help1, spc1, fmaf(help0, spc2, _yi2));
            _yi1 = fmaf(help3, spc1, fmaf(help2, spc2, _yi1));
        }

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

            help0 = C_cross[sec][BLOCK_SIZE - 1][0];
            help1 = C_cross[sec][BLOCK_SIZE - 1][1];
            help2 = C_cross[sec][BLOCK_SIZE - 1][2];
            help3 = C_cross[sec][BLOCK_SIZE - 1][3];

            for (int i = 0; i < num_partcarries; i++) {
                const T p0 = spartc[i * order];
                const T p1 = spartc[i * order + 1];
                const T h0 = fmaf(help1, X1, fmaf(help0, X0, p0));
                const T h1 = fmaf(help3, X1, fmaf(help2, X0, p1));
                X0 = h0;
                X1 = h1;
            }
        }

        _xi2 = X0;
        _xi1 = X1;

        in_reg[N_BLOCKS - 2] = fmaf(C_inner[sec][tid + BLOCK_SIZE], _xi1,
                                     fmaf(C_inner[sec][tid], _xi2, _yi2));
        in_reg[N_BLOCKS - 1] = fmaf(C_inner[sec][tid + 3*BLOCK_SIZE], _xi1,
                                     fmaf(C_inner[sec][tid + 2*BLOCK_SIZE], _xi2, _yi1));

        if (tid == BLOCK_SIZE - 1) {
            fullcarry[chunk_id * order * N_SECTIONS + sec * order] = in_reg[N_BLOCKS - 2];
            fullcarry[chunk_id * order * N_SECTIONS + sec * order + 1] = in_reg[N_BLOCKS - 1];
        }
        __syncwarp();
        __threadfence();
        if (tid == BLOCK_SIZE - 1) {
            status[chunk_id * N_SECTIONS + sec] = full_flag;
        }

        spc2 = __shfl_up_sync(0xffffffff, in_reg[N_BLOCKS - 2], 1);
        spc1 = __shfl_up_sync(0xffffffff, in_reg[N_BLOCKS - 1], 1);
        if (tid > 0) {
            _xi2 = spc2;
            _xi1 = spc1;
        }

        #pragma unroll
        for (int n = 0; n < N_BLOCKS - 2; n++)
            in_reg[n] = fmaf(h1[sec][n], _xi1, fmaf(h2[sec][n], _xi2, in_reg[n]));

        if (sec == N_SECTIONS - 1) {
            #pragma unroll
            for (int n = 0; n < N_BLOCKS; n++)
                in[tid][n] = in_reg[n];
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


static __global__ __launch_bounds__(BLOCK_SIZE, N_TB_PER_SM)
void PH_64(const T* const __restrict__ input,
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
        }

        #pragma unroll
        for (int n = 0; n < N_BLOCKS; n++){

            if (sec == 0)
                _xc = in[tid][n];
            else
                _xc = in_reg[n];

            _yc = fmaf(b1[sec], _xi1, fmaf(b2[sec], _xi2, _xc));

            if (n == 1)
                _yc = fmaf(a1[sec], _yi1, _yc);
            else if (n > 1)
                _yc = fmaf(a1[sec], _yi1, fmaf(a2[sec], _yi2, _yc));

            if (n < N_BLOCKS - 2)
                in_reg[n] = _yc;

            _xi2 = _xi1;
            _xi1 = _xc;
            _yi2 = _yi1;
            _yi1 = _yc;
        }

        int cond;
        T spc2, spc1;
        T help0, help1, help2, help3;

        help0 = C_cross[sec][0][0];
        help1 = C_cross[sec][0][1];
        help2 = C_cross[sec][0][2];
        help3 = C_cross[sec][0][3];

        cond = ((lane & 1) != 0);
        spc2 = __shfl_sync(0xffffffff, _yi2, 0, 2);
        spc1 = __shfl_sync(0xffffffff, _yi1, 0, 2);
        if (cond) {
            _yi2 = fmaf(help1, spc1, fmaf(help0, spc2, _yi2));
            _yi1 = fmaf(help3, spc1, fmaf(help2, spc2, _yi1));
        }


        if (lane < 2) {
            T* Clocal = (T*)C_cross[sec][lane];
            help0 = Clocal[0];
            help1 = Clocal[1];
            help2 = Clocal[2];
            help3 = Clocal[3];
        }
        help0 = __shfl_sync(0xffffffff, help0, lane % 2);
        help1 = __shfl_sync(0xffffffff, help1, lane % 2);
        help2 = __shfl_sync(0xffffffff, help2, lane % 2);
        help3 = __shfl_sync(0xffffffff, help3, lane % 2);

        cond = ((lane & 2) != 0);
        spc2 = __shfl_sync(0xffffffff, _yi2, 1, 4);
        spc1 = __shfl_sync(0xffffffff, _yi1, 1, 4);
        if (cond) {
            _yi2 = fmaf(help1, spc1, fmaf(help0, spc2, _yi2));
            _yi1 = fmaf(help3, spc1, fmaf(help2, spc2, _yi1));
        }


        if (lane < 4) {
            help0 = C_cross[sec][lane][0];
            help1 = C_cross[sec][lane][1];
            help2 = C_cross[sec][lane][2];
            help3 = C_cross[sec][lane][3];
        }
        help0 = __shfl_sync(0xffffffff, help0, lane % 4);
        help1 = __shfl_sync(0xffffffff, help1, lane % 4);
        help2 = __shfl_sync(0xffffffff, help2, lane % 4);
        help3 = __shfl_sync(0xffffffff, help3, lane % 4);

        cond = ((lane & 4) != 0);
        spc2 = __shfl_sync(0xffffffff, _yi2, 3, 8);
        spc1 = __shfl_sync(0xffffffff, _yi1, 3, 8);
        if (cond) {
            _yi2 = fmaf(help1, spc1, fmaf(help0, spc2, _yi2));
            _yi1 = fmaf(help3, spc1, fmaf(help2, spc2, _yi1));
        }


        if (lane < 8) {
            help0 = C_cross[sec][lane][0];
            help1 = C_cross[sec][lane][1];
            help2 = C_cross[sec][lane][2];
            help3 = C_cross[sec][lane][3];
        }
        help0 = __shfl_sync(0xffffffff, help0, lane % 8);
        help1 = __shfl_sync(0xffffffff, help1, lane % 8);
        help2 = __shfl_sync(0xffffffff, help2, lane % 8);
        help3 = __shfl_sync(0xffffffff, help3, lane % 8);

        cond = ((lane & 8) != 0);
        spc2 = __shfl_sync(0xffffffff, _yi2, 7, 16);
        spc1 = __shfl_sync(0xffffffff, _yi1, 7, 16);
        if (cond) {
            _yi2 = fmaf(help1, spc1, fmaf(help0, spc2, _yi2));
            _yi1 = fmaf(help3, spc1, fmaf(help2, spc2, _yi1));
        }

        if (lane < 16) {
            help0 = C_cross[sec][lane][0];
            help1 = C_cross[sec][lane][1];
            help2 = C_cross[sec][lane][2];
            help3 = C_cross[sec][lane][3];
        }
        help0 = __shfl_sync(0xffffffff, help0, lane % 16);
        help1 = __shfl_sync(0xffffffff, help1, lane % 16);
        help2 = __shfl_sync(0xffffffff, help2, lane % 16);
        help3 = __shfl_sync(0xffffffff, help3, lane % 16);

        cond = ((lane & 16) != 0);
        spc2 = __shfl_sync(0xffffffff, _yi2, 15, 32);
        spc1 = __shfl_sync(0xffffffff, _yi1, 15, 32);
        if (cond) {
            _yi2 = fmaf(help1, spc1, fmaf(help0, spc2, _yi2));
            _yi1 = fmaf(help3, spc1, fmaf(help2, spc2, _yi1));
        }

        if (tid == warp_size - 1) {
            sfullc[0] = _yi2;
            sfullc[1] = _yi1;
        }
        __syncthreads();

        if (warp == 1) {
            T* Clocal = (T*)C_cross[sec][lane];
            help0 = Clocal[0];
            help1 = Clocal[1];
            help2 = Clocal[2];
            help3 = Clocal[3];

            _yi2 = fmaf(help1, sfullc[1], fmaf(help0, sfullc[0], _yi2));
            _yi1 = fmaf(help3, sfullc[1], fmaf(help2, sfullc[0], _yi1));

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
                    in[i - start_chunk * order][0] = partcarry[(i / order) * order * N_SECTIONS + sec * order + (i % order)];
                }
                __syncwarp();

                if (lane == 0) {
                    help0 = C_cross[sec][BLOCK_SIZE - 1][0];
                    help1 = C_cross[sec][BLOCK_SIZE - 1][1];
                    help2 = C_cross[sec][BLOCK_SIZE - 1][2];
                    help3 = C_cross[sec][BLOCK_SIZE - 1][3];

                    for (int i = 0; i < num_partcarries; i++) {
                        const T p0 = in[i * order][0];
                        const T p1 = in[i * order + 1][0];
                        const T h0 = fmaf(help1, X1, fmaf(help0, X0, p0));
                        const T h1 = fmaf(help3, X1, fmaf(help2, X0, p1));
                        X0 = h0;
                        X1 = h1;
                    }
                }
            }

            if (lane == 0) {
                sfullc[0] = X0;
                sfullc[1] = X1;
            }
        }

        __syncthreads();
        _xi2 = sfullc[0];
        _xi1 = sfullc[1];

        in_reg[N_BLOCKS - 2] = fmaf(C_inner[sec][tid + BLOCK_SIZE], _xi1,
                                     fmaf(C_inner[sec][tid], _xi2, _yi2));
        in_reg[N_BLOCKS - 1] = fmaf(C_inner[sec][tid + 3*BLOCK_SIZE], _xi1,
                                     fmaf(C_inner[sec][tid + 2*BLOCK_SIZE], _xi2, _yi1));

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

        if (tid > 0) {
            _xi2 = in[tid - 1][N_BLOCKS - 2];
            _xi1 = in[tid - 1][N_BLOCKS - 1];
        }

        #pragma unroll
        for (int n = 0; n < N_BLOCKS - 2; n++) {
            in_reg[n] = fmaf(h1[sec][n], _xi1, fmaf(h2[sec][n], _xi2, in_reg[n]));

            if (sec == N_SECTIONS - 1)
                in[tid][n] = in_reg[n];
        }

    }

    __syncthreads();

    for (int i = tid; i < CHUNK_SIZE; i += BLOCK_SIZE) {
        int global_idx = chunk_start + i;
        int tx = i / N_BLOCKS;
        int bx = i % N_BLOCKS;
        output[global_idx] = in[tx][bx];
    }
}


// ==========================================================================
// Kernel selection macro
// ==========================================================================
#if BLOCK_SIZE == 32
    #define KERNEL_FUNC PH_32
#elif BLOCK_SIZE == 64
    #define KERNEL_FUNC PH_64
#else
    #error "PH kernels only support BLOCK_SIZE 32 or 64"
#endif

#endif // PH_KERNELS_CUH
