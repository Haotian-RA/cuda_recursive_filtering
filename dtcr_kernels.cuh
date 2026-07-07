#ifndef DTCR_KERNELS_CUH
#define DTCR_KERNELS_CUH

// DTCR (decoupled two-sided cyclic reduction) kernels for cascaded biquad
// IIR filtering. Each chunk is processed by 2*BLOCK_SIZE threads arranged as
// dim3(BLOCK_SIZE, 2): threadIdx.y selects the parity side (0 = even block
// positions, 1 = odd). After the first CR elimination (fused into the FIR
// pass via the decoupled coefficients cb/db/eb/fb/hb/gb) the two sides are
// independent and run concurrently.
//
// REQUIRES the following to be defined before inclusion:
//   N_SECTIONS   — number of cascaded biquads
//   BLOCK_SIZE   — lanes per parity side (32 or 64)
//   N_BLOCKS     — register-resident samples per lane (per thread: N_BLOCKS/2)
//   N_TB_PER_SM  — occupancy target computed by the driver (thread-capped
//                  here into DTCR_N_TB_PER_SM because TBs are 2*BLOCK_SIZE)
//   gpu_specs.hpp must already be included.
//
// Exposes:
//   KERNEL_FUNC     — one of six kernels, selected by BLOCK_SIZE, N_BLOCKS
//                     and the DTCR_HANDUNROLLED define:
//                       default:            DTCR_32_LOOP / DTCR_64_LOOP
//                       DTCR_HANDUNROLLED:  DTCR_32_UNROLL_{32,64,128},
//                                           DTCR_64_UNROLL_64
//   KERNEL_TB_DIM   — thread-block dimensions for the <<<>>> launch
//   setup_kernel_coefficients(sos) — copies SOS-derived factors to the device
//
// Launch-versioned status flags (same protocol as ph/stcr_kernels.cuh):
//   Each launch passes a monotonically increasing `launch` index. Status
//   values are launch-relative: part_flag = 2*launch+1, full_flag = 2*launch+2.
//   Any value < part_flag reads as "not ready", so the status array is zeroed
//   ONCE at setup and never reset. Assumes serialized launches on one stream
//   with a constant grid, `launch` incrementing by exactly 1 per launch.
//
// Cross-warp note: DTCR_64 exchanges lane-neighbor values through shared
// memory with __syncthreads between rounds; one barrier missing in the
// original (companion write -> first round read) has been made explicit.

#include <vector>
#include <array>
#include <cassert>

#ifndef N_BLOCKS_LOG2
#define N_BLOCKS_LOG2   __builtin_ctz(N_BLOCKS)
#endif
#ifndef HALF_N_BLOCKS
#define HALF_N_BLOCKS   (N_BLOCKS >> 1)
#endif
#ifndef QUARTER_N_BLOCKS
#define QUARTER_N_BLOCKS (HALF_N_BLOCKS >> 1)
#endif
#ifndef CHUNK_SIZE
#define CHUNK_SIZE      (BLOCK_SIZE * N_BLOCKS)
#endif

#define KERNEL_TB_DIM dim3(BLOCK_SIZE, 2)

// DTCR thread blocks hold 2*BLOCK_SIZE threads: cap the driver-computed
// occupancy target by the SM thread limit so __launch_bounds__ is feasible.
constexpr int DTCR_N_TB_PER_SM =
    (N_TB_PER_SM < gpu_specs::MAX_THREADS_PER_SM / (2 * BLOCK_SIZE))
        ? N_TB_PER_SM
        : gpu_specs::MAX_THREADS_PER_SM / (2 * BLOCK_SIZE);

static const int order = 2;
static const int warp_size = 32;

static __device__ unsigned int counter = 0;

// Per-section coefficients
static __constant__ T b1[N_SECTIONS], b2[N_SECTIONS], a1[N_SECTIONS], a2[N_SECTIONS];
static __constant__ T xi1[N_SECTIONS], xi2[N_SECTIONS], yi1[N_SECTIONS], yi2[N_SECTIONS];

// Decoupled first-round coefficients (even/odd fused FIR+CR pass)
static __constant__ T cb[N_SECTIONS], db[N_SECTIONS], fb[N_SECTIONS], eb[N_SECTIONS];
static __constant__ T hb[N_SECTIONS], gb[N_SECTIONS];

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
        T* h_h, T* h_g, T* h_p, T* h_q, T* h_d, T* h_c, T h_b1, T h_b2, T* h_dc){

    h_f[0] = - h_a2;
    h_e[0] = - h_a1;
    h_fde[0] = h_f[0]/h_e[0];
    h_h[0] = h_f[0];
    h_g[0] = h_e[0];
    h_p[1] = h_f[0];
    h_q[1] = h_e[0];
    
    for (int n = 1; n < N_BLOCKS_LOG2 + 1; n++){
        h_f[n] = h_f[n-1]*h_f[n-1];
        h_e[n] = 2*h_f[n-1] - h_e[n-1]*h_e[n-1];
        h_fde[n] = h_f[n]/h_e[n];
        h_h[n] = -h_e[n-1]*h_h[n-1];
        h_g[n] = h_f[n-1] - h_e[n-1]*h_g[n-1];
        h_d[n] = -h_f[n-1]*h_f[n-1]/h_e[n-1];
        h_c[n] = h_e[n-1] - h_f[n-1]/h_e[n-1];

        if (n > 1){
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

inline void setup_kernel_coefficients(const std::vector<std::array<T, 6>>& sos) {
    assert((int)sos.size() == N_SECTIONS);

    T h_b1[N_SECTIONS], h_b2[N_SECTIONS], h_a1[N_SECTIONS], h_a2[N_SECTIONS];
    T h_xi1[N_SECTIONS], h_xi2[N_SECTIONS], h_yi1[N_SECTIONS], h_yi2[N_SECTIONS];

    for (int sec = 0; sec < N_SECTIONS; sec++) {
        h_b1[sec] = sos[sec][1];
        h_b2[sec] = sos[sec][2];
        h_a1[sec] = -sos[sec][4];
        h_a2[sec] = -sos[sec][5];
        h_xi1[sec] = 0.0f;  h_xi2[sec] = 0.0f;
        h_yi1[sec] = 0.0f;  h_yi2[sec] = 0.0f;
    }

    static T h_e[N_SECTIONS][N_BLOCKS_LOG2 + 1], h_f[N_SECTIONS][N_BLOCKS_LOG2 + 1];
    static T h_fde[N_SECTIONS][N_BLOCKS_LOG2 + 1];
    static T h_h[N_SECTIONS][N_BLOCKS_LOG2 + 1], h_g[N_SECTIONS][N_BLOCKS_LOG2 + 1];
    static T h_p[N_SECTIONS][N_BLOCKS_LOG2 + 1], h_q[N_SECTIONS][N_BLOCKS_LOG2 + 1];
    static T h_d[N_SECTIONS][N_BLOCKS_LOG2 + 1], h_c[N_SECTIONS][N_BLOCKS_LOG2 + 1];
    static T h_dc[N_SECTIONS][6];
    static T h_h0[N_SECTIONS][BLOCK_SIZE];
    static T h_hb2[N_SECTIONS][BLOCK_SIZE], h_hb1[N_SECTIONS][BLOCK_SIZE];
    static T h_he2[N_SECTIONS][BLOCK_SIZE], h_he1[N_SECTIONS][BLOCK_SIZE];
    static T h_C_cross[N_SECTIONS][BLOCK_SIZE][4];
    static T h_h1[N_SECTIONS][N_BLOCKS], h_h2[N_SECTIONS][N_BLOCKS];

    T h_cb[N_SECTIONS], h_db[N_SECTIONS], h_fb[N_SECTIONS], h_eb[N_SECTIONS];
    T h_hb_coef[N_SECTIONS], h_gb[N_SECTIONS];

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

        // Decoupled coefficients from h_dc
        h_cb[sec] = h_dc[sec][0];
        h_db[sec] = h_dc[sec][1];
        h_fb[sec] = h_dc[sec][2];
        h_eb[sec] = h_dc[sec][3];
        h_hb_coef[sec] = h_dc[sec][4];
        h_gb[sec] = h_dc[sec][5];
    }

    assert(cudaSuccess == cudaMemcpyToSymbol(b1, h_b1, N_SECTIONS * sizeof(T)));
    assert(cudaSuccess == cudaMemcpyToSymbol(b2, h_b2, N_SECTIONS * sizeof(T)));
    assert(cudaSuccess == cudaMemcpyToSymbol(a1, h_a1, N_SECTIONS * sizeof(T)));
    assert(cudaSuccess == cudaMemcpyToSymbol(a2, h_a2, N_SECTIONS * sizeof(T)));
    assert(cudaSuccess == cudaMemcpyToSymbol(xi1, h_xi1, N_SECTIONS * sizeof(T)));
    assert(cudaSuccess == cudaMemcpyToSymbol(xi2, h_xi2, N_SECTIONS * sizeof(T)));
    assert(cudaSuccess == cudaMemcpyToSymbol(yi1, h_yi1, N_SECTIONS * sizeof(T)));
    assert(cudaSuccess == cudaMemcpyToSymbol(yi2, h_yi2, N_SECTIONS * sizeof(T)));
    assert(cudaSuccess == cudaMemcpyToSymbol(cb, h_cb, N_SECTIONS * sizeof(T)));
    assert(cudaSuccess == cudaMemcpyToSymbol(db, h_db, N_SECTIONS * sizeof(T)));
    assert(cudaSuccess == cudaMemcpyToSymbol(fb, h_fb, N_SECTIONS * sizeof(T)));
    assert(cudaSuccess == cudaMemcpyToSymbol(eb, h_eb, N_SECTIONS * sizeof(T)));
    assert(cudaSuccess == cudaMemcpyToSymbol(hb, h_hb_coef, N_SECTIONS * sizeof(T)));
    assert(cudaSuccess == cudaMemcpyToSymbol(gb, h_gb, N_SECTIONS * sizeof(T)));
    assert(cudaSuccess == cudaMemcpyToSymbol(e, h_e, N_SECTIONS * (N_BLOCKS_LOG2 + 1) * sizeof(T)));
    assert(cudaSuccess == cudaMemcpyToSymbol(f, h_f, N_SECTIONS * (N_BLOCKS_LOG2 + 1) * sizeof(T)));
    assert(cudaSuccess == cudaMemcpyToSymbol(fde, h_fde, N_SECTIONS * (N_BLOCKS_LOG2 + 1) * sizeof(T)));
    assert(cudaSuccess == cudaMemcpyToSymbol(cr_h, h_h, N_SECTIONS * (N_BLOCKS_LOG2 + 1) * sizeof(T)));
    assert(cudaSuccess == cudaMemcpyToSymbol(cr_g, h_g, N_SECTIONS * (N_BLOCKS_LOG2 + 1) * sizeof(T)));
    assert(cudaSuccess == cudaMemcpyToSymbol(cr_p, h_p, N_SECTIONS * (N_BLOCKS_LOG2 + 1) * sizeof(T)));
    assert(cudaSuccess == cudaMemcpyToSymbol(cr_q, h_q, N_SECTIONS * (N_BLOCKS_LOG2 + 1) * sizeof(T)));
    assert(cudaSuccess == cudaMemcpyToSymbol(cr_d, h_d, N_SECTIONS * (N_BLOCKS_LOG2 + 1) * sizeof(T)));
    assert(cudaSuccess == cudaMemcpyToSymbol(cr_c, h_c, N_SECTIONS * (N_BLOCKS_LOG2 + 1) * sizeof(T)));
    assert(cudaSuccess == cudaMemcpyToSymbol(h0, h_h0, N_SECTIONS * BLOCK_SIZE * sizeof(T)));
    assert(cudaSuccess == cudaMemcpyToSymbol(hb2, h_hb2, N_SECTIONS * BLOCK_SIZE * sizeof(T)));
    assert(cudaSuccess == cudaMemcpyToSymbol(hb1, h_hb1, N_SECTIONS * BLOCK_SIZE * sizeof(T)));
    assert(cudaSuccess == cudaMemcpyToSymbol(he2, h_he2, N_SECTIONS * BLOCK_SIZE * sizeof(T)));
    assert(cudaSuccess == cudaMemcpyToSymbol(he1, h_he1, N_SECTIONS * BLOCK_SIZE * sizeof(T)));
    assert(cudaSuccess == cudaMemcpyToSymbol(C_cross, h_C_cross, N_SECTIONS * BLOCK_SIZE * 4 * sizeof(T)));
}

static __global__ __launch_bounds__(2 * BLOCK_SIZE, DTCR_N_TB_PER_SM)
void DTCR_32_LOOP(const T* const __restrict__ input, 
        T* const __restrict__ output,
        volatile int* const __restrict__ status, 
        volatile T* const __restrict__ partcarry, 
        volatile T* const __restrict__ fullcarry,
        const int launch)
{
    __shared__ T in[BLOCK_SIZE][N_BLOCKS + 1];
    __shared__ int cid;
    __shared__ T spartc[CHUNK_SIZE / warp_size * order];
    __shared__ T sfullc[order];

    const int tx = threadIdx.x;  
    const int ty = threadIdx.y;  
    const int tid = ty * BLOCK_SIZE + tx;  // 0-63
    const int lane = tid % warp_size;
    const int warp = tid / warp_size;

    // Launch-relative status values (see header comment).
    const int part_flag = 2 * launch + 1;
    const int full_flag = part_flag + 1;

    if (tid == 0) {
        cid = (int)(atomicAdd(&counter, 1u) - (unsigned int)launch * gridDim.x);
    }

    __syncthreads();

    const int chunk_id = cid;
    const int chunk_start = chunk_id * CHUNK_SIZE;

    for (int i = tid; i < CHUNK_SIZE; i += 2*BLOCK_SIZE) {
        int global_idx = chunk_start + i;
        int ti = i / N_BLOCKS;
        int bi = i % N_BLOCKS;
        in[ti][bi] = input[global_idx];
    }

    T in_reg[HALF_N_BLOCKS];
    T _xc, _xi1, _xi2, _xi3, _xi4;

    for (int sec = 0; sec < N_SECTIONS; sec++) {

        __syncthreads();

        // Phase 1: FIR + CR
        if (ty == 0) {
            if (tx == 0) {
                _xi3 = 0;
                _xi4 = 0;
                if (chunk_id == 0) {
                    _xi1 = xi1[sec];
                    _xi2 = xi2[sec];
                } else if (sec == 0) {
                    _xi1 = input[chunk_start - 1];
                    _xi2 = input[chunk_start - 2];
                }
            } else {
                _xi1 = in[tx - 1][N_BLOCKS - 1];
                _xi2 = in[tx - 1][N_BLOCKS - 2];
                _xi3 = in[tx - 1][N_BLOCKS - 3];
                _xi4 = in[tx - 1][N_BLOCKS - 4];
            }
        } else {
            _xi1 = in[tx][0];
            if (tx == 0) {
                _xi4 = 0;
                if (chunk_id == 0) {
                    _xi2 = xi1[sec];
                    _xi3 = xi2[sec];
                } else if (sec == 0) {
                    _xi2 = input[chunk_start - 1];
                    _xi3 = input[chunk_start - 2];
                }
            } else {
                _xi2 = in[tx - 1][N_BLOCKS - 1];
                _xi3 = in[tx - 1][N_BLOCKS - 2];
                _xi4 = in[tx - 1][N_BLOCKS - 3];
            }
        }

        // the first block
        T _h1, _h2, _h3;
        
        if (ty == 0) {
            if (tx == 0) {
                _h1 = b1[sec];
                _h2 = b2[sec];
                _h3 = 0;
            } else {
                _h1 = db[sec];
                _h2 = eb[sec];
                _h3 = cb[sec];
            }
        } else {
            _h1 = db[sec];  
            if (tx == 0) {
                _h2 = hb[sec];
                _h3 = gb[sec];
            } else {
                _h2 = eb[sec];
                _h3 = cb[sec];
            }
        }

        _xc = in[tx][ty];

        in_reg[0] = fmaf(_h1, _xi1, fmaf(_h2, _xi2, fmaf(_h3, _xi3, fmaf(fb[sec], _xi4, _xc))));

        _xi4 = _xi2;
        _xi3 = _xi1;
        _xi2 = _xc;

        #pragma unroll
        for (int n = 1; n < HALF_N_BLOCKS; n++) {
            _xc = in[tx][2 * n + ty];
            _xi1 = in[tx][2 * n + ty - 1];

            in_reg[n] = fmaf(db[sec], _xi1, fmaf(eb[sec], _xi2, fmaf(cb[sec], _xi3, fmaf(fb[sec], _xi4, _xc))));

            _xi4 = _xi2;
            _xi3 = _xi1;
            _xi2 = _xc;
        }

        // Remaining CR rounds - using warp shuffle (each warp independent)
        #pragma unroll
        for (int ro = 1; ro < N_BLOCKS_LOG2; ro++) {
       
            const int step = 1 << ro;
            const int sub = step >> 1; 
            const int off = sub - 1;

            _xi2 = __shfl_up_sync(0xffffffff, in_reg[HALF_N_BLOCKS - 1], 1);
            if (tx == 0) _xi2 = 0;

            #pragma unroll
            for (int n = 0; n < HALF_N_BLOCKS; n += step) {
                in_reg[n + off] = fmaf(-fde[sec][ro], _xi2, in_reg[n + off]);

                _xi2 = in_reg[n + off + sub];
                in_reg[n + off + sub] = fmaf(-e[sec][ro], in_reg[n + off], _xi2);
            }
        }

        // block filtering: each warp independent
        T _yi = h0[sec][0] * in_reg[HALF_N_BLOCKS - 1];

        #pragma unroll
        for (int n = 1; n < BLOCK_SIZE; n++) {

            _xc = __shfl_up_sync(0xffffffff, in_reg[HALF_N_BLOCKS - 1], n);

            if (tx < n) 
                _xc = 0;
            
            _yi = fmaf(h0[sec][n], _xc, _yi);
        }

        // Phase 2: Lookback
        if (tx == BLOCK_SIZE - 1) {
            partcarry[chunk_id * order * N_SECTIONS + sec * order + ty] = _yi;
        }

        __syncthreads();
        __threadfence();
        if (tid == BLOCK_SIZE - 1) 
            status[chunk_id * N_SECTIONS + sec] = part_flag;

        T X0, X1;

        if (warp == 0) {
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
                    spartc[i - start_chunk * order] = partcarry[(i / order) * order * N_SECTIONS + sec * order + (i % order)];
                }
                __syncwarp();
                
                if (lane == 0) {
                    _xi1 = C_cross[sec][BLOCK_SIZE - 1][0];
                    _xi2 = C_cross[sec][BLOCK_SIZE - 1][1];
                    _xi3 = C_cross[sec][BLOCK_SIZE - 1][2];
                    _xi4 = C_cross[sec][BLOCK_SIZE - 1][3];

                    for (int i = 0; i < num_partcarries; i++) {
                        const T p0 = spartc[i * order];
                        const T p1 = spartc[i * order + 1];
                        const T h0_val = fmaf(_xi2, X1, fmaf(_xi1, X0, p0));
                        const T h1_val = fmaf(_xi4, X1, fmaf(_xi3, X0, p1));
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
        T _h2_bs, _h1_bs;
        if (ty == 0) {
            _h2_bs = he2[sec][tx];
            _h1_bs = he1[sec][tx];
        } else {
            _h2_bs = hb2[sec][tx];
            _h1_bs = hb1[sec][tx];
        }

        in_reg[HALF_N_BLOCKS - 1] = fmaf(-_h1_bs, X1, fmaf(-_h2_bs, X0, _yi));

        if (tx == BLOCK_SIZE - 1) 
            fullcarry[chunk_id * order * N_SECTIONS + sec * order + ty] = in_reg[HALF_N_BLOCKS - 1];
        
        __syncthreads();
        __threadfence();
        if (tid == BLOCK_SIZE - 1) {
            status[chunk_id * N_SECTIONS + sec] = full_flag;
        }

        in[tx][N_BLOCKS - 2 + ty] = in_reg[HALF_N_BLOCKS - 1];

        T _yi2 = __shfl_up_sync(0xffffffff, in_reg[HALF_N_BLOCKS - 1], 2 - ty);
        T _yi1 = __shfl_up_sync(0xffffffff, in_reg[HALF_N_BLOCKS - 1], 1 + ty);
        T tmp;

        if (tx == 0) {
            tmp = X1;
            if (ty == 0) {
                _h2_bs = cr_p[sec][N_BLOCKS_LOG2 - 1];
                _h1_bs = cr_q[sec][N_BLOCKS_LOG2 - 1];
                _xc = X0;
                _xi1 = X1;
                _xi2 = X0;
            } else {
                _h2_bs = cr_h[sec][N_BLOCKS_LOG2 - 1]; 
                _h1_bs = cr_g[sec][N_BLOCKS_LOG2 - 1];  
                _xc = X1;
                _xi2 = X1;
                _xi3 = X0;
            }
            _yi2 = X0;  
            _yi1 = X1;
        } else {
            if (ty == 0) { 
                _h2_bs = cr_d[sec][N_BLOCKS_LOG2];
                _h1_bs = cr_c[sec][N_BLOCKS_LOG2];
                tmp = _yi1;
            } else {
                _h2_bs = cr_c[sec][N_BLOCKS_LOG2];
                _h1_bs = cr_d[sec][N_BLOCKS_LOG2];
                tmp = _yi2;
            }
            if (tx == 1) {
                if (ty == 0) 
                    _yi2 = X0;
                else 
                    _yi1 = X1;
            }
            _xc = tmp;
        }

        in_reg[QUARTER_N_BLOCKS - 1] = fmaf(-_h1_bs, _yi1, fmaf(-_h2_bs, _yi2, in_reg[QUARTER_N_BLOCKS - 1]));
        in[tx][HALF_N_BLOCKS - 2 + ty] = in_reg[QUARTER_N_BLOCKS - 1];

        T _zi2, _zi1;

        #pragma unroll 
        for (int ro = N_BLOCKS_LOG2 - 2; ro > 0; ro--) {

            const int P = HALF_N_BLOCKS >> ro; 
            const int step = 2 << ro; 
            const int sub = 1 << ro; 
            const int sub2 = sub >> 1; 

            #pragma unroll
            for (int n = 0; n < P; n++) {

                if (n == 0) {
                    _yi2 = __shfl_up_sync(0xffffffff, in_reg[HALF_N_BLOCKS - 1 - sub], 1);
                    if (tx == 0) {
                        if (ty == 0){
                            _h2_bs = cr_p[sec][ro];
                            _h1_bs = cr_q[sec][ro];
                        } else {
                            _h2_bs = cr_h[sec][ro];
                            _h1_bs = cr_g[sec][ro];
                        }
                        _yi2 = X0;
                    } else {
                        _h2_bs = cr_d[sec][ro + 1];
                        _h1_bs = cr_c[sec][ro + 1];
                    }
                    _zi2 = _yi2;
                    _zi1 = tmp;
                } else if (n == 1) {
                    _h2_bs = cr_d[sec][ro + 1];
                    _h1_bs = cr_c[sec][ro + 1];
                    _zi2 = _xc;
                    _zi1 = in_reg[sub - 1];
                } else {
                    _zi2 = _zi1;
                    _zi1 = in_reg[sub * n - 1];
                }

                in_reg[sub*n + sub2 - 1] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[sub*n + sub2 - 1]));
                in[tx][step*n + sub - 2 + ty] = in_reg[sub*n + sub2 - 1];
            
            }
        }


    } // end section loop

    __syncthreads();

    for (int i = tid; i < CHUNK_SIZE; i += 2 * BLOCK_SIZE) {
        int global_idx = chunk_start + i;
        int ti = i / N_BLOCKS;
        int bi = i % N_BLOCKS;
        output[global_idx] = in[ti][bi];  
    }
}


#if N_BLOCKS == 32
static __global__ __launch_bounds__(2 * BLOCK_SIZE, DTCR_N_TB_PER_SM)
void DTCR_32_UNROLL_32(const T* const __restrict__ input, 
        T* const __restrict__ output,
        volatile int* const __restrict__ status, 
        volatile T* const __restrict__ partcarry, 
        volatile T* const __restrict__ fullcarry,
        const int launch)
{
    __shared__ T in[BLOCK_SIZE][N_BLOCKS + 1];
    __shared__ int cid;
    __shared__ T spartc[CHUNK_SIZE / warp_size * order];
    __shared__ T sfullc[order];

    const int tx = threadIdx.x;  
    const int ty = threadIdx.y;  
    const int tid = ty * BLOCK_SIZE + tx;  // 0-63
    const int lane = tid % warp_size;
    const int warp = tid / warp_size;

    // Launch-relative status values (see header comment).
    const int part_flag = 2 * launch + 1;
    const int full_flag = part_flag + 1;

    if (tid == 0) {
        cid = (int)(atomicAdd(&counter, 1u) - (unsigned int)launch * gridDim.x);
    }

    __syncthreads();

    const int chunk_id = cid;
    const int chunk_start = chunk_id * CHUNK_SIZE;

    for (int i = tid; i < CHUNK_SIZE; i += 2*BLOCK_SIZE) {
        int global_idx = chunk_start + i;
        int ti = i / N_BLOCKS;
        int bi = i % N_BLOCKS;
        in[ti][bi] = input[global_idx];
    }

    T in_reg[HALF_N_BLOCKS];
    T _xc, _xi1, _xi2, _xi3, _xi4;

    for (int sec = 0; sec < N_SECTIONS; sec++) {

        __syncthreads();

        // Phase 1: FIR + CR
        if (ty == 0) {
            if (tx == 0) {
                _xi3 = 0;
                _xi4 = 0;
                if (chunk_id == 0) {
                    _xi1 = xi1[sec];
                    _xi2 = xi2[sec];
                } else if (sec == 0) {
                    _xi1 = input[chunk_start - 1];
                    _xi2 = input[chunk_start - 2];
                }
            } else {
                _xi1 = in[tx - 1][N_BLOCKS - 1];
                _xi2 = in[tx - 1][N_BLOCKS - 2];
                _xi3 = in[tx - 1][N_BLOCKS - 3];
                _xi4 = in[tx - 1][N_BLOCKS - 4];
            }
        } else {
            _xi1 = in[tx][0];
            if (tx == 0) {
                _xi4 = 0;
                if (chunk_id == 0) {
                    _xi2 = xi1[sec];
                    _xi3 = xi2[sec];
                } else if (sec == 0) {
                    _xi2 = input[chunk_start - 1];
                    _xi3 = input[chunk_start - 2];
                }
            } else {
                _xi2 = in[tx - 1][N_BLOCKS - 1];
                _xi3 = in[tx - 1][N_BLOCKS - 2];
                _xi4 = in[tx - 1][N_BLOCKS - 3];
            }
        }

        // the first block
        T _h1, _h2, _h3;
        
        if (ty == 0) {
            if (tx == 0) {
                _h1 = b1[sec];
                _h2 = b2[sec];
                _h3 = 0;
            } else {
                _h1 = db[sec];
                _h2 = eb[sec];
                _h3 = cb[sec];
            }
        } else {
            _h1 = db[sec];  
            if (tx == 0) {
                _h2 = hb[sec];
                _h3 = gb[sec];
            } else {
                _h2 = eb[sec];
                _h3 = cb[sec];
            }
        }

        _xc = in[tx][ty];

        in_reg[0] = fmaf(_h1, _xi1, fmaf(_h2, _xi2, fmaf(_h3, _xi3, fmaf(fb[sec], _xi4, _xc))));

        _xi4 = _xi2;
        _xi3 = _xi1;
        _xi2 = _xc;

        #pragma unroll
        for (int n = 1; n < HALF_N_BLOCKS; n++) {
            _xc = in[tx][2 * n + ty];
            _xi1 = in[tx][2 * n + ty - 1];

            in_reg[n] = fmaf(db[sec], _xi1, fmaf(eb[sec], _xi2, fmaf(cb[sec], _xi3, fmaf(fb[sec], _xi4, _xc))));

            _xi4 = _xi2;
            _xi3 = _xi1;
            _xi2 = _xc;
        }

        // Remaining CR rounds - using warp shuffle (each warp independent)
        #pragma unroll
        for (int ro = 1; ro < N_BLOCKS_LOG2; ro++) {
       
            const int step = 1 << ro;
            const int sub = step >> 1; 
            const int off = sub - 1;

            _xi2 = __shfl_up_sync(0xffffffff, in_reg[HALF_N_BLOCKS - 1], 1);
            if (tx == 0) _xi2 = 0;

            #pragma unroll
            for (int n = 0; n < HALF_N_BLOCKS; n += step) {
                in_reg[n + off] = fmaf(-fde[sec][ro], _xi2, in_reg[n + off]);

                _xi2 = in_reg[n + off + sub];
                in_reg[n + off + sub] = fmaf(-e[sec][ro], in_reg[n + off], _xi2);
            }
        }

        // block filtering: each warp independent
        T _yi = h0[sec][0] * in_reg[HALF_N_BLOCKS - 1];

        #pragma unroll
        for (int n = 1; n < BLOCK_SIZE; n++) {

            _xc = __shfl_up_sync(0xffffffff, in_reg[HALF_N_BLOCKS - 1], n);

            if (tx < n) 
                _xc = 0;
            
            _yi = fmaf(h0[sec][n], _xc, _yi);
        }

        // Phase 2: Lookback
        if (tx == BLOCK_SIZE - 1) {
            partcarry[chunk_id * order * N_SECTIONS + sec * order + ty] = _yi;
        }

        __syncthreads();
        __threadfence();
        if (tid == BLOCK_SIZE - 1) 
            status[chunk_id * N_SECTIONS + sec] = part_flag;

        T X0, X1;

        if (warp == 0) {
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
                    spartc[i - start_chunk * order] = partcarry[(i / order) * order * N_SECTIONS + sec * order + (i % order)];
                }
                __syncwarp();
                
                if (lane == 0) {
                    _xi1 = C_cross[sec][BLOCK_SIZE - 1][0];
                    _xi2 = C_cross[sec][BLOCK_SIZE - 1][1];
                    _xi3 = C_cross[sec][BLOCK_SIZE - 1][2];
                    _xi4 = C_cross[sec][BLOCK_SIZE - 1][3];

                    for (int i = 0; i < num_partcarries; i++) {
                        const T p0 = spartc[i * order];
                        const T p1 = spartc[i * order + 1];
                        const T h0_val = fmaf(_xi2, X1, fmaf(_xi1, X0, p0));
                        const T h1_val = fmaf(_xi4, X1, fmaf(_xi3, X0, p1));
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
        T _h2_bs, _h1_bs;
        if (ty == 0) {
            _h2_bs = he2[sec][tx];
            _h1_bs = he1[sec][tx];
        } else {
            _h2_bs = hb2[sec][tx];
            _h1_bs = hb1[sec][tx];
        }

        in_reg[HALF_N_BLOCKS - 1] = fmaf(-_h1_bs, X1, fmaf(-_h2_bs, X0, _yi));

        if (tx == BLOCK_SIZE - 1) 
            fullcarry[chunk_id * order * N_SECTIONS + sec * order + ty] = in_reg[HALF_N_BLOCKS - 1];
        
        __syncthreads();
        __threadfence();
        if (tid == BLOCK_SIZE - 1) {
            status[chunk_id * N_SECTIONS + sec] = full_flag;
        }

        in[tx][N_BLOCKS - 2 + ty] = in_reg[HALF_N_BLOCKS - 1];

        T _yi2 = __shfl_up_sync(0xffffffff, in_reg[HALF_N_BLOCKS - 1], 2 - ty);
        T _yi1 = __shfl_up_sync(0xffffffff, in_reg[HALF_N_BLOCKS - 1], 1 + ty);
        T tmp;

        if (tx == 0) {
            tmp = X1;
            if (ty == 0) {
                _h2_bs = cr_p[sec][N_BLOCKS_LOG2 - 1];
                _h1_bs = cr_q[sec][N_BLOCKS_LOG2 - 1];
                _xc = X0;
                _xi1 = X1;
                _xi2 = X0;
            } else {
                _h2_bs = cr_h[sec][N_BLOCKS_LOG2 - 1]; 
                _h1_bs = cr_g[sec][N_BLOCKS_LOG2 - 1];  
                _xc = X1;
                _xi2 = X1;
                _xi3 = X0;
            }
            _yi2 = X0;  
            _yi1 = X1;
        } else {
            if (ty == 0) { 
                _h2_bs = cr_d[sec][N_BLOCKS_LOG2];
                _h1_bs = cr_c[sec][N_BLOCKS_LOG2];
                tmp = _yi1;
            } else {
                _h2_bs = cr_c[sec][N_BLOCKS_LOG2];
                _h1_bs = cr_d[sec][N_BLOCKS_LOG2];
                tmp = _yi2;
            }
            if (tx == 1) {
                if (ty == 0) 
                    _yi2 = X0;
                else 
                    _yi1 = X1;
            }
            _xc = tmp;
        }

        in_reg[QUARTER_N_BLOCKS - 1] = fmaf(-_h1_bs, _yi1, fmaf(-_h2_bs, _yi2, in_reg[QUARTER_N_BLOCKS - 1]));
        in[tx][HALF_N_BLOCKS - 2 + ty] = in_reg[QUARTER_N_BLOCKS - 1];

        T _zi2, _zi1;

        // ============ ro = 3 ============
        // P = 2, step = 16, sub = 8, sub2 = 4
        
        // n = 0
        _yi2 = __shfl_up_sync(0xffffffff, in_reg[7], 1);
        if (tx == 0) {
            if (ty == 0) {
                _h2_bs = cr_p[sec][3];
                _h1_bs = cr_q[sec][3];
            } else {
                _h2_bs = cr_h[sec][3];
                _h1_bs = cr_g[sec][3];
            }
            _yi2 = X0;
        } else {
            _h2_bs = cr_d[sec][4];
            _h1_bs = cr_c[sec][4];
        }
        _zi2 = _yi2;
        _zi1 = tmp;
        in_reg[3] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[3]));
        in[tx][6 + ty] = in_reg[3];

        // n = 1
        _h2_bs = cr_d[sec][4];
        _h1_bs = cr_c[sec][4];
        _zi2 = _xc;
        _zi1 = in_reg[7];
        in_reg[11] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[11]));
        in[tx][22 + ty] = in_reg[11];

        // ============ ro = 2 ============
        // P = 4, step = 8, sub = 4, sub2 = 2

        // n = 0
        _yi2 = __shfl_up_sync(0xffffffff, in_reg[11], 1);
        if (tx == 0) {
            if (ty == 0) {
                _h2_bs = cr_p[sec][2];
                _h1_bs = cr_q[sec][2];
            } else {
                _h2_bs = cr_h[sec][2];
                _h1_bs = cr_g[sec][2];
            }
            _yi2 = X0;
        } else {
            _h2_bs = cr_d[sec][3];
            _h1_bs = cr_c[sec][3];
        }
        _zi2 = _yi2;
        _zi1 = tmp;
        in_reg[1] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[1]));
        in[tx][2 + ty] = in_reg[1];

        // n = 1
        _h2_bs = cr_d[sec][3];
        _h1_bs = cr_c[sec][3];
        _zi2 = _xc;
        _zi1 = in_reg[3];
        in_reg[5] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[5]));
        in[tx][10 + ty] = in_reg[5];

        // n = 2
        _zi2 = _zi1;
        _zi1 = in_reg[7];
        in_reg[9] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[9]));
        in[tx][18 + ty] = in_reg[9];

        // n = 3
        _zi2 = _zi1;
        _zi1 = in_reg[11];
        in_reg[13] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[13]));
        in[tx][26 + ty] = in_reg[13];

        // ============ ro = 1 ============
        // P = 8, step = 4, sub = 2, sub2 = 1

        // n = 0
        _yi2 = __shfl_up_sync(0xffffffff, in_reg[13], 1);
        if (tx == 0) {
            if (ty == 0) {
                _h2_bs = cr_p[sec][1];
                _h1_bs = cr_q[sec][1];
            } else {
                _h2_bs = cr_h[sec][1];
                _h1_bs = cr_g[sec][1];
            }
            _yi2 = X0;
        } else {
            _h2_bs = cr_d[sec][2];
            _h1_bs = cr_c[sec][2];
        }
        _zi2 = _yi2;
        _zi1 = tmp;
        in_reg[0] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[0]));
        in[tx][0 + ty] = in_reg[0];

        // n = 1
        _h2_bs = cr_d[sec][2];
        _h1_bs = cr_c[sec][2];
        _zi2 = _xc;
        _zi1 = in_reg[1];
        in_reg[2] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[2]));
        in[tx][4 + ty] = in_reg[2];

        // n = 2
        _zi2 = _zi1;
        _zi1 = in_reg[3];
        in_reg[4] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[4]));
        in[tx][8 + ty] = in_reg[4];

        // n = 3
        _zi2 = _zi1;
        _zi1 = in_reg[5];
        in_reg[6] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[6]));
        in[tx][12 + ty] = in_reg[6];

        // n = 4
        _zi2 = _zi1;
        _zi1 = in_reg[7];
        in_reg[8] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[8]));
        in[tx][16 + ty] = in_reg[8];

        // n = 5
        _zi2 = _zi1;
        _zi1 = in_reg[9];
        in_reg[10] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[10]));
        in[tx][20 + ty] = in_reg[10];

        // n = 6
        _zi2 = _zi1;
        _zi1 = in_reg[11];
        in_reg[12] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[12]));
        in[tx][24 + ty] = in_reg[12];

        // n = 7
        _zi2 = _zi1;
        _zi1 = in_reg[13];
        in_reg[14] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[14]));
        in[tx][28 + ty] = in_reg[14];

    } // end section loop

    __syncthreads();

    for (int i = tid; i < CHUNK_SIZE; i += 2 * BLOCK_SIZE) {
        int global_idx = chunk_start + i;
        int ti = i / N_BLOCKS;
        int bi = i % N_BLOCKS;
        output[global_idx] = in[ti][bi];  
    }
}
#endif  // N_BLOCKS == 32


#if N_BLOCKS == 64
static __global__ __launch_bounds__(2 * BLOCK_SIZE, DTCR_N_TB_PER_SM)
void DTCR_32_UNROLL_64(const T* const __restrict__ input, 
        T* const __restrict__ output,
        volatile int* const __restrict__ status, 
        volatile T* const __restrict__ partcarry, 
        volatile T* const __restrict__ fullcarry,
        const int launch)
{
    __shared__ T in[BLOCK_SIZE][N_BLOCKS + 1];
    __shared__ int cid;
    __shared__ T spartc[CHUNK_SIZE / warp_size * order];
    __shared__ T sfullc[order];

    const int tx = threadIdx.x;  
    const int ty = threadIdx.y;  
    const int tid = ty * BLOCK_SIZE + tx;  // 0-63
    const int lane = tid % warp_size;
    const int warp = tid / warp_size;

    // Launch-relative status values (see header comment).
    const int part_flag = 2 * launch + 1;
    const int full_flag = part_flag + 1;

    if (tid == 0) {
        cid = (int)(atomicAdd(&counter, 1u) - (unsigned int)launch * gridDim.x);
    }

    __syncthreads();

    const int chunk_id = cid;
    const int chunk_start = chunk_id * CHUNK_SIZE;

    for (int i = tid; i < CHUNK_SIZE; i += 2*BLOCK_SIZE) {
        int global_idx = chunk_start + i;
        int ti = i / N_BLOCKS;
        int bi = i % N_BLOCKS;
        in[ti][bi] = input[global_idx];
    }

    T in_reg[HALF_N_BLOCKS];
    T _xc, _xi1, _xi2, _xi3, _xi4;

    for (int sec = 0; sec < N_SECTIONS; sec++) {

        __syncthreads();

        // Phase 1: FIR + CR
        if (ty == 0) {
            if (tx == 0) {
                _xi3 = 0;
                _xi4 = 0;
                if (chunk_id == 0) {
                    _xi1 = xi1[sec];
                    _xi2 = xi2[sec];
                } else if (sec == 0) {
                    _xi1 = input[chunk_start - 1];
                    _xi2 = input[chunk_start - 2];
                }
            } else {
                _xi1 = in[tx - 1][N_BLOCKS - 1];
                _xi2 = in[tx - 1][N_BLOCKS - 2];
                _xi3 = in[tx - 1][N_BLOCKS - 3];
                _xi4 = in[tx - 1][N_BLOCKS - 4];
            }
        } else {
            _xi1 = in[tx][0];
            if (tx == 0) {
                _xi4 = 0;
                if (chunk_id == 0) {
                    _xi2 = xi1[sec];
                    _xi3 = xi2[sec];
                } else if (sec == 0) {
                    _xi2 = input[chunk_start - 1];
                    _xi3 = input[chunk_start - 2];
                }
            } else {
                _xi2 = in[tx - 1][N_BLOCKS - 1];
                _xi3 = in[tx - 1][N_BLOCKS - 2];
                _xi4 = in[tx - 1][N_BLOCKS - 3];
            }
        }

        // the first block
        T _h1, _h2, _h3;
        
        if (ty == 0) {
            if (tx == 0) {
                _h1 = b1[sec];
                _h2 = b2[sec];
                _h3 = 0;
            } else {
                _h1 = db[sec];
                _h2 = eb[sec];
                _h3 = cb[sec];
            }
        } else {
            _h1 = db[sec];  
            if (tx == 0) {
                _h2 = hb[sec];
                _h3 = gb[sec];
            } else {
                _h2 = eb[sec];
                _h3 = cb[sec];
            }
        }

        _xc = in[tx][ty];

        in_reg[0] = fmaf(_h1, _xi1, fmaf(_h2, _xi2, fmaf(_h3, _xi3, fmaf(fb[sec], _xi4, _xc))));

        _xi4 = _xi2;
        _xi3 = _xi1;
        _xi2 = _xc;

        #pragma unroll
        for (int n = 1; n < HALF_N_BLOCKS; n++) {
            _xc = in[tx][2 * n + ty];
            _xi1 = in[tx][2 * n + ty - 1];

            in_reg[n] = fmaf(db[sec], _xi1, fmaf(eb[sec], _xi2, fmaf(cb[sec], _xi3, fmaf(fb[sec], _xi4, _xc))));

            _xi4 = _xi2;
            _xi3 = _xi1;
            _xi2 = _xc;
        }

        // Remaining CR rounds - using warp shuffle (each warp independent)
        #pragma unroll
        for (int ro = 1; ro < N_BLOCKS_LOG2; ro++) {
       
            const int step = 1 << ro;
            const int sub = step >> 1; 
            const int off = sub - 1;

            _xi2 = __shfl_up_sync(0xffffffff, in_reg[HALF_N_BLOCKS - 1], 1);
            if (tx == 0) _xi2 = 0;

            #pragma unroll
            for (int n = 0; n < HALF_N_BLOCKS; n += step) {
                in_reg[n + off] = fmaf(-fde[sec][ro], _xi2, in_reg[n + off]);

                _xi2 = in_reg[n + off + sub];
                in_reg[n + off + sub] = fmaf(-e[sec][ro], in_reg[n + off], _xi2);
            }
        }

        // block filtering: each warp independent
        T _yi = h0[sec][0] * in_reg[HALF_N_BLOCKS - 1];

        #pragma unroll
        for (int n = 1; n < BLOCK_SIZE; n++) {

            _xc = __shfl_up_sync(0xffffffff, in_reg[HALF_N_BLOCKS - 1], n);

            if (tx < n) 
                _xc = 0;
            
            _yi = fmaf(h0[sec][n], _xc, _yi);
        }

        // Phase 2: Lookback
        if (tx == BLOCK_SIZE - 1) {
            partcarry[chunk_id * order * N_SECTIONS + sec * order + ty] = _yi;
        }

        __syncthreads();
        __threadfence();
        if (tid == BLOCK_SIZE - 1) 
            status[chunk_id * N_SECTIONS + sec] = part_flag;

        T X0, X1;

        if (warp == 0) {
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
                    spartc[i - start_chunk * order] = partcarry[(i / order) * order * N_SECTIONS + sec * order + (i % order)];
                }
                __syncwarp();
                
                if (lane == 0) {
                    _xi1 = C_cross[sec][BLOCK_SIZE - 1][0];
                    _xi2 = C_cross[sec][BLOCK_SIZE - 1][1];
                    _xi3 = C_cross[sec][BLOCK_SIZE - 1][2];
                    _xi4 = C_cross[sec][BLOCK_SIZE - 1][3];

                    for (int i = 0; i < num_partcarries; i++) {
                        const T p0 = spartc[i * order];
                        const T p1 = spartc[i * order + 1];
                        const T h0_val = fmaf(_xi2, X1, fmaf(_xi1, X0, p0));
                        const T h1_val = fmaf(_xi4, X1, fmaf(_xi3, X0, p1));
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
        T _h2_bs, _h1_bs;
        if (ty == 0) {
            _h2_bs = he2[sec][tx];
            _h1_bs = he1[sec][tx];
        } else {
            _h2_bs = hb2[sec][tx];
            _h1_bs = hb1[sec][tx];
        }

        in_reg[HALF_N_BLOCKS - 1] = fmaf(-_h1_bs, X1, fmaf(-_h2_bs, X0, _yi));

        if (tx == BLOCK_SIZE - 1) 
            fullcarry[chunk_id * order * N_SECTIONS + sec * order + ty] = in_reg[HALF_N_BLOCKS - 1];
        
        __syncthreads();
        __threadfence();
        if (tid == BLOCK_SIZE - 1) {
            status[chunk_id * N_SECTIONS + sec] = full_flag;
        }

        in[tx][N_BLOCKS - 2 + ty] = in_reg[HALF_N_BLOCKS - 1];

        T _yi2 = __shfl_up_sync(0xffffffff, in_reg[HALF_N_BLOCKS - 1], 2 - ty);
        T _yi1 = __shfl_up_sync(0xffffffff, in_reg[HALF_N_BLOCKS - 1], 1 + ty);
        T tmp;

        if (tx == 0) {
            tmp = X1;
            if (ty == 0) {
                _h2_bs = cr_p[sec][N_BLOCKS_LOG2 - 1];
                _h1_bs = cr_q[sec][N_BLOCKS_LOG2 - 1];
                _xc = X0;
                _xi1 = X1;
                _xi2 = X0;
            } else {
                _h2_bs = cr_h[sec][N_BLOCKS_LOG2 - 1]; 
                _h1_bs = cr_g[sec][N_BLOCKS_LOG2 - 1];  
                _xc = X1;
                _xi2 = X1;
                _xi3 = X0;
            }
            _yi2 = X0;  
            _yi1 = X1;
        } else {
            if (ty == 0) { 
                _h2_bs = cr_d[sec][N_BLOCKS_LOG2];
                _h1_bs = cr_c[sec][N_BLOCKS_LOG2];
                tmp = _yi1;
            } else {
                _h2_bs = cr_c[sec][N_BLOCKS_LOG2];
                _h1_bs = cr_d[sec][N_BLOCKS_LOG2];
                tmp = _yi2;
            }
            if (tx == 1) {
                if (ty == 0) 
                    _yi2 = X0;
                else 
                    _yi1 = X1;
            }
            _xc = tmp;
        }

        in_reg[QUARTER_N_BLOCKS - 1] = fmaf(-_h1_bs, _yi1, fmaf(-_h2_bs, _yi2, in_reg[QUARTER_N_BLOCKS - 1]));
        in[tx][HALF_N_BLOCKS - 2 + ty] = in_reg[QUARTER_N_BLOCKS - 1];

        T _zi2, _zi1;

        // ============ ro = 4 ============
        // P = 2, step = 32, sub = 16, sub2 = 8

        // n = 0
        _yi2 = __shfl_up_sync(0xffffffff, in_reg[15], 1);
        if (tx == 0) {
            if (ty == 0) {
                _h2_bs = cr_p[sec][4];
                _h1_bs = cr_q[sec][4];
            } else {
                _h2_bs = cr_h[sec][4];
                _h1_bs = cr_g[sec][4];
            }
            _yi2 = X0;
        } else {
            _h2_bs = cr_d[sec][5];
            _h1_bs = cr_c[sec][5];
        }
        _zi2 = _yi2;
        _zi1 = tmp;
        in_reg[7] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[7]));
        in[tx][14 + ty] = in_reg[7];

        // n = 1
        _h2_bs = cr_d[sec][5];
        _h1_bs = cr_c[sec][5];
        _zi2 = _xc;
        _zi1 = in_reg[15];
        in_reg[23] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[23]));
        in[tx][46 + ty] = in_reg[23];

        // ============ ro = 3 ============
        // P = 4, step = 16, sub = 8, sub2 = 4

        // n = 0
        _yi2 = __shfl_up_sync(0xffffffff, in_reg[23], 1);
        if (tx == 0) {
            if (ty == 0) {
                _h2_bs = cr_p[sec][3];
                _h1_bs = cr_q[sec][3];
            } else {
                _h2_bs = cr_h[sec][3];
                _h1_bs = cr_g[sec][3];
            }
            _yi2 = X0;
        } else {
            _h2_bs = cr_d[sec][4];
            _h1_bs = cr_c[sec][4];
        }
        _zi2 = _yi2;
        _zi1 = tmp;
        in_reg[3] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[3]));
        in[tx][6 + ty] = in_reg[3];

        // n = 1
        _h2_bs = cr_d[sec][4];
        _h1_bs = cr_c[sec][4];
        _zi2 = _xc;
        _zi1 = in_reg[7];
        in_reg[11] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[11]));
        in[tx][22 + ty] = in_reg[11];

        // n = 2
        _zi2 = _zi1;
        _zi1 = in_reg[15];
        in_reg[19] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[19]));
        in[tx][38 + ty] = in_reg[19];

        // n = 3
        _zi2 = _zi1;
        _zi1 = in_reg[23];
        in_reg[27] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[27]));
        in[tx][54 + ty] = in_reg[27];

        // ============ ro = 2 ============
        // P = 8, step = 8, sub = 4, sub2 = 2

        // n = 0
        _yi2 = __shfl_up_sync(0xffffffff, in_reg[27], 1);
        if (tx == 0) {
            if (ty == 0) {
                _h2_bs = cr_p[sec][2];
                _h1_bs = cr_q[sec][2];
            } else {
                _h2_bs = cr_h[sec][2];
                _h1_bs = cr_g[sec][2];
            }
            _yi2 = X0;
        } else {
            _h2_bs = cr_d[sec][3];
            _h1_bs = cr_c[sec][3];
        }
        _zi2 = _yi2;
        _zi1 = tmp;
        in_reg[1] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[1]));
        in[tx][2 + ty] = in_reg[1];

        // n = 1
        _h2_bs = cr_d[sec][3];
        _h1_bs = cr_c[sec][3];
        _zi2 = _xc;
        _zi1 = in_reg[3];
        in_reg[5] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[5]));
        in[tx][10 + ty] = in_reg[5];

        // n = 2
        _zi2 = _zi1;
        _zi1 = in_reg[7];
        in_reg[9] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[9]));
        in[tx][18 + ty] = in_reg[9];

        // n = 3
        _zi2 = _zi1;
        _zi1 = in_reg[11];
        in_reg[13] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[13]));
        in[tx][26 + ty] = in_reg[13];

        // n = 4
        _zi2 = _zi1;
        _zi1 = in_reg[15];
        in_reg[17] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[17]));
        in[tx][34 + ty] = in_reg[17];

        // n = 5
        _zi2 = _zi1;
        _zi1 = in_reg[19];
        in_reg[21] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[21]));
        in[tx][42 + ty] = in_reg[21];

        // n = 6
        _zi2 = _zi1;
        _zi1 = in_reg[23];
        in_reg[25] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[25]));
        in[tx][50 + ty] = in_reg[25];

        // n = 7
        _zi2 = _zi1;
        _zi1 = in_reg[27];
        in_reg[29] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[29]));
        in[tx][58 + ty] = in_reg[29];

        // ============ ro = 1 ============
        // P = 16, step = 4, sub = 2, sub2 = 1

        // n = 0
        _yi2 = __shfl_up_sync(0xffffffff, in_reg[29], 1);
        if (tx == 0) {
            if (ty == 0) {
                _h2_bs = cr_p[sec][1];
                _h1_bs = cr_q[sec][1];
            } else {
                _h2_bs = cr_h[sec][1];
                _h1_bs = cr_g[sec][1];
            }
            _yi2 = X0;
        } else {
            _h2_bs = cr_d[sec][2];
            _h1_bs = cr_c[sec][2];
        }
        _zi2 = _yi2;
        _zi1 = tmp;
        in_reg[0] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[0]));
        in[tx][0 + ty] = in_reg[0];

        // n = 1
        _h2_bs = cr_d[sec][2];
        _h1_bs = cr_c[sec][2];
        _zi2 = _xc;
        _zi1 = in_reg[1];
        in_reg[2] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[2]));
        in[tx][4 + ty] = in_reg[2];

        // n = 2
        _zi2 = _zi1;
        _zi1 = in_reg[3];
        in_reg[4] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[4]));
        in[tx][8 + ty] = in_reg[4];

        // n = 3
        _zi2 = _zi1;
        _zi1 = in_reg[5];
        in_reg[6] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[6]));
        in[tx][12 + ty] = in_reg[6];

        // n = 4
        _zi2 = _zi1;
        _zi1 = in_reg[7];
        in_reg[8] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[8]));
        in[tx][16 + ty] = in_reg[8];

        // n = 5
        _zi2 = _zi1;
        _zi1 = in_reg[9];
        in_reg[10] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[10]));
        in[tx][20 + ty] = in_reg[10];

        // n = 6
        _zi2 = _zi1;
        _zi1 = in_reg[11];
        in_reg[12] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[12]));
        in[tx][24 + ty] = in_reg[12];

        // n = 7
        _zi2 = _zi1;
        _zi1 = in_reg[13];
        in_reg[14] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[14]));
        in[tx][28 + ty] = in_reg[14];

        // n = 8
        _zi2 = _zi1;
        _zi1 = in_reg[15];
        in_reg[16] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[16]));
        in[tx][32 + ty] = in_reg[16];

        // n = 9
        _zi2 = _zi1;
        _zi1 = in_reg[17];
        in_reg[18] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[18]));
        in[tx][36 + ty] = in_reg[18];

        // n = 10
        _zi2 = _zi1;
        _zi1 = in_reg[19];
        in_reg[20] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[20]));
        in[tx][40 + ty] = in_reg[20];

        // n = 11
        _zi2 = _zi1;
        _zi1 = in_reg[21];
        in_reg[22] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[22]));
        in[tx][44 + ty] = in_reg[22];

        // n = 12
        _zi2 = _zi1;
        _zi1 = in_reg[23];
        in_reg[24] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[24]));
        in[tx][48 + ty] = in_reg[24];

        // n = 13
        _zi2 = _zi1;
        _zi1 = in_reg[25];
        in_reg[26] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[26]));
        in[tx][52 + ty] = in_reg[26];

        // n = 14
        _zi2 = _zi1;
        _zi1 = in_reg[27];
        in_reg[28] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[28]));
        in[tx][56 + ty] = in_reg[28];

        // n = 15
        _zi2 = _zi1;
        _zi1 = in_reg[29];
        in_reg[30] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[30]));
        in[tx][60 + ty] = in_reg[30];
    } // end section loop

    __syncthreads();

    for (int i = tid; i < CHUNK_SIZE; i += 2 * BLOCK_SIZE) {
        int global_idx = chunk_start + i;
        int ti = i / N_BLOCKS;
        int bi = i % N_BLOCKS;
        output[global_idx] = in[ti][bi];  
    }
}
#endif  // N_BLOCKS == 64


#if N_BLOCKS == 128
static __global__ __launch_bounds__(2 * BLOCK_SIZE, DTCR_N_TB_PER_SM)
void DTCR_32_UNROLL_128(const T* const __restrict__ input, 
        T* const __restrict__ output,
        volatile int* const __restrict__ status, 
        volatile T* const __restrict__ partcarry, 
        volatile T* const __restrict__ fullcarry,
        const int launch)
{
    __shared__ T in[BLOCK_SIZE][N_BLOCKS + 1];
    __shared__ int cid;
    __shared__ T spartc[CHUNK_SIZE / warp_size * order];
    __shared__ T sfullc[order];

    const int tx = threadIdx.x;  
    const int ty = threadIdx.y;  
    const int tid = ty * BLOCK_SIZE + tx;  // 0-63
    const int lane = tid % warp_size;
    const int warp = tid / warp_size;

    // Launch-relative status values (see header comment).
    const int part_flag = 2 * launch + 1;
    const int full_flag = part_flag + 1;

    if (tid == 0) {
        cid = (int)(atomicAdd(&counter, 1u) - (unsigned int)launch * gridDim.x);
    }

    __syncthreads();

    const int chunk_id = cid;
    const int chunk_start = chunk_id * CHUNK_SIZE;

    for (int i = tid; i < CHUNK_SIZE; i += 2*BLOCK_SIZE) {
        int global_idx = chunk_start + i;
        int ti = i / N_BLOCKS;
        int bi = i % N_BLOCKS;
        in[ti][bi] = input[global_idx];
    }

    T in_reg[HALF_N_BLOCKS];
    T _xc, _xi1, _xi2, _xi3, _xi4;

    for (int sec = 0; sec < N_SECTIONS; sec++) {

        __syncthreads();

        // Phase 1: FIR + CR
        if (ty == 0) {
            if (tx == 0) {
                _xi3 = 0;
                _xi4 = 0;
                if (chunk_id == 0) {
                    _xi1 = xi1[sec];
                    _xi2 = xi2[sec];
                } else if (sec == 0) {
                    _xi1 = input[chunk_start - 1];
                    _xi2 = input[chunk_start - 2];
                }
            } else {
                _xi1 = in[tx - 1][N_BLOCKS - 1];
                _xi2 = in[tx - 1][N_BLOCKS - 2];
                _xi3 = in[tx - 1][N_BLOCKS - 3];
                _xi4 = in[tx - 1][N_BLOCKS - 4];
            }
        } else {
            _xi1 = in[tx][0];
            if (tx == 0) {
                _xi4 = 0;
                if (chunk_id == 0) {
                    _xi2 = xi1[sec];
                    _xi3 = xi2[sec];
                } else if (sec == 0) {
                    _xi2 = input[chunk_start - 1];
                    _xi3 = input[chunk_start - 2];
                }
            } else {
                _xi2 = in[tx - 1][N_BLOCKS - 1];
                _xi3 = in[tx - 1][N_BLOCKS - 2];
                _xi4 = in[tx - 1][N_BLOCKS - 3];
            }
        }

        // the first block
        T _h1, _h2, _h3;
        
        if (ty == 0) {
            if (tx == 0) {
                _h1 = b1[sec];
                _h2 = b2[sec];
                _h3 = 0;
            } else {
                _h1 = db[sec];
                _h2 = eb[sec];
                _h3 = cb[sec];
            }
        } else {
            _h1 = db[sec];  
            if (tx == 0) {
                _h2 = hb[sec];
                _h3 = gb[sec];
            } else {
                _h2 = eb[sec];
                _h3 = cb[sec];
            }
        }

        _xc = in[tx][ty];

        in_reg[0] = fmaf(_h1, _xi1, fmaf(_h2, _xi2, fmaf(_h3, _xi3, fmaf(fb[sec], _xi4, _xc))));

        _xi4 = _xi2;
        _xi3 = _xi1;
        _xi2 = _xc;

        #pragma unroll
        for (int n = 1; n < HALF_N_BLOCKS; n++) {
            _xc = in[tx][2 * n + ty];
            _xi1 = in[tx][2 * n + ty - 1];

            in_reg[n] = fmaf(db[sec], _xi1, fmaf(eb[sec], _xi2, fmaf(cb[sec], _xi3, fmaf(fb[sec], _xi4, _xc))));

            _xi4 = _xi2;
            _xi3 = _xi1;
            _xi2 = _xc;
        }

        // Remaining CR rounds - using warp shuffle (each warp independent)
        #pragma unroll
        for (int ro = 1; ro < N_BLOCKS_LOG2; ro++) {
       
            const int step = 1 << ro;
            const int sub = step >> 1; 
            const int off = sub - 1;

            _xi2 = __shfl_up_sync(0xffffffff, in_reg[HALF_N_BLOCKS - 1], 1);
            if (tx == 0) _xi2 = 0;

            #pragma unroll
            for (int n = 0; n < HALF_N_BLOCKS; n += step) {
                in_reg[n + off] = fmaf(-fde[sec][ro], _xi2, in_reg[n + off]);

                _xi2 = in_reg[n + off + sub];
                in_reg[n + off + sub] = fmaf(-e[sec][ro], in_reg[n + off], _xi2);
            }
        }

        // block filtering: each warp independent
        T _yi = h0[sec][0] * in_reg[HALF_N_BLOCKS - 1];

        #pragma unroll
        for (int n = 1; n < BLOCK_SIZE; n++) {

            _xc = __shfl_up_sync(0xffffffff, in_reg[HALF_N_BLOCKS - 1], n);

            if (tx < n) 
                _xc = 0;
            
            _yi = fmaf(h0[sec][n], _xc, _yi);
        }

        // Phase 2: Lookback
        if (tx == BLOCK_SIZE - 1) {
            partcarry[chunk_id * order * N_SECTIONS + sec * order + ty] = _yi;
        }

        __syncthreads();
        __threadfence();
        if (tid == BLOCK_SIZE - 1) 
            status[chunk_id * N_SECTIONS + sec] = part_flag;

        T X0, X1;

        if (warp == 0) {
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
                    spartc[i - start_chunk * order] = partcarry[(i / order) * order * N_SECTIONS + sec * order + (i % order)];
                }
                __syncwarp();
                
                if (lane == 0) {
                    _xi1 = C_cross[sec][BLOCK_SIZE - 1][0];
                    _xi2 = C_cross[sec][BLOCK_SIZE - 1][1];
                    _xi3 = C_cross[sec][BLOCK_SIZE - 1][2];
                    _xi4 = C_cross[sec][BLOCK_SIZE - 1][3];

                    for (int i = 0; i < num_partcarries; i++) {
                        const T p0 = spartc[i * order];
                        const T p1 = spartc[i * order + 1];
                        const T h0_val = fmaf(_xi2, X1, fmaf(_xi1, X0, p0));
                        const T h1_val = fmaf(_xi4, X1, fmaf(_xi3, X0, p1));
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
        T _h2_bs, _h1_bs;
        if (ty == 0) {
            _h2_bs = he2[sec][tx];
            _h1_bs = he1[sec][tx];
        } else {
            _h2_bs = hb2[sec][tx];
            _h1_bs = hb1[sec][tx];
        }

        in_reg[HALF_N_BLOCKS - 1] = fmaf(-_h1_bs, X1, fmaf(-_h2_bs, X0, _yi));

        if (tx == BLOCK_SIZE - 1) 
            fullcarry[chunk_id * order * N_SECTIONS + sec * order + ty] = in_reg[HALF_N_BLOCKS - 1];
        
        __syncthreads();
        __threadfence();
        if (tid == BLOCK_SIZE - 1) {
            status[chunk_id * N_SECTIONS + sec] = full_flag;
        }

        in[tx][N_BLOCKS - 2 + ty] = in_reg[HALF_N_BLOCKS - 1];

        T _yi2 = __shfl_up_sync(0xffffffff, in_reg[HALF_N_BLOCKS - 1], 2 - ty);
        T _yi1 = __shfl_up_sync(0xffffffff, in_reg[HALF_N_BLOCKS - 1], 1 + ty);
        T tmp;

        if (tx == 0) {
            tmp = X1;
            if (ty == 0) {
                _h2_bs = cr_p[sec][N_BLOCKS_LOG2 - 1];
                _h1_bs = cr_q[sec][N_BLOCKS_LOG2 - 1];
                _xc = X0;
                _xi1 = X1;
                _xi2 = X0;
            } else {
                _h2_bs = cr_h[sec][N_BLOCKS_LOG2 - 1]; 
                _h1_bs = cr_g[sec][N_BLOCKS_LOG2 - 1];  
                _xc = X1;
                _xi2 = X1;
                _xi3 = X0;
            }
            _yi2 = X0;  
            _yi1 = X1;
        } else {
            if (ty == 0) { 
                _h2_bs = cr_d[sec][N_BLOCKS_LOG2];
                _h1_bs = cr_c[sec][N_BLOCKS_LOG2];
                tmp = _yi1;
            } else {
                _h2_bs = cr_c[sec][N_BLOCKS_LOG2];
                _h1_bs = cr_d[sec][N_BLOCKS_LOG2];
                tmp = _yi2;
            }
            if (tx == 1) {
                if (ty == 0) 
                    _yi2 = X0;
                else 
                    _yi1 = X1;
            }
            _xc = tmp;
        }

        in_reg[QUARTER_N_BLOCKS - 1] = fmaf(-_h1_bs, _yi1, fmaf(-_h2_bs, _yi2, in_reg[QUARTER_N_BLOCKS - 1]));
        in[tx][HALF_N_BLOCKS - 2 + ty] = in_reg[QUARTER_N_BLOCKS - 1];

        T _zi2, _zi1;

        // ============ ro = 5 ============
        // P = 2, step = 64, sub = 32, sub2 = 16

        // n = 0
        _yi2 = __shfl_up_sync(0xffffffff, in_reg[31], 1);
        if (tx == 0) {
            if (ty == 0) {
                _h2_bs = cr_p[sec][5];
                _h1_bs = cr_q[sec][5];
            } else {
                _h2_bs = cr_h[sec][5];
                _h1_bs = cr_g[sec][5];
            }
            _yi2 = X0;
        } else {
            _h2_bs = cr_d[sec][6];
            _h1_bs = cr_c[sec][6];
        }
        _zi2 = _yi2;
        _zi1 = tmp;
        in_reg[15] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[15]));
        in[tx][30 + ty] = in_reg[15];

        // n = 1
        _h2_bs = cr_d[sec][6];
        _h1_bs = cr_c[sec][6];
        _zi2 = _xc;
        _zi1 = in_reg[31];
        in_reg[47] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[47]));
        in[tx][94 + ty] = in_reg[47];

        // ============ ro = 4 ============
        // P = 4, step = 32, sub = 16, sub2 = 8

        // n = 0
        _yi2 = __shfl_up_sync(0xffffffff, in_reg[47], 1);
        if (tx == 0) {
            if (ty == 0) {
                _h2_bs = cr_p[sec][4];
                _h1_bs = cr_q[sec][4];
            } else {
                _h2_bs = cr_h[sec][4];
                _h1_bs = cr_g[sec][4];
            }
            _yi2 = X0;
        } else {
            _h2_bs = cr_d[sec][5];
            _h1_bs = cr_c[sec][5];
        }
        _zi2 = _yi2;
        _zi1 = tmp;
        in_reg[7] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[7]));
        in[tx][14 + ty] = in_reg[7];

        // n = 1
        _h2_bs = cr_d[sec][5];
        _h1_bs = cr_c[sec][5];
        _zi2 = _xc;
        _zi1 = in_reg[15];
        in_reg[23] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[23]));
        in[tx][46 + ty] = in_reg[23];

        // n = 2
        _zi2 = _zi1;
        _zi1 = in_reg[31];
        in_reg[39] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[39]));
        in[tx][78 + ty] = in_reg[39];

        // n = 3
        _zi2 = _zi1;
        _zi1 = in_reg[47];
        in_reg[55] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[55]));
        in[tx][110 + ty] = in_reg[55];

        // ============ ro = 3 ============
        // P = 8, step = 16, sub = 8, sub2 = 4

        // n = 0
        _yi2 = __shfl_up_sync(0xffffffff, in_reg[55], 1);
        if (tx == 0) {
            if (ty == 0) {
                _h2_bs = cr_p[sec][3];
                _h1_bs = cr_q[sec][3];
            } else {
                _h2_bs = cr_h[sec][3];
                _h1_bs = cr_g[sec][3];
            }
            _yi2 = X0;
        } else {
            _h2_bs = cr_d[sec][4];
            _h1_bs = cr_c[sec][4];
        }
        _zi2 = _yi2;
        _zi1 = tmp;
        in_reg[3] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[3]));
        in[tx][6 + ty] = in_reg[3];

        // n = 1
        _h2_bs = cr_d[sec][4];
        _h1_bs = cr_c[sec][4];
        _zi2 = _xc;
        _zi1 = in_reg[7];
        in_reg[11] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[11]));
        in[tx][22 + ty] = in_reg[11];

        // n = 2
        _zi2 = _zi1;
        _zi1 = in_reg[15];
        in_reg[19] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[19]));
        in[tx][38 + ty] = in_reg[19];

        // n = 3
        _zi2 = _zi1;
        _zi1 = in_reg[23];
        in_reg[27] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[27]));
        in[tx][54 + ty] = in_reg[27];

        // n = 4
        _zi2 = _zi1;
        _zi1 = in_reg[31];
        in_reg[35] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[35]));
        in[tx][70 + ty] = in_reg[35];

        // n = 5
        _zi2 = _zi1;
        _zi1 = in_reg[39];
        in_reg[43] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[43]));
        in[tx][86 + ty] = in_reg[43];

        // n = 6
        _zi2 = _zi1;
        _zi1 = in_reg[47];
        in_reg[51] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[51]));
        in[tx][102 + ty] = in_reg[51];

        // n = 7
        _zi2 = _zi1;
        _zi1 = in_reg[55];
        in_reg[59] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[59]));
        in[tx][118 + ty] = in_reg[59];

        // ============ ro = 2 ============
        // P = 16, step = 8, sub = 4, sub2 = 2

        // n = 0
        _yi2 = __shfl_up_sync(0xffffffff, in_reg[59], 1);
        if (tx == 0) {
            if (ty == 0) {
                _h2_bs = cr_p[sec][2];
                _h1_bs = cr_q[sec][2];
            } else {
                _h2_bs = cr_h[sec][2];
                _h1_bs = cr_g[sec][2];
            }
            _yi2 = X0;
        } else {
            _h2_bs = cr_d[sec][3];
            _h1_bs = cr_c[sec][3];
        }
        _zi2 = _yi2;
        _zi1 = tmp;
        in_reg[1] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[1]));
        in[tx][2 + ty] = in_reg[1];

        // n = 1
        _h2_bs = cr_d[sec][3];
        _h1_bs = cr_c[sec][3];
        _zi2 = _xc;
        _zi1 = in_reg[3];
        in_reg[5] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[5]));
        in[tx][10 + ty] = in_reg[5];

        // n = 2
        _zi2 = _zi1;
        _zi1 = in_reg[7];
        in_reg[9] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[9]));
        in[tx][18 + ty] = in_reg[9];

        // n = 3
        _zi2 = _zi1;
        _zi1 = in_reg[11];
        in_reg[13] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[13]));
        in[tx][26 + ty] = in_reg[13];

        // n = 4
        _zi2 = _zi1;
        _zi1 = in_reg[15];
        in_reg[17] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[17]));
        in[tx][34 + ty] = in_reg[17];

        // n = 5
        _zi2 = _zi1;
        _zi1 = in_reg[19];
        in_reg[21] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[21]));
        in[tx][42 + ty] = in_reg[21];

        // n = 6
        _zi2 = _zi1;
        _zi1 = in_reg[23];
        in_reg[25] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[25]));
        in[tx][50 + ty] = in_reg[25];

        // n = 7
        _zi2 = _zi1;
        _zi1 = in_reg[27];
        in_reg[29] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[29]));
        in[tx][58 + ty] = in_reg[29];

        // n = 8
        _zi2 = _zi1;
        _zi1 = in_reg[31];
        in_reg[33] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[33]));
        in[tx][66 + ty] = in_reg[33];

        // n = 9
        _zi2 = _zi1;
        _zi1 = in_reg[35];
        in_reg[37] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[37]));
        in[tx][74 + ty] = in_reg[37];

        // n = 10
        _zi2 = _zi1;
        _zi1 = in_reg[39];
        in_reg[41] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[41]));
        in[tx][82 + ty] = in_reg[41];

        // n = 11
        _zi2 = _zi1;
        _zi1 = in_reg[43];
        in_reg[45] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[45]));
        in[tx][90 + ty] = in_reg[45];

        // n = 12
        _zi2 = _zi1;
        _zi1 = in_reg[47];
        in_reg[49] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[49]));
        in[tx][98 + ty] = in_reg[49];

        // n = 13
        _zi2 = _zi1;
        _zi1 = in_reg[51];
        in_reg[53] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[53]));
        in[tx][106 + ty] = in_reg[53];

        // n = 14
        _zi2 = _zi1;
        _zi1 = in_reg[55];
        in_reg[57] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[57]));
        in[tx][114 + ty] = in_reg[57];

        // n = 15
        _zi2 = _zi1;
        _zi1 = in_reg[59];
        in_reg[61] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[61]));
        in[tx][122 + ty] = in_reg[61];

        // ============ ro = 1 ============
        // P = 32, step = 4, sub = 2, sub2 = 1

        // n = 0
        _yi2 = __shfl_up_sync(0xffffffff, in_reg[61], 1);
        if (tx == 0) {
            if (ty == 0) {
                _h2_bs = cr_p[sec][1];
                _h1_bs = cr_q[sec][1];
            } else {
                _h2_bs = cr_h[sec][1];
                _h1_bs = cr_g[sec][1];
            }
            _yi2 = X0;
        } else {
            _h2_bs = cr_d[sec][2];
            _h1_bs = cr_c[sec][2];
        }
        _zi2 = _yi2;
        _zi1 = tmp;
        in_reg[0] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[0]));
        in[tx][0 + ty] = in_reg[0];

        // n = 1
        _h2_bs = cr_d[sec][2];
        _h1_bs = cr_c[sec][2];
        _zi2 = _xc;
        _zi1 = in_reg[1];
        in_reg[2] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[2]));
        in[tx][4 + ty] = in_reg[2];

        // n = 2
        _zi2 = _zi1;
        _zi1 = in_reg[3];
        in_reg[4] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[4]));
        in[tx][8 + ty] = in_reg[4];

        // n = 3
        _zi2 = _zi1;
        _zi1 = in_reg[5];
        in_reg[6] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[6]));
        in[tx][12 + ty] = in_reg[6];

        // n = 4
        _zi2 = _zi1;
        _zi1 = in_reg[7];
        in_reg[8] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[8]));
        in[tx][16 + ty] = in_reg[8];

        // n = 5
        _zi2 = _zi1;
        _zi1 = in_reg[9];
        in_reg[10] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[10]));
        in[tx][20 + ty] = in_reg[10];

        // n = 6
        _zi2 = _zi1;
        _zi1 = in_reg[11];
        in_reg[12] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[12]));
        in[tx][24 + ty] = in_reg[12];

        // n = 7
        _zi2 = _zi1;
        _zi1 = in_reg[13];
        in_reg[14] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[14]));
        in[tx][28 + ty] = in_reg[14];

        // n = 8
        _zi2 = _zi1;
        _zi1 = in_reg[15];
        in_reg[16] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[16]));
        in[tx][32 + ty] = in_reg[16];

        // n = 9
        _zi2 = _zi1;
        _zi1 = in_reg[17];
        in_reg[18] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[18]));
        in[tx][36 + ty] = in_reg[18];

        // n = 10
        _zi2 = _zi1;
        _zi1 = in_reg[19];
        in_reg[20] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[20]));
        in[tx][40 + ty] = in_reg[20];

        // n = 11
        _zi2 = _zi1;
        _zi1 = in_reg[21];
        in_reg[22] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[22]));
        in[tx][44 + ty] = in_reg[22];

        // n = 12
        _zi2 = _zi1;
        _zi1 = in_reg[23];
        in_reg[24] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[24]));
        in[tx][48 + ty] = in_reg[24];

        // n = 13
        _zi2 = _zi1;
        _zi1 = in_reg[25];
        in_reg[26] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[26]));
        in[tx][52 + ty] = in_reg[26];

        // n = 14
        _zi2 = _zi1;
        _zi1 = in_reg[27];
        in_reg[28] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[28]));
        in[tx][56 + ty] = in_reg[28];

        // n = 15
        _zi2 = _zi1;
        _zi1 = in_reg[29];
        in_reg[30] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[30]));
        in[tx][60 + ty] = in_reg[30];

        // n = 16
        _zi2 = _zi1;
        _zi1 = in_reg[31];
        in_reg[32] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[32]));
        in[tx][64 + ty] = in_reg[32];

        // n = 17
        _zi2 = _zi1;
        _zi1 = in_reg[33];
        in_reg[34] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[34]));
        in[tx][68 + ty] = in_reg[34];

        // n = 18
        _zi2 = _zi1;
        _zi1 = in_reg[35];
        in_reg[36] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[36]));
        in[tx][72 + ty] = in_reg[36];

        // n = 19
        _zi2 = _zi1;
        _zi1 = in_reg[37];
        in_reg[38] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[38]));
        in[tx][76 + ty] = in_reg[38];

        // n = 20
        _zi2 = _zi1;
        _zi1 = in_reg[39];
        in_reg[40] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[40]));
        in[tx][80 + ty] = in_reg[40];

        // n = 21
        _zi2 = _zi1;
        _zi1 = in_reg[41];
        in_reg[42] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[42]));
        in[tx][84 + ty] = in_reg[42];

        // n = 22
        _zi2 = _zi1;
        _zi1 = in_reg[43];
        in_reg[44] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[44]));
        in[tx][88 + ty] = in_reg[44];

        // n = 23
        _zi2 = _zi1;
        _zi1 = in_reg[45];
        in_reg[46] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[46]));
        in[tx][92 + ty] = in_reg[46];

        // n = 24
        _zi2 = _zi1;
        _zi1 = in_reg[47];
        in_reg[48] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[48]));
        in[tx][96 + ty] = in_reg[48];

        // n = 25
        _zi2 = _zi1;
        _zi1 = in_reg[49];
        in_reg[50] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[50]));
        in[tx][100 + ty] = in_reg[50];

        // n = 26
        _zi2 = _zi1;
        _zi1 = in_reg[51];
        in_reg[52] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[52]));
        in[tx][104 + ty] = in_reg[52];

        // n = 27
        _zi2 = _zi1;
        _zi1 = in_reg[53];
        in_reg[54] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[54]));
        in[tx][108 + ty] = in_reg[54];

        // n = 28
        _zi2 = _zi1;
        _zi1 = in_reg[55];
        in_reg[56] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[56]));
        in[tx][112 + ty] = in_reg[56];

        // n = 29
        _zi2 = _zi1;
        _zi1 = in_reg[57];
        in_reg[58] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[58]));
        in[tx][116 + ty] = in_reg[58];

        // n = 30
        _zi2 = _zi1;
        _zi1 = in_reg[59];
        in_reg[60] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[60]));
        in[tx][120 + ty] = in_reg[60];

        // n = 31
        _zi2 = _zi1;
        _zi1 = in_reg[61];
        in_reg[62] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[62]));
        in[tx][124 + ty] = in_reg[62];
    } // end section loop

    __syncthreads();

    for (int i = tid; i < CHUNK_SIZE; i += 2 * BLOCK_SIZE) {
        int global_idx = chunk_start + i;
        int ti = i / N_BLOCKS;
        int bi = i % N_BLOCKS;
        output[global_idx] = in[ti][bi];  
    }
}
#endif  // N_BLOCKS == 128


static __global__ __launch_bounds__(2 * BLOCK_SIZE, DTCR_N_TB_PER_SM)
void DTCR_64_LOOP(const T* const __restrict__ input, 
        T* const __restrict__ output,
        volatile int* const __restrict__ status, 
        volatile T* const __restrict__ partcarry, 
        volatile T* const __restrict__ fullcarry,
        const int launch)
{
    __shared__ T in[BLOCK_SIZE][N_BLOCKS + 1];  // 64 x 65
    __shared__ int cid;
    __shared__ T sfullc[order];

    const int tx = threadIdx.x;  // 0-63
    const int ty = threadIdx.y;  // 0-1
    const int tid = ty * BLOCK_SIZE + tx;  // 0-127
    const int lane = tid % warp_size;
    const int warp = tid / warp_size;  // 0-3

    // Launch-relative status values (see header comment).
    const int part_flag = 2 * launch + 1;
    const int full_flag = part_flag + 1;

    if (tid == 0) {
        cid = (int)(atomicAdd(&counter, 1u) - (unsigned int)launch * gridDim.x);
    }

    __syncthreads();

    const int chunk_id = cid;
    const int chunk_start = chunk_id * CHUNK_SIZE;

    for (int i = tid; i < CHUNK_SIZE; i += 2*BLOCK_SIZE) {
        int global_idx = chunk_start + i;
        int ti = i / N_BLOCKS;  // i / 64
        int bi = i % N_BLOCKS;  // i % 64
        in[ti][bi] = input[global_idx];
    }

    T in_reg[HALF_N_BLOCKS];  // 32 registers
    T _xc, _xi1, _xi2, _xi3, _xi4;

    __syncthreads();

    for (int sec = 0; sec < N_SECTIONS; sec++) {

        // Phase 1: FIR + CR
        if (ty == 0) {
            if (tx == 0) {
                _xi3 = 0;
                _xi4 = 0;
                if (chunk_id == 0) {
                    _xi1 = xi1[sec];
                    _xi2 = xi2[sec];
                } else if (sec == 0) {
                    _xi1 = input[chunk_start - 1];
                    _xi2 = input[chunk_start - 2];
                }
            } else {
                _xi1 = in[tx - 1][N_BLOCKS - 1];
                _xi2 = in[tx - 1][N_BLOCKS - 2];
                _xi3 = in[tx - 1][N_BLOCKS - 3];
                _xi4 = in[tx - 1][N_BLOCKS - 4];
            }
        } else {
            _xi1 = in[tx][0];
            if (tx == 0) {
                _xi4 = 0;
                if (chunk_id == 0) {
                    _xi2 = xi1[sec];
                    _xi3 = xi2[sec];
                } else if (sec == 0) {
                    _xi2 = input[chunk_start - 1];
                    _xi3 = input[chunk_start - 2];
                }
            } else {
                _xi2 = in[tx - 1][N_BLOCKS - 1];
                _xi3 = in[tx - 1][N_BLOCKS - 2];
                _xi4 = in[tx - 1][N_BLOCKS - 3];
            }
        }

        // the first block
        T _h1, _h2, _h3;
        
        if (ty == 0) {
            if (tx == 0) {
                _h1 = b1[sec];
                _h2 = b2[sec];
                _h3 = 0;
            } else {
                _h1 = db[sec];
                _h2 = eb[sec];
                _h3 = cb[sec];
            }
        } else {
            _h1 = db[sec];  
            if (tx == 0) {
                _h2 = hb[sec];
                _h3 = gb[sec];
            } else {
                _h2 = eb[sec];
                _h3 = cb[sec];
            }
        }

        _xc = in[tx][ty];

        in_reg[0] = fmaf(_h1, _xi1, fmaf(_h2, _xi2, fmaf(_h3, _xi3, fmaf(fb[sec], _xi4, _xc))));

        _xi4 = _xi2;
        _xi3 = _xi1;
        _xi2 = _xc;

        #pragma unroll
        for (int n = 1; n < HALF_N_BLOCKS; n++) {
            _xc = in[tx][2 * n + ty];
            _xi1 = in[tx][2 * n + ty - 1];

            in_reg[n] = fmaf(db[sec], _xi1, fmaf(eb[sec], _xi2, fmaf(cb[sec], _xi3, fmaf(fb[sec], _xi4, _xc))));

            _xi4 = _xi2;
            _xi3 = _xi1;
            _xi2 = _xc;
        }

        // RACE FIX: the publish below overwrites in[tx][0] / in[tx][1] --
        // the first two samples of this lane -- which threads of the OPPOSITE
        // parity (different warps) may still be reading during their fused
        // pass or boundary setup. The original code synchronized only AFTER
        // the publish; a fast warp could clobber the slots first (observed on
        // Ampere as a batch-dependent accuracy failure at bit-identical
        // positions). All pass reads must complete before any publish:
        __syncthreads();

        in[tx][ty] = in_reg[HALF_N_BLOCKS - 1];

        __syncthreads();

        #pragma unroll
        for (int r = 1; r < N_BLOCKS_LOG2; r++) {

            const int step = 1 << r;
            const int sub = step >> 1; // 1
            const int off = sub - 1; // 0  

            if (tx == 0)
                _xi2 = 0;
            else
                _xi2 = in[tx - 1][ty];

            __syncthreads();
            
            #pragma unroll
            for (int n = 0; n < HALF_N_BLOCKS; n += step) { // no ty bc symm

                in_reg[n + off] = fmaf(-fde[sec][r], _xi2, in_reg[n + off]);

                _xi2 = in_reg[n + off + sub];

                in_reg[n + off + sub] = fmaf(-e[sec][r], in_reg[n + off], _xi2);

            }

            in[tx][ty] = in_reg[HALF_N_BLOCKS - 1];

            __syncthreads();
        }

        // block filtering
        T _yi = h0[sec][0] * in_reg[HALF_N_BLOCKS - 1];

        #pragma unroll
        for (int n = 1; n < BLOCK_SIZE; n++) {

            if (tx >= n) {
                _xc = in[tx - n][ty];
            } else {
                _xc = 0;
            }

            _yi = fmaf(h0[sec][n], _xc, _yi);
        }

        // Phase 2: Lookback
        if (tx == BLOCK_SIZE - 1) {
            partcarry[chunk_id * order * N_SECTIONS + sec * order + ty] = _yi;
        }

        __syncthreads();
        __threadfence();
        if (tid == BLOCK_SIZE - 1) 
            status[chunk_id * N_SECTIONS + sec] = part_flag;

        T X0, X1;

        if (warp == 0) {
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
                    _xi1 = C_cross[sec][BLOCK_SIZE - 1][0];
                    _xi2 = C_cross[sec][BLOCK_SIZE - 1][1];
                    _xi3 = C_cross[sec][BLOCK_SIZE - 1][2];
                    _xi4 = C_cross[sec][BLOCK_SIZE - 1][3];

                    for (int i = 0; i < num_partcarries; i++) {
                        const T p0 = in[i * order][0];
                        const T p1 = in[i * order + 1][0];
                        const T h0_val = fmaf(_xi2, X1, fmaf(_xi1, X0, p0));
                        const T h1_val = fmaf(_xi4, X1, fmaf(_xi3, X0, p1));
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
        T _h2_bs, _h1_bs;
        if (ty == 0) {
            _h2_bs = he2[sec][tx];
            _h1_bs = he1[sec][tx];
        } else {
            _h2_bs = hb2[sec][tx];
            _h1_bs = hb1[sec][tx];
        }

        in_reg[HALF_N_BLOCKS - 1] = fmaf(-_h1_bs, X1, fmaf(-_h2_bs, X0, _yi));

        if (tx == BLOCK_SIZE - 1) 
            fullcarry[chunk_id * order * N_SECTIONS + sec * order + ty] = in_reg[HALF_N_BLOCKS - 1];
        
        __syncthreads();
        __threadfence();
        if (tid == BLOCK_SIZE - 1) {
            status[chunk_id * N_SECTIONS + sec] = full_flag;
        }

        in[tx][N_BLOCKS - 2 + ty] = in_reg[HALF_N_BLOCKS - 1];
        __syncthreads();


        T _yi2, _yi1;
        T tmp;


        if (tx == 0) {
            tmp = X1;
            if (ty == 0) {
                _h2_bs = cr_p[sec][N_BLOCKS_LOG2 - 1];
                _h1_bs = cr_q[sec][N_BLOCKS_LOG2 - 1];
                _xc = X0;
                _xi1 = X1;
                _xi2 = X0;
            } else {
                _h2_bs = cr_h[sec][N_BLOCKS_LOG2 - 1];
                _h1_bs = cr_g[sec][N_BLOCKS_LOG2 - 1];
                _xc = X1;
                _xi2 = X1;
                _xi3 = X0;
            } 
            _yi2 = X0;  
            _yi1 = X1;
        } else {
            tmp = in[tx - 1][N_BLOCKS - 2 + ty];
            _xc = tmp;
            if (ty == 0) {
                _h2_bs = cr_d[sec][N_BLOCKS_LOG2];
                _h1_bs = cr_c[sec][N_BLOCKS_LOG2];
                _yi1 = tmp;
                if (tx == 1) {
                    _yi2 = X0;
                } else 
                    _yi2 = in[tx - 2][N_BLOCKS - 2];
            } else {
                _h2_bs = cr_c[sec][N_BLOCKS_LOG2];
                _h1_bs = cr_d[sec][N_BLOCKS_LOG2];
                _yi2 = in[tx - 1][N_BLOCKS - 1];
                if (tx == 1) {
                    _yi1 = X1;
                } else 
                    _yi1 = in[tx - 2][N_BLOCKS - 1];
            }

        }

        in_reg[QUARTER_N_BLOCKS - 1] = fmaf(-_h1_bs, _yi1, fmaf(-_h2_bs, _yi2, in_reg[QUARTER_N_BLOCKS - 1]));
        in[tx][HALF_N_BLOCKS - 2 + ty] = in_reg[QUARTER_N_BLOCKS - 1];

        // The first back-substitution round reads in[tx-1][...] written just
        // above by threads of another warp; the original code had no barrier
        // here (latent cross-warp race, tolerated on Pascal). Made explicit:
        __syncthreads();
    
        T _zi1,_zi2;

        #pragma unroll 
        for (int r = N_BLOCKS_LOG2 - 2; r > 0; r--) {

            const int P = HALF_N_BLOCKS >> r; 
            const int stride = 2 << r; 
            const int sub = 1 << r; 
            const int sub2 = sub >> 1; 

            #pragma unroll
            for (int n = 0; n < P; n++) {

                if (n == 0) {
                    if (tx == 0) {
                        if (ty == 0){
                            _h2_bs = cr_p[sec][r];
                            _h1_bs = cr_q[sec][r];
                        } else {
                            _h2_bs = cr_h[sec][r];
                            _h1_bs = cr_g[sec][r];
                        }
                    } else {
                        _h2_bs = cr_d[sec][r + 1];
                        _h1_bs = cr_c[sec][r + 1];
                        _yi2 = in[tx - 1][N_BLOCKS - stride - 2 + ty];
                    }
                    _zi2 = _yi2;
                    _zi1 = tmp;

                } else if (n == 1) {

                    _h2_bs = cr_d[sec][r + 1];
                    _h1_bs = cr_c[sec][r + 1];

                    _zi2 = _xc;
                    _zi1 = in_reg[sub - 1]; 

                } else {
                    _zi2 = _zi1;
                    _zi1 = in_reg[sub * n - 1];
                }

                in_reg[sub*n + sub2 - 1] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[sub*n + sub2 - 1]));
                in[tx][stride*n + sub - 2 + ty] = in_reg[sub*n + sub2 - 1];
            }
            

            __syncthreads();

        }


    } // end section loop

    for (int i = tid; i < CHUNK_SIZE; i += 2 * BLOCK_SIZE) {
        int global_idx = chunk_start + i;
        int ti = i / N_BLOCKS;
        int bi = i % N_BLOCKS;
        output[global_idx] = in[ti][bi];  
    }
}


#if N_BLOCKS == 64
static __global__ __launch_bounds__(2 * BLOCK_SIZE, DTCR_N_TB_PER_SM)
void DTCR_64_UNROLL_64(const T* const __restrict__ input, 
        T* const __restrict__ output,
        volatile int* const __restrict__ status, 
        volatile T* const __restrict__ partcarry, 
        volatile T* const __restrict__ fullcarry,
        const int launch)
{
    __shared__ T in[BLOCK_SIZE][N_BLOCKS + 1];  // 64 x 65
    __shared__ int cid;
    __shared__ T sfullc[order];

    const int tx = threadIdx.x;  // 0-63
    const int ty = threadIdx.y;  // 0-1
    const int tid = ty * BLOCK_SIZE + tx;  // 0-127
    const int lane = tid % warp_size;
    const int warp = tid / warp_size;  // 0-3

    // Launch-relative status values (see header comment).
    const int part_flag = 2 * launch + 1;
    const int full_flag = part_flag + 1;

    if (tid == 0) {
        cid = (int)(atomicAdd(&counter, 1u) - (unsigned int)launch * gridDim.x);
    }

    __syncthreads();

    const int chunk_id = cid;
    const int chunk_start = chunk_id * CHUNK_SIZE;

    for (int i = tid; i < CHUNK_SIZE; i += 2*BLOCK_SIZE) {
        int global_idx = chunk_start + i;
        int ti = i / N_BLOCKS;  // i / 64
        int bi = i % N_BLOCKS;  // i % 64
        in[ti][bi] = input[global_idx];
    }

    T in_reg[HALF_N_BLOCKS];  // 32 registers
    T _xc, _xi1, _xi2, _xi3, _xi4;

    __syncthreads();

    for (int sec = 0; sec < N_SECTIONS; sec++) {

        // Phase 1: FIR + CR
        if (ty == 0) {
            if (tx == 0) {
                _xi3 = 0;
                _xi4 = 0;
                if (chunk_id == 0) {
                    _xi1 = xi1[sec];
                    _xi2 = xi2[sec];
                } else if (sec == 0) {
                    _xi1 = input[chunk_start - 1];
                    _xi2 = input[chunk_start - 2];
                }
            } else {
                _xi1 = in[tx - 1][N_BLOCKS - 1];
                _xi2 = in[tx - 1][N_BLOCKS - 2];
                _xi3 = in[tx - 1][N_BLOCKS - 3];
                _xi4 = in[tx - 1][N_BLOCKS - 4];
            }
        } else {
            _xi1 = in[tx][0];
            if (tx == 0) {
                _xi4 = 0;
                if (chunk_id == 0) {
                    _xi2 = xi1[sec];
                    _xi3 = xi2[sec];
                } else if (sec == 0) {
                    _xi2 = input[chunk_start - 1];
                    _xi3 = input[chunk_start - 2];
                }
            } else {
                _xi2 = in[tx - 1][N_BLOCKS - 1];
                _xi3 = in[tx - 1][N_BLOCKS - 2];
                _xi4 = in[tx - 1][N_BLOCKS - 3];
            }
        }

        // the first block
        T _h1, _h2, _h3;
        
        if (ty == 0) {
            if (tx == 0) {
                _h1 = b1[sec];
                _h2 = b2[sec];
                _h3 = 0;
            } else {
                _h1 = db[sec];
                _h2 = eb[sec];
                _h3 = cb[sec];
            }
        } else {
            _h1 = db[sec];  
            if (tx == 0) {
                _h2 = hb[sec];
                _h3 = gb[sec];
            } else {
                _h2 = eb[sec];
                _h3 = cb[sec];
            }
        }

        _xc = in[tx][ty];

        in_reg[0] = fmaf(_h1, _xi1, fmaf(_h2, _xi2, fmaf(_h3, _xi3, fmaf(fb[sec], _xi4, _xc))));

        _xi4 = _xi2;
        _xi3 = _xi1;
        _xi2 = _xc;

        #pragma unroll
        for (int n = 1; n < HALF_N_BLOCKS; n++) {
            _xc = in[tx][2 * n + ty];
            _xi1 = in[tx][2 * n + ty - 1];

            in_reg[n] = fmaf(db[sec], _xi1, fmaf(eb[sec], _xi2, fmaf(cb[sec], _xi3, fmaf(fb[sec], _xi4, _xc))));

            _xi4 = _xi2;
            _xi3 = _xi1;
            _xi2 = _xc;
        }

        // RACE FIX: the publish below overwrites in[tx][0] / in[tx][1] --
        // the first two samples of this lane -- which threads of the OPPOSITE
        // parity (different warps) may still be reading during their fused
        // pass or boundary setup. The original code synchronized only AFTER
        // the publish; a fast warp could clobber the slots first (observed on
        // Ampere as a batch-dependent accuracy failure at bit-identical
        // positions). All pass reads must complete before any publish:
        __syncthreads();

        in[tx][ty] = in_reg[HALF_N_BLOCKS - 1];

        __syncthreads();

        #pragma unroll
        for (int r = 1; r < N_BLOCKS_LOG2; r++) {

            const int step = 1 << r;
            const int sub = step >> 1; // 1
            const int off = sub - 1; // 0  

            if (tx == 0)
                _xi2 = 0;
            else
                _xi2 = in[tx - 1][ty];

            __syncthreads();
            
            #pragma unroll
            for (int n = 0; n < HALF_N_BLOCKS; n += step) { // no ty bc symm

                in_reg[n + off] = fmaf(-fde[sec][r], _xi2, in_reg[n + off]);

                _xi2 = in_reg[n + off + sub];

                in_reg[n + off + sub] = fmaf(-e[sec][r], in_reg[n + off], _xi2);

            }

            in[tx][ty] = in_reg[HALF_N_BLOCKS - 1];

            __syncthreads();
        }

        // block filtering
        T _yi = h0[sec][0] * in_reg[HALF_N_BLOCKS - 1];

        #pragma unroll
        for (int n = 1; n < BLOCK_SIZE; n++) {

            if (tx >= n) {
                _xc = in[tx - n][ty];
            } else {
                _xc = 0;
            }

            _yi = fmaf(h0[sec][n], _xc, _yi);
        }

        // Phase 2: Lookback
        if (tx == BLOCK_SIZE - 1) {
            partcarry[chunk_id * order * N_SECTIONS + sec * order + ty] = _yi;
        }

        __syncthreads();
        __threadfence();
        if (tid == BLOCK_SIZE - 1) 
            status[chunk_id * N_SECTIONS + sec] = part_flag;

        T X0, X1;

        if (warp == 0) {
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
                    _xi1 = C_cross[sec][BLOCK_SIZE - 1][0];
                    _xi2 = C_cross[sec][BLOCK_SIZE - 1][1];
                    _xi3 = C_cross[sec][BLOCK_SIZE - 1][2];
                    _xi4 = C_cross[sec][BLOCK_SIZE - 1][3];

                    for (int i = 0; i < num_partcarries; i++) {
                        const T p0 = in[i * order][0];
                        const T p1 = in[i * order + 1][0];
                        const T h0_val = fmaf(_xi2, X1, fmaf(_xi1, X0, p0));
                        const T h1_val = fmaf(_xi4, X1, fmaf(_xi3, X0, p1));
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
        T _h2_bs, _h1_bs;
        if (ty == 0) {
            _h2_bs = he2[sec][tx];
            _h1_bs = he1[sec][tx];
        } else {
            _h2_bs = hb2[sec][tx];
            _h1_bs = hb1[sec][tx];
        }

        in_reg[HALF_N_BLOCKS - 1] = fmaf(-_h1_bs, X1, fmaf(-_h2_bs, X0, _yi));

        if (tx == BLOCK_SIZE - 1) 
            fullcarry[chunk_id * order * N_SECTIONS + sec * order + ty] = in_reg[HALF_N_BLOCKS - 1];
        
        __syncthreads();
        __threadfence();
        if (tid == BLOCK_SIZE - 1) {
            status[chunk_id * N_SECTIONS + sec] = full_flag;
        }

        in[tx][N_BLOCKS - 2 + ty] = in_reg[HALF_N_BLOCKS - 1];
        __syncthreads();


        T _yi2, _yi1;
        T tmp;


        if (tx == 0) {
            tmp = X1;
            if (ty == 0) {
                _h2_bs = cr_p[sec][N_BLOCKS_LOG2 - 1];
                _h1_bs = cr_q[sec][N_BLOCKS_LOG2 - 1];
                _xc = X0;
                _xi1 = X1;
                _xi2 = X0;
            } else {
                _h2_bs = cr_h[sec][N_BLOCKS_LOG2 - 1];
                _h1_bs = cr_g[sec][N_BLOCKS_LOG2 - 1];
                _xc = X1;
                _xi2 = X1;
                _xi3 = X0;
            } 
            _yi2 = X0;  
            _yi1 = X1;
        } else {
            tmp = in[tx - 1][N_BLOCKS - 2 + ty];
            _xc = tmp;
            if (ty == 0) {
                _h2_bs = cr_d[sec][N_BLOCKS_LOG2];
                _h1_bs = cr_c[sec][N_BLOCKS_LOG2];
                _yi1 = tmp;
                if (tx == 1) {
                    _yi2 = X0;
                } else 
                    _yi2 = in[tx - 2][N_BLOCKS - 2];
            } else {
                _h2_bs = cr_c[sec][N_BLOCKS_LOG2];
                _h1_bs = cr_d[sec][N_BLOCKS_LOG2];
                _yi2 = in[tx - 1][N_BLOCKS - 1];
                if (tx == 1) {
                    _yi1 = X1;
                } else 
                    _yi1 = in[tx - 2][N_BLOCKS - 1];
            }

        }

        in_reg[QUARTER_N_BLOCKS - 1] = fmaf(-_h1_bs, _yi1, fmaf(-_h2_bs, _yi2, in_reg[QUARTER_N_BLOCKS - 1]));
        in[tx][HALF_N_BLOCKS - 2 + ty] = in_reg[QUARTER_N_BLOCKS - 1];

        // The first back-substitution round reads in[tx-1][...] written just
        // above by threads of another warp; the original code had no barrier
        // here (latent cross-warp race, tolerated on Pascal). Made explicit:
        __syncthreads();
    
        T _zi1, _zi2;

        // ============ r = 4 ============
        // P = 2, stride = 32, sub = 16, sub2 = 8

        // n = 0
        if (tx == 0) {
            if (ty == 0) {
                _h2_bs = cr_p[sec][4];
                _h1_bs = cr_q[sec][4];
            } else {
                _h2_bs = cr_h[sec][4];
                _h1_bs = cr_g[sec][4];
            }
        } else {
            _h2_bs = cr_d[sec][5];
            _h1_bs = cr_c[sec][5];
            _yi2 = in[tx - 1][30 + ty];  // N_BLOCKS - 32 - 2 + ty
        }
        _zi2 = _yi2;
        _zi1 = tmp;
        in_reg[7] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[7]));
        in[tx][14 + ty] = in_reg[7];

        // n = 1
        _h2_bs = cr_d[sec][5];
        _h1_bs = cr_c[sec][5];
        _zi2 = _xc;
        _zi1 = in_reg[15];
        in_reg[23] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[23]));
        in[tx][46 + ty] = in_reg[23];

        __syncthreads();

        // ============ r = 3 ============
        // P = 4, stride = 16, sub = 8, sub2 = 4

        // n = 0
        if (tx == 0) {
            if (ty == 0) {
                _h2_bs = cr_p[sec][3];
                _h1_bs = cr_q[sec][3];
            } else {
                _h2_bs = cr_h[sec][3];
                _h1_bs = cr_g[sec][3];
            }
        } else {
            _h2_bs = cr_d[sec][4];
            _h1_bs = cr_c[sec][4];
            _yi2 = in[tx - 1][46 + ty];  // N_BLOCKS - 16 - 2 + ty
        }
        _zi2 = _yi2;
        _zi1 = tmp;
        in_reg[3] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[3]));
        in[tx][6 + ty] = in_reg[3];

        // n = 1
        _h2_bs = cr_d[sec][4];
        _h1_bs = cr_c[sec][4];
        _zi2 = _xc;
        _zi1 = in_reg[7];
        in_reg[11] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[11]));
        in[tx][22 + ty] = in_reg[11];

        // n = 2
        _zi2 = _zi1;
        _zi1 = in_reg[15];
        in_reg[19] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[19]));
        in[tx][38 + ty] = in_reg[19];

        // n = 3
        _zi2 = _zi1;
        _zi1 = in_reg[23];
        in_reg[27] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[27]));
        in[tx][54 + ty] = in_reg[27];

        __syncthreads();

        // ============ r = 2 ============
        // P = 8, stride = 8, sub = 4, sub2 = 2

        // n = 0
        if (tx == 0) {
            if (ty == 0) {
                _h2_bs = cr_p[sec][2];
                _h1_bs = cr_q[sec][2];
            } else {
                _h2_bs = cr_h[sec][2];
                _h1_bs = cr_g[sec][2];
            }
        } else {
            _h2_bs = cr_d[sec][3];
            _h1_bs = cr_c[sec][3];
            _yi2 = in[tx - 1][54 + ty];  // N_BLOCKS - 8 - 2 + ty
        }
        _zi2 = _yi2;
        _zi1 = tmp;
        in_reg[1] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[1]));
        in[tx][2 + ty] = in_reg[1];

        // n = 1
        _h2_bs = cr_d[sec][3];
        _h1_bs = cr_c[sec][3];
        _zi2 = _xc;
        _zi1 = in_reg[3];
        in_reg[5] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[5]));
        in[tx][10 + ty] = in_reg[5];

        // n = 2
        _zi2 = _zi1;
        _zi1 = in_reg[7];
        in_reg[9] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[9]));
        in[tx][18 + ty] = in_reg[9];

        // n = 3
        _zi2 = _zi1;
        _zi1 = in_reg[11];
        in_reg[13] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[13]));
        in[tx][26 + ty] = in_reg[13];

        // n = 4
        _zi2 = _zi1;
        _zi1 = in_reg[15];
        in_reg[17] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[17]));
        in[tx][34 + ty] = in_reg[17];

        // n = 5
        _zi2 = _zi1;
        _zi1 = in_reg[19];
        in_reg[21] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[21]));
        in[tx][42 + ty] = in_reg[21];

        // n = 6
        _zi2 = _zi1;
        _zi1 = in_reg[23];
        in_reg[25] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[25]));
        in[tx][50 + ty] = in_reg[25];

        // n = 7
        _zi2 = _zi1;
        _zi1 = in_reg[27];
        in_reg[29] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[29]));
        in[tx][58 + ty] = in_reg[29];

        __syncthreads();

        // ============ r = 1 ============
        // P = 16, stride = 4, sub = 2, sub2 = 1

        // n = 0
        if (tx == 0) {
            if (ty == 0) {
                _h2_bs = cr_p[sec][1];
                _h1_bs = cr_q[sec][1];
            } else {
                _h2_bs = cr_h[sec][1];
                _h1_bs = cr_g[sec][1];
            }
        } else {
            _h2_bs = cr_d[sec][2];
            _h1_bs = cr_c[sec][2];
            _yi2 = in[tx - 1][58 + ty];  // N_BLOCKS - 4 - 2 + ty
        }
        _zi2 = _yi2;
        _zi1 = tmp;
        in_reg[0] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[0]));
        in[tx][0 + ty] = in_reg[0];

        // n = 1
        _h2_bs = cr_d[sec][2];
        _h1_bs = cr_c[sec][2];
        _zi2 = _xc;
        _zi1 = in_reg[1];
        in_reg[2] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[2]));
        in[tx][4 + ty] = in_reg[2];

        // n = 2
        _zi2 = _zi1;
        _zi1 = in_reg[3];
        in_reg[4] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[4]));
        in[tx][8 + ty] = in_reg[4];

        // n = 3
        _zi2 = _zi1;
        _zi1 = in_reg[5];
        in_reg[6] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[6]));
        in[tx][12 + ty] = in_reg[6];

        // n = 4
        _zi2 = _zi1;
        _zi1 = in_reg[7];
        in_reg[8] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[8]));
        in[tx][16 + ty] = in_reg[8];

        // n = 5
        _zi2 = _zi1;
        _zi1 = in_reg[9];
        in_reg[10] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[10]));
        in[tx][20 + ty] = in_reg[10];

        // n = 6
        _zi2 = _zi1;
        _zi1 = in_reg[11];
        in_reg[12] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[12]));
        in[tx][24 + ty] = in_reg[12];

        // n = 7
        _zi2 = _zi1;
        _zi1 = in_reg[13];
        in_reg[14] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[14]));
        in[tx][28 + ty] = in_reg[14];

        // n = 8
        _zi2 = _zi1;
        _zi1 = in_reg[15];
        in_reg[16] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[16]));
        in[tx][32 + ty] = in_reg[16];

        // n = 9
        _zi2 = _zi1;
        _zi1 = in_reg[17];
        in_reg[18] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[18]));
        in[tx][36 + ty] = in_reg[18];

        // n = 10
        _zi2 = _zi1;
        _zi1 = in_reg[19];
        in_reg[20] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[20]));
        in[tx][40 + ty] = in_reg[20];

        // n = 11
        _zi2 = _zi1;
        _zi1 = in_reg[21];
        in_reg[22] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[22]));
        in[tx][44 + ty] = in_reg[22];

        // n = 12
        _zi2 = _zi1;
        _zi1 = in_reg[23];
        in_reg[24] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[24]));
        in[tx][48 + ty] = in_reg[24];

        // n = 13
        _zi2 = _zi1;
        _zi1 = in_reg[25];
        in_reg[26] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[26]));
        in[tx][52 + ty] = in_reg[26];

        // n = 14
        _zi2 = _zi1;
        _zi1 = in_reg[27];
        in_reg[28] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[28]));
        in[tx][56 + ty] = in_reg[28];

        // n = 15
        _zi2 = _zi1;
        _zi1 = in_reg[29];
        in_reg[30] = fmaf(-_h1_bs, _zi1, fmaf(-_h2_bs, _zi2, in_reg[30]));
        in[tx][60 + ty] = in_reg[30];

        __syncthreads();

    } // end section loop

    for (int i = tid; i < CHUNK_SIZE; i += 2 * BLOCK_SIZE) {
        int global_idx = chunk_start + i;
        int ti = i / N_BLOCKS;
        int bi = i % N_BLOCKS;
        output[global_idx] = in[ti][bi];  
    }
}
#endif  // N_BLOCKS == 64

// ----------------------------------------------------------------------------
// Kernel selection
// ----------------------------------------------------------------------------
#ifdef DTCR_HANDUNROLLED
    #if BLOCK_SIZE == 32 && N_BLOCKS == 32
        #define KERNEL_FUNC DTCR_32_UNROLL_32
    #elif BLOCK_SIZE == 32 && N_BLOCKS == 64
        #define KERNEL_FUNC DTCR_32_UNROLL_64
    #elif BLOCK_SIZE == 32 && N_BLOCKS == 128
        #define KERNEL_FUNC DTCR_32_UNROLL_128
    #elif BLOCK_SIZE == 64 && N_BLOCKS == 64
        #define KERNEL_FUNC DTCR_64_UNROLL_64
    #else
        #error "DTCR_HANDUNROLLED supports (BLOCK_SIZE, N_BLOCKS) = (32,32), (32,64), (32,128), (64,64) only"
    #endif
#else
    #if BLOCK_SIZE == 32
        #define KERNEL_FUNC DTCR_32_LOOP
    #elif BLOCK_SIZE == 64
        #define KERNEL_FUNC DTCR_64_LOOP
    #else
        #error "DTCR kernels only support BLOCK_SIZE 32 or 64"
    #endif
#endif

#endif // DTCR_KERNELS_CUH
