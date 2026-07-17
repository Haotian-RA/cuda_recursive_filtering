// plr_kernels.cuh — PLR direct-form kernels (Maleki & Burtscher, ASPLOS'18).
//
// Literature-faithful lane: the whole Butterworth filter is realized as ONE
// direct-form recurrence of order 2*N_SECTIONS (the paper's only route for
// multi-stage filters: offline z-transform combination), computed by the
// hierarchical merge algorithm of the paper. Kernels are transplanted from
// the group's prior hand implementation (main_PLR.cu) with two surgical
// changes only:
//   1. launch-versioned decoupled-lookback flags (protocol identical to
//      ph_kernels.cuh: part_flag = 2*launch+1, full_flag = 2*launch+2,
//      versioned chunk ticket; status zeroed ONCE at allocation).
//   2. geometry from gpu_specs + build flags instead of hardcoded constants.
//
// Exports to the drivers:
//   KERNEL_FUNC        — PLR_2 / PLR_4 / PLR_8 / PLR_16  (order = 2*N_SECTIONS)
//   KERNEL_TB_DIM      — PLR_BLOCK_SIZE (per-GPU, per-order table below)
//   KERNEL_GRID_DIM    — PLR_N_TB (one thread block per chunk)
//   KERNEL_NUM_CHUNKS  — PLR_N_TB (sizes the status/carry buffers)
//   setup_kernel_coefficients(sos) — ignores the SOS argument; consumes the
//     direct-form taps PLR_FILTER_B / PLR_FILTER_A emitted by ref_generate.py
//     and precomputes the n-nacci correction-factor arrays.
//
// Geometry:
//   PLR_BLOCK_SIZE — halves per order doubling so the per-thread register
//     budget (65536 / (PLR_N_TB_PER_SM * PLR_BLOCK_SIZE)) grows with the
//     order-k carry footprint. Locked table (per-GPU, per-order):
//        GTX 1070 : 1024 / 512 / 256 / 128   at 2 thread blocks per SM
//        RTX 3060 :  512 / 256 / 128 /  64   at 3 thread blocks per SM
//   PLR_X (values per thread) — effective x = min(adaptive, PLR_X_MAX):
//     the adaptive term N_SAMPLES / (PLR_SM_P2 * PLR_BLOCK_SIZE) shrinks
//     chunks at small batches so every SM receives work (the paper's sizing
//     rule); the ceiling PLR_X_MAX governs the saturated plateau and is
//     calibrated per (GPU x order) by the ascending spill search in the run
//     scripts ('auto'), or forced with -DPLR_X_MAX=<n>.

#ifndef PLR_KERNELS_CUH
#define PLR_KERNELS_CUH

#include "iir_utils.hpp"
#include "gpu_specs.hpp"
#include "filter_taps.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <array>
#include <cassert>
#include <cstdlib>

// Direct-form order: the full transfer function as one recurrence.
#define PLR_ORDER (2 * N_SECTIONS)

// Values-per-thread ceiling (see header comment). Overridden by the run
// scripts via -DPLR_X_MAX=<n> ('auto' calibration or forced integer).
#ifndef PLR_X_MAX
#define PLR_X_MAX 10
#endif

// ---- per-GPU, per-order launch geometry (locked table) -------------------
#ifdef GPU_GTX1070
    #if PLR_ORDER == 2
        #define PLR_BLOCK_SIZE 1024
    #elif PLR_ORDER == 4
        #define PLR_BLOCK_SIZE 512
    #elif PLR_ORDER == 8
        #define PLR_BLOCK_SIZE 256
    #elif PLR_ORDER == 16
        #define PLR_BLOCK_SIZE 128
    #else
        #error "PLR supports N_SECTIONS in {1, 2, 4, 8}"
    #endif
    #define PLR_TB_TARGET 2
#else
    #if PLR_ORDER == 2
        #define PLR_BLOCK_SIZE 512
    #elif PLR_ORDER == 4
        #define PLR_BLOCK_SIZE 256
    #elif PLR_ORDER == 8
        #define PLR_BLOCK_SIZE 128
    #elif PLR_ORDER == 16
        #define PLR_BLOCK_SIZE 64
    #else
        #error "PLR supports N_SECTIONS in {1, 2, 4, 8}"
    #endif
    #define PLR_TB_TARGET 3
#endif

// The drivers size the carry buffers as num_chunks * order * N_SECTIONS;
// with order = PLR_ORDER this over-allocates by N_SECTIONS (harmless) and
// always covers the kernels' [chunk * order + i] indexing.
static const int order = PLR_ORDER;
static const int warp_size = 32;

constexpr int plr_next_pow2(int v) { int p = 1; while (p < v) p <<= 1; return p; }

// Adaptive chunk sizing (paper rule, "closest power of 2" of the SM count so
// chunk counts and x stay exact for power-of-two batches): 15 -> 16, 28 -> 32.
constexpr int PLR_SM_P2 = plr_next_pow2(gpu_specs::NUM_SMS);
constexpr int PLR_X_RAW = N_SAMPLES / (PLR_SM_P2 * PLR_BLOCK_SIZE);
constexpr int PLR_X     = (PLR_X_RAW < 1) ? 1
                        : ((PLR_X_RAW > PLR_X_MAX) ? PLR_X_MAX : PLR_X_RAW);
constexpr int PLR_CHUNK_SIZE = PLR_X * PLR_BLOCK_SIZE;
constexpr int PLR_N_TB  = (N_SAMPLES + PLR_CHUNK_SIZE - 1) / PLR_CHUNK_SIZE;

// __launch_bounds__ residency target, capped by the thread budget.
constexpr int PLR_N_TB_PER_SM_CAP = gpu_specs::MAX_THREADS_PER_SM / PLR_BLOCK_SIZE;
constexpr int PLR_N_TB_PER_SM = (PLR_TB_TARGET < PLR_N_TB_PER_SM_CAP)
                              ? PLR_TB_TARGET : PLR_N_TB_PER_SM_CAP;

static_assert(PLR_BLOCK_SIZE >= 2 * warp_size,
              "PLR kernels need at least two warps per thread block");
static_assert(PLR_ORDER <= warp_size,
              "carry handling assumes order <= warp size");
static_assert(PLR_X >= 1 && PLR_X <= PLR_X_MAX, "PLR_X out of range");

// ---- device state ---------------------------------------------------------
static __device__ unsigned int counter = 0;

// n-nacci correction-factor arrays (one per carry of the recurrence). Only
// the first PLR_ORDER arrays are filled/used; the rest cost idle global
// memory. The merge tree reads the first PLR_BLOCK_SIZE entries (cached in
// shared memory); the inter-chunk carry correction reads up to PLR_CHUNK_SIZE.
static __device__ T facA[PLR_CHUNK_SIZE], facB[PLR_CHUNK_SIZE], facC[PLR_CHUNK_SIZE], facD[PLR_CHUNK_SIZE];
static __device__ T facE[PLR_CHUNK_SIZE], facF[PLR_CHUNK_SIZE], facG[PLR_CHUNK_SIZE], facH[PLR_CHUNK_SIZE];
static __device__ T facI[PLR_CHUNK_SIZE], facJ[PLR_CHUNK_SIZE], facK[PLR_CHUNK_SIZE], facL[PLR_CHUNK_SIZE];
static __device__ T facM[PLR_CHUNK_SIZE], facN[PLR_CHUNK_SIZE], facO[PLR_CHUNK_SIZE], facP[PLR_CHUNK_SIZE];

// Direct-form coefficients (b = FIR taps past the implicit b0 == 1, a = the
// feedback taps with the recurrence sign convention y[n] += a_j * y[n-j]).
static __constant__ T b1, b2, b3, b4, b5, b6, b7, b8;
static __constant__ T b9, b10, b11, b12, b13, b14, b15, b16;
static __constant__ T a1, a2, a3, a4, a5, a6, a7, a8;
static __constant__ T a9, a10, a11, a12, a13, a14, a15, a16;

// Zero initial conditions (input and output history before sample 0).
static __constant__ const T xi1 = 0, xi2 = 0, xi3 = 0, xi4 = 0;
static __constant__ const T xi5 = 0, xi6 = 0, xi7 = 0, xi8 = 0;
static __constant__ const T xi9 = 0, xi10 = 0, xi11 = 0, xi12 = 0;
static __constant__ const T xi13 = 0, xi14 = 0, xi15 = 0, xi16 = 0;

static __constant__ const T yi1 = 0, yi2 = 0, yi3 = 0, yi4 = 0;
static __constant__ const T yi5 = 0, yi6 = 0, yi7 = 0, yi8 = 0;
static __constant__ const T yi9 = 0, yi10 = 0, yi11 = 0, yi12 = 0;
static __constant__ const T yi13 = 0, yi14 = 0, yi15 = 0, yi16 = 0;


// ===========================================================================
// Host-side n-nacci correction-factor generation (transplanted verbatim).
// ===========================================================================

void iterative_doubling_factor_order2(T a1, T a2, T* facA, T* facB, int chunk_size) {
    std::vector<T> vfacA(chunk_size + 2, 0.0f);
    std::vector<T> vfacB(chunk_size + 2, 0.0f);
    
    vfacA[0] = 1.0f; vfacB[0] = 0.0f; vfacA[1] = 0.0f; vfacB[1] = 1.0f;
    
    for (int i = 2; i < chunk_size + 2; i++) {
        vfacA[i] = a1 * vfacA[i-1] + a2 * vfacA[i-2];
        vfacB[i] = a1 * vfacB[i-1] + a2 * vfacB[i-2];
    }
    
    for (int i = 0; i < chunk_size; i++) {
        facA[i] = vfacA[i + 2];
        facB[i] = vfacB[i + 2];
    }
}

void iterative_doubling_factor_order4(T a1, T a2, T a3, T a4, 
                                      T* facA, T* facB, T* facC, T* facD, 
                                      int chunk_size) {

    std::vector<T> vfacA(chunk_size + 4, 0.0f);
    std::vector<T> vfacB(chunk_size + 4, 0.0f);
    std::vector<T> vfacC(chunk_size + 4, 0.0f);
    std::vector<T> vfacD(chunk_size + 4, 0.0f);
    
    vfacA[0] = 1.0f;  vfacB[0] = 0.0f;  vfacC[0] = 0.0f;  vfacD[0] = 0.0f;  
    vfacA[1] = 0.0f;  vfacB[1] = 1.0f;  vfacC[1] = 0.0f;  vfacD[1] = 0.0f;  
    vfacA[2] = 0.0f;  vfacB[2] = 0.0f;  vfacC[2] = 1.0f;  vfacD[2] = 0.0f;  
    vfacA[3] = 0.0f;  vfacB[3] = 0.0f;  vfacC[3] = 0.0f;  vfacD[3] = 1.0f; 
    
    for (int i = 4; i < chunk_size + 4; i++) {
        vfacA[i] = a1 * vfacA[i-1] + a2 * vfacA[i-2] + a3 * vfacA[i-3] + a4 * vfacA[i-4];
        vfacB[i] = a1 * vfacB[i-1] + a2 * vfacB[i-2] + a3 * vfacB[i-3] + a4 * vfacB[i-4];
        vfacC[i] = a1 * vfacC[i-1] + a2 * vfacC[i-2] + a3 * vfacC[i-3] + a4 * vfacC[i-4];
        vfacD[i] = a1 * vfacD[i-1] + a2 * vfacD[i-2] + a3 * vfacD[i-3] + a4 * vfacD[i-4];
    }
    
    for (int i = 0; i < chunk_size; i++) {
        facA[i] = vfacA[i + 4];
        facB[i] = vfacB[i + 4];
        facC[i] = vfacC[i + 4];
        facD[i] = vfacD[i + 4];
    }
}

void iterative_doubling_factor_order8(T a1, T a2, T a3, T a4, T a5, T a6, T a7, T a8,
                                      T* facA, T* facB, T* facC, T* facD,
                                      T* facE, T* facF, T* facG, T* facH,
                                      int chunk_size) {

    std::vector<T> vfacA(chunk_size + 8, 0.0f);
    std::vector<T> vfacB(chunk_size + 8, 0.0f);
    std::vector<T> vfacC(chunk_size + 8, 0.0f);
    std::vector<T> vfacD(chunk_size + 8, 0.0f);
    std::vector<T> vfacE(chunk_size + 8, 0.0f);
    std::vector<T> vfacF(chunk_size + 8, 0.0f);
    std::vector<T> vfacG(chunk_size + 8, 0.0f);
    std::vector<T> vfacH(chunk_size + 8, 0.0f);
    
    // Initial conditions: identity matrix pattern
    vfacA[0] = 1.0f;  vfacB[0] = 0.0f;  vfacC[0] = 0.0f;  vfacD[0] = 0.0f;
    vfacE[0] = 0.0f;  vfacF[0] = 0.0f;  vfacG[0] = 0.0f;  vfacH[0] = 0.0f;
    
    vfacA[1] = 0.0f;  vfacB[1] = 1.0f;  vfacC[1] = 0.0f;  vfacD[1] = 0.0f;
    vfacE[1] = 0.0f;  vfacF[1] = 0.0f;  vfacG[1] = 0.0f;  vfacH[1] = 0.0f;
    
    vfacA[2] = 0.0f;  vfacB[2] = 0.0f;  vfacC[2] = 1.0f;  vfacD[2] = 0.0f;
    vfacE[2] = 0.0f;  vfacF[2] = 0.0f;  vfacG[2] = 0.0f;  vfacH[2] = 0.0f;
    
    vfacA[3] = 0.0f;  vfacB[3] = 0.0f;  vfacC[3] = 0.0f;  vfacD[3] = 1.0f;
    vfacE[3] = 0.0f;  vfacF[3] = 0.0f;  vfacG[3] = 0.0f;  vfacH[3] = 0.0f;
    
    vfacA[4] = 0.0f;  vfacB[4] = 0.0f;  vfacC[4] = 0.0f;  vfacD[4] = 0.0f;
    vfacE[4] = 1.0f;  vfacF[4] = 0.0f;  vfacG[4] = 0.0f;  vfacH[4] = 0.0f;
    
    vfacA[5] = 0.0f;  vfacB[5] = 0.0f;  vfacC[5] = 0.0f;  vfacD[5] = 0.0f;
    vfacE[5] = 0.0f;  vfacF[5] = 1.0f;  vfacG[5] = 0.0f;  vfacH[5] = 0.0f;
    
    vfacA[6] = 0.0f;  vfacB[6] = 0.0f;  vfacC[6] = 0.0f;  vfacD[6] = 0.0f;
    vfacE[6] = 0.0f;  vfacF[6] = 0.0f;  vfacG[6] = 1.0f;  vfacH[6] = 0.0f;
    
    vfacA[7] = 0.0f;  vfacB[7] = 0.0f;  vfacC[7] = 0.0f;  vfacD[7] = 0.0f;
    vfacE[7] = 0.0f;  vfacF[7] = 0.0f;  vfacG[7] = 0.0f;  vfacH[7] = 1.0f;
    
    for (int i = 8; i < chunk_size + 8; i++) {
        vfacA[i] = a1 * vfacA[i-1] + a2 * vfacA[i-2] + a3 * vfacA[i-3] + a4 * vfacA[i-4]
                 + a5 * vfacA[i-5] + a6 * vfacA[i-6] + a7 * vfacA[i-7] + a8 * vfacA[i-8];
        vfacB[i] = a1 * vfacB[i-1] + a2 * vfacB[i-2] + a3 * vfacB[i-3] + a4 * vfacB[i-4]
                 + a5 * vfacB[i-5] + a6 * vfacB[i-6] + a7 * vfacB[i-7] + a8 * vfacB[i-8];
        vfacC[i] = a1 * vfacC[i-1] + a2 * vfacC[i-2] + a3 * vfacC[i-3] + a4 * vfacC[i-4]
                 + a5 * vfacC[i-5] + a6 * vfacC[i-6] + a7 * vfacC[i-7] + a8 * vfacC[i-8];
        vfacD[i] = a1 * vfacD[i-1] + a2 * vfacD[i-2] + a3 * vfacD[i-3] + a4 * vfacD[i-4]
                 + a5 * vfacD[i-5] + a6 * vfacD[i-6] + a7 * vfacD[i-7] + a8 * vfacD[i-8];
        vfacE[i] = a1 * vfacE[i-1] + a2 * vfacE[i-2] + a3 * vfacE[i-3] + a4 * vfacE[i-4]
                 + a5 * vfacE[i-5] + a6 * vfacE[i-6] + a7 * vfacE[i-7] + a8 * vfacE[i-8];
        vfacF[i] = a1 * vfacF[i-1] + a2 * vfacF[i-2] + a3 * vfacF[i-3] + a4 * vfacF[i-4]
                 + a5 * vfacF[i-5] + a6 * vfacF[i-6] + a7 * vfacF[i-7] + a8 * vfacF[i-8];
        vfacG[i] = a1 * vfacG[i-1] + a2 * vfacG[i-2] + a3 * vfacG[i-3] + a4 * vfacG[i-4]
                 + a5 * vfacG[i-5] + a6 * vfacG[i-6] + a7 * vfacG[i-7] + a8 * vfacG[i-8];
        vfacH[i] = a1 * vfacH[i-1] + a2 * vfacH[i-2] + a3 * vfacH[i-3] + a4 * vfacH[i-4]
                 + a5 * vfacH[i-5] + a6 * vfacH[i-6] + a7 * vfacH[i-7] + a8 * vfacH[i-8];
    }
    
    for (int i = 0; i < chunk_size; i++) {
        facA[i] = vfacA[i + 8];
        facB[i] = vfacB[i + 8];
        facC[i] = vfacC[i + 8];
        facD[i] = vfacD[i + 8];
        facE[i] = vfacE[i + 8];
        facF[i] = vfacF[i + 8];
        facG[i] = vfacG[i + 8];
        facH[i] = vfacH[i + 8];
    }
}

void iterative_doubling_factor_order16(T a1, T a2, T a3, T a4, T a5, T a6, T a7, T a8,
                                       T a9, T a10, T a11, T a12, T a13, T a14, T a15, T a16,
                                       T* facA, T* facB, T* facC, T* facD,
                                       T* facE, T* facF, T* facG, T* facH,
                                       T* facI, T* facJ, T* facK, T* facL,
                                       T* facM, T* facN, T* facO, T* facP,
                                       int chunk_size) {

    std::vector<T> vfacA(chunk_size + 16, 0.0f);
    std::vector<T> vfacB(chunk_size + 16, 0.0f);
    std::vector<T> vfacC(chunk_size + 16, 0.0f);
    std::vector<T> vfacD(chunk_size + 16, 0.0f);
    std::vector<T> vfacE(chunk_size + 16, 0.0f);
    std::vector<T> vfacF(chunk_size + 16, 0.0f);
    std::vector<T> vfacG(chunk_size + 16, 0.0f);
    std::vector<T> vfacH(chunk_size + 16, 0.0f);
    std::vector<T> vfacI(chunk_size + 16, 0.0f);
    std::vector<T> vfacJ(chunk_size + 16, 0.0f);
    std::vector<T> vfacK(chunk_size + 16, 0.0f);
    std::vector<T> vfacL(chunk_size + 16, 0.0f);
    std::vector<T> vfacM(chunk_size + 16, 0.0f);
    std::vector<T> vfacN(chunk_size + 16, 0.0f);
    std::vector<T> vfacO(chunk_size + 16, 0.0f);
    std::vector<T> vfacP(chunk_size + 16, 0.0f);
    
    // Initial conditions: identity matrix pattern
    // Row 0: [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    vfacA[0] = 1.0f;
    // Row 1: [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    vfacB[1] = 1.0f;
    // Row 2: [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]
    vfacC[2] = 1.0f;
    // Row 3: [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]
    vfacD[3] = 1.0f;
    // Row 4: [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]
    vfacE[4] = 1.0f;
    // Row 5: [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]
    vfacF[5] = 1.0f;
    // Row 6: [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]
    vfacG[6] = 1.0f;
    // Row 7: [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]
    vfacH[7] = 1.0f;
    // Row 8: [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]
    vfacI[8] = 1.0f;
    // Row 9: [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]
    vfacJ[9] = 1.0f;
    // Row 10: [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]
    vfacK[10] = 1.0f;
    // Row 11: [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]
    vfacL[11] = 1.0f;
    // Row 12: [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]
    vfacM[12] = 1.0f;
    // Row 13: [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]
    vfacN[13] = 1.0f;
    // Row 14: [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]
    vfacO[14] = 1.0f;
    // Row 15: [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
    vfacP[15] = 1.0f;
    
    for (int i = 16; i < chunk_size + 16; i++) {
        vfacA[i] = a1 * vfacA[i-1] + a2 * vfacA[i-2] + a3 * vfacA[i-3] + a4 * vfacA[i-4]
                 + a5 * vfacA[i-5] + a6 * vfacA[i-6] + a7 * vfacA[i-7] + a8 * vfacA[i-8]
                 + a9 * vfacA[i-9] + a10 * vfacA[i-10] + a11 * vfacA[i-11] + a12 * vfacA[i-12]
                 + a13 * vfacA[i-13] + a14 * vfacA[i-14] + a15 * vfacA[i-15] + a16 * vfacA[i-16];
        vfacB[i] = a1 * vfacB[i-1] + a2 * vfacB[i-2] + a3 * vfacB[i-3] + a4 * vfacB[i-4]
                 + a5 * vfacB[i-5] + a6 * vfacB[i-6] + a7 * vfacB[i-7] + a8 * vfacB[i-8]
                 + a9 * vfacB[i-9] + a10 * vfacB[i-10] + a11 * vfacB[i-11] + a12 * vfacB[i-12]
                 + a13 * vfacB[i-13] + a14 * vfacB[i-14] + a15 * vfacB[i-15] + a16 * vfacB[i-16];
        vfacC[i] = a1 * vfacC[i-1] + a2 * vfacC[i-2] + a3 * vfacC[i-3] + a4 * vfacC[i-4]
                 + a5 * vfacC[i-5] + a6 * vfacC[i-6] + a7 * vfacC[i-7] + a8 * vfacC[i-8]
                 + a9 * vfacC[i-9] + a10 * vfacC[i-10] + a11 * vfacC[i-11] + a12 * vfacC[i-12]
                 + a13 * vfacC[i-13] + a14 * vfacC[i-14] + a15 * vfacC[i-15] + a16 * vfacC[i-16];
        vfacD[i] = a1 * vfacD[i-1] + a2 * vfacD[i-2] + a3 * vfacD[i-3] + a4 * vfacD[i-4]
                 + a5 * vfacD[i-5] + a6 * vfacD[i-6] + a7 * vfacD[i-7] + a8 * vfacD[i-8]
                 + a9 * vfacD[i-9] + a10 * vfacD[i-10] + a11 * vfacD[i-11] + a12 * vfacD[i-12]
                 + a13 * vfacD[i-13] + a14 * vfacD[i-14] + a15 * vfacD[i-15] + a16 * vfacD[i-16];
        vfacE[i] = a1 * vfacE[i-1] + a2 * vfacE[i-2] + a3 * vfacE[i-3] + a4 * vfacE[i-4]
                 + a5 * vfacE[i-5] + a6 * vfacE[i-6] + a7 * vfacE[i-7] + a8 * vfacE[i-8]
                 + a9 * vfacE[i-9] + a10 * vfacE[i-10] + a11 * vfacE[i-11] + a12 * vfacE[i-12]
                 + a13 * vfacE[i-13] + a14 * vfacE[i-14] + a15 * vfacE[i-15] + a16 * vfacE[i-16];
        vfacF[i] = a1 * vfacF[i-1] + a2 * vfacF[i-2] + a3 * vfacF[i-3] + a4 * vfacF[i-4]
                 + a5 * vfacF[i-5] + a6 * vfacF[i-6] + a7 * vfacF[i-7] + a8 * vfacF[i-8]
                 + a9 * vfacF[i-9] + a10 * vfacF[i-10] + a11 * vfacF[i-11] + a12 * vfacF[i-12]
                 + a13 * vfacF[i-13] + a14 * vfacF[i-14] + a15 * vfacF[i-15] + a16 * vfacF[i-16];
        vfacG[i] = a1 * vfacG[i-1] + a2 * vfacG[i-2] + a3 * vfacG[i-3] + a4 * vfacG[i-4]
                 + a5 * vfacG[i-5] + a6 * vfacG[i-6] + a7 * vfacG[i-7] + a8 * vfacG[i-8]
                 + a9 * vfacG[i-9] + a10 * vfacG[i-10] + a11 * vfacG[i-11] + a12 * vfacG[i-12]
                 + a13 * vfacG[i-13] + a14 * vfacG[i-14] + a15 * vfacG[i-15] + a16 * vfacG[i-16];
        vfacH[i] = a1 * vfacH[i-1] + a2 * vfacH[i-2] + a3 * vfacH[i-3] + a4 * vfacH[i-4]
                 + a5 * vfacH[i-5] + a6 * vfacH[i-6] + a7 * vfacH[i-7] + a8 * vfacH[i-8]
                 + a9 * vfacH[i-9] + a10 * vfacH[i-10] + a11 * vfacH[i-11] + a12 * vfacH[i-12]
                 + a13 * vfacH[i-13] + a14 * vfacH[i-14] + a15 * vfacH[i-15] + a16 * vfacH[i-16];
        vfacI[i] = a1 * vfacI[i-1] + a2 * vfacI[i-2] + a3 * vfacI[i-3] + a4 * vfacI[i-4]
                 + a5 * vfacI[i-5] + a6 * vfacI[i-6] + a7 * vfacI[i-7] + a8 * vfacI[i-8]
                 + a9 * vfacI[i-9] + a10 * vfacI[i-10] + a11 * vfacI[i-11] + a12 * vfacI[i-12]
                 + a13 * vfacI[i-13] + a14 * vfacI[i-14] + a15 * vfacI[i-15] + a16 * vfacI[i-16];
        vfacJ[i] = a1 * vfacJ[i-1] + a2 * vfacJ[i-2] + a3 * vfacJ[i-3] + a4 * vfacJ[i-4]
                 + a5 * vfacJ[i-5] + a6 * vfacJ[i-6] + a7 * vfacJ[i-7] + a8 * vfacJ[i-8]
                 + a9 * vfacJ[i-9] + a10 * vfacJ[i-10] + a11 * vfacJ[i-11] + a12 * vfacJ[i-12]
                 + a13 * vfacJ[i-13] + a14 * vfacJ[i-14] + a15 * vfacJ[i-15] + a16 * vfacJ[i-16];
        vfacK[i] = a1 * vfacK[i-1] + a2 * vfacK[i-2] + a3 * vfacK[i-3] + a4 * vfacK[i-4]
                 + a5 * vfacK[i-5] + a6 * vfacK[i-6] + a7 * vfacK[i-7] + a8 * vfacK[i-8]
                 + a9 * vfacK[i-9] + a10 * vfacK[i-10] + a11 * vfacK[i-11] + a12 * vfacK[i-12]
                 + a13 * vfacK[i-13] + a14 * vfacK[i-14] + a15 * vfacK[i-15] + a16 * vfacK[i-16];
        vfacL[i] = a1 * vfacL[i-1] + a2 * vfacL[i-2] + a3 * vfacL[i-3] + a4 * vfacL[i-4]
                 + a5 * vfacL[i-5] + a6 * vfacL[i-6] + a7 * vfacL[i-7] + a8 * vfacL[i-8]
                 + a9 * vfacL[i-9] + a10 * vfacL[i-10] + a11 * vfacL[i-11] + a12 * vfacL[i-12]
                 + a13 * vfacL[i-13] + a14 * vfacL[i-14] + a15 * vfacL[i-15] + a16 * vfacL[i-16];
        vfacM[i] = a1 * vfacM[i-1] + a2 * vfacM[i-2] + a3 * vfacM[i-3] + a4 * vfacM[i-4]
                 + a5 * vfacM[i-5] + a6 * vfacM[i-6] + a7 * vfacM[i-7] + a8 * vfacM[i-8]
                 + a9 * vfacM[i-9] + a10 * vfacM[i-10] + a11 * vfacM[i-11] + a12 * vfacM[i-12]
                 + a13 * vfacM[i-13] + a14 * vfacM[i-14] + a15 * vfacM[i-15] + a16 * vfacM[i-16];
        vfacN[i] = a1 * vfacN[i-1] + a2 * vfacN[i-2] + a3 * vfacN[i-3] + a4 * vfacN[i-4]
                 + a5 * vfacN[i-5] + a6 * vfacN[i-6] + a7 * vfacN[i-7] + a8 * vfacN[i-8]
                 + a9 * vfacN[i-9] + a10 * vfacN[i-10] + a11 * vfacN[i-11] + a12 * vfacN[i-12]
                 + a13 * vfacN[i-13] + a14 * vfacN[i-14] + a15 * vfacN[i-15] + a16 * vfacN[i-16];
        vfacO[i] = a1 * vfacO[i-1] + a2 * vfacO[i-2] + a3 * vfacO[i-3] + a4 * vfacO[i-4]
                 + a5 * vfacO[i-5] + a6 * vfacO[i-6] + a7 * vfacO[i-7] + a8 * vfacO[i-8]
                 + a9 * vfacO[i-9] + a10 * vfacO[i-10] + a11 * vfacO[i-11] + a12 * vfacO[i-12]
                 + a13 * vfacO[i-13] + a14 * vfacO[i-14] + a15 * vfacO[i-15] + a16 * vfacO[i-16];
        vfacP[i] = a1 * vfacP[i-1] + a2 * vfacP[i-2] + a3 * vfacP[i-3] + a4 * vfacP[i-4]
                 + a5 * vfacP[i-5] + a6 * vfacP[i-6] + a7 * vfacP[i-7] + a8 * vfacP[i-8]
                 + a9 * vfacP[i-9] + a10 * vfacP[i-10] + a11 * vfacP[i-11] + a12 * vfacP[i-12]
                 + a13 * vfacP[i-13] + a14 * vfacP[i-14] + a15 * vfacP[i-15] + a16 * vfacP[i-16];
    }
    
    for (int i = 0; i < chunk_size; i++) {
        facA[i] = vfacA[i + 16];
        facB[i] = vfacB[i + 16];
        facC[i] = vfacC[i + 16];
        facD[i] = vfacD[i + 16];
        facE[i] = vfacE[i + 16];
        facF[i] = vfacF[i + 16];
        facG[i] = vfacG[i + 16];
        facH[i] = vfacH[i + 16];
        facI[i] = vfacI[i + 16];
        facJ[i] = vfacJ[i + 16];
        facK[i] = vfacK[i + 16];
        facL[i] = vfacL[i + 16];
        facM[i] = vfacM[i + 16];
        facN[i] = vfacN[i + 16];
        facO[i] = vfacO[i + 16];
        facP[i] = vfacP[i + 16];
    }
}

// ===========================================================================
// Host-side setup — driver-facing interface (same signature as the other
// kernel families). The SOS argument is ignored: PLR consumes the merged
// direct-form taps PLR_FILTER_B / PLR_FILTER_A from filter_taps.hpp.
// ===========================================================================
inline void setup_kernel_coefficients(const std::vector<std::array<T, 6>>& sos) {
    (void)sos;
    static_assert(PLR_FILTER_ORDER == PLR_ORDER,
                  "filter_taps.hpp was generated for a different N_SECTIONS - "
                  "rerun ref_generate.py");

    // float32-cast taps: exactly what the emulator and reference share.
    T hb[PLR_ORDER], ha[PLR_ORDER];
    for (int i = 0; i < PLR_ORDER; i++) {
        hb[i] = (T)PLR_FILTER_B[i];
        ha[i] = (T)PLR_FILTER_A[i];
    }

    T* hf[PLR_ORDER];
    for (int i = 0; i < PLR_ORDER; i++) {
        hf[i] = (T*)malloc(PLR_CHUNK_SIZE * sizeof(T));
        assert(hf[i] != NULL);
    }

#if PLR_ORDER == 2
    iterative_doubling_factor_order2(ha[0], ha[1], hf[0], hf[1], PLR_CHUNK_SIZE);
#elif PLR_ORDER == 4
    iterative_doubling_factor_order4(ha[0], ha[1], ha[2], ha[3],
                                     hf[0], hf[1], hf[2], hf[3], PLR_CHUNK_SIZE);
#elif PLR_ORDER == 8
    iterative_doubling_factor_order8(ha[0], ha[1], ha[2], ha[3], ha[4], ha[5], ha[6], ha[7],
                                     hf[0], hf[1], hf[2], hf[3], hf[4], hf[5], hf[6], hf[7],
                                     PLR_CHUNK_SIZE);
#elif PLR_ORDER == 16
    iterative_doubling_factor_order16(ha[0], ha[1], ha[2],  ha[3],  ha[4],  ha[5],  ha[6],  ha[7],
                                      ha[8], ha[9], ha[10], ha[11], ha[12], ha[13], ha[14], ha[15],
                                      hf[0], hf[1], hf[2],  hf[3],  hf[4],  hf[5],  hf[6],  hf[7],
                                      hf[8], hf[9], hf[10], hf[11], hf[12], hf[13], hf[14], hf[15],
                                      PLR_CHUNK_SIZE);
#endif

    // Upload the used factor arrays and coefficients.
    static const void* fac_syms[16] = {
        (const void*)&facA, (const void*)&facB, (const void*)&facC, (const void*)&facD,
        (const void*)&facE, (const void*)&facF, (const void*)&facG, (const void*)&facH,
        (const void*)&facI, (const void*)&facJ, (const void*)&facK, (const void*)&facL,
        (const void*)&facM, (const void*)&facN, (const void*)&facO, (const void*)&facP };
    static const void* b_syms[16] = {
        (const void*)&b1,  (const void*)&b2,  (const void*)&b3,  (const void*)&b4,
        (const void*)&b5,  (const void*)&b6,  (const void*)&b7,  (const void*)&b8,
        (const void*)&b9,  (const void*)&b10, (const void*)&b11, (const void*)&b12,
        (const void*)&b13, (const void*)&b14, (const void*)&b15, (const void*)&b16 };
    static const void* a_syms[16] = {
        (const void*)&a1,  (const void*)&a2,  (const void*)&a3,  (const void*)&a4,
        (const void*)&a5,  (const void*)&a6,  (const void*)&a7,  (const void*)&a8,
        (const void*)&a9,  (const void*)&a10, (const void*)&a11, (const void*)&a12,
        (const void*)&a13, (const void*)&a14, (const void*)&a15, (const void*)&a16 };

    for (int i = 0; i < PLR_ORDER; i++) {
        assert(cudaSuccess == cudaMemcpyToSymbol(fac_syms[i], hf[i], PLR_CHUNK_SIZE * sizeof(T)));
        assert(cudaSuccess == cudaMemcpyToSymbol(b_syms[i], &hb[i], sizeof(T)));
        assert(cudaSuccess == cudaMemcpyToSymbol(a_syms[i], &ha[i], sizeof(T)));
    }

    for (int i = 0; i < PLR_ORDER; i++) free(hf[i]);
}


// ===========================================================================
// Kernels (transplanted from main_PLR.cu; see header comment for the two
// surgical changes).
// ===========================================================================
#if PLR_ORDER == 2
static __global__ __launch_bounds__(PLR_BLOCK_SIZE, PLR_N_TB_PER_SM)
void PLR_2(const T* const __restrict__ input, 
                       T* const __restrict__ output, 
                       volatile int* const __restrict__ status, 
                       volatile T* const __restrict__ partcarry, 
                       volatile T* const __restrict__ fullcarry,
                       const int launch)
{
    constexpr int num_warps = PLR_BLOCK_SIZE / warp_size;
    constexpr int last_warp = num_warps - 1;
    const int tid = threadIdx.x;
    const int warp = tid / warp_size;
    const int lane = tid % warp_size;

    // Launch-versioned status flags (protocol identical to ph_kernels.cuh):
    // values below part_flag - including everything a previous launch
    // wrote - read as "not ready", so status is zeroed once, never reset.
    const int part_flag = 2 * launch + 1;
    const int full_flag = part_flag + 1;

    __shared__ T spartc[PLR_CHUNK_SIZE / warp_size * order];
    __shared__ T sfullc[order];
    __shared__ int cid;
    __shared__ T sbuf[PLR_BLOCK_SIZE + 2];

    __shared__ T sfacA[PLR_BLOCK_SIZE];
    __shared__ T sfacB[PLR_BLOCK_SIZE];
    sfacA[tid] = facA[tid];
    sfacB[tid] = facB[tid];

    if (tid == 0) {
        cid = (int)(atomicAdd(&counter, 1u) - (unsigned int)launch * gridDim.x);
    }
    __syncthreads();

    const int chunk_id = cid;
    const int offs = tid + chunk_id * PLR_CHUNK_SIZE;

    T val[PLR_X];

    if (chunk_id == gridDim.x - 1) {
        #pragma unroll
        for (int v = 0; v < PLR_X; v++) {
            val[v] = 0;
            if (offs + (v * PLR_BLOCK_SIZE) < N_SAMPLES) {
                val[v] = input[offs + (v * PLR_BLOCK_SIZE)];
            }
        }
    } else {
        #pragma unroll
        for (int v = 0; v < PLR_X; v++) {
            val[v] = input[offs + (v * PLR_BLOCK_SIZE)];
        }
    }

    // phase 1: FIR 
    #pragma unroll
    for (int v = PLR_X - 1; v >= 0; v--) {
        sbuf[tid + 2] = val[v];
        if ((tid + 2 - PLR_BLOCK_SIZE) >= 0) {
            if (v > 0) {
                sbuf[tid + 2 - PLR_BLOCK_SIZE] = val[v - 1];
            } else {
                // First segment - handle initial conditions or previous chunk
                if (chunk_id == 0) {
                    const int idx_init = tid + 2 - PLR_BLOCK_SIZE;
                    sbuf[idx_init] = (idx_init == 0) ? xi2 : xi1;
                } else {
                    sbuf[tid + 2 - PLR_BLOCK_SIZE] = input[offs - PLR_BLOCK_SIZE];
                }
            }
        }
        __syncthreads();
        
        val[v] += b1 * sbuf[tid + 1];
        val[v] += b2 * sbuf[tid + 0];
        
        if (v > 0) __syncthreads();
    }

    // phase 2a: intra-warp iterative doubling
    const T sfA = sfacA[lane];
    const T sfB = sfacB[lane];
    int cond;
    T help, spc;

    help = a1;
    cond = ((lane & 1) != 0);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 0, 2);
        if (cond) val[v] += spc;
    }

    help = __shfl_sync(0xffffffff, sfA, lane % 2);
    cond = ((lane & 2) != 0);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 0, 4);
        if (cond) val[v] += spc;
    }
    help = __shfl_sync(0xffffffff, sfB, lane % 2);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 1, 4);
        if (cond) val[v] += spc;
    }

    help = __shfl_sync(0xffffffff, sfA, lane % 4);
    cond = ((lane & 4) != 0);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 2, 8);
        if (cond) val[v] += spc;
    }
    help = __shfl_sync(0xffffffff, sfB, lane % 4);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 3, 8);
        if (cond) val[v] += spc;
    }

    help = __shfl_sync(0xffffffff, sfA, lane % 8);
    cond = ((lane & 8) != 0);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 6, 16);
        if (cond) val[v] += spc;
    }
    help = __shfl_sync(0xffffffff, sfB, lane % 8);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 7, 16);
        if (cond) val[v] += spc;
    }

    help = __shfl_sync(0xffffffff, sfA, lane % 16);
    cond = ((lane & 16) != 0);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 14, 32);
        if (cond) val[v] += spc;
    }
    help = __shfl_sync(0xffffffff, sfB, lane % 16);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 15, 32);
        if (cond) val[v] += spc;
    }

    // phase 2b: inter-warp iterative doubling using shared memory
    // Number of merge stages depends on number of warps:
    // 8 warps (256 threads): 3 stages (1, 2, 4)
    // 16 warps (512 threads): 4 stages (1, 2, 4, 8)
    // 32 warps (1024 threads): 5 stages (1, 2, 4, 8, 16)

    const int delta = PLR_BLOCK_SIZE / warp_size * order;
    const int clane = lane - (warp_size - order);
    const int clwo = clane + warp * order;

    if (((warp & 1) == 0) && (clane >= 0)) {
        #pragma unroll
        for (int v = 0; v < PLR_X; v++) {
            spartc[clwo + v * delta] = val[v];
        }
    }

    __syncthreads();

    if ((warp & 1) != 0) {
        const int cwarp = ((warp & ~1) | 0) * order;
        const T helpA = sfacA[tid % (warp_size * 1)];
        const T helpB = sfacB[tid % (warp_size * 1)];
        #pragma unroll
        for (int v = 0; v < PLR_X; v++) {
            val[v] += helpA * spartc[cwarp + (v * delta + 0)];
            val[v] += helpB * spartc[cwarp + (v * delta + 1)];
        }
        if constexpr (num_warps > 2) {
            if (((warp & 3) != 0) && (clane >= 0)) {
                #pragma unroll
                for (int v = 0; v < PLR_X; v++) {
                    spartc[clwo + v * delta] = val[v];
                }
            }
        } else {
            if ((warp == last_warp) && (clane >= 0)) {
                #pragma unroll
                for (int v = 0; v < PLR_X; v++) {
                    spartc[clane + (last_warp * order + v * delta)] = val[v];
                }
            }
        }
    }

    if constexpr (num_warps >= 4) {
        __syncthreads();
        if ((warp & 2) != 0) {
            const int cwarp = ((warp & ~3) | 1) * order;
            const T helpA = sfacA[tid % (warp_size * 2)];
            const T helpB = sfacB[tid % (warp_size * 2)];
            #pragma unroll
            for (int v = 0; v < PLR_X; v++) {
                val[v] += helpA * spartc[cwarp + (v * delta + 0)];
                val[v] += helpB * spartc[cwarp + (v * delta + 1)];
            }
            if constexpr (num_warps > 4) {
                if (((warp & 7) != 0) && (clane >= 0)) {
                    #pragma unroll
                    for (int v = 0; v < PLR_X; v++) {
                        spartc[clwo + v * delta] = val[v];
                    }
                }
            } else {
                if ((warp == last_warp) && (clane >= 0)) {
                    #pragma unroll
                    for (int v = 0; v < PLR_X; v++) {
                        spartc[clane + (last_warp * order + v * delta)] = val[v];
                    }
                }
            }
        }
    }

    if constexpr (num_warps >= 8) {
        __syncthreads();
        if ((warp & 4) != 0) {
            const int cwarp = ((warp & ~7) | 3) * order;
            const T helpA = sfacA[tid % (warp_size * 4)];
            const T helpB = sfacB[tid % (warp_size * 4)];
            #pragma unroll
            for (int v = 0; v < PLR_X; v++) {
                val[v] += helpA * spartc[cwarp + (v * delta + 0)];
                val[v] += helpB * spartc[cwarp + (v * delta + 1)];
            }
            if constexpr (num_warps > 8) {
                if (((warp & 15) != 0) && (clane >= 0)) {
                    #pragma unroll
                    for (int v = 0; v < PLR_X; v++) {
                        spartc[clwo + v * delta] = val[v];
                    }
                }
            } else {
                if ((warp == last_warp) && (clane >= 0)) {
                    #pragma unroll
                    for (int v = 0; v < PLR_X; v++) {
                        spartc[clane + (last_warp * order + v * delta)] = val[v];
                    }
                }
            }
        }
    }

    if constexpr (num_warps >= 16) {
        __syncthreads();
        if ((warp & 8) != 0) {
            const int cwarp = ((warp & ~15) | 7) * order;
            const T helpA = sfacA[tid % (warp_size * 8)];
            const T helpB = sfacB[tid % (warp_size * 8)];
            #pragma unroll
            for (int v = 0; v < PLR_X; v++) {
                val[v] += helpA * spartc[cwarp + (v * delta + 0)];
                val[v] += helpB * spartc[cwarp + (v * delta + 1)];
            }
            if constexpr (num_warps > 16) {
                if ((warp == 15) && (clane >= 0)) {
                    #pragma unroll
                    for (int v = 0; v < PLR_X; v++) {
                        spartc[clane + (15 * order + v * delta)] = val[v];
                    }
                }
            } else {
                if ((warp == last_warp) && (clane >= 0)) {
                    #pragma unroll
                    for (int v = 0; v < PLR_X; v++) {
                        spartc[clane + (last_warp * order + v * delta)] = val[v];
                    }
                }
            }
        }
    }

    if constexpr (num_warps == 32) {
        __syncthreads();
        if ((warp & 16) != 0) {
            const int cwarp = 15 * order;
            const T helpA = sfacA[tid % (warp_size * 16)];
            const T helpB = sfacB[tid % (warp_size * 16)];
            #pragma unroll
            for (int v = 0; v < PLR_X; v++) {
                val[v] += helpA * spartc[cwarp + (v * delta + 0)];
                val[v] += helpB * spartc[cwarp + (v * delta + 1)];
            }
            // warp 31 is the last warp
            if ((warp == last_warp) && (clane >= 0)) {
                #pragma unroll
                for (int v = 0; v < PLR_X; v++) {
                    spartc[clane + (last_warp * order + v * delta)] = val[v];
                }
            }
        }
    }

    if constexpr (PLR_X > 1) {
        // Store segment 0's last values (from last warp)
        if ((warp == last_warp) && (clane >= 0)) {
            spartc[clane + (last_warp * order + 0 * delta)] = val[0];
        }
        __syncthreads();
        
        // Propagate through segments sequentially
        #pragma unroll
        for (int v = 1; v < PLR_X; v++) {
            // All threads update segment v based on segment (v-1)
            val[v] += sfacA[tid] * spartc[last_warp * order + ((v-1) * delta + 0)];
            val[v] += sfacB[tid] * spartc[last_warp * order + ((v-1) * delta + 1)];
            
            // Store current segment's result for next iteration
            if ((warp == last_warp) && (clane >= 0)) {
                spartc[clane + (last_warp * order + v * delta)] = val[v];
            }
            
            // Sync before next iteration reads the stored values
            if (v < PLR_X - 1) {
                __syncthreads();
            }
        }
    }

    // phase 3a: inter-block carry propagation (look-back)
    const int idx = tid - (PLR_BLOCK_SIZE - order);
    const int last_val_idx = PLR_X - 1;

    if (idx >= 0) {
        partcarry[chunk_id * order + idx] = val[last_val_idx];
    }
    __syncwarp();
    __threadfence();
    if (idx == 0) {
        status[chunk_id] = part_flag;
    }

    __syncthreads();

    if (warp == 0) {
        int flag = part_flag;
        bool no_zeros, has_status_2, reached_origin;
        
        do {
            if (chunk_id > lane) {
                flag = status[chunk_id - 1 - lane];
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
            X0 = yi2;
            X1 = yi1;
            start_chunk = 0;
        } else {
            const int pos = __ffs(mask) - 1;
            const int full_chunk = chunk_id - 1 - pos;
            start_chunk = full_chunk + 1;
            
            T fc;
            if (lane < order) {
                fc = fullcarry[full_chunk * order + lane];
            }
            X0 = __shfl_sync(0xffffffff, fc, 0);
            X1 = __shfl_sync(0xffffffff, fc, 1);
        }
        
        const int num_partcarries = chunk_id - start_chunk;
        
        if (num_partcarries > 0) {
            for (int i = (start_chunk * order + lane); i < (chunk_id * order); i += warp_size) {
                spartc[i - start_chunk * order] = partcarry[i];
            }
            __syncwarp();
            
            if (lane == 0) {
                for (int i = 0; i < num_partcarries; i++) {
                    const T h0 = spartc[i * order + 0] + X0 * facA[PLR_CHUNK_SIZE-2] + X1 * facB[PLR_CHUNK_SIZE-2];
                    const T h1 = spartc[i * order + 1] + X0 * facA[PLR_CHUNK_SIZE-1] + X1 * facB[PLR_CHUNK_SIZE-1];
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

    // phase 3b: apply carry to all values
    T X0 = sfullc[0];
    T X1 = sfullc[1];

    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        val[v] += facA[tid + v * PLR_BLOCK_SIZE] * X0;
        val[v] += facB[tid + v * PLR_BLOCK_SIZE] * X1;
    }

    if (idx >= 0) {
        fullcarry[chunk_id * order + idx] = val[last_val_idx];
    }
    __syncwarp();
    __threadfence();
    if (idx == 0) {
        status[chunk_id] = full_flag;
    }

    if (chunk_id == gridDim.x - 1) {
        #pragma unroll
        for (int v = 0; v < PLR_X; v++) {
            if (offs + (v * PLR_BLOCK_SIZE) < N_SAMPLES) {
                output[offs + (v * PLR_BLOCK_SIZE)] = val[v];
            }
        }
    } else {
        #pragma unroll
        for (int v = 0; v < PLR_X; v++) {
            output[offs + (v * PLR_BLOCK_SIZE)] = val[v];
        }
    }
}
#endif

#if PLR_ORDER == 4
static __global__ __launch_bounds__(PLR_BLOCK_SIZE, PLR_N_TB_PER_SM)
void PLR_4(const T* const __restrict__ input, 
                       T* const __restrict__ output, 
                       volatile int* const __restrict__ status, 
                       volatile T* const __restrict__ partcarry, 
                       volatile T* const __restrict__ fullcarry,
                       const int launch)
{
    constexpr int num_warps = PLR_BLOCK_SIZE / warp_size;
    constexpr int last_warp = num_warps - 1;
    const int tid = threadIdx.x;
    const int warp = tid / warp_size;
    const int lane = tid % warp_size;

    // Launch-versioned status flags (protocol identical to ph_kernels.cuh):
    // values below part_flag - including everything a previous launch
    // wrote - read as "not ready", so status is zeroed once, never reset.
    const int part_flag = 2 * launch + 1;
    const int full_flag = part_flag + 1;

    __shared__ T spartc[PLR_CHUNK_SIZE / warp_size * order];
    __shared__ T sfullc[order];
    __shared__ int cid;
    __shared__ T sbuf[PLR_BLOCK_SIZE + 4];  // +4 for 4th order

    __shared__ T sfacA[PLR_BLOCK_SIZE];
    __shared__ T sfacB[PLR_BLOCK_SIZE];
    __shared__ T sfacC[PLR_BLOCK_SIZE];
    __shared__ T sfacD[PLR_BLOCK_SIZE];
    sfacA[tid] = facA[tid];
    sfacB[tid] = facB[tid];
    sfacC[tid] = facC[tid];
    sfacD[tid] = facD[tid];

    if (tid == 0) {
        cid = (int)(atomicAdd(&counter, 1u) - (unsigned int)launch * gridDim.x);
    }
    __syncthreads();

    const int chunk_id = cid;
    const int offs = tid + chunk_id * PLR_CHUNK_SIZE;

    T val[PLR_X];

    if (chunk_id == gridDim.x - 1) {
        #pragma unroll
        for (int v = 0; v < PLR_X; v++) {
            val[v] = 0;
            if (offs + (v * PLR_BLOCK_SIZE) < N_SAMPLES) {
                val[v] = input[offs + (v * PLR_BLOCK_SIZE)];
            }
        }
    } else {
        #pragma unroll
        for (int v = 0; v < PLR_X; v++) {
            val[v] = input[offs + (v * PLR_BLOCK_SIZE)];
        }
    }

    // phase 1: FIR 
    #pragma unroll
    for (int v = PLR_X - 1; v >= 0; v--) {
        sbuf[tid + 4] = val[v];
        if ((tid + 4 - PLR_BLOCK_SIZE) >= 0) {
            if (v > 0) {
                sbuf[tid + 4 - PLR_BLOCK_SIZE] = val[v - 1];
            } else {
                // First segment - handle initial conditions or previous chunk
                if (chunk_id == 0) {
                    const int idx_init = tid + 4 - PLR_BLOCK_SIZE;
                    if (idx_init == 0) sbuf[idx_init] = xi4;
                    else if (idx_init == 1) sbuf[idx_init] = xi3;
                    else if (idx_init == 2) sbuf[idx_init] = xi2;
                    else if (idx_init == 3) sbuf[idx_init] = xi1;
                } else {
                    sbuf[tid + 4 - PLR_BLOCK_SIZE] = input[offs - PLR_BLOCK_SIZE];
                }
            }
        }
        __syncthreads();
        
        val[v] += b1 * sbuf[tid + 3];
        val[v] += b2 * sbuf[tid + 2];
        val[v] += b3 * sbuf[tid + 1];
        val[v] += b4 * sbuf[tid + 0];
        
        if (v > 0) __syncthreads();
    }

    // phase 2a: intra-warp iterative doubling 
    const T sfA = sfacA[lane];
    const T sfB = sfacB[lane];
    const T sfC = sfacC[lane];
    const T sfD = sfacD[lane];
    int cond;
    T help, spc;

    help = a1;
    cond = ((lane & 1) != 0);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 0, 2);
        if (cond) val[v] += spc;
    }

    cond = ((lane & 2) != 0);
    help = __shfl_sync(0xffffffff, sfC, lane % 2);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 0, 4);
        if (cond) val[v] += spc;
    }
    help = __shfl_sync(0xffffffff, sfD, lane % 2);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 1, 4);
        if (cond) val[v] += spc;
    }

    cond = ((lane & 4) != 0);
    help = __shfl_sync(0xffffffff, sfA, lane % 4);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 0, 8);
        if (cond) val[v] += spc;
    }
    help = __shfl_sync(0xffffffff, sfB, lane % 4);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 1, 8);
        if (cond) val[v] += spc;
    }
    help = __shfl_sync(0xffffffff, sfC, lane % 4);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 2, 8);
        if (cond) val[v] += spc;
    }
    help = __shfl_sync(0xffffffff, sfD, lane % 4);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 3, 8);
        if (cond) val[v] += spc;
    }

    cond = ((lane & 8) != 0);
    help = __shfl_sync(0xffffffff, sfA, lane % 8);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 4, 16);
        if (cond) val[v] += spc;
    }
    help = __shfl_sync(0xffffffff, sfB, lane % 8);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 5, 16);
        if (cond) val[v] += spc;
    }
    help = __shfl_sync(0xffffffff, sfC, lane % 8);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 6, 16);
        if (cond) val[v] += spc;
    }
    help = __shfl_sync(0xffffffff, sfD, lane % 8);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 7, 16);
        if (cond) val[v] += spc;
    }

    cond = ((lane & 16) != 0);
    help = __shfl_sync(0xffffffff, sfA, lane % 16);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 12, 32);
        if (cond) val[v] += spc;
    }
    help = __shfl_sync(0xffffffff, sfB, lane % 16);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 13, 32);
        if (cond) val[v] += spc;
    }
    help = __shfl_sync(0xffffffff, sfC, lane % 16);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 14, 32);
        if (cond) val[v] += spc;
    }
    help = __shfl_sync(0xffffffff, sfD, lane % 16);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 15, 32);
        if (cond) val[v] += spc;
    }


    // phase 2b: inter-warp iterative doubling
    const int delta = PLR_BLOCK_SIZE / warp_size * order;
    const int clane = lane - (warp_size - order);  // -28 to 3 for order=4
    const int clwo = clane + warp * order;

    if (((warp & 1) == 0) && (clane >= 0)) {
        #pragma unroll
        for (int v = 0; v < PLR_X; v++) {
            spartc[clwo + v * delta] = val[v];
        }
    }

    __syncthreads();

    if ((warp & 1) != 0) {
        const int cwarp = ((warp & ~1) | 0) * order;
        const T helpA = sfacA[tid % (warp_size * 1)];
        const T helpB = sfacB[tid % (warp_size * 1)];
        const T helpC = sfacC[tid % (warp_size * 1)];
        const T helpD = sfacD[tid % (warp_size * 1)];
        #pragma unroll
        for (int v = 0; v < PLR_X; v++) {
            val[v] += helpA * spartc[cwarp + (v * delta + 0)];
            val[v] += helpB * spartc[cwarp + (v * delta + 1)];
            val[v] += helpC * spartc[cwarp + (v * delta + 2)];
            val[v] += helpD * spartc[cwarp + (v * delta + 3)];
        }
        if constexpr (num_warps > 2) {
            if (((warp & 3) != 0) && (clane >= 0)) {
                #pragma unroll
                for (int v = 0; v < PLR_X; v++) {
                    spartc[clwo + v * delta] = val[v];
                }
            }
        } else {
            if ((warp == last_warp) && (clane >= 0)) {
                #pragma unroll
                for (int v = 0; v < PLR_X; v++) {
                    spartc[clane + (last_warp * order + v * delta)] = val[v];
                }
            }
        }
    }

    if constexpr (num_warps >= 4) {
        __syncthreads();
        if ((warp & 2) != 0) {
            const int cwarp = ((warp & ~3) | 1) * order;
            const T helpA = sfacA[tid % (warp_size * 2)];
            const T helpB = sfacB[tid % (warp_size * 2)];
            const T helpC = sfacC[tid % (warp_size * 2)];
            const T helpD = sfacD[tid % (warp_size * 2)];
            #pragma unroll
            for (int v = 0; v < PLR_X; v++) {
                val[v] += helpA * spartc[cwarp + (v * delta + 0)];
                val[v] += helpB * spartc[cwarp + (v * delta + 1)];
                val[v] += helpC * spartc[cwarp + (v * delta + 2)];
                val[v] += helpD * spartc[cwarp + (v * delta + 3)];
            }
            if constexpr (num_warps > 4) {
                if (((warp & 7) != 0) && (clane >= 0)) {
                    #pragma unroll
                    for (int v = 0; v < PLR_X; v++) {
                        spartc[clwo + v * delta] = val[v];
                    }
                }
            } else {
                if ((warp == last_warp) && (clane >= 0)) {
                    #pragma unroll
                    for (int v = 0; v < PLR_X; v++) {
                        spartc[clane + (last_warp * order + v * delta)] = val[v];
                    }
                }
            }
        }
    }

    if constexpr (num_warps >= 8) {
        __syncthreads();
        if ((warp & 4) != 0) {
            const int cwarp = ((warp & ~7) | 3) * order;
            const T helpA = sfacA[tid % (warp_size * 4)];
            const T helpB = sfacB[tid % (warp_size * 4)];
            const T helpC = sfacC[tid % (warp_size * 4)];
            const T helpD = sfacD[tid % (warp_size * 4)];
            #pragma unroll
            for (int v = 0; v < PLR_X; v++) {
                val[v] += helpA * spartc[cwarp + (v * delta + 0)];
                val[v] += helpB * spartc[cwarp + (v * delta + 1)];
                val[v] += helpC * spartc[cwarp + (v * delta + 2)];
                val[v] += helpD * spartc[cwarp + (v * delta + 3)];
            }
            if constexpr (num_warps > 8) {
                if (((warp & 15) != 0) && (clane >= 0)) {
                    #pragma unroll
                    for (int v = 0; v < PLR_X; v++) {
                        spartc[clwo + v * delta] = val[v];
                    }
                }
            } else {
                if ((warp == last_warp) && (clane >= 0)) {
                    #pragma unroll
                    for (int v = 0; v < PLR_X; v++) {
                        spartc[clane + (last_warp * order + v * delta)] = val[v];
                    }
                }
            }
        }
    }

    if constexpr (num_warps >= 16) {
        __syncthreads();
        if ((warp & 8) != 0) {
            const int cwarp = ((warp & ~15) | 7) * order;
            const T helpA = sfacA[tid % (warp_size * 8)];
            const T helpB = sfacB[tid % (warp_size * 8)];
            const T helpC = sfacC[tid % (warp_size * 8)];
            const T helpD = sfacD[tid % (warp_size * 8)];
            #pragma unroll
            for (int v = 0; v < PLR_X; v++) {
                val[v] += helpA * spartc[cwarp + (v * delta + 0)];
                val[v] += helpB * spartc[cwarp + (v * delta + 1)];
                val[v] += helpC * spartc[cwarp + (v * delta + 2)];
                val[v] += helpD * spartc[cwarp + (v * delta + 3)];
            }
            if constexpr (num_warps > 16) {
                if ((warp == 15) && (clane >= 0)) {
                    #pragma unroll
                    for (int v = 0; v < PLR_X; v++) {
                        spartc[clane + (15 * order + v * delta)] = val[v];
                    }
                }
            } else {
                if ((warp == last_warp) && (clane >= 0)) {
                    #pragma unroll
                    for (int v = 0; v < PLR_X; v++) {
                        spartc[clane + (last_warp * order + v * delta)] = val[v];
                    }
                }
            }
        }
    }

    if constexpr (num_warps == 32) {
        __syncthreads();
        if ((warp & 16) != 0) {
            const int cwarp = 15 * order;
            const T helpA = sfacA[tid % (warp_size * 16)];
            const T helpB = sfacB[tid % (warp_size * 16)];
            const T helpC = sfacC[tid % (warp_size * 16)];
            const T helpD = sfacD[tid % (warp_size * 16)];
            #pragma unroll
            for (int v = 0; v < PLR_X; v++) {
                val[v] += helpA * spartc[cwarp + (v * delta + 0)];
                val[v] += helpB * spartc[cwarp + (v * delta + 1)];
                val[v] += helpC * spartc[cwarp + (v * delta + 2)];
                val[v] += helpD * spartc[cwarp + (v * delta + 3)];
            }
            if ((warp == last_warp) && (clane >= 0)) {
                #pragma unroll
                for (int v = 0; v < PLR_X; v++) {
                    spartc[clane + (last_warp * order + v * delta)] = val[v];
                }
            }
        }
    }

    if constexpr (PLR_X > 1) {
        if ((warp == last_warp) && (clane >= 0)) {
            spartc[clane + (last_warp * order + 0 * delta)] = val[0];
        }
        __syncthreads();
        
        #pragma unroll
        for (int v = 1; v < PLR_X; v++) {
            val[v] += sfacA[tid] * spartc[last_warp * order + ((v-1) * delta + 0)];
            val[v] += sfacB[tid] * spartc[last_warp * order + ((v-1) * delta + 1)];
            val[v] += sfacC[tid] * spartc[last_warp * order + ((v-1) * delta + 2)];
            val[v] += sfacD[tid] * spartc[last_warp * order + ((v-1) * delta + 3)];
            
            if ((warp == last_warp) && (clane >= 0)) {
                spartc[clane + (last_warp * order + v * delta)] = val[v];
            }
            
            if (v < PLR_X - 1) {
                __syncthreads();
            }
        }
    }

    // phase 3a: inter-block carry propagation 
    const int idx = tid - (PLR_BLOCK_SIZE - order);
    const int last_val_idx = PLR_X - 1;

    if (idx >= 0) {
        partcarry[chunk_id * order + idx] = val[last_val_idx];
    }
    __syncwarp();
    __threadfence();
    if (idx == 0) {
        status[chunk_id] = part_flag;
    }

    __syncthreads();

    if (warp == 0) {
        int flag = part_flag;
        bool no_zeros, has_status_2, reached_origin;
        
        do {
            if (chunk_id > lane) {
                flag = status[chunk_id - 1 - lane];
            }
            no_zeros = !__any_sync(0xffffffff, flag < part_flag);
            has_status_2 = !__all_sync(0xffffffff, flag != full_flag) && no_zeros;
            reached_origin = (chunk_id < warp_size) && no_zeros;
        } while (!(has_status_2 || reached_origin));
        
        __threadfence();
        
        int mask = __ballot_sync(0xffffffff, flag == full_flag);
        
        T X0, X1, X2, X3;
        int start_chunk;
        
        if (mask == 0) {
            X0 = yi4;
            X1 = yi3;
            X2 = yi2;
            X3 = yi1;
            start_chunk = 0;
        } else {
            const int pos = __ffs(mask) - 1;
            const int full_chunk = chunk_id - 1 - pos;
            start_chunk = full_chunk + 1;
            
            T fc;
            if (lane < order) {
                fc = fullcarry[full_chunk * order + lane];
            }
            X0 = __shfl_sync(0xffffffff, fc, 0);
            X1 = __shfl_sync(0xffffffff, fc, 1);
            X2 = __shfl_sync(0xffffffff, fc, 2);
            X3 = __shfl_sync(0xffffffff, fc, 3);
        }
        
        const int num_partcarries = chunk_id - start_chunk;
        
        if (num_partcarries > 0) {
            for (int i = (start_chunk * order + lane); i < (chunk_id * order); i += warp_size) {
                spartc[i - start_chunk * order] = partcarry[i];
            }
            __syncwarp();
            
            if (lane == 0) {
                for (int i = 0; i < num_partcarries; i++) {
                    const T h0 = spartc[i * order + 0] + X0 * facA[PLR_CHUNK_SIZE-4] + X1 * facB[PLR_CHUNK_SIZE-4] + X2 * facC[PLR_CHUNK_SIZE-4] + X3 * facD[PLR_CHUNK_SIZE-4];
                    const T h1 = spartc[i * order + 1] + X0 * facA[PLR_CHUNK_SIZE-3] + X1 * facB[PLR_CHUNK_SIZE-3] + X2 * facC[PLR_CHUNK_SIZE-3] + X3 * facD[PLR_CHUNK_SIZE-3];
                    const T h2 = spartc[i * order + 2] + X0 * facA[PLR_CHUNK_SIZE-2] + X1 * facB[PLR_CHUNK_SIZE-2] + X2 * facC[PLR_CHUNK_SIZE-2] + X3 * facD[PLR_CHUNK_SIZE-2];
                    const T h3 = spartc[i * order + 3] + X0 * facA[PLR_CHUNK_SIZE-1] + X1 * facB[PLR_CHUNK_SIZE-1] + X2 * facC[PLR_CHUNK_SIZE-1] + X3 * facD[PLR_CHUNK_SIZE-1];
                    X0 = h0;
                    X1 = h1;
                    X2 = h2;
                    X3 = h3;
                }
            }
        }
        
        if (lane == 0) {
            sfullc[0] = X0;
            sfullc[1] = X1;
            sfullc[2] = X2;
            sfullc[3] = X3;
        }
    }

    __syncthreads();

    // phase 3b: apply carry to all values
    T X0 = sfullc[0];
    T X1 = sfullc[1];
    T X2 = sfullc[2];
    T X3 = sfullc[3];

    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        val[v] += facA[tid + v * PLR_BLOCK_SIZE] * X0;
        val[v] += facB[tid + v * PLR_BLOCK_SIZE] * X1;
        val[v] += facC[tid + v * PLR_BLOCK_SIZE] * X2;
        val[v] += facD[tid + v * PLR_BLOCK_SIZE] * X3;
    }

    if (idx >= 0) {
        fullcarry[chunk_id * order + idx] = val[last_val_idx];
    }
    __syncwarp();
    __threadfence();
    if (idx == 0) {
        status[chunk_id] = full_flag;
    }

    if (chunk_id == gridDim.x - 1) {
        #pragma unroll
        for (int v = 0; v < PLR_X; v++) {
            if (offs + (v * PLR_BLOCK_SIZE) < N_SAMPLES) {
                output[offs + (v * PLR_BLOCK_SIZE)] = val[v];
            }
        }
    } else {
        #pragma unroll
        for (int v = 0; v < PLR_X; v++) {
            output[offs + (v * PLR_BLOCK_SIZE)] = val[v];
        }
    }
}
#endif

#if PLR_ORDER == 8
static __global__ __launch_bounds__(PLR_BLOCK_SIZE, PLR_N_TB_PER_SM)
void PLR_8(const T* const __restrict__ input, 
                       T* const __restrict__ output, 
                       volatile int* const __restrict__ status, 
                       volatile T* const __restrict__ partcarry, 
                       volatile T* const __restrict__ fullcarry,
                       const int launch)
{
    constexpr int num_warps = PLR_BLOCK_SIZE / warp_size;
    constexpr int last_warp = num_warps - 1;
    const int tid = threadIdx.x;
    const int warp = tid / warp_size;
    const int lane = tid % warp_size;

    // Launch-versioned status flags (protocol identical to ph_kernels.cuh):
    // values below part_flag - including everything a previous launch
    // wrote - read as "not ready", so status is zeroed once, never reset.
    const int part_flag = 2 * launch + 1;
    const int full_flag = part_flag + 1;
    
    __shared__ T spartc[PLR_CHUNK_SIZE / warp_size * order];
    __shared__ T sfullc[order];
    __shared__ int cid;
    __shared__ T sbuf[PLR_BLOCK_SIZE + 8];

    __shared__ T sfacA[PLR_BLOCK_SIZE];
    __shared__ T sfacB[PLR_BLOCK_SIZE];
    __shared__ T sfacC[PLR_BLOCK_SIZE];
    __shared__ T sfacD[PLR_BLOCK_SIZE];
    __shared__ T sfacE[PLR_BLOCK_SIZE];
    __shared__ T sfacF[PLR_BLOCK_SIZE];
    __shared__ T sfacG[PLR_BLOCK_SIZE];
    __shared__ T sfacH[PLR_BLOCK_SIZE];
    sfacA[tid] = facA[tid];
    sfacB[tid] = facB[tid];
    sfacC[tid] = facC[tid];
    sfacD[tid] = facD[tid];
    sfacE[tid] = facE[tid];
    sfacF[tid] = facF[tid];
    sfacG[tid] = facG[tid];
    sfacH[tid] = facH[tid];

    if (tid == 0) {
        cid = (int)(atomicAdd(&counter, 1u) - (unsigned int)launch * gridDim.x);
    }
    __syncthreads();

    const int chunk_id = cid;
    const int offs = tid + chunk_id * PLR_CHUNK_SIZE;

    T val[PLR_X];

    if (chunk_id == gridDim.x - 1) {
        #pragma unroll
        for (int v = 0; v < PLR_X; v++) {
            val[v] = 0;
            if (offs + (v * PLR_BLOCK_SIZE) < N_SAMPLES) {
                val[v] = input[offs + (v * PLR_BLOCK_SIZE)];
            }
        }
    } else {
        #pragma unroll
        for (int v = 0; v < PLR_X; v++) {
            val[v] = input[offs + (v * PLR_BLOCK_SIZE)];
        }
    }

    // phase 1: FIR 
    #pragma unroll
    for (int v = PLR_X - 1; v >= 0; v--) {
        sbuf[tid + 8] = val[v];
        if ((tid + 8 - PLR_BLOCK_SIZE) >= 0) {
            if (v > 0) {
                sbuf[tid + 8 - PLR_BLOCK_SIZE] = val[v - 1];
            } else {
                // First segment - handle initial conditions or previous chunk
                if (chunk_id == 0) {
                    const int idx_init = tid + 8 - PLR_BLOCK_SIZE;
                    if (idx_init == 0) sbuf[idx_init] = xi8;
                    else if (idx_init == 1) sbuf[idx_init] = xi7;
                    else if (idx_init == 2) sbuf[idx_init] = xi6;
                    else if (idx_init == 3) sbuf[idx_init] = xi5;
                    else if (idx_init == 4) sbuf[idx_init] = xi4;
                    else if (idx_init == 5) sbuf[idx_init] = xi3;
                    else if (idx_init == 6) sbuf[idx_init] = xi2;
                    else if (idx_init == 7) sbuf[idx_init] = xi1;
                } else {
                    sbuf[tid + 8 - PLR_BLOCK_SIZE] = input[offs - PLR_BLOCK_SIZE];
                }
            }
        }
        __syncthreads();
        
        val[v] += b1 * sbuf[tid + 7];
        val[v] += b2 * sbuf[tid + 6];
        val[v] += b3 * sbuf[tid + 5];
        val[v] += b4 * sbuf[tid + 4];
        val[v] += b5 * sbuf[tid + 3];
        val[v] += b6 * sbuf[tid + 2];
        val[v] += b7 * sbuf[tid + 1];
        val[v] += b8 * sbuf[tid + 0];
        
        if (v > 0) __syncthreads();
    }

    // phase 2a: intra-warp iterative doubling 
    const T sfA = sfacA[lane];
    const T sfB = sfacB[lane];
    const T sfC = sfacC[lane];
    const T sfD = sfacD[lane];
    const T sfE = sfacE[lane];
    const T sfF = sfacF[lane];
    const T sfG = sfacG[lane];
    const T sfH = sfacH[lane];
    int cond;
    T help, spc;

    // Width 2: use a1
    help = a1;
    cond = ((lane & 1) != 0);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 0, 2);
        if (cond) val[v] += spc;
    }

    // Width 4: use G, H (last 2 coefficients for 2 positions)
    cond = ((lane & 2) != 0);
    help = __shfl_sync(0xffffffff, sfG, lane % 2);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 0, 4);
        if (cond) val[v] += spc;
    }
    help = __shfl_sync(0xffffffff, sfH, lane % 2);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 1, 4);
        if (cond) val[v] += spc;
    }

    // Width 8: use E, F, G, H (last 4 coefficients for 4 positions)
    cond = ((lane & 4) != 0);
    help = __shfl_sync(0xffffffff, sfE, lane % 4);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 0, 8);
        if (cond) val[v] += spc;
    }
    help = __shfl_sync(0xffffffff, sfF, lane % 4);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 1, 8);
        if (cond) val[v] += spc;
    }
    help = __shfl_sync(0xffffffff, sfG, lane % 4);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 2, 8);
        if (cond) val[v] += spc;
    }
    help = __shfl_sync(0xffffffff, sfH, lane % 4);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 3, 8);
        if (cond) val[v] += spc;
    }

    // Width 16: use all 8 coefficients A-H for 8 positions
    cond = ((lane & 8) != 0);
    help = __shfl_sync(0xffffffff, sfA, lane % 8);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 0, 16);
        if (cond) val[v] += spc;
    }
    help = __shfl_sync(0xffffffff, sfB, lane % 8);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 1, 16);
        if (cond) val[v] += spc;
    }
    help = __shfl_sync(0xffffffff, sfC, lane % 8);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 2, 16);
        if (cond) val[v] += spc;
    }
    help = __shfl_sync(0xffffffff, sfD, lane % 8);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 3, 16);
        if (cond) val[v] += spc;
    }
    help = __shfl_sync(0xffffffff, sfE, lane % 8);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 4, 16);
        if (cond) val[v] += spc;
    }
    help = __shfl_sync(0xffffffff, sfF, lane % 8);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 5, 16);
        if (cond) val[v] += spc;
    }
    help = __shfl_sync(0xffffffff, sfG, lane % 8);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 6, 16);
        if (cond) val[v] += spc;
    }
    help = __shfl_sync(0xffffffff, sfH, lane % 8);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 7, 16);
        if (cond) val[v] += spc;
    }

    // Width 32: use all 8 coefficients A-H for lanes 8-15
    cond = ((lane & 16) != 0);
    help = __shfl_sync(0xffffffff, sfA, lane % 16);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 8, 32);
        if (cond) val[v] += spc;
    }
    help = __shfl_sync(0xffffffff, sfB, lane % 16);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 9, 32);
        if (cond) val[v] += spc;
    }
    help = __shfl_sync(0xffffffff, sfC, lane % 16);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 10, 32);
        if (cond) val[v] += spc;
    }
    help = __shfl_sync(0xffffffff, sfD, lane % 16);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 11, 32);
        if (cond) val[v] += spc;
    }
    help = __shfl_sync(0xffffffff, sfE, lane % 16);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 12, 32);
        if (cond) val[v] += spc;
    }
    help = __shfl_sync(0xffffffff, sfF, lane % 16);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 13, 32);
        if (cond) val[v] += spc;
    }
    help = __shfl_sync(0xffffffff, sfG, lane % 16);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 14, 32);
        if (cond) val[v] += spc;
    }
    help = __shfl_sync(0xffffffff, sfH, lane % 16);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 15, 32);
        if (cond) val[v] += spc;
    }

    // phase 2b: inter-warp iterative doubling
    const int delta = PLR_BLOCK_SIZE / warp_size * order;
    const int clane = lane - (warp_size - order);  // -24 to 7 for order=8
    const int clwo = clane + warp * order;

    if (((warp & 1) == 0) && (clane >= 0)) {
        #pragma unroll
        for (int v = 0; v < PLR_X; v++) {
            spartc[clwo + v * delta] = val[v];
        }
    }

    __syncthreads();

    if ((warp & 1) != 0) {
        const int cwarp = ((warp & ~1) | 0) * order;
        const T helpA = sfacA[tid % (warp_size * 1)];
        const T helpB = sfacB[tid % (warp_size * 1)];
        const T helpC = sfacC[tid % (warp_size * 1)];
        const T helpD = sfacD[tid % (warp_size * 1)];
        const T helpE = sfacE[tid % (warp_size * 1)];
        const T helpF = sfacF[tid % (warp_size * 1)];
        const T helpG = sfacG[tid % (warp_size * 1)];
        const T helpH = sfacH[tid % (warp_size * 1)];
        #pragma unroll
        for (int v = 0; v < PLR_X; v++) {
            val[v] += helpA * spartc[cwarp + (v * delta + 0)];
            val[v] += helpB * spartc[cwarp + (v * delta + 1)];
            val[v] += helpC * spartc[cwarp + (v * delta + 2)];
            val[v] += helpD * spartc[cwarp + (v * delta + 3)];
            val[v] += helpE * spartc[cwarp + (v * delta + 4)];
            val[v] += helpF * spartc[cwarp + (v * delta + 5)];
            val[v] += helpG * spartc[cwarp + (v * delta + 6)];
            val[v] += helpH * spartc[cwarp + (v * delta + 7)];
        }
        if constexpr (num_warps > 2) {
            if (((warp & 3) != 0) && (clane >= 0)) {
                #pragma unroll
                for (int v = 0; v < PLR_X; v++) {
                    spartc[clwo + v * delta] = val[v];
                }
            }
        } else {
            if ((warp == last_warp) && (clane >= 0)) {
                #pragma unroll
                for (int v = 0; v < PLR_X; v++) {
                    spartc[clane + (last_warp * order + v * delta)] = val[v];
                }
            }
        }
    }

    if constexpr (num_warps >= 4) {
        __syncthreads();
        if ((warp & 2) != 0) {
            const int cwarp = ((warp & ~3) | 1) * order;
            const T helpA = sfacA[tid % (warp_size * 2)];
            const T helpB = sfacB[tid % (warp_size * 2)];
            const T helpC = sfacC[tid % (warp_size * 2)];
            const T helpD = sfacD[tid % (warp_size * 2)];
            const T helpE = sfacE[tid % (warp_size * 2)];
            const T helpF = sfacF[tid % (warp_size * 2)];
            const T helpG = sfacG[tid % (warp_size * 2)];
            const T helpH = sfacH[tid % (warp_size * 2)];
            #pragma unroll
            for (int v = 0; v < PLR_X; v++) {
                val[v] += helpA * spartc[cwarp + (v * delta + 0)];
                val[v] += helpB * spartc[cwarp + (v * delta + 1)];
                val[v] += helpC * spartc[cwarp + (v * delta + 2)];
                val[v] += helpD * spartc[cwarp + (v * delta + 3)];
                val[v] += helpE * spartc[cwarp + (v * delta + 4)];
                val[v] += helpF * spartc[cwarp + (v * delta + 5)];
                val[v] += helpG * spartc[cwarp + (v * delta + 6)];
                val[v] += helpH * spartc[cwarp + (v * delta + 7)];
            }
            if constexpr (num_warps > 4) {
                if (((warp & 7) != 0) && (clane >= 0)) {
                    #pragma unroll
                    for (int v = 0; v < PLR_X; v++) {
                        spartc[clwo + v * delta] = val[v];
                    }
                }
            } else {
                if ((warp == last_warp) && (clane >= 0)) {
                    #pragma unroll
                    for (int v = 0; v < PLR_X; v++) {
                        spartc[clane + (last_warp * order + v * delta)] = val[v];
                    }
                }
            }
        }
    }

    if constexpr (num_warps >= 8) {
        __syncthreads();
        if ((warp & 4) != 0) {
            const int cwarp = ((warp & ~7) | 3) * order;
            const T helpA = sfacA[tid % (warp_size * 4)];
            const T helpB = sfacB[tid % (warp_size * 4)];
            const T helpC = sfacC[tid % (warp_size * 4)];
            const T helpD = sfacD[tid % (warp_size * 4)];
            const T helpE = sfacE[tid % (warp_size * 4)];
            const T helpF = sfacF[tid % (warp_size * 4)];
            const T helpG = sfacG[tid % (warp_size * 4)];
            const T helpH = sfacH[tid % (warp_size * 4)];
            #pragma unroll
            for (int v = 0; v < PLR_X; v++) {
                val[v] += helpA * spartc[cwarp + (v * delta + 0)];
                val[v] += helpB * spartc[cwarp + (v * delta + 1)];
                val[v] += helpC * spartc[cwarp + (v * delta + 2)];
                val[v] += helpD * spartc[cwarp + (v * delta + 3)];
                val[v] += helpE * spartc[cwarp + (v * delta + 4)];
                val[v] += helpF * spartc[cwarp + (v * delta + 5)];
                val[v] += helpG * spartc[cwarp + (v * delta + 6)];
                val[v] += helpH * spartc[cwarp + (v * delta + 7)];
            }
            if constexpr (num_warps > 8) {
                if (((warp & 15) != 0) && (clane >= 0)) {
                    #pragma unroll
                    for (int v = 0; v < PLR_X; v++) {
                        spartc[clwo + v * delta] = val[v];
                    }
                }
            } else {
                if ((warp == last_warp) && (clane >= 0)) {
                    #pragma unroll
                    for (int v = 0; v < PLR_X; v++) {
                        spartc[clane + (last_warp * order + v * delta)] = val[v];
                    }
                }
            }
        }
    }

    if constexpr (num_warps >= 16) {
        __syncthreads();
        if ((warp & 8) != 0) {
            const int cwarp = ((warp & ~15) | 7) * order;
            const T helpA = sfacA[tid % (warp_size * 8)];
            const T helpB = sfacB[tid % (warp_size * 8)];
            const T helpC = sfacC[tid % (warp_size * 8)];
            const T helpD = sfacD[tid % (warp_size * 8)];
            const T helpE = sfacE[tid % (warp_size * 8)];
            const T helpF = sfacF[tid % (warp_size * 8)];
            const T helpG = sfacG[tid % (warp_size * 8)];
            const T helpH = sfacH[tid % (warp_size * 8)];
            #pragma unroll
            for (int v = 0; v < PLR_X; v++) {
                val[v] += helpA * spartc[cwarp + (v * delta + 0)];
                val[v] += helpB * spartc[cwarp + (v * delta + 1)];
                val[v] += helpC * spartc[cwarp + (v * delta + 2)];
                val[v] += helpD * spartc[cwarp + (v * delta + 3)];
                val[v] += helpE * spartc[cwarp + (v * delta + 4)];
                val[v] += helpF * spartc[cwarp + (v * delta + 5)];
                val[v] += helpG * spartc[cwarp + (v * delta + 6)];
                val[v] += helpH * spartc[cwarp + (v * delta + 7)];
            }
            if constexpr (num_warps > 16) {
                if ((warp == 15) && (clane >= 0)) {
                    #pragma unroll
                    for (int v = 0; v < PLR_X; v++) {
                        spartc[clane + (15 * order + v * delta)] = val[v];
                    }
                }
            } else {
                if ((warp == last_warp) && (clane >= 0)) {
                    #pragma unroll
                    for (int v = 0; v < PLR_X; v++) {
                        spartc[clane + (last_warp * order + v * delta)] = val[v];
                    }
                }
            }
        }
    }

    if constexpr (num_warps == 32) {
        __syncthreads();
        if ((warp & 16) != 0) {
            const int cwarp = 15 * order;
            const T helpA = sfacA[tid % (warp_size * 16)];
            const T helpB = sfacB[tid % (warp_size * 16)];
            const T helpC = sfacC[tid % (warp_size * 16)];
            const T helpD = sfacD[tid % (warp_size * 16)];
            const T helpE = sfacE[tid % (warp_size * 16)];
            const T helpF = sfacF[tid % (warp_size * 16)];
            const T helpG = sfacG[tid % (warp_size * 16)];
            const T helpH = sfacH[tid % (warp_size * 16)];
            #pragma unroll
            for (int v = 0; v < PLR_X; v++) {
                val[v] += helpA * spartc[cwarp + (v * delta + 0)];
                val[v] += helpB * spartc[cwarp + (v * delta + 1)];
                val[v] += helpC * spartc[cwarp + (v * delta + 2)];
                val[v] += helpD * spartc[cwarp + (v * delta + 3)];
                val[v] += helpE * spartc[cwarp + (v * delta + 4)];
                val[v] += helpF * spartc[cwarp + (v * delta + 5)];
                val[v] += helpG * spartc[cwarp + (v * delta + 6)];
                val[v] += helpH * spartc[cwarp + (v * delta + 7)];
            }
            if ((warp == last_warp) && (clane >= 0)) {
                #pragma unroll
                for (int v = 0; v < PLR_X; v++) {
                    spartc[clane + (last_warp * order + v * delta)] = val[v];
                }
            }
        }
    }

    if constexpr (PLR_X > 1) {
        if ((warp == last_warp) && (clane >= 0)) {
            spartc[clane + (last_warp * order + 0 * delta)] = val[0];
        }
        __syncthreads();
        
        #pragma unroll
        for (int v = 1; v < PLR_X; v++) {
            val[v] += sfacA[tid] * spartc[last_warp * order + ((v-1) * delta + 0)];
            val[v] += sfacB[tid] * spartc[last_warp * order + ((v-1) * delta + 1)];
            val[v] += sfacC[tid] * spartc[last_warp * order + ((v-1) * delta + 2)];
            val[v] += sfacD[tid] * spartc[last_warp * order + ((v-1) * delta + 3)];
            val[v] += sfacE[tid] * spartc[last_warp * order + ((v-1) * delta + 4)];
            val[v] += sfacF[tid] * spartc[last_warp * order + ((v-1) * delta + 5)];
            val[v] += sfacG[tid] * spartc[last_warp * order + ((v-1) * delta + 6)];
            val[v] += sfacH[tid] * spartc[last_warp * order + ((v-1) * delta + 7)];
            
            if ((warp == last_warp) && (clane >= 0)) {
                spartc[clane + (last_warp * order + v * delta)] = val[v];
            }
            
            if (v < PLR_X - 1) {
                __syncthreads();
            }
        }
    }

        // phase 3a: inter-block carry propagation 
    const int idx = tid - (PLR_BLOCK_SIZE - order);
    const int last_val_idx = PLR_X - 1;

    if (idx >= 0) {
        partcarry[chunk_id * order + idx] = val[last_val_idx];
    }
    __syncwarp();
    __threadfence();
    if (idx == 0) {
        status[chunk_id] = part_flag;
    }

    __syncthreads();

    if (warp == 0) {
        int flag = part_flag;
        bool no_zeros, has_status_2, reached_origin;
        
        do {
            if (chunk_id > lane) {
                flag = status[chunk_id - 1 - lane];
            }
            no_zeros = !__any_sync(0xffffffff, flag < part_flag);
            has_status_2 = !__all_sync(0xffffffff, flag != full_flag) && no_zeros;
            reached_origin = (chunk_id < warp_size) && no_zeros;
        } while (!(has_status_2 || reached_origin));
        
        __threadfence();
        
        int mask = __ballot_sync(0xffffffff, flag == full_flag);
        
        T X0, X1, X2, X3, X4, X5, X6, X7;
        int start_chunk;
        
        if (mask == 0) {
            X0 = yi8;
            X1 = yi7;
            X2 = yi6;
            X3 = yi5;
            X4 = yi4;
            X5 = yi3;
            X6 = yi2;
            X7 = yi1;
            start_chunk = 0;
        } else {
            const int pos = __ffs(mask) - 1;
            const int full_chunk = chunk_id - 1 - pos;
            start_chunk = full_chunk + 1;
            
            T fc;
            if (lane < order) {
                fc = fullcarry[full_chunk * order + lane];
            }
            X0 = __shfl_sync(0xffffffff, fc, 0);
            X1 = __shfl_sync(0xffffffff, fc, 1);
            X2 = __shfl_sync(0xffffffff, fc, 2);
            X3 = __shfl_sync(0xffffffff, fc, 3);
            X4 = __shfl_sync(0xffffffff, fc, 4);
            X5 = __shfl_sync(0xffffffff, fc, 5);
            X6 = __shfl_sync(0xffffffff, fc, 6);
            X7 = __shfl_sync(0xffffffff, fc, 7);
        }
        
        const int num_partcarries = chunk_id - start_chunk;
        
        if (num_partcarries > 0) {
            for (int i = (start_chunk * order + lane); i < (chunk_id * order); i += warp_size) {
                spartc[i - start_chunk * order] = partcarry[i];
            }
            __syncwarp();
            
            if (lane == 0) {
                for (int i = 0; i < num_partcarries; i++) {
                    const T h0 = spartc[i * order + 0] 
                               + X0 * facA[PLR_CHUNK_SIZE-8] + X1 * facB[PLR_CHUNK_SIZE-8] 
                               + X2 * facC[PLR_CHUNK_SIZE-8] + X3 * facD[PLR_CHUNK_SIZE-8]
                               + X4 * facE[PLR_CHUNK_SIZE-8] + X5 * facF[PLR_CHUNK_SIZE-8]
                               + X6 * facG[PLR_CHUNK_SIZE-8] + X7 * facH[PLR_CHUNK_SIZE-8];
                    const T h1 = spartc[i * order + 1] 
                               + X0 * facA[PLR_CHUNK_SIZE-7] + X1 * facB[PLR_CHUNK_SIZE-7] 
                               + X2 * facC[PLR_CHUNK_SIZE-7] + X3 * facD[PLR_CHUNK_SIZE-7]
                               + X4 * facE[PLR_CHUNK_SIZE-7] + X5 * facF[PLR_CHUNK_SIZE-7]
                               + X6 * facG[PLR_CHUNK_SIZE-7] + X7 * facH[PLR_CHUNK_SIZE-7];
                    const T h2 = spartc[i * order + 2] 
                               + X0 * facA[PLR_CHUNK_SIZE-6] + X1 * facB[PLR_CHUNK_SIZE-6] 
                               + X2 * facC[PLR_CHUNK_SIZE-6] + X3 * facD[PLR_CHUNK_SIZE-6]
                               + X4 * facE[PLR_CHUNK_SIZE-6] + X5 * facF[PLR_CHUNK_SIZE-6]
                               + X6 * facG[PLR_CHUNK_SIZE-6] + X7 * facH[PLR_CHUNK_SIZE-6];
                    const T h3 = spartc[i * order + 3] 
                               + X0 * facA[PLR_CHUNK_SIZE-5] + X1 * facB[PLR_CHUNK_SIZE-5] 
                               + X2 * facC[PLR_CHUNK_SIZE-5] + X3 * facD[PLR_CHUNK_SIZE-5]
                               + X4 * facE[PLR_CHUNK_SIZE-5] + X5 * facF[PLR_CHUNK_SIZE-5]
                               + X6 * facG[PLR_CHUNK_SIZE-5] + X7 * facH[PLR_CHUNK_SIZE-5];
                    const T h4 = spartc[i * order + 4] 
                               + X0 * facA[PLR_CHUNK_SIZE-4] + X1 * facB[PLR_CHUNK_SIZE-4] 
                               + X2 * facC[PLR_CHUNK_SIZE-4] + X3 * facD[PLR_CHUNK_SIZE-4]
                               + X4 * facE[PLR_CHUNK_SIZE-4] + X5 * facF[PLR_CHUNK_SIZE-4]
                               + X6 * facG[PLR_CHUNK_SIZE-4] + X7 * facH[PLR_CHUNK_SIZE-4];
                    const T h5 = spartc[i * order + 5] 
                               + X0 * facA[PLR_CHUNK_SIZE-3] + X1 * facB[PLR_CHUNK_SIZE-3] 
                               + X2 * facC[PLR_CHUNK_SIZE-3] + X3 * facD[PLR_CHUNK_SIZE-3]
                               + X4 * facE[PLR_CHUNK_SIZE-3] + X5 * facF[PLR_CHUNK_SIZE-3]
                               + X6 * facG[PLR_CHUNK_SIZE-3] + X7 * facH[PLR_CHUNK_SIZE-3];
                    const T h6 = spartc[i * order + 6] 
                               + X0 * facA[PLR_CHUNK_SIZE-2] + X1 * facB[PLR_CHUNK_SIZE-2] 
                               + X2 * facC[PLR_CHUNK_SIZE-2] + X3 * facD[PLR_CHUNK_SIZE-2]
                               + X4 * facE[PLR_CHUNK_SIZE-2] + X5 * facF[PLR_CHUNK_SIZE-2]
                               + X6 * facG[PLR_CHUNK_SIZE-2] + X7 * facH[PLR_CHUNK_SIZE-2];
                    const T h7 = spartc[i * order + 7] 
                               + X0 * facA[PLR_CHUNK_SIZE-1] + X1 * facB[PLR_CHUNK_SIZE-1] 
                               + X2 * facC[PLR_CHUNK_SIZE-1] + X3 * facD[PLR_CHUNK_SIZE-1]
                               + X4 * facE[PLR_CHUNK_SIZE-1] + X5 * facF[PLR_CHUNK_SIZE-1]
                               + X6 * facG[PLR_CHUNK_SIZE-1] + X7 * facH[PLR_CHUNK_SIZE-1];
                    X0 = h0;
                    X1 = h1;
                    X2 = h2;
                    X3 = h3;
                    X4 = h4;
                    X5 = h5;
                    X6 = h6;
                    X7 = h7;
                }
            }
        }
        
        if (lane == 0) {
            sfullc[0] = X0;
            sfullc[1] = X1;
            sfullc[2] = X2;
            sfullc[3] = X3;
            sfullc[4] = X4;
            sfullc[5] = X5;
            sfullc[6] = X6;
            sfullc[7] = X7;
        }
    }

    __syncthreads();

    // phase 3b: apply carry to all values
    T X0 = sfullc[0];
    T X1 = sfullc[1];
    T X2 = sfullc[2];
    T X3 = sfullc[3];
    T X4 = sfullc[4];
    T X5 = sfullc[5];
    T X6 = sfullc[6];
    T X7 = sfullc[7];

    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        val[v] += facA[tid + v * PLR_BLOCK_SIZE] * X0;
        val[v] += facB[tid + v * PLR_BLOCK_SIZE] * X1;
        val[v] += facC[tid + v * PLR_BLOCK_SIZE] * X2;
        val[v] += facD[tid + v * PLR_BLOCK_SIZE] * X3;
        val[v] += facE[tid + v * PLR_BLOCK_SIZE] * X4;
        val[v] += facF[tid + v * PLR_BLOCK_SIZE] * X5;
        val[v] += facG[tid + v * PLR_BLOCK_SIZE] * X6;
        val[v] += facH[tid + v * PLR_BLOCK_SIZE] * X7;
    }

    if (idx >= 0) {
        fullcarry[chunk_id * order + idx] = val[last_val_idx];
    }
    __syncwarp();
    __threadfence();
    if (idx == 0) {
        status[chunk_id] = full_flag;
    }

    if (chunk_id == gridDim.x - 1) {
        #pragma unroll
        for (int v = 0; v < PLR_X; v++) {
            if (offs + (v * PLR_BLOCK_SIZE) < N_SAMPLES) {
                output[offs + (v * PLR_BLOCK_SIZE)] = val[v];
            }
        }
    } else {
        #pragma unroll
        for (int v = 0; v < PLR_X; v++) {
            output[offs + (v * PLR_BLOCK_SIZE)] = val[v];
        }
    }

}
#endif

#if PLR_ORDER == 16
static __global__ __launch_bounds__(PLR_BLOCK_SIZE, PLR_N_TB_PER_SM)
void PLR_16(const T* const __restrict__ input, 
                        T* const __restrict__ output, 
                        volatile int* const __restrict__ status, 
                        volatile T* const __restrict__ partcarry, 
                        volatile T* const __restrict__ fullcarry,
                       const int launch)
{
    constexpr int num_warps = PLR_BLOCK_SIZE / warp_size;
    constexpr int last_warp = num_warps - 1;
    const int tid = threadIdx.x;
    const int warp = tid / warp_size;
    const int lane = tid % warp_size;

    // Launch-versioned status flags (protocol identical to ph_kernels.cuh):
    // values below part_flag - including everything a previous launch
    // wrote - read as "not ready", so status is zeroed once, never reset.
    const int part_flag = 2 * launch + 1;
    const int full_flag = part_flag + 1;

    __shared__ T spartc[PLR_CHUNK_SIZE / warp_size * order];
    __shared__ T sfullc[order];
    __shared__ int cid;
    __shared__ T sbuf[PLR_BLOCK_SIZE + 16];  // +16 for 16th order

    // 16 shared factor arrays
    __shared__ T sfacA[PLR_BLOCK_SIZE];
    __shared__ T sfacB[PLR_BLOCK_SIZE];
    __shared__ T sfacC[PLR_BLOCK_SIZE];
    __shared__ T sfacD[PLR_BLOCK_SIZE];
    __shared__ T sfacE[PLR_BLOCK_SIZE];
    __shared__ T sfacF[PLR_BLOCK_SIZE];
    __shared__ T sfacG[PLR_BLOCK_SIZE];
    __shared__ T sfacH[PLR_BLOCK_SIZE];
    __shared__ T sfacI[PLR_BLOCK_SIZE];
    __shared__ T sfacJ[PLR_BLOCK_SIZE];
    __shared__ T sfacK[PLR_BLOCK_SIZE];
    __shared__ T sfacL[PLR_BLOCK_SIZE];
    __shared__ T sfacM[PLR_BLOCK_SIZE];
    __shared__ T sfacN[PLR_BLOCK_SIZE];
    __shared__ T sfacO[PLR_BLOCK_SIZE];
    __shared__ T sfacP[PLR_BLOCK_SIZE];
    
    sfacA[tid] = facA[tid];
    sfacB[tid] = facB[tid];
    sfacC[tid] = facC[tid];
    sfacD[tid] = facD[tid];
    sfacE[tid] = facE[tid];
    sfacF[tid] = facF[tid];
    sfacG[tid] = facG[tid];
    sfacH[tid] = facH[tid];
    sfacI[tid] = facI[tid];
    sfacJ[tid] = facJ[tid];
    sfacK[tid] = facK[tid];
    sfacL[tid] = facL[tid];
    sfacM[tid] = facM[tid];
    sfacN[tid] = facN[tid];
    sfacO[tid] = facO[tid];
    sfacP[tid] = facP[tid];

    if (tid == 0) {
        cid = (int)(atomicAdd(&counter, 1u) - (unsigned int)launch * gridDim.x);
    }
    __syncthreads();

    const int chunk_id = cid;
    const int offs = tid + chunk_id * PLR_CHUNK_SIZE;

    T val[PLR_X];

    if (chunk_id == gridDim.x - 1) {
        #pragma unroll
        for (int v = 0; v < PLR_X; v++) {
            val[v] = 0;
            if (offs + (v * PLR_BLOCK_SIZE) < N_SAMPLES) {
                val[v] = input[offs + (v * PLR_BLOCK_SIZE)];
            }
        }
    } else {
        #pragma unroll
        for (int v = 0; v < PLR_X; v++) {
            val[v] = input[offs + (v * PLR_BLOCK_SIZE)];
        }
    }

    // phase 1: FIR 
    #pragma unroll
    for (int v = PLR_X - 1; v >= 0; v--) {
        sbuf[tid + 16] = val[v];
        if ((tid + 16 - PLR_BLOCK_SIZE) >= 0) {
            if (v > 0) {
                sbuf[tid + 16 - PLR_BLOCK_SIZE] = val[v - 1];
            } else {
                // First segment - handle initial conditions or previous chunk
                if (chunk_id == 0) {
                    const int idx_init = tid + 16 - PLR_BLOCK_SIZE;
                    if (idx_init == 0) sbuf[idx_init] = xi16;
                    else if (idx_init == 1) sbuf[idx_init] = xi15;
                    else if (idx_init == 2) sbuf[idx_init] = xi14;
                    else if (idx_init == 3) sbuf[idx_init] = xi13;
                    else if (idx_init == 4) sbuf[idx_init] = xi12;
                    else if (idx_init == 5) sbuf[idx_init] = xi11;
                    else if (idx_init == 6) sbuf[idx_init] = xi10;
                    else if (idx_init == 7) sbuf[idx_init] = xi9;
                    else if (idx_init == 8) sbuf[idx_init] = xi8;
                    else if (idx_init == 9) sbuf[idx_init] = xi7;
                    else if (idx_init == 10) sbuf[idx_init] = xi6;
                    else if (idx_init == 11) sbuf[idx_init] = xi5;
                    else if (idx_init == 12) sbuf[idx_init] = xi4;
                    else if (idx_init == 13) sbuf[idx_init] = xi3;
                    else if (idx_init == 14) sbuf[idx_init] = xi2;
                    else if (idx_init == 15) sbuf[idx_init] = xi1;
                } else {
                    sbuf[tid + 16 - PLR_BLOCK_SIZE] = input[offs - PLR_BLOCK_SIZE];
                }
            }
        }
        __syncthreads();
        
        val[v] += b1 * sbuf[tid + 15];
        val[v] += b2 * sbuf[tid + 14];
        val[v] += b3 * sbuf[tid + 13];
        val[v] += b4 * sbuf[tid + 12];
        val[v] += b5 * sbuf[tid + 11];
        val[v] += b6 * sbuf[tid + 10];
        val[v] += b7 * sbuf[tid + 9];
        val[v] += b8 * sbuf[tid + 8];
        val[v] += b9 * sbuf[tid + 7];
        val[v] += b10 * sbuf[tid + 6];
        val[v] += b11 * sbuf[tid + 5];
        val[v] += b12 * sbuf[tid + 4];
        val[v] += b13 * sbuf[tid + 3];
        val[v] += b14 * sbuf[tid + 2];
        val[v] += b15 * sbuf[tid + 1];
        val[v] += b16 * sbuf[tid + 0];
        
        if (v > 0) __syncthreads();
    }

    // phase 2a: intra-warp iterative doubling 
    const T sfA = sfacA[lane];
    const T sfB = sfacB[lane];
    const T sfC = sfacC[lane];
    const T sfD = sfacD[lane];
    const T sfE = sfacE[lane];
    const T sfF = sfacF[lane];
    const T sfG = sfacG[lane];
    const T sfH = sfacH[lane];
    const T sfI = sfacI[lane];
    const T sfJ = sfacJ[lane];
    const T sfK = sfacK[lane];
    const T sfL = sfacL[lane];
    const T sfM = sfacM[lane];
    const T sfN = sfacN[lane];
    const T sfO = sfacO[lane];
    const T sfP = sfacP[lane];
    int cond;
    T help, spc;

    // Width 2: use a1
    help = a1;
    cond = ((lane & 1) != 0);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 0, 2);
        if (cond) val[v] += spc;
    }

    // Width 4: use O, P (last 2 coefficients for 2 positions)
    cond = ((lane & 2) != 0);
    help = __shfl_sync(0xffffffff, sfO, lane % 2);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 0, 4);
        if (cond) val[v] += spc;
    }
    help = __shfl_sync(0xffffffff, sfP, lane % 2);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 1, 4);
        if (cond) val[v] += spc;
    }

    // Width 8: use M, N, O, P (last 4 coefficients for 4 positions)
    cond = ((lane & 4) != 0);
    help = __shfl_sync(0xffffffff, sfM, lane % 4);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 0, 8);
        if (cond) val[v] += spc;
    }
    help = __shfl_sync(0xffffffff, sfN, lane % 4);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 1, 8);
        if (cond) val[v] += spc;
    }
    help = __shfl_sync(0xffffffff, sfO, lane % 4);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 2, 8);
        if (cond) val[v] += spc;
    }
    help = __shfl_sync(0xffffffff, sfP, lane % 4);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 3, 8);
        if (cond) val[v] += spc;
    }

    // Width 16: use I-P (last 8 coefficients for 8 positions)
    cond = ((lane & 8) != 0);
    help = __shfl_sync(0xffffffff, sfI, lane % 8);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 0, 16);
        if (cond) val[v] += spc;
    }
    help = __shfl_sync(0xffffffff, sfJ, lane % 8);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 1, 16);
        if (cond) val[v] += spc;
    }
    help = __shfl_sync(0xffffffff, sfK, lane % 8);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 2, 16);
        if (cond) val[v] += spc;
    }
    help = __shfl_sync(0xffffffff, sfL, lane % 8);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 3, 16);
        if (cond) val[v] += spc;
    }
    help = __shfl_sync(0xffffffff, sfM, lane % 8);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 4, 16);
        if (cond) val[v] += spc;
    }
    help = __shfl_sync(0xffffffff, sfN, lane % 8);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 5, 16);
        if (cond) val[v] += spc;
    }
    help = __shfl_sync(0xffffffff, sfO, lane % 8);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 6, 16);
        if (cond) val[v] += spc;
    }
    help = __shfl_sync(0xffffffff, sfP, lane % 8);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 7, 16);
        if (cond) val[v] += spc;
    }

    // Width 32: use all 16 coefficients A-P for lanes 0-15
    cond = ((lane & 16) != 0);
    help = __shfl_sync(0xffffffff, sfA, lane % 16);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 0, 32);
        if (cond) val[v] += spc;
    }
    help = __shfl_sync(0xffffffff, sfB, lane % 16);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 1, 32);
        if (cond) val[v] += spc;
    }
    help = __shfl_sync(0xffffffff, sfC, lane % 16);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 2, 32);
        if (cond) val[v] += spc;
    }
    help = __shfl_sync(0xffffffff, sfD, lane % 16);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 3, 32);
        if (cond) val[v] += spc;
    }
    help = __shfl_sync(0xffffffff, sfE, lane % 16);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 4, 32);
        if (cond) val[v] += spc;
    }
    help = __shfl_sync(0xffffffff, sfF, lane % 16);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 5, 32);
        if (cond) val[v] += spc;
    }
    help = __shfl_sync(0xffffffff, sfG, lane % 16);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 6, 32);
        if (cond) val[v] += spc;
    }
    help = __shfl_sync(0xffffffff, sfH, lane % 16);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 7, 32);
        if (cond) val[v] += spc;
    }
    help = __shfl_sync(0xffffffff, sfI, lane % 16);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 8, 32);
        if (cond) val[v] += spc;
    }
    help = __shfl_sync(0xffffffff, sfJ, lane % 16);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 9, 32);
        if (cond) val[v] += spc;
    }
    help = __shfl_sync(0xffffffff, sfK, lane % 16);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 10, 32);
        if (cond) val[v] += spc;
    }
    help = __shfl_sync(0xffffffff, sfL, lane % 16);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 11, 32);
        if (cond) val[v] += spc;
    }
    help = __shfl_sync(0xffffffff, sfM, lane % 16);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 12, 32);
        if (cond) val[v] += spc;
    }
    help = __shfl_sync(0xffffffff, sfN, lane % 16);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 13, 32);
        if (cond) val[v] += spc;
    }
    help = __shfl_sync(0xffffffff, sfO, lane % 16);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 14, 32);
        if (cond) val[v] += spc;
    }
    help = __shfl_sync(0xffffffff, sfP, lane % 16);
    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        spc = help * __shfl_sync(0xffffffff, val[v], 15, 32);
        if (cond) val[v] += spc;
    }

    // phase 2b: inter-warp iterative doubling
    const int delta = PLR_BLOCK_SIZE / warp_size * order;
    const int clane = lane - (warp_size - order);  // -16 to 15 for order=16
    const int clwo = clane + warp * order;

    if (((warp & 1) == 0) && (clane >= 0)) {
        #pragma unroll
        for (int v = 0; v < PLR_X; v++) {
            spartc[clwo + v * delta] = val[v];
        }
    }

    __syncthreads();

    if ((warp & 1) != 0) {
        const int cwarp = ((warp & ~1) | 0) * order;
        const T helpA = sfacA[tid % (warp_size * 1)];
        const T helpB = sfacB[tid % (warp_size * 1)];
        const T helpC = sfacC[tid % (warp_size * 1)];
        const T helpD = sfacD[tid % (warp_size * 1)];
        const T helpE = sfacE[tid % (warp_size * 1)];
        const T helpF = sfacF[tid % (warp_size * 1)];
        const T helpG = sfacG[tid % (warp_size * 1)];
        const T helpH = sfacH[tid % (warp_size * 1)];
        const T helpI = sfacI[tid % (warp_size * 1)];
        const T helpJ = sfacJ[tid % (warp_size * 1)];
        const T helpK = sfacK[tid % (warp_size * 1)];
        const T helpL = sfacL[tid % (warp_size * 1)];
        const T helpM = sfacM[tid % (warp_size * 1)];
        const T helpN = sfacN[tid % (warp_size * 1)];
        const T helpO = sfacO[tid % (warp_size * 1)];
        const T helpP = sfacP[tid % (warp_size * 1)];
        #pragma unroll
        for (int v = 0; v < PLR_X; v++) {
            val[v] += helpA * spartc[cwarp + (v * delta + 0)];
            val[v] += helpB * spartc[cwarp + (v * delta + 1)];
            val[v] += helpC * spartc[cwarp + (v * delta + 2)];
            val[v] += helpD * spartc[cwarp + (v * delta + 3)];
            val[v] += helpE * spartc[cwarp + (v * delta + 4)];
            val[v] += helpF * spartc[cwarp + (v * delta + 5)];
            val[v] += helpG * spartc[cwarp + (v * delta + 6)];
            val[v] += helpH * spartc[cwarp + (v * delta + 7)];
            val[v] += helpI * spartc[cwarp + (v * delta + 8)];
            val[v] += helpJ * spartc[cwarp + (v * delta + 9)];
            val[v] += helpK * spartc[cwarp + (v * delta + 10)];
            val[v] += helpL * spartc[cwarp + (v * delta + 11)];
            val[v] += helpM * spartc[cwarp + (v * delta + 12)];
            val[v] += helpN * spartc[cwarp + (v * delta + 13)];
            val[v] += helpO * spartc[cwarp + (v * delta + 14)];
            val[v] += helpP * spartc[cwarp + (v * delta + 15)];
        }
        if constexpr (num_warps > 2) {
            if (((warp & 3) != 0) && (clane >= 0)) {
                #pragma unroll
                for (int v = 0; v < PLR_X; v++) {
                    spartc[clwo + v * delta] = val[v];
                }
            }
        } else {
            if ((warp == last_warp) && (clane >= 0)) {
                #pragma unroll
                for (int v = 0; v < PLR_X; v++) {
                    spartc[clane + (last_warp * order + v * delta)] = val[v];
                }
            }
        }
    }

    if constexpr (num_warps >= 4) {
        __syncthreads();
        if ((warp & 2) != 0) {
            const int cwarp = ((warp & ~3) | 1) * order;
            const T helpA = sfacA[tid % (warp_size * 2)];
            const T helpB = sfacB[tid % (warp_size * 2)];
            const T helpC = sfacC[tid % (warp_size * 2)];
            const T helpD = sfacD[tid % (warp_size * 2)];
            const T helpE = sfacE[tid % (warp_size * 2)];
            const T helpF = sfacF[tid % (warp_size * 2)];
            const T helpG = sfacG[tid % (warp_size * 2)];
            const T helpH = sfacH[tid % (warp_size * 2)];
            const T helpI = sfacI[tid % (warp_size * 2)];
            const T helpJ = sfacJ[tid % (warp_size * 2)];
            const T helpK = sfacK[tid % (warp_size * 2)];
            const T helpL = sfacL[tid % (warp_size * 2)];
            const T helpM = sfacM[tid % (warp_size * 2)];
            const T helpN = sfacN[tid % (warp_size * 2)];
            const T helpO = sfacO[tid % (warp_size * 2)];
            const T helpP = sfacP[tid % (warp_size * 2)];
            #pragma unroll
            for (int v = 0; v < PLR_X; v++) {
                val[v] += helpA * spartc[cwarp + (v * delta + 0)];
                val[v] += helpB * spartc[cwarp + (v * delta + 1)];
                val[v] += helpC * spartc[cwarp + (v * delta + 2)];
                val[v] += helpD * spartc[cwarp + (v * delta + 3)];
                val[v] += helpE * spartc[cwarp + (v * delta + 4)];
                val[v] += helpF * spartc[cwarp + (v * delta + 5)];
                val[v] += helpG * spartc[cwarp + (v * delta + 6)];
                val[v] += helpH * spartc[cwarp + (v * delta + 7)];
                val[v] += helpI * spartc[cwarp + (v * delta + 8)];
                val[v] += helpJ * spartc[cwarp + (v * delta + 9)];
                val[v] += helpK * spartc[cwarp + (v * delta + 10)];
                val[v] += helpL * spartc[cwarp + (v * delta + 11)];
                val[v] += helpM * spartc[cwarp + (v * delta + 12)];
                val[v] += helpN * spartc[cwarp + (v * delta + 13)];
                val[v] += helpO * spartc[cwarp + (v * delta + 14)];
                val[v] += helpP * spartc[cwarp + (v * delta + 15)];
            }
            if constexpr (num_warps > 4) {
                if (((warp & 7) != 0) && (clane >= 0)) {
                    #pragma unroll
                    for (int v = 0; v < PLR_X; v++) {
                        spartc[clwo + v * delta] = val[v];
                    }
                }
            } else {
                if ((warp == last_warp) && (clane >= 0)) {
                    #pragma unroll
                    for (int v = 0; v < PLR_X; v++) {
                        spartc[clane + (last_warp * order + v * delta)] = val[v];
                    }
                }
            }
        }
    }

    if constexpr (num_warps >= 8) {
        __syncthreads();
        if ((warp & 4) != 0) {
            const int cwarp = ((warp & ~7) | 3) * order;
            const T helpA = sfacA[tid % (warp_size * 4)];
            const T helpB = sfacB[tid % (warp_size * 4)];
            const T helpC = sfacC[tid % (warp_size * 4)];
            const T helpD = sfacD[tid % (warp_size * 4)];
            const T helpE = sfacE[tid % (warp_size * 4)];
            const T helpF = sfacF[tid % (warp_size * 4)];
            const T helpG = sfacG[tid % (warp_size * 4)];
            const T helpH = sfacH[tid % (warp_size * 4)];
            const T helpI = sfacI[tid % (warp_size * 4)];
            const T helpJ = sfacJ[tid % (warp_size * 4)];
            const T helpK = sfacK[tid % (warp_size * 4)];
            const T helpL = sfacL[tid % (warp_size * 4)];
            const T helpM = sfacM[tid % (warp_size * 4)];
            const T helpN = sfacN[tid % (warp_size * 4)];
            const T helpO = sfacO[tid % (warp_size * 4)];
            const T helpP = sfacP[tid % (warp_size * 4)];
            #pragma unroll
            for (int v = 0; v < PLR_X; v++) {
                val[v] += helpA * spartc[cwarp + (v * delta + 0)];
                val[v] += helpB * spartc[cwarp + (v * delta + 1)];
                val[v] += helpC * spartc[cwarp + (v * delta + 2)];
                val[v] += helpD * spartc[cwarp + (v * delta + 3)];
                val[v] += helpE * spartc[cwarp + (v * delta + 4)];
                val[v] += helpF * spartc[cwarp + (v * delta + 5)];
                val[v] += helpG * spartc[cwarp + (v * delta + 6)];
                val[v] += helpH * spartc[cwarp + (v * delta + 7)];
                val[v] += helpI * spartc[cwarp + (v * delta + 8)];
                val[v] += helpJ * spartc[cwarp + (v * delta + 9)];
                val[v] += helpK * spartc[cwarp + (v * delta + 10)];
                val[v] += helpL * spartc[cwarp + (v * delta + 11)];
                val[v] += helpM * spartc[cwarp + (v * delta + 12)];
                val[v] += helpN * spartc[cwarp + (v * delta + 13)];
                val[v] += helpO * spartc[cwarp + (v * delta + 14)];
                val[v] += helpP * spartc[cwarp + (v * delta + 15)];
            }
            if constexpr (num_warps > 8) {
                if (((warp & 15) != 0) && (clane >= 0)) {
                    #pragma unroll
                    for (int v = 0; v < PLR_X; v++) {
                        spartc[clwo + v * delta] = val[v];
                    }
                }
            } else {
                if ((warp == last_warp) && (clane >= 0)) {
                    #pragma unroll
                    for (int v = 0; v < PLR_X; v++) {
                        spartc[clane + (last_warp * order + v * delta)] = val[v];
                    }
                }
            }
        }
    }

    if constexpr (num_warps >= 16) {
        __syncthreads();
        if ((warp & 8) != 0) {
            const int cwarp = ((warp & ~15) | 7) * order;
            const T helpA = sfacA[tid % (warp_size * 8)];
            const T helpB = sfacB[tid % (warp_size * 8)];
            const T helpC = sfacC[tid % (warp_size * 8)];
            const T helpD = sfacD[tid % (warp_size * 8)];
            const T helpE = sfacE[tid % (warp_size * 8)];
            const T helpF = sfacF[tid % (warp_size * 8)];
            const T helpG = sfacG[tid % (warp_size * 8)];
            const T helpH = sfacH[tid % (warp_size * 8)];
            const T helpI = sfacI[tid % (warp_size * 8)];
            const T helpJ = sfacJ[tid % (warp_size * 8)];
            const T helpK = sfacK[tid % (warp_size * 8)];
            const T helpL = sfacL[tid % (warp_size * 8)];
            const T helpM = sfacM[tid % (warp_size * 8)];
            const T helpN = sfacN[tid % (warp_size * 8)];
            const T helpO = sfacO[tid % (warp_size * 8)];
            const T helpP = sfacP[tid % (warp_size * 8)];
            #pragma unroll
            for (int v = 0; v < PLR_X; v++) {
                val[v] += helpA * spartc[cwarp + (v * delta + 0)];
                val[v] += helpB * spartc[cwarp + (v * delta + 1)];
                val[v] += helpC * spartc[cwarp + (v * delta + 2)];
                val[v] += helpD * spartc[cwarp + (v * delta + 3)];
                val[v] += helpE * spartc[cwarp + (v * delta + 4)];
                val[v] += helpF * spartc[cwarp + (v * delta + 5)];
                val[v] += helpG * spartc[cwarp + (v * delta + 6)];
                val[v] += helpH * spartc[cwarp + (v * delta + 7)];
                val[v] += helpI * spartc[cwarp + (v * delta + 8)];
                val[v] += helpJ * spartc[cwarp + (v * delta + 9)];
                val[v] += helpK * spartc[cwarp + (v * delta + 10)];
                val[v] += helpL * spartc[cwarp + (v * delta + 11)];
                val[v] += helpM * spartc[cwarp + (v * delta + 12)];
                val[v] += helpN * spartc[cwarp + (v * delta + 13)];
                val[v] += helpO * spartc[cwarp + (v * delta + 14)];
                val[v] += helpP * spartc[cwarp + (v * delta + 15)];
            }
            if constexpr (num_warps > 16) {
                if ((warp == 15) && (clane >= 0)) {
                    #pragma unroll
                    for (int v = 0; v < PLR_X; v++) {
                        spartc[clane + (15 * order + v * delta)] = val[v];
                    }
                }
            } else {
                if ((warp == last_warp) && (clane >= 0)) {
                    #pragma unroll
                    for (int v = 0; v < PLR_X; v++) {
                        spartc[clane + (last_warp * order + v * delta)] = val[v];
                    }
                }
            }
        }
    }

    if constexpr (num_warps == 32) {
        __syncthreads();
        if ((warp & 16) != 0) {
            const int cwarp = 15 * order;
            const T helpA = sfacA[tid % (warp_size * 16)];
            const T helpB = sfacB[tid % (warp_size * 16)];
            const T helpC = sfacC[tid % (warp_size * 16)];
            const T helpD = sfacD[tid % (warp_size * 16)];
            const T helpE = sfacE[tid % (warp_size * 16)];
            const T helpF = sfacF[tid % (warp_size * 16)];
            const T helpG = sfacG[tid % (warp_size * 16)];
            const T helpH = sfacH[tid % (warp_size * 16)];
            const T helpI = sfacI[tid % (warp_size * 16)];
            const T helpJ = sfacJ[tid % (warp_size * 16)];
            const T helpK = sfacK[tid % (warp_size * 16)];
            const T helpL = sfacL[tid % (warp_size * 16)];
            const T helpM = sfacM[tid % (warp_size * 16)];
            const T helpN = sfacN[tid % (warp_size * 16)];
            const T helpO = sfacO[tid % (warp_size * 16)];
            const T helpP = sfacP[tid % (warp_size * 16)];
            #pragma unroll
            for (int v = 0; v < PLR_X; v++) {
                val[v] += helpA * spartc[cwarp + (v * delta + 0)];
                val[v] += helpB * spartc[cwarp + (v * delta + 1)];
                val[v] += helpC * spartc[cwarp + (v * delta + 2)];
                val[v] += helpD * spartc[cwarp + (v * delta + 3)];
                val[v] += helpE * spartc[cwarp + (v * delta + 4)];
                val[v] += helpF * spartc[cwarp + (v * delta + 5)];
                val[v] += helpG * spartc[cwarp + (v * delta + 6)];
                val[v] += helpH * spartc[cwarp + (v * delta + 7)];
                val[v] += helpI * spartc[cwarp + (v * delta + 8)];
                val[v] += helpJ * spartc[cwarp + (v * delta + 9)];
                val[v] += helpK * spartc[cwarp + (v * delta + 10)];
                val[v] += helpL * spartc[cwarp + (v * delta + 11)];
                val[v] += helpM * spartc[cwarp + (v * delta + 12)];
                val[v] += helpN * spartc[cwarp + (v * delta + 13)];
                val[v] += helpO * spartc[cwarp + (v * delta + 14)];
                val[v] += helpP * spartc[cwarp + (v * delta + 15)];
            }
            if ((warp == last_warp) && (clane >= 0)) {
                #pragma unroll
                for (int v = 0; v < PLR_X; v++) {
                    spartc[clane + (last_warp * order + v * delta)] = val[v];
                }
            }
        }
    }

    if constexpr (PLR_X > 1) {
        if ((warp == last_warp) && (clane >= 0)) {
            spartc[clane + (last_warp * order + 0 * delta)] = val[0];
        }
        __syncthreads();
        
        #pragma unroll
        for (int v = 1; v < PLR_X; v++) {
            val[v] += sfacA[tid] * spartc[last_warp * order + ((v-1) * delta + 0)];
            val[v] += sfacB[tid] * spartc[last_warp * order + ((v-1) * delta + 1)];
            val[v] += sfacC[tid] * spartc[last_warp * order + ((v-1) * delta + 2)];
            val[v] += sfacD[tid] * spartc[last_warp * order + ((v-1) * delta + 3)];
            val[v] += sfacE[tid] * spartc[last_warp * order + ((v-1) * delta + 4)];
            val[v] += sfacF[tid] * spartc[last_warp * order + ((v-1) * delta + 5)];
            val[v] += sfacG[tid] * spartc[last_warp * order + ((v-1) * delta + 6)];
            val[v] += sfacH[tid] * spartc[last_warp * order + ((v-1) * delta + 7)];
            val[v] += sfacI[tid] * spartc[last_warp * order + ((v-1) * delta + 8)];
            val[v] += sfacJ[tid] * spartc[last_warp * order + ((v-1) * delta + 9)];
            val[v] += sfacK[tid] * spartc[last_warp * order + ((v-1) * delta + 10)];
            val[v] += sfacL[tid] * spartc[last_warp * order + ((v-1) * delta + 11)];
            val[v] += sfacM[tid] * spartc[last_warp * order + ((v-1) * delta + 12)];
            val[v] += sfacN[tid] * spartc[last_warp * order + ((v-1) * delta + 13)];
            val[v] += sfacO[tid] * spartc[last_warp * order + ((v-1) * delta + 14)];
            val[v] += sfacP[tid] * spartc[last_warp * order + ((v-1) * delta + 15)];
            
            if ((warp == last_warp) && (clane >= 0)) {
                spartc[clane + (last_warp * order + v * delta)] = val[v];
            }
            
            if (v < PLR_X - 1) {
                __syncthreads();
            }
        }
    }

    // phase 3a: inter-block carry propagation 
    const int idx = tid - (PLR_BLOCK_SIZE - order);
    const int last_val_idx = PLR_X - 1;

    if (idx >= 0) {
        partcarry[chunk_id * order + idx] = val[last_val_idx];
    }
    __syncwarp();
    __threadfence();
    if (idx == 0) {
        status[chunk_id] = part_flag;
    }

    __syncthreads();

    if (warp == 0) {
        int flag = part_flag;
        bool no_zeros, has_status_2, reached_origin;
        
        do {
            if (chunk_id > lane) {
                flag = status[chunk_id - 1 - lane];
            }
            no_zeros = !__any_sync(0xffffffff, flag < part_flag);
            has_status_2 = !__all_sync(0xffffffff, flag != full_flag) && no_zeros;
            reached_origin = (chunk_id < warp_size) && no_zeros;
        } while (!(has_status_2 || reached_origin));
        
        __threadfence();
        
        int mask = __ballot_sync(0xffffffff, flag == full_flag);
        
        T X0, X1, X2, X3, X4, X5, X6, X7;
        T X8, X9, X10, X11, X12, X13, X14, X15;
        int start_chunk;
        
        if (mask == 0) {
            X0 = yi16; X1 = yi15; X2 = yi14; X3 = yi13;
            X4 = yi12; X5 = yi11; X6 = yi10; X7 = yi9;
            X8 = yi8;  X9 = yi7;  X10 = yi6; X11 = yi5;
            X12 = yi4; X13 = yi3; X14 = yi2; X15 = yi1;
            start_chunk = 0;
        } else {
            const int pos = __ffs(mask) - 1;
            const int full_chunk = chunk_id - 1 - pos;
            start_chunk = full_chunk + 1;
            
            T fc;
            if (lane < order) {
                fc = fullcarry[full_chunk * order + lane];
            }
            X0 = __shfl_sync(0xffffffff, fc, 0);
            X1 = __shfl_sync(0xffffffff, fc, 1);
            X2 = __shfl_sync(0xffffffff, fc, 2);
            X3 = __shfl_sync(0xffffffff, fc, 3);
            X4 = __shfl_sync(0xffffffff, fc, 4);
            X5 = __shfl_sync(0xffffffff, fc, 5);
            X6 = __shfl_sync(0xffffffff, fc, 6);
            X7 = __shfl_sync(0xffffffff, fc, 7);
            X8 = __shfl_sync(0xffffffff, fc, 8);
            X9 = __shfl_sync(0xffffffff, fc, 9);
            X10 = __shfl_sync(0xffffffff, fc, 10);
            X11 = __shfl_sync(0xffffffff, fc, 11);
            X12 = __shfl_sync(0xffffffff, fc, 12);
            X13 = __shfl_sync(0xffffffff, fc, 13);
            X14 = __shfl_sync(0xffffffff, fc, 14);
            X15 = __shfl_sync(0xffffffff, fc, 15);
        }
        
        const int num_partcarries = chunk_id - start_chunk;
        
        if (num_partcarries > 0) {
            for (int i = (start_chunk * order + lane); i < (chunk_id * order); i += warp_size) {
                spartc[i - start_chunk * order] = partcarry[i];
            }
            __syncwarp();
            
            if (lane == 0) {
                for (int i = 0; i < num_partcarries; i++) {
                    const T h0 = spartc[i * order + 0] 
                               + X0 * facA[PLR_CHUNK_SIZE-16] + X1 * facB[PLR_CHUNK_SIZE-16] 
                               + X2 * facC[PLR_CHUNK_SIZE-16] + X3 * facD[PLR_CHUNK_SIZE-16]
                               + X4 * facE[PLR_CHUNK_SIZE-16] + X5 * facF[PLR_CHUNK_SIZE-16]
                               + X6 * facG[PLR_CHUNK_SIZE-16] + X7 * facH[PLR_CHUNK_SIZE-16]
                               + X8 * facI[PLR_CHUNK_SIZE-16] + X9 * facJ[PLR_CHUNK_SIZE-16]
                               + X10 * facK[PLR_CHUNK_SIZE-16] + X11 * facL[PLR_CHUNK_SIZE-16]
                               + X12 * facM[PLR_CHUNK_SIZE-16] + X13 * facN[PLR_CHUNK_SIZE-16]
                               + X14 * facO[PLR_CHUNK_SIZE-16] + X15 * facP[PLR_CHUNK_SIZE-16];
                    const T h1 = spartc[i * order + 1] 
                               + X0 * facA[PLR_CHUNK_SIZE-15] + X1 * facB[PLR_CHUNK_SIZE-15] 
                               + X2 * facC[PLR_CHUNK_SIZE-15] + X3 * facD[PLR_CHUNK_SIZE-15]
                               + X4 * facE[PLR_CHUNK_SIZE-15] + X5 * facF[PLR_CHUNK_SIZE-15]
                               + X6 * facG[PLR_CHUNK_SIZE-15] + X7 * facH[PLR_CHUNK_SIZE-15]
                               + X8 * facI[PLR_CHUNK_SIZE-15] + X9 * facJ[PLR_CHUNK_SIZE-15]
                               + X10 * facK[PLR_CHUNK_SIZE-15] + X11 * facL[PLR_CHUNK_SIZE-15]
                               + X12 * facM[PLR_CHUNK_SIZE-15] + X13 * facN[PLR_CHUNK_SIZE-15]
                               + X14 * facO[PLR_CHUNK_SIZE-15] + X15 * facP[PLR_CHUNK_SIZE-15];
                    const T h2 = spartc[i * order + 2] 
                               + X0 * facA[PLR_CHUNK_SIZE-14] + X1 * facB[PLR_CHUNK_SIZE-14] 
                               + X2 * facC[PLR_CHUNK_SIZE-14] + X3 * facD[PLR_CHUNK_SIZE-14]
                               + X4 * facE[PLR_CHUNK_SIZE-14] + X5 * facF[PLR_CHUNK_SIZE-14]
                               + X6 * facG[PLR_CHUNK_SIZE-14] + X7 * facH[PLR_CHUNK_SIZE-14]
                               + X8 * facI[PLR_CHUNK_SIZE-14] + X9 * facJ[PLR_CHUNK_SIZE-14]
                               + X10 * facK[PLR_CHUNK_SIZE-14] + X11 * facL[PLR_CHUNK_SIZE-14]
                               + X12 * facM[PLR_CHUNK_SIZE-14] + X13 * facN[PLR_CHUNK_SIZE-14]
                               + X14 * facO[PLR_CHUNK_SIZE-14] + X15 * facP[PLR_CHUNK_SIZE-14];
                    const T h3 = spartc[i * order + 3] 
                               + X0 * facA[PLR_CHUNK_SIZE-13] + X1 * facB[PLR_CHUNK_SIZE-13] 
                               + X2 * facC[PLR_CHUNK_SIZE-13] + X3 * facD[PLR_CHUNK_SIZE-13]
                               + X4 * facE[PLR_CHUNK_SIZE-13] + X5 * facF[PLR_CHUNK_SIZE-13]
                               + X6 * facG[PLR_CHUNK_SIZE-13] + X7 * facH[PLR_CHUNK_SIZE-13]
                               + X8 * facI[PLR_CHUNK_SIZE-13] + X9 * facJ[PLR_CHUNK_SIZE-13]
                               + X10 * facK[PLR_CHUNK_SIZE-13] + X11 * facL[PLR_CHUNK_SIZE-13]
                               + X12 * facM[PLR_CHUNK_SIZE-13] + X13 * facN[PLR_CHUNK_SIZE-13]
                               + X14 * facO[PLR_CHUNK_SIZE-13] + X15 * facP[PLR_CHUNK_SIZE-13];
                    const T h4 = spartc[i * order + 4] 
                               + X0 * facA[PLR_CHUNK_SIZE-12] + X1 * facB[PLR_CHUNK_SIZE-12] 
                               + X2 * facC[PLR_CHUNK_SIZE-12] + X3 * facD[PLR_CHUNK_SIZE-12]
                               + X4 * facE[PLR_CHUNK_SIZE-12] + X5 * facF[PLR_CHUNK_SIZE-12]
                               + X6 * facG[PLR_CHUNK_SIZE-12] + X7 * facH[PLR_CHUNK_SIZE-12]
                               + X8 * facI[PLR_CHUNK_SIZE-12] + X9 * facJ[PLR_CHUNK_SIZE-12]
                               + X10 * facK[PLR_CHUNK_SIZE-12] + X11 * facL[PLR_CHUNK_SIZE-12]
                               + X12 * facM[PLR_CHUNK_SIZE-12] + X13 * facN[PLR_CHUNK_SIZE-12]
                               + X14 * facO[PLR_CHUNK_SIZE-12] + X15 * facP[PLR_CHUNK_SIZE-12];
                    const T h5 = spartc[i * order + 5] 
                               + X0 * facA[PLR_CHUNK_SIZE-11] + X1 * facB[PLR_CHUNK_SIZE-11] 
                               + X2 * facC[PLR_CHUNK_SIZE-11] + X3 * facD[PLR_CHUNK_SIZE-11]
                               + X4 * facE[PLR_CHUNK_SIZE-11] + X5 * facF[PLR_CHUNK_SIZE-11]
                               + X6 * facG[PLR_CHUNK_SIZE-11] + X7 * facH[PLR_CHUNK_SIZE-11]
                               + X8 * facI[PLR_CHUNK_SIZE-11] + X9 * facJ[PLR_CHUNK_SIZE-11]
                               + X10 * facK[PLR_CHUNK_SIZE-11] + X11 * facL[PLR_CHUNK_SIZE-11]
                               + X12 * facM[PLR_CHUNK_SIZE-11] + X13 * facN[PLR_CHUNK_SIZE-11]
                               + X14 * facO[PLR_CHUNK_SIZE-11] + X15 * facP[PLR_CHUNK_SIZE-11];
                    const T h6 = spartc[i * order + 6] 
                               + X0 * facA[PLR_CHUNK_SIZE-10] + X1 * facB[PLR_CHUNK_SIZE-10] 
                               + X2 * facC[PLR_CHUNK_SIZE-10] + X3 * facD[PLR_CHUNK_SIZE-10]
                               + X4 * facE[PLR_CHUNK_SIZE-10] + X5 * facF[PLR_CHUNK_SIZE-10]
                               + X6 * facG[PLR_CHUNK_SIZE-10] + X7 * facH[PLR_CHUNK_SIZE-10]
                               + X8 * facI[PLR_CHUNK_SIZE-10] + X9 * facJ[PLR_CHUNK_SIZE-10]
                               + X10 * facK[PLR_CHUNK_SIZE-10] + X11 * facL[PLR_CHUNK_SIZE-10]
                               + X12 * facM[PLR_CHUNK_SIZE-10] + X13 * facN[PLR_CHUNK_SIZE-10]
                               + X14 * facO[PLR_CHUNK_SIZE-10] + X15 * facP[PLR_CHUNK_SIZE-10];
                    const T h7 = spartc[i * order + 7] 
                               + X0 * facA[PLR_CHUNK_SIZE-9] + X1 * facB[PLR_CHUNK_SIZE-9] 
                               + X2 * facC[PLR_CHUNK_SIZE-9] + X3 * facD[PLR_CHUNK_SIZE-9]
                               + X4 * facE[PLR_CHUNK_SIZE-9] + X5 * facF[PLR_CHUNK_SIZE-9]
                               + X6 * facG[PLR_CHUNK_SIZE-9] + X7 * facH[PLR_CHUNK_SIZE-9]
                               + X8 * facI[PLR_CHUNK_SIZE-9] + X9 * facJ[PLR_CHUNK_SIZE-9]
                               + X10 * facK[PLR_CHUNK_SIZE-9] + X11 * facL[PLR_CHUNK_SIZE-9]
                               + X12 * facM[PLR_CHUNK_SIZE-9] + X13 * facN[PLR_CHUNK_SIZE-9]
                               + X14 * facO[PLR_CHUNK_SIZE-9] + X15 * facP[PLR_CHUNK_SIZE-9];
                    const T h8 = spartc[i * order + 8] 
                               + X0 * facA[PLR_CHUNK_SIZE-8] + X1 * facB[PLR_CHUNK_SIZE-8] 
                               + X2 * facC[PLR_CHUNK_SIZE-8] + X3 * facD[PLR_CHUNK_SIZE-8]
                               + X4 * facE[PLR_CHUNK_SIZE-8] + X5 * facF[PLR_CHUNK_SIZE-8]
                               + X6 * facG[PLR_CHUNK_SIZE-8] + X7 * facH[PLR_CHUNK_SIZE-8]
                               + X8 * facI[PLR_CHUNK_SIZE-8] + X9 * facJ[PLR_CHUNK_SIZE-8]
                               + X10 * facK[PLR_CHUNK_SIZE-8] + X11 * facL[PLR_CHUNK_SIZE-8]
                               + X12 * facM[PLR_CHUNK_SIZE-8] + X13 * facN[PLR_CHUNK_SIZE-8]
                               + X14 * facO[PLR_CHUNK_SIZE-8] + X15 * facP[PLR_CHUNK_SIZE-8];
                    const T h9 = spartc[i * order + 9] 
                               + X0 * facA[PLR_CHUNK_SIZE-7] + X1 * facB[PLR_CHUNK_SIZE-7] 
                               + X2 * facC[PLR_CHUNK_SIZE-7] + X3 * facD[PLR_CHUNK_SIZE-7]
                               + X4 * facE[PLR_CHUNK_SIZE-7] + X5 * facF[PLR_CHUNK_SIZE-7]
                               + X6 * facG[PLR_CHUNK_SIZE-7] + X7 * facH[PLR_CHUNK_SIZE-7]
                               + X8 * facI[PLR_CHUNK_SIZE-7] + X9 * facJ[PLR_CHUNK_SIZE-7]
                               + X10 * facK[PLR_CHUNK_SIZE-7] + X11 * facL[PLR_CHUNK_SIZE-7]
                               + X12 * facM[PLR_CHUNK_SIZE-7] + X13 * facN[PLR_CHUNK_SIZE-7]
                               + X14 * facO[PLR_CHUNK_SIZE-7] + X15 * facP[PLR_CHUNK_SIZE-7];
                    const T h10 = spartc[i * order + 10] 
                               + X0 * facA[PLR_CHUNK_SIZE-6] + X1 * facB[PLR_CHUNK_SIZE-6] 
                               + X2 * facC[PLR_CHUNK_SIZE-6] + X3 * facD[PLR_CHUNK_SIZE-6]
                               + X4 * facE[PLR_CHUNK_SIZE-6] + X5 * facF[PLR_CHUNK_SIZE-6]
                               + X6 * facG[PLR_CHUNK_SIZE-6] + X7 * facH[PLR_CHUNK_SIZE-6]
                               + X8 * facI[PLR_CHUNK_SIZE-6] + X9 * facJ[PLR_CHUNK_SIZE-6]
                               + X10 * facK[PLR_CHUNK_SIZE-6] + X11 * facL[PLR_CHUNK_SIZE-6]
                               + X12 * facM[PLR_CHUNK_SIZE-6] + X13 * facN[PLR_CHUNK_SIZE-6]
                               + X14 * facO[PLR_CHUNK_SIZE-6] + X15 * facP[PLR_CHUNK_SIZE-6];
                    const T h11 = spartc[i * order + 11] 
                               + X0 * facA[PLR_CHUNK_SIZE-5] + X1 * facB[PLR_CHUNK_SIZE-5] 
                               + X2 * facC[PLR_CHUNK_SIZE-5] + X3 * facD[PLR_CHUNK_SIZE-5]
                               + X4 * facE[PLR_CHUNK_SIZE-5] + X5 * facF[PLR_CHUNK_SIZE-5]
                               + X6 * facG[PLR_CHUNK_SIZE-5] + X7 * facH[PLR_CHUNK_SIZE-5]
                               + X8 * facI[PLR_CHUNK_SIZE-5] + X9 * facJ[PLR_CHUNK_SIZE-5]
                               + X10 * facK[PLR_CHUNK_SIZE-5] + X11 * facL[PLR_CHUNK_SIZE-5]
                               + X12 * facM[PLR_CHUNK_SIZE-5] + X13 * facN[PLR_CHUNK_SIZE-5]
                               + X14 * facO[PLR_CHUNK_SIZE-5] + X15 * facP[PLR_CHUNK_SIZE-5];
                    const T h12 = spartc[i * order + 12] 
                               + X0 * facA[PLR_CHUNK_SIZE-4] + X1 * facB[PLR_CHUNK_SIZE-4] 
                               + X2 * facC[PLR_CHUNK_SIZE-4] + X3 * facD[PLR_CHUNK_SIZE-4]
                               + X4 * facE[PLR_CHUNK_SIZE-4] + X5 * facF[PLR_CHUNK_SIZE-4]
                               + X6 * facG[PLR_CHUNK_SIZE-4] + X7 * facH[PLR_CHUNK_SIZE-4]
                               + X8 * facI[PLR_CHUNK_SIZE-4] + X9 * facJ[PLR_CHUNK_SIZE-4]
                               + X10 * facK[PLR_CHUNK_SIZE-4] + X11 * facL[PLR_CHUNK_SIZE-4]
                               + X12 * facM[PLR_CHUNK_SIZE-4] + X13 * facN[PLR_CHUNK_SIZE-4]
                               + X14 * facO[PLR_CHUNK_SIZE-4] + X15 * facP[PLR_CHUNK_SIZE-4];
                    const T h13 = spartc[i * order + 13] 
                               + X0 * facA[PLR_CHUNK_SIZE-3] + X1 * facB[PLR_CHUNK_SIZE-3] 
                               + X2 * facC[PLR_CHUNK_SIZE-3] + X3 * facD[PLR_CHUNK_SIZE-3]
                               + X4 * facE[PLR_CHUNK_SIZE-3] + X5 * facF[PLR_CHUNK_SIZE-3]
                               + X6 * facG[PLR_CHUNK_SIZE-3] + X7 * facH[PLR_CHUNK_SIZE-3]
                               + X8 * facI[PLR_CHUNK_SIZE-3] + X9 * facJ[PLR_CHUNK_SIZE-3]
                               + X10 * facK[PLR_CHUNK_SIZE-3] + X11 * facL[PLR_CHUNK_SIZE-3]
                               + X12 * facM[PLR_CHUNK_SIZE-3] + X13 * facN[PLR_CHUNK_SIZE-3]
                               + X14 * facO[PLR_CHUNK_SIZE-3] + X15 * facP[PLR_CHUNK_SIZE-3];
                    const T h14 = spartc[i * order + 14] 
                               + X0 * facA[PLR_CHUNK_SIZE-2] + X1 * facB[PLR_CHUNK_SIZE-2] 
                               + X2 * facC[PLR_CHUNK_SIZE-2] + X3 * facD[PLR_CHUNK_SIZE-2]
                               + X4 * facE[PLR_CHUNK_SIZE-2] + X5 * facF[PLR_CHUNK_SIZE-2]
                               + X6 * facG[PLR_CHUNK_SIZE-2] + X7 * facH[PLR_CHUNK_SIZE-2]
                               + X8 * facI[PLR_CHUNK_SIZE-2] + X9 * facJ[PLR_CHUNK_SIZE-2]
                               + X10 * facK[PLR_CHUNK_SIZE-2] + X11 * facL[PLR_CHUNK_SIZE-2]
                               + X12 * facM[PLR_CHUNK_SIZE-2] + X13 * facN[PLR_CHUNK_SIZE-2]
                               + X14 * facO[PLR_CHUNK_SIZE-2] + X15 * facP[PLR_CHUNK_SIZE-2];
                    const T h15 = spartc[i * order + 15] 
                               + X0 * facA[PLR_CHUNK_SIZE-1] + X1 * facB[PLR_CHUNK_SIZE-1] 
                               + X2 * facC[PLR_CHUNK_SIZE-1] + X3 * facD[PLR_CHUNK_SIZE-1]
                               + X4 * facE[PLR_CHUNK_SIZE-1] + X5 * facF[PLR_CHUNK_SIZE-1]
                               + X6 * facG[PLR_CHUNK_SIZE-1] + X7 * facH[PLR_CHUNK_SIZE-1]
                               + X8 * facI[PLR_CHUNK_SIZE-1] + X9 * facJ[PLR_CHUNK_SIZE-1]
                               + X10 * facK[PLR_CHUNK_SIZE-1] + X11 * facL[PLR_CHUNK_SIZE-1]
                               + X12 * facM[PLR_CHUNK_SIZE-1] + X13 * facN[PLR_CHUNK_SIZE-1]
                               + X14 * facO[PLR_CHUNK_SIZE-1] + X15 * facP[PLR_CHUNK_SIZE-1];
                    X0 = h0; X1 = h1; X2 = h2; X3 = h3;
                    X4 = h4; X5 = h5; X6 = h6; X7 = h7;
                    X8 = h8; X9 = h9; X10 = h10; X11 = h11;
                    X12 = h12; X13 = h13; X14 = h14; X15 = h15;
                }
            }
        }
        
        if (lane == 0) {
            sfullc[0] = X0;   sfullc[1] = X1;   sfullc[2] = X2;   sfullc[3] = X3;
            sfullc[4] = X4;   sfullc[5] = X5;   sfullc[6] = X6;   sfullc[7] = X7;
            sfullc[8] = X8;   sfullc[9] = X9;   sfullc[10] = X10; sfullc[11] = X11;
            sfullc[12] = X12; sfullc[13] = X13; sfullc[14] = X14; sfullc[15] = X15;
        }
    }

    __syncthreads();

    // phase 3b: apply carry to all values
    T X0 = sfullc[0];   T X1 = sfullc[1];   T X2 = sfullc[2];   T X3 = sfullc[3];
    T X4 = sfullc[4];   T X5 = sfullc[5];   T X6 = sfullc[6];   T X7 = sfullc[7];
    T X8 = sfullc[8];   T X9 = sfullc[9];   T X10 = sfullc[10]; T X11 = sfullc[11];
    T X12 = sfullc[12]; T X13 = sfullc[13]; T X14 = sfullc[14]; T X15 = sfullc[15];

    #pragma unroll
    for (int v = 0; v < PLR_X; v++) {
        val[v] += facA[tid + v * PLR_BLOCK_SIZE] * X0;
        val[v] += facB[tid + v * PLR_BLOCK_SIZE] * X1;
        val[v] += facC[tid + v * PLR_BLOCK_SIZE] * X2;
        val[v] += facD[tid + v * PLR_BLOCK_SIZE] * X3;
        val[v] += facE[tid + v * PLR_BLOCK_SIZE] * X4;
        val[v] += facF[tid + v * PLR_BLOCK_SIZE] * X5;
        val[v] += facG[tid + v * PLR_BLOCK_SIZE] * X6;
        val[v] += facH[tid + v * PLR_BLOCK_SIZE] * X7;
        val[v] += facI[tid + v * PLR_BLOCK_SIZE] * X8;
        val[v] += facJ[tid + v * PLR_BLOCK_SIZE] * X9;
        val[v] += facK[tid + v * PLR_BLOCK_SIZE] * X10;
        val[v] += facL[tid + v * PLR_BLOCK_SIZE] * X11;
        val[v] += facM[tid + v * PLR_BLOCK_SIZE] * X12;
        val[v] += facN[tid + v * PLR_BLOCK_SIZE] * X13;
        val[v] += facO[tid + v * PLR_BLOCK_SIZE] * X14;
        val[v] += facP[tid + v * PLR_BLOCK_SIZE] * X15;
    }

    if (idx >= 0) {
        fullcarry[chunk_id * order + idx] = val[last_val_idx];
    }
    __syncwarp();
    __threadfence();
    if (idx == 0) {
        status[chunk_id] = full_flag;
    }

    if (chunk_id == gridDim.x - 1) {
        #pragma unroll
        for (int v = 0; v < PLR_X; v++) {
            if (offs + (v * PLR_BLOCK_SIZE) < N_SAMPLES) {
                output[offs + (v * PLR_BLOCK_SIZE)] = val[v];
            }
        }
    } else {
        #pragma unroll
        for (int v = 0; v < PLR_X; v++) {
            output[offs + (v * PLR_BLOCK_SIZE)] = val[v];
        }
    }
}
#endif

// ---- driver-facing macros -------------------------------------------------
#if PLR_ORDER == 2
    #define KERNEL_FUNC PLR_2
#elif PLR_ORDER == 4
    #define KERNEL_FUNC PLR_4
#elif PLR_ORDER == 8
    #define KERNEL_FUNC PLR_8
#elif PLR_ORDER == 16
    #define KERNEL_FUNC PLR_16
#endif

#define KERNEL_TB_DIM     PLR_BLOCK_SIZE
#define KERNEL_GRID_DIM   PLR_N_TB
#define KERNEL_NUM_CHUNKS PLR_N_TB

#endif // PLR_KERNELS_CUH
