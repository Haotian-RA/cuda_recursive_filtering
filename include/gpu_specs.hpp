#ifndef GPU_SPECS_HPP
#define GPU_SPECS_HPP

// GPU hardware specs and occupancy calculation.
//
// Profile selection (compile time):
//   default          ->  RTX 3060  (Ampere GA106, compute capability 8.6)
//   -DGPU_GTX1070    ->  GTX 1070  (Pascal GP104, compute capability 6.1)
//
// Keeping both profiles in the source makes each build's provenance
// explicit: the test drivers print gpu_specs::GPU_NAME at startup.
// To port to another architecture, add a profile below.

namespace gpu_specs {

#ifdef GPU_GTX1070
// -------------------------------------------------------------
// GTX 1070 hardware constants (Pascal GP104, CC 6.1)
// -------------------------------------------------------------
constexpr const char* GPU_NAME          = "GTX 1070 (Pascal GP104)";
constexpr int NUM_SMS                   = 15;
constexpr int REGISTERS_PER_SM          = 65536;
constexpr int SHARED_MEM_PER_SM         = 96 * 1024;   // 96 KB
constexpr int MAX_SHARED_MEM_PER_TB     = 48 * 1024;   // 48 KB
constexpr int MAX_THREADS_PER_SM        = 2048;
constexpr int MAX_WARPS_PER_SM          = 64;
constexpr int MAX_TBS_PER_SM            = 32;
constexpr int MAX_THREADS_PER_TB        = 1024;
constexpr int MAX_REGISTERS_PER_THREAD  = 255;
constexpr int WARP_SIZE                 = 32;
#else
// -------------------------------------------------------------
// RTX 3060 hardware constants (Ampere GA106, CC 8.6) — default
// -------------------------------------------------------------
constexpr const char* GPU_NAME          = "RTX 3060 (Ampere GA106)";
constexpr int NUM_SMS                   = 28;
constexpr int REGISTERS_PER_SM          = 65536;
constexpr int SHARED_MEM_PER_SM         = 100 * 1024;  // 100 KB
constexpr int MAX_SHARED_MEM_PER_TB     = 48 * 1024;   // 48 KB (static)
constexpr int MAX_THREADS_PER_SM        = 1536;
constexpr int MAX_WARPS_PER_SM          = 48;
constexpr int MAX_TBS_PER_SM            = 16;
constexpr int MAX_THREADS_PER_TB        = 1024;
constexpr int MAX_REGISTERS_PER_THREAD  = 255;
constexpr int WARP_SIZE                 = 32;
#endif




// -------------------------------------------------------------
// compute_supply_tb_per_sm
//   Hardware-limited maximum TBs per SM (supply side).
//   Determined by shared memory, registers, thread count, warps, and the
//   hardware TB cap. Independent of batch size.
// -------------------------------------------------------------
constexpr int compute_supply_tb_per_sm(int block_size, int n_blocks) {
    const int smem_per_TB    = block_size * (n_blocks + 1) * 4;
    const int regs_per_TB    = n_blocks * block_size;
    const int warps_per_TB   = (block_size + WARP_SIZE - 1) / WARP_SIZE;

    const int tb_from_smem    = SHARED_MEM_PER_SM  / smem_per_TB;
    const int tb_from_regs    = REGISTERS_PER_SM   / regs_per_TB;
    const int tb_from_threads = MAX_THREADS_PER_SM / block_size;
    const int tb_from_warps   = MAX_WARPS_PER_SM   / warps_per_TB;
    const int tb_from_hw      = MAX_TBS_PER_SM;

    int result = tb_from_smem;
    if (tb_from_regs    < result) result = tb_from_regs;
    if (tb_from_threads < result) result = tb_from_threads;
    if (tb_from_warps   < result) result = tb_from_warps;
    if (tb_from_hw      < result) result = tb_from_hw;
    return result;
}


// -------------------------------------------------------------
// compute_demand_tb_per_sm
//   Batch-driven demand — number of TBs per SM when the launch's grid is
//   evenly distributed across the num_SMs. Uses ceiling division because
//   some SMs may get one more TB than others when total_TBs is not
//   divisible by num_SMs.
// -------------------------------------------------------------
constexpr int compute_demand_tb_per_sm(int block_size, int n_blocks, int batch_size) {
    const int total_TBs = batch_size / (block_size * n_blocks);
    return (total_TBs + NUM_SMS - 1) / NUM_SMS;   // ceiling
}


// -------------------------------------------------------------
// compute_n_tb_per_sm
//   Effective TB residency per SM = min(supply, demand).
//   Matches the notebook's `TBs_per_SM_effective` field. This is what
//   actually runs on the hardware.
//
//   block_size : threads per TB (= N_T = L)
//   n_blocks   : register-resident samples per thread (= N_reg = N)
//   batch_size : total number of samples in the launch (= N_SAMPLES)
// -------------------------------------------------------------
constexpr int compute_n_tb_per_sm(int block_size, int n_blocks, int batch_size) {
    const int supply = compute_supply_tb_per_sm(block_size, n_blocks);
    const int demand = compute_demand_tb_per_sm(block_size, n_blocks, batch_size);
    return (supply < demand) ? supply : demand;
}


// -------------------------------------------------------------
// Compile-time validity checks — independent of batch size.
// -------------------------------------------------------------
constexpr bool config_is_valid(int block_size, int n_blocks) {
    return
        (block_size * (n_blocks + 1) * 4) <= MAX_SHARED_MEM_PER_TB
        && block_size <= MAX_THREADS_PER_TB
        && n_blocks <= MAX_REGISTERS_PER_THREAD
        && compute_supply_tb_per_sm(block_size, n_blocks) >= 1;
}

} // namespace gpu_specs

#endif // GPU_SPECS_HPP
