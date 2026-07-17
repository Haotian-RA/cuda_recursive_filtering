#ifndef MEASURE_CUH
#define MEASURE_CUH

// Kernel timing infrastructure.
//
// Provides:
//   keep_alive_kernel — dummy kernel to keep the GPU busy between measurements
//   measure_kernel_time<LaunchFunc> — templated timing with buffer rotation
//
// The buffer rotation is added specifically to defeat L2 cache retention
// across launches. Without rotation, the same input buffer is read on every
// iteration and stays warm in L2, artificially inflating measured throughput
// at small batch sizes. By rotating through multiple identical input buffers,
// each launch sees data that was recently evicted from L2 by other buffers'
// traffic — matching the "streaming from DRAM" behavior of real workloads.
//
// Output format is preserved from the original main_PH.cu measurement code.

#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <algorithm>


__global__ void keep_alive_kernel(float* dummy, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dummy[idx] = sinf(dummy[idx]) * cosf(dummy[idx]);
    }
}


// LaunchFunc must be callable as `launch(T* d_in)` where d_in is one of the
// rotated input buffers. The caller sets up buffers with identical contents
// so that verification against reference.bin works regardless of which
// buffer was used last.
template<typename LaunchFunc>
float measure_kernel_time(LaunchFunc launch,
                          const std::vector<float*>& input_buffers,
                          int iterations = 4000,
                          int warmup_iterations = 1000) {

    // Dummy buffer for keep-alive
    float* d_dummy;
    int dummy_size = 1 << 20;   // 1M elements
    cudaMalloc(&d_dummy, dummy_size * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int num_buffers = input_buffers.size();

    // Warmup — get GPU to stable boost clock
    printf("[Timing] Warming up GPU...\n");
    for (int i = 0; i < warmup_iterations; i++) {
        launch(input_buffers[i % num_buffers]);
        if (i % 100 == 0) {
            keep_alive_kernel<<<256, 256>>>(d_dummy, dummy_size);
        }
    }
    cudaDeviceSynchronize();
    printf("[Timing] Warmup complete, starting measurements...\n");

    // Batched measurements with keep-alive between batches
    constexpr int BATCH_SIZE = 200;
    int num_batches = iterations / BATCH_SIZE;
    std::vector<float> batch_times(num_batches);

    int global_iter = 0;
    for (int b = 0; b < num_batches; b++) {
        keep_alive_kernel<<<256, 256>>>(d_dummy, dummy_size);
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        for (int i = 0; i < BATCH_SIZE; i++) {
            launch(input_buffers[global_iter % num_buffers]);
            global_iter++;
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float batch_time;
        cudaEventElapsedTime(&batch_time, start, stop);
        batch_times[b] = batch_time / BATCH_SIZE;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_dummy);

    // Sort for percentile stats
    std::sort(batch_times.begin(), batch_times.end());

    int p20_idx = num_batches / 5;
    if (p20_idx < 1) p20_idx = 0;
    float p20 = batch_times[p20_idx];
    float median = batch_times[num_batches / 2];
    float min_time = batch_times.front();
    float max_time = batch_times.back();

    printf("[Timing] Results over %d batches (x%d each):\n", num_batches, BATCH_SIZE);
    printf("[Timing]   Min (peak boost): %.4f ms\n", min_time);
    printf("[Timing]   P20 (stable):     %.4f ms\n", p20);
    printf("[Timing]   Median:           %.4f ms\n", median);
    printf("[Timing]   Max (throttled):  %.4f ms\n", max_time);
    printf("[Timing]   Spread:           %.1fx\n", max_time / min_time);

    return p20;
}

#endif // MEASURE_CUH
