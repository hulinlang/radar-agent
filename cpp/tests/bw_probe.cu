#include <cstdio>
#include <cuda_runtime.h>

int main() {
    const size_t bytes = 1ull << 30;  // 1 GiB
    void *src, *dst;
    cudaMalloc(&src, bytes);
    cudaMalloc(&dst, bytes);
    cudaMemset(src, 1, bytes);
    cudaMemset(dst, 0, bytes);
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    // warmup
    cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
    cudaEventRecord(a);
    const int iters = 20;
    for (int i = 0; i < iters; ++i) cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice);
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0;
    cudaEventElapsedTime(&ms, a, b);
    const double sec = ms / 1000.0 / iters;
    // D2D copy reads+writes -> traffic = 2x bytes
    std::printf("WSL2 D2D bandwidth: %.1f GB/s (%.3f ms/GiB)\n", 2.0 * bytes / sec / 1e9, sec * 1000);
    return 0;
}
