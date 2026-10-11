#include <cstdio>
#include <cuda_runtime.h>
#include <cublas_v2.h>

int main() {
    cublasHandle_t h;
    cublasStatus_t s = cublasCreate(&h);
    std::printf("create: %d\n", s);

    float *A, *x, *y;
    cudaMalloc(&A, 4 * 4 * 4);
    cudaMalloc(&x, 4 * 4);
    cudaMalloc(&y, 4 * 4);
    float ha[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    float hx[4] = {1, 2, 3, 4};
    cudaMemcpy(A, ha, 64, cudaMemcpyHostToDevice);
    cudaMemcpy(x, hx, 16, cudaMemcpyHostToDevice);
    float a = 1.f, b = 0.f;
    s = cublasSgemv(h, CUBLAS_OP_N, 4, 4, &a, A, 4, x, 1, &b, y, 1);
    std::printf("sgemv f32: %d\n", s);

    __half *Ah, *xh;
    cudaMalloc(&Ah, 4 * 4 * 2);
    cudaMalloc(&xh, 4 * 2);
    float alpha = 1.f, beta = 0.f;
    s = cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, 4, 1, 4, &alpha,
                     Ah, CUDA_R_16F, 4, xh, CUDA_R_16F, 4, &beta,
                     y, CUDA_R_32F, 4, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    std::printf("gemmEx f16: %d\n", s);
    std::printf("cuda err: %s\n", cudaGetErrorString(cudaGetLastError()));
    return 0;
}
