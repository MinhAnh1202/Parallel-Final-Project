#pragma once
#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK_CUDA(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA Error %s:%d: %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));       \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

// NCHW layout: [N, C, H, W]
__device__ __host__ inline int idx4(int n, int c, int h, int w,
                                    int C, int H, int W) {
    return ((n * C + c) * H + h) * W + w;
}

// ==== KERNEL DECLARATIONS ====

__global__ void conv2d_forward_naive(
    const float* __restrict__ input,    // [N, C_in, H, W]
    const float* __restrict__ weight,   // [C_out, C_in, K, K]
    const float* __restrict__ bias,     // [C_out]
    float* __restrict__ output,         // [N, C_out, H_out, W_out]
    int N, int C_in, int H, int W,
    int C_out, int K, int pad, int stride);

__global__ void relu_forward(float* x, int size);

__global__ void maxpool2x2_forward(
    const float* __restrict__ input,  // [N, C, H, W]
    float* __restrict__ output,       // [N, C, H/2, W/2]
    int N, int C, int H, int W);

__global__ void upsample2x2_forward(
    const float* __restrict__ input,  // [N, C, H, W]
    float* __restrict__ output,       // [N, C, 2H, 2W]
    int N, int C, int H, int W);

__global__ void mse_loss_forward(
    const float* __restrict__ output,
    const float* __restrict__ target,
    float* __restrict__ loss,   // single float on device
    int size);
