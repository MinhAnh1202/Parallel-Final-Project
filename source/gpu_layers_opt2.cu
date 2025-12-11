%%writefile gpu_layers_opt2.cu
// ============================================================================
// OPTIMIZED CONV2D - FIX CRITICAL ISSUES
// Target: 12.83ms → 7-8ms (forward), similar for backward
// ============================================================================

#include <cuda_runtime.h>
#include "gpu_layers_opt2.h"


// Optimal configurations
#define TILE_W 16
#define TILE_H 16
#define K 3

void update_dc_bias(float* d_bias_ptr, int count) {
    CHECK_CUDA(cudaMemcpyToSymbol(dc_bias, d_bias_ptr, count * sizeof(float),
                                   0, cudaMemcpyDeviceToDevice));
}

// ============================================================================
// FORWARD PASS - FIXED
// Key fix: Efficient cooperative loading
// ============================================================================
__global__ void conv2d_forward_opt2(
    float* __restrict__ input,    // [N, C_in, H, W]
    float* __restrict__ weight,   // [C_out, C_in, K, K]
    float* __restrict__ output,   // [N, C_out, H_out, W_out]
    int N, int C_in, int H, int W,
    int C_out, int pad, int stride)
{
    int H_out = (H + 2 * pad - K) / stride + 1;
    int W_out = (W + 2 * pad - K) / stride + 1;

    __shared__ float smem[BLOCK_H][BLOCK_W];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;
    int num_threads = blockDim.x * blockDim.y;  // 256

    int w_out = blockIdx.x * TILE_W + tx;
    int h_out = blockIdx.y * TILE_H + ty;

    int nc = blockIdx.z;
    int c_out = nc % C_out;
    int n = nc / C_out;

    float value = 0.0f;

    // Base position for loading (với padding)
    int row_start = blockIdx.y * TILE_H * stride - pad;
    int col_start = blockIdx.x * TILE_W * stride - pad;

    // Loop over input channels
    for (int c_in = 0; c_in < C_in; c_in++) {
        // ✅ OPTIMIZED: Cooperative loading
        // Total elements = 18x18 = 324
        // 256 threads → mỗi thread load 1-2 elements
        int total_elements = BLOCK_H * BLOCK_W;

        for (int idx = tid; idx < total_elements; idx += num_threads) {
            int i = idx / BLOCK_W;  // row
            int j = idx % BLOCK_W;  // col

            int h_in = row_start + i;
            int w_in = col_start + j;

            // Load with bounds checking
            if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                smem[i][j] = input[idx4(n, c_in, h_in, w_in, C_in, H, W)];
            } else {
                smem[i][j] = 0.0f;
            }
        }
        __syncthreads();

        // ✅ OPTIMIZED: Compute convolution with unrolling
        if (h_out < H_out && w_out < W_out && n < N) {
            int smem_row = ty * stride;
            int smem_col = tx * stride;

            #pragma unroll
            for (int i = 0; i < K; ++i) {
                #pragma unroll
                for (int j = 0; j < K; ++j) {
                    float in_val = smem[smem_row + i][smem_col + j];
                    float w_val = weight[idx4(c_out, c_in, i, j, C_in, K, K)];
                    value += in_val * w_val;
                }
            }
        }
        __syncthreads();
    }

    // Write output with bias from constant memory
    if (w_out < W_out && h_out < H_out && n < N) {
        value += dc_bias[c_out];
        output[idx4(n, c_out, h_out, w_out, C_out, H_out, W_out)] = value;
    }
}

// ============================================================================
// BACKWARD INPUT - FIXED
// Key fix: Parallel halo loading
// ============================================================================

__global__ void conv2d_backward_input_opt2(
    float* __restrict__ dY,     // [N, C_out, H, W]
    float* __restrict__ weight, // [C_out, C_in, K, K]
    float* __restrict__ dX,     // [N, C_in, H, W]
    int N, int C_in, int H, int W,
    int C_out, int pad, int stride)
{
    // Only support K=3, pad=1, stride=1
    if (stride != 1 || pad != 1 || K != 3) return;

    const int H_out = H;
    const int W_out = W;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int base_h = blockIdx.y * TILE_H;
    int base_w = blockIdx.x * TILE_W;

    int h_in = base_h + ty;
    int w_in = base_w + tx;

    int nc = blockIdx.z;
    int c_in = nc % C_in;
    int n    = nc / C_in;

    if (n >= N || c_in >= C_in) return;

    // Shared memory for tile + halo
    __shared__ float s_dY[TILE_H + 2][TILE_W + 2];

    float value = 0.0f;

    for (int c_out = 0; c_out < C_out; ++c_out) {
        // ---- Load dY tile (n, c_out, :, :) into shared memory ----
        for (int sy = ty; sy < TILE_H + 2; sy += blockDim.y) {
            for (int sx = tx; sx < TILE_W + 2; sx += blockDim.x) {
                int ho = base_h + sy - 1;  // -pad
                int wo = base_w + sx - 1;  // -pad

                float v = 0.0f;
                if (ho >= 0 && ho < H_out && wo >= 0 && wo < W_out) {
                    v = dY[idx4(n, c_out, ho, wo, C_out, H_out, W_out)];
                }
                s_dY[sy][sx] = v;
            }
        }
        __syncthreads();

        // ---- Compute gradient for (n, c_in, h_in, w_in) ----
        if (h_in < H && w_in < W) {
            #pragma unroll
            for (int kh = 0; kh < K; ++kh) {
                #pragma unroll
                for (int kw = 0; kw < K; ++kw) {
                    // Shared memory indices
                    int sy = ty + 2 - kh;
                    int sx = tx + 2 - kw;

                    float dy = s_dY[sy][sx];

                    // CRITICAL FIX: Flip the kernel indices
                    int kh_flip = K - 1 - kh;  // 2, 1, 0
                    int kw_flip = K - 1 - kw;  // 2, 1, 0

                    // Weight layout [C_out, C_in, K, K]
                    float w = weight[idx4(c_out, c_in, kh_flip, kw_flip, C_in, K, K)];
                    value += dy * w;
                }
            }
        }

        __syncthreads();
    }

    if (h_in < H && w_in < W) {
        dX[idx4(n, c_in, h_in, w_in, C_in, H, W)] = value;
    }
}
// ============================================================================
// BACKWARD WEIGHT - FIXED
// Key fix: Block-level reduction BEFORE atomic
// ============================================================================
__global__ void conv2d_backward_weight_opt2(
    float* __restrict__ input,
    float* __restrict__ dY,
    float* __restrict__ dW,
    int N, int C_in, int H, int W,
    int C_out, int pad, int stride)
{
    int H_out = (H + 2 * pad - K) / stride + 1;
    int W_out = (W + 2 * pad - K) / stride + 1;

    __shared__ float s_in[BLOCK_H][BLOCK_W];
    __shared__ float s_dY[TILE_H][TILE_W];
    __shared__ float s_dw[K * K][256];  // Reduction buffer

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;
    int num_threads = blockDim.x * blockDim.y;

    int index = blockIdx.z;
    int c_in = index % C_in;
    int c_out = index / C_in;

    // Local accumulator
    float dw[K][K];
    #pragma unroll
    for (int i = 0; i < K; ++i)
        #pragma unroll
        for (int j = 0; j < K; ++j)
            dw[i][j] = 0.0f;

    for (int n = 0; n < N; ++n) {
        int num_blocks_h = (H_out + TILE_H - 1) / TILE_H;
        int num_blocks_w = (W_out + TILE_W - 1) / TILE_W;

        for (int block_h = 0; block_h < num_blocks_h; ++block_h) {
            for (int block_w = 0; block_w < num_blocks_w; ++block_w) {

                // ✅ FIX: Cooperative loading cho dY
                int h_out = block_h * TILE_H;
                int w_out = block_w * TILE_W;

                for (int idx = tid; idx < TILE_H * TILE_W; idx += num_threads) {
                    int i = idx / TILE_W;
                    int j = idx % TILE_W;
                    int h = h_out + i;
                    int w = w_out + j;

                    if (h < H_out && w < W_out) {
                        s_dY[i][j] = dY[idx4(n, c_out, h, w, C_out, H_out, W_out)];
                    } else {
                        s_dY[i][j] = 0.0f;
                    }
                }

                // ✅ FIX: Cooperative loading cho input với halo
                int h_in_base = block_h * TILE_H - pad;
                int w_in_base = block_w * TILE_W - pad;

                for (int idx = tid; idx < BLOCK_H * BLOCK_W; idx += num_threads) {
                    int i = idx / BLOCK_W;
                    int j = idx % BLOCK_W;
                    int h = h_in_base + i;
                    int w = w_in_base + j;

                    if (h >= 0 && h < H && w >= 0 && w < W) {
                        s_in[i][j] = input[idx4(n, c_in, h, w, C_in, H, W)];
                    } else {
                        s_in[i][j] = 0.0f;
                    }
                }
                __syncthreads();

                // Compute local dW
                if (ty < TILE_H && tx < TILE_W) {
                    float val_dy = s_dY[ty][tx];
                    #pragma unroll
                    for (int kh = 0; kh < K; ++kh) {
                        #pragma unroll
                        for (int kw = 0; kw < K; ++kw) {
                            dw[kh][kw] += s_in[ty * stride + kh][tx * stride + kw] * val_dy;
                        }
                    }
                }
                __syncthreads();
            }
        }
    }

    // ✅ FIX: Block-level reduction TRƯỚC atomic
    // Mỗi kernel element có reduction riêng
    #pragma unroll
    for (int kh = 0; kh < K; ++kh) {
        #pragma unroll
        for (int kw = 0; kw < K; ++kw) {
            int k_idx = kh * K + kw;

            // Store to shared memory
            s_dw[k_idx][tid] = dw[kh][kw];
            __syncthreads();

            // Tree reduction
            for (int s = num_threads / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    s_dw[k_idx][tid] += s_dw[k_idx][tid + s];
                }
                __syncthreads();
            }

            // ONLY thread 0 does atomic
            if (tid == 0) {
                size_t dw_idx = idx4(c_out, c_in, kh, kw, C_in, K, K);
                atomicAdd(&dW[dw_idx], s_dw[k_idx][0]);
            }
        }
    }
}

// ============================================================================
// BACKWARD BIAS - OPTIMIZED (warp-level reduction)
// ============================================================================
__global__ void conv2d_backward_bias_opt2(
    float* __restrict__ dY,
    float* __restrict__ dB,
    int N, int C_out, int H_out, int W_out)
{
    int c = blockIdx.x;
    if (c >= C_out) return;

    int spatial_size = H_out * W_out;
    int channel_size = N * spatial_size;

    int tid = threadIdx.x;
    int lane = tid % 32;

    float sum = 0.0f;

    // Grid-stride loop
    for (int i = tid; i < channel_size; i += blockDim.x) {
        int n = i / spatial_size;
        int rem = i % spatial_size;
        int global_idx = n * (C_out * spatial_size) + c * spatial_size + rem;
        sum += dY[global_idx];
    }

    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // First thread in each warp writes to shared memory
    __shared__ float warp_sums[32];
    int warp_id = tid / 32;

    if (lane == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();

    // Final reduction by first warp
    if (warp_id == 0) {
        sum = (lane < (blockDim.x / 32)) ? warp_sums[lane] : 0.0f;

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        if (lane == 0) {
            dB[c] = sum;
        }
    }
}

// --------------- ReLU ------------------
__global__ void relu_forward(float* __restrict__ x, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float v = x[i];
        x[i] = (v > 0.0f) ? v : 0.0f;
    }
}

// --------------- MaxPool 2x2 (stride 2) ------------------
__global__ void maxpool2x2_forward(
    float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W)
{
    int H_out = H / 2;
    int W_out = W / 2;

    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int nc = blockIdx.z;

    if (w_out >= W_out || h_out >= H_out) return;

    int n = nc / C;
    int c = nc % C;
    if (n >= N) return;

    int h_in0 = h_out * 2;
    int w_in0 = w_out * 2;

    float m = -1e30f;
    for (int dh = 0; dh < 2; dh++) {
        for (int dw = 0; dw < 2; dw++) {
            int h_in = h_in0 + dh;
            int w_in = w_in0 + dw;
            int idx = idx4(n, c, h_in, w_in, C, H, W);
            float v = input[idx];
            if (v > m) m = v;
        }
    }

    int out_idx = idx4(n, c, h_out, w_out, C, H_out, W_out);
    output[out_idx] = m;
}

// --------------- UpSample 2x2 (nearest) ------------------
__global__ void upsample2x2_forward(
    float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W)
{
    int H_out = H * 2;
    int W_out = W * 2;

    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int nc = blockIdx.z;

    if (w_out >= W_out || h_out >= H_out) return;

    int n = nc / C;
    int c = nc % C;
    if (n >= N) return;

    int h_in = h_out / 2;
    int w_in = w_out / 2;

    int idx_in = idx4(n, c, h_in, w_in, C, H, W);
    int idx_out = idx4(n, c, h_out, w_out, C, H_out, W_out);
    output[idx_out] = input[idx_in];
}

// --------------- MSE loss ------------------
__global__ void mse_loss_forward(
    float* __restrict__ output,
    float* __restrict__ target,
    float* __restrict__ loss,
    int size)
{
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float val = 0.0f;
    if (idx < size) {
        float diff = output[idx] - target[idx];
        val = diff * diff;
    }
    sdata[tid] = val;
    __syncthreads();

    // Parallel reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(loss, sdata[0]);
    }
}

// --------------- ReLU backward ------------------
__global__ void relu_backward(
    float* __restrict__ x,
    float* __restrict__ grad_y,
    float* __restrict__ grad_x,
    int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        grad_x[i] = (x[i] > 0.0f) ? grad_y[i] : 0.0f;
    }
}

// --------------- MaxPool 2x2 backward ------------------
__global__ void maxpool2x2_backward(
    float* __restrict__ input,
    float* __restrict__ grad_out,
    float* __restrict__ grad_in,
    int N, int C, int H, int W)
{
    int H_out = H / 2;
    int W_out = W / 2;

    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int nc = blockIdx.z;

    if (w_out >= W_out || h_out >= H_out) return;

    int n = nc / C;
    int c = nc % C;
    if (n >= N) return;

    int h_in0 = h_out * 2;
    int w_in0 = w_out * 2;

    int idx00 = idx4(n, c, h_in0 + 0, w_in0 + 0, C, H, W);
    int idx01 = idx4(n, c, h_in0 + 0, w_in0 + 1, C, H, W);
    int idx10 = idx4(n, c, h_in0 + 1, w_in0 + 0, C, H, W);
    int idx11 = idx4(n, c, h_in0 + 1, w_in0 + 1, C, H, W);

    float v00 = input[idx00];
    float v01 = input[idx01];
    float v10 = input[idx10];
    float v11 = input[idx11];

    float g = grad_out[idx4(n, c, h_out, w_out, C, H_out, W_out)];

    float m = v00;
    int max_idx = 0;
    if (v01 > m) { m = v01; max_idx = 1; }
    if (v10 > m) { m = v10; max_idx = 2; }
    if (v11 > m) { m = v11; max_idx = 3; }

    if (max_idx == 0) grad_in[idx00] = g;
    else if (max_idx == 1) grad_in[idx01] = g;
    else if (max_idx == 2) grad_in[idx10] = g;
    else grad_in[idx11] = g;
}

// --------------- UpSample 2x2 backward ------------------
__global__ void upsample2x2_backward(
    float* __restrict__ grad_out,
    float* __restrict__ grad_in,
    int N, int C, int H, int W)
{
    int H_out = H * 2;
    int W_out = W * 2;

    int w_in = blockIdx.x * blockDim.x + threadIdx.x;
    int h_in = blockIdx.y * blockDim.y + threadIdx.y;
    int nc = blockIdx.z;

    if (w_in >= W || h_in >= H) return;

    int n = nc / C;
    int c = nc % C;
    if (n >= N) return;

    int h_out0 = h_in * 2;
    int w_out0 = w_in * 2;

    float sum = 0.0f;
    for (int dh = 0; dh < 2; ++dh) {
        for (int dw = 0; dw < 2; ++dw) {
            int h_out = h_out0 + dh;
            int w_out = w_out0 + dw;
            int idx_o = idx4(n, c, h_out, w_out, C, H_out, W_out);
            sum += grad_out[idx_o];
        }
    }

    int idx_in = idx4(n, c, h_in, w_in, C, H, W);
    grad_in[idx_in] = sum;
}

// --------------- MSE loss backward ------------------
__global__ void mse_loss_backward(
    float* __restrict__ output,
    float* __restrict__ target,
    float* __restrict__ grad_out,
    int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    grad_out[i] = 2.0f * (output[i] - target[i]) / size;
}

// --------------- SGD update ------------------
__global__ void sgd_update(
    float* __restrict__ param,
    float* __restrict__ grad,
    int size,
    float lr)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        param[i] -= lr * grad[i];
    }
}
