// your kernels
#include "gpu_layers.h"

// --------------- Conv2D forward (naive) ------------------
// each thread computes ONE output pixel (n, c_out, h_out, w_out)
__global__ void conv2d_forward_naive(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int H, int W,
    int C_out, int K, int pad, int stride)
{
    int H_out = (H + 2 * pad - K) / stride + 1;
    int W_out = (W + 2 * pad - K) / stride + 1;

    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int nc    = blockIdx.z;  // pack (n, c_out) into z-dimension

    if (w_out >= W_out || h_out >= H_out) return;

    int n      = nc / C_out;
    int c_out  = nc % C_out;
    if (n >= N) return;

    float sum = bias ? bias[c_out] : 0.0f;

    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                int h_in = h_out * stride + kh - pad;
                int w_in = w_out * stride + kw - pad;
                if (h_in < 0 || h_in >= H || w_in < 0 || w_in >= W)
                    continue;

                int in_idx = idx4(n, c_in, h_in, w_in, C_in, H, W);

                int w_idx = (((c_out * C_in + c_in) * K) + kh) * K + kw;
                sum += weight[w_idx] * input[in_idx];
            }
        }
    }

    int out_idx = idx4(n, c_out, h_out, w_out, C_out, H_out, W_out);
    output[out_idx] = sum;
}

// --------------- ReLU ------------------
__global__ void relu_forward(float* x, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float v = x[i];
        x[i] = v > 0.0f ? v : 0.0f;
    }
}

// --------------- MaxPool 2x2 (stride 2) ------------------
__global__ void maxpool2x2_forward(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W)
{
    int H_out = H / 2;
    int W_out = W / 2;

    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int nc    = blockIdx.z;

    if (w_out >= W_out || h_out >= H_out) return;

    int n = nc / C;
    int c = nc % C;
    if (n >= N) return;

    int h_in0 = h_out * 2;
    int w_in0 = w_out * 2;

    float m = -1e30f;
    for (int dh = 0; dh < 2; ++dh) {
        for (int dw = 0; dw < 2; ++dw) {
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
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W)
{
    int H_out = H * 2;
    int W_out = W * 2;

    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int nc    = blockIdx.z;

    if (w_out >= W_out || h_out >= H_out) return;

    int n = nc / C;
    int c = nc % C;
    if (n >= N) return;

    int h_in = h_out / 2;
    int w_in = w_out / 2;

    int idx_in  = idx4(n, c, h_in, w_in, C, H, W);
    int idx_out = idx4(n, c, h_out, w_out, C, H_out, W_out);
    output[idx_out] = input[idx_in];
}

// --------------- MSE loss (naive) ------------------
__global__ void mse_loss_forward(
    const float* __restrict__ output,
    const float* __restrict__ target,
    float* __restrict__ loss,
    int size)
{
    // many threads atomically add into one float – naive but simple
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;

    float diff = output[i] - target[i];
    float val = diff * diff;

    atomicAdd(loss, val / size);
}
