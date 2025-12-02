%%writefile gpu_layers.cu
// your kernels
#include "gpu_layers.h"

// --------------- Conv2D forward (naive) ------------------
// each thread computes ONE output pixel (n, c_out, h_out, w_out)
__global__ void conv2d_forward_naive(
    const float* __restrict__ input,   // [N, C_in, H, W]
    const float* __restrict__ weight,  // [C_out, C_in, K, K]
    const float* __restrict__ bias,    // [C_out]
    float* __restrict__ output,        // [N, C_out, H_out, W_out]
    int N, int C_in, int H, int W,
    int C_out, int K, int pad, int stride)
{
    int H_out = (H + 2 * pad - K) / stride + 1;
    int W_out = (W + 2 * pad - K) / stride + 1;

    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int nc    = blockIdx.z;  // pack (n, c_out)

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

                // NCHW: ((n*C + c)*H + h)*W + w
                int in_idx = ((n * C_in + c_in) * H + h_in) * W + w_in;
                int w_idx = (((c_out * C_in + c_in) * K) + kh) * K + kw;
                sum += weight[w_idx] * input[in_idx];
            }
        }
    }
int out_idx = ((n * C_out + c_out) * H_out + h_out) * W_out + w_out;
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
    extern __shared__ float sdata[];   // shared mem size = blockDim.x * sizeof(float)

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float val = 0.0f;
    if (idx < size) {
        float diff = output[idx] - target[idx];
        val = diff * diff;             // chưa chia size, để chia sau
    }
    sdata[tid] = val;
    __syncthreads();

    // parallel reduction trong block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // thread 0 của mỗi block ghi kết quả block vào loss global
    if (tid == 0) {
        atomicAdd(loss, sdata[0]);
    }
}

// --------------- ReLU backward ------------------
__global__ void relu_backward(
    const float* __restrict__ x,       // forward output/input to ReLU
    const float* __restrict__ grad_y,  // dL/dy
    float* __restrict__ grad_x,        // dL/dx
    int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float v = x[i];
        grad_x[i] = (v > 0.0f) ? grad_y[i] : 0.0f;
    }
}

// --------------- MaxPool 2x2 backward ------------------
// input: x [N,C,H,W]  (same as forward)
// grad_out: dL/d(pool_out) [N,C,H/2,W/2]
// grad_in: dL/dx [N,C,H,W]
__global__ void maxpool2x2_backward(
    const float* __restrict__ input,
    const float* __restrict__ grad_out,
    float* __restrict__ grad_in,
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

    int idx00 = idx4(n, c, h_in0 + 0, w_in0 + 0, C, H, W);
    int idx01 = idx4(n, c, h_in0 + 0, w_in0 + 1, C, H, W);
    int idx10 = idx4(n, c, h_in0 + 1, w_in0 + 0, C, H, W);
    int idx11 = idx4(n, c, h_in0 + 1, w_in0 + 1, C, H, W);

    float v00 = input[idx00];
    float v01 = input[idx01];
    float v10 = input[idx10];
    float v11 = input[idx11];

    float g = grad_out[idx4(n, c, h_out, w_out, C, H_out, W_out)];

    // chọn phần tử max giống forward
    float m = v00;
    int max_idx = 0;
    if (v01 > m) { m = v01; max_idx = 1; }
    if (v10 > m) { m = v10; max_idx = 2; }
    if (v11 > m) { m = v11; max_idx = 3; }

    // mỗi input chỉ thuộc một ô pool 2x2, nên có thể set trực tiếp
    grad_in[idx00] = (max_idx == 0) ? g : 0.0f;
    grad_in[idx01] = (max_idx == 1) ? g : 0.0f;
    grad_in[idx10] = (max_idx == 2) ? g : 0.0f;
    grad_in[idx11] = (max_idx == 3) ? g : 0.0f;
}

// --------------- UpSample 2x2 backward ------------------
// forward: out[n,c,2h+dh,2w+dw] = in[n,c,h,w]
// grad_out: dL/d(out) [N,C,2H,2W]
// grad_in:  dL/d(in)  [N,C,H,W]
__global__ void upsample2x2_backward(
    const float* __restrict__ grad_out,
    float* __restrict__ grad_in,
    int N, int C, int H, int W)
{
    int H_out = H * 2;
    int W_out = W * 2;

    int w_in = blockIdx.x * blockDim.x + threadIdx.x;
    int h_in = blockIdx.y * blockDim.y + threadIdx.y;
    int nc   = blockIdx.z;

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
// L = 1/size * sum (out - target)^2  => dL/dout = 2*(out - target)/size
__global__ void mse_loss_backward(
    const float* __restrict__ output,
    const float* __restrict__ target,
    float* __restrict__ grad_out,
    int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    grad_out[i] = 2.0f * (output[i] - target[i]) / size;
}

// --------------- Conv2D backward: dX ------------------
// dX[n,c_in,h,w] = sum_{c_out,kh,kw} dY[n,c_out,h_out,w_out] * W[c_out,c_in,kh,kw]
__global__ void conv2d_backward_input_naive(
    const float* __restrict__ dY,
    const float* __restrict__ weight,
    float* __restrict__ dX,
    int N, int C_in, int H, int W,
    int C_out, int K, int pad, int stride)
{
    int H_out = (H + 2 * pad - K) / stride + 1;
    int W_out = (W + 2 * pad - K) / stride + 1;

    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int nc = blockIdx.z;

    if (w >= W || h >= H) return;

    int n = nc / C_in;
    int c_in = nc % C_in;
    if (n >= N) return;

    float sum = 0.0f;
    for (int c_out = 0; c_out < C_out; ++c_out) {
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                int h_out = h + pad - kh;
                int w_out = w + pad - kw;

                if (h_out % stride != 0 || w_out % stride != 0) continue;

                h_out /= stride;
                w_out /= stride;

                if (h_out < 0 || h_out >= H_out ||
                    w_out < 0 || w_out >= W_out)
                    continue;

                int dy_idx = idx4(n, c_out, h_out, w_out,
                                  C_out, H_out, W_out);
                int w_idx = (((c_out * C_in + c_in) * K) + kh) * K + kw;
                sum += dY[dy_idx] * weight[w_idx];
            }
        }
    }

    int dx_idx = idx4(n, c_in, h, w, C_in, H, W);
    dX[dx_idx] = sum;
}

// --------------- Conv2D backward: dW ------------------
__global__ void conv2d_backward_weight_naive(
    const float* __restrict__ input,
    const float* __restrict__ dY,
    float* __restrict__ dW,
    int N, int C_in, int H, int W,
    int C_out, int K, int pad, int stride)
{
    int H_out = (H + 2 * pad - K) / stride + 1;
    int W_out = (W + 2 * pad - K) / stride + 1;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C_out * C_in * K * K;
    if (idx >= total) return;

    int kw = idx % K;
    int tmp = idx / K;
    int kh = tmp % K;
    tmp /= K;
    int c_in = tmp % C_in;
    int c_out = tmp / C_in;

    float sum = 0.0f;
    for (int n = 0; n < N; ++n) {
        for (int h_out = 0; h_out < H_out; ++h_out) {
            for (int w_out = 0; w_out < W_out; ++w_out) {
                int h_in = h_out * stride + kh - pad;
                int w_in = w_out * stride + kw - pad;
                if (h_in < 0 || h_in >= H || w_in < 0 || w_in >= W)
                    continue;

                int in_idx = idx4(n, c_in, h_in, w_in, C_in, H, W);
                int dy_idx = idx4(n, c_out, h_out, w_out,
                                  C_out, H_out, W_out);
                sum += dY[dy_idx] * input[in_idx];
            }
        }
    }

    dW[idx] = sum;
}

// --------------- Conv2D backward: dB ------------------
__global__ void conv2d_backward_bias_naive(
    const float* __restrict__ dY,
    float* __restrict__ dB,
    int N, int C_out, int H_out, int W_out)
{
    int c_out = blockIdx.x * blockDim.x + threadIdx.x;
    if (c_out >= C_out) return;

    float sum = 0.0f;
    for (int n = 0; n < N; ++n) {
        for (int h = 0; h < H_out; ++h) {
            for (int w = 0; w < W_out; ++w) {
                int idx = idx4(n, c_out, h, w, C_out, H_out, W_out);
                sum += dY[idx];
            }
        }
    }
    dB[c_out] = sum;
}

// --------------- SGD update ------------------
__global__ void sgd_update(
    float* __restrict__ param,
    const float* __restrict__ grad,
    int size,
    float lr)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        param[i] -= lr * grad[i];
    }
}