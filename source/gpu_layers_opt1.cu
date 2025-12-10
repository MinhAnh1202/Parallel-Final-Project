#include "gpu_layers_opt1.h"

// --------------- Conv2D forward (optimization 1) ------------------
void update_dc_bias(float* d_bias_ptr, int count) {
    CHECK_CUDA(cudaMemcpyToSymbol(dc_bias, d_bias_ptr, count * sizeof(float),
                                   0, cudaMemcpyDeviceToDevice));
}

__global__ void conv2d_forward(
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

    int w_out = blockIdx.x * TILE_W + tx;
    int h_out = blockIdx.y * TILE_H + ty;

    int nc = blockIdx.z;
    int c_out = nc % C_out;
    int n = nc / C_out;

    float value = 0.0f;

    int row_start = blockIdx.y * TILE_H * stride - pad;
    int col_start = blockIdx.x * TILE_W * stride - pad;

    for (int c_in = 0; c_in < C_in; c_in++) {
        // Load input tile into shared memory
        for (int i = ty; i < BLOCK_H; i += blockDim.y) {
            for (int j = tx; j < BLOCK_W; j += blockDim.x) {
                int h_in = row_start + i;
                int w_in = col_start + j;

                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                    size_t input_idx = idx4(n, c_in, h_in, w_in, C_in, H, W);
                    smem[i][j] = input[input_idx];
                } else {
                    smem[i][j] = 0.0f;
                }
            }
        }
        __syncthreads();

        // Compute convolution
        for (int i = 0; i < K; ++i) {
            for (int j = 0; j < K; ++j) {
                size_t weight_idx = idx4(c_out, c_in, i, j, C_in, K, K);
                value += smem[ty + i][tx + j] * weight[weight_idx];
            }
        }
        __syncthreads();
    }

    if (w_out < W_out && h_out < H_out && n < N) {
        value += dc_bias[c_out];
        output[idx4(n, c_out, h_out, w_out, C_out, H_out, W_out)] = value;
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

// --------------- Conv2D backward: dX ------------------
__global__ void conv2d_backward_input(
    float* __restrict__ dY,
    float* __restrict__ weight,
    float* __restrict__ dX,
    int N, int C_in, int H, int W,
    int C_out, int pad, int stride)
{
    int H_out = (H + 2 * pad - K) / stride + 1;
    int W_out = (W + 2 * pad - K) / stride + 1;

    // Shared Memory chứa dY
    __shared__ float s_dY[BLOCK_H][BLOCK_W];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Tọa độ dX (output của kernel này)
    int h_out = blockIdx.y * TILE_H + ty;
    int w_out = blockIdx.x * TILE_W + tx;

    int nc = blockIdx.z;
    int c_in = nc % C_in;
    int n = nc / C_in;
    //if (h_out >= H || w_out >= W || n >= N) return;

    float value = 0.0f;

    for (int c_out = 0; c_out < C_out; ++c_out) {
        // Tọa độ dY cần tải
        int h_global = h_out;
        int w_global = w_out;

        // Tải main tile
        if (h_global >= 0 && h_global < H_out && w_global >= 0 && w_global < W_out) {
            s_dY[ty + pad][tx + pad] = dY[idx4(n, c_out, h_global, w_global, C_out, H_out, W_out)];
        } else {
            s_dY[ty + pad][tx + pad] = 0.0f;
        }

        // Tải biên trên/dưới
        if (ty < pad) {
            // Biên trên
            int h_top = blockIdx.y * TILE_H + ty - pad;
            if (h_top >= 0 && h_top < H_out && w_global >= 0 && w_global < W_out) {
                s_dY[ty][tx + pad] = dY[idx4(n, c_out, h_top, w_global, C_out, H_out, W_out)];
            } else {
                s_dY[ty][tx + pad] = 0.0f;
            }

            // Biên dưới
            int h_bottom = blockIdx.y * TILE_H + TILE_H + pad - 1 - ty;
            if (h_bottom >= 0 && h_bottom < H_out && w_global >= 0 && w_global < W_out) {
                s_dY[BLOCK_H - 1 - ty][tx + pad] = dY[idx4(n, c_out, h_bottom, w_global, C_out, H_out, W_out)];
            } else {
                s_dY[BLOCK_H - 1 - ty][tx + pad] = 0.0f;
            }
        }

        // Tải biên trái/phải
        if (tx < pad) {
            // Biên trái
            int w_left = blockIdx.x * TILE_W + tx - pad;
            if (w_left >= 0 && w_left < W_out && h_global >= 0 && h_global < H_out) {
                s_dY[ty + pad][tx] = dY[idx4(n, c_out, h_global, w_left, C_out, H_out, W_out)];
            } else {
                s_dY[ty + pad][tx] = 0.0f;
            }

            // Biên phải
            int w_right = blockIdx.x * TILE_W + TILE_W + pad - 1 - tx;
            if (w_right >= 0 && w_right < W_out && h_global >= 0 && h_global < H_out) {
                s_dY[ty + pad][BLOCK_W - 1 - tx] = dY[idx4(n, c_out, h_global, w_right, C_out, H_out, W_out)];
            } else {
                s_dY[ty + pad][BLOCK_W - 1 - tx] = 0.0f;
            }
        }

        // Tải 4 góc (Thread (0,0) tải)
        if (tx == 0 && ty == 0) {
            // Góc trên trái [0][0]
            int h_c = blockIdx.y * TILE_H - pad;
            int w_c = blockIdx.x * TILE_W - pad;
            s_dY[0][0] = (h_c >= 0 && h_c < H_out && w_c >= 0 && w_c < W_out) ? dY[idx4(n, c_out, h_c, w_c, C_out, H_out, W_out)] : 0.0f;

            // Góc trên phải [0][17]
            w_c = blockIdx.x * TILE_W + TILE_W + pad - 1;
            s_dY[0][BLOCK_W - 1] = (h_c >= 0 && h_c < H_out && w_c >= 0 && w_c < W_out) ? dY[idx4(n, c_out, h_c, w_c, C_out, H_out, W_out)] : 0.0f;

            // Góc dưới trái [17][0]
            h_c = blockIdx.y * TILE_H + TILE_H + pad - 1;
            w_c = blockIdx.x * TILE_W - pad;
            s_dY[BLOCK_H - 1][0] = (h_c >= 0 && h_c < H_out && w_c >= 0 && w_c < W_out) ? dY[idx4(n, c_out, h_c, w_c, C_out, H_out, W_out)] : 0.0f;

            // Góc dưới phải [17][17]
            w_c = blockIdx.x * TILE_W + TILE_W + pad - 1;
            s_dY[BLOCK_H - 1][BLOCK_W - 1] = (h_c >= 0 && h_c < H_out && w_c >= 0 && w_c < W_out) ? dY[idx4(n, c_out, h_c, w_c, C_out, H_out, W_out)] : 0.0f;
        }

        __syncthreads();

        // Tính convolution
            for (int kh = 0; kh < K; ++kh) {
                for (int kw = 0; kw < K; ++kw) {
                    int smem_y = ty + 2 * pad - kh;
                    int smem_x = tx + 2* pad - kw;

                    size_t w_idx = idx4(c_out, c_in, K - 1 - kh, K - 1 - kw, C_in, K, K);
                    value += s_dY[smem_y][smem_x] * weight[w_idx];
                }
            }
        __syncthreads();
    }

    if (h_out < H && w_out < W && n < N) {
        dX[idx4(n, c_in, h_out, w_out, C_in, H, W)] = value;
    }
}

// --------------- Conv2D backward: dW ------------------
__global__ void conv2d_backward_weight(
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

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int index = blockIdx.z;
    int c_in = index % C_in;
    int c_out = index / C_in;

    float dw[K][K];
    for (int i = 0; i < K; ++i)
        for (int j = 0; j < K; ++j)
            dw[i][j] = 0.0f;

    for (int n = 0; n < N; ++n) {
        int num_blocks_h = (H_out + TILE_H - 1) / TILE_H;
        int num_blocks_w = (W_out + TILE_W - 1) / TILE_W;

        for (int block_h = 0; block_h < num_blocks_h; ++block_h) {
            for (int block_w = 0; block_w < num_blocks_w; ++block_w) {

                // Load s_dY
                int h_out = block_h * TILE_H + ty;
                int w_out = block_w * TILE_W + tx;

                if (h_out < H_out && w_out < W_out) {
                    s_dY[ty][tx] = dY[idx4(n, c_out, h_out, w_out, C_out, H_out, W_out)];
                } else {
                    s_dY[ty][tx] = 0.0f;
                }

                // Load s_in
                int h_in_base = block_h * TILE_H + ty;
                int w_in_base = block_w * TILE_W + tx;

                // Main tile
                if (h_in_base >= 0 && h_in_base < H && w_in_base >= 0 && w_in_base < W) {
                    s_in[ty + pad][tx + pad] = input[idx4(n, c_in, h_in_base, w_in_base, C_in, H, W)];
                } else {
                    s_in[ty + pad][tx + pad] = 0.0f;
                }

                // Top/bottom borders
                if (ty < pad) {
                    int h_top = block_h * TILE_H - pad + ty;
                    if (h_top >= 0 && h_top < H && w_in_base >= 0 && w_in_base < W) {
                        s_in[ty][tx + pad] = input[idx4(n, c_in, h_top, w_in_base, C_in, H, W)];
                    } else {
                        s_in[ty][tx + pad] = 0.0f;
                    }

                    int h_bottom = block_h * TILE_H + TILE_H + pad - 1 - ty;
                    if (h_bottom >= 0 && h_bottom < H && w_in_base >= 0 && w_in_base < W) {
                        s_in[BLOCK_H - 1 - ty][tx + pad] = input[idx4(n, c_in, h_bottom, w_in_base, C_in, H, W)];
                    } else {
                        s_in[BLOCK_H - 1 - ty][tx + pad] = 0.0f;
                    }
                }

                // Left/right borders
                if (tx < pad) {
                    int w_left = block_w * TILE_W - pad + tx;
                    if (w_left >= 0 && w_left < W && h_in_base >= 0 && h_in_base < H) {
                        s_in[ty + pad][tx] = input[idx4(n, c_in, h_in_base, w_left, C_in, H, W)];
                    } else {
                        s_in[ty + pad][tx] = 0.0f;
                    }

                    int w_right = block_w * TILE_W + TILE_W + pad - 1 - tx;
                    if (w_right >= 0 && w_right < W && h_in_base >= 0 && h_in_base < H) {
                        s_in[ty + pad][BLOCK_W - 1 - tx] = input[idx4(n, c_in, h_in_base, w_right, C_in, H, W)];
                    } else {
                        s_in[ty + pad][BLOCK_W - 1 - tx] = 0.0f;
                    }
                }

                // Thread (0,0) loads 4 corners
                if (tx == 0 && ty == 0) {
                    int h_c = block_h * TILE_H - pad;
                    int w_c = block_w * TILE_W - pad;
                    s_in[0][0] = (h_c >= 0 && h_c < H && w_c >= 0 && w_c < W)
                        ? input[idx4(n, c_in, h_c, w_c, C_in, H, W)] : 0.0f;

                    w_c = block_w * TILE_W + TILE_W + pad - 1;
                    s_in[0][BLOCK_W - 1] = (h_c >= 0 && h_c < H && w_c >= 0 && w_c < W)
                        ? input[idx4(n, c_in, h_c, w_c, C_in, H, W)] : 0.0f;

                    h_c = block_h * TILE_H + TILE_H + pad - 1;
                    w_c = block_w * TILE_W - pad;
                    s_in[BLOCK_H - 1][0] = (h_c >= 0 && h_c < H && w_c >= 0 && w_c < W)
                        ? input[idx4(n, c_in, h_c, w_c, C_in, H, W)] : 0.0f;

                    w_c = block_w * TILE_W + TILE_W + pad - 1;
                    s_in[BLOCK_H - 1][BLOCK_W - 1] = (h_c >= 0 && h_c < H && w_c >= 0 && w_c < W)
                        ? input[idx4(n, c_in, h_c, w_c, C_in, H, W)] : 0.0f;
                }

                __syncthreads();

                // Compute dW: X * dY
                float val_dy = s_dY[ty][tx];
                for (int kh = 0; kh < K; ++kh) {
                    for (int kw = 0; kw < K; ++kw) {
                        dw[kh][kw] += s_in[ty + kh][tx + kw] * val_dy;
                    }
                }

                __syncthreads();
            }
        }
    }

    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < K; ++j) {
            // Tính index global của dW[i][j] cho cặp filter (c_out, c_in)
            size_t dw_idx = idx4(c_out, c_in, i, j, C_in, K, K);
            // Cộng giá trị dw[i][j] của thread hiện tại vào bộ nhớ global
            atomicAdd(&dW[dw_idx], dw[i][j]);
        }
    }
}

// --------------- Conv2D backward: dB ------------------
__global__ void conv2d_backward_bias(
    float* __restrict__ dY,
    float* __restrict__ dB,
    int N, int C_out, int H_out, int W_out)
{
    int spatial_size = H_out * W_out;
    int channel_size = N * spatial_size;
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int c = blockIdx.x;
    if (c >= C_out) return;
    float sum = 0.0f;
    for (int i = tid; i < channel_size; i += blockDim.x) {
        int n = i / spatial_size;
        int rem = i % spatial_size;
        int global_idx = n * (C_out * spatial_size) + c * spatial_size + rem;
        sum += dY[global_idx];
    }
    sdata[tid] = sum;
    __syncthreads();

    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        dB[c] = sdata[0];
    }
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
