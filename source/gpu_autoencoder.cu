// your GPUAutoencoder implementation
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
#include "gpu_layers.h"
#include "gpu_autoencoder.h"

static inline float rand_uniform(float min_val, float max_val) {
    float r = (float)rand() / (float)RAND_MAX;
    return min_val + r * (max_val - min_val);
}


void gpu_autoencoder_init(GPUAutoencoder *ae, int batch_size) {
    ae->N = batch_size;
    ae->H = 32;
    ae->W = 32;

    const int N = ae->N;
    const int H = ae->H;
    const int W = ae->W;

    // ---------- allocate weights ----------
    const int K = 3;

    int C_in1 = 3,   C_out1 = 256;
    int C_in2 = 256, C_out2 = 128;
    int C_in3 = 128, C_out3 = 128;
    int C_in4 = 128, C_out4 = 256;
    int C_in5 = 256, C_out5 = 3;

    size_t w1_bytes = C_out1 * C_in1 * K * K * sizeof(float);
    size_t b1_bytes = C_out1 * sizeof(float);
    size_t w2_bytes = C_out2 * C_in2 * K * K * sizeof(float);
    size_t b2_bytes = C_out2 * sizeof(float);
    size_t w3_bytes = C_out3 * C_in3 * K * K * sizeof(float);
    size_t b3_bytes = C_out3 * sizeof(float);
    size_t w4_bytes = C_out4 * C_in4 * K * K * sizeof(float);
    size_t b4_bytes = C_out4 * sizeof(float);
    size_t w5_bytes = C_out5 * C_in5 * K * K * sizeof(float);
    size_t b5_bytes = C_out5 * sizeof(float);

    CHECK_CUDA(cudaMalloc(&ae->d_w1, w1_bytes));
    CHECK_CUDA(cudaMalloc(&ae->d_b1, b1_bytes));
    CHECK_CUDA(cudaMalloc(&ae->d_w2, w2_bytes));
    CHECK_CUDA(cudaMalloc(&ae->d_b2, b2_bytes));
    CHECK_CUDA(cudaMalloc(&ae->d_w3, w3_bytes));
    CHECK_CUDA(cudaMalloc(&ae->d_b3, b3_bytes));
    CHECK_CUDA(cudaMalloc(&ae->d_w4, w4_bytes));
    CHECK_CUDA(cudaMalloc(&ae->d_b4, b4_bytes));
    CHECK_CUDA(cudaMalloc(&ae->d_w5, w5_bytes));
    CHECK_CUDA(cudaMalloc(&ae->d_b5, b5_bytes));

    // init weights on host
    // find max weight bytes
    size_t max_w_bytes = w1_bytes;
    if (w2_bytes > max_w_bytes) max_w_bytes = w2_bytes;
    if (w3_bytes > max_w_bytes) max_w_bytes = w3_bytes;
    if (w4_bytes > max_w_bytes) max_w_bytes = w4_bytes;
    if (w5_bytes > max_w_bytes) max_w_bytes = w5_bytes;

    // find max bias bytes
    size_t max_b_bytes = b1_bytes;
    if (b2_bytes > max_b_bytes) max_b_bytes = b2_bytes;
    if (b3_bytes > max_b_bytes) max_b_bytes = b3_bytes;
    if (b4_bytes > max_b_bytes) max_b_bytes = b4_bytes;
    if (b5_bytes > max_b_bytes) max_b_bytes = b5_bytes;

    float *h_w = (float*)malloc(max_w_bytes);
    float *h_b = (float*)malloc(max_b_bytes);

    auto init_wb = [&](float *d_w, size_t w_bytes, float *d_b, size_t b_bytes) {
        size_t w_cnt = w_bytes / sizeof(float);
        size_t b_cnt = b_bytes / sizeof(float);
        for (size_t i = 0; i < w_cnt; ++i) h_w[i] = rand_uniform(-0.05f, 0.05f);
        for (size_t i = 0; i < b_cnt; ++i) h_b[i] = rand_uniform(-0.05f, 0.05f);
        CHECK_CUDA(cudaMemcpy(d_w, h_w, w_bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_b, h_b, b_bytes, cudaMemcpyHostToDevice));
    };

    init_wb(ae->d_w1, w1_bytes, ae->d_b1, b1_bytes);
    init_wb(ae->d_w2, w2_bytes, ae->d_b2, b2_bytes);
    init_wb(ae->d_w3, w3_bytes, ae->d_b3, b3_bytes);
    init_wb(ae->d_w4, w4_bytes, ae->d_b4, b4_bytes);
    init_wb(ae->d_w5, w5_bytes, ae->d_b5, b5_bytes);

    free(h_w);
    free(h_b);

    // ---------- allocate activations ----------
    size_t bytes_x0  = N * 3   * 32 * 32 * sizeof(float);
    size_t bytes_h1  = N * 256 * 32 * 32 * sizeof(float);
    size_t bytes_p1  = N * 256 * 16 * 16 * sizeof(float);
    size_t bytes_h2  = N * 128 * 16 * 16 * sizeof(float);
    size_t bytes_p2  = N * 128 *  8 *  8 * sizeof(float);
    size_t bytes_h3  = N * 128 *  8 *  8 * sizeof(float);
    size_t bytes_u1  = N * 128 * 16 * 16 * sizeof(float);
    size_t bytes_h4  = N * 256 * 16 * 16 * sizeof(float);
    size_t bytes_u2  = N * 256 * 32 * 32 * sizeof(float);
    size_t bytes_out = N * 3   * 32 * 32 * sizeof(float);

    CHECK_CUDA(cudaMalloc(&ae->d_x0,  bytes_x0));
    CHECK_CUDA(cudaMalloc(&ae->d_h1,  bytes_h1));
    CHECK_CUDA(cudaMalloc(&ae->d_p1,  bytes_p1));
    CHECK_CUDA(cudaMalloc(&ae->d_h2,  bytes_h2));
    CHECK_CUDA(cudaMalloc(&ae->d_p2,  bytes_p2));
    CHECK_CUDA(cudaMalloc(&ae->d_h3,  bytes_h3));
    CHECK_CUDA(cudaMalloc(&ae->d_u1,  bytes_u1));
    CHECK_CUDA(cudaMalloc(&ae->d_h4,  bytes_h4));
    CHECK_CUDA(cudaMalloc(&ae->d_u2,  bytes_u2));
    CHECK_CUDA(cudaMalloc(&ae->d_out, bytes_out));

    // loss buffer
    CHECK_CUDA(cudaMalloc(&ae->d_loss, sizeof(float)));

    // ---------- allocate activation gradients ----------
    CHECK_CUDA(cudaMalloc(&ae->d_gx0,  bytes_x0));
    CHECK_CUDA(cudaMalloc(&ae->d_gh1,  bytes_h1));
    CHECK_CUDA(cudaMalloc(&ae->d_gp1,  bytes_p1));
    CHECK_CUDA(cudaMalloc(&ae->d_gh2,  bytes_h2));
    CHECK_CUDA(cudaMalloc(&ae->d_gp2,  bytes_p2));
    CHECK_CUDA(cudaMalloc(&ae->d_gh3,  bytes_h3));
    CHECK_CUDA(cudaMalloc(&ae->d_gu1,  bytes_u1));
    CHECK_CUDA(cudaMalloc(&ae->d_gh4,  bytes_h4));
    CHECK_CUDA(cudaMalloc(&ae->d_gu2,  bytes_u2));
    CHECK_CUDA(cudaMalloc(&ae->d_gout, bytes_out));

    // ---------- allocate weight gradients ----------
    CHECK_CUDA(cudaMalloc(&ae->d_gw1, w1_bytes));
    CHECK_CUDA(cudaMalloc(&ae->d_gb1, b1_bytes));
    CHECK_CUDA(cudaMalloc(&ae->d_gw2, w2_bytes));
    CHECK_CUDA(cudaMalloc(&ae->d_gb2, b2_bytes));
    CHECK_CUDA(cudaMalloc(&ae->d_gw3, w3_bytes));
    CHECK_CUDA(cudaMalloc(&ae->d_gb3, b3_bytes));
    CHECK_CUDA(cudaMalloc(&ae->d_gw4, w4_bytes));
    CHECK_CUDA(cudaMalloc(&ae->d_gb4, b4_bytes));
    CHECK_CUDA(cudaMalloc(&ae->d_gw5, w5_bytes));
    CHECK_CUDA(cudaMalloc(&ae->d_gb5, b5_bytes));
}

void gpu_autoencoder_free(GPUAutoencoder *ae) {
    // weights
    cudaFree(ae->d_w1); cudaFree(ae->d_b1);
    cudaFree(ae->d_w2); cudaFree(ae->d_b2);
    cudaFree(ae->d_w3); cudaFree(ae->d_b3);
    cudaFree(ae->d_w4); cudaFree(ae->d_b4);
    cudaFree(ae->d_w5); cudaFree(ae->d_b5);

    // activations
    cudaFree(ae->d_x0);
    cudaFree(ae->d_h1);
    cudaFree(ae->d_p1);
    cudaFree(ae->d_h2);
    cudaFree(ae->d_p2);
    cudaFree(ae->d_h3);
    cudaFree(ae->d_u1);
    cudaFree(ae->d_h4);
    cudaFree(ae->d_u2);
    cudaFree(ae->d_out);

    cudaFree(ae->d_loss);

    // activation gradients
    cudaFree(ae->d_gx0);
    cudaFree(ae->d_gh1);
    cudaFree(ae->d_gp1);
    cudaFree(ae->d_gh2);
    cudaFree(ae->d_gp2);
    cudaFree(ae->d_gh3);
    cudaFree(ae->d_gu1);
    cudaFree(ae->d_gh4);
    cudaFree(ae->d_gu2);
    cudaFree(ae->d_gout);

    // weight gradients
    cudaFree(ae->d_gw1); cudaFree(ae->d_gb1);
    cudaFree(ae->d_gw2); cudaFree(ae->d_gb2);
    cudaFree(ae->d_gw3); cudaFree(ae->d_gb3);
    cudaFree(ae->d_gw4); cudaFree(ae->d_gb4);
    cudaFree(ae->d_gw5); cudaFree(ae->d_gb5);
}

void gpu_autoencoder_copy_weights_to_host(
    GPUAutoencoder *ae,
    float *h_w1, float *h_b1,
    float *h_w2, float *h_b2,
    float *h_w3, float *h_b3,
    float *h_w4, float *h_b4,
    float *h_w5, float *h_b5)
{
    const int K = 3;
    int C_in1 = 3,   C_out1 = 256;
    int C_in2 = 256, C_out2 = 128;
    int C_in3 = 128, C_out3 = 128;
    int C_in4 = 128, C_out4 = 256;
    int C_in5 = 256, C_out5 = 3;

    size_t w1_bytes = C_out1 * C_in1 * K * K * sizeof(float);
    size_t b1_bytes = C_out1 * sizeof(float);
    size_t w2_bytes = C_out2 * C_in2 * K * K * sizeof(float);
    size_t b2_bytes = C_out2 * sizeof(float);
    size_t w3_bytes = C_out3 * C_in3 * K * K * sizeof(float);
    size_t b3_bytes = C_out3 * sizeof(float);
    size_t w4_bytes = C_out4 * C_in4 * K * K * sizeof(float);
    size_t b4_bytes = C_out4 * sizeof(float);
    size_t w5_bytes = C_out5 * C_in5 * K * K * sizeof(float);
    size_t b5_bytes = C_out5 * sizeof(float);

    CHECK_CUDA(cudaMemcpy(h_w1, ae->d_w1, w1_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_b1, ae->d_b1, b1_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_w2, ae->d_w2, w2_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_b2, ae->d_b2, b2_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_w3, ae->d_w3, w3_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_b3, ae->d_b3, b3_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_w4, ae->d_w4, w4_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_b4, ae->d_b4, b4_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_w5, ae->d_w5, w5_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_b5, ae->d_b5, b5_bytes, cudaMemcpyDeviceToHost));
}

void gpu_autoencoder_copy_weights_to_device(
    GPUAutoencoder *ae,
    const float *h_w1, const float *h_b1,
    const float *h_w2, const float *h_b2,
    const float *h_w3, const float *h_b3,
    const float *h_w4, const float *h_b4,
    const float *h_w5, const float *h_b5)
{
    const int K = 3;
    int C_in1 = 3,   C_out1 = 256;
    int C_in2 = 256, C_out2 = 128;
    int C_in3 = 128, C_out3 = 128;
    int C_in4 = 128, C_out4 = 256;
    int C_in5 = 256, C_out5 = 3;

    size_t w1_bytes = C_out1 * C_in1 * K * K * sizeof(float);
    size_t b1_bytes = C_out1 * sizeof(float);
    size_t w2_bytes = C_out2 * C_in2 * K * K * sizeof(float);
    size_t b2_bytes = C_out2 * sizeof(float);
    size_t w3_bytes = C_out3 * C_in3 * K * K * sizeof(float);
    size_t b3_bytes = C_out3 * sizeof(float);
    size_t w4_bytes = C_out4 * C_in4 * K * K * sizeof(float);
    size_t b4_bytes = C_out4 * sizeof(float);
    size_t w5_bytes = C_out5 * C_in5 * K * K * sizeof(float);
    size_t b5_bytes = C_out5 * sizeof(float);

    CHECK_CUDA(cudaMemcpy(ae->d_w1, h_w1, w1_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ae->d_b1, h_b1, b1_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ae->d_w2, h_w2, w2_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ae->d_b2, h_b2, b2_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ae->d_w3, h_w3, w3_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ae->d_b3, h_b3, b3_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ae->d_w4, h_w4, w4_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ae->d_b4, h_b4, b4_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ae->d_w5, h_w5, w5_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ae->d_b5, h_b5, b5_bytes, cudaMemcpyHostToDevice));
}

float gpu_autoencoder_forward(
    GPUAutoencoder *ae,
    const float *h_input,
    float *h_output,
    bool compute_loss)
{
    const int N = ae->N;
    const int H = ae->H;
    const int W = ae->W;
    const int K = 3;
    const int pad = 1;
    const int stride = 1;

    // ------------- copy input to device -------------
    size_t in_bytes = N * 3 * H * W * sizeof(float);
    CHECK_CUDA(cudaMemcpy(ae->d_x0, h_input, in_bytes, cudaMemcpyHostToDevice));

    dim3 block2d(16, 16);

    // ========= ENCODER =========
    // conv1: 3 -> 256, same 32x32
    {
        int C_in = 3, C_out = 256;
        int H_out = 32, W_out = 32;
        dim3 gridConv(
            (W_out + block2d.x - 1) / block2d.x,
            (H_out + block2d.y - 1) / block2d.y,
            N * C_out);

        conv2d_forward_naive<<<gridConv, block2d>>>(
            ae->d_x0, ae->d_w1, ae->d_b1, ae->d_h1,
            N, C_in, H, W, C_out, K, pad, stride);

        // ReLU
        int size = N * C_out * H_out * W_out;
        int t = 256;
        int b = (size + t - 1) / t;
        relu_forward<<<b, t>>>(ae->d_h1, size);

        // MaxPool 2x2 -> 16x16
        int Hp = 16, Wp = 16;
        dim3 gridPool(
            (Wp + block2d.x - 1) / block2d.x,
            (Hp + block2d.y - 1) / block2d.y,
            N * C_out);

        maxpool2x2_forward<<<gridPool, block2d>>>(
            ae->d_h1, ae->d_p1,
            N, C_out, H_out, W_out);
    }

    // conv2: 256 -> 128, 16x16, then pool -> 8x8
    {
        int C_in = 256, C_out = 128;
        int H_in = 16, W_in = 16;
        int H_out = 16, W_out = 16;
        dim3 gridConv(
            (W_out + block2d.x - 1) / block2d.x,
            (H_out + block2d.y - 1) / block2d.y,
            N * C_out);

        conv2d_forward_naive<<<gridConv, block2d>>>(
            ae->d_p1, ae->d_w2, ae->d_b2, ae->d_h2,
            N, C_in, H_in, W_in, C_out, K, pad, stride);

        int size = N * C_out * H_out * W_out;
        int t = 256;
        int b = (size + t - 1) / t;
        relu_forward<<<b, t>>>(ae->d_h2, size);

        // pool -> 8x8
        int Hp = 8, Wp = 8;
        dim3 gridPool(
            (Wp + block2d.x - 1) / block2d.x,
            (Hp + block2d.y - 1) / block2d.y,
            N * C_out);

        maxpool2x2_forward<<<gridPool, block2d>>>(
            ae->d_h2, ae->d_p2,
            N, C_out, H_out, W_out);
    }

    // LATENT is ae->d_p2: [N, 128, 8, 8]

    // ========= DECODER =========
    // conv3: 128 -> 128, 8x8
    {
        int C_in = 128, C_out = 128;
        int H_in = 8, W_in = 8;
        int H_out = 8, W_out = 8;

        dim3 gridConv(
            (W_out + block2d.x - 1) / block2d.x,
            (H_out + block2d.y - 1) / block2d.y,
            N * C_out);

        conv2d_forward_naive<<<gridConv, block2d>>>(
            ae->d_p2, ae->d_w3, ae->d_b3, ae->d_h3,
            N, C_in, H_in, W_in, C_out, K, pad, stride);

        int size = N * C_out * H_out * W_out;
        int t = 256;
        int b = (size + t - 1) / t;
        relu_forward<<<b, t>>>(ae->d_h3, size);

        // upsample 8x8 -> 16x16
        int Hu = 16, Wu = 16;
        dim3 gridUp(
            (Wu + block2d.x - 1) / block2d.x,
            (Hu + block2d.y - 1) / block2d.y,
            N * C_out);

        upsample2x2_forward<<<gridUp, block2d>>>(
            ae->d_h3, ae->d_u1,
            N, C_out, H_in, W_in);
    }

    // conv4: 128 -> 256, 16x16, then upsample 16->32
    {
        int C_in = 128, C_out = 256;
        int H_in = 16, W_in = 16;
        int H_out = 16, W_out = 16;

        dim3 gridConv(
            (W_out + block2d.x - 1) / block2d.x,
            (H_out + block2d.y - 1) / block2d.y,
            N * C_out);

        conv2d_forward_naive<<<gridConv, block2d>>>(
            ae->d_u1, ae->d_w4, ae->d_b4, ae->d_h4,
            N, C_in, H_in, W_in, C_out, K, pad, stride);

        int size = N * C_out * H_out * W_out;
        int t = 256;
        int b = (size + t - 1) / t;
        relu_forward<<<b, t>>>(ae->d_h4, size);

        // upsample 16x16 -> 32x32
        int Hu = 32, Wu = 32;
        dim3 gridUp(
            (Wu + block2d.x - 1) / block2d.x,
            (Hu + block2d.y - 1) / block2d.y,
            N * C_out);

        upsample2x2_forward<<<gridUp, block2d>>>(
            ae->d_h4, ae->d_u2,
            N, C_out, H_in, W_in);
    }

    // conv5: 256 -> 3, 32x32 (no activation, usually MSE on raw output)
    {
        int C_in = 256, C_out = 3;
        int H_in = 32, W_in = 32;
        int H_out = 32, W_out = 32;

        dim3 gridConv(
            (W_out + block2d.x - 1) / block2d.x,
            (H_out + block2d.y - 1) / block2d.y,
            N * C_out);

        conv2d_forward_naive<<<gridConv, block2d>>>(
            ae->d_u2, ae->d_w5, ae->d_b5, ae->d_out,
            N, C_in, H_in, W_in, C_out, K, pad, stride);
    }

    // ------------- (optional) compute MSE loss -------------
    float loss_value = 0.0f;
        if (compute_loss) {
        int size = N * 3 * 32 * 32;
        CHECK_CUDA(cudaMemset(ae->d_loss, 0, sizeof(float)));

        int t = 256;
        int b = (size + t - 1) / t;
        size_t shmem_bytes = t * sizeof(float);

        // kernel giờ trả về SUM(diff^2) vào d_loss
        mse_loss_forward<<<b, t, shmem_bytes>>>(
            ae->d_out, ae->d_x0, ae->d_loss, size);

        float loss_sum = 0.0f;
        CHECK_CUDA(cudaMemcpy(&loss_sum, ae->d_loss,
                              sizeof(float),
                              cudaMemcpyDeviceToHost));

        loss_value = loss_sum / size;  // MSE = sum / size
    }


    // ------------- copy output back to host -------------
    size_t out_bytes = N * 3 * 32 * 32 * sizeof(float);
    CHECK_CUDA(cudaMemcpy(h_output, ae->d_out,
                          out_bytes,
                          cudaMemcpyDeviceToHost));

    return loss_value;
}

void gpu_autoencoder_backward(GPUAutoencoder *ae, float lr)
{
    const int N = ae->N;
    const int H0 = ae->H; // 32
    const int W0 = ae->W; // 32
    const int K = 3;
    const int pad = 1;
    const int stride = 1;

    // Zero all gradient buffers
    CHECK_CUDA(cudaMemset(ae->d_gw1, 0, 256 * 3 * K * K * sizeof(float)));
    CHECK_CUDA(cudaMemset(ae->d_gb1, 0, 256 * sizeof(float)));
    CHECK_CUDA(cudaMemset(ae->d_gw2, 0, 128 * 256 * K * K * sizeof(float)));
    CHECK_CUDA(cudaMemset(ae->d_gb2, 0, 128 * sizeof(float)));
    CHECK_CUDA(cudaMemset(ae->d_gw3, 0, 128 * 128 * K * K * sizeof(float)));
    CHECK_CUDA(cudaMemset(ae->d_gb3, 0, 128 * sizeof(float)));
    CHECK_CUDA(cudaMemset(ae->d_gw4, 0, 256 * 128 * K * K * sizeof(float)));
    CHECK_CUDA(cudaMemset(ae->d_gb4, 0, 256 * sizeof(float)));
    CHECK_CUDA(cudaMemset(ae->d_gw5, 0, 3 * 256 * K * K * sizeof(float)));
    CHECK_CUDA(cudaMemset(ae->d_gb5, 0, 3 * sizeof(float)));

    dim3 block2d(16, 16);

    // ===== 1. dL/dout (MSE) =====
    int size_out = N * 3 * 32 * 32;
    {
        int t = 256;
        int b = (size_out + t - 1) / t;
        mse_loss_backward<<<b, t>>>(
            ae->d_out, ae->d_x0, ae->d_gout, size_out);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // ===== 2. Backward conv5: 256->3, 32x32 =====
    {
        int C_in = 256, C_out = 3;
        int H = 32, W = 32;

        dim3 gridIn(
            (W + block2d.x - 1) / block2d.x,
            (H + block2d.y - 1) / block2d.y,
            N * C_in);

        conv2d_backward_input_naive<<<gridIn, block2d>>>(
            ae->d_gout, ae->d_w5, ae->d_gu2,
            N, C_in, H, W, C_out, K, pad, stride);

        int num_w = C_out * C_in * K * K;
        int t = 256;
        int b = (num_w + t - 1) / t;
        conv2d_backward_weight_naive<<<b, t>>>(
            ae->d_u2, ae->d_gout, ae->d_gw5,
            N, C_in, H, W, C_out, K, pad, stride);

        int tb = 256;
        int bb = (C_out + tb - 1) / tb;
        conv2d_backward_bias_naive<<<bb, tb>>>(
            ae->d_gout, ae->d_gb5,
            N, C_out, H, W);

        sgd_update<<<b, t>>>(ae->d_w5, ae->d_gw5, num_w, lr);

        int bbp = (C_out + t - 1) / t;
        sgd_update<<<bbp, t>>>(ae->d_b5, ae->d_gb5, C_out, lr);
    }

    // ===== 3. UpSample2x2 backward =====
    {
        int C = 256;
        int H = 16, W = 16;

        dim3 grid(
            (W + block2d.x - 1) / block2d.x,
            (H + block2d.y - 1) / block2d.y,
            N * C);

        upsample2x2_backward<<<grid, block2d>>>(
            ae->d_gu2, ae->d_gh4,
            N, C, H, W);
    }

    // ===== 4. ReLU backward h4 =====
    {
        int size = N * 256 * 16 * 16;
        int t = 256;
        int b = (size + t - 1) / t;
        relu_backward<<<b, t>>>(
            ae->d_h4, ae->d_gh4, ae->d_gh4, size);
    }

    // ===== 5. conv4 backward =====
    {
        int C_in = 128, C_out = 256;
        int H = 16, W = 16;

        dim3 gridIn(
            (W + block2d.x - 1) / block2d.x,
            (H + block2d.y - 1) / block2d.y,
            N * C_in);

        conv2d_backward_input_naive<<<gridIn, block2d>>>(
            ae->d_gh4, ae->d_w4, ae->d_gu1,
            N, C_in, H, W, C_out, K, pad, stride);

        int num_w = C_out * C_in * K * K;
        int t = 256;
        int b = (num_w + t - 1) / t;
        conv2d_backward_weight_naive<<<b, t>>>(
            ae->d_u1, ae->d_gh4, ae->d_gw4,
            N, C_in, H, W, C_out, K, pad, stride);

        int tb = 256;
        int bb = (C_out + tb - 1) / tb;
        conv2d_backward_bias_naive<<<bb, tb>>>(
            ae->d_gh4, ae->d_gb4,
            N, C_out, H, W);

        sgd_update<<<b, t>>>(ae->d_w4, ae->d_gw4, num_w, lr);

        int bbp = (C_out + t - 1) / t;
        sgd_update<<<bbp, t>>>(ae->d_b4, ae->d_gb4, C_out, lr);
    }

    // ===== 6. UpSample2x2 backward =====
    {
        int C = 128;
        int H = 8, W = 8;

        dim3 grid(
            (W + block2d.x - 1) / block2d.x,
            (H + block2d.y - 1) / block2d.y,
            N * C);

        upsample2x2_backward<<<grid, block2d>>>(
            ae->d_gu1, ae->d_gh3,
            N, C, H, W);
    }

    // ===== 7. ReLU backward h3 =====
    {
        int size = N * 128 * 8 * 8;
        int t = 256;
        int b = (size + t - 1) / t;
        relu_backward<<<b, t>>>(
            ae->d_h3, ae->d_gh3, ae->d_gh3, size);
    }

    // ===== 8. conv3 backward =====
    {
        int C_in = 128, C_out = 128;
        int H = 8, W = 8;

        dim3 gridIn(
            (W + block2d.x - 1) / block2d.x,
            (H + block2d.y - 1) / block2d.y,
            N * C_in);

        conv2d_backward_input_naive<<<gridIn, block2d>>>(
            ae->d_gh3, ae->d_w3, ae->d_gp2,
            N, C_in, H, W, C_out, K, pad, stride);

        int num_w = C_out * C_in * K * K;
        int t = 256;
        int b = (num_w + t - 1) / t;
        conv2d_backward_weight_naive<<<b, t>>>(
            ae->d_p2, ae->d_gh3, ae->d_gw3,
            N, C_in, H, W, C_out, K, pad, stride);

        int tb = 256;
        int bb = (C_out + tb - 1) / tb;
        conv2d_backward_bias_naive<<<bb, tb>>>(
            ae->d_gh3, ae->d_gb3,
            N, C_out, H, W);

        sgd_update<<<b, t>>>(ae->d_w3, ae->d_gw3, num_w, lr);

        int bbp = (C_out + t - 1) / t;
        sgd_update<<<bbp, t>>>(ae->d_b3, ae->d_gb3, C_out, lr);
    }

    // ===== 9. MaxPool2x2 backward: P2 <- H2 =====
    {
        int C = 128;
        int H = 16, W = 16;
        
        CHECK_CUDA(cudaMemset(ae->d_gh2, 0, N * C * H * W * sizeof(float)));

        dim3 grid(
            (W/2 + block2d.x - 1) / block2d.x,
            (H/2 + block2d.y - 1) / block2d.y,
            N * C);

        maxpool2x2_backward<<<grid, block2d>>>(
            ae->d_h2, ae->d_gp2, ae->d_gh2,
            N, C, H, W);
    }

    // ===== 10. ReLU backward h2 =====
    {
        int size = N * 128 * 16 * 16;
        int t = 256;
        int b = (size + t - 1) / t;
        relu_backward<<<b, t>>>(
            ae->d_h2, ae->d_gh2, ae->d_gh2, size);
    }

    // ===== 11. conv2 backward =====
    {
        int C_in = 256, C_out = 128;
        int H = 16, W = 16;

        dim3 gridIn(
            (W + block2d.x - 1) / block2d.x,
            (H + block2d.y - 1) / block2d.y,
            N * C_in);

        conv2d_backward_input_naive<<<gridIn, block2d>>>(
            ae->d_gh2, ae->d_w2, ae->d_gp1,
            N, C_in, H, W, C_out, K, pad, stride);

        int num_w = C_out * C_in * K * K;
        int t = 256;
        int b = (num_w + t - 1) / t;
        conv2d_backward_weight_naive<<<b, t>>>(
            ae->d_p1, ae->d_gh2, ae->d_gw2,
            N, C_in, H, W, C_out, K, pad, stride);

        int tb = 256;
        int bb = (C_out + tb - 1) / tb;
        conv2d_backward_bias_naive<<<bb, tb>>>(
            ae->d_gh2, ae->d_gb2,
            N, C_out, H, W);

        sgd_update<<<b, t>>>(ae->d_w2, ae->d_gw2, num_w, lr);

        int bbp = (C_out + t - 1) / t;
        sgd_update<<<bbp, t>>>(ae->d_b2, ae->d_gb2, C_out, lr);
    }

    // ===== 12. MaxPool2x2 backward: P1 <- H1 =====
    {
        int C = 256;
        int H = 32, W = 32;

        // Zero gradient buffer
        CHECK_CUDA(cudaMemset(ae->d_gh1, 0, N * C * H * W * sizeof(float)));

        dim3 grid(
            (W/2 + block2d.x - 1) / block2d.x,
            (H/2 + block2d.y - 1) / block2d.y,
            N * C);

        maxpool2x2_backward<<<grid, block2d>>>(
            ae->d_h1, ae->d_gp1, ae->d_gh1,
            N, C, H, W);
    }

    // ===== 13. ReLU backward h1 =====
    {
        int size = N * 256 * 32 * 32;
        int t = 256;
        int b = (size + t - 1) / t;
        relu_backward<<<b, t>>>(
            ae->d_h1, ae->d_gh1, ae->d_gh1, size);
    }

    // ===== 14. conv1 backward =====
    {
        int C_in = 3, C_out = 256;
        int H = 32, W = 32;

        dim3 gridIn(
            (W + block2d.x - 1) / block2d.x,
            (H + block2d.y - 1) / block2d.y,
            N * C_in);

        conv2d_backward_input_naive<<<gridIn, block2d>>>(
            ae->d_gh1, ae->d_w1, ae->d_gx0,
            N, C_in, H, W, C_out, K, pad, stride);

        int num_w = C_out * C_in * K * K;
        int t = 256;
        int b = (num_w + t - 1) / t;
        conv2d_backward_weight_naive<<<b, t>>>(
            ae->d_x0, ae->d_gh1, ae->d_gw1,
            N, C_in, H, W, C_out, K, pad, stride);

        int tb = 256;
        int bb = (C_out + tb - 1) / tb;
        conv2d_backward_bias_naive<<<bb, tb>>>(
            ae->d_gh1, ae->d_gb1,
            N, C_out, H, W);

        sgd_update<<<b, t>>>(ae->d_w1, ae->d_gw1, num_w, lr);

        int bbp = (C_out + t - 1) / t;
        sgd_update<<<bbp, t>>>(ae->d_b1, ae->d_gb1, C_out, lr);
    }
}

void gpu_autoencoder_save_weights(GPUAutoencoder *ae, const char *filename)
{
    const int K = 3;
    int C_in1 = 3,   C_out1 = 256;
    int C_in2 = 256, C_out2 = 128;
    int C_in3 = 128, C_out3 = 128;
    int C_in4 = 128, C_out4 = 256;
    int C_in5 = 256, C_out5 = 3;

    size_t w1_cnt = C_out1 * C_in1 * K * K;
    size_t b1_cnt = C_out1;
    size_t w2_cnt = C_out2 * C_in2 * K * K;
    size_t b2_cnt = C_out2;
    size_t w3_cnt = C_out3 * C_in3 * K * K;
    size_t b3_cnt = C_out3;
    size_t w4_cnt = C_out4 * C_in4 * K * K;
    size_t b4_cnt = C_out4;
    size_t w5_cnt = C_out5 * C_in5 * K * K;
    size_t b5_cnt = C_out5;

    float *h_w1 = (float*)malloc(w1_cnt * sizeof(float));
    float *h_b1 = (float*)malloc(b1_cnt * sizeof(float));
    float *h_w2 = (float*)malloc(w2_cnt * sizeof(float));
    float *h_b2 = (float*)malloc(b2_cnt * sizeof(float));
    float *h_w3 = (float*)malloc(w3_cnt * sizeof(float));
    float *h_b3 = (float*)malloc(b3_cnt * sizeof(float));
    float *h_w4 = (float*)malloc(w4_cnt * sizeof(float));
    float *h_b4 = (float*)malloc(b4_cnt * sizeof(float));
    float *h_w5 = (float*)malloc(w5_cnt * sizeof(float));
    float *h_b5 = (float*)malloc(b5_cnt * sizeof(float));

    gpu_autoencoder_copy_weights_to_host(
        ae,
        h_w1, h_b1,
        h_w2, h_b2,
        h_w3, h_b3,
        h_w4, h_b4,
        h_w5, h_b5);

    FILE *f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "Cannot open %s for writing\n", filename);
    } else {
        fwrite(h_w1, sizeof(float), w1_cnt, f);
        fwrite(h_b1, sizeof(float), b1_cnt, f);
        fwrite(h_w2, sizeof(float), w2_cnt, f);
        fwrite(h_b2, sizeof(float), b2_cnt, f);
        fwrite(h_w3, sizeof(float), w3_cnt, f);
        fwrite(h_b3, sizeof(float), b3_cnt, f);
        fwrite(h_w4, sizeof(float), w4_cnt, f);
        fwrite(h_b4, sizeof(float), b4_cnt, f);
        fwrite(h_w5, sizeof(float), w5_cnt, f);
        fwrite(h_b5, sizeof(float), b5_cnt, f);
        fclose(f);
        printf("Saved weights to %s\n", filename);
    }

    free(h_w1); free(h_b1);
    free(h_w2); free(h_b2);
    free(h_w3); free(h_b3);
    free(h_w4); free(h_b4);
    free(h_w5); free(h_b5);
}

void gpu_autoencoder_load_weights(GPUAutoencoder *ae, const char *filename)
{
    const int K = 3;
    int C_in1 = 3,   C_out1 = 256;
    int C_in2 = 256, C_out2 = 128;
    int C_in3 = 128, C_out3 = 128;
    int C_in4 = 128, C_out4 = 256;
    int C_in5 = 256, C_out5 = 3;

    size_t w1_cnt = C_out1 * C_in1 * K * K;
    size_t b1_cnt = C_out1;
    size_t w2_cnt = C_out2 * C_in2 * K * K;
    size_t b2_cnt = C_out2;
    size_t w3_cnt = C_out3 * C_in3 * K * K;
    size_t b3_cnt = C_out3;
    size_t w4_cnt = C_out4 * C_in4 * K * K;
    size_t b4_cnt = C_out4;
    size_t w5_cnt = C_out5 * C_in5 * K * K;
    size_t b5_cnt = C_out5;

    float *h_w1 = (float*)malloc(w1_cnt * sizeof(float));
    float *h_b1 = (float*)malloc(b1_cnt * sizeof(float));
    float *h_w2 = (float*)malloc(w2_cnt * sizeof(float));
    float *h_b2 = (float*)malloc(b2_cnt * sizeof(float));
    float *h_w3 = (float*)malloc(w3_cnt * sizeof(float));
    float *h_b3 = (float*)malloc(b3_cnt * sizeof(float));
    float *h_w4 = (float*)malloc(w4_cnt * sizeof(float));
    float *h_b4 = (float*)malloc(b4_cnt * sizeof(float));
    float *h_w5 = (float*)malloc(w5_cnt * sizeof(float));
    float *h_b5 = (float*)malloc(b5_cnt * sizeof(float));

    FILE *f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Cannot open %s for reading\n", filename);
        exit(1);
    }

    size_t r1 = fread(h_w1, sizeof(float), w1_cnt, f);
    size_t r2 = fread(h_b1, sizeof(float), b1_cnt, f);
    size_t r3 = fread(h_w2, sizeof(float), w2_cnt, f);
    size_t r4 = fread(h_b2, sizeof(float), b2_cnt, f);
    size_t r5 = fread(h_w3, sizeof(float), w3_cnt, f);
    size_t r6 = fread(h_b3, sizeof(float), b3_cnt, f);
    size_t r7 = fread(h_w4, sizeof(float), w4_cnt, f);
    size_t r8 = fread(h_b4, sizeof(float), b4_cnt, f);
    size_t r9 = fread(h_w5, sizeof(float), w5_cnt, f);
    size_t r10 = fread(h_b5, sizeof(float), b5_cnt, f);
    fclose(f);

    if (r1 != w1_cnt || r2 != b1_cnt ||
        r3 != w2_cnt || r4 != b2_cnt ||
        r5 != w3_cnt || r6 != b3_cnt ||
        r7 != w4_cnt || r8 != b4_cnt ||
        r9 != w5_cnt || r10 != b5_cnt)
    {
        fprintf(stderr, "Error reading weights from %s\n", filename);
        exit(1);
    }

    gpu_autoencoder_copy_weights_to_device(
        ae,
        h_w1, h_b1,
        h_w2, h_b2,
        h_w3, h_b3,
        h_w4, h_b4,
        h_w5, h_b5
    );

    free(h_w1); free(h_b1);
    free(h_w2); free(h_b2);
    free(h_w3); free(h_b3);
    free(h_w4); free(h_b4);
    free(h_w5); free(h_b5);

    printf("Loaded weights from %s\n", filename);
}

void gpu_autoencoder_encode_batch(
    GPUAutoencoder *ae,
    const float *h_input,
    float *h_latent,
    int N_batch)
{
    const int H = ae->H;    // 32
    const int W = ae->W;    // 32
    const int K = 3;
    const int pad = 1;
    const int stride = 1;

    // Copy input [N_batch, 3, 32, 32] to GPU
    size_t in_bytes = (size_t)N_batch * 3 * H * W * sizeof(float);
    CHECK_CUDA(cudaMemcpy(ae->d_x0, h_input, in_bytes, cudaMemcpyHostToDevice));

    dim3 block2d(16, 16);

    // ===== ENCODER =====
    // conv1: 3 -> 256, 32x32 -> h1, ReLU + MaxPool -> p1 (16x16)
    {
        int C_in = 3, C_out = 256;
        int H_out = 32, W_out = 32;

        dim3 gridConv(
            (W_out + block2d.x - 1) / block2d.x,
            (H_out + block2d.y - 1) / block2d.y,
            N_batch * C_out
        );

        conv2d_forward_naive<<<gridConv, block2d>>>(
            ae->d_x0, ae->d_w1, ae->d_b1, ae->d_h1,
            N_batch, C_in, H, W, C_out, K, pad, stride);

        int size = N_batch * C_out * H_out * W_out;
        int t = 256;
        int b = (size + t - 1) / t;
        relu_forward<<<b, t>>>(ae->d_h1, size);

        int Hp = 16, Wp = 16;
        dim3 gridPool(
            (Wp + block2d.x - 1) / block2d.x,
            (Hp + block2d.y - 1) / block2d.y,
            N_batch * C_out
        );

        maxpool2x2_forward<<<gridPool, block2d>>>(
            ae->d_h1, ae->d_p1,
            N_batch, C_out, H_out, W_out);
    }

    // conv2: 256 -> 128, 16x16 -> h2, ReLU + MaxPool -> p2 (8x8)
    {
        int C_in = 256, C_out = 128;
        int H_in = 16, W_in = 16;
        int H_out = 16, W_out = 16;

        dim3 gridConv(
            (W_out + block2d.x - 1) / block2d.x,
            (H_out + block2d.y - 1) / block2d.y,
            N_batch * C_out
        );

        conv2d_forward_naive<<<gridConv, block2d>>>(
            ae->d_p1, ae->d_w2, ae->d_b2, ae->d_h2,
            N_batch, C_in, H_in, W_in, C_out, K, pad, stride);

        int size = N_batch * C_out * H_out * W_out;
        int t = 256;
        int b = (size + t - 1) / t;
        relu_forward<<<b, t>>>(ae->d_h2, size);

        int Hp = 8, Wp = 8;
        dim3 gridPool(
            (Wp + block2d.x - 1) / block2d.x,
            (Hp + block2d.y - 1) / block2d.y,
            N_batch * C_out
        );

        maxpool2x2_forward<<<gridPool, block2d>>>(
            ae->d_h2, ae->d_p2,
            N_batch, C_out, H_out, W_out);
    }

    // FIX: Copy latent [N_batch, 128, 8, 8] correctly
    size_t latent_bytes = (size_t)N_batch * 128 * 8 * 8 * sizeof(float);
    CHECK_CUDA(cudaMemcpy(h_latent, ae->d_p2, latent_bytes, cudaMemcpyDeviceToHost));
}
