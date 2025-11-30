//%%writefile gpu_autoencoder.cu
// your GPUAutoencoder implementation
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
#include "gpu_layers.h"
#include "gpu_autoencoder.h"

static inline float rand_uniform(float scale = 0.01f) {
    return scale * ( (float)rand() / (float)RAND_MAX - 0.5f );
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
        for (size_t i = 0; i < w_cnt; ++i) h_w[i] = rand_uniform(0.01f);
        for (size_t i = 0; i < b_cnt; ++i) h_b[i] = 0.0f;
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
}

// Forward pass for ONE batch (no backward yet)
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
        CHECK_CUDA(cudaDeviceSynchronize());

        // ReLU
        int size = N * C_out * H_out * W_out;
        int t = 256;
        int b = (size + t - 1) / t;
        relu_forward<<<b, t>>>(ae->d_h1, size);
        CHECK_CUDA(cudaDeviceSynchronize());

        // MaxPool 2x2 -> 16x16
        int Hp = 16, Wp = 16;
        dim3 gridPool(
            (Wp + block2d.x - 1) / block2d.x,
            (Hp + block2d.y - 1) / block2d.y,
            N * C_out);

        maxpool2x2_forward<<<gridPool, block2d>>>(
            ae->d_h1, ae->d_p1,
            N, C_out, H_out, W_out);
        CHECK_CUDA(cudaDeviceSynchronize());
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
        CHECK_CUDA(cudaDeviceSynchronize());

        int size = N * C_out * H_out * W_out;
        int t = 256;
        int b = (size + t - 1) / t;
        relu_forward<<<b, t>>>(ae->d_h2, size);
        CHECK_CUDA(cudaDeviceSynchronize());

        // pool -> 8x8
        int Hp = 8, Wp = 8;
        dim3 gridPool(
            (Wp + block2d.x - 1) / block2d.x,
            (Hp + block2d.y - 1) / block2d.y,
            N * C_out);

        maxpool2x2_forward<<<gridPool, block2d>>>(
            ae->d_h2, ae->d_p2,
            N, C_out, H_out, W_out);
        CHECK_CUDA(cudaDeviceSynchronize());
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
        CHECK_CUDA(cudaDeviceSynchronize());

        int size = N * C_out * H_out * W_out;
        int t = 256;
        int b = (size + t - 1) / t;
        relu_forward<<<b, t>>>(ae->d_h3, size);
        CHECK_CUDA(cudaDeviceSynchronize());

        // upsample 8x8 -> 16x16
        int Hu = 16, Wu = 16;
        dim3 gridUp(
            (Wu + block2d.x - 1) / block2d.x,
            (Hu + block2d.y - 1) / block2d.y,
            N * C_out);

        upsample2x2_forward<<<gridUp, block2d>>>(
            ae->d_h3, ae->d_u1,
            N, C_out, H_in, W_in);
        CHECK_CUDA(cudaDeviceSynchronize());
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
        CHECK_CUDA(cudaDeviceSynchronize());

        int size = N * C_out * H_out * W_out;
        int t = 256;
        int b = (size + t - 1) / t;
        relu_forward<<<b, t>>>(ae->d_h4, size);
        CHECK_CUDA(cudaDeviceSynchronize());

        // upsample 16x16 -> 32x32
        int Hu = 32, Wu = 32;
        dim3 gridUp(
            (Wu + block2d.x - 1) / block2d.x,
            (Hu + block2d.y - 1) / block2d.y,
            N * C_out);

        upsample2x2_forward<<<gridUp, block2d>>>(
            ae->d_h4, ae->d_u2,
            N, C_out, H_in, W_in);
        CHECK_CUDA(cudaDeviceSynchronize());
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
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // ------------- (optional) compute MSE loss -------------
    float loss_value = 0.0f;
    if (compute_loss) {
        int size = N * 3 * 32 * 32;
        CHECK_CUDA(cudaMemset(ae->d_loss, 0, sizeof(float)));
        int t = 256;
        int b = (size + t - 1) / t;
        mse_loss_forward<<<b, t>>>(
            ae->d_out, ae->d_x0, ae->d_loss, size);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(&loss_value, ae->d_loss,
                              sizeof(float),
                              cudaMemcpyDeviceToHost));
    }

    // ------------- copy output back to host -------------
    size_t out_bytes = N * 3 * 32 * 32 * sizeof(float);
    CHECK_CUDA(cudaMemcpy(h_output, ae->d_out,
                          out_bytes,
                          cudaMemcpyDeviceToHost));

    return loss_value;
}
