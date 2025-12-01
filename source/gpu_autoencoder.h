//%%writefile gpu_autoencoder.h
// header for GPUAutoencoder (the struct + declarations)
#pragma once
#include "gpu_layers.h"

// This autoencoder matches the project architecture exactly.
// Layout: NCHW [batch, channels, height, width]
struct GPUAutoencoder {
    int N;   // batch size
    int H;   // 32
    int W;   // 32;

    // --- Conv layer parameters ---
    // conv1: 3 -> 256 (3x3)
    float *d_w1, *d_b1;
    // conv2: 256 -> 128
    float *d_w2, *d_b2;
    // conv3: 128 -> 128
    float *d_w3, *d_b3;
    // conv4: 128 -> 256
    float *d_w4, *d_b4;
    // conv5: 256 -> 3
    float *d_w5, *d_b5;

    // --- Activations ---
    // Input batch
    float *d_x0;   // [N, 3, 32, 32]

    // Encoder
    float *d_h1;   // conv1 out: [N, 256, 32, 32]
    float *d_p1;   // pool1   : [N, 256, 16, 16]
    float *d_h2;   // conv2   : [N, 128, 16, 16]
    float *d_p2;   // pool2   : [N, 128,  8,  8]   (latent)

    // Decoder
    float *d_h3;   // conv3   : [N, 128,  8,  8]
    float *d_u1;   // up1     : [N, 128, 16, 16]
    float *d_h4;   // conv4   : [N, 256, 16, 16]
    float *d_u2;   // up2     : [N, 256, 32, 32]
    float *d_out;  // conv5   : [N,   3, 32, 32]

    // Loss buffer
    float *d_loss; // single float on device

    // ---- gradients for activations ----
    float *d_gx0;
    float *d_gh1;
    float *d_gp1;
    float *d_gh2;
    float *d_gp2;
    float *d_gh3;
    float *d_gu1;
    float *d_gh4;
    float *d_gu2;
    float *d_gout;

    // ---- gradients for weights ----
    float *d_gw1, *d_gb1;
    float *d_gw2, *d_gb2;
    float *d_gw3, *d_gb3;
    float *d_gw4, *d_gb4;
    float *d_gw5, *d_gb5;
};

// API
void gpu_autoencoder_init(GPUAutoencoder *ae, int batch_size);
void gpu_autoencoder_free(GPUAutoencoder *ae);

void gpu_autoencoder_copy_weights_to_host(
    GPUAutoencoder *ae,
    float *h_w1, float *h_b1,
    float *h_w2, float *h_b2,
    float *h_w3, float *h_b3,
    float *h_w4, float *h_b4,
    float *h_w5, float *h_b5);

void gpu_autoencoder_copy_weights_to_device(
    GPUAutoencoder *ae,
    const float *h_w1, const float *h_b1,
    const float *h_w2, const float *h_b2,
    const float *h_w3, const float *h_b3,
    const float *h_w4, const float *h_b4,
    const float *h_w5, const float *h_b5);


// Forward on GPU:
//   h_input  : host pointer [N * 3 * 32 * 32]
//   h_output : host pointer [N * 3 * 32 * 32]
//   returns loss value (MSE(x_hat, x)) if compute_loss=true;
//   otherwise returns 0.0f.
float gpu_autoencoder_forward(
    GPUAutoencoder *ae,
    const float *h_input,
    float *h_output,
    bool compute_loss = true);

void gpu_autoencoder_backward(GPUAutoencoder *ae, float lr);

void gpu_autoencoder_save_weights(GPUAutoencoder *ae, const char *filename);
