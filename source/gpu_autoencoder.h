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
};

// API
void gpu_autoencoder_init(GPUAutoencoder *ae, int batch_size);
void gpu_autoencoder_free(GPUAutoencoder *ae);

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
