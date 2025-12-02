#pragma once
#include "cpu_layers.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
// Định nghĩa kích thước Kernel/Stride/Padding
#define KERNEL_SIZE 3
#define POOL_SIZE 2
#define UPSAMPLE_SIZE 2
#define CONV_PADDING 1
#define CONV_STRIDE 1
#define POOL_STRIDE 2

typedef struct {
    int batch_size;
    double learning_rate;
    // kích thước input
    int input_height;       // 32
    int input_width;        // 32
    int input_channels;     // 3
    // weight, bias và gradient của từng lớp Conv2D
    float* w1; float* b1; float* d_w1; float* d_b1;
    float* w2; float* b2; float* d_w2; float* d_b2;
    float* w3; float* b3; float* d_w3; float* d_b3;
    float* w4; float* b4; float* d_w4; float* d_b4;
    float* w5; float* b5; float* d_w5; float* d_b5;
    
    float* batch_input;
    float* final_output;
    float* loss_gradient;
    // ouput và gradient của từng lớp Conv2D/MaxPool/UpSample
    float* conv1_output;   float* d_conv1_output; 
    float* pool1_output;   float* d_pool1_output;  
    float* conv2_output;   float* d_conv2_output; 
    float* pool2_output;   float* d_pool2_output; // LATENT SPACE
    float* conv3_output;   float* d_conv3_output;  
    float* upsample1_output; float* d_upsample1_output;
    float* conv4_output;   float* d_conv4_output;  
    float* upsample2_output; float* d_upsample2_output;
} CPUAutoEncoder;

void random_initialize(float* array, int size, float min, float max);
void initialize_autoencoder(CPUAutoEncoder* autoencoder, int batch_size, double learning_rate);
void free_autoencoder(CPUAutoEncoder* autoencoder);
void forward_autoencoder(CPUAutoEncoder* autoencoder);
void backward_autoencoder(CPUAutoEncoder* autoencoder);
void update_autoencoder_parameters(CPUAutoEncoder* autoencoder);
void save_weights(CPUAutoEncoder* autoencoder, const char* filename);