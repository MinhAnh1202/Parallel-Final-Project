#pragma once
#include <stdio.h>
#include <float.h>

void Relu(float* input, int N, float* output);
void Conv2D_Forward(float* input, int input_width, int input_height, int input_channels,
    float* kernel, int kernel_width, int kernel_height,
    float* biases, int padding, int stride, int filter_count,
    float* output, int output_height, int output_width);
void MaxPool2D_Forward(float* input, int input_width, int input_height, 
    int filter_width, int filter_height, int stride, int filter_count, 
    float* output, int output_height, int output_width);
void UpSample2D_Forward(float* input, int input_width, int input_height,
    int scale_factor, int filter_count,
    float* output, int output_height, int output_width);
float MSE(float* input, float* output, int size);
void Relu_Backward(float* d_output, float* input,int N);
void MSE_Gradient(float* input, float* output, int size, float* d_output);
void UpSample2D_Backward(float* d_output, int d_output_width, int d_output_height, int scale_factor, int filter_count,
    float* d_input, int d_input_height, int d_input_width);
void MaxPool2D_Backward(float* d_output, int d_output_width, int d_output_height, float* input,
    int input_width, int input_height, int filter_width, int filter_height, int stride, int filter_count, float* d_input);
void Conv2D_Backward_Input(float* d_output, int d_output_width, int d_output_height, float* kernel, int kernel_width, int kernel_height, 
    int input_width, int input_height, int input_channels, int padding, int stride, int filter_count, float* d_input);
void Conv2D_Backward_Kernel(float* d_output, int d_output_width, int d_output_height, float* input,
    int input_width, int input_height, int input_channels, int kernel_width, int kernel_height, int padding, int stride, int filter_count, float* d_weights);
void Conv2D_Backward_Biases(float* d_output, int d_output_width, int d_output_height, int filter_count, float* d_biases);
void SGD_Update(float* weights, float* d_weights, double learning_rate, int N_params);
