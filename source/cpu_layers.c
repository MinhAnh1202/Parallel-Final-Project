#include "cpu_layers.h"

void Relu(float* input, int N, float* output) {
    for (int i = 0; i < N; i++) {
        output[i] = input[i] > 0.0f ? input[i] : 0.0f;
    }
}

void Conv2D_Forward(float* input, int input_width, int input_height, int input_channels,
    float* kernel, int kernel_width, int kernel_height,
    float* biases, int padding, int stride, int filter_count,
    float* output, int output_height, int output_width) 
{
    // // Tính toán kích thước output
    // int H_out = (input_height + 2 * padding - kernel_height) / stride + 1;
    // int W_out = (input_width + 2 * padding - kernel_width) / stride + 1;
    // int output_size = filter_count * H_out * W_out;
    // output = (float*)malloc(output_size * sizeof(float));
    // if (output == NULL) {
    //     fprintf(stderr, "ERROR: Memory allocation failed!\n");
    //     return;
    // }
    // Lặp qua kênh đầu ra (filter)
    for (int c_out = 0; c_out < filter_count; c_out++) {  
        // Lặp qua chiều cao output
        for (int h_out = 0; h_out < output_height; h_out++) {
            // Lặp qua chiều rộng output
            for (int w_out = 0; w_out < output_width; w_out++) {
                float sum = 0.0f;
                // Lặp qua kênh đầu vào (c_in)
                for (int c_in = 0; c_in < input_channels; c_in++) {
                    // Lặp qua kernel height
                    for (int k_h = 0; k_h < kernel_height; k_h++) {
                        // Lặp qua kernel width
                        for (int k_w = 0; k_w < kernel_width; k_w++) {
                            // Vị trí input tương ứng
                            int h_in = h_out * stride + k_h - padding;
                            int w_in = w_out * stride + k_w - padding;
                            float val = 0.0f; 
                            // Kiểm tra zero padding
                            if (h_in >= 0 && h_in < input_height && w_in >= 0 && w_in < input_width) {
                                int channel_size = input_width * input_height;
                                val = input[c_in * channel_size + h_in * input_width + w_in];
                            }
                            
                            int weight_idx = c_out * input_channels * kernel_height * kernel_width + c_in * kernel_height * kernel_width + 
                                            k_h * kernel_width + k_w;
                            
                            sum += val * kernel[weight_idx];
                        }
                    }
                } 
                sum += biases[c_out];  // Thêm bias
                int output_idx = h_out * output_width + w_out + c_out * output_width * output_height;
                output[output_idx] = sum;
            }
        }
    }
}


void MaxPool2D_Forward(float* input, int input_width, int input_height, int filter_width, int filter_height, int stride, int filter_count, 
    float* output, int output_height, int output_width) {
    // int H_out = (input_height - filter_height) / stride + 1;
    // int W_out = (input_width - filter_width) / stride + 1;
    
    int plane_size_in = input_height * input_width;
    int plane_size_out = output_height * output_width;
    for (int c = 0; c < filter_count; c++) {
        for (int h_out = 0; h_out < output_height; h_out++) {
            for (int w_out = 0; w_out < output_width; w_out++) {
                float max_val = -FLT_MAX;
                int h_start = h_out * stride;
                int w_start = w_out * stride;
                for (int fh = 0; fh < filter_height; fh++) {
                    for (int fw = 0; fw < filter_width; fw++) {
                        int h_in = h_start + fh;
                        int w_in = w_start + fw;
                        int input_idx = c * plane_size_in + h_in * input_width + w_in;
                        float val = input[input_idx];
                        if (val > max_val) {
                            max_val = val;
                        }
                    }
                }
                int output_idx = c * plane_size_out + h_out * output_width + w_out;
                output[output_idx] = max_val;
            }
        }
    }   
}

void UpSample2D_Forward(float* input, int input_width, int input_height,
int scale_factor, int filter_count, float* output, int output_height, int output_width) {
    int plane_size_in = input_height * input_width;
    int plane_size_out = output_height * output_width;
    for (int c = 0; c < filter_count; c++) {
        for (int h_in = 0; h_in < input_height; h_in++) {
            for (int w_in = 0; w_in < input_width; w_in++) {
                float val = input[c * plane_size_in + h_in * input_width + w_in];
                for (int sh = 0; sh < scale_factor; sh++) { // Gấp đôi hàng
                    for (int sw = 0; sw < scale_factor; sw++) { // Gấp đôi cột
                        int h_out = h_in * scale_factor + sh;
                        int w_out = w_in * scale_factor + sw;
                        int output_idx = c * plane_size_out + h_out * output_width + w_out;
                        output[output_idx] = val;
                    }
                }
            }
        }
    }
}

float MSE(float* input, float* output, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += (output[i] - input[i]) * (output[i] - input[i]);
    }
    return sum / size;
}

void Relu_Backward(float* d_output, float* input,int N) {
    for (int i = 0; i < N; i++) {
        d_output[i] = input[i] > 0.0f ? d_output[i] : 0.0f;
    }
}

void MSE_Gradient(float* input, float* output, int size, float* d_output) {
    float sum = 0.0f;
    float factor = 2.0f / size;
    for (int i = 0; i < size; i++) {
        d_output[i] = factor * (output[i] - input[i]);
    }
}

void MaxPool2D_Backward(float* d_output, int d_output_width, int d_output_height, float* input,
    int input_width, int input_height, int filter_width, int filter_height, int stride, int filter_count,
    float* d_input) 
{
    // Chỉ gán vị trí giá trị max của input ban đầu là gradient của lớp tiếp theo (d_output), còn lại là 0
    int plane_size_in = input_height * input_width;
    int plane_size_out = d_output_height * d_output_width;
    int total_input_size = filter_count * plane_size_in;
    for (int i = 0; i < total_input_size; i++) { // Khởi tạo gradient của input ban đầu là 0
        d_input[i] = 0.0f;
    }

    for (int c = 0; c < filter_count; c++) {
        
        int channel_offset_in = c * plane_size_in;
        int channel_offset_out = c * plane_size_out;

        for (int h_out = 0; h_out < d_output_height; h_out++) {
            for (int w_out = 0; w_out < d_output_width; w_out++) {
                
                int h_start = h_out * stride;
                int w_start = w_out * stride;

                float max_val = -FLT_MAX;
                int max_input_idx = -1;
                
                for (int fh = 0; fh < filter_height; fh++) {
                    for (int fw = 0; fw < filter_width; fw++) {
                        int h_in = h_start + fh;
                        int w_in = w_start + fw;
                        int input_idx = channel_offset_in + h_in * input_width + w_in;
                        float val = input[input_idx];
                        if (val > max_val) {
                            max_val = val;
                            max_input_idx = input_idx;
                        }
                    }
                }
                //Lấy gradient từ output
                if (max_input_idx != -1) {
                    int output_idx = channel_offset_out + h_out * d_output_width + w_out;
                    d_input[max_input_idx] += d_output[output_idx]; 
                }
            }
        }
    }  
}

void UpSample2D_Backward(float* d_output, int d_output_width, int d_output_height, int scale_factor, int filter_count,
    float* d_input, int d_input_height, int d_input_width) {

    int plane_size_in = d_input_height * d_input_width;
    int plane_size_out = d_output_height * d_output_width;
    int total_input_size = filter_count * plane_size_in;
    for (int i = 0; i < total_input_size; i++) {
        d_input[i] = 0.0f;
    }
    for (int c = 0; c < filter_count; c++) {
        int channel_offset_in = c * plane_size_in;
        int channel_offset_out = c * plane_size_out;
        // Lặp qua input grid (d_input)
        for (int h_in = 0; h_in < d_input_height; h_in++) {
            for (int w_in = 0; w_in < d_input_width; w_in++) {
                float sum_gradient = 0.0f;
                int h_start_out = h_in * scale_factor;
                int w_start_out = w_in * scale_factor;
                for (int sh = 0; sh < scale_factor; sh++) {
                    for (int sw = 0; sw < scale_factor; sw++) {
                        int h_out = h_start_out + sh;
                        int w_out = w_start_out + sw;
                        if (h_out < d_output_height && w_out < d_output_width) {
                            int output_idx = channel_offset_out + h_out * d_output_width + w_out;
                            sum_gradient += d_output[output_idx];
                        }
                    }
                }
                int input_idx = channel_offset_in + h_in * d_input_width + w_in;
                d_input[input_idx] = sum_gradient;
            }
        }
    }
}

void Conv2D_Backward_Input(float* d_output, int d_output_width, int d_output_height, float* kernel, int kernel_width, int kernel_height, 
    int input_width, int input_height, int input_channels, int padding, int stride, int filter_count, float* d_input) {
    // Thực hiện tích chập giữa dE/dO và kernel (xoay 180 độ) để tính dE/dI
    int plane_size_in = input_height * input_width;
    int plane_size_out = d_output_height * d_output_width;
    // Lặp qua kênh input (kênh output gradient)
    for (int c_in = 0; c_in < input_channels; c_in++) { 
        // Lặp qua input grid (d_input)
        for (int h_in = 0; h_in < input_height; h_in++) {
            for (int w_in = 0; w_in < input_width; w_in++) {
                float sum_gradient = 0.0f;
                // Lặp qua kênh output (số lượng filter)
                for (int c_out = 0; c_out < filter_count; c_out++) {
                    // Lặp qua kernel (xoay 180 độ)
                    for (int kh = 0; kh < kernel_height; kh++) {
                        for (int kw = 0; kw < kernel_width; kw++) {
                            int h_out = h_in - kh + padding;
                            int w_out = w_in - kw + padding;
                            float d_output_val = 0.0f;
                            // Kiểm tra padding
                            if (h_out >= 0 && h_out < d_output_height && w_out >= 0 && w_out < d_output_width) {
                                int d_output_idx = c_out * plane_size_out + h_out * d_output_width + w_out;
                                d_output_val = d_output[d_output_idx];
                            }
                            // Tính chỉ số kernel (xoay 180 độ)
                            int kernel_idx = c_out * input_channels * kernel_height * kernel_width + c_in * kernel_height * kernel_width + 
                                            (kernel_height - 1 - kh) * kernel_width + (kernel_width - 1 - kw);
                            
                            sum_gradient += d_output_val * kernel[kernel_idx];
                        }
                    }
                } 
                int d_input_idx = c_in * plane_size_in + h_in * input_width + w_in;
                d_input[d_input_idx] = sum_gradient;
            }
        }
    }
}

void Conv2D_Backward_Kernel(float* d_output, int d_output_width, int d_output_height, float* input,
int input_width, int input_height, int input_channels, int kernel_width, int kernel_height, int padding, int stride, int filter_count, float* d_weights) {
    // Thực hiện tích chập giữa dE/dO và input để tính dE/dW
    int plane_size_in = input_height * input_width;
    int plane_size_out = d_output_height * d_output_width; 
    // Lặp qua kênh đầu ra (filter)
    for (int c_out = 0; c_out < filter_count; c_out++) {
        // Lặp qua kênh đầu vào
        for (int c_in = 0; c_in < input_channels; c_in++) {
            // Lặp qua kích thước kernel
            for (int k_h = 0; k_h < kernel_height; k_h++) {
                for (int k_w = 0; k_w < kernel_width; k_w++) {
                    float sum_gradient = 0.0f;
                    // Lặp qua output grid (d_output) để tích lũy
                    for (int h_out = 0; h_out < d_output_height; h_out++) {
                        for (int w_out = 0; w_out < d_output_width; w_out++) {
                            int h_in = h_out * stride + k_h - padding;
                            int w_in = w_out * stride + k_w - padding;
                            float input_val = 0.0f;
                            // Kiểm tra padding
                            if (h_in >= 0 && h_in < input_height && w_in >= 0 && w_in < input_width) {
                                int input_idx = c_in * plane_size_in + h_in * input_width + w_in;
                                input_val = input[input_idx];
                            }
                            int d_output_idx = c_out * plane_size_out + h_out * d_output_width + w_out;
                            float d_output_val = d_output[d_output_idx];
                            // Tính gradient
                            sum_gradient += input_val * d_output_val;
                        }
                    } 
                    int kernel_idx = c_out * input_channels * kernel_height * kernel_width + c_in * kernel_height * kernel_width + 
                                    k_h * kernel_width + k_w;               
                    d_weights[kernel_idx] += sum_gradient;
                }
            }
        }
    }
}

void Conv2D_Backward_Biases(float* d_output, int d_output_width, int d_output_height, 
    int filter_count, float* d_biases) {
    int plane_size_out = d_output_height * d_output_width;
    // Lặp qua từng filter
    for (int c_out = 0; c_out < filter_count; c_out++) {
        float sum_gradient = 0.0f;
        // Lặp qua từng vị trí trong output
        for (int h_out = 0; h_out < d_output_height; h_out++) {
            for (int w_out = 0; w_out < d_output_width; w_out++) {
                // Tính chỉ số trong mảng d_output
                int output_idx = c_out * plane_size_out + h_out * d_output_width + w_out;
                // Cộng dồn gradient từ d_output
                sum_gradient += d_output[output_idx];
            }
        }
        d_biases[c_out] += sum_gradient;
    }
}

void SGD_Update(float* weights, float* d_weights, double learning_rate, int N_params) {
    for (int i = 0; i < N_params; i++) {
        weights[i] -= (learning_rate * d_weights[i]);
    }
}