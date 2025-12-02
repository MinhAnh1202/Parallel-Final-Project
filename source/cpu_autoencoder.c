#include "cpu_autoencoder.h"

// Hàm khởi tạo mảng trọng số với giá trị ngẫu nhiên trong khoảng [min, max]
void random_initialize(float* array, int size, float min, float max) {
    for (int i = 0; i < size; i++) {
        float scale = (float)rand() / (float)RAND_MAX; 
        array[i] = min + scale * (max - min); 
    }
}

void zero_initialize(float* array, int size) {
    for (int i = 0; i < size; i++) {
        array[i] = 0.0f;
    }
}


void initialize_conv_layer(float** w, float** b, float** dw, float** db, int C_in, int C_out) {
    int size_W = C_out * C_in * KERNEL_SIZE * KERNEL_SIZE;
    *w = (float*)malloc(size_W * sizeof(float));
    *b = (float*)malloc(C_out * sizeof(float));
    *dw = (float*)malloc(size_W * sizeof(float));
    *db = (float*)malloc(C_out * sizeof(float));

    random_initialize(*w, size_W, -0.05f, 0.05f);
    random_initialize(*b, C_out, -0.05f, 0.05f);
    zero_initialize(*dw, size_W);
    zero_initialize(*db, C_out);
}

float* allocate_buffer(int batch_size, int H, int W, int C) {
    int size = batch_size * H * W * C;
    return (float*)malloc(size * sizeof(float));
}


void initialize_autoencoder(CPUAutoEncoder* autoencoder, int batch_size, double learning_rate) {
    // Tham số chung
    autoencoder->batch_size = batch_size;
    autoencoder->learning_rate = learning_rate;
    autoencoder->input_height = 32;
    autoencoder->input_width = 32;
    autoencoder->input_channels = 3;

    // Output channels của các lớp
    int C_in = 3, C1 = 256, C2 = 128, C3 = 128, C4 = 256, C5 = 3; 
    // Kích thước không gian (Pixel/kênh)
    int P1 = 32 * 32, P2 = 16 * 16, P3 = 8 * 8; 
    // Khởi tạo trọng số, bias và gradient cho từng lớp Conv2D
    initialize_conv_layer(&autoencoder->w1, &autoencoder->b1, &autoencoder->d_w1, &autoencoder->d_b1, C_in, C1);
    initialize_conv_layer(&autoencoder->w2, &autoencoder->b2, &autoencoder->d_w2, &autoencoder->d_b2, C1, C2);
    initialize_conv_layer(&autoencoder->w3, &autoencoder->b3, &autoencoder->d_w3, &autoencoder->d_b3, C2, C3);
    initialize_conv_layer(&autoencoder->w4, &autoencoder->b4, &autoencoder->d_w4, &autoencoder->d_b4, C3, C4);
    initialize_conv_layer(&autoencoder->w5, &autoencoder->b5, &autoencoder->d_w5, &autoencoder->d_b5, C4, C5);
    // Khởi tạo Buffers cho activations và gradients
    int input_height = 32, input_width = 32;
    autoencoder->batch_input = allocate_buffer(batch_size, input_height, input_width, C_in);
    autoencoder->final_output = allocate_buffer(batch_size, input_height, input_width, C5); // Output size (32x32x3)
    autoencoder->loss_gradient = allocate_buffer(batch_size, input_height, input_width, C5);
    // Layer 1 (Conv1): 32x32x256
    autoencoder->conv1_output = allocate_buffer(batch_size, input_height, input_width, C1);
    autoencoder->d_conv1_output = allocate_buffer(batch_size, input_height, input_width, C1);
    // Layer 2 (Pool1): 16x16x256
    int H2 = 16, W2 = 16;
    autoencoder->pool1_output = allocate_buffer(batch_size, H2, W2, C1);
    autoencoder->d_pool1_output = allocate_buffer(batch_size, H2, W2, C1);
    // Layer 3 (Conv2): 16x16x128
    autoencoder->conv2_output = allocate_buffer(batch_size, H2, W2, C2);
    autoencoder->d_conv2_output = allocate_buffer(batch_size, H2, W2, C2);
    // Layer 4 (Pool2 - Latent): 8x8x128
    int H3 = 8, W3 = 8;
    autoencoder->pool2_output = allocate_buffer(batch_size, H3, W3, C2);
    autoencoder->d_pool2_output = allocate_buffer(batch_size, H3, W3, C2);
    // Layer 5 (Conv3): 8x8x128
    autoencoder->conv3_output = allocate_buffer(batch_size, H3, W3, C3);
    autoencoder->d_conv3_output = allocate_buffer(batch_size, H3, W3, C3);
    // Layer 6 (UpSample1): 16x16x128
    autoencoder->upsample1_output = allocate_buffer(batch_size, H2, W2, C3);
    autoencoder->d_upsample1_output = allocate_buffer(batch_size, H2, W2, C3);
    // Layer 7 (Conv4): 16x16x256
    autoencoder->conv4_output = allocate_buffer(batch_size, H2, W2, C4);
    autoencoder->d_conv4_output = allocate_buffer(batch_size, H2, W2, C4);
    // Layer 8 (UpSample2): 32x32x256
    autoencoder->upsample2_output = allocate_buffer(batch_size, input_height, input_width, C4);
    autoencoder->d_upsample2_output = allocate_buffer(batch_size, input_height, input_width, C4);
}

void free_autoencoder(CPUAutoEncoder* autoencoder) {
    // Giải phóng trọng số và gradient
    free(autoencoder->w1); free(autoencoder->b1); free(autoencoder->d_w1); free(autoencoder->d_b1);
    free(autoencoder->w2); free(autoencoder->b2); free(autoencoder->d_w2); free(autoencoder->d_b2);
    free(autoencoder->w3); free(autoencoder->b3); free(autoencoder->d_w3); free(autoencoder->d_b3);
    free(autoencoder->w4); free(autoencoder->b4); free(autoencoder->d_w4); free(autoencoder->d_b4);
    free(autoencoder->w5); free(autoencoder->b5); free(autoencoder->d_w5); free(autoencoder->d_b5);

    // Giải phóng buffers activation/gradient
    free(autoencoder->batch_input);
    free(autoencoder->final_output);
    free(autoencoder->loss_gradient);
    free(autoencoder->conv1_output); free(autoencoder->d_conv1_output);
    free(autoencoder->pool1_output); free(autoencoder->d_pool1_output);
    free(autoencoder->conv2_output); free(autoencoder->d_conv2_output);
    free(autoencoder->pool2_output); free(autoencoder->d_pool2_output);
    free(autoencoder->conv3_output); free(autoencoder->d_conv3_output);
    free(autoencoder->upsample1_output); free(autoencoder->d_upsample1_output);
    free(autoencoder->conv4_output); free(autoencoder->d_conv4_output);
    free(autoencoder->upsample2_output); free(autoencoder->d_upsample2_output);
}

// Forward
void forward_autoencoder(CPUAutoEncoder* autoencoder) {
    int bs = autoencoder->batch_size;
    
    // Kích thước activation của 1 ảnh tại các lớp
    int size_input = 32 * 32 * 3;
    int size_L1 = 32 * 32 * 256;
    int size_L2 = 16 * 16 * 256;
    int size_L3 = 16 * 16 * 128;
    int size_L4 = 8 * 8 * 128; // Latent
    // Decoder sizes
    int size_L5 = 8 * 8 * 128;
    int size_L6 = 16 * 16 * 128;
    int size_L7 = 16 * 16 * 256;
    int size_L8 = 32 * 32 * 256;
    int size_Out = 32 * 32 * 3;
    for (int b = 0; b < bs; b++) {
        // Tính offset con trỏ cho ảnh thứ b
        float* ptr_input = autoencoder->batch_input + b * size_input;
        float* ptr_L1 = autoencoder->conv1_output + b * size_L1;
        float* ptr_L2 = autoencoder->pool1_output + b * size_L2;
        float* ptr_L3 = autoencoder->conv2_output + b * size_L3;
        float* ptr_L4 = autoencoder->pool2_output + b * size_L4;
        float* ptr_L5 = autoencoder->conv3_output + b * size_L5;
        float* ptr_L6 = autoencoder->upsample1_output + b * size_L6;
        float* ptr_L7 = autoencoder->conv4_output + b * size_L7;
        float* ptr_L8 = autoencoder->upsample2_output + b * size_L8;
        float* ptr_Out = autoencoder->final_output + b * size_Out;
        // --- ENCODER ---
        // L1: Conv1 + ReLU
        Conv2D_Forward(ptr_input, 32, 32, 3, autoencoder->w1, KERNEL_SIZE, KERNEL_SIZE, autoencoder->b1, CONV_PADDING, CONV_STRIDE, 256, ptr_L1, 32, 32);
        Relu(ptr_L1, size_L1, ptr_L1);

        // L2: Pool1
        MaxPool2D_Forward(ptr_L1, 32, 32, POOL_SIZE, POOL_SIZE, POOL_STRIDE, 256, ptr_L2, 16, 16);

        // L3: Conv2 + ReLU
        Conv2D_Forward(ptr_L2, 16, 16, 256, autoencoder->w2, KERNEL_SIZE, KERNEL_SIZE, autoencoder->b2, CONV_PADDING, CONV_STRIDE, 128, ptr_L3, 16, 16);
        Relu(ptr_L3, size_L3, ptr_L3);

        // L4: Pool2 (Latent)
        MaxPool2D_Forward(ptr_L3, 16, 16, POOL_SIZE, POOL_SIZE, POOL_STRIDE, 128, ptr_L4, 8, 8);

        // --- DECODER ---
        // L5: Conv3 + ReLU
        Conv2D_Forward(ptr_L4, 8, 8, 128, autoencoder->w3, KERNEL_SIZE, KERNEL_SIZE, autoencoder->b3, CONV_PADDING, CONV_STRIDE, 128, ptr_L5, 8, 8);
        Relu(ptr_L5, size_L5, ptr_L5);

        // L6: UpSample1
        UpSample2D_Forward(ptr_L5, 8, 8, UPSAMPLE_SIZE, 128, ptr_L6, 16, 16);

        // L7: Conv4 + ReLU
        Conv2D_Forward(ptr_L6, 16, 16, 128, autoencoder->w4, KERNEL_SIZE, KERNEL_SIZE, autoencoder->b4, CONV_PADDING, CONV_STRIDE, 256, ptr_L7, 16, 16);
        Relu(ptr_L7, size_L7, ptr_L7);

        // L8: UpSample2
        UpSample2D_Forward(ptr_L7, 16, 16, UPSAMPLE_SIZE, 256, ptr_L8, 32, 32);

        // L9: Conv5 (Output)
        Conv2D_Forward(ptr_L8, 32, 32, 256, autoencoder->w5, KERNEL_SIZE, KERNEL_SIZE, autoencoder->b5, CONV_PADDING, CONV_STRIDE, 3, ptr_Out, 32, 32);
    }
}


// Backward
void backward_autoencoder(CPUAutoEncoder* autoencoder) {
    int bs = autoencoder->batch_size;
    int total_elements = bs * 32 * 32 * 3;
    MSE_Gradient(autoencoder->batch_input, autoencoder->final_output, total_elements, autoencoder->loss_gradient);
    // Khởi tạo gradient về 0 trước khi cộng dồn
    zero_initialize(autoencoder->d_w1, 256*3*3*3); zero_initialize(autoencoder->d_b1, 256);
    zero_initialize(autoencoder->d_w2, 128*256*3*3); zero_initialize(autoencoder->d_b2, 128);
    zero_initialize(autoencoder->d_w3, 128*128*3*3); zero_initialize(autoencoder->d_b3, 128);
    zero_initialize(autoencoder->d_w4, 256*128*3*3); zero_initialize(autoencoder->d_b4, 256);
    zero_initialize(autoencoder->d_w5, 3*256*3*3); zero_initialize(autoencoder->d_b5, 3);   
    // Kích thước 1 ảnh tại các lớp (như Forward)
    int size_Out = 32*32*3;
    int size_L8 = 32*32*256;
    int size_L7 = 16*16*256;
    int size_L6 = 16*16*128;
    int size_L5 = 8*8*128;
    int size_L4 = 8*8*128;
    int size_L3 = 16*16*128;
    int size_L2 = 16*16*256;
    int size_L1 = 32*32*256;
    int size_In = 32*32*3;

    // 3. Vòng lặp Batch cho Backward
    for (int b = 0; b < bs; b++) {
        // Offset pointers
        float* ptr_dOut = autoencoder->loss_gradient + b * size_Out;
        float* ptr_Upsample2_Out = autoencoder->upsample2_output + b * size_L8;
        float* ptr_d_Upsample2_Out = autoencoder->d_upsample2_output + b * size_L8;
        float* ptr_d_Conv4_Out = autoencoder->d_conv4_output + b * size_L7;
        float* ptr_Upsample1_Out = autoencoder->upsample1_output + b * size_L6;
        float* ptr_d_Upsample1_Out = autoencoder->d_upsample1_output + b * size_L6;
        float* ptr_d_Conv3_Out = autoencoder->d_conv3_output + b * size_L5;
        float* ptr_Pool2_Out = autoencoder->pool2_output + b * size_L4;
        float* ptr_d_Pool2_Out = autoencoder->d_pool2_output + b * size_L4;
        float* ptr_Conv2_Out = autoencoder->conv2_output + b * size_L3;
        float* ptr_d_Conv2_Out = autoencoder->d_conv2_output + b * size_L3;
        float* ptr_Pool1_Out = autoencoder->pool1_output + b * size_L2;
        float* ptr_d_Pool1_Out = autoencoder->d_pool1_output + b * size_L2;
        float* ptr_Conv1_Out = autoencoder->conv1_output + b * size_L1;
        float* ptr_d_Conv1_Out = autoencoder->d_conv1_output + b * size_L1;
        float* ptr_Input = autoencoder->batch_input + b * size_In;

        // === L9 (Conv5) ===
        // dW5, dB5 
        Conv2D_Backward_Kernel(ptr_dOut, 32, 32, ptr_Upsample2_Out, 32, 32, 256, 3, 3, 1, 1, 3, autoencoder->d_w5);
        Conv2D_Backward_Biases(ptr_dOut, 32, 32, 3, autoencoder->d_b5);
        // dInput cho L8
        Conv2D_Backward_Input(ptr_dOut, 32, 32, autoencoder->w5, 3, 3, 32, 32, 256, 1, 1, 3, ptr_d_Upsample2_Out);

        // === L8 (Upsample2) ===
        UpSample2D_Backward(ptr_d_Upsample2_Out, 32, 32, UPSAMPLE_SIZE, 256, autoencoder->d_conv4_output + b * size_L7, 16, 16);

        // === L7 (Conv4) ===
        // ReLU Backward 
        Relu_Backward(ptr_d_Conv4_Out, autoencoder->conv4_output + b * size_L7, 16*16*256);
        Conv2D_Backward_Kernel(ptr_d_Conv4_Out, 16, 16, ptr_Upsample1_Out, 16, 16, 128, 3, 3, 1, 1, 256, autoencoder->d_w4);
        Conv2D_Backward_Biases(ptr_d_Conv4_Out, 16, 16, 256, autoencoder->d_b4);
        Conv2D_Backward_Input(ptr_d_Conv4_Out, 16, 16, autoencoder->w4, 3, 3, 16, 16, 128, 1, 1, 256, ptr_d_Upsample1_Out);

        // === L6 (Upsample1) ===
        UpSample2D_Backward(ptr_d_Upsample1_Out, 16, 16, UPSAMPLE_SIZE, 128, ptr_d_Conv3_Out, 8, 8);

        // === L5 (Conv3) ===
        Relu_Backward(ptr_d_Conv3_Out, autoencoder->conv3_output + b * size_L5, 8*8*128);
        Conv2D_Backward_Kernel(ptr_d_Conv3_Out, 8, 8, ptr_Pool2_Out, 8, 8, 128, 3, 3, 1, 1, 128, autoencoder->d_w3);
        Conv2D_Backward_Biases(ptr_d_Conv3_Out, 8, 8, 128, autoencoder->d_b3);
        Conv2D_Backward_Input(ptr_d_Conv3_Out, 8, 8, autoencoder->w3, 3, 3, 8, 8, 128, 1, 1, 128, ptr_d_Pool2_Out);

        // === L4 (Pool2) ===
        MaxPool2D_Backward(ptr_d_Pool2_Out, 8, 8, ptr_Conv2_Out, 16, 16, 2, 2, 2, 128, ptr_d_Conv2_Out);

        // === L3 (Conv2) ===
        Relu_Backward(ptr_d_Conv2_Out, ptr_Conv2_Out, 16*16*128);
        Conv2D_Backward_Kernel(ptr_d_Conv2_Out, 16, 16, ptr_Pool1_Out, 16, 16, 256, 3, 3, 1, 1, 128, autoencoder->d_w2);
        Conv2D_Backward_Biases(ptr_d_Conv2_Out, 16, 16, 128, autoencoder->d_b2);
        Conv2D_Backward_Input(ptr_d_Conv2_Out, 16, 16, autoencoder->w2, 3, 3, 16, 16, 256, 1, 1, 128, ptr_d_Pool1_Out);

        // === L2 (Pool1) ===
        MaxPool2D_Backward(ptr_d_Pool1_Out, 16, 16, ptr_Conv1_Out, 32, 32, 2, 2, 2, 256, ptr_d_Conv1_Out);

        // === L1 (Conv1) ===
        Relu_Backward(ptr_d_Conv1_Out, ptr_Conv1_Out, 32*32*256);
        Conv2D_Backward_Kernel(ptr_d_Conv1_Out, 32, 32, ptr_Input, 32, 32, 3, 3, 3, 1, 1, 256, autoencoder->d_w1);
        Conv2D_Backward_Biases(ptr_d_Conv1_Out, 32, 32, 256, autoencoder->d_b1);
    }
}

void update_autoencoder_parameters(CPUAutoEncoder* autoencoder) {
    // Cập nhật tất cả 5 lớp Conv: W += -learning_rate * dW
    int size_W1 = 256 * 3 * 3 * 3; 
    SGD_Update(autoencoder->w1, autoencoder->d_w1, autoencoder->learning_rate, size_W1);
    SGD_Update(autoencoder->b1, autoencoder->d_b1, autoencoder->learning_rate, 256);
    int size_W2 = 128 * 256 * 3 * 3; 
    SGD_Update(autoencoder->w2, autoencoder->d_w2, autoencoder->learning_rate, size_W2);
    SGD_Update(autoencoder->b2, autoencoder->d_b2, autoencoder->learning_rate, 128);
    int size_W3  = 128 * 128 * 3 * 3;   
    SGD_Update(autoencoder->w3, autoencoder->d_w3, autoencoder->learning_rate, size_W3);
    SGD_Update(autoencoder->b3, autoencoder->d_b3, autoencoder->learning_rate, 128);
    int size_W4 = 256 * 128 * 3 * 3;
    SGD_Update(autoencoder->w4, autoencoder->d_w4, autoencoder->learning_rate, size_W4);
    SGD_Update(autoencoder->b4, autoencoder->d_b4, autoencoder->learning_rate, 256);
    int size_W5 = 3 * 256 * 3 * 3;
    SGD_Update(autoencoder->w5, autoencoder->d_w5, autoencoder->learning_rate, size_W5);
    SGD_Update(autoencoder->b5, autoencoder->d_b5, autoencoder->learning_rate, 3);
}

void save_weights(CPUAutoEncoder* autoencoder, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (file == NULL) {
        printf("Error opening file for writing weights.\n");
        return;
    }
    // Lưu trọng số và bias của từng lớp Conv2D
    fwrite(autoencoder->w1, sizeof(float), 256*3*3*3, file);
    fwrite(autoencoder->b1, sizeof(float), 256, file);
    fwrite(autoencoder->w2, sizeof(float), 128*256*3*3, file);
    fwrite(autoencoder->b2, sizeof(float), 128, file);
    fwrite(autoencoder->w3, sizeof(float), 128*128*3*3, file);
    fwrite(autoencoder->b3, sizeof(float), 128, file);
    fwrite(autoencoder->w4, sizeof(float), 256*128*3*3, file);
    fwrite(autoencoder->b4, sizeof(float), 256, file);
    fwrite(autoencoder->w5, sizeof(float), 3*256*3*3, file);
    fwrite(autoencoder->b5, sizeof(float), 3, file);
    fclose(file);
}