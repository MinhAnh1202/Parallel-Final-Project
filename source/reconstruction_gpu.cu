// File này dùng cho cả GPU naive, GPU opt1 và GPU opt2
%%writefile reconstruction_gpu.cu
#include "load_data.h"
#include "gpu_autoencoder.h"
#include <stdint.h>

uint8_t float_to_pixel(float val) {
    if (val < 0.0f) val = 0.0f;
    if (val > 1.0f) val = 1.0f;
    return (uint8_t)(val * 255.0f);
}

void save_image_pnm(const char* filename, float* planar_data, int width, int height) {
    FILE* f = fopen(filename, "wb");
    if (!f) {
        printf("Error opening file %s for writing\n", filename);
        return;
    }

    // Header PNM: P6 format (binary)
    fprintf(f, "P6\n%d %d\n255\n", width, height);

    int plane_size = width * height;
    uint8_t* pixel_buffer = (uint8_t*)malloc(width * height * 3 * sizeof(uint8_t));
    if (!pixel_buffer) {
        printf("Error allocating pixel buffer\n");
        fclose(f);
        return;
    }

    // Convert float data to uint8_t and interleave RGB channels
    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            int pixel_idx = (h * width + w) * 3;
            int data_idx = h * width + w;

            pixel_buffer[pixel_idx] = float_to_pixel(planar_data[data_idx]);
            pixel_buffer[pixel_idx + 1] = float_to_pixel(planar_data[plane_size + data_idx]);
            pixel_buffer[pixel_idx + 2] = float_to_pixel(planar_data[2 * plane_size + data_idx]);
        }
    }

    // Write all pixel data at once
    fwrite(pixel_buffer, 1, width * height * 3, f);
    fclose(f);
    free(pixel_buffer);
}

void sample_reconstructions(GPUAutoencoder* ae, Cifar10* data,
                            int num_samples) {
    printf("\n*** Sampling Reconstructed Images ***\n");

    const int batch_size = ae->N;        // batch size lưu trong struct
    const int H = ae->H;                 // 32
    const int W = ae->W;                 // 32
    const int img_size = 3 * H * W;      // 3*32*32

    // Đảm bảo không lấy nhiều hơn batch
    if (num_samples > batch_size) {
        num_samples = batch_size;
    }

    // Cấp phát buffer host cho 1 batch input / output
    float* h_input  =
        (float*)malloc(batch_size * img_size * sizeof(float));
    float* h_output =
        (float*)malloc(batch_size * img_size * sizeof(float));

    if (!h_input || !h_output) {
        printf("Error allocating host buffers for batch\n");
        free(h_input);
        free(h_output);
        return;
    }

    // Ở đây mình đơn giản lấy batch đầu tiên của test set
    // data->test_images được giả định là [N_test, 3, 32, 32] NCHW planar
    memcpy(h_input,
           data->test_images,
           batch_size * img_size * sizeof(float));

    // Forward: copy h_input -> GPU, chạy autoencoder, copy d_out -> h_output
    // Không cần loss, nên compute_loss = false cho nhẹ
    gpu_autoencoder_forward(ae, h_input, h_output, false);

    char filename[64];

    for (int i = 0; i < num_samples; i++) {
        // Save original
        snprintf(filename, sizeof(filename),
                 "sample_%d_original.pnm", i);
        save_image_pnm(filename,
                       h_input + i * img_size,
                       W, H);

        // Save reconstructed
        snprintf(filename, sizeof(filename),
                 "sample_%d_reconstructed.pnm", i);
        save_image_pnm(filename,
                       h_output + i * img_size,
                       W, H);

        printf("Saved pair %d: %s vs %s\n",
               i, "original", "reconstructed");
    }

    free(h_input);
    free(h_output);
}
        
int main(int argc, char** argv) {
    srand((unsigned int)time(NULL)); 

    if (argc < 2) {
        fprintf(stderr,
                "Usage: %s <ae_weights_gpu_naive.bin>\n",
                argv[0]);
        return 1;
    }
    const char* weight_file = argv[1];
    
    //Load Data
    Cifar10 data;
    load_cifar10(&data);
    normalize_cifar10(&data);
    printf("Data loaded and normalized.\n");

    int batch_size = 64;
    GPUAutoencoder ae;
    gpu_autoencoder_init(&ae, batch_size);
    gpu_autoencoder_load_weights(&ae, weight_file);
    
    //Save 5 pairs of original and reconstructed images
    sample_reconstructions(&ae, &data, 5);
    
    // Free memory 
    gpu_autoencoder_free(&ae);
    free_cifar10(&data);
    
    return 0;
}