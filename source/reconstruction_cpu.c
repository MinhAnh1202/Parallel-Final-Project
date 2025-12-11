#include "load_data.h"
#include "cpu_autoencoder.h"
#include <time.h>
#include <sys/resource.h>
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

void sample_reconstructions(CPUAutoEncoder* ae, Cifar10* data, int num_samples) {
    printf("\n*** Sampling Reconstructed Images ***\n");
    int batch_size = ae->batch_size;
    
    float* sample_batch = (float*)malloc(batch_size * 32 * 32 * 3 * sizeof(float));
    if (!sample_batch) {
        printf("Error allocating sample batch\n");
        return;
    }

    memcpy(ae->batch_input, data->test_images, batch_size * 32 * 32 * 3 * sizeof(float));
    forward_autoencoder(ae);
    char filename[64];
    int img_size = 32 * 32 * 3;

    for (int i = 0; i < num_samples; i++) {
        // Save original image
        snprintf(filename, sizeof(filename), "sample_%d_original.pnm", i);
        save_image_pnm(filename, ae->batch_input + i * img_size, 32, 32);

        // Save reconstructed image
        snprintf(filename, sizeof(filename), "sample_%d_reconstructed.pnm", i);
        save_image_pnm(filename, ae->final_output + i * img_size, 32, 32);
        
        printf("Saved pair %d: %s vs %s\n", i, "original", "reconstructed");
    }

    free(sample_batch);
}
        
int main(int argc, char** argv) {
    srand((unsigned int)time(NULL)); 

    if (argc < 2) {
        fprintf(stderr,
                "Usage: %s <autoencoder_weights_cpu.bin>\n",
                argv[0]);
        return 1;
    }
    const char* weight_file = argv[1];
    
    //Load Data
    Cifar10 data;
    load_cifar10(&data);
    normalize_cifar10(&data);
    printf("Data loaded and normalized.\n");

    int batch_size = 32;
    float learning_rate = 0.001;

    // Initialize AutoEncoder
    CPUAutoEncoder autoencoder;
    initialize_autoencoder(&autoencoder, batch_size, learning_rate);

    // Load weights
    cpu_load_weights(&autoencoder, weight_file);
    
    //Save 5 pairs of original and reconstructed images
    sample_reconstructions(&autoencoder, &data, 5);
    
    // Free memory 
    free_autoencoder(&autoencoder);
    free_cifar10(&data);
    
    return 0;
}