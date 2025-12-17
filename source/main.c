#include "load_data.h"
#include "cpu_autoencoder.h"
#include <time.h>
#include <sys/resource.h>
#include <stdint.h>

void print_memory_usage() {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        double memory_usage_mb = usage.ru_maxrss / 1024.0;
        double memory_usage_gb = memory_usage_mb / 1024.0;
        printf("[SYSTEM] Memory Usage: %.2f MB (%.4f GB)\n", memory_usage_mb, memory_usage_gb);
    } else {
        printf("[SYSTEM] Error checking memory usage.\n");
    }
}

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
    
    //Load Data
    Cifar10 data;
    load_cifar10(&data);
    normalize_cifar10(&data);
    printf("Data loaded and normalized.\n");

    // Hyperparameters
    int train_subset_size = 1000;
    int batch_size = 32; // Can be changed 
    int num_epochs = 20; // Can be changed 
    int num_batches = train_subset_size / batch_size;
    float learning_rate = 0.001; 
    float* batch_images = (float*)malloc(batch_size * IMG_SIZE * sizeof(float));
    double total_time = 0.0;
    double final_loss = 0.0;

    // Initialize AutoEncoder
    CPUAutoEncoder autoencoder;
    initialize_autoencoder(&autoencoder, batch_size, learning_rate);
    printf("Autoencoder initialized (batch_size=%d, learning_rate=%f)\n", batch_size, learning_rate);
    printf("Start training...\n");
    // Training Loop
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        // Shuffle the training indices at the beginning of each epoch
        int start_time = clock();
        shuffle_cifar10(&data); 
        double epoch_loss = 0.0;
        printf("Epoch %d/%d\n", epoch + 1, num_epochs);
        for (int batch_id = 0; batch_id < num_batches; batch_id++) {
            // Get the current batch data from the shuffled array
            get_next_batch(&data, batch_size, batch_id, batch_images);
            // forward + backward autoencoder on batch_images
            // copy into autoencoder input buffer
            for (int i = 0; i < batch_size * IMG_SIZE; i++) {
                autoencoder.batch_input[i] = batch_images[i];
            }
            // Training process
            forward_autoencoder(&autoencoder);
            // Calculate loss for display
            float current_loss = MSE(autoencoder.batch_input, autoencoder.final_output, batch_size * IMG_SIZE);
            epoch_loss += current_loss;
            backward_autoencoder(&autoencoder);
            update_autoencoder_parameters(&autoencoder);
            if ((batch_id + 1) % 100 == 0) {
                printf("[TRAIN] Epoch %d, batch %d/%d, loss = %f\n", epoch + 1, batch_id + 1, num_batches, current_loss);
            }
        }
        int end_time = clock();
        double epoch_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
        total_time += epoch_time;
        printf("==> Epoch %d finished. Avg Loss: %f, time: %.2f seconds\n", epoch + 1, epoch_loss / num_batches, epoch_time);
        if(epoch + 1 == 20) final_loss = epoch_loss / num_batches;
    }

    printf("\n*** Training Summary ***\n");
    printf("Total training time: %.2f seconds.\n", total_time);
    printf("Final reconstruction loss: %f\n", final_loss);
    print_memory_usage();
    
    //Save weights after training
    save_weights(&autoencoder, "autoencoder_weights_cpu.bin");
    //Save 5 pairs of original and reconstructed images
    sample_reconstructions(&autoencoder, &data, 5);
    // Free memory 
    free_autoencoder(&autoencoder);
    free(batch_images);
    free_cifar10(&data);
    
    return 0;
}