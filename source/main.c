#include "load_data.h"
#include "cpu_autoencoder.h"
#include <time.h>

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
            if ((batch_id + 1) % 100 == 0) printf("[TRAIN]Epoch %d, batch %d/%d, loss = %f\n", epoch, batch_id + 1, num_batches, current_loss);
        }
        int end_time = clock();
        double epoch_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
        printf("==> Epoch %d finished. Avg Loss: %f, time: %.2f seconds\n", epoch + 1, epoch_loss / num_batches, epoch_time);
    }

    //Save weights after training
    save_weights(&autoencoder, "autoencoder_weights_cpu.bin");
    // Free memory (Only free after training is done)
    free_autoencoder(&autoencoder);
    free(batch_images);
    free_cifar10(&data);
    
    return 0;
}