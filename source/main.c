#include "load_data.h"
#include <time.h>

int main(int argc, char** argv) {
    srand((unsigned int)time(NULL)); 
    
    Cifar10 data;
    load_cifar10(&data);
    normalize_cifar10(&data);

    int batch_size = 32; // Can be changed 
    int num_epochs = 20; // Can be changed 
    int num_batches = TRAIN_NUM / batch_size;
    float* batch_images = (float*)malloc(batch_size * IMG_SIZE * sizeof(float));

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        // Shuffle the training indices at the beginning of each epoch
        shuffle_cifar10(&data); 

        for (int batch_id = 0; batch_id < num_batches; batch_id++) {
            // Get the current batch data from the shuffled array
            get_next_batch(&data, batch_size, batch_id, batch_images);
            // forward + backward autoencoder on batch_images
        }
        printf("Epoch %d finished.\n", epoch);
    }

    free(batch_images);
    free_cifar10(&data);
    return 0;
}