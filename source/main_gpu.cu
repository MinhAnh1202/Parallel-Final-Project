//%%writefile main_gpu.cu
// the UPDATED main() code with argc/argv
#include <cstdio>
#include <ctime>
#include "load_data.h"
#include "gpu_autoencoder.h"   // your header

int main(int argc, char** argv) {
    srand((unsigned int)time(NULL));

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <path_to_cifar-10-batches-bin>\n", argv[0]);
        return 1;
    }
    const char* data_dir = argv[1];

    // ---- Load CIFAR-10 on CPU ----
    Cifar10 data;
    load_cifar10(&data, data_dir);
    normalize_cifar10(&data);

    int batch_size = 64;           // GPU phase suggests 64
    int num_batches = TRAIN_NUM / batch_size;

    float *h_batch  = (float*)malloc(batch_size * IMG_SIZE * sizeof(float));
    float *h_output = (float*)malloc(batch_size * IMG_SIZE * sizeof(float));

    // ---- Init GPU autoencoder ----
    GPUAutoencoder ae;
    gpu_autoencoder_init(&ae, batch_size);

    // Take one batch, just to test forward
    shuffle_cifar10(&data);
    get_next_batch(&data, batch_size, 0, h_batch);

    float loss = gpu_autoencoder_forward(&ae, h_batch, h_output, true);
    printf("Single GPU forward done. MSE loss = %f\n", loss);

    // inspect first few output pixels
    for (int i = 0; i < 10; ++i) {
        printf("%f ", h_output[i]);
    }
    printf("\n");

    // ---- cleanup ----
    gpu_autoencoder_free(&ae);
    free(h_batch);
    free(h_output);
    free_cifar10(&data);

    return 0;
}
