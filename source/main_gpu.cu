// the UPDATED main() code with argc/argv
// main_gpu.cu (DEBUG VERSION)
#include <cstdio>
#include <ctime>
#include <cuda_runtime.h>

#include "load_data.h"
#include "gpu_autoencoder_opt1.h"

// GpuTimer dùng cudaEvent để đo time (đúng spec Phase 2.5)
struct GpuTimer {
    cudaEvent_t start, stop;
    GpuTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    ~GpuTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    void tic() {
        cudaEventRecord(start, 0);
    }
    float toc() {
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

int main(int argc, char** argv) {
    srand((unsigned int)time(NULL));


    printf("[MAIN] Start program\n");
    fflush(stdout);

    // ---- Check GPU device ----
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "[MAIN] cudaGetDeviceCount error: %s\n",
                cudaGetErrorString(err));
        return 1;
    }
    printf("[MAIN] Num CUDA devices = %d\n", deviceCount);
    fflush(stdout);

    // ---- Load CIFAR-10 on CPU ----
    Cifar10 data;
    load_cifar10(&data);   // câu lệnh này in: CIFAR-10 loaded successfully ...
    printf("[MAIN] After load_cifar10\n");
    fflush(stdout);

    normalize_cifar10(&data);
    printf("[MAIN] After normalize_cifar10\n");
    fflush(stdout);

    GpuTimer epoch_timer;

    int batch_size = 64;
    int epochs     = 20;
    float lr       = 1e-3f;
    float total_time = 0.0f;

    int num_batches = TRAIN_NUM / batch_size;

    printf("[MAIN] Start training loop (epochs=%d, num_batches=%d)\n",
       epochs, num_batches);
    fflush(stdout);


    float *h_batch  = (float*)malloc(batch_size * IMG_SIZE * sizeof(float));
    float *h_output = (float*)malloc(batch_size * IMG_SIZE * sizeof(float)); // dùng làm buffer tạm

    // ---- Init GPU autoencoder ----
    GPUAutoencoder ae;
    gpu_autoencoder_init(&ae, batch_size);


    for (int epoch = 0; epoch < epochs; ++epoch) {
        shuffle_cifar10(&data);
        double epoch_loss = 0.0;

        epoch_timer.tic();

        for (int b = 0; b < num_batches; ++b) {
            get_next_batch(&data, batch_size, b, h_batch);

            float loss = gpu_autoencoder_forward(&ae, h_batch, h_output, true);
            gpu_autoencoder_backward(&ae, lr);

            epoch_loss += loss;

            if ((b + 1) % 100 == 0) {
                printf("[TRAIN] Epoch %d, batch %d/%d, loss = %f\n",
                       epoch, b + 1, num_batches, loss);
                fflush(stdout);
            }
        }

        float ms = epoch_timer.toc();
        total_time += ms;
        printf("==> Epoch %d done. Avg loss = %f, time = %.3f ms (%.3f s)\n",
               epoch, epoch_loss / num_batches, ms, ms / 1000.0f);
        fflush(stdout);
    }

    printf("[MAIN] Training finished\n");
    printf("[MAIN] Total training time = %.3f s\n", total_time / 1000.0f);
    fflush(stdout);

    // save weights
    gpu_autoencoder_save_weights(&ae, "ae_weights.bin");

    // ---- cleanup ----
    gpu_autoencoder_free(&ae);
    free(h_batch);
    free(h_output);
    free_cifar10(&data);

    printf("[MAIN] Program finished\n");
    fflush(stdout);

    return 0;
}
