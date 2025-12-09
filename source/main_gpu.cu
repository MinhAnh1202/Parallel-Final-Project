%%writefile main_gpu.cu
#include <cstdio>
#include <ctime>
#include <cuda_runtime.h>

#include "load_data.h"
#include "gpu_autoencoder.h"

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

size_t g_base_free = 0;

void init_base_mem() {
    size_t total;
    cudaMemGetInfo(&g_base_free, &total);
}

void print_program_mem(const char* msg) {
    size_t free_b, total_b;
    cudaMemGetInfo(&free_b, &total_b);
    float used_by_program_mb = (g_base_free > free_b)
        ? (g_base_free - free_b) / (1024.0f * 1024.0f)
        : 0.0f;

    printf("[PROG MEM] %s: approx used by this program = %.2f MB\n",
           msg, used_by_program_mb);
}

int main(int argc, char** argv) {
    srand((unsigned int)time(NULL));

    printf("[MAIN] Start program\n");
    fflush(stdout);

    init_base_mem();
    print_program_mem("At start");

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
    load_cifar10(&data);   // in: CIFAR-10 loaded successfully ...
    printf("[MAIN] After load_cifar10\n");
    fflush(stdout);

    print_program_mem("After load_cifar10");

    normalize_cifar10(&data);
    printf("[MAIN] After normalize_cifar10\n");
    fflush(stdout);

    // ---- Mở file log GPU (giống format CPU) ----
    FILE* log_gpu = fopen("training_gpu.txt", "w");
    if (!log_gpu) {
        fprintf(stderr, "[MAIN] Cannot open training_gpu.txt for writing\n");
        return 1;
    }

    GpuTimer epoch_timer;

    int batch_size = 64;
    int epochs     = 20;
    float lr       = 1e-3f;

    int num_batches = TRAIN_NUM / batch_size;

    printf("[MAIN] Start training loop (epochs=%d, num_batches=%d)\n",
           epochs, num_batches);
    fflush(stdout);

    float *h_batch  = (float*)malloc(batch_size * IMG_SIZE * sizeof(float));
    float *h_output = (float*)malloc(batch_size * IMG_SIZE * sizeof(float)); // buffer tạm

    // ---- Init GPU autoencoder ----
    GPUAutoencoder ae;
    gpu_autoencoder_init(&ae, batch_size);
    print_program_mem("After model alloc");

    // Biến để tích lũy total time & final loss
    double total_gpu_time_ms = 0.0;
    double final_loss = 0.0;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        shuffle_cifar10(&data);
        double epoch_loss = 0.0;

        // In giống CPU
        printf("Epoch %d/%d\n", epoch + 1, epochs);
        fflush(stdout);
        fprintf(log_gpu, "Epoch %d/%d\n", epoch + 1, epochs);

        epoch_timer.tic();

        for (int b = 0; b < num_batches; ++b) {
            get_next_batch(&data, batch_size, b, h_batch);

            float loss = gpu_autoencoder_forward(&ae, h_batch, h_output, true);
            gpu_autoencoder_backward(&ae, lr);

            epoch_loss += loss;

            if ((b + 1) % 100 == 0) {
                printf("[TRAIN] Epoch %d, batch %d/%d, loss = %f\n",
                       epoch + 1, b + 1, num_batches, loss);
                fflush(stdout);
            }
        }

        float ms = epoch_timer.toc();
        total_gpu_time_ms += ms;

        double avg_loss = epoch_loss / num_batches;
        final_loss = avg_loss;  // loss của epoch cuối sẽ là final loss

        double epoch_time_sec = ms / 1000.0;

        // In ra màn hình
        printf("Epoch %d finished. Avg Loss: %f, time: %.2f seconds\n",
               epoch + 1, avg_loss, epoch_time_sec);
        fflush(stdout);

        // Ghi giống hệt CPU vào file training_gpu.txt
        fprintf(log_gpu,
                "Epoch %d finished. Avg Loss: %f, time: %.2f seconds\n",
                epoch + 1, avg_loss, epoch_time_sec);
        fflush(log_gpu);
    }

    printf("[MAIN] Training finished\n");
    fflush(stdout);
    print_program_mem("At end");

    // ---- SUMMARY trên màn hình ----
    printf("\n*** Training Summary (GPU) ***\n");
    printf("Total training time: %.2f seconds.\n", total_gpu_time_ms / 1000.0);
    printf("Final reconstruction loss: %f\n", final_loss);
    fflush(stdout);

    // ---- SUMMARY ghi xuống training_gpu.txt ----
    fprintf(log_gpu, "\n*** Training Summary ***\n");
    fprintf(log_gpu, "Total training time: %.2f seconds.\n",
            total_gpu_time_ms / 1000.0);
    fprintf(log_gpu, "Final reconstruction loss: %f\n", final_loss);
    fclose(log_gpu);

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
