#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

extern "C" {
  #include "load_data.h"
  //#include "gpu_autoencoder.h"
  #include "cpu_autoencoder.h"
}

#define AE_LATENT_DIM 128 * 8 * 8

// ghi 1 dòng theo format LIBSVM: label index:val ...
void write_svm_line(FILE* f, int label,
                    const float* feat, int dim)
{
    fprintf(f, "%d", label);

    // In toàn bộ feature, không bỏ qua giá trị 0
    for (int j = 0; j < dim; ++j) {
        float v = feat[j];
        fprintf(f, " %d:%g", j + 1, v);
    }
    fprintf(f, "\n");
}


int main(int argc, char** argv)
{
    char* weight_file_cpu = "autoencoder_weights_cpu.bin";
    float learning_rate = 0.001;

    printf("[SVM] Loading CIFAR-10...\n");
    Cifar10 data;
    load_cifar10(&data);
    normalize_cifar10(&data);

    // batch_size cho encoder khi extract feature
    int batch_size = 64;
    //GPUAutoencoder ae;
    //gpu_autoencoder_init(&ae, batch_size);
    //gpu_autoencoder_load_weights(&ae, weight_file);

    CPUAutoEncoder autoencoder;
    initialize_autoencoder(&autoencoder, batch_size, learning_rate);
    cpu_load_weights(&autoencoder, weight_file_cpu);


    float* h_batch  = (float*)malloc(batch_size * IMG_SIZE * sizeof(float));
    float* h_latent = (float*)malloc(batch_size * AE_LATENT_DIM * sizeof(float));
    if (!h_batch || !h_latent) {
        fprintf(stderr, "Host malloc failed\n");
        return 1;
    }

    // ====== TRAIN: 50k ảnh -> train_svm.txt ======
    FILE* f_train = fopen("train_svm.txt", "w");
    if (!f_train) {
        perror("train_svm.txt");
        return 1;
    }

    int N_train           = 10000;//TRAIN_NUM; // 50000
    int num_batches_train = (N_train + batch_size - 1) / batch_size;

    printf("[SVM] Extracting train features...\n");
    for (int b = 0; b < num_batches_train; ++b) {
        int start = b * batch_size;
        int cur_bs = batch_size;
        if (start + cur_bs > N_train) {
            cur_bs = N_train - start;
        }

        // copy ảnh [start, start+cur_bs) vào h_batch
        for (int i = 0; i < cur_bs; ++i) {
            int idx = start + i;
            memcpy(h_batch + i * IMG_SIZE,
                   data.train_images + idx * IMG_SIZE,
                   IMG_SIZE * sizeof(float));
        }

        // encoder-only
        //gpu_autoencoder_encode_batch(&ae, h_batch, h_latent, cur_bs);
        cpu_extract_features(&autoencoder, h_batch, cur_bs, h_latent);


        // ghi ra file theo format LIBSVM
        for (int i = 0; i < cur_bs; ++i) {
            int idx = start + i;
            int label = data.train_labels[idx];
            const float* feat = h_latent + i * AE_LATENT_DIM;
            write_svm_line(f_train, label, feat, AE_LATENT_DIM);
        }

        printf("[SVM][TRAIN] Batch %d/%d done\n",
               b + 1, num_batches_train);
        fflush(stdout);
    }
    fclose(f_train);
    printf("[SVM] Saved train_svm.txt\n");

    // ====== TEST: 10k ảnh -> test_svm.txt ======
    FILE* f_test = fopen("test_svm.txt", "w");
    if (!f_test) {
        perror("test_svm.txt");
        return 1;
    }

    int N_test           = 2000;//TEST_NUM; // 10000
    int num_batches_test = (N_test + batch_size - 1) / batch_size;

    printf("[SVM] Extracting test features...\n");
    for (int b = 0; b < num_batches_test; ++b) {
        int start = b * batch_size;
        int cur_bs = batch_size;
        if (start + cur_bs > N_test) {
            cur_bs = N_test - start;
        }

        for (int i = 0; i < cur_bs; ++i) {
            int idx = start + i;
            memcpy(h_batch + i * IMG_SIZE,
                   data.test_images + idx * IMG_SIZE,
                   IMG_SIZE * sizeof(float));
        }

        // **Không còn debug cudaMemcpy w1/b1, không in input nữa**

        //gpu_autoencoder_encode_batch(&ae, h_batch, h_latent, cur_bs);
        cpu_extract_features(&autoencoder, h_batch, cur_bs, h_latent);

        for (int i = 0; i < cur_bs; ++i) {
            int idx = start + i;
            int label = data.test_labels[idx];
            const float* feat = h_latent + i * AE_LATENT_DIM;
            write_svm_line(f_test, label, feat, AE_LATENT_DIM);
        }

        printf("[SVM][TEST] Batch %d/%d done\n",
               b + 1, num_batches_test);
        fflush(stdout);
    }
    fclose(f_test);
    printf("[SVM] Saved test_svm.txt\n");

    // cleanup
    //gpu_autoencoder_free(&ae);
    free_autoencoder(&autoencoder);
    free(h_batch);
    free(h_latent);
    free_cifar10(&data);

    printf("[SVM] Done.\n");
    return 0;
}
