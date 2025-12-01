//%%writefile load_data.cu
#include "load_data.h"

static void read_batch(const char* filename, float* images_start, uint8_t* labels) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        perror(filename);
        exit(EXIT_FAILURE);
    }

    uint8_t buffer[3073];
    for (int i = 0; i < 10000; i++) {
        if (fread(buffer, 1, 3073, f) != 3073) {
            fprintf(stderr, "Error: incomplete read in %s at image %d\n", filename, i);
            fclose(f);
            exit(EXIT_FAILURE);
        }
        labels[i] = buffer[0];
        for (int j = 0; j < 3072; j++) {
            images_start[i * 3072 + j] = (float)buffer[1 + j];  // uint8 -> float
        }
    }
    fclose(f);
}

// must match the prototype in load_data.h
void load_cifar10(Cifar10* data, const char* data_dir) {
    data->train_images = (float*)malloc(TRAIN_NUM * IMG_SIZE * sizeof(float));
    data->test_images  = (float*)malloc(TEST_NUM  * IMG_SIZE * sizeof(float));
    data->train_labels = (uint8_t*)malloc(TRAIN_NUM * sizeof(uint8_t));
    data->test_labels  = (uint8_t*)malloc(TEST_NUM  * sizeof(uint8_t));

    if (!data->train_images || !data->test_images ||
        !data->train_labels  || !data->test_labels) {
        fprintf(stderr, "ERROR: Memory allocation failed!\n");
        exit(EXIT_FAILURE);
    }

    data->train_indices = (int*)malloc(TRAIN_NUM * sizeof(int));
    for (int i = 0; i < TRAIN_NUM; i++) {
        data->train_indices[i] = i;
    }

    // Load training data: data_batch_1.bin ... data_batch_5.bin
    for (int i = 1; i <= 5; i++) {
        char filename[512];
        snprintf(filename, sizeof(filename), "%s/data_batch_%d.bin", data_dir, i);
        read_batch(filename,
                   data->train_images + (i-1) * 10000 * IMG_SIZE,
                   data->train_labels + (i-1) * 10000);
    }

    // Load test data: test_batch.bin
    char test_file[512];
    snprintf(test_file, sizeof(test_file), "%s/test_batch.bin", data_dir);
    read_batch(test_file,
               data->test_images, data->test_labels);

    printf("CIFAR-10 loaded successfully from %s\n", data_dir);
}

void normalize_cifar10(Cifar10* data) {
    for (size_t i = 0; i < (size_t)TRAIN_NUM * IMG_SIZE; i++) {
        data->train_images[i] /= 255.0f;
    }
    for (size_t i = 0; i < (size_t)TEST_NUM * IMG_SIZE; i++) {
        data->test_images[i] /= 255.0f;
    }
}

void shuffle_cifar10(Cifar10* data) {
    for (int i = TRAIN_NUM - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = data->train_indices[i];
        data->train_indices[i] = data->train_indices[j];
        data->train_indices[j] = temp;
    }
}

void get_next_batch(Cifar10* data, size_t batch_size, size_t batch_id, float* batch_images) {
    size_t start = batch_id * batch_size;
    for (size_t i = 0; i < batch_size; i++) {
        int idx = data->train_indices[start + i];
        memcpy(batch_images + i * IMG_SIZE,
               data->train_images + idx * IMG_SIZE,
               IMG_SIZE * sizeof(float));
    }
}

void free_cifar10(Cifar10* data) {
    free(data->train_images);
    free(data->test_images);
    free(data->train_labels);
    free(data->test_labels);
    free(data->train_indices);

    data->train_images = data->test_images = NULL;
    data->train_labels = data->test_labels = NULL;
    data->train_indices = NULL;
}
