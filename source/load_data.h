#pragma once
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TRAIN_NUM    50000
#define TEST_NUM     10000
#define IMG_SIZE     (32*32*3)     // 3072

typedef struct {
    float* train_images;   // [50000 * 3072]
    float* test_images;    // [10000 * 3072]
    uint8_t* train_labels; // [50000]
    uint8_t* test_labels;  // [10000]
    int* train_indices;
} Cifar10;

void load_cifar10(Cifar10* data);
void normalize_cifar10(Cifar10* data);
void shuffle_cifar10(Cifar10* data);
void get_next_batch(Cifar10* data, size_t batch_size, size_t batch_id, float* batch_images);
void print_cifar10(Cifar10* data);
void free_cifar10(Cifar10* data);