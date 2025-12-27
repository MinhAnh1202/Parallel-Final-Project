# Bài toán phân loại ảnh cho CIFAR-10

Triển khai mô hình Autoencoder cho tập dữ liệu CIFAR-10, sau đó trích xuất đặc trưng đưa qua mô hình SVM để thực hiện tác vụ phân loại ảnh.

## Tổng Quan Dự Án

Dự án này triển khai mô hình autoencoder cho việc tái tạo hình ảnh CIFAR-10 sử dụng:
- **Triển Khai CPU**: Triển khai bằng ngôn ngữ C
- **Triển Khai GPU**: Kernel CUDA với ba mức độ tối ưu hóa:
  - Triển khai naive (đơn thuần thực hiện song song hóa từ phiên bản CPU)
  - Tối ưu hóa 1: Memory coalescing và shared memory
  - Tối ưu hóa 2: Quản lý bộ nhớ nâng cao và kernel fusion
Sau đó trích xuất đặc trưng và đưa qua mô hinh SVM để phân loại.

## Lưu ý
Trong các notebook, nhóm thực hiện copy trực tiếp source code các file vào và dùng lệnh `%%writefile` ngay trên kernel của notebook Colab. Khi thực hiện chạy lại các notebook nhóm đã thực hiện, chỉ cần thực hiện re-run notebook đó chứ không cần thêm file,...

Để thực thi file `train_svm.ipynb` trên Google Colab, cần thiết lập cấu trúc thư mục tại `/content` bao gồm: raw, cpu, naive, opt1 và opt2. Trong đó, các thư mục dùng để chứa các file đặc trưng đã trích xuất (như train_svm.txt, test_svm.txt).

## Yêu Cầu Phần Cứng

### Yêu Cầu Tối Thiểu
- **RAM**: Tối thiểu 4GB, khuyến nghị 8GB
- **GPU**: NVIDIA GPU với CUDA Compute Capability 3.5 trở lên
- **GPU Memory**: Tối thiểu 2GB, khuyến nghị 4GB+ cho batch size lớn hơn
- **Lưu trữ**: 500MB cho dataset CIFAR-10 và kết quả đầu ra

### Khuyến Nghị cho Google Colab
- **Tesla T4**: 16GB VRAM, CUDA Compute Capability 7.5
- **Runtime**: Bật GPU (Python 3.x với hỗ trợ CUDA)
- **RAM**: 12.7GB RAM có sẵn trong Colab
- **Storage**: ~100GB lưu trữ tạm thời

### Cấu Hình Đã Kiểm Tra
- **NVIDIA Tesla T4** (16GB VRAM) - Google Colab 

## Phụ Thuộc và Cài Đặt

### Yêu Cầu Phần Mềm
- **CUDA Toolkit**: Phiên bản 11.0 trở lên
  - Tải từ [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- **GCC/G++**: Phiên bản 7.0 trở lên (cho triển khai CPU)
- **NVCC**: Trình biên dịch CUDA (đi kèm với CUDA Toolkit)
- **OpenMP**: Cho song song hóa CPU (thường đi kèm với GCC)

### Thư Viện
- **Thư Viện C Tiêu Chuẩn**: stdio.h, stdlib.h, string.h, time.h
- **CUDA Runtime**: cuda_runtime.h
- **Thư Viện Hệ Thống**: sys/resource.h (để giám sát bộ nhớ)

### Các Bước Cài Đặt

#### 1. Cài Đặt CUDA Toolkit
```bash
# Cho Ubuntu/Debian
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

# Cho Windows: Tải và chạy file .exe installer từ trang web NVIDIA
```

#### 2. Xác Minh Cài Đặt CUDA
```bash
nvcc --version
nvidia-smi
```

#### 3. Thiết Lập Biến Môi Trường (nếu cần)
```bash
# Linux
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Windows (thêm vào system PATH)
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\lib\x64
```

#### 4. Tải Dataset CIFAR-10
Chương trình sẽ tự động tải và giải nén dataset CIFAR-10 khi chạy lần đầu.
Cách khác: Tải thủ công từ [trang web CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)

## Lệnh Biên Dịch

Di chuyển đến thư mục `source/` và sử dụng các lệnh sau:

### Triển Khai CPU
```bash
gcc -O2 \
    main.c \
    load_data.c \
    cpu_autoencoder.c \
    cpu_layers.c \
    -o main_cpu
```

### Triển Khai GPU (Naive) - Cho Tesla T4
```bash
nvcc -O3 -arch=sm_75 \
    main_gpu.cu \
    load_data.cu \
    gpu_autoencoder.cu \
    gpu_layers.cu \
    -o main_gpu
```

### Triển Khai GPU (Tối Ưu Hóa 1)
```bash
nvcc -arch=sm_75 \
    main_gpu_opt1.cu \
    load_data.cu \
    gpu_autoencoder_opt1.cu \
    gpu_layers_opt1.cu \
    -o main_gpu_opt1
```

### Triển Khai GPU (Tối Ưu Hóa 2)
```bash
nvcc -arch=sm_75 \
    main_gpu_opt2.cu \
    load_data.cu \
    gpu_autoencoder_opt2.cu \
    gpu_layers_opt2.cu \
    -o main_gpu_opt2
```

## Hướng Dẫn Thực Thi

### Thực Thi Cơ Bản
```bash
# Triển Khai CPU
./main_cpu

# Triển Khai GPU (Naive)
./main_gpu

# Triển Khai GPU (Tối Ưu Hóa 1)
./main_gpu_opt1

# Triển Khai GPU (Tối Ưu Hóa 2)
./main_gpu_opt2
```

### Ví Dụ Lệnh với Các Cấu Hình Khác Nhau

#### Huấn Luyện CPU với Tham Số Tùy Chỉnh
```bash
# Huấn luyện mặc định (1000 mẫu, 20 epochs)
./main_cpu

# Chương trình sử dụng các tham số cố định:
# - Tập huấn luyện con: 1000 ảnh
# - Batch size: 64
# - Learning rate: 0.001
# - Epochs: 20
```

#### Ví Dụ Huấn Luyện GPU
```bash
# Huấn luyện GPU tiêu chuẩn
./main_gpu

# Huấn luyện tối ưu hóa bộ nhớ
./main_gpu_opt1

# Huấn luyện tối ưu hóa hoàn toàn
./main_gpu_opt2
```

## Kết Quả Mong Đợi

### Kết Quả Huấn Luyện
```
[MAIN] Start program
[MAIN] Num CUDA devices = 1
Data loaded and normalized.
[MAIN] After load_cifar10
[MAIN] Start training loop (epochs=20, num_batches=15)

Epoch 1/20:
[EPOCH] epoch= 1, total_loss=125.67, avg_loss_per_sample=0.1257, time=1234.56 ms
[PROG MEM] After epoch 1: approx used by this program = 1024.50 MB

Epoch 2/20:
[EPOCH] epoch= 2, total_loss=98.45, avg_loss_per_sample=0.0985, time=1198.23 ms
...

*** Sampling Reconstructed Images ***
Saved pair 0: original vs reconstructed
Saved pair 1: original vs reconstructed
...
```

### Các File Được Tạo
1. **Log Huấn Luyện**:
   - `training_cpu.txt` - Metrics huấn luyện CPU
   - `training_gpu.txt` - Metrics huấn luyện GPU
   
2. **Hình Ảnh Tái Tạo** (định dạng PNM):
   - `sample_0_original.pnm`
   - `sample_0_reconstructed.pnm`
   - `sample_1_original.pnm`
   - `sample_1_reconstructed.pnm`
   - ... (lên tới 10 mẫu)

3. **Đặc Trưng SVM**:
   - `svm_features.txt` - Các đặc trưng được trích xuất để phân loại

### Metrics Hiệu Suất
- **Thời Gian Huấn Luyện**: Tổng thời gian cho tất cả các epochs
- **Sử Dụng Bộ Nhớ**: Lượng bộ nhớ GPU/CPU tiêu thụ
- **Giá Trị Loss**: Mean squared error mỗi epoch
- **Chất Lượng Tái Tạo**: File so sánh trực quan

## Cấu Trúc Dự Án (Cấu trúc folder Drive)
```
├── Source Code/                 # Mã nguồn
│   ├── main.c             # Chương trình chính CPU
│   ├── main_gpu*.cu       # Chương trình chính GPU
│   ├── *_autoencoder.*    # Triển khai autoencoder
│   ├── *_layers.*         # Triển khai các lớp
│   ├── load_data.*        # Tiện ích tải dữ liệu
│   └── README.md             # File này
├── Notebooks/              # Jupyter notebooks để phân tích
│   ├── CPU.ipynb          # Notebook train Autoencoder và trích xuất đặc trưng của CPU
│   ├── GPU naive.ipynb    # Notebook train Autoencoder và trích xuất đặc trưng GPU naive
|   ├── GPU_opt1.ipynb     # Notebook train Autoencoder và trích xuất đặc trưng GPU version 1
|   ├── GPU_opt2.ipynb     # Notebook train Autoencoder và trích xuất đặc trưng GPU version 2
|   ├── Demo CPU.ipynb     # Notebook chạy demo CPU trong video
|   ├── Demo GPU naive.ipynb    # Notebook chạy demo GPU naive trong video
|   ├── Demo GPU_opt1.ipynb     # Notebook chạy demo GPU version 1 trong video
|   ├── Demo GPU opt2.ipynb     # Notebook chạy demo GPU version 2 trong video
│   ├── Reconstruction.ipynb         # Notebook hiển thị ảnh Reconstructed vs Original
│   ├── Verify kernel outputs.ipynb         # Notebook kiểm tra kết quả kernel các version so với CPU
│   ├── Verify outputs.ipynb         # Notebook kiểm tra kết quả output của các version với CPU
│   └── train_svm.ipynb         # Notebook thực hiện train SVM các version
├── Trained model weights/
│   ├── ae_weights_gpu_naive.bin          # Trọng số của phiên bản GPU naive
│   ├── ae_weights_gpu_opt1.bin           # Trọng số của phiên bản GPU version 1
│   ├── ae_weights_gpu_opt2.bin           # Trọng số của phiên bản GPU version 2
│   └── autoencoder_weights_cpu.bin       # Trọng số của phiên bản CPU
├── Link of Presentation Video.txt # File txt chứa video thuyết trình
├── Team Plan and Work Distribution.pdf # File pdf chứa plan và work distribution
└── Project Report.ipynb         # Report notebook chứa nội dung đồ án
```

## Tác Giả
- Đoàn Thị Minh Anh - 22120213
- Trần Hoàng Kim Ngân - 22120224

## Giấy Phép
Dự án này dành cho mục đích giáo dục.
