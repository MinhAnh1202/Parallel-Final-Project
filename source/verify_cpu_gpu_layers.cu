%%writefile verify_gpu_cpu_layers.cu
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

extern "C" {
    #include "cpu_layers.h"   // đã có sẵn
}

#include "gpu_layers.h"       // chứa các kernel GPU + CHECK_CUDA

// Hàm tiện ích so sánh 2 mảng
void compare_arrays(const char* name,
                    const float* a,
                    const float* b,
                    int n)
{
    double max_abs = 0.0;
    double sum_abs = 0.0;
    double sum_sq  = 0.0;

    for (int i = 0; i < n; ++i) {
        double diff = (double)a[i] - (double)b[i];
        double ad   = fabs(diff);

        if (ad > max_abs) max_abs = ad;
        sum_abs += ad;
        sum_sq  += diff * diff;
    }

    double mean_abs = sum_abs / n;
    double rmse     = std::sqrt(sum_sq / n);

    printf("[%s] max|diff| = %.6g, mean|diff| = %.6g, RMSE = %.6g\n",
           name, max_abs, mean_abs, rmse);
}

/*-------------------------------------------------------------
  TEST CONV2D (forward + backward) – cái này bạn đang chạy OK,
  mình để ví dụ đơn giản N=1 cho gọn.
-------------------------------------------------------------*/
void test_conv2d()
{
    printf("==================== test_conv2d ====================\n");

    int N      = 1;
    int C_in   = 3;
    int C_out  = 4;
    int H      = 8;
    int W      = 8;
    int K      = 3;
    int pad    = 1;
    int stride = 1;

    int H_out = (H + 2 * pad - K) / stride + 1;
    int W_out = (W + 2 * pad - K) / stride + 1;

    int in_size   = N * C_in * H * W;
    int w_size    = C_out * C_in * K * K;
    int b_size    = C_out;
    int out_size  = N * C_out * H_out * W_out;

    // Cấp phát host
    float *h_x     = (float*)malloc(in_size  * sizeof(float));
    float *h_w     = (float*)malloc(w_size   * sizeof(float));
    float *h_b     = (float*)malloc(b_size   * sizeof(float));
    float *h_y_cpu = (float*)malloc(out_size * sizeof(float));
    float *h_y_gpu = (float*)malloc(out_size * sizeof(float));

    float *h_dy    = (float*)malloc(out_size * sizeof(float));
    float *h_dx_cpu= (float*)malloc(in_size  * sizeof(float));
    float *h_dx_gpu= (float*)malloc(in_size  * sizeof(float));
    float *h_dw_cpu= (float*)malloc(w_size   * sizeof(float));
    float *h_dw_gpu= (float*)malloc(w_size   * sizeof(float));
    float *h_db_cpu= (float*)malloc(b_size   * sizeof(float));
    float *h_db_gpu= (float*)malloc(b_size   * sizeof(float));

    // Init dữ liệu determinisitc
    for (int i = 0; i < in_size; ++i)  h_x[i] = (float)((i % 17) - 8) / 10.0f;
    for (int i = 0; i < w_size; ++i)   h_w[i] = (float)((i % 13) - 6) / 7.0f;
    for (int i = 0; i < b_size; ++i)   h_b[i] = (float)((i % 5) - 2) / 3.0f;
    for (int i = 0; i < out_size; ++i) h_dy[i]= (float)((i % 11) - 5) / 9.0f;

    // ===== CPU FORWARD =====
    Conv2D_Forward(
        h_x, W, H, C_in,
        h_w, K, K,
        h_b,
        pad, stride,
        C_out,
        h_y_cpu,
        H_out, W_out);

    // ===== GPU FORWARD =====
    float *d_x=nullptr, *d_w=nullptr, *d_b=nullptr, *d_y=nullptr;
    CHECK_CUDA(cudaMalloc(&d_x, in_size  * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_w, w_size   * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b, b_size   * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, out_size * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_x, h_x, in_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_w, h_w, w_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, b_size * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block2d(16,16);
    dim3 gridConv(
        (W_out + block2d.x - 1)/block2d.x,
        (H_out + block2d.y - 1)/block2d.y,
        N * C_out);

    conv2d_forward_naive<<<gridConv, block2d>>>(
        d_x, d_w, d_b, d_y,
        N, C_in, H, W,
        C_out, K, pad, stride);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_y_gpu, d_y, out_size * sizeof(float), cudaMemcpyDeviceToHost));

    compare_arrays("Conv2D_Forward", h_y_cpu, h_y_gpu, out_size);

    // ===== CPU BACKWARD =====
    Conv2D_Backward_Input(
        h_dy, W_out, H_out,
        h_w, K, K,
        W, H, C_in,
        pad, stride, C_out,
        h_dx_cpu);

    Conv2D_Backward_Kernel(
        h_dy, W_out, H_out,
        h_x, W, H, C_in,
        K, K,
        pad, stride, C_out,
        h_dw_cpu);

    Conv2D_Backward_Biases(
        h_dy, W_out, H_out,
        C_out,
        h_db_cpu);

    // ===== GPU BACKWARD =====
    float *d_dy=nullptr, *d_dx=nullptr, *d_dw=nullptr, *d_db=nullptr;
    CHECK_CUDA(cudaMalloc(&d_dy, out_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dx, in_size  * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dw, w_size   * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_db, b_size   * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_dy, h_dy, out_size * sizeof(float), cudaMemcpyHostToDevice));

    // dX
    dim3 gridIn(
        (W + block2d.x - 1)/block2d.x,
        (H + block2d.y - 1)/block2d.y,
        N * C_in);
    conv2d_backward_input_naive<<<gridIn, block2d>>>(
        d_dy, d_w, d_dx,
        N, C_in, H, W,
        C_out, K, pad, stride);
    CHECK_CUDA(cudaDeviceSynchronize());

    // dW
    int num_w = w_size;
    int t = 256;
    int b = (num_w + t - 1)/t;
    conv2d_backward_weight_naive<<<b, t>>>(
        d_x, d_dy, d_dw,
        N, C_in, H, W,
        C_out, K, pad, stride);
    CHECK_CUDA(cudaDeviceSynchronize());

    // dB
    int tb = 256;
    int bb = (C_out + tb - 1)/tb;
    conv2d_backward_bias_naive<<<bb, tb>>>(
        d_dy, d_db,
        N, C_out, H_out, W_out);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_dx_gpu, d_dx, in_size  * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_dw_gpu, d_dw, w_size   * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_db_gpu, d_db, b_size   * sizeof(float), cudaMemcpyDeviceToHost));

    compare_arrays("Conv2D_Backward_Input (dX)", h_dx_cpu, h_dx_gpu, in_size);
    compare_arrays("Conv2D_Backward_Kernel (dW)", h_dw_cpu, h_dw_gpu, w_size);
    compare_arrays("Conv2D_Backward_Biases (dB)", h_db_cpu, h_db_gpu, b_size);

    // cleanup
    free(h_x); free(h_w); free(h_b); free(h_y_cpu); free(h_y_gpu);
    free(h_dy); free(h_dx_cpu); free(h_dx_gpu);
    free(h_dw_cpu); free(h_dw_gpu);
    free(h_db_cpu); free(h_db_gpu);

    cudaFree(d_x); cudaFree(d_w); cudaFree(d_b); cudaFree(d_y);
    cudaFree(d_dy); cudaFree(d_dx); cudaFree(d_dw); cudaFree(d_db);
}

/*-------------------------------------------------------------
  TEST MaxPool + UpSample (forward + backward)
  Quan trọng: truyền đúng H,W (H_input/H_pool), không dùng H_out nhầm.
-------------------------------------------------------------*/
void test_pool_upsample()
{
    printf("\n==================== test_pool_upsample ====================\n");

    int N = 1;
    int C = 3;
    int H = 8;
    int W = 8;

    int pool_k     = 2;
    int pool_stride= 2;
    int H_pool     = H / pool_k;  // 4
    int W_pool     = W / pool_k;  // 4

    int scale      = 2;
    int H_up       = H_pool * scale; // 8
    int W_up       = W_pool * scale; // 8

    int in_size    = N * C * H      * W;
    int pool_size  = N * C * H_pool * W_pool;
    int up_size    = N * C * H_up   * W_up;

    // Host buffers
    float *h_in              = (float*)malloc(in_size   * sizeof(float));
    float *h_pool_cpu        = (float*)malloc(pool_size * sizeof(float));
    float *h_pool_gpu        = (float*)malloc(pool_size * sizeof(float));
    float *h_up_cpu          = (float*)malloc(up_size   * sizeof(float));
    float *h_up_gpu          = (float*)malloc(up_size   * sizeof(float));
    float *h_pool_grad       = (float*)malloc(pool_size * sizeof(float));
    float *h_in_grad_cpu     = (float*)malloc(in_size   * sizeof(float));
    float *h_in_grad_gpu     = (float*)malloc(in_size   * sizeof(float));
    float *h_up_grad         = (float*)malloc(up_size   * sizeof(float));
    float *h_pool_grad2_cpu  = (float*)malloc(pool_size * sizeof(float));
    float *h_pool_grad2_gpu  = (float*)malloc(pool_size * sizeof(float));

    // Init input
    for (int i = 0; i < in_size; ++i)
        h_in[i] = (float)((i % 13) - 6) / 7.0f;

    // ===== CPU MaxPool Forward =====
    MaxPool2D_Forward(
        h_in,
        W, H,
        pool_k, pool_k,
        pool_stride,
        C,
        h_pool_cpu,
        H_pool, W_pool);

    // ===== GPU MaxPool Forward =====
    float *d_in=nullptr, *d_pool=nullptr, *d_up=nullptr;
    float *d_pool_grad=nullptr, *d_in_grad=nullptr;
    float *d_up_grad=nullptr, *d_pool_grad2=nullptr;

    CHECK_CUDA(cudaMalloc(&d_in,   in_size   * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_pool, pool_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_up,   up_size   * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_in, h_in, in_size * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block2d(16,16);
    dim3 gridPool(
        (W_pool + block2d.x - 1) / block2d.x,
        (H_pool + block2d.y - 1) / block2d.y,
        N * C);

    // GPU pool fwd: H,W là kích thước input (8x8)
    maxpool2x2_forward<<<gridPool, block2d>>>(
        d_in, d_pool,
        N, C, H, W);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_pool_gpu, d_pool, pool_size * sizeof(float),
                          cudaMemcpyDeviceToHost));

    compare_arrays("MaxPool2D_Forward", h_pool_cpu, h_pool_gpu, pool_size);

    // ===== CPU MaxPool Backward =====
    for (int i = 0; i < pool_size; ++i)
        h_pool_grad[i] = (float)((i % 7) - 3) / 5.0f;

    MaxPool2D_Backward(
        h_pool_grad, W_pool, H_pool,
        h_in,
        W, H,
        pool_k, pool_k, pool_stride,
        C,
        h_in_grad_cpu);

    // ===== GPU MaxPool Backward =====
    CHECK_CUDA(cudaMalloc(&d_pool_grad, pool_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_in_grad,   in_size   * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_pool_grad, h_pool_grad, pool_size * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_in_grad, 0, in_size * sizeof(float)));

    dim3 gridPoolB(
        (W_pool + block2d.x - 1) / block2d.x,
        (H_pool + block2d.y - 1) / block2d.y,
        N * C);

    maxpool2x2_backward<<<gridPoolB, block2d>>>(
        d_in, d_pool_grad, d_in_grad,
        N, C, H, W);   // H,W = kích thước INPUT (8x8)
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_in_grad_gpu, d_in_grad, in_size * sizeof(float),
                          cudaMemcpyDeviceToHost));

    compare_arrays("MaxPool2D_Backward", h_in_grad_cpu, h_in_grad_gpu, in_size);

    // ===== CPU UpSample Forward (4x4 -> 8x8) =====
    UpSample2D_Forward(
        h_pool_cpu,
        W_pool, H_pool,
        scale,
        C,
        h_up_cpu,
        H_up, W_up);

    // ===== GPU UpSample Forward =====
    dim3 gridUp(
        (W_up + block2d.x - 1) / block2d.x,
        (H_up + block2d.y - 1) / block2d.y,
        N * C);

    upsample2x2_forward<<<gridUp, block2d>>>(
        d_pool, d_up,
        N, C,
        H_pool, W_pool);   // H,W = kích thước input (4x4)
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_up_gpu, d_up, up_size * sizeof(float),
                          cudaMemcpyDeviceToHost));

    compare_arrays("UpSample2D_Forward", h_up_cpu, h_up_gpu, up_size);

    // ===== CPU UpSample Backward =====
    for (int i = 0; i < up_size; ++i)
        h_up_grad[i] = (float)((i % 11) - 5) / 9.0f;

    UpSample2D_Backward(
        h_up_grad,
        W_up, H_up,
        scale,
        C,
        h_pool_grad2_cpu,
        H_pool, W_pool);

    // ===== GPU UpSample Backward =====
    CHECK_CUDA(cudaMalloc(&d_up_grad,    up_size   * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_pool_grad2, pool_size * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_up_grad, h_up_grad, up_size * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_pool_grad2, 0, pool_size * sizeof(float)));

    dim3 gridUpB(
        (W_pool + block2d.x - 1) / block2d.x,
        (H_pool + block2d.y - 1) / block2d.y,
        N * C);

    upsample2x2_backward<<<gridUpB, block2d>>>(
        d_up_grad, d_pool_grad2,
        N, C,
        H_pool, W_pool);   // CHÚ Ý: H,W = kích thước INPUT (4x4), KHÔNG phải 8x8
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_pool_grad2_gpu, d_pool_grad2, pool_size * sizeof(float),
                          cudaMemcpyDeviceToHost));

    compare_arrays("UpSample2D_Backward", h_pool_grad2_cpu, h_pool_grad2_gpu, pool_size);

    // cleanup
    free(h_in);
    free(h_pool_cpu); free(h_pool_gpu);
    free(h_up_cpu);   free(h_up_gpu);
    free(h_pool_grad);
    free(h_in_grad_cpu); free(h_in_grad_gpu);
    free(h_up_grad);
    free(h_pool_grad2_cpu); free(h_pool_grad2_gpu);

    cudaFree(d_in);
    cudaFree(d_pool);
    cudaFree(d_up);
    cudaFree(d_pool_grad);
    cudaFree(d_in_grad);
    cudaFree(d_up_grad);
    cudaFree(d_pool_grad2);
}

int main()
{
    test_conv2d();
    test_pool_upsample();

    CHECK_CUDA(cudaDeviceReset());
    return 0;
}
