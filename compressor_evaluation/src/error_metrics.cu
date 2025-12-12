#include "error_metrics.h"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

__global__ void mse_kernel(const double* a, const double* b, double* out, size_t size) {
    __shared__ double shared[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    double diff = 0.0;
    if (idx < size) {
        double d = a[idx] - b[idx];
        diff = d * d;
    }
    shared[tid] = diff;
    __syncthreads();

    // Reduce within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(out, shared[0]);
    }
}

__global__ void norm_kernel(const double* a, const double* b, double* out_diff, double* out_orig, size_t size) {
    __shared__ double shared_diff[256];
    __shared__ double shared_orig[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    double diff_sq = 0.0, orig_sq = 0.0;
    if (idx < size) {
        double x = a[idx];
        double y = b[idx];
        diff_sq = (x - y) * (x - y);
        orig_sq = x * x;
    }

    shared_diff[tid] = diff_sq;
    shared_orig[tid] = orig_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_diff[tid] += shared_diff[tid + s];
            shared_orig[tid] += shared_orig[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(out_diff, shared_diff[0]);
        atomicAdd(out_orig, shared_orig[0]);
    }
}

double compute_relative_l2_error_gpu(const double* d_orig, const double* d_comp, size_t size) {
    if (size == 0) return 0.0;

    double *d_diff_sum = nullptr, *d_orig_sum = nullptr;
    cudaMalloc(&d_diff_sum, sizeof(double));
    cudaMalloc(&d_orig_sum, sizeof(double));
    cudaMemset(d_diff_sum, 0, sizeof(double));
    cudaMemset(d_orig_sum, 0, sizeof(double));

    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    norm_kernel<<<numBlocks, blockSize>>>(d_orig, d_comp, d_diff_sum, d_orig_sum, size);
    cudaDeviceSynchronize();

    double diff_sum = 0.0, orig_sum = 0.0;
    cudaMemcpy(&diff_sum, d_diff_sum, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&orig_sum, d_orig_sum, sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_diff_sum);
    cudaFree(d_orig_sum);

    if (orig_sum == 0.0) return std::numeric_limits<double>::infinity(); // avoid divide-by-zero
    return std::sqrt(diff_sum) / std::sqrt(orig_sum);
}

__global__ void max_error_kernel(const double* a, const double* b, double* max_out, size_t size) {
    __shared__ double shared[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    double local = 0.0;
    if (idx < size)
        local = fabs(a[idx] - b[idx]);

    shared[tid] = local;
    __syncthreads();

    // In-block max reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            shared[tid] = fmax(shared[tid], shared[tid + s]);
        __syncthreads();
    }

    if (tid == 0)
        atomicMax((unsigned long long int*)max_out, __double_as_longlong(shared[0]));
}

double compute_max_error_gpu(const double* d_orig, const double* d_comp, size_t size) {
    double* d_max;
    cudaMalloc(&d_max, sizeof(double));
    double zero = 0.0;
    cudaMemcpy(d_max, &zero, sizeof(double), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    max_error_kernel<<<numBlocks, blockSize>>>(d_orig, d_comp, d_max, size);
    cudaDeviceSynchronize();

    double max_error = 0.0;
    cudaMemcpy(&max_error, d_max, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_max);
    return max_error;
}



double compute_mse_gpu(const double* d_orig, const double* d_comp, size_t size) {
    if (size == 0) return 0.0;

    double* d_result = nullptr;
    cudaMalloc(&d_result, sizeof(double));
    cudaMemset(d_result, 0, sizeof(double));

    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    mse_kernel<<<numBlocks, blockSize>>>(d_orig, d_comp, d_result, size);
    cudaDeviceSynchronize(); 

    double mse = 0.0;
    cudaMemcpy(&mse, d_result, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_result);

    return mse / static_cast<double>(size);
}


