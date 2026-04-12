#include "logistic.h"
#include <cmath>
#include <cuda_runtime.h>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <string>
#include <fstream>

#define CUDA_CHECK(x) if((x) != cudaSuccess) throw std::runtime_error(cudaGetErrorString(x));

__global__ void sigmoidKernel(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] = 1.0f / (1.0f + expf(-x[i]));
    }
}

__global__ void errorKernel(
    const float* yhat,
    const float* y,
    float* error,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        error[i] = yhat[i] - y[i];
    }
}

float LogisticRegression::predict(const std::vector<float>& x) {

    int n_samples = 1;

    float* d_X;
    float* d_logits;

    cudaMalloc(&d_X, n_features * sizeof(float));
    cudaMalloc(&d_logits, sizeof(float));

    cudaMemcpy(d_X, x.data(),
               n_features * sizeof(float),
               cudaMemcpyHostToDevice);

    forward(d_X, d_logits, n_samples);

    sigmoidKernel<<<1, 1>>>(d_logits, n_samples);

    float h_out;
    cudaMemcpy(&h_out, d_logits,
               sizeof(float),
               cudaMemcpyDeviceToHost);

    cudaFree(d_X);
    cudaFree(d_logits);

    return h_out;
}

void LogisticRegression::train_step(const std::vector<float>& x, int y, float lr) {
  
    float* d_X;
    cudaMalloc(&d_X, n_features * sizeof(float));
    cudaMemcpy(d_X, x.data(), n_features * sizeof(float), cudaMemcpyHostToDevice);

    float label = (float)y;
    float* d_y;
    cudaMalloc(&d_y, sizeof(float));
    cudaMemcpy(d_y, &label, sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&d_yhat,  sizeof(float));
    cudaMalloc(&d_error, sizeof(float));
    cudaMalloc(&d_grad,  n_features * sizeof(float));

    forward(d_X, d_yhat, 1);
    sigmoidKernel<<<1, 1>>>(d_yhat, 1);

    errorKernel<<<1, 1>>>(d_yhat, d_y, d_error, 1);

    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
        n_features, 1, 1,
        &alpha, d_X, 1, d_error, 1,
        &beta,  d_grad, n_features);

    float neg_lr = -lr;
    cublasSaxpy(handle, n_features, &neg_lr, d_grad, 1, d_weights, 1);

    cudaFree(d_X);
    cudaFree(d_y);
    cudaFree(d_yhat);
    cudaFree(d_error);
    cudaFree(d_grad);
}

// Constructor — initialize cuBLAS handle and copy weights to GPU
LogisticRegression::LogisticRegression(int n_features, int n_classes)
    : n_features(n_features), n_classes(n_classes)
{
    cublasCreate(&handle);
    cudaMalloc(&d_weights, n_features * sizeof(float));
    cudaMemset(d_weights, 0, n_features * sizeof(float));
    cudaMalloc(&d_grad, n_features * sizeof(float));
}

// Destructor — clean up
LogisticRegression::~LogisticRegression()
{
    cublasDestroy(handle);
    cudaFree(d_weights);
}

// Forward pass: X (n_samples x n_features) * W (n_features x 1) = out (n_samples x 1)
// This uses cuBLAS (heavily CUDA optimized) for matmul https://docs.nvidia.com/cuda/cublas/
void LogisticRegression::forward(const float *d_X, float *d_out, int n_samples)
{
    float alpha = 1.0f;
    float beta = 0.0f;

    // C = alpha * A * B + beta * C
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                n_samples,  // m — rows of output
                1,          // n — cols of output (single output per sample)
                n_features, // k — shared dimension
                &alpha,
                d_X, n_samples,        // input matrix
                d_weights, n_features, // weight vector
                &beta,
                d_out, n_samples // output
    );
}

// Kernel: per sample loss
__global__ void logisticLossKernel(const float *__restrict__ y, const float *__restrict__ yhat, float *__restrict__ losses, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        float p = fmaxf(yhat[i], 1e-7f);
        p = fminf(p, 1.0f - 1e-7f);
        losses[i] = -(y[i] * logf(p) + (1.0f - y[i]) * logf(1.0f - p));
    }
}

// blockSize defaults to 256 (@ bindings)
float LogisticRegression::LossFunction(const float *d_y, const float *d_yhat, int n, int blockSize) const
{
    float *d_losses;
    cudaMalloc(&d_losses, n * sizeof(float));

    int gridSize = (n + blockSize - 1) / blockSize;
    logisticLossKernel<<<gridSize, blockSize>>>(d_y, d_yhat, d_losses, n);

    float *h_losses = new float[n];
    cudaMemcpy(h_losses, d_losses, n * sizeof(float), cudaMemcpyDeviceToHost);

    float total = 0.0f;
    for (int i = 0; i < n; i++)
        total += h_losses[i];

    cudaFree(d_losses);
    delete[] h_losses;

    return total / static_cast<float>(n);
}

void LogisticRegression::save(const std::string& path) const {
    std::vector<float> host_weights(n_features);

    cudaMemcpy(
        host_weights.data(),
        d_weights,
        n_features * sizeof(float),
        cudaMemcpyDeviceToHost
    );

    std::ofstream out(path, std::ios::binary);
    out.write((char*)host_weights.data(), n_features * sizeof(float));
}

void LogisticRegression::load(const std::string& path) {
    std::vector<float> host_weights(n_features);

    std::ifstream in(path, std::ios::binary);
    in.read((char*)host_weights.data(), n_features * sizeof(float));

    cudaMemcpy(
        d_weights,
        host_weights.data(),
        n_features * sizeof(float),
        cudaMemcpyHostToDevice
    );
}
