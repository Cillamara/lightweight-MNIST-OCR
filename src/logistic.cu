#include "logistic.h"
#include <cmath>
#include <cuda_runtime.h>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <string>
#include <fstream>

#define CUDA_CHECK(x) if((x) != cudaSuccess) throw std::runtime_error(cudaGetErrorString(x));

// Hardware Accelerated Fast matmul

// Matrix-vector multiply: out[k] = sum_j X[j] * W[j * K + k]
// Forward pass: X (1 x n_features) * W (n_features x n_classes) = out (1 x n_classes)
__global__ void matvecKernel(const float* X, const float* W, float* out,
                             int n_features, int K) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    float sum = 0.0f;
    for (int j = 0; j < n_features; j++)
        sum += X[j] * W[j * K + k];
    out[k] = sum;
}

// Add bias: out[k] += bias[k]
__global__ void addBiasKernel(float* out, const float* bias, int K) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    out[k] += bias[k];
}

// Softmax in-place (single row of K elements, numerically stable)
__global__ void softmaxKernel(float* x, int K) {
    float max_val = x[0];
    for (int i = 1; i < K; i++)
        max_val = fmaxf(max_val, x[i]);

    float sum = 0.0f;
    for (int i = 0; i < K; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < K; i++)
        x[i] /= sum;
}

// Softmax gradient: error[k] = probs[k] - (k == y ? 1 : 0)
__global__ void softmaxGradKernel(const float* probs, int y,
                                  float* error, int K) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    error[k] = probs[k] - (k == y ? 1.0f : 0.0f);
}

// Outer product: grad[j * K + k] = x[j] * error[k]
// This is the weight gradient for a single sample
__global__ void outerProductKernel(const float* x, const float* error,
                                   float* grad, int n_features, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_features * K;
    if (idx >= total) return;
    int j = idx / K;
    int k = idx % K;
    grad[idx] = x[j] * error[k];
}

// SAXPY: y[i] = alpha * x[i] + y[i]
__global__ void saxpyKernel(float alpha, const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = alpha * x[i] + y[i];
}

// ---- Original kernels (kept for reference / binary path) ----

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

std::vector<float> LogisticRegression::predict(const std::vector<float>& x) {

    float* d_X;
    CUDA_CHECK(cudaMalloc(&d_X, n_features * sizeof(float)));

    float* d_out;
    CUDA_CHECK(cudaMalloc(&d_out, n_classes * sizeof(float)));

    cudaMemcpy(d_X, x.data(),
               n_features * sizeof(float),
               cudaMemcpyHostToDevice);

    forward(d_X, d_out, 1);

    softmaxKernel<<<1, 1>>>(d_out, n_classes);

    std::vector<float> h_out(n_classes);
    cudaMemcpy(h_out.data(), d_out,
               n_classes * sizeof(float),
               cudaMemcpyDeviceToHost);

    cudaFree(d_X);
    cudaFree(d_out);

    return h_out;
}

void LogisticRegression::train_step(const std::vector<float>& x, int y, float lr) {

    float* d_X;
    CUDA_CHECK(cudaMalloc(&d_X, n_features * sizeof(float)));
    cudaMemcpy(d_X, x.data(), n_features * sizeof(float), cudaMemcpyHostToDevice);

    CUDA_CHECK(cudaMalloc(&d_yhat,  n_classes * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_error, n_classes * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad,  n_features * n_classes * sizeof(float)));

    // Forward pass + softmax
    forward(d_X, d_yhat, 1);
    softmaxKernel<<<1, 1>>>(d_yhat, n_classes);

    // error[k] = P[k] - onehot(y)[k]
    int block = 256;
    int grid = (n_classes + block - 1) / block;
    softmaxGradKernel<<<grid, block>>>(d_yhat, y, d_error, n_classes);

    // grad[j,k] = x[j] * error[k]  (outer product for single sample)
    int total = n_features * n_classes;
    grid = (total + block - 1) / block;
    outerProductKernel<<<grid, block>>>(d_X, d_error, d_grad, n_features, n_classes);

    // W -= lr * grad ;  b -= lr * error
    float neg_lr = -lr;
    saxpyKernel<<<(total + block - 1) / block, block>>>(neg_lr, d_grad, d_weights, total);
    saxpyKernel<<<(n_classes + block - 1) / block, block>>>(neg_lr, d_error, d_bias, n_classes);

    cudaFree(d_X);
    cudaFree(d_yhat);
    cudaFree(d_error);
    cudaFree(d_grad);
}

// Constructor — initialize weights and bias on GPU
LogisticRegression::LogisticRegression(int n_features, int n_classes)
    : n_features(n_features), n_classes(n_classes),
      d_yhat(nullptr), d_error(nullptr), d_grad(nullptr)
{
    CUDA_CHECK(cudaMalloc(&d_weights, n_features * n_classes * sizeof(float)));
    cudaMemset(d_weights, 0, n_features * n_classes * sizeof(float));
    CUDA_CHECK(cudaMalloc(&d_bias, n_classes * sizeof(float)));
    cudaMemset(d_bias, 0, n_classes * sizeof(float));
}

// Destructor — clean up
LogisticRegression::~LogisticRegression()
{
    cudaFree(d_weights);
    cudaFree(d_bias);
}

// Forward pass: X (1 x n_features) * W (n_features x n_classes) + b = out (1 x n_classes)
// This uses custom CUDA kernels for matmul (replacing cuBLAS)
// https://docs.nvidia.com/cuda/cublas/ (reference for the math)
void LogisticRegression::forward(const float *d_X, float *d_out, int n_samples)
{
    int block = 256;
    int grid = (n_classes + block - 1) / block;

    // out = X @ W   (matrix-vector multiply)
    matvecKernel<<<grid, block>>>(d_X, d_weights, d_out, n_features, n_classes);

    // out += bias
    addBiasKernel<<<grid, block>>>(d_out, d_bias, n_classes);
}

// Kernel: per sample loss (binary cross-entropy — kept from original)
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
    int w_size = n_features * n_classes;
    std::vector<float> host_weights(w_size);
    std::vector<float> host_bias(n_classes);

    cudaMemcpy(
        host_weights.data(),
        d_weights,
        w_size * sizeof(float),
        cudaMemcpyDeviceToHost
    );
    cudaMemcpy(
        host_bias.data(),
        d_bias,
        n_classes * sizeof(float),
        cudaMemcpyDeviceToHost
    );

    std::ofstream out(path, std::ios::binary);
    out.write((char*)&n_features, sizeof(int));
    out.write((char*)&n_classes, sizeof(int));
    out.write((char*)host_weights.data(), w_size * sizeof(float));
    out.write((char*)host_bias.data(), n_classes * sizeof(float));
}

void LogisticRegression::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);

    int file_features, file_classes;
    in.read((char*)&file_features, sizeof(int));
    in.read((char*)&file_classes, sizeof(int));

    if (file_features != n_features || file_classes != n_classes)
        throw std::runtime_error("Model file shape mismatch");

    int w_size = n_features * n_classes;
    std::vector<float> host_weights(w_size);
    std::vector<float> host_bias(n_classes);

    in.read((char*)host_weights.data(), w_size * sizeof(float));
    in.read((char*)host_bias.data(), n_classes * sizeof(float));

    cudaMemcpy(
        d_weights,
        host_weights.data(),
        w_size * sizeof(float),
        cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        d_bias,
        host_bias.data(),
        n_classes * sizeof(float),
        cudaMemcpyHostToDevice
    );
}
