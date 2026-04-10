#include "logistic.h"
#include <cmath>
#include <cuda_runtime.h>
#include <stdexcept>
#include <algorithm>
//#include <cuda_runtime.h>

static std::vector<float> softmax(const std::vector<float>& z) {
    std::vector<float> probs(z.size());

    float max_z = *std::max_element(z.begin(), z.end());
    float sum = 0.0f;

    for (int i = 0; i < z.size(); i++) {
        probs[i] = std::exp(z[i] - max_z);
        sum += probs[i];
    }

    for (int i = 0; i < z.size(); i++) {
        probs[i] /= sum;
    }

    return probs;
}

std::vector<float> LogisticRegression::predict(const std::vector<float>& x) const {
    // TODO
    return std::vector<float>(10, 0.1f);
}


void LogisticRegression::train_step(const std::vector<float>& x, int y, float lr) {
    // TODO
}

// Constructor — initialize cuBLAS handle and copy weights to GPU
LogisticRegression::LogisticRegression(int n_features): weights(n_features, 0.0f), n_features(n_features)
{
    cublasCreate(&handle);
    cudaMalloc(&d_weights, n_features * sizeof(float));
    cudaMemcpy(d_weights, weights.data(), n_features * sizeof(float), cudaMemcpyHostToDevice);
}

// Destructor — clean up
LogisticRegression::~LogisticRegression()
{
    cublasDestroy(handle);
    cudaFree(d_weights);
}

// Forward pass: X (n_samples x n_features) * W (n_features x 1) = out (n_samples x 1)
// This uses cuBLAS (heavily CUDA optimized) for matmul https://docs.nvidia.com/cuda/cublas/
void LogisticRegression::forward(const float* d_X, float* d_out, int n_samples)
{
    float alpha = 1.0f;
    float beta  = 0.0f;

    // C = alpha * A * B + beta * C
    cublasSgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n_samples,  // m — rows of output
        1,          // n — cols of output (single output per sample)
        n_features, // k — shared dimension
        &alpha,
        d_X,       n_samples,   // input matrix
        d_weights, n_features,  // weight vector
        &beta,
        d_out,     n_samples    // output
    );
}


// Kernel: per sample loss
__global__ void logisticLossKernel(const float* __restrict__ y, const float* __restrict__ yhat, float* __restrict__ losses, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float p = fmaxf(yhat[i], 1e-7f);
        p = fminf(p, 1.0f - 1e-7f);
        losses[i] = -(y[i] * logf(p) + (1.0f - y[i]) * logf(1.0f - p));
    }
}

// blockSize defaults to 256 (@ bindings)
float LogisticRegression::LossFunction(const float* d_y, const float* d_yhat, int n, int blockSize) const
{
    float* d_losses;
    cudaMalloc(&d_losses, n * sizeof(float));

    int gridSize = (n + blockSize - 1) / blockSize;
    logisticLossKernel<<<gridSize, blockSize>>>(d_y, d_yhat, d_losses, n);

    float* h_losses = new float[n];
    cudaMemcpy(h_losses, d_losses, n * sizeof(float), cudaMemcpyDeviceToHost);

    float total = 0.0f;
    for (int i = 0; i < n; i++) total += h_losses[i];

    cudaFree(d_losses);
    delete[] h_losses;

    return total / static_cast<float>(n);
}

void LogisticRegression::save(const std::string& path) const {
    // TODO
}

void LogisticRegression::load(const std::string& path) {
    // TODO
}
