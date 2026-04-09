#include "logistic.h"
#include <cmath>
#include <stdexcept>
#include <cuda_runtime.h>

static float sigmoid(float z) {
    return 1.0f / (1.0f + std::exp(-z));
}

float LogisticRegression::predict(const std::vector<float>& x) const {
    
}

void LogisticRegression::train_step(const std::vector<float>& x, float y, float lr) {

}

float LogisticRegression::LossFunction(const float* d_y, const float* d_yhat, int n) {
    float* d_losses;
    cudaMalloc(&d_losses, n * sizeof(float));

    int gridSize = (n + blockSize - 1) / blockSize; // I want to be able to contrll the block size in python cuz i dont know how many threads we 
    //can use so i dont want to hard code for example blocksize 256

    //TODO also i need to add the gpu kernel i will try to finialize loss function tonight
    
}
