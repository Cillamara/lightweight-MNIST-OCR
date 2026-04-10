#include "logistic.h"
#include <cmath>
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

float LogisticRegression::LossFunction(const float* d_y, const float* d_yhat, int n) const {
    //float* d_losses;
    //cudaMalloc(&d_losses, n * sizeof(float));

    //int gridSize = (n + blockSize - 1) / blockSize; // I want to be able to contrll the block size in python cuz i dont know how many threads we 
    //can use so i dont want to hard code for example blocksize 256

    //TODO also i need to add the gpu kernel i will try to finialize loss function tonight
    
}

void LogisticRegression::save(const std::string& path) const {
    // TODO
}

void LogisticRegression::load(const std::string& path) {
    // TODO
}
