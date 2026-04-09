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
    //sugma balls gotta go to class 
}
