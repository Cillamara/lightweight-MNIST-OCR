#include "logistic.h"
#include <cmath>
#include <stdexcept>

static float sigmoid(float z) {
    return 1.0f / (1.0f + std::exp(-z));
}

float LogisticRegression::predict(const std::vector<float>& x) const {
    
}

void LogisticRegression::train_step(const std::vector<float>& x, float y, float lr) {

}