#ifndef LOGISTIC_H
#define LOGISTIC_H

#include <vector>
#include <cmath>
#include <string>

class LogisticRegression 
{
public:
    LogisticRegression() {}

    LogisticRegression(int n_features, int n_classes = 10): weights(n_classes, std::vector<float>(n_features, 0.0f)) {}

    ~LogisticRegression();

    std::vector<float> predict(const std::vector<float>& x) const;

    float LossFunction(const float* d_y, const float* d_yhat, int n, int blockSize = 256) const;

    void forward(const float* d_X, float* d_out, int n_samples);

    void train_step(const std::vector<float>& x, int y, float lr = 0.01f);

    void save(const std::string& path) const;

    void load(const std::string& path);

private:
    std::vector<std::vector<float>> weights;
    std::vector<float> bias;
    int n_features;
    float* d_weights;       // weights on GPU
    cublasHandle_t handle;  // cuBLAS handle lives in the class
};

#endif
