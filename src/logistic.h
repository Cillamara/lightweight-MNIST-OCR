#ifndef LOGISTIC_H
#define LOGISTIC_H

#include <vector>
#include <cmath>
#include <string>

#include <cublas_v2.h>

class LogisticRegression 
{
public:
    LogisticRegression(int n_features, int n_classes = 1);

    ~LogisticRegression();

    float predict(const std::vector<float>& x);

    float LossFunction(const float* d_y, const float* d_yhat, int n, int blockSize = 256) const;

    void train_step(const std::vector<float>& x, int y, float lr = 0.01f);

    void save(const std::string& path) const;

    void load(const std::string& path);

private:
    void forward(const float* d_X, float* d_out, int n_samples);

    int n_features;
    int n_classes;
    float* d_weights;       // weights on GPU
    float* d_bias;
    cublasHandle_t handle;  // cuBLAS handle lives in the class

    float* d_yhat;
    float* d_error;
    float* d_grad;
};

#endif
