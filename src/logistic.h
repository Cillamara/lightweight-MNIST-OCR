#ifndef LOGISTIC_H
#define LOGISTIC_H

#include <vector>
#include <cmath>
#include <string>

class LogisticRegression 
{
public:
    LogisticRegression() {}

    LogisticRegression(int n_features, int n_classes = 10)
    : weights(n_classes, std::vector<float>(n_features, 0.0f)) {}

    std::vector<float> predict(const std::vector<float>& x) const;

    float LossFunction(const float* d_y, const float* d_yhat, int n) const;

    void train_step(const std::vector<float>& x, int y, float lr = 0.01f);

    void save(const std::string& path) const;

    void load(const std::string& path);

private:
    std::vector<std::vector<float>> weights;
    std::vector<float> bias;
    
};

#endif
