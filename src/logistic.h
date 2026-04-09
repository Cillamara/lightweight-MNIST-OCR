#ifndef LOGISTIC_H
#define LOGISTIC_H

#include <vector>

class LogisticRegression 
{
public:
    LogisticRegression() {}

    LogisticRegression(int n_features) : weights(n_features, 0.0f) {}

    float predict(const std::vector<float>& x) const;
    
    void train_step(const std::vector<float>& x, float y, float lr = 0.01f);

private:
    std::vector<float> weights;
    
};

#endif