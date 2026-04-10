#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include "logistic.h"

namespace py = pybind11;

float loss_function(
    LogisticRegression& model,
    torch::Tensor y,
    torch::Tensor yhat,
    int blockSize
) {
    TORCH_CHECK(y.is_cuda(),          "y must be a CUDA tensor");
    TORCH_CHECK(yhat.is_cuda(),       "yhat must be a CUDA tensor");
    TORCH_CHECK(y.is_contiguous(),    "y must be contiguous");
    TORCH_CHECK(yhat.is_contiguous(), "yhat must be contiguous");

    return model.LossFunction(y.data_ptr<float>(), yhat.data_ptr<float>(), y.numel(), blockSize);
}

torch::Tensor forward(
    LogisticRegression& model,
    torch::Tensor X,
    int n_samples
) {
    TORCH_CHECK(X.is_cuda(),       "X must be a CUDA tensor");
    TORCH_CHECK(X.is_contiguous(), "X must be contiguous");

    torch::Tensor out = torch::zeros({n_samples, 1}, X.options());

    model.forward(X.data_ptr<float>(), out.data_ptr<float>(), n_samples);

    return out;
}

PYBIND11_MODULE(mnistocr, m) {
    py::class_<LogisticRegression>(m, "LogisticRegression")
        .def(py::init<int>())
        .def("loss_function", &loss_function, py::arg("y"), py::arg("yhat"), py::arg("block_size") = 256)
        .def("forward", &forward, py::arg("X"), py::arg("n_samples"));
        .def("predict", &LogisticRegression::predict)
        .def("train_step", &LogisticRegression::train_step);
}