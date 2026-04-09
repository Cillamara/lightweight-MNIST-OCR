#include <pybind11/pybind11.h>
#include "logistic.h"

namespace py = pybind11;

PYBIND11_MODULE(mnistocr, m) {
    py::class_<LogisticRegression>(m, "LogisticRegression")
        .def(py::init<int>()) 
        .def("predict", &LogisticRegression::predict)
        .def("train_step", &LogisticRegression::train_step);
}