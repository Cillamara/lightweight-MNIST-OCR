#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "logistic.h"

namespace py = pybind11;

PYBIND11_MODULE(mnistocr, m) {
    py::class_<LogisticRegression>(m, "LogisticRegression")
        .def(py::init<int, int>(), py::arg("n_features"), py::arg("n_classes") = 10)
        .def("train_step", &LogisticRegression::train_step)
        .def("predict",    &LogisticRegression::predict)
        .def("save",       &LogisticRegression::save)
        .def("load",       &LogisticRegression::load);
}