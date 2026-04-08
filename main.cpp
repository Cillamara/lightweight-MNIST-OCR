#include <pybind11/pybind11.h>

namespace py = pybind11;

float func(float arg1, float arg2) {
    return arg1 + arg2;
}

PYBIND11_MODULE(module_name, handle){
    handle.doc() = "example";
    handle.def("test_func", &func);
}