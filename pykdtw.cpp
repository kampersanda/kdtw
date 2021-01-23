#include <pybind11/pybind11.h>

#include "kdtw.hpp"

namespace py = pybind11;

PYBIND11_MODULE(pykdtw, m) {
    m.doc() = "Regularized Dynamic Time Warping Kernel (KDTW)";

    py::class_<MatrixF64>(m, "MatrixF64")
        .def(py::init<int64_t, int64_t>())
        .def("get", &MatrixF64::get)
        .def("set", &MatrixF64::set)
        .def("nrows", &MatrixF64::nrows)
        .def("ncols", &MatrixF64::ncols)
        .def("resize", &MatrixF64::resize);

    m.def("log_KDTW", &log_KDTW);
}
