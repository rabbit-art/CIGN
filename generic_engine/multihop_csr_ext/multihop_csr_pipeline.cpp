#include "multihop_csr_pipeline.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<MultiHopCsrPlan, std::shared_ptr<MultiHopCsrPlan>>(m, "MultiHopCsrPlan")
        .def(py::init<torch::Tensor, torch::Tensor, torch::Tensor,
                      torch::Tensor, torch::Tensor, torch::Tensor,
                      int64_t, int64_t, int64_t>(),
             py::arg("p_crow"),
             py::arg("p_col"),
             py::arg("p_values"),
             py::arg("pt_crow"),
             py::arg("pt_col"),
             py::arg("pt_values"),
             py::arg("rows"),
             py::arg("dense_cols"),
             py::arg("algorithm"))
        .def("prepare", &MultiHopCsrPlan::prepare)
        .def("forward3", &MultiHopCsrPlan::forward3)
        .def("backward3", &MultiHopCsrPlan::backward3)
        .def("algorithm", &MultiHopCsrPlan::algorithm)
        .def("workspace_bytes", &MultiHopCsrPlan::workspace_bytes)
        .def("info", &MultiHopCsrPlan::info);
}
