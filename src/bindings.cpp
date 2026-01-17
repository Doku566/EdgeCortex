#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "edgecortex/allocator.hpp"
#include "edgecortex/ops.hpp"

namespace py = pybind11;
using namespace edgecortex;

// Helper to convert py::array to MatrixView
MatrixView get_view(py::array_t<float> arr) {
    py::buffer_info info = arr.request();
    if (info.ndim != 2) throw std::runtime_error("Inputs must be 2D arrays");
    return MatrixView{
        static_cast<float*>(info.ptr),
        static_cast<size_t>(info.shape[0]),
        static_cast<size_t>(info.shape[1])
    };
}

PYBIND11_MODULE(edgecortex_core, m) {
    m.doc() = "EdgeCortex Core C++ Module";

    // ... MemoryArena bindings ...
    
    // Expose GEMMs for benchmarking
    m.def("gemm_naive", [](py::array_t<float> A, py::array_t<float> B, py::array_t<float> C) {
        gemm_naive(get_view(A), get_view(B), get_view(C));
    }, "Naive Matrix Multiplication O(N^3)");

    m.def("gemm_tiled", [](py::array_t<float> A, py::array_t<float> B, py::array_t<float> C, int block_size) {
        gemm_tiled(get_view(A), get_view(B), get_view(C), block_size);
    }, py::arg("A"), py::arg("B"), py::arg("C"), py::arg("block_size")=32, "Tiled Matrix Multiplication (Cache Optimized)");

    py::class_<MemoryArena>(m, "MemoryArena", py::buffer_protocol())
        .def(py::init<size_t>(), py::arg("size_bytes"))
        .def("size", &MemoryArena::size)
        .def("used", &MemoryArena::used)
        .def("reset", &MemoryArena::reset)
        .def("allocate", [](MemoryArena& self, size_t size) {
            void* ptr = self.allocate(size);
            return reinterpret_cast<std::uintptr_t>(ptr);
        })
        .def_buffer([](MemoryArena &m) -> py::buffer_info {
            return py::buffer_info(
                m.data(),                               /* Pointer to buffer */
                sizeof(uint8_t),                        /* Size of one scalar */
                py::format_descriptor<uint8_t>::format(), /* Python struct-style format descriptor */
                1,                                      /* Number of dimensions */
                { m.size() },                           /* Buffer dimensions */
                { sizeof(uint8_t) }                     /* Strides (in bytes) for each index */
            );
        });
}
