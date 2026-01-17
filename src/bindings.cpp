#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "edgecortex/allocator.hpp"

namespace py = pybind11;
using namespace edgecortex;

PYBIND11_MODULE(edgecortex_core, m) {
    m.doc() = "EdgeCortex Core C++ Module";

    py::class_<MemoryArena>(m, "MemoryArena", py::buffer_protocol())
        .def(py::init<size_t>(), py::arg("size_bytes"))
        .def("size", &MemoryArena::size)
        .def("used", &MemoryArena::used)
        .def("reset", &MemoryArena::reset)
        .def("allocate", [](MemoryArena& self, size_t size) {
            // Helper to return a raw pointer as an integer (for debugging mostly)
            // In real usage, we use the buffer protocol.
            void* ptr = self.allocate(size);
            return reinterpret_cast<std::uintptr_t>(ptr);
        })
        // Buffer Protocol Implementation
        // This allows `numpy.array(arena)` or `memoryview(arena)`
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
