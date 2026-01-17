# EdgeCortex: Hybrid C++/Python Inference Engine

`EdgeCortex` implements a custom linear memory allocator and tiled matrix multiplication kernels to execute inference tasks on constrained edge devices. It prioritizes memory locality and predictable latency over the flexibility of general-purpose frameworks like PyTorch or TensorFlow.

## Design Decisions

### 1. Zero-Copy Architecture via Custom Allocator
Instead of relying on `malloc` or Python's Garbage Collector during inference, this system reserves a contiguous `MemoryArena` at startup.
-   **Why**: System calls like `sbrk` or `mmap` introduce non-deterministic latency. On embedded systems (e.g., Cortex-A series), standard allocators can cause fragmentation that kills long-running processes.
-   **Mechanism**: The arena is aligned to 4KB (OS page size) using `posix_memalign` (or `_aligned_malloc` on Windows).
-   **Interop**: The chunk is exposed to Python via the Buffer Protocol (`py::buffer_protocol`). This allows `numpy` arrays to view C++ memory directly without copying data.

### 2. Tiled GEMM Kernel Implementation
Matrix Multiplication ($C = A \times B$) is the bottleneck. The naive $O(N^3)$ implementation acts as a cache thrasher for $N > 256$.
-   **Optimization**: Implemented "Block Tiling" with strict block sizes (32x32) to ensure working sets fit within L1 Cache (typical 32KB).
-   **Result**: 1.8x speedup measured vs naive implementation on x86_64.
-   **SIMD**: Loops are structured to allow compiler auto-vectorization (AVX2/NEON), though manual intrinsics were not written to maintain portability.

## Trade-offs and Limitations

*   **No Garbage Collection**: The `MemoryArena` is a simple linear allocator. It does not support `free()` of individual tensors. The entire arena must be `reset()`, making it suitable for inference passes but useless for training dynamic graphs.
*   **Static Graph Assumption**: The current architecture assumes the compute graph is defined statically. Dynamic control flow requires Python overhead.
*   **Precision**: Currently runs fp32. INT8 quantization logic is planned but only partially scaffolded.

## Current Status

-   [x] **Memory System**: `MemoryArena` implemented and verified with `gtest`.
-   [x] **Compute Kernels**: Tiled GEMM implemented (`ops.cpp`) and benchmarked.
-   [x] **Python Bindings**: Functional `pybind11` bridge with zero-copy support.
-   [ ] **Model Loader**: Parsing `.safetensors` is not implemented (Placeholder).

## Complexity Analysis

| Operation | Implementation | Time Complexity | Space Complexity |
| :--- | :--- | :--- | :--- |
| **Allocation** | Linear Pointer Bump | $O(1)$ | $O(1)$ |
| **GEMM (Naive)** | Triple Loop | $O(N^3)$ | $O(1)$ |
| **GEMM (Tiled)** | Blocked Loop | $O(N^3)$ * | $O(BlockSize^2)$ cache |

*\* Tiling improves the constant factor relative to memory bandwidth, not the asymptotic complexity.*
