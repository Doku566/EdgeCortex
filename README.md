# EdgeCortex: Linear Memory Tensor Engine

Implementation of a bare-metal tensor engine in C++ to investigate manual memory management on commodity hardware. This project replaces standard `malloc` chains with a contiguous arena allocator to predict and control cache behavior during inference.

## Memory Architecture: The Arena
The core component is `MemoryArena` (`src/allocator.cpp`), which reserves a monolithic block of virtual memory at startup.

### Design Decision: 64-Byte Alignment
I utilized `posix_memalign` (Linux) / `_aligned_malloc` (Windows) with a 64-byte boundary.
*   **Why**: A standard CPU cache line is 64 bytes.
*   **Effect**: Ensures that every tensor's data payload begins at the start of a cache line. This eliminates "false sharing" in potential multi-threaded scenarios and allows SIMD instructions (AVX2 requires 32-byte alignment) to load data without penalty.
*   **Cost**: We potentially waste average 32 bytes of padding per allocation (Internal Fragmentation).

## Kernel Optimization: GEMM
Matrix Multiplication is the CPU-bound operation. I implemented a Tiled approach to benchmark against the naive implementation.

**Benchmark Results (Intel Core i7 / 512x512 Matrices)**
| Implementation | Latency (avg) | Notes |
| :--- | :--- | :--- |
| **Naive Loop ($O(N^3)$)** | 110.3 ms | Heavy L1/L2 cache misses. |
| **Blocked Tiling (32x32)** | 67.0 ms | **1.65x Speedup**. working set fits in L1 (32KB). |

*Verified via `python/benchmarks/benchmark_gemm.py`*

## Technical Challenges
**Pointer Arithmetic for Alignment**:
Implementing `allocate()` was not just `ptr += size`. To maintain the 64-byte guarantee, I had to calculate the padding offset manually:
```cpp
size_t padding = (alignment - (current_addr % alignment)) % alignment;
```
Getting this wrong resulted in immediate Segfaults when AVX intrinsics attempted to load unaligned data during testing.

## Known Limitations (Trade-offs)
1.  **Thread Safety**: The `MemoryArena::allocate` method is **NOT** thread-safe.
    *   *Trade-off*: Adding a `std::mutex` would introduce lock contention during high-frequency small allocations. I prioritized single-threaded throughput for the inference loop.
2.  **No Deallocation**: There is no `free(ptr)`. You must call `arena.reset()` to wipe everything. This is acceptable for inference (load model -> run batch -> reset), but effectively useless for general-purpose computing.
3.  **Fixed Buffer Size**: The bindings assume a static pool size. If the model exceeds the pre-allocated arena, the process crashes.
