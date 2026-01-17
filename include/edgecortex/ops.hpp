#pragma once

#include <vector>
#include <cstddef>

namespace edgecortex {

/**
 * @brief Lightweight view over a raw memory buffer to interpret it as a matrix.
 * Does NOT own the memory.
 */
struct MatrixView {
    float* data;
    size_t rows;
    size_t cols;

    float& at(size_t r, size_t c) {
        return data[r * cols + c];
    }
    
    const float& at(size_t r, size_t c) const {
        return data[r * cols + c];
    }
};

/**
 * @brief Naive Matrix Multiplication (Baseline).
 * O(N^3) complexity. Trashes CPU cache for large N.
 */
void gemm_naive(const MatrixView& A, const MatrixView& B, MatrixView& C);

/**
 * @brief Tiled Matrix Multiplication (Optimized).
 * Uses Loop Blocking to keep active data subsets in L1/L2 Cache.
 * @param block_size Size of the tile (e.g., 32 or 64).
 */
void gemm_tiled(const MatrixView& A, const MatrixView& B, MatrixView& C, size_t block_size = 32);

} // namespace edgecortex
