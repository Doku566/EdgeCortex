#include "edgecortex/ops.hpp"
#include <algorithm>

namespace edgecortex {

void gemm_naive(const MatrixView& A, const MatrixView& B, MatrixView& C) {
    // C = A * B
    // A: MxK, B: KxN, C: MxN
    
    // Ensure dimensions match
    if (A.cols != B.rows || C.rows != A.rows || C.cols != B.cols) {
        // In prod: throw proper error. Here we assume check done by caller.
        return;
    }

    // Initialize C to 0
    // (Or assume it's zeroed/being accumulated into. Here we overwrite)
    for (size_t i = 0; i < C.rows * C.cols; ++i) C.data[i] = 0.0f;

    for (size_t i = 0; i < A.rows; ++i) {
        for (size_t j = 0; j < B.cols; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < A.cols; ++k) {
                sum += A.at(i, k) * B.at(k, j);
            }
            C.at(i, j) = sum;
        }
    }
}

void gemm_tiled(const MatrixView& A, const MatrixView& B, MatrixView& C, size_t block_size) {
    // Check dims
    if (A.cols != B.rows || C.rows != A.rows || C.cols != B.cols) return;

    size_t M = A.rows;
    size_t N = B.cols;
    size_t K = A.cols;
    size_t BS = block_size;

    // Zero out C
    for (size_t i = 0; i < M * N; ++i) C.data[i] = 0.0f;

    // Loop over blocks
    for (size_t ii = 0; ii < M; ii += BS) {
        for (size_t jj = 0; jj < N; jj += BS) {
            for (size_t kk = 0; kk < K; kk += BS) {
                
                // Compute block bounds
                size_t i_max = std::min(ii + BS, M);
                size_t j_max = std::min(jj + BS, N);
                size_t k_max = std::min(kk + BS, K);

                // Small micro-kernel for the block
                // This "inner loop" fits in L1 Cache
                for (size_t i = ii; i < i_max; ++i) {
                    for (size_t j = jj; j < j_max; ++j) { // Note: J loop here can be improved, but usually K or J works
                        float sum = 0.0f;
                        // Pre-load C value if accumulating
                        sum = C.at(i, j);
                        
                        // Vectorization friendly loop
                        // #pragma omp simd
                        for (size_t k = kk; k < k_max; ++k) {
                            sum += A.at(i, k) * B.at(k, j);
                        }
                        
                        C.at(i, j) = sum;
                    }
                }
            }
        }
    }
}

} // namespace edgecortex
