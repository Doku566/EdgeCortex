import time
import numpy as np
import sys
import os

# Import C++ Core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../build/Release")))
try:
    import edgecortex_core
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../build")))
    import edgecortex_core

def benchmark(N=512, iterations=3):
    print(f"Benchmarking Matrix Multiplication (N={N})...")
    
    A = np.random.rand(N, N).astype(np.float32)
    B = np.random.rand(N, N).astype(np.float32)
    C = np.zeros((N, N), dtype=np.float32)
    
    # 1. Naive
    start = time.time()
    for _ in range(iterations):
        edgecortex_core.gemm_naive(A, B, C)
    dt_naive = (time.time() - start) / iterations
    print(f"Naive: {dt_naive:.4f}s")
    
    # 2. Tiled
    start = time.time()
    for _ in range(iterations):
        edgecortex_core.gemm_tiled(A, B, C, 32)
    dt_tiled = (time.time() - start) / iterations
    print(f"Tiled (BLK=32): {dt_tiled:.4f}s")
    
    # Speedup
    if dt_tiled > 0:
        print(f"Speedup: {dt_naive / dt_tiled:.2f}x")

    # 3. Correctness Check (vs NumPy/BLAS)
    print("Verifying correctness...")
    Ref = A @ B
    edgecortex_core.gemm_tiled(A, B, C, 32)
    
    # Allow some float error
    if np.allclose(C, Ref, atol=1e-3):
        print("PASS: Tiled implementation matches NumPy")
    else:
        print("FAIL: Output mismatch")
        diff = np.abs(C - Ref)
        print(f"Max Diff: {np.max(diff)}")

if __name__ == "__main__":
    benchmark(N=256)
    benchmark(N=512)
    # benchmark(N=1024) # Uncomment for larger stress test
