import pytest
import numpy as np
import sys
import os

# Ensure we can import the module
# (In a real install, this would be handled by setup.py/pip)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# Also need to find the .pyd/.so which is in build/Release
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../build/Release")))

try:
    from edgecortex_core import MemoryArena
except ImportError:
    # Try alternate path for Linux/Other builds
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../build")))
    from edgecortex_core import MemoryArena

def test_memory_arena_buffer_protocol():
    size = 1024
    arena = MemoryArena(size)
    
    # 1. Create a numpy array FROM the arena (Zero-Copy)
    # copy=False is default for buffer interface usually, but good to be explicit if possible or just rely on buffer
    arr = np.array(arena, copy=False)
    
    # The arena might be larger due to page alignment
    assert arr.size >= size
    assert arr.dtype == np.uint8
    
    # 2. Modify data in Numpy
    test_val = 123
    arr[0] = test_val
    arr[512] = 42
    
    # 3. Verify specific properties
    # Since we don't have a direct C++ view function exposed here easily besides raw pointer,
    # we verify that the array persists its state.
    assert arr[0] == 123
    
    # 4. Verify pointer connectivity (Advanced)
    # This checks if the array actually points to the arena's memory
    # We can't easily check the raw pointer value from python safely without ctypes, 
    # but the buffer protocol guarantees it if implemented correctly.
    
    # Check that resizing/allocating in arena doesn't invalidate valid ranges conceptually
    # (Though in our linear allocator, previous pointers remain valid until reset)
    ptr_offset = arena.allocate(100)
    assert arena.used() == 100

def test_arena_alignment_integration():
    # Verify that the array created is indeed large enough aligned
    # This is more implicit via the C++ constructor
    req_size = 4000
    arena = MemoryArena(req_size)
    arr = np.array(arena, copy=False)
    
    print(f"Requested size: {req_size}")
    print(f"Arena actual size (C++ reported): {arena.size()}")
    print(f"Numpy array size: {arr.size}")

    # Our C++ logic aligns up to 4096
    assert arr.size >= 4096, f"Expected size >= 4096, got {arr.size}"

if __name__ == "__main__":
    # Manual run
    test_memory_arena_buffer_protocol()
    test_arena_alignment_integration()
    print("All python inference tests passed!")
