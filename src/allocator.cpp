#include "edgecortex/allocator.hpp"
#include <cstdlib>
#include <iostream>
#include <new>

// Platform specific includes for aligned_alloc
#if defined(_MSC_VER)
    #include <malloc.h>
    #define ALIGNED_ALLOC(alignment, size) _aligned_malloc(size, alignment)
    #define ALIGNED_FREE(ptr) _aligned_free(ptr)
#else
    #define ALIGNED_ALLOC(alignment, size) aligned_alloc(alignment, size)
    #define ALIGNED_FREE(ptr) free(ptr)
#endif

namespace edgecortex {

MemoryArena::MemoryArena(size_t size_bytes) 
    : size_(size_bytes), offset_(0) {
    
    // Default alignment to 4KB (Page Size) for the main block
    // This facilitates better interactions with OS paging mechanisms
    size_t page_size = 4096;
    
    // Ensure size is a multiple of page size
    if (size_ % page_size != 0) {
        size_ = ((size_ / page_size) + 1) * page_size;
    }

    memory_block_ = ALIGNED_ALLOC(page_size, size_);
    
    if (!memory_block_) {
        throw std::bad_alloc();
    }
}

MemoryArena::~MemoryArena() {
    if (memory_block_) {
        ALIGNED_FREE(memory_block_);
    }
}

void* MemoryArena::allocate(size_t size, size_t alignment) {
    // Calculate memory address for current offset
    std::uintptr_t current_ptr = reinterpret_cast<std::uintptr_t>(memory_block_) + offset_;
    
    // Calculate padding needed for alignment
    std::uintptr_t padding = 0;
    if (alignment > 0) {
        std::uintptr_t misalignment = current_ptr % alignment;
        if (misalignment > 0) {
            padding = alignment - misalignment;
        }
    }

    if (offset_ + padding + size > size_) {
        // OOM in Arena
        // In a real inference engine, we might want to fallback or log comprehensively
        throw std::runtime_error("EdgeCortex MemoryArena: Out of Memory");
    }

    offset_ += padding;
    void* ptr = reinterpret_cast<void*>(reinterpret_cast<std::uintptr_t>(memory_block_) + offset_);
    offset_ += size;

    return ptr;
}

void MemoryArena::reset() {
    offset_ = 0;
    // Note: We don't zero out memory here for performance reasons. 
    // Inference runs assume they write output buffers before reading.
}

} // namespace edgecortex
