#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>

namespace edgecortex {

/**
 * @brief High-performance linear memory allocator.
 * 
 * Manages a large pre-allocated block of memory to minimize syscall overhead
 * (malloc/free) during inference. Designed for zero-copy integration with Python.
 */
class MemoryArena {
public:
    /**
     * @brief Initialize the arena with a fixed size.
     * @param size_bytes Total size of the arena. Must be page-aligned ideally.
     */
    explicit MemoryArena(size_t size_bytes);

    ~MemoryArena();

    // Disable copy/move to prevent double-free logic for now
    MemoryArena(const MemoryArena&) = delete;
    MemoryArena& operator=(const MemoryArena&) = delete;

    /**
     * @brief Allocate a block of memory from the arena.
     * @param size Size in bytes.
     * @param alignment Alignment requirement (default 64 bytes for AVX-512).
     * @return Pointer to the allocated block.
     */
    [[nodiscard]] void* allocate(size_t size, size_t alignment = 64);

    /**
     * @brief Reset the allocator to the beginning. 
     * Extremely fast "free" for all objects in the arena.
     */
    void reset();

    /**
     * @brief Get the raw pointer to the start of the arena.
     */
    void* data() const { return memory_block_; }
    
    /**
     * @brief Get total capacity.
     */
    size_t size() const { return size_; }

    /**
     * @brief Get currently widely used bytes.
     */
    size_t used() const { return offset_; }

private:
    void* memory_block_;
    size_t size_;
    size_t offset_;
};

} // namespace edgecortex
