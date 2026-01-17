#include <gtest/gtest.h>
#include "edgecortex/allocator.hpp"

// Simple fixture
class AllocatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup code if needed
    }
    
    void TearDown() override {
        // Cleanup code if needed
    }
};

TEST_F(AllocatorTest, Initialization) {
    // Should align size to page size (4096)
    // Requesting 100 bytes should get at least 4096
    edgecortex::MemoryArena arena(100);
    EXPECT_GE(arena.size(), 4096);
    EXPECT_EQ(arena.used(), 0);
}

TEST_F(AllocatorTest, BasicAllocation) {
    edgecortex::MemoryArena arena(1024);
    void* ptr1 = arena.allocate(128);
    EXPECT_NE(ptr1, nullptr);
    EXPECT_EQ(arena.used(), 128);

    void* ptr2 = arena.allocate(128);
    EXPECT_NE(ptr2, nullptr);
    EXPECT_EQ(arena.used(), 256);
    
    // Pointers should be distinct
    EXPECT_NE(ptr1, ptr2);
}

TEST_F(AllocatorTest, Alignment) {
    edgecortex::MemoryArena arena(4096);
    
    // Allocate 1 byte to throw off alignment
    arena.allocate(1);
    
    // Request 64-byte alignment
    void* ptr = arena.allocate(128, 64);
    std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(ptr);
    
    EXPECT_EQ(addr % 64, 0);
}

TEST_F(AllocatorTest, OutOfMemory) {
    edgecortex::MemoryArena arena(4096);
    // Allocate all
    arena.allocate(4090);
    
    // Try to allocate more than remains
    EXPECT_THROW(arena.allocate(100), std::runtime_error);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
