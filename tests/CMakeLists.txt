include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/5376968f6948923e2411081fd9372e71a59d8e77.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(test_allocator test_allocator.cpp ../src/allocator.cpp)
target_link_libraries(test_allocator GTest::gtest_main)
target_include_directories(test_allocator PRIVATE ../include)

add_test(NAME AllocatorTest COMMAND test_allocator)
