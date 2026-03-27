include(FetchContent)

FetchContent_Declare(googletest
  GIT_REPOSITORY https://github.com/google/googletest
  GIT_TAG v1.14.0
)

# Prevent GoogleTest from overriding our compiler/linker options
set(gtest_force_shared_crt OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

add_executable(modelbuilder_tests
  test_modelbuilder.cpp
)

target_link_libraries(modelbuilder_tests
  PRIVATE
    ModelBuilder::ModelBuilder
    GTest::gtest_main
)

target_compile_definitions(modelbuilder_tests
  PRIVATE
    MODELBUILDER_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/.."
)

include(GoogleTest)
gtest_discover_tests(modelbuilder_tests)

