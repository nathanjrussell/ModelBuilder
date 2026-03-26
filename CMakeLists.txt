cmake_minimum_required(VERSION 3.24)

project(ModelBuilder
  VERSION 0.1.0
  DESCRIPTION "Modular components to build a model on categorical data"
  LANGUAGES CXX
)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(MODELBUILDER_BUILD_EXAMPLES "Build ModelBuilder examples" ON)
option(MODELBUILDER_BUILD_TESTS "Build ModelBuilder tests" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(ModelBuilder
  src/ModelBuilder.cpp
)
add_library(ModelBuilder::ModelBuilder ALIAS ModelBuilder)

target_include_directories(ModelBuilder
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Dependencies
include(FetchContent)

FetchContent_Declare(LeftTree
  GIT_REPOSITORY https://github.com/nathanjrussell/LeftTree
  GIT_TAG v1.0.0
)

FetchContent_Declare(ContingencyTable
  GIT_REPOSITORY https://github.com/nathanjrussell/ContingencyTable
  GIT_TAG v1.0.0
)

FetchContent_Declare(CPPTypeConcepts
  GIT_REPOSITORY https://github.com/nathanjrussell/CPP-Type-Concepts.git
  GIT_TAG v1.0.0
)

FetchContent_MakeAvailable(LeftTree ContingencyTable CPPTypeConcepts)

target_link_libraries(ModelBuilder
  PUBLIC
    # These target names depend on what the dependency projects export.
    # Uncomment/adjust once confirmed (build will still succeed if unused here).
    # LeftTree::LeftTree
    # ContingencyTable::ContingencyTable
  PRIVATE
    CPPTypeConcepts::cpp_type_concepts
)

# Warnings (nice defaults for dev)
if(MSVC)
  target_compile_options(ModelBuilder PRIVATE /W4)
else()
  target_compile_options(ModelBuilder PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(MODELBUILDER_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

if(MODELBUILDER_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# ---- Install / Package ----
install(TARGETS ModelBuilder
  EXPORT ModelBuilderTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT ModelBuilderTargets
  FILE ModelBuilderTargets.cmake
  NAMESPACE ModelBuilder::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ModelBuilder
)

configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ModelBuilderConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/ModelBuilderConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ModelBuilder
)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/ModelBuilderConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/ModelBuilderConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/ModelBuilderConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ModelBuilder
)

