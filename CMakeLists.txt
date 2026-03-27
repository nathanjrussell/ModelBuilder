cmake_minimum_required(VERSION 3.24)

project(ModelBuilder
  VERSION 0.2.0
  DESCRIPTION "Modular components to build a model on categorical data"
  LANGUAGES CXX
)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(MODELBUILDER_BUILD_EXAMPLES "Build ModelBuilder examples" ON)
option(MODELBUILDER_BUILD_TESTS "Build ModelBuilder tests" ON)
option(MODELBUILDER_ENABLE_INSTALL "Enable install/export rules (requires dependencies available as packages)" OFF)

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
  GIT_TAG v2.1.0
)

FetchContent_Declare(CPPTypeConcepts
  GIT_REPOSITORY https://github.com/nathanjrussell/CPP-Type-Concepts.git
  GIT_TAG v1.0.0
)

FetchContent_MakeAvailable(LeftTree ContingencyTable CPPTypeConcepts)

get_target_property(_LEFTTREE_INCLUDE_DIRS LeftTree::LeftTree INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(_CONTINGENCY_INCLUDE_DIRS ContingencyTable::ContingencyTable INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(_DATATABLE_INCLUDE_DIRS DataTable::DataTable INTERFACE_INCLUDE_DIRECTORIES)

target_link_libraries(ModelBuilder
  PUBLIC
    CPPTypeConcepts::cpp_type_concepts
  PRIVATE
    LeftTree::LeftTree
    ContingencyTable::ContingencyTable
)

# We need the dependency headers when building ModelBuilder itself, but we don't want to
# export these FetchContent deps as requirements of the installed target.
target_include_directories(ModelBuilder
  PUBLIC
    $<BUILD_INTERFACE:${_LEFTTREE_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${_CONTINGENCY_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${_DATATABLE_INCLUDE_DIRS}>
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
# When developing this repo, we build dependencies via FetchContent. Those targets are not meant
# to be exported from this package. To keep development builds working, we only enable the install
# rules when ModelBuilder is the top-level project.
if(MODELBUILDER_ENABLE_INSTALL)
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
endif()

