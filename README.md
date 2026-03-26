# ModelBuilder
This combines a significant number of modular components to build a model on categorical data.

## Build
```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
```

## Run tests
```sh
ctest --test-dir build --output-on-failure
```

## CMake usage (as a dependency)
This project exports a CMake target:
- `ModelBuilder::ModelBuilder`

It also fetches (via `FetchContent`) the following dependencies:
- `LeftTree` (tag `v1.0.0`)
- `ContingencyTable` (tag `v1.0.0`)
- `CPP-Type-Concepts` (tag `v1.0.0`, target `CPPTypeConcepts::cpp_type_concepts`)
