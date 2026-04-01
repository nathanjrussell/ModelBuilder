# ModelBuilder
This combines a significant number of modular components to build a model on categorical data.

## Build
```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
```

### Build with optimizations (-O3)
The usual way to get `-O3` with CMake is to build in `Release` mode:

```sh
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release -j
```

If you want to force `-O3` explicitly (in addition to what your toolchain uses for Release):

```sh
cmake -S . -B build-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS_RELEASE="-O3"
cmake --build build-release -j
```

## Run tests
```sh
ctest --test-dir build --output-on-failure
```

## Run example
This repo builds an example executable named `modelbuilder_example`.

### Build (if you haven't already)
```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
```

### Run
For single-config generators (Makefiles/Ninja), the binary is typically here:
```sh
./build/examples/modelbuilder_example
```

For multi-config generators (e.g. Xcode/Visual Studio), it is typically under a config subdir:
```sh
./build/examples/Debug/modelbuilder_example
```

If you're using the checked-in `cmake-build-debug/` folder instead of `build/`:
```sh
./cmake-build-debug/examples/modelbuilder_example
```

## CMake usage (as a dependency)
This project exports a CMake target:
- `ModelBuilder::ModelBuilder`

It also fetches (via `FetchContent`) the following dependencies:
- `LeftTree` (tag `v1.0.0`)
 - `ContingencyTable` (tag `v4.2.0`)
- `CPP-Type-Concepts` (tag `v1.0.0`, target `CPPTypeConcepts::cpp_type_concepts`)
