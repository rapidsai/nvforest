# cuForest Build From Source Guide

## Setting up your build environment

To install cuForest from source, ensure the following dependencies are met:

> **Note:** The easiest way to setup a fully functional build environment is to use the conda environment files located in `conda/environments/all_*.yaml`. These files contain all the dependencies listed below except for clang-format (only needed for development/contributing). To create a development environment, see the [recommended Conda setup](#recommended-conda-setup) at the end of this section.

**Hardware needed to run cuForest:**
cuForest is able to support following NVIDIA GPUs and CUDA versions:
- **CUDA 12.x**: compute capability 7.0 or higher (Volta™ architecture or newer)
- **CUDA 13.x**: compute capability 7.5 or higher (Turing™ architecture or newer)

It is possible to build and run cuForest on machines without a GPU; in such machines, cuForest will use the CPU to run inference.

**Software dependencies:**
1. gcc (>= 13.0)
2. cmake (>= 3.30.4)
3. ninja - build system used by default
4. Python (>= 3.11 and <= 3.13)
5. Cython (>= 3.0.0)
6. CUDA Toolkit (>= 12.2). Not needed when building cuForest without GPU support.

**RAPIDS Ecosystem Libraries:**

These RAPIDS libraries must match the cuML version (e.g., all version 25.10 if building cuML 25.10):

*C++ Libraries:*
- [librmm](https://github.com/rapidsai/rmm) - RAPIDS Memory Manager (C++ library)
- [libraft](https://github.com/rapidsai/raft) - RAPIDS CUDA accelerated algorithms (C++ library)

*Python Packages:*
- [rmm](https://github.com/rapidsai/rmm) - RAPIDS Memory Manager (Python package)
- [pylibraft](https://github.com/rapidsai/raft) - RAPIDS CUDA accelerated algorithms (Python package)

**Python Build Dependencies:**
- scikit-build-core
- rapids-build-backend

**Python Runtime Dependencies:**

For detailed version requirements of runtime dependencies (numpy, scikit-learn, scipy, joblib, numba, cupy, treelite, etc.), please see [docs/source/supported_versions.rst](docs/source/supported_versions.rst).

**Other External Libraries:**
- treelite
- rapids-logger

**For development only:**
- clang-format (= 20.1.4) - enforces uniform C++ coding style; required for pre-commit hooks and CI checks. The packages `clang=20` and `clang-tools=20` from the conda-forge channel should be sufficient, if you are using conda. If not using conda, install the right version using your OS package manager.

### Recommended Conda Setup

It is recommended to use conda for environment/package management. If doing so, development environment .yaml files are located in `conda/environments/all_*.yaml`. These files contain most of the dependencies mentioned above. To create a development environment named `cuforest_dev`, you can use the following commands (adjust the YAML filename to match your CUDA version and architecture):

```bash
conda create -n cuforest_dev python=3.13
conda env update -n cuforest_dev --file=conda/environments/all_cuda-131_arch-$(uname -m).yaml
conda activate cuforest_dev
```

## Installing from Source

### Recommended Process

As a convenience, a `build.sh` script is provided to simplify the build process. The libraries will be installed to `$INSTALL_PREFIX` if set (e.g., `export INSTALL_PREFIX=/install/path`), otherwise to `$CONDA_PREFIX`.
```bash
$ ./build.sh                           # build the cuForest libraries, tests, and python package, then
                                       # install them to $INSTALL_PREFIX if set, otherwise $CONDA_PREFIX
```
For workflows that involve frequent switching among branches or between debug and release builds, it is recommended that you install [ccache](https://ccache.dev/) and make use of it by passing the `--ccache` flag to `build.sh`.

To build individual components, specify them as arguments to `build.sh`:
```bash
$ ./build.sh libcuforest               # build and install the cuForest C++ and C-wrapper libraries
$ ./build.sh cuforest                  # build and install the cuForest Python package
```

Other `build.sh` options:
```bash
$ ./build.sh clean                         # remove any prior build artifacts and configuration (start over)
$ ./build.sh libcuforest -v                # build and install libcuforest with verbose output
$ ./build.sh libcuforest -g                # build and install libcuforest for debug
$ PARALLEL_LEVEL=8 ./build.sh libcuforest  # build and install libcuforest limiting parallel build jobs to 8 (ninja -j8)
$ ./build.sh libcuforest -n                # build libcuforest but do not install
$ ./build.sh --ccache                      # use ccache to cache compilations, speeding up subsequent builds
```

By default, Ninja is used as the cmake generator. To override this and use, e.g., `make`, define the `CMAKE_GENERATOR` environment variable accordingly:
```bash
CMAKE_GENERATOR='Unix Makefiles' ./build.sh
```

To run the C++ unit tests (optional), from the repo root:

```bash
$ cd cpp/build
$ ctest
```

If you want a list of the available C++ tests:
```bash
$ ctest -N
```

To run all Python tests, from the repo root:
```bash
$ cd python
$ pytest -v
```

If you want a list of the available Python tests:
```bash
$ pytest cuforest/tests --collect-only
```

**Note:** Some tests require `xgboost`. If running tests in conda devcontainers, you must install the `xgboost` conda package manually. See `dependencies.yaml` for version information.

### Manual Process

Once dependencies are present, follow the steps below:

1. Clone the repository:
```bash
$ git clone https://github.com/rapidsai/cuforest.git
```

2. Build and install `libcuforest++` (C++/CUDA library containing the cuML algorithms), starting from the repository root folder:
```bash
$ cd cpp
$ mkdir build && cd build
$ cmake ..
```

**Note:** If CUDA is not in your PATH, you may need to set `CUDA_BIN_PATH` before running cmake:
```bash
$ export CUDA_BIN_PATH=$CUDA_HOME  # Default: /usr/local/cuda
```

If using a conda environment (recommended), configure cmake for `libcuforest++`:

```bash
$ cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
```

**Note:** You may see the following warning depending on your cmake version and `CMAKE_INSTALL_PREFIX`. This warning can be safely ignored:
```
Cannot generate a safe runtime search path for target ml_test because files
in some directories may conflict with libraries in implicit directories:
```
To silence it, add `-DCMAKE_IGNORE_PATH=$CONDA_PREFIX/lib` to your `cmake` command.

To reduce compile times, you can specify GPU compute capabilities to compile for. For example, for Volta GPUs:

```bash
$ cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DCMAKE_CUDA_ARCHITECTURES="70"
```

Or for multiple architectures (e.g., Ampere and Hopper):

```bash
$ cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DCMAKE_CUDA_ARCHITECTURES="80;86;90"
```

You may also wish to make use of `ccache` to reduce build times when switching among branches or between debug and release builds:

```bash
$ cmake .. -DUSE_CCACHE=ON
```

There are many options to configure the build process, see the [customizing build section](#custom-build-options).

3. Build `libcuforest++` and `libcuforest`:

```bash
$ ninja -j
$ ninja install
```

To run tests (optional):
```bash
$ ctest
```

To build doxygen docs for all C/C++ source files:
```bash
$ ninja doc
```

4. Build and install the `cuforest` python package:

From the repository root:
```bash
$ python -m pip install --no-build-isolation --no-deps --config-settings rapidsai.disable-cuda=true python/cuforest
```

To run Python tests (optional):

```bash
$ cd python
$ pytest -v
```

If you want a list of the available tests:
```bash
$ pytest cuforest/tests --collect-only
```

### Custom Build Options

#### libcuforest++

cuForest's cmake has the following configurable flags available:

| Flag | Possible Values | Default Value | Behavior |
| --- | --- | --- | --- |
| CUFOREST_ENABLE_GPU | [ON, OFF] | ON | Enable/disable GPU support |
| BUILD_SHARED_LIBS | [ON, OFF] | ON | Whether to build libcuforest++ as a shared library |
| BUILD_CUFOREST_TESTS | [ON, OFF]  | ON  |  Enable/disable building cuForest C++ test executables |
| CUDA_ENABLE_KERNEL_INFO | [ON, OFF]  | OFF  | Enable/disable kernel resource usage info in nvcc. |
| CUDA_ENABLE_LINE_INFO | [ON, OFF]  | OFF  | Enable/disable lineinfo in nvcc.  |
| DETECT_CONDA_ENV | [ON, OFF] | ON | Use detection of conda environment for dependencies. If set to ON, and no value for CMAKE_INSTALL_PREFIX is passed, then it will assign it to $CONDA_PREFIX (to install in the active environment).  |
| DIABLE_DEPRECATION_WARNINGS | [ON, OFF]  | ON  | Set to `ON` to disable deprecation warnings  |
| DISABLE_OPENMP | [ON, OFF]  | OFF  | Set to `ON` to disable OpenMP  |
| NVTX | [ON, OFF]  | OFF  | Enable/disable nvtx markers in libcuforest++.  |
| USE_CCACHE | [ON, OFF] | OFF | Whether to cache build artifacts with ccache. |
| CUDA_STATIC_RUNTIME | [ON, OFF] | OFF | Whether to statically link the CUDA runtime. |
| CUFOREST_USE_RAFT_STATIC | [ON, OFF] | OFF | Whether to statically link the RAFT library. |
| CUFOREST_USE_TREELITE_STATIC | [ON, OFF] | OFF | Whether to statically link the Treelite library. |
| CUFOREST_EXPORT_TREELITE_LINKAGE | [ON, OFF] | OFF | Whether to publicly link Treelite to libcuforest++ |
| CUDA_WARNINGS_AS_ERRORS | [ON, OFF] | ON | Treat all warnings from CUDA as errors |
| CMAKE_CUDA_ARCHITECTURES |  List of GPU architectures, semicolon-separated | Empty  | List the GPU architectures to compile the GPU targets for. Set to "NATIVE" to auto detect GPU architecture of the system, set to "ALL" to compile for all RAPIDS supported archs.  |
