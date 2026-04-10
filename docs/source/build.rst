########################
Building from the source
########################

Setting up your build environment
=================================

To install nvForest from source, ensure the following dependencies are met:

**Hardware needed to run nvForest.**
nvForest is part of RAPIDS and follows the RAPIDS support matrix.
See https://docs.rapids.ai/platform-support/.
It is possible to build and run nvForest on machines without a GPU; in such machines, nvForest will use the CPU to run inference.

**Software dependencies.**
See https://docs.rapids.ai/platform-support/ for the list of required C++ compilers and Python interpreters.
In addition, nvForest requires Cython 3.0 or later.

.. note:: Building nvForest without GPU support

   It is possible to build nvForest without GPU support; in this case, the CUDA toolkit is not required.
   To build nvForest without GPU, set the CMake option ``NVFOREST_ENABLE_GPU=OFF``.

**RAPIDS libraries.**
The nvForest code base is updated in tandem with the rest of RAPIDS. So to build the latest nvForest, you
should use the latest version of RAPIDS as well. (For example, nvForest 26.04 will require 26.04 version of
all RAPIDS packages.)

**Python dependencies.**
Please see https://docs.rapids.ai/install/ for RAPIDS-wide version support.

We aim to meet the `SPEC 0 guidelines <https://scientific-python.org/specs/spec-0000/>`_ for minimal supported versions.

**For development only.**

* clang-format (= 20.1.8): enforces uniform C++ coding style; required for pre-commit hooks and CI checks. The packages ``clang=20`` and ``clang-tools=20`` from the conda-forge channel should be sufficient, if you are using conda. If not using conda, install the right version using your OS package manager.

.. note:: Use Conda to install all software dependencies

  We highly recommend the use of Conda, a package manager that lets you obtain all necessary
  software dependencies in a virtual environment.
  We provide environment definition files ``conda/environments/all_*.yaml`` containing all software
  dependencies for nvForest.

  To create a development environment named ``nvforest_dev``, use the following commands.

  .. code-block:: console

    $ conda create -n nvforest_dev python=3.13
    $ conda env update -n nvforest_dev \
        --file=conda/environments/all_cuda-131_arch-$(uname -m).yaml
    $ conda activate nvforest_dev

Installing from Source
======================

Option 1. Use the convenience wrapper script (Recommended)
----------------------------------------------------------

As a convenience, a ``build.sh`` script is provided to simplify the build process.
The libraries will be installed to ``$INSTALL_PREFIX`` if set (e.g., ``export INSTALL_PREFIX=/install/path``);
otherwise it will be installed to ``$CONDA_PREFIX``.

.. code-block:: bash

  # Build the nvForest libraries, tests, and python package, then
  # Install them to $INSTALL_PREFIX if set, otherwise $CONDA_PREFIX
  ./build.sh

For workflows that involve frequent switching among branches or between debug and release builds, it is recommended that you install `ccache <https://ccache.dev/>`_ and make use of it by passing the ``--ccache`` flag to ``build.sh``.

To build individual components, specify them as arguments to ``build.sh``:

.. code-block:: bash

  # Build and install the nvForest C++ and C-wrapper libraries
  ./build.sh libnvforest

  # Build and install the nvForest Python package
  ./build.sh nvforest

Other ``build.sh`` options:

.. code-block:: bash

  # Remove any prior build artifacts and configuration (start over)
  ./build.sh clean

  # Build and install libnvforest with verbose output
  ./build.sh libnvforest -v

  # Build and install libnvforest for debug
  ./build.sh libnvforest -g

  # Build and install libnvforest limiting parallel build jobs to 8 (ninja -j8)
  PARALLEL_LEVEL=8 ./build.sh libnvforest

  # Build libnvforest but do not install
  ./build.sh libnvforest -n

  # Use ccache to cache compilations, speeding up subsequent builds
  ./build.sh --ccache

By default, Ninja is used as the cmake generator. To override this and use, e.g., GNU Make, define the ``CMAKE_GENERATOR`` environment variable accordingly:

.. code-block:: bash

  CMAKE_GENERATOR='Unix Makefiles' ./build.sh

To run the C++ unit tests (optional), from the repo root:

.. code-block:: bash

  ctest --test-dir cpp/build

If you want a list of the available C++ tests:

.. code-block:: bash

  ctest -N --test-dir cpp/build

To run all Python tests, from the repo root:

.. code-block:: bash

  pytest -v python/nvforest/tests

If you want a list of the available Python tests:

.. code-block:: bash

  pytest -v python/nvforest/tests --collect-only

Option 2. Manually invoke CMake and build toolchain
---------------------------------------------------

Once dependencies are present, follow the steps below:

1. Clone the repository:

.. code-block:: bash

  git clone https://github.com/rapidsai/nvforest.git

2. Build and install ``libnvforest++`` (C++/CUDA library containing the nvForest algorithms), starting from the repository root folder:

.. code-block:: bash

  mkdir cpp/build
  cmake -B cpp/build -S cpp/ -GNinja

.. note::

  If CUDA is not in your PATH, you may need to set ``CUDA_BIN_PATH`` before running CMake:

  .. code-block:: bash

    export CUDA_BIN_PATH=$CUDA_HOME  # Default: /usr/local/cuda

If using a Conda environment (recommended), configure CMake to install ``libnvforest++`` into the Conda environment:

.. code-block:: bash

  cmake -B cpp/build -S cpp/ -GNinja -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX

.. note::

  You may see the following warning depending on your cmake version and ``CMAKE_INSTALL_PREFIX``. This warning can be safely ignored:

  .. code-block::

    Cannot generate a safe runtime search path for target ml_test because files
    in some directories may conflict with libraries in implicit directories:

  To silence it, add ``-DCMAKE_IGNORE_PATH=$CONDA_PREFIX/lib`` to your ``cmake`` command.

To reduce compile times, you can specify GPU compute capabilities to compile for. For example, for Volta GPUs:

.. code-block:: bash

  cmake -B cpp/build -S cpp/ -GNinja -DCMAKE_CUDA_ARCHITECTURES="70"

Or for multiple architectures (e.g., Ampere and Hopper):

.. code-block:: bash

  cmake -B cpp/build -S cpp/ -GNinja -DCMAKE_CUDA_ARCHITECTURES="80;86;90"

You may also wish to make use of ``ccache`` to reduce build times when switching among branches or between debug and release builds:

.. code-block:: bash

  cmake -B cpp/build -S cpp/ -GNinja -DUSE_CCACHE=ON

There are many options to configure the build process, see the :ref:`custom-build-options` section.

3. Build ``libnvforest++`` and ``libnvforest``:

.. code-block:: bash

  cmake --build cpp/build --target all -v
  cmake --build cpp/build --target install -v

To run tests (optional):

.. code-block:: bash

  ctest --test-dir cpp/build

To build doxygen docs for all C/C++ source files:

.. code-block:: bash

  cmake --build cpp/build --target docs_nvforest

4. Build and install the ``nvforest`` python package.

From the repository root:

.. code-block:: bash

  python -m pip install --no-build-isolation --no-deps \
    --config-settings rapidsai.disable-cuda=true python/nvforest

To run Python tests (optional):

.. code-block:: bash

  pytest -v python/nvforest/tests

If you want a list of the available tests:

.. code-block:: bash

  pytest -v python/nvforest/tests --collect-only

.. _custom-build-options:

Custom Build Options
====================

libnvforest++
-------------

nvForest's cmake has the following configurable flags available:

.. list-table::
   :header-rows: 1
   :widths: 25 20 15 40

   * - Flag
     - Possible Values
     - Default Value
     - Behavior
   * - NVFOREST_ENABLE_GPU
     - [ON, OFF]
     - ON
     - Enable/disable GPU support
   * - BUILD_SHARED_LIBS
     - [ON, OFF]
     - ON
     - Whether to build libnvforest++ as a shared library
   * - BUILD_NVFOREST_TESTS
     - [ON, OFF]
     - ON
     - Enable/disable building nvForest C++ test executables
   * - CUDA_ENABLE_KERNEL_INFO
     - [ON, OFF]
     - OFF
     - Enable/disable kernel resource usage info in nvcc.
   * - CUDA_ENABLE_LINE_INFO
     - [ON, OFF]
     - OFF
     - Enable/disable lineinfo in nvcc.
   * - DETECT_CONDA_ENV
     - [ON, OFF]
     - ON
     - Use detection of conda environment for dependencies. If set to ON, and no value for CMAKE_INSTALL_PREFIX is passed, then it will assign it to $CONDA_PREFIX (to install in the active environment).
   * - DISABLE_DEPRECATION_WARNINGS
     - [ON, OFF]
     - ON
     - Set to ON to disable deprecation warnings
   * - DISABLE_OPENMP
     - [ON, OFF]
     - OFF
     - Set to ON to disable OpenMP
   * - NVTX
     - [ON, OFF]
     - OFF
     - Enable/disable nvtx markers in libnvforest++.
   * - USE_CCACHE
     - [ON, OFF]
     - OFF
     - Whether to cache build artifacts with ccache.
   * - CUDA_STATIC_RUNTIME
     - [ON, OFF]
     - OFF
     - Whether to statically link the CUDA runtime.
   * - NVFOREST_USE_RAFT_STATIC
     - [ON, OFF]
     - OFF
     - Whether to statically link the RAFT library.
   * - NVFOREST_USE_TREELITE_STATIC
     - [ON, OFF]
     - OFF
     - Whether to statically link the Treelite library.
   * - NVFOREST_EXPORT_TREELITE_LINKAGE
     - [ON, OFF]
     - OFF
     - Whether to publicly link Treelite to libnvforest++
   * - CUDA_WARNINGS_AS_ERRORS
     - [ON, OFF]
     - ON
     - Treat all warnings from CUDA as errors
   * - CMAKE_CUDA_ARCHITECTURES
     - List of GPU architectures, semicolon-separated
     - Empty
     - List the GPU architectures to compile the GPU targets for. Set to "NATIVE" to auto detect GPU architecture of the system, set to "ALL" to compile for all RAPIDS supported archs.
