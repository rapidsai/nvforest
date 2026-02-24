########################
Building from the source
########################

Setting up your build environment
=================================

To install cuForest from source, ensure the following dependencies are met:

**Hardware needed to run cuForest.**
cuForest is part of RAPIDS and follows the RAPIDS support matrix.
See https://docs.rapids.ai/platform-support/.
It is possible to build and run cuForest on machines without a GPU; in such machines, cuForest will use the CPU to run inference.

**Software dependencies.**
See https://docs.rapids.ai/platform-support/ for the list of required C++ compilers and Python interpreters.
In addition, cuForest requires Cython 3.0 or later.

.. note:: Building cuForest without GPU support

   It is possible to build cuForest without GPU support; in this case, the CUDA toolkit is not required.
   To build cuForest without GPU, set the CMake option ``CUFOREST_ENABLE_GPU=ON``.

**RAPIDS libraries.**
The cuForest code base is updated in tandem with the rest of RAPIDS. So to build the latest cuForest, you
should use the latest version of RAPIDS as well. (For example, cuForest 26.04 will require 26.04 version of
all RAPIDS packages.)

**Python dependencies.**
For detailed version requirements of runtime dependencies
(numpy, scikit-learn, scipy, joblib, numba, cupy, treelite, etc.),
please see :doc:`supported_versions`.

**For development only.**

* clang-format (= 20.1.4): enforces uniform C++ coding style; required for pre-commit hooks and CI checks. The packages ``clang=20`` and ``clang-tools=20`` from the conda-forge channel should be sufficient, if you are using conda. If not using conda, install the right version using your OS package manager.

.. note:: Use Conda to install all software dependencies

  We highly recommend the use of Conda, a package manager that lets you obtain all necessary
  software dependencies in a virtual environment.
  We provide environment definition files ``conda/environments/all_*.yaml`` containing all software
  dependencies for cuForest.

  To create a development environment named ``cuforest_dev``, use the following commands.

  .. code-block:: console

    $ conda create -n cuforest_dev python=3.13
    $ conda env update -n cuforest_dev \
        --file=conda/environments/all_cuda-131_arch-$(uname -m).yaml
    $ conda activate cuforest_dev

Installing from Source
======================

Option 1. Use the convenience wrapper script (Recommended)
----------------------------------------------------------

As a convenience, a ``build.sh`` script is provided to simplify the build process.
The libraries will be installed to ``$INSTALL_PREFIX`` if set (e.g., ``export INSTALL_PREFIX=/install/path``);
otherwise it will be installed to ``$CONDA_PREFIX``.

.. code-block:: bash

  # Build the cuForest libraries, tests, and python package, then
  # Install them to $INSTALL_PREFIX if set, otherwise $CONDA_PREFIX
  ./build.sh

For workflows that involve frequent switching among branches or between debug and release builds, it is recommended that you install `ccache <https://ccache.dev/>`_ and make use of it by passing the ``--ccache`` flag to ``build.sh``.

To build individual components, specify them as arguments to ``build.sh``:

.. code-block:: bash

  # Build and install the cuForest C++ and C-wrapper libraries
  ./build.sh libcuforest

  # Build and install the cuForest Python package
  ./build.sh cuforest

Other ``build.sh`` options:

.. code-block:: bash

  # Remove any prior build artifacts and configuration (start over)
  ./build.sh clean

  # Build and install libcuforest with verbose output
  ./build.sh libcuforest -v

  # Build and install libcuforest for debug
  ./build.sh libcuforest -g

  # Build and install libcuforest limiting parallel build jobs to 8 (ninja -j8)
  PARALLEL_LEVEL=8 ./build.sh libcuforest

  # Build libcuforest but do not install
  ./build.sh libcuforest -n

  # Use ccache to cache compilations, speeding up subsequent builds
  ./build.sh --ccache

By default, Ninja is used as the cmake generator. To override this and use, e.g., GNU Make, define the ``CMAKE_GENERATOR`` environment variable accordingly:

.. code-block:: bash

  CMAKE_GENERATOR='Unix Makefiles' ./build.sh

To run the C++ unit tests (optional), from the repo root:

.. code-block:: bash

  cd cpp/build
  ctest

If you want a list of the available C++ tests:

.. code-block:: bash

  ctest -N

To run all Python tests, from the repo root:

.. code-block:: bash

  cd python
  pytest -v

If you want a list of the available Python tests:

.. code-block:: bash

  pytest cuforest/tests --collect-only

Option 2. Manually invoke CMake and build toolchain
---------------------------------------------------

Once dependencies are present, follow the steps below:

1. Clone the repository:

.. code-block:: bash

  git clone https://github.com/rapidsai/cuforest.git

2. Build and install ``libcuforest++`` (C++/CUDA library containing the cuForest algorithms), starting from the repository root folder:

.. code-block:: bash

  cd cpp
  mkdir build && cd build
  cmake ..

.. note::

  If CUDA is not in your PATH, you may need to set ``CUDA_BIN_PATH`` before running CMake:

  .. code-block:: bash

    export CUDA_BIN_PATH=$CUDA_HOME  # Default: /usr/local/cuda

If using a Conda environment (recommended), configure cmake for ``libcuforest++``:

.. code-block:: bash

  cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX

.. note::

  You may see the following warning depending on your cmake version and ``CMAKE_INSTALL_PREFIX``. This warning can be safely ignored:

  .. code-block::

    Cannot generate a safe runtime search path for target ml_test because files
    in some directories may conflict with libraries in implicit directories:

  To silence it, add ``-DCMAKE_IGNORE_PATH=$CONDA_PREFIX/lib`` to your ``cmake`` command.

To reduce compile times, you can specify GPU compute capabilities to compile for. For example, for Volta GPUs:

.. code-block:: bash

  cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DCMAKE_CUDA_ARCHITECTURES="70"

Or for multiple architectures (e.g., Ampere and Hopper):

.. code-block:: bash

  cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DCMAKE_CUDA_ARCHITECTURES="80;86;90"

You may also wish to make use of ``ccache`` to reduce build times when switching among branches or between debug and release builds:

.. code-block:: bash

  cmake .. -DUSE_CCACHE=ON

There are many options to configure the build process, see the :ref:`custom-build-options` section.

3. Build ``libcuforest++`` and ``libcuforest``:

.. code-block:: bash

  ninja -j
  ninja install

To run tests (optional):

.. code-block:: bash

  ctest

To build doxygen docs for all C/C++ source files:

.. code-block:: bash

  ninja doc

4. Build and install the ``cuforest`` python package.

From the repository root:

.. code-block:: bash

  python -m pip install --no-build-isolation --no-deps --config-settings rapidsai.disable-cuda=true python/cuforest

To run Python tests (optional):

.. code-block:: bash

  cd python
  pytest -v

If you want a list of the available tests:

.. code-block:: bash

  pytest cuforest/tests --collect-only

.. _custom-build-options:

Custom Build Options
====================

libcuforest++
-------------

cuForest's cmake has the following configurable flags available:

.. list-table::
   :header-rows: 1
   :widths: 25 20 15 40

   * - Flag
     - Possible Values
     - Default Value
     - Behavior
   * - CUFOREST_ENABLE_GPU
     - [ON, OFF]
     - ON
     - Enable/disable GPU support
   * - BUILD_SHARED_LIBS
     - [ON, OFF]
     - ON
     - Whether to build libcuforest++ as a shared library
   * - BUILD_CUFOREST_TESTS
     - [ON, OFF]
     - ON
     - Enable/disable building cuForest C++ test executables
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
   * - DIABLE_DEPRECATION_WARNINGS
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
     - Enable/disable nvtx markers in libcuforest++.
   * - USE_CCACHE
     - [ON, OFF]
     - OFF
     - Whether to cache build artifacts with ccache.
   * - CUDA_STATIC_RUNTIME
     - [ON, OFF]
     - OFF
     - Whether to statically link the CUDA runtime.
   * - CUFOREST_USE_RAFT_STATIC
     - [ON, OFF]
     - OFF
     - Whether to statically link the RAFT library.
   * - CUFOREST_USE_TREELITE_STATIC
     - [ON, OFF]
     - OFF
     - Whether to statically link the Treelite library.
   * - CUFOREST_EXPORT_TREELITE_LINKAGE
     - [ON, OFF]
     - OFF
     - Whether to publicly link Treelite to libcuforest++
   * - CUDA_WARNINGS_AS_ERRORS
     - [ON, OFF]
     - ON
     - Treat all warnings from CUDA as errors
   * - CMAKE_CUDA_ARCHITECTURES
     - List of GPU architectures, semicolon-separated
     - Empty
     - List the GPU architectures to compile the GPU targets for. Set to "NATIVE" to auto detect GPU architecture of the system, set to "ALL" to compile for all RAPIDS supported archs.
