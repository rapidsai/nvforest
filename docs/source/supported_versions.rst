Supported Versions
==================

Please see https://docs.rapids.ai/install/ for RAPIDS-wide version support.

We aim to meet the `SPEC 0 guidelines <https://scientific-python.org/specs/spec-0000/>`_ for minimal supported versions.

Required Runtime Dependencies
-----------------------------

The following dependencies are required for the cuForest library:

* **NumPy**: >=1.23,<3.0a0
* **scikit-learn**: >=1.5
* **cupy**: cupy-cuda12x>=13.6.0 (CUDA 12), cupy-cuda13x>=13.6.0 (CUDA 13)
* **treelite**: ==4.6.1

RAPIDS Dependencies
-------------------

cuML dependencies within the RAPIDS ecosystem are pinned to the same version. For example, cuML 26.02 is compatible with and only with cuDF 26.02, cuVS 26.02, and other RAPIDS libraries at version 26.02.
