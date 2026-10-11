====================================================
Supercharge Tree-Based Model Inference with nvForest
====================================================

*Published 2026-09-03*

Tree-based models such as gradient boosted trees and random forests remain
some of the most widely used models in machine learning. For structured data,
they are often more accurate, easier to train, and easier to interpret than
other types of models, such as deep learning models. Libraries like XGBoost,
LightGBM, and scikit-learn have made tree models a default choice for tasks
such as fraud detection, recommendation systems, and risk scoring.

Once a model is trained, inference is where most of its life is spent, often
with high throughput or low latency requirements, and on different hardware
than it was trained on. Getting good inference performance is critical, and
performing inference on CPUs does not always meet those latency or throughput
demands.

We are proud to announce nvForest, a lightweight library for fast inference
for tree-based models on NVIDIA GPUs. nvForest builds on our past work on the
`Forest Inference Library (FIL) <https://developer.nvidia.com/blog/supercharge-tree-based-model-inference-with-forest-inference-library-in-nvidia-cuml/>`_,
which has lived inside cuML until now. With the release of nvForest, this
interface is easier to install, easier to deploy, and useful to a wider range
of applications.

nvForest enables fast inference for a variety of model formats, including
cuML, scikit-learn, XGBoost, LightGBM, and any other Treelite-compatible
model. It provides state-of-the-art performance on tree inference, a small
binary footprint, and a simple API.

In this post, we explore the motivation behind releasing nvForest as a
standalone library, provide updated benchmarks showing the benefits of
performing inference with nvForest, and share code samples to help you get
started running on your own data today.

Why a standalone library
========================

FIL has always been a strong inference engine, but bundling it with cuML
created some friction for the people who relied on it most. Splitting it out
addresses a few specific problems.

First, nvForest is now much lighter to install. Because FIL accelerates models
from scikit-learn, XGBoost, and LightGBM, many of its users work primarily in
those ecosystems. Bundling it with cuML imposed cuML and all of its
dependencies on those users, which was not always possible in constrained
environments. nvForest depends only on what it needs, and has a much smaller
binary size relative to cuML (50 MiB vs 250 MiB).

Additionally, nvForest does not require a GPU to be installed. One of the
features of nvForest is a CPU-only execution mode, and one common usage
pattern is to train a model on GPUs and then deploy it on CPUs. Until now,
that required a machine with a GPU present, even if the GPU was never used
for inference. As a standalone library, nvForest removes that requirement and
provides forest inference out of the box on CPUs as well as NVIDIA GPUs.

Quick-start examples
====================

To get started with nvForest, import the module, load your model, and call
:py:meth:`~nvforest.GPUForestInferenceRegressor.predict`.

Inferencing with XGBoost models
-------------------------------

.. code-block:: python

   import nvforest

   fm = nvforest.load_model("xgboost_model.json", device="gpu")
   y = fm.predict(X)

Inferencing with LightGBM models
--------------------------------

.. code-block:: python

   import nvforest

   fm = nvforest.load_model("lightgbm_model.txt", device="gpu")
   y = fm.predict(X)

Inferencing with scikit-learn random forest models
--------------------------------------------------

.. code-block:: python

   import nvforest

   # skl_model is RandomForestRegressor or RandomForestClassifier
   fm = nvforest.load_from_sklearn(skl_model, device="gpu")
   y = fm.predict(X)

In addition to the simple Python API, nvForest provides a CMake configuration
and a C++ API, to make it easy for C++ applications to provide fast tree
inference. See :ref:`nvforest-with-c-advanced` for more details.

Auto-optimization
=================

nvForest allows users to fine-tune inference performance with two
hyperparameters: ``layout`` and ``default_chunk_size``. It is difficult to
predict what the optimal values will be for any given model, so it is often
necessary to determine them empirically. nvForest significantly simplifies
this process with a built-in method for auto-optimization at any given batch
size:

.. code-block:: python

   fm_optimized = fm.optimize(batch_size=1000)

Here, the batch size is a typical size of each inference request. The
``default_chunk_size`` parameter controls the unit of work for each inference
kernel launch and is typically no more than 32. As part of the optimization,
we need to find the best size for ``default_chunk_size`` so that the given
batch is optimally partitioned between inference kernel launches.

:py:meth:`~nvforest.GPUForestInferenceRegressor.optimize` returns a new model
instance. Subsequent prediction calls on that instance use the optimal
performance hyperparameters found for the indicated batch size. You can also
check what hyperparameters were selected by looking at the attributes:

.. code-block:: python

   print(fm_optimized.layout)
   print(fm_optimized.default_chunk_size)

Performance benchmark
=====================

In order to obtain the most complete understanding of nvForest's performance
characteristics, we performed an exhaustive sweep across a broad range of
each of these variables, as summarized in the following table:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Variable
     - Values
   * - Maximum tree depth
     - 2, 4, 8, 16, 32
   * - Tree count
     - 16, 128, 1024
   * - Feature count
     - 8, 32, 128, 512
   * - Batch size
     - 1, 16, 128, 1024, 1048576

We trained scikit-learn ``RandomForestRegressor`` models with every
combination of maximum tree depth, tree count, and feature count from the
above table using synthetically generated data. We then measured inference
performance across each of the batch sizes above.

Runtime performance for inference was assessed using synthetically generated
input batches that were drawn from the same distribution as the training
data. We used ``RandomForestRegressor.predict`` as the baseline.

To summarize the performance over a variety of model configurations, we plot
the 0%, 25%, 50%, 75% and 100% percentile of speedup values for each subset
of configurations. The box plot labeled "all-avg" shows the data for all
configurations.

.. figure:: /_static/blog/speedup-boxplots-1.png
   :alt: Speedup factor by tree depth for batch size 1 and 1048576
   :width: 100%

   Speedup factor by tree depth.

.. figure:: /_static/blog/speedup-boxplots-2.png
   :alt: Speedup factor by feature count for batch size 1 and 1048576
   :width: 100%

   Speedup factor by feature count.

.. figure:: /_static/blog/speedup-boxplots-3.png
   :alt: Speedup factor by tree count for batch size 1 and 1048576
   :width: 100%

   Speedup factor by tree count.

A single NVIDIA H100 (80GB HBM3) was used for GPU results, and a 2-socket
Intel Xeon Platinum 8480CL machine was used for CPU results.

In general, nvForest outperforms scikit-learn by 2-3 orders of magnitude for
inference latency and throughput. The difference in performance is greatest
for shallow tree models (tree depth 8 or less) and models with a high number
of trees (1024).

Migrating from FIL
==================

If you are using FIL today through ``cuml.fil.ForestInference``, you do not
need to rewrite your code immediately. Going forward,
``cuml.fil.ForestInference`` will become a lightweight shim that uses
nvForest underneath and emits a deprecation warning pointing you to nvForest.
Starting in the 26.10 release, it will raise an error directing you to the
standalone library. We recommend moving to nvForest directly prior to this
release; for more details, see the :doc:`../fil_migration`.

Conclusion
==========

nvForest is available today. You can install it by following instructions in
the `RAPIDS installation guide <https://docs.rapids.ai/install/>`_.

To get started, head to :doc:`../getting_started`, try the examples above,
and get started performing lightning fast inference on your own tree models!
