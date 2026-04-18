#############################
Getting started with nvForest
#############################

nvForest with Python
====================

.. testsetup:: workflow

    import numpy as np
    from sklearn.ensemble import RandomForestClassifier

    X = np.array([[1, 2], [-1, 2]], dtype="float32")
    y = np.array([0, 1], dtype="int32")

    skl_model = RandomForestClassifier(n_estimators=1, max_depth=1)
    skl_model.fit(X, y)

First example
-------------

To run inference for decision tree models, it takes only two lines of code:

.. testcode:: workflow

    import nvforest

    # model_dir: pathlib.Path object, pointing to a directory containing model files.
    fm = nvforest.load_model(model_dir / "xgboost_model.json")
    y = fm.predict(X)

Let us now look into all available options for nvForest.

Model import and device selection
---------------------------------

With nvForest, you can run tree models using CPUs and NVIDIA GPUs.
You may explicitly select which device to use by specifying the ``device`` parameter
in :py:meth:`~nvforest.load_model`. If no ``device`` is given, it is set to ``"auto"``.
(See the note below for the behavior of ``"auto"``.)

.. testcode:: workflow

    # Load XGBoost JSON model, to run inference on CPU
    fm = nvforest.load_model(model_dir / "xgboost_model.json", device="cpu")

    # Load LightGBM model, to run inference on GPU
    fm = nvforest.load_model(model_dir / "lightgbm_model.txt", device="gpu")

    # Load scikit-learn random forest, to run inference on GPU
    fm = nvforest.load_from_sklearn(skl_model, device="gpu")

.. note:: Automatically detecting ``device``

    Setting ``device="auto"`` in :py:meth:`~nvforest.load_model` will load the tree model
    onto GPU memory, if a GPU is available. If no GPU is available, the tree model will be
    loaded to the main memory instead. If no ``device`` parameter is specified, ``device="auto"``
    will be used.

    This feature allows you to use a single script to deploy tree models to heterogeneous array
    of machines, some with NVIDIA GPUs and some without.

.. note:: Automatically detecting ``model_type``

    By default, nvForest will attempt to detect the type of the model file using the file
    extension:

    * ``.json``: XGBoost JSON
    * ``.ubj``: XGBoost UBJSON
    * ``.txt``: LightGBM
    * Other: Treelite checkpoint (produced with :py:meth:`treelite.Model.serialize`)

    In cases where nvForest fails to detect the right model type, you may want to
    specify the ``model_type`` explicitly:

    .. testcode:: workflow

        fm = nvforest.load_model(model_dir / "lightgbm_model.txt", device="gpu",
                                 model_type="lightgbm")

The ``fm`` object will be one of the following types:

* :py:class:`~nvforest.GPUForestInferenceClassifier`: a classification model, to run on GPU.
  The model will reside in the GPU memory.
* :py:class:`~nvforest.GPUForestInferenceRegressor`: a regression model, to run on GPU.
  The model will reside in the GPU memory.
* :py:class:`~nvforest.CPUForestInferenceClassifier`: a classification model, to run on CPU.
  The model object will reside in the main memory.
* :py:class:`~nvforest.CPUForestInferenceRegressor`: a regression model, to run on CPU.
  The model object will reside in the main memory.

You can inspect the type of the model by printing its type:

.. doctest:: workflow

    >>> print(type(fm).__name__)
    GPUForestInferenceClassifier

.. note:: Selecting among multiple GPUs

    If your system has more than one NVIDIA GPU, you can select one of them to run tree inference
    by passing ``device_id`` parameter.

    .. code-block:: python

        # Load model to GPU device 1
        fm = nvforest.load_model(model_dir / "xgboost_model.json",
                                 device="gpu", device_id=1)
        fm = nvforest.load_from_sklearn(skl_model, device="gpu", device_id=1)

    Each model object is associated with a single device. Use the ``device_id`` property to look up
    which device the model object is located on.

Running inference
-----------------

After importing the model, run inference using :py:meth:`~nvforest.GPUForestInferenceRegressor.predict`
or its variants.

.. testcode:: workflow

    # Run inference
    pred = fm.predict(X)

    # Run inference and output class probabilities
    # Only applicable for classification models
    class_probs = fm.predict_proba(X)

    # Run inference and obtain leaf indices in each decision tree
    leaf_ids = fm.apply(X)

    # Run inference and obtain prediction per individual tree
    pred_per_tree = fm.predict_per_tree(X)

.. testcode:: workflow
    :hide:

    assert pred.shape == (X.shape[0], 1)
    assert class_probs.shape == (X.shape[0], fm.num_outputs)
    assert leaf_ids.shape == (X.shape[0], fm.num_trees)
    assert pred_per_tree.shape == (X.shape[0], fm.num_trees)

nvForest with C++ (Advanced)
============================

Integrating your C++ application with nvForest
----------------------------------------------
nvForest provides a CMake config file so that other C++ projects can find and use it.

.. code-block:: cmake

    find_package(nvforest CONFIG REQUIRED)

    target_link_libraries(my_target PRIVATE nvforest::nvforest++ treelite::treelite)

To ensure that CMake can locate nvForest and Treelite, we recommend the use of Conda to install nvForest.

How to load tree models and run inference
-----------------------------------------

To import tree models into nvForest, first load the tree models as a Treelite model object.

.. code-block:: cpp

    #include <treelite/model_loader.h>
    #include <treelite/tree.h>
    #include <memory>

    std::unique_ptr<treelite::Model> treelite_model
        = treelite::model_loader::LoadXGBoostModelUBJSON(
            "/path/to/xgboost_model.ubj", "{}");

Refer to `the Treelite documentation <https://treelite.readthedocs.io/en/latest/>`_ for
the full list of model loader utilities.

Once the tree model is available as a Treelite object, pass it to the
:cpp:func:`~nvforest::import_from_treelite_model` to load it into nvForest.

.. code-block:: cpp

    #include <nvforest/constants.hpp>
    #include <nvforest/treelite_importer.hpp>
    #include <nvforest/detail/index_type.hpp>
    #include <nvforest/detail/raft_proto/device_type.hpp>
    #include <optional>

    auto fm = nvforest::import_from_treelite_model(
        *treelite_model,
        nvforest::preferred_tree_layout,
        nvforest::index_type{},
        std::nullopt,
        raft_proto::device_type::gpu);

Now that the tree model is fully imported into nvForest, let's run inference:

.. code-block:: cpp

    // Assumption:
    // * Both output and input are in the GPU memory.
    // * The input buffer should be of dimension (num_rows, num_features)
    // * The output buffer should be of dimension (num_rows, fm.num_outputs())
    fm.predict(output, input, num_rows,
               raft_proto::device_type::gpu, raft_proto::device_type::gpu,
               nvforest::infer_kind::default_kind);

.. note:: Reuse the resource handle to reduce overhead

    nvForest internally creates a resource handle (``raft::device_resources``)
    to manage GPU resources. Creation of the resource handle adds a slight
    performance overhead. If you plan to call :cpp:func:`nvforest::predict`
    multiple times, consider creating the resource handle explicitly
    and re-using the handle between the function calls.

    .. code-block:: cpp

        #include <raft/core/device_resources.hpp>

        auto resource = raft::device_resources{};
        fm.predict(resource, output, input, num_rows,
                   raft_proto::device_type::gpu, raft_proto::device_type::gpu,
                   nvforest::infer_kind::default_kind);
