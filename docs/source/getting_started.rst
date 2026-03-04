#############################
Getting started with nvForest
#############################

nvForest with Python
====================

To run inference for decision tree models, first **import** tree models into nvForest.

.. testsetup:: workflow

    import numpy as np
    from sklearn.ensemble import RandomForestClassifier

    X = np.array([[1, 2], [-1, 2]], dtype="float32")
    y = np.array([0, 1], dtype="int32")

    skl_model = RandomForestClassifier(n_estimators=1, max_depth=1)
    skl_model.fit(X, y)

.. testcode:: workflow

    import nvforest

    # model_dir: pathlib.Path object, pointing to a directory containing model files.

    # Load XGBoost JSON model, to run inference on CPU
    fm = nvforest.load_model(model_dir / "xgboost_model.json", device="cpu",
                             model_type="xgboost_json")

    # Load LightGBM model, to run inference on GPU
    fm = nvforest.load_model(model_dir / "lightgbm_model.txt", device="gpu",
                             model_type="lightgbm")

    # Load scikit-learn random forest, to run inference on GPU
    fm = nvforest.load_from_sklearn(skl_model, device="gpu")

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

.. note:: Automatically detecting ``device``

    Setting ``device="auto"`` in :py:meth:`~nvforest.load_model` will load the tree model
    onto GPU memory, if a GPU is available. If no GPU is available, the tree model will be
    loaded to the main memory instead. If no ``device`` parameter is specified, ``device="auto"``
    will be used.

    This feature allows you to use a single script to deploy tree models to heterogeneous array
    of machines, some with NVIDIA GPUs and some without.

.. note:: Selecting among multiple GPUs

    If your system has more than one NVIDIA GPUs, you can select one of them to run tree inference
    by passing ``device_id`` parameter.

    .. code-block:: python

        # Load model to GPU device 1
        fm = nvforest.load_model(model_dir / "xgboost_model.json",
                                 model_type="xgboost_json",
                                 device="gpu", device_id=1)
        fm = nvforest.load_from_sklearn(skl_model, device="gpu", device_id=1)

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

    assert pred.shape == (X.shape[0],)
    assert class_probs.shape == (X.shape[0], fm.num_outputs)
    assert leaf_ids.shape == (X.shape[0], fm.num_trees)
    assert pred_per_tree.shape == (X.shape[0], fm.num_trees, fm.num_outputs)

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

    #include <raft/core/handle.hpp>
    #include <nvforest/detail/raft_proto/handle.hpp>

    raft::handle_t raft_handle{};
    raft_proto::handle_t handle{raft_handle};

    // Assumption:
    // * Both output and input are in the GPU memory.
    // * The input buffer should be of dimension (num_rows, num_features)
    // * The output buffer should be of dimension (num_rows, fm.num_outputs())
    fm.predict(handle, output, input, num_rows,
               raft_proto::device_type::gpu, raft_proto::device_type::gpu,
               nvforest::infer_kind::default_kind);
