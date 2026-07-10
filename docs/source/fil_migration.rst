###############
Migration guide
###############

Basic workflow
==============

.. testsetup:: workflow

    import numpy as np
    from sklearn.ensemble import RandomForestClassifier

    X = np.array([[1, 2], [-1, 2]], dtype="float32")
    y = np.array([0, 1], dtype="int32")

    skl_model = RandomForestClassifier(n_estimators=1, max_depth=1)
    skl_model.fit(X, y)

Call :py:meth:`~nvforest.load_model`, :py:meth:`~nvforest.load_from_sklearn`,
or :py:meth:`~nvforest.load_from_treelite_model`. Note that it is no longer
necessary to specify the ``is_classifier`` parameter.

.. code-block:: python

    # BEFORE
    import cuml
    fil_model = cuml.fil.ForestInference.load_from_sklearn(skl_model, is_classifier=True)
    fil_model.optimize(batch_size=1024)
    predictions = fil_model.predict(X)
    probabilities = fil_model.predict_proba(X)
    per_tree_pred = fil_model.predict_per_tree(X)
    lead_ids = fil_model.apply(X)

.. testcode:: workflow

    # AFTER
    import nvforest
    nvforest_model = nvforest.load_from_sklearn(skl_model)
    nvforest_model = nvforest_model.optimize(batch_size=1024)
    predictions = nvforest_model.predict(X)
    probabilities = nvforest_model.predict_proba(X)
    per_tree_pred = nvforest_model.predict_per_tree(X)
    lead_ids = nvforest_model.apply(X)

Device selection
================
Specify the ``device`` parameter when calling :py:meth:`~nvforest.load_model`.

.. code-block:: python

    # BEFORE
    with cuml.fil.set_fil_device_type("cpu"):
        fil_model = cuml.fil.ForestInference.load_from_sklearn(skl_model)

.. testcode:: workflow

    # AFTER
    nvforest_model = nvforest.load_from_sklearn(skl_model, device="cpu")

nvForest also differs from FIL when it comes to the behavior when no device is explicitly
specified. The ``device`` parameter defaults to ``"auto"``. nvForest will attempt to
load the tree model onto a GPU device, if one is available. If no GPU is available,
nvForest will fall back to the CPU.
