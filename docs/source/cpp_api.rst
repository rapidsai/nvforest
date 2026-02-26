~~~~~~~~~~~~~~~~~~~~~
C++ API Documentation
~~~~~~~~~~~~~~~~~~~~~

Model importer utilities
========================

.. doxygenfunction:: nvforest::import_from_treelite_model

.. doxygenfunction:: nvforest::import_from_treelite_handle

.. doxygenstruct:: nvforest::treelite_importer
   :members:

Forest classes
==============

.. doxygenstruct:: nvforest::forest_model
   :members:

.. doxygenstruct:: nvforest::decision_forest
   :members:

Enums and constants
===================

.. doxygenvariable:: nvforest::preferred_tree_layout

.. doxygenenum:: nvforest::tree_layout

.. doxygenenum:: nvforest::infer_kind

.. doxygenenum:: nvforest::row_op

.. doxygenenum:: nvforest::element_op

Type aliases
============

.. doxygentypedef:: nvforest::decision_forest_variant

.. doxygentypedef:: nvforest::detail::preset_decision_forest
