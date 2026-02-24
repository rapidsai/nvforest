#######
C++ API
#######

Model importer utilities
========================

.. doxygenfunction:: cuforest::import_from_treelite_model

.. doxygenfunction:: cuforest::import_from_treelite_handle

.. doxygenstruct:: cuforest::treelite_importer
   :members:

Forest classes
==============

.. doxygenstruct:: cuforest::forest_model
   :members:

.. doxygenstruct:: cuforest::decision_forest
   :members:

Enums and constants
===================

.. doxygenvariable:: cuforest::preferred_tree_layout

.. doxygenenum:: cuforest::tree_layout

.. doxygenenum:: cuforest::infer_kind

.. doxygenenum:: cuforest::row_op

.. doxygenenum:: cuforest::element_op

Type aliases
============

.. doxygentypedef:: cuforest::decision_forest_variant

.. doxygentypedef:: cuforest::detail::preset_decision_forest
