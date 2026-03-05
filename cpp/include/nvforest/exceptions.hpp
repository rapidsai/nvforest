/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <exception>
#include <string>

namespace nvforest {

/** Exception indicating model is incompatible with nvForest */
struct unusable_model_exception : std::exception {
  unusable_model_exception() : msg_{"Model is not compatible with nvForest"} {}
  unusable_model_exception(std::string msg) : msg_{msg} {}
  unusable_model_exception(char const* msg) : msg_{msg} {}
  virtual char const* what() const noexcept { return msg_.c_str(); }

 private:
  std::string msg_;
};

/** Exception indicating model import failed */
struct model_import_error : std::exception {
  model_import_error() : model_import_error("Error while importing model") {}
  model_import_error(char const* msg) : msg_{msg} {}
  virtual char const* what() const noexcept { return msg_; }

 private:
  char const* msg_;
};

/**
 * Exception indicating a mismatch between the type of input data and the
 * model
 *
 * This typically occurs when doubles are provided as input to a model with
 * float thresholds or vice versa.
 */
struct type_error : std::exception {
  type_error() : type_error("Model cannot be used with given data type") {}
  type_error(char const* msg) : msg_{msg} {}
  virtual char const* what() const noexcept { return msg_; }

 private:
  char const* msg_;
};

/**
 * Exception that occurred while traversing a given tree model
 */
struct traversal_exception : std::exception {
  traversal_exception() : msg_{"Error encountered while traversing forest"} {}
  traversal_exception(std::string msg) : msg_{msg} {}
  traversal_exception(char const* msg) : msg_{msg} {}
  virtual char const* what() const noexcept { return msg_.c_str(); }

 private:
  std::string msg_;
};

/**
 * Exception indicating a runtime error.
 */
struct runtime_error : std::exception {
  runtime_error() : runtime_error("Runtime error") {}
  runtime_error(char const* msg) : msg_{msg} {}
  runtime_error(std::string const& msg) : msg_{msg} {}
  virtual char const* what() const noexcept { return msg_.c_str(); }

 private:
  std::string msg_;
};

}  // namespace nvforest
