#!/usr/bin/env python
#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

"""
Comprehensive benchmark comparing cuforest against native ML framework inference.

Supports sklearn, XGBoost, and LightGBM models with both regressor and classifier
variants on CPU and GPU devices.

Usage:
    python -m cuforest.benchmark.benchmark run
    python -m cuforest.benchmark.benchmark run --framework sklearn --framework xgboost
    python -m cuforest.benchmark.benchmark run --dry-run
    python -m cuforest.benchmark.benchmark run --quick-test
"""

import gc
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from time import perf_counter
from typing import Any, Callable, Optional

import click
import numpy as np
import pandas as pd
import treelite

import cuforest

# Conditional imports for ML frameworks
try:
    from sklearn.datasets import make_classification, make_regression
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


# Constants
DEFAULT_WARMUP_CYCLES = 3
DEFAULT_BENCHMARK_CYCLES = 5
MAX_BATCH_SIZE = 16_777_216
TRAIN_SAMPLES = 10_000
RANDOM_STATE = 0

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Parameter spaces
FULL_VALUES = {
    "num_features": [8, 32, 128, 512],
    "max_depth": [2, 4, 8, 16, 32],
    "num_trees": [16, 128, 1024],
    "batch_size": [1, 16, 128, 1024, 1_048_576, MAX_BATCH_SIZE],
}

QUICK_TEST_VALUES = {
    "num_features": [32],
    "max_depth": [4],
    "num_trees": [16],
    "batch_size": [1024],
}


def get_logger():
    """Configure and return the benchmark logger."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


LOGGER = get_logger()


@dataclass
class FrameworkConfig:
    """Configuration for a ML framework."""

    name: str
    regressor_class: Optional[type]
    classifier_class: Optional[type]
    save_model: Callable[[Any, str], None]
    load_native: Callable[[str], Any]
    predict_native: Callable[[Any, np.ndarray], np.ndarray]
    model_extension: str
    supports_gpu_native: bool = False


def _save_sklearn_model(model: Any, path: str) -> None:
    """Save sklearn model as treelite checkpoint."""
    treelite.sklearn.import_model(model).serialize(path)


def _load_sklearn_model(path: str) -> Any:
    """Load sklearn model - returns None as we use the trained model directly."""
    return None


def _predict_sklearn(model: Any, X: np.ndarray) -> np.ndarray:
    """Run sklearn prediction."""
    return model.predict(X)


def _save_xgboost_model(model: Any, path: str) -> None:
    """Save XGBoost model in UBJSON format."""
    booster = model.get_booster() if hasattr(model, "get_booster") else model
    booster.save_model(path)


def _load_xgboost_model(path: str) -> Any:
    """Load XGBoost booster from file."""
    booster = xgb.Booster()
    booster.load_model(path)
    return booster


def _predict_xgboost(model: Any, X: np.ndarray) -> np.ndarray:
    """Run XGBoost prediction."""
    if hasattr(model, "predict"):
        if isinstance(model, xgb.Booster):
            dmatrix = xgb.DMatrix(X)
            return model.predict(dmatrix)
        return model.predict(X)
    return model.predict(xgb.DMatrix(X))


def _save_lightgbm_model(model: Any, path: str) -> None:
    """Save LightGBM model in text format."""
    booster = model.booster_ if hasattr(model, "booster_") else model
    booster.save_model(path)


def _load_lightgbm_model(path: str) -> Any:
    """Load LightGBM booster from file."""
    return lgb.Booster(model_file=path)


def _predict_lightgbm(model: Any, X: np.ndarray) -> np.ndarray:
    """Run LightGBM prediction."""
    if hasattr(model, "predict"):
        return model.predict(X)
    return model.predict(X)


# Framework configurations
FRAMEWORKS: dict[str, FrameworkConfig] = {}

if SKLEARN_AVAILABLE:
    FRAMEWORKS["sklearn"] = FrameworkConfig(
        name="sklearn",
        regressor_class=RandomForestRegressor,
        classifier_class=RandomForestClassifier,
        save_model=_save_sklearn_model,
        load_native=_load_sklearn_model,
        predict_native=_predict_sklearn,
        model_extension=".tl",
        supports_gpu_native=False,
    )

if XGBOOST_AVAILABLE:
    FRAMEWORKS["xgboost"] = FrameworkConfig(
        name="xgboost",
        regressor_class=xgb.XGBRegressor,
        classifier_class=xgb.XGBClassifier,
        save_model=_save_xgboost_model,
        load_native=_load_xgboost_model,
        predict_native=_predict_xgboost,
        model_extension=".ubj",
        supports_gpu_native=True,
    )

if LIGHTGBM_AVAILABLE:
    FRAMEWORKS["lightgbm"] = FrameworkConfig(
        name="lightgbm",
        regressor_class=lgb.LGBMRegressor,
        classifier_class=lgb.LGBMClassifier,
        save_model=_save_lightgbm_model,
        load_native=_load_lightgbm_model,
        predict_native=_predict_lightgbm,
        model_extension=".txt",
        supports_gpu_native=True,
    )


def run_inference_benchmark(
    predict_fn: Callable,
    X: np.ndarray,
    batch_size: int,
    warmup_cycles: int = DEFAULT_WARMUP_CYCLES,
    benchmark_cycles: int = DEFAULT_BENCHMARK_CYCLES,
) -> float:
    """
    Run inference benchmark and return minimum elapsed time.

    Parameters
    ----------
    predict_fn : Callable
        Function that takes a batch and returns predictions.
    X : np.ndarray
        Full dataset to sample batches from.
    batch_size : int
        Size of each batch.
    warmup_cycles : int
        Number of warmup iterations.
    benchmark_cycles : int
        Number of timed iterations.

    Returns
    -------
    float
        Minimum elapsed time across benchmark cycles.
    """
    available_batch_count = X.shape[0] // batch_size

    # Warmup
    for cycle in range(warmup_cycles):
        batch_index = cycle % available_batch_count
        batch = X[batch_index * batch_size : (batch_index + 1) * batch_size]
        predict_fn(batch)

    # Benchmark
    elapsed = float("inf")
    for cycle in range(warmup_cycles, warmup_cycles + benchmark_cycles):
        batch_index = cycle % available_batch_count
        batch = X[batch_index * batch_size : (batch_index + 1) * batch_size]
        begin = perf_counter()
        predict_fn(batch)
        end = perf_counter()
        elapsed = min(elapsed, end - begin)

    # Log throughput
    throughput = batch_size / elapsed
    throughput_gb = throughput * X.shape[1] * X.dtype.itemsize / 1e9
    if throughput_gb >= 0.01:
        LOGGER.info(f"    Throughput: {throughput_gb:.2f} GB/s")
    elif throughput_gb * 1000 >= 0.01:
        LOGGER.info(f"    Throughput: {throughput_gb * 1000:.2f} MB/s")
    else:
        LOGGER.info(f"    Throughput: {throughput_gb * 1e6:.2f} KB/s")

    return elapsed


def write_checkpoint(results: dict, data_dir: str, final: bool = False) -> None:
    """Write results to checkpoint file."""
    if not hasattr(write_checkpoint, "counter"):
        write_checkpoint.counter = 0

    MAX_CHECKPOINTS = 6
    filename = "final_results.csv" if final else f"checkpoint_{write_checkpoint.counter}.csv"

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(data_dir, filename), index=False)

    write_checkpoint.counter = (write_checkpoint.counter + 1) % MAX_CHECKPOINTS


def train_model(
    framework: FrameworkConfig,
    model_type: str,
    num_trees: int,
    max_depth: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Any:
    """Train a model with the given framework and parameters."""
    model_class = (
        framework.regressor_class if model_type == "regressor" else framework.classifier_class
    )

    if framework.name == "sklearn":
        model = model_class(n_estimators=num_trees, max_depth=max_depth, n_jobs=-1)
    elif framework.name == "xgboost":
        model = model_class(
            n_estimators=num_trees,
            max_depth=max_depth,
            tree_method="hist",
            n_jobs=-1,
        )
    elif framework.name == "lightgbm":
        model = model_class(
            n_estimators=num_trees,
            max_depth=max_depth,
            n_jobs=-1,
            verbose=-1,
        )
    else:
        raise ValueError(f"Unknown framework: {framework.name}")

    model.fit(X_train, y_train)
    return model


def generate_data(
    num_features: int,
    num_samples: int,
    model_type: str,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data for benchmarking."""
    if model_type == "regressor":
        X, y = make_regression(
            n_samples=num_samples,
            n_features=num_features,
            random_state=random_state,
        )
    else:
        X, y = make_classification(
            n_samples=num_samples,
            n_features=num_features,
            n_informative=min(num_features, 10),
            n_redundant=0,
            random_state=random_state,
        )

    return X.astype("float32"), y.astype("float32" if model_type == "regressor" else "int32")


def print_dry_run_info(
    frameworks: list[str],
    model_types: list[str],
    devices: list[str],
    param_values: dict,
) -> None:
    """Print benchmark configuration for dry run."""
    print("\nBenchmark Configuration:")
    print(f"  Frameworks: {', '.join(frameworks)}")
    print(f"  Model types: {', '.join(model_types)}")
    print(f"  Devices: {', '.join(devices)}")
    print("\nParameter space:")
    for key, values in param_values.items():
        print(f"  {key}: {values}")

    # Calculate total runs
    total_params = (
        len(param_values["num_features"])
        * len(param_values["max_depth"])
        * len(param_values["num_trees"])
        * len(param_values["batch_size"])
    )
    total_runs = len(frameworks) * len(model_types) * len(devices) * total_params

    print(f"\nTotal benchmark configurations: {total_runs}")
    print(
        f"  = {len(frameworks)} frameworks x {len(model_types)} model_types x "
        f"{len(devices)} devices x {total_params} parameter combinations"
    )


def run_benchmark_suite(
    frameworks: list[str],
    model_types: list[str],
    devices: list[str],
    param_values: dict,
    data_dir: str,
) -> None:
    """
    Run the full benchmark suite.

    Parameters
    ----------
    frameworks : list[str]
        List of framework names to benchmark.
    model_types : list[str]
        List of model types ('regressor', 'classifier').
    devices : list[str]
        List of devices ('cpu', 'gpu').
    param_values : dict
        Dictionary of parameter names to lists of values.
    data_dir : str
        Directory to save results.
    """
    os.makedirs(data_dir, exist_ok=True)

    results = {
        "framework": [],
        "model_type": [],
        "device": [],
        "num_features": [],
        "max_depth": [],
        "num_trees": [],
        "batch_size": [],
        "native_time": [],
        "cuforest_time": [],
        "optimal_layout": [],
        "optimal_chunk_size": [],
        "speedup": [],
    }

    for model_type in model_types:
        for num_features in param_values["num_features"]:
            # Generate data once per feature count
            X, y = generate_data(num_features, MAX_BATCH_SIZE, model_type)
            X_train, y_train = X[:TRAIN_SAMPLES], y[:TRAIN_SAMPLES]

            for max_depth in param_values["max_depth"]:
                for num_trees in param_values["num_trees"]:
                    for framework_name in frameworks:
                        framework = FRAMEWORKS[framework_name]

                        # Train model
                        cur_time = datetime.now().strftime("%H:%M:%S")
                        LOGGER.info(
                            f"{cur_time}: Training {framework_name} {model_type} "
                            f"({num_trees} trees, depth {max_depth}, {num_features} features)"
                        )

                        try:
                            model = train_model(
                                framework, model_type, num_trees, max_depth, X_train, y_train
                            )
                        except Exception as e:
                            LOGGER.warning(f"Failed to train {framework_name} model: {e}")
                            continue

                        # Save model
                        model_path = os.path.join(
                            data_dir, f"model_{framework_name}{framework.model_extension}"
                        )
                        framework.save_model(model, model_path)

                        for device in devices:
                            # Skip GPU native for frameworks that don't support it
                            if device == "gpu" and not framework.supports_gpu_native:
                                native_model = model  # Use sklearn model on CPU for native
                            else:
                                native_model = (
                                    model
                                    if framework.name == "sklearn"
                                    else framework.load_native(model_path)
                                )

                            for batch_size in param_values["batch_size"]:
                                # Skip large batch + large feature combinations
                                if batch_size == MAX_BATCH_SIZE and num_features == 512:
                                    continue

                                LOGGER.info(
                                    f"  {framework_name}/{model_type}/{device} "
                                    f"batch_size={batch_size}"
                                )

                                # Prepare data for device
                                if device == "gpu":
                                    try:
                                        import cupy as cp

                                        X_device = cp.asarray(X)
                                    except ImportError:
                                        LOGGER.warning("cupy not available, skipping GPU benchmark")
                                        continue
                                else:
                                    X_device = X

                                # Native benchmark
                                LOGGER.info("    Running native inference...")
                                try:
                                    native_time = run_inference_benchmark(
                                        lambda batch: framework.predict_native(native_model, batch),
                                        X if device == "cpu" else X,  # Native uses CPU data
                                        batch_size,
                                    )
                                except Exception as e:
                                    LOGGER.warning(f"Native inference failed: {e}")
                                    native_time = float("nan")

                                # cuforest benchmark
                                LOGGER.info("    Running cuforest inference...")
                                try:
                                    cuforest_model = cuforest.load_model(
                                        model_path,
                                        device=device,
                                        precision="single",
                                    )

                                    # Optimize
                                    batch = (
                                        X_device[:batch_size]
                                        if device == "gpu"
                                        else X[:batch_size]
                                    )
                                    cuforest_model = cuforest_model.optimize(data=batch)
                                    optimal_layout = cuforest_model.layout
                                    optimal_chunk_size = cuforest_model.default_chunk_size

                                    cuforest_time = run_inference_benchmark(
                                        lambda batch: cuforest_model.predict(batch),
                                        X_device if device == "gpu" else X,
                                        batch_size,
                                    )
                                except Exception as e:
                                    LOGGER.warning(f"cuforest inference failed: {e}")
                                    cuforest_time = float("nan")
                                    optimal_layout = None
                                    optimal_chunk_size = None

                                # Record results
                                results["framework"].append(framework_name)
                                results["model_type"].append(model_type)
                                results["device"].append(device)
                                results["num_features"].append(num_features)
                                results["max_depth"].append(max_depth)
                                results["num_trees"].append(num_trees)
                                results["batch_size"].append(batch_size)
                                results["native_time"].append(native_time)
                                results["cuforest_time"].append(cuforest_time)
                                results["optimal_layout"].append(optimal_layout)
                                results["optimal_chunk_size"].append(optimal_chunk_size)
                                results["speedup"].append(
                                    native_time / cuforest_time
                                    if cuforest_time and not np.isnan(cuforest_time)
                                    else float("nan")
                                )

                                # Clean up GPU memory
                                if device == "gpu":
                                    del cuforest_model
                                    gc.collect()

                        # Clean up trained model
                        del model
                        if native_model is not model:
                            del native_model
                        gc.collect()

                    # Write checkpoint after each tree/depth combination
                    write_checkpoint(results, data_dir)

            # Clean up data
            del X, y, X_train, y_train
            gc.collect()

    # Write final results
    write_checkpoint(results, data_dir, final=True)
    LOGGER.info(f"Benchmark complete. Results saved to {data_dir}/final_results.csv")


@click.group()
def cli():
    """Benchmark suite for cuforest."""
    pass


@cli.command()
@click.option(
    "--framework",
    "-f",
    "frameworks",
    multiple=True,
    type=click.Choice(["sklearn", "xgboost", "lightgbm"]),
    help="Framework(s) to benchmark. Can be specified multiple times. Default: all available.",
)
@click.option(
    "--dry-run",
    "-n",
    is_flag=True,
    help="Print benchmark configuration without running.",
)
@click.option(
    "--quick-test",
    "-q",
    is_flag=True,
    help="Run with minimal parameters to verify setup.",
)
@click.option(
    "--device",
    "-d",
    "devices",
    multiple=True,
    type=click.Choice(["cpu", "gpu", "both"]),
    default=["both"],
    help="Device(s) to benchmark on. Default: both.",
)
@click.option(
    "--model-type",
    "-m",
    "model_types",
    multiple=True,
    type=click.Choice(["regressor", "classifier", "both"]),
    default=["both"],
    help="Model type(s) to benchmark. Default: both.",
)
@click.option(
    "--output-dir",
    "-o",
    default=None,
    help="Output directory for results. Default: benchmark/data/",
)
def run(
    frameworks: tuple[str, ...],
    dry_run: bool,
    quick_test: bool,
    devices: tuple[str, ...],
    model_types: tuple[str, ...],
    output_dir: Optional[str],
):
    """Run the benchmark suite."""
    # Resolve frameworks
    if not frameworks:
        frameworks = tuple(FRAMEWORKS.keys())
    else:
        # Validate frameworks are available
        unavailable = [f for f in frameworks if f not in FRAMEWORKS]
        if unavailable:
            raise click.ClickException(
                f"Framework(s) not available: {', '.join(unavailable)}. "
                f"Install the required packages."
            )

    # Resolve devices
    device_list = []
    for d in devices:
        if d == "both":
            device_list.extend(["cpu", "gpu"])
        else:
            device_list.append(d)
    device_list = list(dict.fromkeys(device_list))  # Remove duplicates, preserve order

    # Resolve model types
    model_type_list = []
    for m in model_types:
        if m == "both":
            model_type_list.extend(["regressor", "classifier"])
        else:
            model_type_list.append(m)
    model_type_list = list(dict.fromkeys(model_type_list))

    # Select parameter space
    param_values = QUICK_TEST_VALUES if quick_test else FULL_VALUES

    # Resolve output directory
    if output_dir is None:
        output_dir = DATA_DIR

    if dry_run:
        print_dry_run_info(list(frameworks), model_type_list, device_list, param_values)
        return

    # Check for available frameworks
    LOGGER.info(f"Available frameworks: {', '.join(FRAMEWORKS.keys())}")
    LOGGER.info(f"Selected frameworks: {', '.join(frameworks)}")
    LOGGER.info(f"Devices: {', '.join(device_list)}")
    LOGGER.info(f"Model types: {', '.join(model_type_list)}")
    LOGGER.info(f"Quick test: {quick_test}")
    LOGGER.info(f"Output directory: {output_dir}")

    run_benchmark_suite(
        frameworks=list(frameworks),
        model_types=model_type_list,
        devices=device_list,
        param_values=param_values,
        data_dir=output_dir,
    )


if __name__ == "__main__":
    cli()
