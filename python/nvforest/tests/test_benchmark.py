#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import os
import tempfile

import numpy as np
import pytest
from click.testing import CliRunner

# Import benchmark components
from nvforest.benchmark.benchmark import (
    FRAMEWORKS,
    FULL_VALUES,
    QUICK_TEST_VALUES,
    FrameworkConfig,
    cli,
    generate_data,
    print_dry_run_info,
    run_inference_benchmark,
    train_model,
    write_checkpoint,
)


class TestFrameworkConfig:
    """Tests for FrameworkConfig dataclass."""

    @pytest.mark.unit
    def test_sklearn_config_exists(self):
        """Test that sklearn framework config is available."""
        assert "sklearn" in FRAMEWORKS
        config = FRAMEWORKS["sklearn"]
        assert config.name == "sklearn"
        assert config.model_extension == ".tl"
        assert config.supports_gpu_native is False

    @pytest.mark.unit
    def test_framework_config_has_required_fields(self):
        """Test that all framework configs have required fields."""
        for name, config in FRAMEWORKS.items():
            assert isinstance(config, FrameworkConfig)
            assert config.name == name
            assert config.regressor_class is not None
            assert config.classifier_class is not None
            assert callable(config.save_model)
            assert callable(config.load_native)
            assert callable(config.predict_native)
            assert isinstance(config.model_extension, str)


class TestParameterSpace:
    """Tests for parameter space configurations."""

    @pytest.mark.unit
    def test_full_values_structure(self):
        """Test that FULL_VALUES has expected keys."""
        expected_keys = {
            "num_features",
            "max_depth",
            "num_trees",
            "batch_size",
        }
        assert set(FULL_VALUES.keys()) == expected_keys

    @pytest.mark.unit
    def test_quick_test_values_structure(self):
        """Test that QUICK_TEST_VALUES has expected keys."""
        expected_keys = {
            "num_features",
            "max_depth",
            "num_trees",
            "batch_size",
        }
        assert set(QUICK_TEST_VALUES.keys()) == expected_keys

    @pytest.mark.unit
    def test_quick_test_values_are_subset(self):
        """Test that quick test values are subsets of full values."""
        for key in QUICK_TEST_VALUES:
            for val in QUICK_TEST_VALUES[key]:
                assert val in FULL_VALUES[key], (
                    f"{val} not in FULL_VALUES[{key}]"
                )

    @pytest.mark.unit
    def test_quick_test_is_minimal(self):
        """Test that quick test has minimal parameter space."""
        for key in QUICK_TEST_VALUES:
            assert len(QUICK_TEST_VALUES[key]) == 1


class TestGenerateData:
    """Tests for data generation."""

    @pytest.mark.unit
    def test_generate_regression_data(self):
        """Test generating regression data."""
        X, y = generate_data(
            num_features=10, num_samples=100, model_type="regressor"
        )
        assert X.shape == (100, 10)
        assert y.shape == (100,)
        assert X.dtype == np.float32
        assert y.dtype == np.float32

    @pytest.mark.unit
    def test_generate_classification_data(self):
        """Test generating classification data."""
        X, y = generate_data(
            num_features=10, num_samples=100, model_type="classifier"
        )
        assert X.shape == (100, 10)
        assert y.shape == (100,)
        assert X.dtype == np.float32
        assert y.dtype == np.int32

    @pytest.mark.unit
    def test_generate_data_reproducible(self):
        """Test that data generation is reproducible with same seed."""
        X1, y1 = generate_data(
            num_features=10,
            num_samples=100,
            model_type="regressor",
            random_state=42,
        )
        X2, y2 = generate_data(
            num_features=10,
            num_samples=100,
            model_type="regressor",
            random_state=42,
        )
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)


class TestTrainModel:
    """Tests for model training."""

    @pytest.mark.unit
    def test_train_sklearn_regressor(self):
        """Test training sklearn regressor."""
        if "sklearn" not in FRAMEWORKS:
            pytest.skip("sklearn not available")

        X, y = generate_data(
            num_features=10, num_samples=100, model_type="regressor"
        )
        model = train_model(
            FRAMEWORKS["sklearn"],
            model_type="regressor",
            num_trees=5,
            max_depth=3,
            X_train=X,
            y_train=y,
        )
        assert hasattr(model, "predict")
        preds = model.predict(X[:10])
        assert preds.shape == (10,)

    @pytest.mark.unit
    def test_train_sklearn_classifier(self):
        """Test training sklearn classifier."""
        if "sklearn" not in FRAMEWORKS:
            pytest.skip("sklearn not available")

        X, y = generate_data(
            num_features=10, num_samples=100, model_type="classifier"
        )
        model = train_model(
            FRAMEWORKS["sklearn"],
            model_type="classifier",
            num_trees=5,
            max_depth=3,
            X_train=X,
            y_train=y,
        )
        assert hasattr(model, "predict")
        preds = model.predict(X[:10])
        assert preds.shape == (10,)

    @pytest.mark.unit
    def test_train_model_with_device_param(self):
        """Test that train_model accepts device parameter."""
        if "sklearn" not in FRAMEWORKS:
            pytest.skip("sklearn not available")

        X, y = generate_data(
            num_features=10, num_samples=100, model_type="regressor"
        )
        # sklearn ignores device, but should accept the parameter
        model = train_model(
            FRAMEWORKS["sklearn"],
            model_type="regressor",
            num_trees=5,
            max_depth=3,
            X_train=X,
            y_train=y,
            device="cpu",
        )
        assert hasattr(model, "predict")


class TestRunInferenceBenchmark:
    """Tests for inference benchmarking."""

    @pytest.mark.unit
    def test_benchmark_returns_float(self):
        """Test that benchmark returns a float timing."""
        X = np.random.rand(100, 10).astype(np.float32)

        def dummy_predict(batch):
            return batch.sum(axis=1)

        elapsed = run_inference_benchmark(
            predict_fn=dummy_predict,
            X=X,
            batch_size=10,
            warmup_cycles=1,
            benchmark_cycles=2,
        )
        assert isinstance(elapsed, float)
        assert elapsed > 0

    @pytest.mark.unit
    def test_benchmark_with_small_batch(self):
        """Test benchmark with small batch size."""
        X = np.random.rand(50, 5).astype(np.float32)

        def dummy_predict(batch):
            return batch.mean(axis=1)

        elapsed = run_inference_benchmark(
            predict_fn=dummy_predict,
            X=X,
            batch_size=5,
            warmup_cycles=1,
            benchmark_cycles=1,
        )
        assert elapsed > 0


class TestWriteCheckpoint:
    """Tests for checkpoint writing."""

    @pytest.mark.unit
    def test_write_checkpoint_creates_file(self):
        """Test that checkpoint writing creates a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = {
                "framework": ["sklearn"],
                "device": ["cpu"],
                "speedup": [1.5],
            }
            write_checkpoint(results, tmpdir, final=False)

            # Check that a checkpoint file was created
            files = os.listdir(tmpdir)
            assert any(f.startswith("checkpoint_") for f in files)

    @pytest.mark.unit
    def test_write_final_results(self):
        """Test writing final results file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = {
                "framework": ["sklearn", "xgboost"],
                "device": ["cpu", "gpu"],
                "speedup": [1.5, 2.0],
            }
            write_checkpoint(results, tmpdir, final=True)

            assert "final_results.csv" in os.listdir(tmpdir)


class TestPrintDryRunInfo:
    """Tests for dry run info printing."""

    @pytest.mark.unit
    def test_print_dry_run_info_no_error(self, capsys):
        """Test that dry run info prints without error."""
        print_dry_run_info(
            frameworks=["sklearn"],
            model_types=["regressor"],
            devices=["cpu"],
            param_values=QUICK_TEST_VALUES,
        )
        captured = capsys.readouterr()
        assert "Benchmark Configuration" in captured.out
        assert "sklearn" in captured.out
        assert "regressor" in captured.out
        assert "cpu" in captured.out


class TestCLI:
    """Tests for CLI commands."""

    @pytest.mark.unit
    def test_cli_dry_run(self):
        """Test CLI dry run option."""
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--dry-run"])
        assert result.exit_code == 0
        assert "Benchmark Configuration" in result.output

    @pytest.mark.unit
    def test_cli_dry_run_with_framework(self):
        """Test CLI dry run with specific framework."""
        runner = CliRunner()
        result = runner.invoke(
            cli, ["run", "--dry-run", "--framework", "sklearn"]
        )
        assert result.exit_code == 0
        assert "sklearn" in result.output

    @pytest.mark.unit
    def test_cli_dry_run_quick_test(self):
        """Test CLI dry run with quick test."""
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--dry-run", "--quick-test"])
        assert result.exit_code == 0
        # Quick test has fewer configurations
        assert "Total benchmark configurations: 1" in result.output

    @pytest.mark.unit
    def test_cli_dry_run_single_device(self):
        """Test CLI dry run with single device."""
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--dry-run", "--device", "cpu"])
        assert result.exit_code == 0
        assert "cpu" in result.output

    @pytest.mark.unit
    def test_cli_dry_run_single_model_type(self):
        """Test CLI dry run with single model type."""
        runner = CliRunner()
        result = runner.invoke(
            cli, ["run", "--dry-run", "--model-type", "regressor"]
        )
        assert result.exit_code == 0
        assert "regressor" in result.output

    @pytest.mark.unit
    def test_cli_invalid_framework(self):
        """Test CLI with invalid framework."""
        runner = CliRunner()
        result = runner.invoke(
            cli, ["run", "--dry-run", "--framework", "invalid"]
        )
        assert result.exit_code != 0

    @pytest.mark.unit
    def test_cli_multiple_frameworks(self):
        """Test CLI with multiple frameworks."""
        runner = CliRunner()
        result = runner.invoke(
            cli, ["run", "--dry-run", "-f", "sklearn", "-f", "xgboost"]
        )
        # May fail if xgboost not installed, but should not error on parsing
        if result.exit_code == 0:
            assert "sklearn" in result.output


class TestDeviceHandling:
    """Tests for device-aware inference."""

    @pytest.mark.unit
    def test_sklearn_predict_native_accepts_device(self):
        """Test that sklearn predict_native accepts device parameter."""
        if "sklearn" not in FRAMEWORKS:
            pytest.skip("sklearn not available")

        X, y = generate_data(
            num_features=10, num_samples=100, model_type="regressor"
        )
        model = train_model(
            FRAMEWORKS["sklearn"],
            model_type="regressor",
            num_trees=5,
            max_depth=3,
            X_train=X,
            y_train=y,
        )

        framework = FRAMEWORKS["sklearn"]
        # Should work with device parameter
        preds = framework.predict_native(model, X[:10], "cpu")
        assert preds.shape == (10,)

    @pytest.mark.unit
    def test_sklearn_load_native_accepts_device(self):
        """Test that sklearn load_native accepts device parameter."""
        if "sklearn" not in FRAMEWORKS:
            pytest.skip("sklearn not available")

        framework = FRAMEWORKS["sklearn"]
        # sklearn load_native returns None (uses trained model directly)
        result = framework.load_native("dummy_path", "cpu")
        assert result is None

    @pytest.mark.unit
    def test_xgboost_predict_native_accepts_device(self):
        """Test that xgboost predict_native accepts device parameter."""
        if "xgboost" not in FRAMEWORKS:
            pytest.skip("xgboost not available")

        X, y = generate_data(
            num_features=10, num_samples=100, model_type="regressor"
        )
        model = train_model(
            FRAMEWORKS["xgboost"],
            model_type="regressor",
            num_trees=5,
            max_depth=3,
            X_train=X,
            y_train=y,
            device="cpu",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            framework = FRAMEWORKS["xgboost"]
            model_path = os.path.join(
                tmpdir, f"model{framework.model_extension}"
            )
            framework.save_model(model, model_path)

            # Load and predict with device parameter
            native_model = framework.load_native(model_path, "cpu")
            preds = framework.predict_native(native_model, X[:10], "cpu")
            assert preds.shape == (10,)

    @pytest.mark.unit
    def test_lightgbm_predict_native_accepts_device(self):
        """Test that lightgbm predict_native accepts device parameter."""
        if "lightgbm" not in FRAMEWORKS:
            pytest.skip("lightgbm not available")

        X, y = generate_data(
            num_features=10, num_samples=100, model_type="regressor"
        )
        model = train_model(
            FRAMEWORKS["lightgbm"],
            model_type="regressor",
            num_trees=5,
            max_depth=3,
            X_train=X,
            y_train=y,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            framework = FRAMEWORKS["lightgbm"]
            model_path = os.path.join(
                tmpdir, f"model{framework.model_extension}"
            )
            framework.save_model(model, model_path)

            # Load and predict with device parameter
            native_model = framework.load_native(model_path, "cpu")
            preds = framework.predict_native(native_model, X[:10], "cpu")
            assert preds.shape == (10,)

    @pytest.mark.unit
    def test_lightgbm_predict_native_gpu_fallback(self):
        """Test that lightgbm predict_native handles GPU request gracefully."""
        if "lightgbm" not in FRAMEWORKS:
            pytest.skip("lightgbm not available")

        X, y = generate_data(
            num_features=10, num_samples=100, model_type="regressor"
        )
        # Train on CPU (GPU training requires special build)
        model = train_model(
            FRAMEWORKS["lightgbm"],
            model_type="regressor",
            num_trees=5,
            max_depth=3,
            X_train=X,
            y_train=y,
            device="cpu",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            framework = FRAMEWORKS["lightgbm"]
            model_path = os.path.join(
                tmpdir, f"model{framework.model_extension}"
            )
            framework.save_model(model, model_path)

            # Load and predict - should work even with gpu device request
            # (will fall back to CPU if GPU not available)
            native_model = framework.load_native(model_path, "gpu")
            preds = framework.predict_native(native_model, X[:10], "gpu")
            assert preds.shape == (10,)

    @pytest.mark.unit
    def test_framework_supports_gpu_native_flag(self):
        """Test that supports_gpu_native flag is set correctly."""
        if "sklearn" in FRAMEWORKS:
            assert FRAMEWORKS["sklearn"].supports_gpu_native is False
        if "xgboost" in FRAMEWORKS:
            assert FRAMEWORKS["xgboost"].supports_gpu_native is True
        if "lightgbm" in FRAMEWORKS:
            assert FRAMEWORKS["lightgbm"].supports_gpu_native is True

    @pytest.mark.unit
    def test_lightgbm_train_with_gpu_device_param(self):
        """Test that LightGBM train_model accepts GPU device parameter.

        Note: Actual GPU training requires LightGBM built with GPU support.
        This test verifies the code path works (may fall back to CPU).
        """
        if "lightgbm" not in FRAMEWORKS:
            pytest.skip("lightgbm not available")

        X, y = generate_data(
            num_features=10, num_samples=100, model_type="regressor"
        )

        # This may raise an error if LightGBM GPU is not available,
        # which is expected behavior - the training code should handle it
        try:
            model = train_model(
                FRAMEWORKS["lightgbm"],
                model_type="regressor",
                num_trees=5,
                max_depth=3,
                X_train=X,
                y_train=y,
                device="gpu",
            )
            assert hasattr(model, "predict")
        except Exception as e:
            # Expected if LightGBM GPU support is not available
            assert "GPU" in str(e).upper() or "gpu" in str(e).lower()


class TestEndToEnd:
    """End-to-end tests for benchmark functionality."""

    @pytest.mark.unit
    def test_sklearn_model_save_load(self):
        """Test saving and loading sklearn model."""
        if "sklearn" not in FRAMEWORKS:
            pytest.skip("sklearn not available")

        import nvforest

        with tempfile.TemporaryDirectory() as tmpdir:
            framework = FRAMEWORKS["sklearn"]
            X, y = generate_data(
                num_features=10, num_samples=100, model_type="regressor"
            )

            # Train model
            model = train_model(
                framework,
                model_type="regressor",
                num_trees=5,
                max_depth=3,
                X_train=X,
                y_train=y,
            )

            # Save model
            model_path = os.path.join(
                tmpdir, f"model{framework.model_extension}"
            )
            framework.save_model(model, model_path)

            assert os.path.exists(model_path)

            # Load with nvforest
            nvforest_model = nvforest.load_model(model_path, device="cpu")
            assert nvforest_model is not None

            # Verify prediction works
            preds = nvforest_model.predict(X[:10])
            assert preds.shape[0] == 10


class TestAnalyze:
    """Tests for the analyze module."""

    @pytest.mark.unit
    def test_generate_summary_stats(self):
        """Test generating summary statistics."""
        import pandas as pd

        from nvforest.benchmark.analyze import generate_summary_stats

        df = pd.DataFrame(
            {
                "framework": ["sklearn", "sklearn", "xgboost", "xgboost"],
                "model_type": [
                    "regressor",
                    "regressor",
                    "regressor",
                    "regressor",
                ],
                "device": ["cpu", "cpu", "cpu", "cpu"],
                "speedup": [1.5, 2.0, 1.8, 2.2],
                "native_time": [0.1, 0.2, 0.15, 0.25],
                "nvforest_time": [0.05, 0.1, 0.08, 0.12],
            }
        )

        summary = generate_summary_stats(df)
        assert summary is not None
        assert len(summary) == 2  # Two frameworks

    @pytest.mark.unit
    def test_print_summary(self, capsys):
        """Test printing summary."""
        import pandas as pd

        from nvforest.benchmark.analyze import print_summary

        df = pd.DataFrame(
            {
                "framework": ["sklearn", "sklearn"],
                "model_type": ["regressor", "regressor"],
                "device": ["cpu", "cpu"],
                "num_trees": [10, 20],
                "max_depth": [4, 4],
                "batch_size": [1024, 1024],
                "speedup": [1.5, 2.0],
                "native_time": [0.1, 0.2],
                "nvforest_time": [0.05, 0.1],
            }
        )

        print_summary(df)
        captured = capsys.readouterr()
        assert "BENCHMARK SUMMARY" in captured.out
        assert "sklearn" in captured.out.upper()

    @pytest.mark.unit
    def test_analyze_cli_file_not_found(self):
        """Test analyze CLI with non-existent file."""
        from nvforest.benchmark.analyze import analyze

        runner = CliRunner()
        result = runner.invoke(analyze, ["nonexistent.csv"])
        assert result.exit_code != 0

    @pytest.mark.unit
    def test_analyze_cli_with_results_file(self):
        """Test analyze CLI with valid results file."""
        import pandas as pd

        from nvforest.benchmark.analyze import analyze

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a sample results file
            df = pd.DataFrame(
                {
                    "framework": ["sklearn", "sklearn"],
                    "model_type": ["regressor", "regressor"],
                    "device": ["cpu", "cpu"],
                    "num_features": [32, 32],
                    "max_depth": [4, 8],
                    "num_trees": [16, 16],
                    "batch_size": [1024, 1024],
                    "native_time": [0.1, 0.2],
                    "nvforest_time": [0.05, 0.1],
                    "optimal_layout": ["depth_first", "breadth_first"],
                    "optimal_chunk_size": [8, 16],
                    "speedup": [2.0, 2.0],
                }
            )
            results_path = os.path.join(tmpdir, "results.csv")
            df.to_csv(results_path, index=False)

            runner = CliRunner()
            result = runner.invoke(analyze, [results_path, "--summary-only"])
            assert result.exit_code == 0
            assert "BENCHMARK SUMMARY" in result.output

    @pytest.mark.unit
    def test_analyze_cli_with_filter(self):
        """Test analyze CLI with framework filter."""
        import pandas as pd

        from nvforest.benchmark.analyze import analyze

        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame(
                {
                    "framework": ["sklearn", "xgboost"],
                    "model_type": ["regressor", "regressor"],
                    "device": ["cpu", "cpu"],
                    "num_features": [32, 32],
                    "max_depth": [4, 4],
                    "num_trees": [16, 16],
                    "batch_size": [1024, 1024],
                    "native_time": [0.1, 0.15],
                    "nvforest_time": [0.05, 0.08],
                    "optimal_layout": ["depth_first", "depth_first"],
                    "optimal_chunk_size": [8, 8],
                    "speedup": [2.0, 1.875],
                }
            )
            results_path = os.path.join(tmpdir, "results.csv")
            df.to_csv(results_path, index=False)

            runner = CliRunner()
            result = runner.invoke(
                analyze,
                [results_path, "--summary-only", "--framework", "sklearn"],
            )
            assert result.exit_code == 0
            assert "Total benchmark runs: 1" in result.output
