# nvforest Benchmark Suite

Comprehensive benchmark comparing nvforest inference performance against native ML framework inference (sklearn, XGBoost, LightGBM).

## Quick Start

```bash
# Dry run - see what will be benchmarked
python -m nvforest.benchmark.run --dry-run

# Quick test - verify setup with minimal parameters
python -m nvforest.benchmark.run --quick-test

# Full benchmark
python -m nvforest.benchmark.run
```

## Usage

### Running Benchmarks

```bash
python -m nvforest.benchmark.run [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--framework` | `-f` | Framework(s) to benchmark: `sklearn`, `xgboost`, `lightgbm`. Repeatable. Default: all available |
| `--dry-run` | `-n` | Print configuration without running |
| `--quick-test` | `-q` | Run with minimal parameters for quick verification |
| `--device` | `-d` | Device: `cpu`, `gpu`, or `both`. Default: `both` |
| `--model-type` | `-m` | Model type: `regressor`, `classifier`, or `both`. Default: `both` |
| `--output-dir` | `-o` | Output directory for results. Default: `benchmark/data/` |

**Examples:**

```bash
# Benchmark only sklearn on CPU
python -m nvforest.benchmark.run --framework sklearn --device cpu

# Benchmark XGBoost and LightGBM classifiers only
python -m nvforest.benchmark.run -f xgboost -f lightgbm -m classifier

# Quick test with specific framework
python -m nvforest.benchmark.run --quick-test --framework sklearn
```

### Analyzing Results

```bash
python -m nvforest.benchmark.analyze RESULTS_FILE [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--output` | `-o` | Output file for speedup heatmap plot |
| `--framework` | `-f` | Filter results to specific framework |
| `--device` | `-d` | Filter results to specific device (`cpu` or `gpu`) |
| `--plot-only` | | Only generate plot, skip summary |
| `--summary-only` | | Only print summary, skip plot |

**Examples:**

```bash
# Analyze results and generate plots
python -m nvforest.benchmark.analyze data/final_results.csv

# Summary only for GPU results
python -m nvforest.benchmark.analyze data/final_results.csv --device gpu --summary-only
```

## Parameter Space

### Full Benchmark

| Parameter | Values |
|-----------|--------|
| `num_features` | 8, 32, 128, 512 |
| `max_depth` | 2, 4, 8, 16, 32 |
| `num_trees` | 16, 128, 1024 |
| `batch_size` | 1, 16, 128, 1024, 1,048,576, 16,777,216 |

### Quick Test

| Parameter | Values |
|-----------|--------|
| `num_features` | 32 |
| `max_depth` | 4 |
| `num_trees` | 16 |
| `batch_size` | 1024 |

## Device Handling

The `--device` parameter affects how both native frameworks and nvforest run inference:

### nvforest
- **CPU**: Uses CPU inference backend
- **GPU**: Uses GPU inference backend with cupy arrays

### XGBoost
- **CPU**: Standard CPU inference with DMatrix
- **GPU**: Models are trained with `device="cuda"` and use `inplace_predict` for GPU inference

### LightGBM
- **CPU**: Standard CPU inference
- **GPU**: LightGBM GPU training and inference require the library to be built with GPU support:
  ```bash
  pip install lightgbm --no-binary lightgbm --config-settings=cmake.define.USE_GPU=ON
  ```

When GPU is requested and LightGBM GPU support is available, models are trained with `device="gpu"` and inference uses the GPU-trained model. If GPU support is not available, training/inference falls back to CPU with a warning.

### sklearn
- **CPU**: Standard CPU inference
- **GPU**: sklearn is CPU-only. For GPU benchmarks, native inference runs on CPU as a baseline. The speedup comparison reflects nvforest GPU vs sklearn CPU.

> **Note**: When `device=gpu`, XGBoost models are trained on GPU which enables GPU-native inference. This provides a fair comparison between XGBoost GPU inference and nvforest GPU inference.

## Output

Results are saved as CSV files in the output directory:

- `checkpoint_N.csv` - Periodic checkpoints during benchmark
- `final_results.csv` - Complete results

**Columns:**

| Column | Description |
|--------|-------------|
| `framework` | ML framework (sklearn, xgboost, lightgbm) |
| `model_type` | regressor or classifier |
| `device` | cpu or gpu |
| `num_features` | Number of input features |
| `max_depth` | Maximum tree depth |
| `num_trees` | Number of trees in ensemble |
| `batch_size` | Inference batch size |
| `native_time` | Native framework inference time (seconds) |
| `nvforest_time` | nvforest inference time (seconds) |
| `optimal_layout` | Layout selected by nvforest optimize() |
| `optimal_chunk_size` | Chunk size selected by nvforest optimize() |
| `speedup` | native_time / nvforest_time |

## Dependencies

Required:
- `click`
- `pandas`
- `numpy`

Optional (for specific frameworks):
- `scikit-learn` - for sklearn benchmarks
- `xgboost` - for XGBoost benchmarks
- `lightgbm` - for LightGBM benchmarks
- `matplotlib`, `seaborn` - for result visualization
