# cuforest Benchmark Suite

Comprehensive benchmark comparing cuforest inference performance against native ML framework inference (sklearn, XGBoost, LightGBM).

## Quick Start

```bash
# Dry run - see what will be benchmarked
python -m cuforest.benchmark.benchmark run --dry-run

# Quick test - verify setup with minimal parameters
python -m cuforest.benchmark.benchmark run --quick-test

# Full benchmark
python -m cuforest.benchmark.benchmark run
```

## Usage

### Running Benchmarks

```bash
python -m cuforest.benchmark.benchmark run [OPTIONS]
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
python -m cuforest.benchmark.benchmark run --framework sklearn --device cpu

# Benchmark XGBoost and LightGBM classifiers only
python -m cuforest.benchmark.benchmark run -f xgboost -f lightgbm -m classifier

# Quick test with specific framework
python -m cuforest.benchmark.benchmark run --quick-test --framework sklearn
```

### Analyzing Results

```bash
python -m cuforest.benchmark.analyze RESULTS_FILE [OPTIONS]
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
python -m cuforest.benchmark.analyze data/final_results.csv

# Summary only for GPU results
python -m cuforest.benchmark.analyze data/final_results.csv --device gpu --summary-only
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
| `cuforest_time` | cuforest inference time (seconds) |
| `optimal_layout` | Layout selected by cuforest optimize() |
| `optimal_chunk_size` | Chunk size selected by cuforest optimize() |
| `speedup` | native_time / cuforest_time |

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
