# Bike Sharing Demand

This benchmark evaluates data preprocessing pipelines on the bike-sharing demand dataset, comparing different approaches for feature engineering and encoding.

## Overview

The benchmark consists of multiple pipeline implementations (`pipeline0.py` through `pipeline4.py`) that demonstrate various preprocessing strategies for the bike-sharing demand prediction task. The goal is to compare the performance and execution time of different pipeline designs.
Before running the benchmark, please download the dataset and augment the dataset, as described below.

## Running the Benchmark

### Baseline Pipelines

To run the baseline pipeline comparisons:

```bash
python run_base_lines.py
```

This script executes multiple pipeline variants (pipeline0-4) and measures:
- Training time
- Prediction performance
- Memory usage

Each pipeline implements different preprocessing strategies, allowing you to compare trade-offs between complexity and performance.

### Skrubified Pipelines

To run the optimized stratum/skrub-based pipelines:

```bash 
python skrubified_pipelines.py
```

## Data
```bash
kaggle competitions download -c bike-sharing-demand
unzip bike-sharing-demand.zip -d input/
rm bike-sharing-demand.zip
```


The benchmark uses various versions of the bike-sharing demand dataset:
- `train.csv` - Original training data
- `train_augmented_2x.csv` - 2x augmented dataset
- `train_augmented_3x.csv` - 3x augmented dataset
- `train_augmented_stratified.csv` - Stratified augmentation

Data augmentation scripts are available in `bike_data_augmentation.py`.
```bash
python bike_data_augmentation.py
```