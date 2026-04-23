"""
Data augmentation script for bike-sharing demand dataset.
Creates synthetic samples while avoiding duplicates.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def augment_bike_data(input_path, output_path=None, augmentation_factor=2, seed=42):
    """
    Augment bike-sharing demand data by creating synthetic samples.
    
    Parameters:
    -----------
    input_path : str or Path
        Path to the input CSV file
    output_path : str or Path, optional
        Path to save augmented data. If None, returns DataFrame
    augmentation_factor : int
        Multiplier for dataset size (2 = double the data)
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    pd.DataFrame
        Augmented dataset with original + synthetic samples
    """
    np.random.seed(seed)
    
    # Load original data
    df = pd.read_csv(input_path)
    print(f"Original dataset size: {len(df)}")
    
    # Identify column types
    categorical_cols = []
    numeric_cols = []
    target_cols = ['casual', 'registered', 'count']  # Don't augment targets directly
    
    for col in df.columns:
        if col in ['datetime']:
            continue  # Handle datetime separately
        elif col in target_cols:
            continue  # Handle targets separately
        elif df[col].dtype in ['int64', 'float64'] and df[col].nunique() < 20:
            categorical_cols.append(col)
        elif df[col].dtype in ['int64', 'float64']:
            numeric_cols.append(col)
    
    print(f"Categorical columns: {categorical_cols}")
    print(f"Numeric columns: {numeric_cols}")
    
    # Generate synthetic samples
    num_synthetic = len(df) * (augmentation_factor - 1)
    synthetic_samples = []
    
    for i in range(num_synthetic):
        # Randomly select two samples to interpolate/combine
        idx1, idx2 = np.random.choice(len(df), size=2, replace=False)
        sample1 = df.iloc[idx1]
        sample2 = df.iloc[idx2]
        
        new_sample = {}
        
        # Handle datetime: use one of the existing samples or create nearby time
        if 'datetime' in df.columns:
            base_datetime = pd.to_datetime(sample1['datetime'])
            # Add random time offset (±3 hours)
            offset_hours = np.random.randint(-3, 4)
            new_datetime = base_datetime + pd.Timedelta(hours=offset_hours)
            new_sample['datetime'] = new_datetime.strftime('%Y-%m-%d %H:%M:%S')
        
        # Handle categorical columns: choose from one sample or pick random valid value
        for col in categorical_cols:
            if np.random.random() < 0.8:
                # 80% chance: use value from one of the parent samples
                new_sample[col] = sample1[col] if np.random.random() < 0.5 else sample2[col]
            else:
                # 20% chance: pick any valid value from the distribution
                new_sample[col] = np.random.choice(df[col].dropna().values)
        
        # Handle numeric columns: interpolate with noise
        for col in numeric_cols:
            # Interpolation factor between the two samples
            alpha = np.random.beta(2, 2)  # Beta distribution favors middle values
            interpolated = alpha * sample1[col] + (1 - alpha) * sample2[col]
            
            # Add small Gaussian noise (5% of std)
            noise = np.random.normal(0, df[col].std() * 0.05)
            new_value = interpolated + noise
            
            # Ensure value stays in valid range
            new_value = np.clip(new_value, df[col].min(), df[col].max())
            new_sample[col] = new_value
        
        # Handle target columns: use relationship-based generation
        # For bike sharing, casual + registered = count
        if 'temp' in numeric_cols or 'atemp' in numeric_cols:
            # Temperature strongly correlates with demand
            temp_col = 'atemp' if 'atemp' in numeric_cols else 'temp'
            temp_percentile = (new_sample[temp_col] - df[temp_col].min()) / (df[temp_col].max() - df[temp_col].min())
            
            # Find similar weather conditions
            similar_mask = (
                (df[temp_col] >= new_sample[temp_col] - df[temp_col].std() * 0.5) &
                (df[temp_col] <= new_sample[temp_col] + df[temp_col].std() * 0.5)
            )
            
            if similar_mask.sum() > 0:
                similar_samples = df[similar_mask]
                base_casual = similar_samples['casual'].mean()
                base_registered = similar_samples['registered'].mean()
            else:
                base_casual = df['casual'].mean()
                base_registered = df['registered'].mean()
            
            # Add variation based on other factors
            variation = np.random.normal(1.0, 0.15)
            new_sample['casual'] = max(0, int(base_casual * variation))
            new_sample['registered'] = max(0, int(base_registered * variation))
            new_sample['count'] = new_sample['casual'] + new_sample['registered']
        else:
            # Fallback: interpolate targets
            alpha = np.random.beta(2, 2)
            for target in target_cols:
                if target in df.columns:
                    new_sample[target] = int(alpha * sample1[target] + (1 - alpha) * sample2[target])
        
        synthetic_samples.append(new_sample)
    
    # Combine original and synthetic data
    synthetic_df = pd.DataFrame(synthetic_samples)
    augmented_df = pd.concat([df, synthetic_df], ignore_index=True)
    
    # Remove any exact duplicates (should be extremely rare with this approach)
    original_len = len(augmented_df)
    augmented_df = augmented_df.drop_duplicates()
    duplicates_removed = original_len - len(augmented_df)
    
    if duplicates_removed > 0:
        print(f"Removed {duplicates_removed} exact duplicates")
    
    print(f"Augmented dataset size: {len(augmented_df)}")
    print(f"Augmentation ratio: {len(augmented_df) / len(df):.2f}x")
    
    # Verify no exact duplicates exist
    assert augmented_df.duplicated().sum() == 0, "Duplicates detected in augmented data!"
    
    # Save if output path provided
    if output_path:
        augmented_df.to_csv(output_path, index=False)
        print(f"Saved augmented data to {output_path}")
    
    return augmented_df


def create_stratified_augmentation(input_path, output_path=None, target_col='count', 
                                   bins=5, samples_per_bin=None, seed=42):
    """
    Augment data with stratification to maintain target distribution.
    Useful for imbalanced datasets.
    
    Parameters:
    -----------
    input_path : str or Path
        Path to the input CSV file
    output_path : str or Path, optional
        Path to save augmented data
    target_col : str
        Target column name for stratification
    bins : int
        Number of bins for stratification
    samples_per_bin : int, optional
        Number of synthetic samples per bin. If None, uses size of largest bin
    seed : int
        Random seed
    
    Returns:
    --------
    pd.DataFrame
        Augmented dataset
    """
    np.random.seed(seed)
    df = pd.read_csv(input_path)
    
    # Create bins for stratification
    df['_bin'] = pd.qcut(df[target_col], q=bins, labels=False, duplicates='drop')
    
    bin_counts = df['_bin'].value_counts().sort_index()
    print(f"Samples per bin: {bin_counts.to_dict()}")
    
    if samples_per_bin is None:
        samples_per_bin = bin_counts.max()
    
    augmented_dfs = [df.drop('_bin', axis=1)]
    
    for bin_id in df['_bin'].unique():
        bin_df = df[df['_bin'] == bin_id].drop('_bin', axis=1)
        current_count = len(bin_df)
        needed = samples_per_bin - current_count
        
        if needed <= 0:
            continue
        
        print(f"Augmenting bin {bin_id}: adding {needed} samples")
        
        # Generate synthetic samples within this bin
        synthetic_samples = []
        for _ in range(needed):
            idx1, idx2 = np.random.choice(len(bin_df), size=2, replace=True)
            sample1 = bin_df.iloc[idx1]
            sample2 = bin_df.iloc[idx2]
            
            new_sample = {}
            alpha = np.random.uniform(0.3, 0.7)  # Interpolation weight
            
            for col in bin_df.columns:
                if col == 'datetime':
                    new_sample[col] = sample1[col]
                elif bin_df[col].dtype == 'object':
                    new_sample[col] = sample1[col] if np.random.random() < 0.5 else sample2[col]
                else:
                    val = alpha * sample1[col] + (1 - alpha) * sample2[col]
                    if bin_df[col].dtype in ['int64']:
                        val = int(round(val))
                    new_sample[col] = val
            
            synthetic_samples.append(new_sample)
        
        synthetic_df = pd.DataFrame(synthetic_samples)
        augmented_dfs.append(synthetic_df)
    
    result = pd.concat(augmented_dfs, ignore_index=True)
    result = result.drop_duplicates()
    
    print(f"Final augmented size: {len(result)}")
    
    if output_path:
        result.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")
    
    return result


if __name__ == "__main__":
    # Example usage
    input_file = Path(__file__).parent / "input" / "train.csv"
    
    # Basic augmentation (double the dataset)
    output_file = Path(__file__).parent / "input" / "train_augmented_2x.csv"
    augment_bike_data(
        input_file, 
        output_file, 
        augmentation_factor=2,
        seed=42
    )
    
    # Triple the dataset
    output_file_3x = Path(__file__).parent / "input" / "train_augmented_3x.csv"
    augment_bike_data(
        input_file, 
        output_file_3x, 
        augmentation_factor=3,
        seed=42
    )
    
    # Stratified augmentation (maintains target distribution)
    output_file_stratified = Path(__file__).parent / "input" / "train_augmented_stratified.csv"
    create_stratified_augmentation(
        input_file,
        output_file_stratified,
        target_col='count',
        bins=5,
        seed=42
    )
    
    print("\n✅ Data augmentation complete!")
    print("Generated files:")
    print(f"  - {output_file.name}")
    print(f"  - {output_file_3x.name}")
    print(f"  - {output_file_stratified.name}")
