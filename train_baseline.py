"""
Baseline Model Training Script

Trains logistic regression and random forest classifiers on ECG data
with evaluation stratified by demographics.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import ast
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

from src.features.extractor import extract_ecg_features


def prepare_labels(metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare diagnostic labels from metadata.
    
    Args:
        metadata: PTB-XL metadata DataFrame
        
    Returns:
        DataFrame with binary labels for common conditions
    """
    # Load SCP statements
    data_path = Path('physionet.org/files/ptb-xl/1.0.3/')
    agg_df = pd.read_csv(data_path / 'scp_statements.csv', index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]
    
    # Parse SCP codes
    metadata['scp_codes'] = metadata.scp_codes.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    # Aggregate to diagnostic superclass
    def aggregate_diagnostic(y_dict):
        tmp = []
        for key in y_dict.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))
    
    metadata['diagnostic_superclass'] = metadata.scp_codes.apply(aggregate_diagnostic)
    
    # Create binary label: NORM vs abnormal
    metadata['is_normal'] = metadata['diagnostic_superclass'].apply(
        lambda x: 1 if x == ['NORM'] or 'NORM' in x else 0
    )
    
    # Also create label for MI (myocardial infarction)
    metadata['has_mi'] = metadata['diagnostic_superclass'].apply(
        lambda x: 1 if 'MI' in x else 0
    )
    
    return metadata


def prepare_age_groups(metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Create age groups for stratification.
    
    Args:
        metadata: Metadata DataFrame
        
    Returns:
        DataFrame with age groups
    """
    metadata['age_group'] = pd.cut(
        metadata['age'],
        bins=[0, 40, 65, 150],
        labels=['<=40', '41-65', '>65']
    )
    
    return metadata


def load_or_extract_features(
    metadata: pd.DataFrame,
    data_path: Path,
    cache_path: Path,
    max_samples: int = 1000,
    force_recompute: bool = False
) -> pd.DataFrame:
    """
    Load features from cache or extract them.
    
    Args:
        metadata: Metadata DataFrame
        data_path: Path to ECG data
        cache_path: Path to save/load features
        max_samples: Maximum samples to process
        force_recompute: Force recomputation
        
    Returns:
        DataFrame with features and metadata
    """
    cache_file = cache_path / f'features_{max_samples}.csv'
    
    if cache_file.exists() and not force_recompute:
        print(f"Loading cached features from {cache_file}...")
        features_df = pd.read_csv(cache_file, index_col='ecg_id')
        print(f"✓ Loaded {len(features_df)} records with {len(features_df.columns)} columns")
        return features_df
    
    print(f"Extracting features from {max_samples} ECG records...")
    print("(This will take several minutes...)")
    
    features_df = extract_ecg_features(
        metadata_df=metadata.head(max_samples),
        data_path=data_path,
        lead_idx=1,  # Lead II
        fs=100.0,
        apply_preprocessing=True,
        apply_normalization=True,
        quality_check=True,
        verbose=True
    )
    
    # Save to cache
    cache_path.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(cache_file)
    print(f"✓ Features saved to {cache_file}")
    
    return features_df


def train_baseline_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42
) -> Dict:
    """
    Train baseline classification models.
    
    Args:
        X_train: Training features
        y_train: Training labels
        random_state: Random seed
        
    Returns:
        Dictionary of trained models
    """
    models = {}
    
    print("\n" + "="*80)
    print("TRAINING BASELINE MODELS")
    print("="*80)
    
    # Logistic Regression
    print("\n1. Training Logistic Regression...")
    lr_model = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        class_weight='balanced'  # Handle class imbalance
    )
    lr_model.fit(X_train, y_train)
    models['logistic_regression'] = lr_model
    print("   ✓ Logistic Regression trained")
    
    # Random Forest
    print("\n2. Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=random_state,
        class_weight='balanced',
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    models['random_forest'] = rf_model
    print("   ✓ Random Forest trained")
    
    return models


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray
) -> Dict[str, float]:
    """
    Compute classification metrics including FNR, FPR, and calibration error.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (for positive class)
        
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import accuracy_score, confusion_matrix
    
    metrics = {}
    
    # Accuracy
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # False Negative Rate (FNR) = FN / (FN + TP)
    metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    # False Positive Rate (FPR) = FP / (FP + TN)
    metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    # True Positive Rate (TPR) / Sensitivity
    metrics['tpr'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # True Negative Rate (TNR) / Specificity
    metrics['tnr'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # Precision
    metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # F1 Score
    if metrics['precision'] + metrics['tpr'] > 0:
        metrics['f1'] = 2 * (metrics['precision'] * metrics['tpr']) / (metrics['precision'] + metrics['tpr'])
    else:
        metrics['f1'] = 0.0
    
    # Sample counts
    metrics['n_samples'] = len(y_true)
    metrics['n_positive'] = int(np.sum(y_true == 1))
    metrics['n_negative'] = int(np.sum(y_true == 0))
    
    # Expected Calibration Error (ECE)
    metrics['ece'] = expected_calibration_error(y_true, y_prob)
    
    return metrics


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Calculate Expected Calibration Error (ECE).
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        n_bins: Number of bins
        
    Returns:
        ECE value
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Find samples in this bin
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        
        if np.sum(in_bin) > 0:
            # Average confidence in bin
            avg_confidence = np.mean(y_prob[in_bin])
            
            # Average accuracy in bin
            avg_accuracy = np.mean(y_true[in_bin])
            
            # Weighted contribution to ECE
            bin_weight = np.sum(in_bin) / len(y_true)
            ece += bin_weight * np.abs(avg_confidence - avg_accuracy)
    
    return ece


def evaluate_model_stratified(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    metadata_test: pd.DataFrame,
    model_name: str
) -> pd.DataFrame:
    """
    Evaluate model with stratification by demographics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        metadata_test: Test metadata with demographics
        model_name: Name of the model
        
    Returns:
        DataFrame with stratified metrics
    """
    print(f"\n{'='*80}")
    print(f"EVALUATING: {model_name.upper()}")
    print('='*80)
    
    # Overall predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    results = []
    
    # Overall performance
    print("\n1. Overall Performance:")
    overall_metrics = compute_metrics(y_test, y_pred, y_prob)
    overall_metrics['model'] = model_name
    overall_metrics['stratification'] = 'Overall'
    overall_metrics['group'] = 'All'
    results.append(overall_metrics)
    
    print(f"   Accuracy:  {overall_metrics['accuracy']:.4f}")
    print(f"   FNR:       {overall_metrics['fnr']:.4f}")
    print(f"   FPR:       {overall_metrics['fpr']:.4f}")
    print(f"   ECE:       {overall_metrics['ece']:.4f}")
    print(f"   F1:        {overall_metrics['f1']:.4f}")
    
    # Stratified by sex
    print("\n2. Stratified by Sex:")
    for sex in metadata_test['sex'].unique():
        if pd.isna(sex):
            continue
        
        sex_mask = metadata_test['sex'] == sex
        if np.sum(sex_mask) < 10:  # Skip if too few samples
            continue
        
        sex_metrics = compute_metrics(
            y_test[sex_mask],
            y_pred[sex_mask],
            y_prob[sex_mask]
        )
        sex_metrics['model'] = model_name
        sex_metrics['stratification'] = 'Sex'
        sex_metrics['group'] = f"{'Male' if sex == 1 else 'Female'}"
        results.append(sex_metrics)
        
        print(f"   {sex_metrics['group']:10s}: Acc={sex_metrics['accuracy']:.4f}, "
              f"FNR={sex_metrics['fnr']:.4f}, FPR={sex_metrics['fpr']:.4f}, "
              f"ECE={sex_metrics['ece']:.4f}")
    
    # Stratified by age group
    print("\n3. Stratified by Age Group:")
    for age_group in ['<=40', '41-65', '>65']:
        age_mask = metadata_test['age_group'] == age_group
        if np.sum(age_mask) < 10:  # Skip if too few samples
            continue
        
        age_metrics = compute_metrics(
            y_test[age_mask],
            y_pred[age_mask],
            y_prob[age_mask]
        )
        age_metrics['model'] = model_name
        age_metrics['stratification'] = 'Age'
        age_metrics['group'] = age_group
        results.append(age_metrics)
        
        print(f"   {age_group:10s}: Acc={age_metrics['accuracy']:.4f}, "
              f"FNR={age_metrics['fnr']:.4f}, FPR={age_metrics['fpr']:.4f}, "
              f"ECE={age_metrics['ece']:.4f}")
    
    return pd.DataFrame(results)


def main():
    """Main training and evaluation pipeline."""
    
    print("="*80)
    print("BASELINE ECG CLASSIFIER TRAINING")
    print("="*80)
    
    # Paths
    data_path = Path('physionet.org/files/ptb-xl/1.0.3/')
    cache_path = Path('data/processed/')
    
    # Load metadata
    print("\n1. Loading metadata...")
    metadata = pd.read_csv(data_path / 'ptbxl_database.csv', index_col='ecg_id')
    print(f"   ✓ Loaded {len(metadata)} records")
    
    # Prepare labels and age groups
    metadata = prepare_labels(metadata)
    metadata = prepare_age_groups(metadata)
    
    print(f"   ✓ Label distribution: {metadata['is_normal'].value_counts().to_dict()}")
    
    # Extract or load features
    print("\n2. Feature extraction...")
    max_samples = 500  # Adjust based on your needs
    features_df = load_or_extract_features(
        metadata,
        data_path,
        cache_path,
        max_samples=max_samples,
        force_recompute=False
    )
    
    # Prepare feature matrix
    print("\n3. Preparing feature matrix...")
    feature_cols = [c for c in features_df.columns if c.startswith('II_')]
    feature_cols = [c for c in feature_cols if not c.startswith('II_is_good') and not c.startswith('II_snr')]
    
    X = features_df[feature_cols].values
    y = features_df['is_normal'].values
    
    print(f"   ✓ Feature matrix shape: {X.shape}")
    print(f"   ✓ Label distribution: {np.bincount(y)}")
    
    # Handle missing values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Train/test split (stratified)
    print("\n4. Creating train/test split...")
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, features_df.index.values,
        test_size=0.2,
        stratify=y,
        random_state=42
    )
    
    print(f"   ✓ Train: {len(X_train)} samples")
    print(f"   ✓ Test:  {len(X_test)} samples")
    
    # Standardize features
    print("\n5. Standardizing features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train models
    models = train_baseline_models(X_train, y_train)
    
    # Evaluate models
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    all_results = []
    
    for model_name, model in models.items():
        # Get test metadata
        metadata_test = features_df.loc[idx_test]
        
        # Evaluate
        results_df = evaluate_model_stratified(
            model,
            X_test,
            y_test,
            metadata_test,
            model_name
        )
        
        all_results.append(results_df)
    
    # Combine results
    final_results = pd.concat(all_results, ignore_index=True)
    
    # Save results
    results_path = cache_path / 'baseline_evaluation_results.csv'
    final_results.to_csv(results_path, index=False)
    print(f"\n{'='*80}")
    print(f"✓ Results saved to: {results_path}")
    
    # Save models
    models_path = cache_path / 'baseline_models.pkl'
    with open(models_path, 'wb') as f:
        pickle.dump({
            'models': models,
            'scaler': scaler,
            'feature_cols': feature_cols
        }, f)
    print(f"✓ Models saved to: {models_path}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print('='*80)
    
    summary = final_results.groupby(['model', 'stratification']).agg({
        'accuracy': ['mean', 'std'],
        'fnr': ['mean', 'std'],
        'fpr': ['mean', 'std'],
        'ece': ['mean', 'std']
    }).round(4)
    
    print(summary)
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE!")
    print('='*80)
    
    return final_results, models


if __name__ == '__main__':
    results, models = main()

