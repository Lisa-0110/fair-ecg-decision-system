"""
Fairness-Aware Model Training Script

Trains models with subgroup reweighting to reduce false negative disparities
across demographic groups (sex and age).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import sys

from src.models.fairness_aware import SubgroupReweighter, print_weight_statistics
from train_baseline import (
    prepare_labels,
    prepare_age_groups,
    load_or_extract_features,
    compute_metrics
)


def train_fairness_aware_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sample_weights: np.ndarray,
    random_state: int = 42
) -> dict:
    """
    Train fairness-aware classification models with sample weights.
    
    Args:
        X_train: Training features
        y_train: Training labels
        sample_weights: Sample weights for training
        random_state: Random seed
        
    Returns:
        Dictionary of trained models
    """
    models = {}
    
    print("\n" + "="*80)
    print("TRAINING FAIRNESS-AWARE MODELS")
    print("="*80)
    
    # Logistic Regression with reweighting
    print("\n1. Training Fairness-Aware Logistic Regression...")
    lr_model = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        class_weight='balanced'  # Still use class balancing
    )
    lr_model.fit(X_train, y_train, sample_weight=sample_weights)
    models['fairness_aware_lr'] = lr_model
    print("   ✓ Fairness-Aware Logistic Regression trained")
    
    # Random Forest with reweighting
    print("\n2. Training Fairness-Aware Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=random_state,
        class_weight='balanced',
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train, sample_weight=sample_weights)
    models['fairness_aware_rf'] = rf_model
    print("   ✓ Fairness-Aware Random Forest trained")
    
    return models


def evaluate_and_compare(
    baseline_models: dict,
    fairness_models: dict,
    scaler: StandardScaler,
    X_test: np.ndarray,
    y_test: np.ndarray,
    metadata_test: pd.DataFrame
) -> pd.DataFrame:
    """
    Evaluate and compare baseline vs fairness-aware models.
    
    Args:
        baseline_models: Dictionary of baseline models
        fairness_models: Dictionary of fairness-aware models
        scaler: Fitted scaler
        X_test: Test features
        y_test: Test labels
        metadata_test: Test metadata
        
    Returns:
        Combined results DataFrame
    """
    from train_baseline import evaluate_model_stratified
    
    all_results = []
    
    # Evaluate baseline models
    print("\n" + "="*80)
    print("BASELINE MODELS EVALUATION")
    print("="*80)
    
    for model_name, model in baseline_models.items():
        results_df = evaluate_model_stratified(
            model,
            X_test,
            y_test,
            metadata_test,
            f"baseline_{model_name}"
        )
        all_results.append(results_df)
    
    # Evaluate fairness-aware models
    print("\n" + "="*80)
    print("FAIRNESS-AWARE MODELS EVALUATION")
    print("="*80)
    
    for model_name, model in fairness_models.items():
        results_df = evaluate_model_stratified(
            model,
            X_test,
            y_test,
            metadata_test,
            model_name
        )
        all_results.append(results_df)
    
    return pd.concat(all_results, ignore_index=True)


def compare_fnr_improvements(
    baseline_results: pd.DataFrame,
    fairness_results: pd.DataFrame
) -> pd.DataFrame:
    """
    Compare FNR improvements from baseline to fairness-aware models.
    
    Args:
        baseline_results: Baseline evaluation results
        fairness_results: Fairness-aware evaluation results
        
    Returns:
        DataFrame with improvement metrics
    """
    improvements = []
    
    # Map model names
    model_mapping = {
        'logistic_regression': 'lr',
        'random_forest': 'rf'
    }
    
    for baseline_name, short_name in model_mapping.items():
        # Get baseline data for this model
        base_data = baseline_results[
            baseline_results['model'] == f'baseline_{baseline_name}'
        ]
        
        # Get fairness data for corresponding model
        fair_data = fairness_results[
            fairness_results['model'] == f'fairness_aware_{short_name}'
        ]
        
        if len(base_data) == 0 or len(fair_data) == 0:
            continue
        
        # Compare by group
        for _, base_row in base_data.iterrows():
            strat = base_row['stratification']
            group = base_row['group']
            
            fair_row = fair_data[
                (fair_data['stratification'] == strat) & 
                (fair_data['group'] == group)
            ]
            
            if len(fair_row) == 0:
                continue
            
            fair_row = fair_row.iloc[0]
            
            # Calculate improvements
            fnr_improvement = base_row['fnr'] - fair_row['fnr']
            fpr_change = fair_row['fpr'] - base_row['fpr']
            acc_change = fair_row['accuracy'] - base_row['accuracy']
            
            improvements.append({
                'model': baseline_name,
                'stratification': strat,
                'group': group,
                'baseline_fnr': base_row['fnr'],
                'fairness_fnr': fair_row['fnr'],
                'fnr_improvement': fnr_improvement,
                'fnr_reduction_pct': (fnr_improvement / base_row['fnr'] * 100) if base_row['fnr'] > 0 else 0,
                'baseline_fpr': base_row['fpr'],
                'fairness_fpr': fair_row['fpr'],
                'fpr_change': fpr_change,
                'baseline_accuracy': base_row['accuracy'],
                'fairness_accuracy': fair_row['accuracy'],
                'accuracy_change': acc_change
            })
    
    return pd.DataFrame(improvements)


def print_improvement_summary(improvements_df: pd.DataFrame):
    """Print summary of improvements from fairness-aware training."""
    
    print("\n" + "="*80)
    print("FAIRNESS IMPROVEMENTS SUMMARY")
    print("="*80)
    
    print("\nFalse Negative Rate Improvements:")
    print("-" * 80)
    
    # Overall improvement
    avg_fnr_improvement = improvements_df['fnr_improvement'].mean()
    print(f"\nAverage FNR Improvement: {avg_fnr_improvement:.4f} ({avg_fnr_improvement*100:.2f}%)")
    
    # By stratification
    print("\nBy Stratification:")
    for strat in improvements_df['stratification'].unique():
        strat_data = improvements_df[improvements_df['stratification'] == strat]
        avg_imp = strat_data['fnr_improvement'].mean()
        print(f"  {strat}: {avg_imp:.4f} ({avg_imp*100:.2f}%)")
    
    # Highlight biggest improvements
    print("\nBiggest FNR Reductions:")
    top_improvements = improvements_df.nlargest(5, 'fnr_improvement')
    for _, row in top_improvements.iterrows():
        print(f"  {row['group']} ({row['stratification']}): "
              f"{row['baseline_fnr']:.4f} → {row['fairness_fnr']:.4f} "
              f"(-{row['fnr_reduction_pct']:.1f}%)")
    
    # Worst-group FNR comparison
    print("\nWorst-Group FNR Comparison:")
    print("-" * 80)
    for model in improvements_df['model'].unique():
        model_data = improvements_df[improvements_df['model'] == model]
        
        # Find worst group
        worst_baseline = model_data.nlargest(1, 'baseline_fnr').iloc[0]
        
        print(f"\n{model.upper()}:")
        print(f"  Worst group: {worst_baseline['group']} ({worst_baseline['stratification']})")
        print(f"  Baseline FNR: {worst_baseline['baseline_fnr']:.4f}")
        print(f"  Fairness FNR: {worst_baseline['fairness_fnr']:.4f}")
        print(f"  Improvement:  {worst_baseline['fnr_improvement']:.4f} "
              f"(-{worst_baseline['fnr_reduction_pct']:.1f}%)")
    
    # Check for tradeoffs
    print("\nTradeoff Analysis:")
    print("-" * 80)
    avg_fpr_change = improvements_df['fpr_change'].mean()
    avg_acc_change = improvements_df['accuracy_change'].mean()
    
    print(f"Average FPR change: {avg_fpr_change:+.4f}")
    print(f"Average Accuracy change: {avg_acc_change:+.4f}")
    
    if avg_fpr_change > 0:
        print("\n⚠️  Note: FPR increased slightly (expected tradeoff for reducing FNR)")
    if avg_acc_change < 0:
        print("⚠️  Note: Overall accuracy decreased slightly (focusing on fairness)")


def main():
    """Main fairness-aware training pipeline."""
    
    print("="*80)
    print("FAIRNESS-AWARE ECG CLASSIFIER TRAINING")
    print("="*80)
    
    # Paths
    data_path = Path('physionet.org/files/ptb-xl/1.0.3/')
    cache_path = Path('data/processed/')
    
    # Load metadata
    print("\n1. Loading metadata...")
    metadata = pd.read_csv(data_path / 'ptbxl_database.csv', index_col='ecg_id')
    metadata = prepare_labels(metadata)
    metadata = prepare_age_groups(metadata)
    print(f"   ✓ Loaded {len(metadata)} records")
    
    # Load features
    print("\n2. Loading features...")
    max_samples = 500
    features_df = load_or_extract_features(
        metadata,
        data_path,
        cache_path,
        max_samples=max_samples,
        force_recompute=False
    )
    
    # Prepare data
    print("\n3. Preparing feature matrix...")
    feature_cols = [c for c in features_df.columns if c.startswith('II_')]
    feature_cols = [c for c in feature_cols if not c.startswith('II_is_good') and not c.startswith('II_snr')]
    
    X = features_df[feature_cols].values
    y = features_df['is_normal'].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Train/test split
    print("\n4. Creating train/test split...")
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, features_df.index.values,
        test_size=0.2,
        stratify=y,
        random_state=42
    )
    
    # Get metadata for train/test
    metadata_train = features_df.loc[idx_train]
    metadata_test = features_df.loc[idx_test]
    
    # Standardize
    print("\n5. Standardizing features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Load baseline results for reweighting
    print("\n6. Loading baseline evaluation results...")
    baseline_results_path = cache_path / 'baseline_evaluation_results.csv'
    
    if not baseline_results_path.exists():
        print("ERROR: Baseline results not found. Run train_baseline.py first.")
        sys.exit(1)
    
    baseline_results = pd.read_csv(baseline_results_path)
    print(f"   ✓ Loaded baseline results")
    
    # Compute sample weights
    print("\n7. Computing sample weights based on subgroup FNR...")
    reweighter = SubgroupReweighter(
        strategy='inverse_fnr',
        baseline_metrics=baseline_results,
        alpha=1.0  # Full reweighting
    )
    
    sample_weights = reweighter.compute_sample_weights(
        metadata_train,
        y_train,
        fn_penalty=3.0  # Extra penalty for false negatives
    )
    
    print_weight_statistics(sample_weights, metadata_train, y_train)
    
    # Train fairness-aware models
    print("\n8. Training fairness-aware models...")
    fairness_models = train_fairness_aware_models(
        X_train,
        y_train,
        sample_weights,
        random_state=42
    )
    
    # Load baseline models for comparison
    print("\n9. Loading baseline models...")
    baseline_models_path = cache_path / 'baseline_models.pkl'
    with open(baseline_models_path, 'rb') as f:
        baseline_data = pickle.load(f)
        baseline_models = baseline_data['models']
    
    # Evaluate all models
    print("\n10. Evaluating models...")
    all_results = evaluate_and_compare(
        baseline_models,
        fairness_models,
        scaler,
        X_test,
        y_test,
        metadata_test
    )
    
    # Save results
    results_path = cache_path / 'fairness_aware_evaluation_results.csv'
    all_results.to_csv(results_path, index=False)
    print(f"\n✓ Results saved to {results_path}")
    
    # Compare improvements
    print("\n11. Analyzing improvements...")
    baseline_only = all_results[all_results['model'].str.contains('baseline')]
    fairness_only = all_results[all_results['model'].str.contains('fairness_aware')]
    
    improvements = compare_fnr_improvements(baseline_only, fairness_only)
    
    # Save improvements
    improvements_path = cache_path / 'fairness_improvements.csv'
    improvements.to_csv(improvements_path, index=False)
    print(f"✓ Improvements saved to {improvements_path}")
    
    # Print summary
    print_improvement_summary(improvements)
    
    # Save fairness-aware models
    models_path = cache_path / 'fairness_aware_models.pkl'
    with open(models_path, 'wb') as f:
        pickle.dump({
            'models': fairness_models,
            'scaler': scaler,
            'feature_cols': feature_cols,
            'reweighter': reweighter,
            'sample_weights': sample_weights
        }, f)
    print(f"\n✓ Fairness-aware models saved to {models_path}")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    
    return all_results, improvements


if __name__ == '__main__':
    results, improvements = main()

