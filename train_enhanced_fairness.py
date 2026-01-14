"""
Enhanced Fairness-Aware Training with Targeted Reweighting

Implements aggressive reweighting strategy specifically targeting
the elderly (>65) group's high false negative rate.
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
from train_fairness_aware import (
    train_fairness_aware_models,
    evaluate_and_compare,
    compare_fnr_improvements,
    print_improvement_summary
)


class TargetedSubgroupReweighter(SubgroupReweighter):
    """
    Enhanced reweighter with targeted strategy for high-risk groups.
    """
    
    def compute_sample_weights(
        self,
        metadata: pd.DataFrame,
        y: np.ndarray,
        fn_penalty: float = 3.0,
        elderly_boost: float = 5.0
    ) -> np.ndarray:
        """
        Compute sample weights with special boosting for elderly positive samples.
        
        Args:
            metadata: Metadata with demographic information
            y: Labels
            fn_penalty: Base penalty for positive samples
            elderly_boost: Additional boost for elderly (>65) positive samples
            
        Returns:
            Array of sample weights
        """
        n_samples = len(y)
        weights = np.ones(n_samples)
        
        # Compute base group weights
        group_weights = self.compute_group_weights(metadata, y)
        
        # Assign weights to samples
        for idx in range(n_samples):
            sample_weight = 1.0
            
            # Sex-based weight
            sex = metadata.iloc[idx]['sex']
            if not pd.isna(sex):
                sex_label = 'Male' if sex == 1 else 'Female'
                sex_key = f"Sex_{sex_label}"
                
                if sex_key in group_weights:
                    if isinstance(group_weights[sex_key], dict):
                        label_key = 'positive' if y[idx] == 1 else 'negative'
                        sample_weight *= group_weights[sex_key][label_key]
                    else:
                        sample_weight *= group_weights[sex_key]
            
            # Age-based weight
            age_group = metadata.iloc[idx]['age_group']
            if not pd.isna(age_group):
                age_key = f"Age_{age_group}"
                
                if age_key in group_weights:
                    if isinstance(group_weights[age_key], dict):
                        label_key = 'positive' if y[idx] == 1 else 'negative'
                        sample_weight *= group_weights[age_key][label_key]
                    else:
                        sample_weight *= group_weights[age_key]
                
                # CRITICAL: Aggressive boosting for elderly positive samples
                if age_group == '>65' and y[idx] == 1:
                    sample_weight *= elderly_boost
                    print(f"Boosted elderly positive sample {idx}: weight = {sample_weight:.2f}")
            
            # Base FN penalty for all positive samples
            if y[idx] == 1:
                sample_weight *= fn_penalty
            
            weights[idx] = sample_weight
        
        # Normalize weights
        weights = weights / weights.mean()
        
        return weights


def main():
    """Enhanced fairness-aware training pipeline."""
    
    print("="*80)
    print("ENHANCED FAIRNESS-AWARE TRAINING (Targeted Elderly Reweighting)")
    print("="*80)
    
    # Paths
    data_path = Path('physionet.org/files/ptb-xl/1.0.3/')
    cache_path = Path('data/processed/')
    
    # Load metadata
    print("\n1. Loading metadata...")
    metadata = pd.read_csv(data_path / 'ptbxl_database.csv', index_col='ecg_id')
    metadata = prepare_labels(metadata)
    metadata = prepare_age_groups(metadata)
    
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
    
    metadata_train = features_df.loc[idx_train]
    metadata_test = features_df.loc[idx_test]
    
    # Count elderly positive samples in training
    elderly_pos = np.sum((metadata_train['age_group'] == '>65') & (y_train == 1))
    print(f"\n   ⚠️  Elderly (>65) positive samples in training: {elderly_pos}")
    print(f"   This is a critical minority that needs aggressive reweighting!")
    
    # Standardize
    print("\n5. Standardizing features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Load baseline results
    print("\n6. Loading baseline evaluation results...")
    baseline_results_path = cache_path / 'baseline_evaluation_results.csv'
    baseline_results = pd.read_csv(baseline_results_path)
    
    # Compute enhanced sample weights
    print("\n7. Computing ENHANCED sample weights with elderly boosting...")
    print("   Strategy: Aggressive reweighting for elderly (>65) positive samples")
    
    reweighter = TargetedSubgroupReweighter(
        strategy='inverse_fnr',
        baseline_metrics=baseline_results,
        alpha=1.5  # Stronger base reweighting
    )
    
    sample_weights = reweighter.compute_sample_weights(
        metadata_train,
        y_train,
        fn_penalty=4.0,      # Higher base FN penalty
        elderly_boost=10.0   # Very high boost for elderly
    )
    
    print_weight_statistics(sample_weights, metadata_train, y_train)
    
    # Train enhanced models
    print("\n8. Training ENHANCED fairness-aware models...")
    fairness_models = train_fairness_aware_models(
        X_train,
        y_train,
        sample_weights,
        random_state=42
    )
    
    # Load baseline models
    print("\n9. Loading baseline models...")
    baseline_models_path = cache_path / 'baseline_models.pkl'
    with open(baseline_models_path, 'rb') as f:
        baseline_data = pickle.load(f)
        baseline_models = baseline_data['models']
    
    # Evaluate
    print("\n10. Evaluating enhanced models...")
    all_results = evaluate_and_compare(
        baseline_models,
        fairness_models,
        scaler,
        X_test,
        y_test,
        metadata_test
    )
    
    # Save results
    results_path = cache_path / 'enhanced_fairness_evaluation_results.csv'
    all_results.to_csv(results_path, index=False)
    print(f"\n✓ Results saved to {results_path}")
    
    # Compare improvements
    print("\n11. Analyzing improvements...")
    baseline_only = all_results[all_results['model'].str.contains('baseline')]
    fairness_only = all_results[all_results['model'].str.contains('fairness_aware')]
    
    improvements = compare_fnr_improvements(baseline_only, fairness_only)
    
    improvements_path = cache_path / 'enhanced_fairness_improvements.csv'
    improvements.to_csv(improvements_path, index=False)
    
    print_improvement_summary(improvements)
    
    # Save models
    models_path = cache_path / 'enhanced_fairness_aware_models.pkl'
    with open(models_path, 'wb') as f:
        pickle.dump({
            'models': fairness_models,
            'scaler': scaler,
            'feature_cols': feature_cols,
            'reweighter': reweighter,
            'sample_weights': sample_weights
        }, f)
    print(f"\n✓ Enhanced models saved to {models_path}")
    
    print("\n" + "="*80)
    print("ENHANCED TRAINING COMPLETE!")
    print("="*80)
    
    return all_results, improvements


if __name__ == '__main__':
    results, improvements = main()


