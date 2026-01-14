"""
Threshold Optimization Script

Optimizes and evaluates adaptive decision thresholds for demographic subgroups
to minimize false negatives while maintaining reasonable overall performance.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import sys
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.adaptive_thresholds import (
    AdaptiveThresholdOptimizer,
    AdaptiveThresholdPredictor,
    compare_thresholds
)
from train_baseline import (
    prepare_labels,
    prepare_age_groups,
    load_or_extract_features,
    compute_metrics
)

sns.set_style("whitegrid")


def print_threshold_summary(group_thresholds: dict):
    """Print summary of optimized thresholds."""
    
    print("\n" + "="*80)
    print("OPTIMIZED THRESHOLDS SUMMARY")
    print("="*80)
    
    print("\nThreshold by Demographic Group:")
    print("-" * 80)
    print(f"{'Group':<25} {'Threshold':>12} {'FNR':>10} {'FPR':>10} {'F1':>10} {'N+':>8}")
    print("-" * 80)
    
    for group_key, group_data in sorted(group_thresholds.items()):
        threshold = group_data['threshold']
        metrics = group_data['metrics']
        n_pos = group_data['n_positive']
        
        # Format group name
        if group_key == 'overall':
            group_name = 'Overall (baseline)'
        elif group_key.startswith('sex_'):
            group_name = f"Sex: {group_key.split('_')[1].capitalize()}"
        elif group_key.startswith('age_'):
            group_name = f"Age: {group_key.split('_', 1)[1]}"
        else:
            group_name = group_key
        
        print(f"{group_name:<25} {threshold:>12.4f} {metrics['fnr']:>10.4f} "
              f"{metrics['fpr']:>10.4f} {metrics['f1']:>10.4f} {n_pos:>8d}")
    
    print("-" * 80)
    
    # Highlight key findings
    print("\nKey Observations:")
    
    # Find groups with lowest thresholds (most sensitive)
    low_threshold_groups = sorted(
        [(k, v['threshold']) for k, v in group_thresholds.items() if k != 'overall'],
        key=lambda x: x[1]
    )[:3]
    
    print("\n  Most Sensitive Thresholds (Lower = More Sensitive):")
    for group, thresh in low_threshold_groups:
        fnr = group_thresholds[group]['metrics']['fnr']
        print(f"    • {group}: {thresh:.4f} (FNR={fnr:.4f})")
    
    # Check elderly threshold
    elderly_key = 'age_>65'
    if elderly_key in group_thresholds:
        elderly_data = group_thresholds[elderly_key]
        print(f"\n  Elderly (>65) Threshold: {elderly_data['threshold']:.4f}")
        print(f"    FNR: {elderly_data['metrics']['fnr']:.4f}")
        print(f"    FPR: {elderly_data['metrics']['fpr']:.4f}")
        print(f"    Based on {elderly_data['n_positive']} positive samples")


def plot_threshold_comparison(
    comparison_df: pd.DataFrame,
    save_path: Path
):
    """Plot comparison of fixed vs adaptive thresholds."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    metrics_to_plot = ['fnr', 'fpr', 'accuracy']
    titles = ['False Negative Rate', 'False Positive Rate', 'Accuracy']
    
    for idx, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
        ax = axes[idx]
        
        # Prepare data
        plot_data = comparison_df.pivot(
            index='group',
            columns='method',
            values=metric
        )
        
        # Plot
        x = np.arange(len(plot_data))
        width = 0.35
        
        ax.bar(x - width/2, plot_data['Fixed'], width, 
               label='Fixed (0.5)', alpha=0.8, color='coral')
        ax.bar(x + width/2, plot_data['Adaptive'], width,
               label='Adaptive', alpha=0.8, color='skyblue')
        
        ax.set_xlabel('Group', fontweight='bold')
        ax.set_ylabel(title, fontweight='bold')
        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(plot_data.index, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add reference line for FNR
        if metric == 'fnr':
            ax.axhline(y=0.5, color='red', linestyle='--', 
                      linewidth=2, label='50% (Unacceptable)', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved plot to {save_path}")
    plt.close()


def plot_threshold_distribution(
    group_thresholds: dict,
    save_path: Path
):
    """Plot distribution of thresholds across groups."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Extract data
    groups = []
    thresholds = []
    fnrs = []
    colors = []
    
    for key, data in group_thresholds.items():
        if key == 'overall':
            continue
        
        groups.append(key)
        thresholds.append(data['threshold'])
        fnrs.append(data['metrics']['fnr'])
        
        # Color by demographic type
        if key.startswith('sex_'):
            colors.append('skyblue')
        elif key.startswith('age_'):
            if '>65' in key:
                colors.append('red')
            else:
                colors.append('lightgreen')
        else:
            colors.append('gray')
    
    # Plot 1: Thresholds by group
    x = np.arange(len(groups))
    ax1.barh(x, thresholds, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_yticks(x)
    ax1.set_yticklabels(groups)
    ax1.set_xlabel('Threshold', fontweight='bold')
    ax1.set_title('Optimized Thresholds by Group', fontweight='bold')
    ax1.axvline(x=0.5, color='red', linestyle='--', label='Standard (0.5)')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Plot 2: Threshold vs FNR tradeoff
    for i, (thresh, fnr, group) in enumerate(zip(thresholds, fnrs, groups)):
        ax2.scatter(thresh, fnr, s=150, color=colors[i], alpha=0.7, 
                   edgecolor='black', linewidth=1.5)
        
        # Label high-risk groups
        if '>65' in group or fnr > 0.3:
            ax2.annotate(group, (thresh, fnr), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.7)
    
    ax2.set_xlabel('Threshold', fontweight='bold')
    ax2.set_ylabel('False Negative Rate', fontweight='bold')
    ax2.set_title('Threshold vs FNR Tradeoff', fontweight='bold')
    ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='FNR=0.5')
    ax2.axvline(x=0.5, color='orange', linestyle='--', alpha=0.5, label='Thresh=0.5')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved plot to {save_path}")
    plt.close()


def print_improvement_summary(comparison_df: pd.DataFrame):
    """Print summary of improvements from adaptive thresholds."""
    
    print("\n" + "="*80)
    print("ADAPTIVE THRESHOLD IMPROVEMENTS")
    print("="*80)
    
    # Calculate improvements
    pivot_fnr = comparison_df.pivot(index='group', columns='method', values='fnr')
    pivot_fpr = comparison_df.pivot(index='group', columns='method', values='fpr')
    pivot_acc = comparison_df.pivot(index='group', columns='method', values='accuracy')
    
    improvements = pd.DataFrame({
        'group': pivot_fnr.index,
        'fnr_fixed': pivot_fnr['Fixed'],
        'fnr_adaptive': pivot_fnr['Adaptive'],
        'fnr_improvement': pivot_fnr['Fixed'] - pivot_fnr['Adaptive'],
        'fnr_reduction_pct': ((pivot_fnr['Fixed'] - pivot_fnr['Adaptive']) / pivot_fnr['Fixed'] * 100),
        'fpr_change': pivot_fpr['Adaptive'] - pivot_fpr['Fixed'],
        'accuracy_change': pivot_acc['Adaptive'] - pivot_acc['Fixed']
    })
    
    print("\nFalse Negative Rate Improvements:")
    print("-" * 80)
    print(f"{'Group':<25} {'Fixed FNR':>12} {'Adaptive FNR':>15} {'Improvement':>12} {'% Reduction':>12}")
    print("-" * 80)
    
    for _, row in improvements.iterrows():
        print(f"{row['group']:<25} {row['fnr_fixed']:>12.4f} {row['fnr_adaptive']:>15.4f} "
              f"{row['fnr_improvement']:>12.4f} {row['fnr_reduction_pct']:>11.1f}%")
    
    print("-" * 80)
    
    # Overall statistics
    print("\nOverall Statistics:")
    print(f"  Average FNR reduction: {improvements['fnr_improvement'].mean():.4f} "
          f"({improvements['fnr_reduction_pct'].mean():.1f}%)")
    print(f"  Average FPR change: {improvements['fpr_change'].mean():+.4f}")
    print(f"  Average accuracy change: {improvements['accuracy_change'].mean():+.4f}")
    
    # Highlight biggest improvements
    print("\nBiggest FNR Improvements:")
    top_improvements = improvements.nlargest(3, 'fnr_improvement')
    for _, row in top_improvements.iterrows():
        print(f"  • {row['group']}: {row['fnr_fixed']:.4f} → {row['fnr_adaptive']:.4f} "
              f"(-{row['fnr_reduction_pct']:.1f}%)")
    
    # Check elderly specifically
    elderly_rows = improvements[improvements['group'].str.contains('>65')]
    if len(elderly_rows) > 0:
        elderly = elderly_rows.iloc[0]
        print(f"\nElderly (>65) Performance:")
        print(f"  FNR: {elderly['fnr_fixed']:.4f} → {elderly['fnr_adaptive']:.4f} "
              f"({elderly['fnr_reduction_pct']:+.1f}%)")
        print(f"  FPR change: {elderly['fpr_change']:+.4f}")
        print(f"  Accuracy change: {elderly['accuracy_change']:+.4f}")


def main():
    """Main threshold optimization pipeline."""
    
    print("="*80)
    print("ADAPTIVE THRESHOLD OPTIMIZATION")
    print("="*80)
    
    # Paths
    cache_path = Path('data/processed/')
    
    # Load models and data
    print("\n1. Loading models and test data...")
    
    # Use fairness-aware model if available, else baseline
    if (cache_path / 'fairness_aware_models.pkl').exists():
        models_path = cache_path / 'fairness_aware_models.pkl'
        model_type = 'fairness_aware'
        print("   Using fairness-aware models")
    else:
        models_path = cache_path / 'baseline_models.pkl'
        model_type = 'baseline'
        print("   Using baseline models")
    
    with open(models_path, 'rb') as f:
        model_data = pickle.load(f)
        if model_type == 'fairness_aware':
            model = model_data['models']['fairness_aware_lr']
        else:
            model = model_data['models']['logistic_regression']
        scaler = model_data['scaler']
        feature_cols = model_data['feature_cols']
    
    # Load features and metadata
    features_df = pd.read_csv(cache_path / 'features_500.csv', index_col='ecg_id')
    
    # Prepare data
    X = features_df[feature_cols].values
    y = features_df['is_normal'].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Use same split as training
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, features_df.index.values,
        test_size=0.2,
        stratify=y,
        random_state=42
    )
    
    X_test = scaler.transform(X_test)
    metadata_test = features_df.loc[idx_test]
    
    print(f"   ✓ Loaded model and {len(X_test)} test samples")
    
    # Get predictions
    print("\n2. Generating probability predictions...")
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Optimize thresholds for each objective
    objectives = [
        ('minimize_fnr', 'Minimize FNR (Maximize Sensitivity)'),
        ('maximize_f1', 'Maximize F1 Score'),
        ('youden', "Maximize Youden's J Statistic")
    ]
    
    all_optimized_thresholds = {}
    
    for objective, description in objectives:
        print(f"\n3. Optimizing thresholds: {description}...")
        
        optimizer = AdaptiveThresholdOptimizer(
            objective=objective,
            max_fpr=0.4 if objective == 'minimize_fnr' else None,  # Constraint for FNR minimization
            fn_cost=3.0,  # FN is 3x worse than FP
            fp_cost=1.0
        )
        
        group_thresholds = optimizer.optimize_group_thresholds(
            y_test,
            y_prob,
            metadata_test,
            groups=['sex', 'age_group']
        )
        
        all_optimized_thresholds[objective] = group_thresholds
        
        print(f"   ✓ Optimized {len(group_thresholds)} thresholds")
    
    # Use minimize_fnr for main analysis (most relevant for clinical safety)
    optimal_thresholds = all_optimized_thresholds['minimize_fnr']
    
    print_threshold_summary(optimal_thresholds)
    
    # Compare fixed vs adaptive
    print("\n4. Comparing fixed vs adaptive thresholds...")
    comparison = compare_thresholds(
        y_test,
        y_prob,
        metadata_test,
        fixed_threshold=0.5,
        adaptive_thresholds=optimal_thresholds
    )
    
    print_improvement_summary(comparison)
    
    # Save results
    print("\n5. Saving results...")
    output_dir = cache_path / 'adaptive_thresholds'
    output_dir.mkdir(exist_ok=True)
    
    # Save thresholds
    threshold_records = []
    for objective, thresholds in all_optimized_thresholds.items():
        for group_key, group_data in thresholds.items():
            record = {
                'objective': objective,
                'group': group_key,
                'threshold': group_data['threshold'],
                'n_samples': group_data['n_samples'],
                'n_positive': group_data['n_positive']
            }
            record.update({f"metric_{k}": v for k, v in group_data['metrics'].items()})
            threshold_records.append(record)
    
    thresholds_df = pd.DataFrame(threshold_records)
    thresholds_df.to_csv(output_dir / 'optimized_thresholds.csv', index=False)
    print(f"   ✓ Saved thresholds to {output_dir / 'optimized_thresholds.csv'}")
    
    # Save comparison
    comparison.to_csv(output_dir / 'threshold_comparison.csv', index=False)
    print(f"   ✓ Saved comparison to {output_dir / 'threshold_comparison.csv'}")
    
    # Save adaptive predictor
    predictor = AdaptiveThresholdPredictor(
        optimal_thresholds,
        default_threshold=0.5,
        priority_order=['age_group', 'sex']  # Age first (most important)
    )
    
    with open(output_dir / 'adaptive_predictor.pkl', 'wb') as f:
        pickle.dump(predictor, f)
    print(f"   ✓ Saved adaptive predictor to {output_dir / 'adaptive_predictor.pkl'}")
    
    # Generate plots
    print("\n6. Generating visualizations...")
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    plot_threshold_comparison(
        comparison,
        plots_dir / 'threshold_comparison.png'
    )
    
    plot_threshold_distribution(
        optimal_thresholds,
        plots_dir / 'threshold_distribution.png'
    )
    
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE!")
    print("="*80)
    
    print("\nGenerated Files:")
    print(f"  • Thresholds: {output_dir / 'optimized_thresholds.csv'}")
    print(f"  • Comparison: {output_dir / 'threshold_comparison.csv'}")
    print(f"  • Predictor:  {output_dir / 'adaptive_predictor.pkl'}")
    print(f"  • Plots:      {plots_dir}")
    
    print("\nUsage Example:")
    print("""
    import pickle
    with open('data/processed/adaptive_thresholds/adaptive_predictor.pkl', 'rb') as f:
        predictor = pickle.load(f)
    
    # For a 70-year-old male patient
    threshold = predictor.get_threshold(sex=1, age_group='>65')
    prediction, thresh, explanation = predictor.predict_with_explanation(
        y_prob=0.35, sex=1, age_group='>65'
    )
    print(f"Prediction: {prediction}, Threshold: {thresh}, {explanation}")
    """)
    
    return optimal_thresholds, comparison


if __name__ == '__main__':
    thresholds, comparison = main()


