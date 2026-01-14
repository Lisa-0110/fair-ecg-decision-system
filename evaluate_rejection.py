"""
Uncertainty-Based Rejection Evaluation Script

Implements and evaluates uncertainty-based rejection mechanisms
for deferring low-confidence predictions to clinicians.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
from sklearn.model_selection import train_test_split

from src.models.uncertainty_rejection import (
    UncertaintyEstimator,
    RejectionOptimizer,
    UncertaintyRejectPredictor
)
from train_baseline import prepare_labels, prepare_age_groups

sns.set_style("whitegrid")


def plot_confidence_distribution(
    confidence: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    rejection_threshold: float,
    save_path: Path
):
    """Plot confidence distribution for correct vs incorrect predictions."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Determine correct/incorrect
    correct = y_true == y_pred
    
    # Plot 1: Distribution by correctness
    ax1 = axes[0]
    
    ax1.hist(confidence[correct], bins=50, alpha=0.6, label='Correct', 
            color='green', density=True)
    ax1.hist(confidence[~correct], bins=50, alpha=0.6, label='Incorrect', 
            color='red', density=True)
    ax1.axvline(rejection_threshold, color='black', linestyle='--', 
               linewidth=2, label=f'Rejection Threshold ({rejection_threshold:.3f})')
    
    ax1.set_xlabel('Confidence Score', fontweight='bold')
    ax1.set_ylabel('Density', fontweight='bold')
    ax1.set_title('Confidence Distribution: Correct vs Incorrect', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cumulative rejection rate
    ax2 = axes[1]
    
    thresholds = np.linspace(0, 1, 100)
    rejection_rates = []
    accuracy_gains = []
    
    for thresh in thresholds:
        accept_mask = confidence >= thresh
        n_accepted = np.sum(accept_mask)
        rejection_rate = 1 - (n_accepted / len(confidence))
        rejection_rates.append(rejection_rate)
        
        if n_accepted > 0:
            acc_accepted = np.mean((y_true == y_pred)[accept_mask])
            acc_overall = np.mean(y_true == y_pred)
            accuracy_gains.append(acc_accepted - acc_overall)
        else:
            accuracy_gains.append(0)
    
    ax2_twin = ax2.twinx()
    
    line1 = ax2.plot(thresholds, rejection_rates, 'b-', linewidth=2, 
                     label='Rejection Rate')
    line2 = ax2_twin.plot(thresholds, accuracy_gains, 'g-', linewidth=2, 
                          label='Accuracy Gain')
    
    ax2.axvline(rejection_threshold, color='red', linestyle='--', 
               linewidth=2, label='Optimal Threshold')
    
    ax2.set_xlabel('Confidence Threshold', fontweight='bold')
    ax2.set_ylabel('Rejection Rate', fontweight='bold', color='b')
    ax2_twin.set_ylabel('Accuracy Gain', fontweight='bold', color='g')
    ax2.set_title('Rejection-Accuracy Tradeoff', fontweight='bold')
    
    ax2.tick_params(axis='y', labelcolor='b')
    ax2_twin.tick_params(axis='y', labelcolor='g')
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper left')
    
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved plot to {save_path}")
    plt.close()


def plot_rejection_curve(
    results_df: pd.DataFrame,
    save_path: Path
):
    """Plot rejection tradeoff curve."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Accuracy vs Coverage
    ax1 = axes[0]
    
    ax1.scatter(results_df['coverage'], results_df['accuracy_accepted'], 
               s=30, alpha=0.6, color='blue')
    ax1.plot(results_df['coverage'], results_df['accuracy_accepted'], 
            linewidth=2, color='blue', alpha=0.8)
    
    # Mark optimal point
    optimal_idx = results_df['accuracy_accepted'].idxmax()
    optimal_cov = results_df.loc[optimal_idx, 'coverage']
    optimal_acc = results_df.loc[optimal_idx, 'accuracy_accepted']
    
    ax1.scatter([optimal_cov], [optimal_acc], s=200, color='red', 
               marker='*', edgecolor='black', linewidth=2, 
               label='Optimal', zorder=5)
    
    ax1.set_xlabel('Coverage (Fraction Not Rejected)', fontweight='bold')
    ax1.set_ylabel('Accuracy on Accepted Predictions', fontweight='bold')
    ax1.set_title('Accuracy-Coverage Tradeoff', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1.05])
    ax1.set_ylim([0.5, 1.05])
    
    # Plot 2: FNR vs Rejection Rate
    ax2 = axes[1]
    
    ax2.scatter(results_df['rejection_rate'], results_df['fnr_accepted'], 
               s=30, alpha=0.6, color='orange')
    ax2.plot(results_df['rejection_rate'], results_df['fnr_accepted'], 
            linewidth=2, color='orange', alpha=0.8)
    
    ax2.set_xlabel('Rejection Rate', fontweight='bold')
    ax2.set_ylabel('FNR on Accepted Predictions', fontweight='bold')
    ax2.set_title('FNR Reduction via Rejection', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved plot to {save_path}")
    plt.close()


def print_rejection_analysis(
    optimal_metrics: Dict,
    baseline_accuracy: float
):
    """Print detailed rejection analysis."""
    
    print("\n" + "="*80)
    print("REJECTION ANALYSIS")
    print("="*80)
    
    print(f"\nOptimal Rejection Threshold: {optimal_metrics['confidence_threshold']:.4f}")
    print("-" * 80)
    
    print(f"\nCoverage Statistics:")
    print(f"  Total samples:     {int(optimal_metrics['n_total']):>8d}")
    print(f"  Accepted:          {int(optimal_metrics['n_accepted']):>8d} ({optimal_metrics['coverage']*100:>5.1f}%)")
    print(f"  Rejected:          {int(optimal_metrics['n_rejected']):>8d} ({optimal_metrics['rejection_rate']*100:>5.1f}%)")
    
    print(f"\nPerformance on Accepted Predictions:")
    print(f"  Accuracy:          {optimal_metrics['accuracy_accepted']:>8.4f}")
    print(f"  False Negative Rate: {optimal_metrics['fnr_accepted']:>6.4f}")
    print(f"  False Positive Rate: {optimal_metrics['fpr_accepted']:>6.4f}")
    
    if 'accuracy_rejected' in optimal_metrics:
        print(f"\nPerformance on Rejected Predictions (if accepted):")
        print(f"  Accuracy:          {optimal_metrics['accuracy_rejected']:>8.4f}")
        print(f"  Accuracy Gain:     {optimal_metrics['accuracy_gain']:>8.4f}")
    
    print(f"\nComparison to Baseline (No Rejection):")
    print(f"  Baseline Accuracy:         {baseline_accuracy:>6.4f}")
    print(f"  Accepted Accuracy:         {optimal_metrics['accuracy_accepted']:>6.4f}")
    print(f"  Improvement:               {optimal_metrics['accuracy_accepted'] - baseline_accuracy:>+6.4f}")
    print(f"  Cost (rejection rate):     {optimal_metrics['rejection_rate']*100:>5.1f}%")
    
    # Calculate effective accuracy (conservative: rejected = errors)
    effective_acc = optimal_metrics['accuracy_accepted'] * optimal_metrics['coverage']
    print(f"\nEffective Accuracy (conservative):")
    print(f"  With rejection:    {effective_acc:>8.4f}")
    print(f"  Baseline:          {baseline_accuracy:>8.4f}")
    print(f"  Difference:        {effective_acc - baseline_accuracy:>+8.4f}")
    
    print("\n" + "="*80)


def demonstrate_rejection_use_cases(
    predictor: UncertaintyRejectPredictor,
    X_test: np.ndarray,
    metadata_test: pd.DataFrame,
    n_examples: int = 5
):
    """Demonstrate rejection mechanism with real examples."""
    
    print("\n" + "="*80)
    print("REJECTION MECHANISM DEMONSTRATION")
    print("="*80)
    
    # Get predictions with explanations
    explanations = predictor.predict_with_explanation(X_test[:n_examples], metadata_test.head(n_examples))
    
    print("\nExample Predictions:")
    print("-" * 80)
    
    for i, exp in enumerate(explanations):
        print(f"\nPatient {i+1}:")
        if 'age' in exp:
            print(f"  Age: {exp['age']} ({exp.get('age_group', 'N/A')})")
        if 'sex' in exp:
            print(f"  Sex: {exp['sex']}")
        print(f"  Probability: {exp['probability']:.4f}")
        print(f"  Confidence:  {exp['confidence']:.4f}")
        print(f"  Decision:    {exp['decision']}")
        print(f"  Reason:      {exp['reason']}")
        print(f"  Action:      {exp['recommendation']}")
    
    print("\n" + "="*80)


def main():
    """Main evaluation pipeline."""
    
    print("="*80)
    print("UNCERTAINTY-BASED REJECTION EVALUATION")
    print("="*80)
    
    # Paths
    cache_path = Path('data/processed/')
    
    # Load model and data
    print("\n1. Loading model and test data...")
    
    # Use adaptive thresholds if available
    if (cache_path / 'adaptive_thresholds/adaptive_predictor.pkl').exists():
        print("   Using adaptive threshold model")
        with open(cache_path / 'adaptive_thresholds/adaptive_predictor.pkl', 'rb') as f:
            adaptive_predictor = pickle.load(f)
            classification_threshold = adaptive_predictor.group_thresholds['overall']['threshold']
    else:
        classification_threshold = 0.5
    
    # Load fairness-aware model if available
    if (cache_path / 'fairness_aware_models.pkl').exists():
        models_path = cache_path / 'fairness_aware_models.pkl'
        model_name = 'fairness_aware_lr'
        print("   Using fairness-aware Logistic Regression")
    else:
        models_path = cache_path / 'baseline_models.pkl'
        model_name = 'logistic_regression'
        print("   Using baseline Logistic Regression")
    
    with open(models_path, 'rb') as f:
        model_data = pickle.load(f)
        if 'models' in model_data:
            if model_name in model_data['models']:
                model = model_data['models'][model_name]
            else:
                model = list(model_data['models'].values())[0]
        else:
            model = model_data
        scaler = model_data['scaler']
        feature_cols = model_data['feature_cols']
    
    # Load features
    features_df = pd.read_csv(cache_path / 'features_500.csv', index_col='ecg_id')
    
    # Prepare data
    X = features_df[feature_cols].values
    y = features_df['is_normal'].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Same split as training
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, features_df.index.values,
        test_size=0.2,
        stratify=y,
        random_state=42
    )
    
    X_test = scaler.transform(X_test)
    metadata_test = features_df.loc[idx_test]
    
    print(f"   ✓ Loaded model and {len(X_test)} test samples")
    
    # Get baseline predictions
    print("\n2. Computing baseline predictions...")
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= classification_threshold).astype(int)
    baseline_accuracy = np.mean(y_test == y_pred)
    print(f"   Baseline accuracy: {baseline_accuracy:.4f}")
    
    # Estimate uncertainty
    print("\n3. Estimating prediction uncertainty...")
    uncertainty_methods = ['probability', 'entropy', 'margin']
    
    estimators = {}
    for method in uncertainty_methods:
        estimator = UncertaintyEstimator(method=method)
        estimators[method] = estimator
        confidence = estimator.get_confidence(y_prob, classification_threshold)
        print(f"   {method:12s}: mean confidence = {confidence.mean():.4f}")
    
    # Use probability method for main analysis
    estimator = estimators['probability']
    confidence = estimator.get_confidence(y_prob, classification_threshold)
    
    # Optimize rejection threshold
    print("\n4. Optimizing rejection threshold...")
    
    optimizer = RejectionOptimizer(
        target_accuracy=None,  # Will find best tradeoff
        target_coverage=None,
        min_accuracy_gain=0.03
    )
    
    optimal_threshold, optimal_metrics = optimizer.optimize_rejection_threshold(
        y_test,
        y_pred,
        y_prob,
        confidence
    )
    
    print(f"   ✓ Optimal rejection threshold: {optimal_threshold:.4f}")
    print(f"   ✓ Coverage: {optimal_metrics['coverage']*100:.1f}%")
    print(f"   ✓ Accuracy on accepted: {optimal_metrics['accuracy_accepted']:.4f}")
    
    # Create rejection predictor
    print("\n5. Creating rejection predictor...")
    reject_predictor = UncertaintyRejectPredictor(
        model=model,
        uncertainty_estimator=estimator,
        rejection_threshold=optimal_threshold,
        classification_threshold=classification_threshold
    )
    
    # Get rejection statistics
    reject_stats = reject_predictor.get_rejection_statistics(X_test, y_test)
    
    print_rejection_analysis(optimal_metrics, baseline_accuracy)
    
    # Demonstrate use cases
    demonstrate_rejection_use_cases(
        reject_predictor,
        X_test,
        metadata_test,
        n_examples=10
    )
    
    # Analyze full rejection curve
    print("\n6. Analyzing rejection curve...")
    
    thresholds = np.linspace(0, 1, 101)
    rejection_results = []
    
    for thresh in thresholds:
        metrics = optimizer.compute_metrics_with_rejection(
            y_test, y_pred, y_prob, confidence, thresh
        )
        rejection_results.append(metrics)
    
    rejection_df = pd.DataFrame(rejection_results)
    
    # Save results
    print("\n7. Saving results...")
    output_dir = cache_path / 'uncertainty_rejection'
    output_dir.mkdir(exist_ok=True)
    
    # Save rejection curve
    rejection_df.to_csv(output_dir / 'rejection_curve.csv', index=False)
    print(f"   ✓ Saved rejection curve to {output_dir / 'rejection_curve.csv'}")
    
    # Save optimal configuration
    with open(output_dir / 'rejection_predictor.pkl', 'wb') as f:
        pickle.dump(reject_predictor, f)
    print(f"   ✓ Saved rejection predictor to {output_dir / 'rejection_predictor.pkl'}")
    
    # Save optimal metrics
    optimal_df = pd.DataFrame([optimal_metrics])
    optimal_df.to_csv(output_dir / 'optimal_rejection_config.csv', index=False)
    print(f"   ✓ Saved optimal config to {output_dir / 'optimal_rejection_config.csv'}")
    
    # Generate plots
    print("\n8. Generating visualizations...")
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    plot_confidence_distribution(
        confidence,
        y_test,
        y_pred,
        optimal_threshold,
        plots_dir / 'confidence_distribution.png'
    )
    
    plot_rejection_curve(
        rejection_df,
        plots_dir / 'rejection_curve.png'
    )
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    
    print("\nGenerated Files:")
    print(f"  • Rejection curve:    {output_dir / 'rejection_curve.csv'}")
    print(f"  • Optimal config:     {output_dir / 'optimal_rejection_config.csv'}")
    print(f"  • Rejection predictor: {output_dir / 'rejection_predictor.pkl'}")
    print(f"  • Plots:              {plots_dir}")
    
    print("\nUsage Example:")
    print("""
    import pickle
    with open('data/processed/uncertainty_rejection/rejection_predictor.pkl', 'rb') as f:
        predictor = pickle.load(f)
    
    # Make predictions with rejection
    predictions = predictor.predict_with_rejection(X_new)
    # predictions: 0 (negative), 1 (positive), or -1 (refer to clinician)
    
    # Get explanations
    explanations = predictor.predict_with_explanation(X_new, metadata_new)
    for exp in explanations:
        print(f"Decision: {exp['decision']}")
        print(f"Reason: {exp['reason']}")
        print(f"Action: {exp['recommendation']}")
    """)
    
    return reject_predictor, rejection_df, optimal_metrics


if __name__ == '__main__':
    predictor, rejection_df, metrics = main()

