"""
Fairness Evaluation Script

Analyzes model fairness across demographic groups and generates
comprehensive reports with tables and visualizations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

from src.evaluation.fairness import (
    compute_fairness_summary,
    format_fairness_table
)
from src.evaluation.visualization import create_fairness_report_plots


def print_section_header(title: str):
    """Print formatted section header."""
    print("\n" + "="*80)
    print(title.center(80))
    print("="*80)


def print_fairness_metrics(fairness_metrics: dict):
    """Print fairness metrics in formatted tables."""
    
    # 1. Subgroup Performance Gaps
    print_section_header("SUBGROUP PERFORMANCE GAPS")
    gaps_df = fairness_metrics['subgroup_gaps']
    
    if len(gaps_df) > 0:
        print("\nPerformance gaps measure the difference between best and worst performing subgroups.")
        print("Smaller gaps indicate better fairness.\n")
        
        for stratification in gaps_df['stratification'].unique():
            strat_data = gaps_df[gaps_df['stratification'] == stratification]
            
            print(f"\n{stratification}-based Gaps:")
            print("-" * 80)
            
            display_cols = ['model', 'metric', 'gap', 'relative_gap', 'max_group', 'min_group']
            display_df = strat_data[display_cols].copy()
            display_df['gap'] = display_df['gap'].apply(lambda x: f"{x:.4f}")
            display_df['relative_gap'] = display_df['relative_gap'].apply(lambda x: f"{x:.4f}")
            
            print(display_df.to_string(index=False))
    else:
        print("\nNo subgroup gap data available.")
    
    # 2. Worst-Group Performance
    print_section_header("WORST-GROUP PERFORMANCE")
    worst_df = fairness_metrics['worst_group']
    
    if len(worst_df) > 0:
        print("\nWorst-group metrics identify the subgroup with poorest performance.")
        print("Key fairness concern: ensuring worst-group performance is acceptable.\n")
        
        for model in worst_df['model'].unique():
            model_data = worst_df[worst_df['model'] == model]
            
            print(f"\n{model.replace('_', ' ').title()}:")
            print("-" * 80)
            
            display_cols = ['metric', 'worst_value', 'worst_group', 'best_value', 'gap_from_best']
            display_df = model_data[display_cols].copy()
            
            for col in ['worst_value', 'best_value', 'gap_from_best']:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
            
            print(display_df.to_string(index=False))
            
            # Highlight concerning metrics
            print("\nKey Concerns:")
            fnr_row = model_data[model_data['metric'] == 'fnr']
            if len(fnr_row) > 0:
                worst_fnr = fnr_row.iloc[0]['worst_value']
                worst_fnr_group = fnr_row.iloc[0]['worst_group']
                print(f"  • Worst FNR: {worst_fnr:.4f} ({worst_fnr_group})")
                if worst_fnr > 0.2:
                    print(f"    WARNING: High false negative rate for {worst_fnr_group}!")
    else:
        print("\nNo worst-group data available.")
    
    # 3. Disparity Ratios
    print_section_header("DISPARITY RATIOS")
    disparity_df = fairness_metrics['disparity_ratios']
    
    if len(disparity_df) > 0:
        print("\nDisparity ratios compare each group to a reference group.")
        print("Ratio = 1.0 indicates parity with reference group.")
        print("For accuracy: ratio < 1.0 indicates disadvantage.")
        print("For error rates (FNR/FPR): ratio > 1.0 indicates disadvantage.\n")
        
        for model in disparity_df['model'].unique():
            model_data = disparity_df[disparity_df['model'] == model]
            
            print(f"\n{model.replace('_', ' ').title()}:")
            print("-" * 80)
            
            # Show only non-reference groups
            non_ref = model_data[model_data['group'] != model_data['reference_group']]
            
            display_cols = ['stratification', 'metric', 'group', 'disparity_ratio', 'is_disadvantaged']
            display_df = non_ref[display_cols].copy()
            display_df['disparity_ratio'] = display_df['disparity_ratio'].apply(lambda x: f"{x:.4f}")
            
            print(display_df.to_string(index=False))
            
            # Count disadvantaged groups
            n_disadvantaged = display_df['is_disadvantaged'].sum()
            n_total = len(display_df)
            print(f"\nDisadvantaged groups: {n_disadvantaged}/{n_total}")
    else:
        print("\nNo disparity ratio data available.")
    
    # 4. Demographic Parity
    print_section_header("DEMOGRAPHIC PARITY DIFFERENCE")
    dpd_df = fairness_metrics['demographic_parity']
    
    if len(dpd_df) > 0:
        print("\nDemographic parity difference measures how positive prediction rates differ.")
        print("Smaller values indicate better demographic parity.\n")
        
        for model in dpd_df['model'].unique():
            model_data = dpd_df[dpd_df['model'] == model]
            
            print(f"\n{model.replace('_', ' ').title()}:")
            print("-" * 80)
            
            display_cols = ['stratification', 'group_a', 'group_b', 'demographic_parity_diff']
            display_df = model_data[display_cols].copy()
            display_df['demographic_parity_diff'] = display_df['demographic_parity_diff'].apply(
                lambda x: f"{x:.4f}"
            )
            
            print(display_df.to_string(index=False))
    else:
        print("\nNo demographic parity data available.")
    
    # 5. Equalized Odds
    print_section_header("EQUALIZED ODDS DIFFERENCE")
    eod_df = fairness_metrics['equalized_odds']
    
    if len(eod_df) > 0:
        print("\nEqualized odds difference measures TPR and FPR differences.")
        print("Smaller values indicate better equalized odds.\n")
        
        for model in eod_df['model'].unique():
            model_data = eod_df[eod_df['model'] == model]
            
            print(f"\n{model.replace('_', ' ').title()}:")
            print("-" * 80)
            
            display_cols = ['stratification', 'group_a', 'group_b', 
                          'tpr_diff', 'fpr_diff', 'equalized_odds_diff']
            display_df = model_data[display_cols].copy()
            
            for col in ['tpr_diff', 'fpr_diff', 'equalized_odds_diff']:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
            
            print(display_df.to_string(index=False))
    else:
        print("\nNo equalized odds data available.")


def generate_fairness_summary(fairness_metrics: dict) -> pd.DataFrame:
    """Generate high-level fairness summary."""
    
    summary_rows = []
    
    # Get unique models
    if len(fairness_metrics['worst_group']) > 0:
        models = fairness_metrics['worst_group']['model'].unique()
    else:
        return pd.DataFrame()
    
    for model in models:
        row = {'model': model}
        
        # Worst-group FNR
        worst_fnr = fairness_metrics['worst_group'][
            (fairness_metrics['worst_group']['model'] == model) & 
            (fairness_metrics['worst_group']['metric'] == 'fnr')
        ]
        if len(worst_fnr) > 0:
            row['worst_group_fnr'] = worst_fnr.iloc[0]['worst_value']
            row['worst_fnr_group'] = worst_fnr.iloc[0]['worst_group']
        
        # Max gap
        gaps = fairness_metrics['subgroup_gaps'][
            fairness_metrics['subgroup_gaps']['model'] == model
        ]
        if len(gaps) > 0:
            row['max_performance_gap'] = gaps['gap'].max()
            max_gap_row = gaps.loc[gaps['gap'].idxmax()]
            row['max_gap_metric'] = max_gap_row['metric']
        
        # Disadvantaged groups count
        disparities = fairness_metrics['disparity_ratios'][
            fairness_metrics['disparity_ratios']['model'] == model
        ]
        if len(disparities) > 0:
            row['n_disadvantaged_groups'] = disparities['is_disadvantaged'].sum()
            row['total_groups_compared'] = len(disparities)
        
        summary_rows.append(row)
    
    return pd.DataFrame(summary_rows)


def main():
    """Main fairness evaluation pipeline."""
    
    print("="*80)
    print("FAIRNESS EVALUATION ANALYSIS".center(80))
    print("="*80)
    
    # Load evaluation results
    print("\n1. Loading evaluation results...")
    results_path = Path('data/processed/baseline_evaluation_results.csv')
    
    if not results_path.exists():
        print(f"ERROR: Results file not found at {results_path}")
        print("Please run train_baseline.py first.")
        sys.exit(1)
    
    results_df = pd.read_csv(results_path)
    print(f"   ✓ Loaded results for {len(results_df)} configurations")
    
    # Compute fairness metrics
    print("\n2. Computing fairness metrics...")
    fairness_metrics = compute_fairness_summary(results_df)
    print("   ✓ Computed:")
    print(f"      - {len(fairness_metrics['subgroup_gaps'])} subgroup gaps")
    print(f"      - {len(fairness_metrics['worst_group'])} worst-group metrics")
    print(f"      - {len(fairness_metrics['disparity_ratios'])} disparity ratios")
    print(f"      - {len(fairness_metrics['demographic_parity'])} demographic parity comparisons")
    print(f"      - {len(fairness_metrics['equalized_odds'])} equalized odds comparisons")
    
    # Print detailed metrics
    print_fairness_metrics(fairness_metrics)
    
    # Generate high-level summary
    print_section_header("FAIRNESS SUMMARY")
    summary_df = generate_fairness_summary(fairness_metrics)
    
    if len(summary_df) > 0:
        print("\nHigh-Level Fairness Indicators:\n")
        
        for col in summary_df.select_dtypes(include=[np.number]).columns:
            if col not in ['n_disadvantaged_groups', 'total_groups_compared']:
                summary_df[col] = summary_df[col].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A")
        
        print(summary_df.to_string(index=False))
    
    # Save metrics to files
    print("\n\n3. Saving fairness metrics...")
    output_dir = Path('data/processed/fairness')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for name, df in fairness_metrics.items():
        output_path = output_dir / f'{name}.csv'
        df.to_csv(output_path, index=False)
        print(f"   ✓ Saved {name} to {output_path}")
    
    # Save summary
    if len(summary_df) > 0:
        summary_path = output_dir / 'fairness_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        print(f"   ✓ Saved summary to {summary_path}")
    
    # Generate visualizations
    print("\n4. Generating fairness visualizations...")
    plots_dir = Path('data/processed/fairness/plots')
    plots = create_fairness_report_plots(results_df, fairness_metrics, plots_dir)
    
    print(f"\n   ✓ Generated {len(plots)} plots in {plots_dir}")
    
    # Final summary
    print_section_header("EVALUATION COMPLETE")
    print("\nGenerated Files:")
    print(f"  • Fairness metrics: {output_dir}")
    print(f"  • Visualizations: {plots_dir}")
    print("\nKey Findings:")
    
    if len(fairness_metrics['worst_group']) > 0:
        # Find worst FNR
        worst_fnr_all = fairness_metrics['worst_group'][
            fairness_metrics['worst_group']['metric'] == 'fnr'
        ]
        if len(worst_fnr_all) > 0:
            max_fnr = worst_fnr_all['worst_value'].max()
            max_fnr_row = worst_fnr_all.loc[worst_fnr_all['worst_value'].idxmax()]
            print(f"  • Worst-group FNR: {max_fnr:.4f} ({max_fnr_row['worst_group']}, {max_fnr_row['model']})")
        
        # Find largest gap
        if len(fairness_metrics['subgroup_gaps']) > 0:
            max_gap = fairness_metrics['subgroup_gaps']['gap'].max()
            max_gap_row = fairness_metrics['subgroup_gaps'].loc[
                fairness_metrics['subgroup_gaps']['gap'].idxmax()
            ]
            print(f"  • Largest performance gap: {max_gap:.4f} ({max_gap_row['metric']}, {max_gap_row['model']})")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()

