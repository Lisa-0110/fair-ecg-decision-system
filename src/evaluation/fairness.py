"""
Fairness Evaluation Utilities

Provides functions to compute fairness metrics including subgroup performance gaps,
worst-group metrics, and disparity ratios across demographic groups.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def compute_subgroup_performance_gaps(
    results_df: pd.DataFrame,
    metrics: List[str] = ['accuracy', 'fnr', 'fpr', 'ece']
) -> pd.DataFrame:
    """
    Compute performance gaps between subgroups.
    
    Args:
        results_df: DataFrame with evaluation results by model and subgroup
        metrics: List of metrics to compute gaps for
        
    Returns:
        DataFrame with performance gaps for each metric
    """
    gaps = []
    
    for model in results_df['model'].unique():
        model_data = results_df[results_df['model'] == model]
        
        for stratification in ['Sex', 'Age']:
            strat_data = model_data[model_data['stratification'] == stratification]
            
            if len(strat_data) < 2:
                continue
            
            for metric in metrics:
                values = strat_data[metric].values
                groups = strat_data['group'].values
                
                # Remove NaN values
                valid_mask = ~np.isnan(values)
                values = values[valid_mask]
                groups = groups[valid_mask]
                
                if len(values) < 2:
                    continue
                
                # Compute gap statistics
                max_val = np.max(values)
                min_val = np.min(values)
                gap = max_val - min_val
                
                max_group = groups[np.argmax(values)]
                min_group = groups[np.argmin(values)]
                
                gaps.append({
                    'model': model,
                    'stratification': stratification,
                    'metric': metric,
                    'gap': gap,
                    'max_value': max_val,
                    'min_value': min_val,
                    'max_group': max_group,
                    'min_group': min_group,
                    'relative_gap': gap / max_val if max_val > 0 else 0.0
                })
    
    return pd.DataFrame(gaps)


def compute_worst_group_metrics(
    results_df: pd.DataFrame,
    metrics: List[str] = ['accuracy', 'fnr', 'fpr']
) -> pd.DataFrame:
    """
    Compute worst-group performance for each model.
    
    For accuracy/precision, worst = minimum.
    For error rates (FNR, FPR), worst = maximum.
    
    Args:
        results_df: DataFrame with evaluation results
        metrics: List of metrics to analyze
        
    Returns:
        DataFrame with worst-group metrics
    """
    worst_metrics = []
    
    for model in results_df['model'].unique():
        model_data = results_df[results_df['model'] == model]
        
        # Exclude overall performance
        subgroup_data = model_data[model_data['stratification'] != 'Overall']
        
        for metric in metrics:
            values = subgroup_data[metric].values
            groups = subgroup_data['group'].values
            strats = subgroup_data['stratification'].values
            
            # Remove NaN values
            valid_mask = ~np.isnan(values)
            values = values[valid_mask]
            groups = groups[valid_mask]
            strats = strats[valid_mask]
            
            if len(values) == 0:
                continue
            
            # For error rates, worst = max; for accuracy, worst = min
            if metric in ['fnr', 'fpr', 'ece']:
                worst_idx = np.argmax(values)
                worst_val = np.max(values)
                best_val = np.min(values)
            else:
                worst_idx = np.argmin(values)
                worst_val = np.min(values)
                best_val = np.max(values)
            
            worst_metrics.append({
                'model': model,
                'metric': metric,
                'worst_value': worst_val,
                'best_value': best_val,
                'worst_group': groups[worst_idx],
                'stratification': strats[worst_idx],
                'gap_from_best': abs(worst_val - best_val)
            })
    
    return pd.DataFrame(worst_metrics)


def compute_disparity_ratios(
    results_df: pd.DataFrame,
    reference_groups: Optional[Dict[str, str]] = None,
    metrics: List[str] = ['accuracy', 'fnr', 'fpr']
) -> pd.DataFrame:
    """
    Compute disparity ratios relative to reference groups.
    
    Disparity ratio = Group metric / Reference group metric
    
    Args:
        results_df: DataFrame with evaluation results
        reference_groups: Dict mapping stratification to reference group
                         (e.g., {'Sex': 'Male', 'Age': '41-65'})
        metrics: List of metrics to compute ratios for
        
    Returns:
        DataFrame with disparity ratios
    """
    if reference_groups is None:
        reference_groups = {
            'Sex': 'Male',
            'Age': '41-65'
        }
    
    disparities = []
    
    for model in results_df['model'].unique():
        model_data = results_df[results_df['model'] == model]
        
        for stratification, ref_group in reference_groups.items():
            strat_data = model_data[model_data['stratification'] == stratification]
            
            # Get reference group metrics
            ref_data = strat_data[strat_data['group'] == ref_group]
            
            if len(ref_data) == 0:
                continue
            
            for metric in metrics:
                ref_value = ref_data[metric].values[0]
                
                if np.isnan(ref_value) or ref_value == 0:
                    continue
                
                # Compute ratio for each group
                for _, row in strat_data.iterrows():
                    group = row['group']
                    value = row[metric]
                    
                    if np.isnan(value):
                        continue
                    
                    # For error rates, ratio > 1 means worse; for accuracy, ratio < 1 means worse
                    ratio = value / ref_value if ref_value != 0 else np.nan
                    
                    disparities.append({
                        'model': model,
                        'stratification': stratification,
                        'metric': metric,
                        'group': group,
                        'reference_group': ref_group,
                        'group_value': value,
                        'reference_value': ref_value,
                        'disparity_ratio': ratio,
                        'is_disadvantaged': (ratio < 1.0) if metric == 'accuracy' else (ratio > 1.0)
                    })
    
    return pd.DataFrame(disparities)


def compute_demographic_parity_difference(
    results_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute demographic parity difference.
    
    DPD = |P(Y_hat=1|A=a) - P(Y_hat=1|A=b)|
    
    This requires prediction distributions, which we approximate using
    the confusion matrix components.
    
    Args:
        results_df: DataFrame with evaluation results
        
    Returns:
        DataFrame with demographic parity differences
    """
    dpd_results = []
    
    for model in results_df['model'].unique():
        model_data = results_df[results_df['model'] == model]
        
        for stratification in ['Sex', 'Age']:
            strat_data = model_data[model_data['stratification'] == stratification]
            
            if len(strat_data) < 2:
                continue
            
            # Approximate positive prediction rate from TPR and FPR
            # P(Y_hat=1) ≈ TPR * P(Y=1) + FPR * P(Y=0)
            groups = []
            pred_rates = []
            
            for _, row in strat_data.iterrows():
                tpr = row['tpr']
                fpr = row['fpr']
                n_pos = row['n_positive']
                n_neg = row['n_negative']
                n_total = n_pos + n_neg
                
                if n_total == 0 or np.isnan(tpr) or np.isnan(fpr):
                    continue
                
                # Estimate positive prediction rate
                pred_rate = (tpr * n_pos + fpr * n_neg) / n_total
                
                groups.append(row['group'])
                pred_rates.append(pred_rate)
            
            # Compute pairwise differences
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    dpd = abs(pred_rates[i] - pred_rates[j])
                    
                    dpd_results.append({
                        'model': model,
                        'stratification': stratification,
                        'group_a': groups[i],
                        'group_b': groups[j],
                        'pred_rate_a': pred_rates[i],
                        'pred_rate_b': pred_rates[j],
                        'demographic_parity_diff': dpd
                    })
    
    return pd.DataFrame(dpd_results)


def compute_equalized_odds_difference(
    results_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute equalized odds difference.
    
    EOD = max(|TPR_a - TPR_b|, |FPR_a - FPR_b|)
    
    Args:
        results_df: DataFrame with evaluation results
        
    Returns:
        DataFrame with equalized odds differences
    """
    eod_results = []
    
    for model in results_df['model'].unique():
        model_data = results_df[results_df['model'] == model]
        
        for stratification in ['Sex', 'Age']:
            strat_data = model_data[model_data['stratification'] == stratification]
            
            if len(strat_data) < 2:
                continue
            
            groups = strat_data['group'].values
            tprs = strat_data['tpr'].values
            fprs = strat_data['fpr'].values
            
            # Compute pairwise differences
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    if np.isnan(tprs[i]) or np.isnan(tprs[j]) or \
                       np.isnan(fprs[i]) or np.isnan(fprs[j]):
                        continue
                    
                    tpr_diff = abs(tprs[i] - tprs[j])
                    fpr_diff = abs(fprs[i] - fprs[j])
                    eod = max(tpr_diff, fpr_diff)
                    
                    eod_results.append({
                        'model': model,
                        'stratification': stratification,
                        'group_a': groups[i],
                        'group_b': groups[j],
                        'tpr_diff': tpr_diff,
                        'fpr_diff': fpr_diff,
                        'equalized_odds_diff': eod
                    })
    
    return pd.DataFrame(eod_results)


def compute_fairness_summary(
    results_df: pd.DataFrame
) -> Dict[str, pd.DataFrame]:
    """
    Compute comprehensive fairness summary.
    
    Args:
        results_df: DataFrame with evaluation results
        
    Returns:
        Dictionary containing various fairness metrics DataFrames
    """
    summary = {
        'subgroup_gaps': compute_subgroup_performance_gaps(results_df),
        'worst_group': compute_worst_group_metrics(results_df),
        'disparity_ratios': compute_disparity_ratios(results_df),
        'demographic_parity': compute_demographic_parity_difference(results_df),
        'equalized_odds': compute_equalized_odds_difference(results_df)
    }
    
    return summary


def format_fairness_table(
    df: pd.DataFrame,
    title: str = "Fairness Metrics",
    max_rows: Optional[int] = None
) -> str:
    """
    Format fairness metrics as a readable table.
    
    Args:
        df: DataFrame to format
        title: Title for the table
        max_rows: Maximum rows to display
        
    Returns:
        Formatted string table
    """
    if len(df) == 0:
        return f"\n{title}\n{'='*80}\nNo data available.\n"
    
    lines = [
        "",
        "=" * 80,
        title,
        "=" * 80,
        ""
    ]
    
    display_df = df.head(max_rows) if max_rows else df
    
    # Format numeric columns
    formatted_df = display_df.copy()
    for col in formatted_df.select_dtypes(include=[np.number]).columns:
        formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.4f}" if not np.isnan(x) else "N/A")
    
    lines.append(formatted_df.to_string(index=False))
    lines.append("")
    
    return "\n".join(lines)


