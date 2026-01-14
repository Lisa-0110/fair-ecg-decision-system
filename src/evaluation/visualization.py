"""
Fairness Visualization Utilities

Provides functions to create clear visualizations of fairness metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def plot_subgroup_performance(
    results_df: pd.DataFrame,
    metrics: List[str] = ['accuracy', 'fnr', 'fpr'],
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot performance metrics by subgroup.
    
    Args:
        results_df: DataFrame with evaluation results
        metrics: List of metrics to plot
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))
    
    if n_metrics == 1:
        axes = [axes]
    
    # Filter out overall performance
    subgroup_data = results_df[results_df['stratification'] != 'Overall'].copy()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Prepare data for plotting
        plot_data = []
        for _, row in subgroup_data.iterrows():
            plot_data.append({
                'Model': row['model'].replace('_', ' ').title(),
                'Group': f"{row['group']}\n({row['stratification']})",
                'Value': row[metric],
                'Stratification': row['stratification']
            })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create grouped bar plot
        models = plot_df['Model'].unique()
        groups = plot_df['Group'].unique()
        
        x = np.arange(len(groups))
        width = 0.35
        
        for i, model in enumerate(models):
            model_data = plot_df[plot_df['Model'] == model]
            values = [model_data[model_data['Group'] == g]['Value'].values[0] 
                     if len(model_data[model_data['Group'] == g]) > 0 else 0 
                     for g in groups]
            
            ax.bar(x + i * width, values, width, label=model, alpha=0.8)
        
        ax.set_xlabel('Demographic Group', fontsize=11, fontweight='bold')
        ax.set_ylabel(metric.upper(), fontsize=11, fontweight='bold')
        ax.set_title(f'{metric.upper()} by Subgroup', fontsize=12, fontweight='bold')
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(groups, rotation=0, ha='center')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Add reference line
        overall_mean = results_df[results_df['stratification'] == 'Overall'][metric].mean()
        if not np.isnan(overall_mean):
            ax.axhline(y=overall_mean, color='red', linestyle='--', 
                      alpha=0.5, label='Overall Mean')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to {save_path}")
    
    return fig


def plot_performance_gaps(
    gaps_df: pd.DataFrame,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot performance gaps between subgroups.
    
    Args:
        gaps_df: DataFrame with gap metrics
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Absolute gaps
    ax1 = axes[0]
    
    pivot_data = gaps_df.pivot_table(
        values='gap',
        index='metric',
        columns='model',
        aggfunc='mean'
    )
    
    pivot_data.plot(kind='bar', ax=ax1, width=0.7, alpha=0.8)
    ax1.set_xlabel('Metric', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Absolute Gap (Max - Min)', fontsize=11, fontweight='bold')
    ax1.set_title('Subgroup Performance Gaps', fontsize=12, fontweight='bold')
    ax1.legend(title='Model', loc='best')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Plot 2: Relative gaps
    ax2 = axes[1]
    
    pivot_rel = gaps_df.pivot_table(
        values='relative_gap',
        index='metric',
        columns='model',
        aggfunc='mean'
    )
    
    pivot_rel.plot(kind='bar', ax=ax2, width=0.7, alpha=0.8)
    ax2.set_xlabel('Metric', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Relative Gap (Gap / Max)', fontsize=11, fontweight='bold')
    ax2.set_title('Relative Subgroup Performance Gaps', fontsize=12, fontweight='bold')
    ax2.legend(title='Model', loc='best')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to {save_path}")
    
    return fig


def plot_worst_group_comparison(
    worst_df: pd.DataFrame,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot worst-group metrics comparison.
    
    Args:
        worst_df: DataFrame with worst-group metrics
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Prepare data
    plot_data = []
    for _, row in worst_df.iterrows():
        plot_data.append({
            'Model': row['model'].replace('_', ' ').title(),
            'Metric': row['metric'].upper(),
            'Type': 'Worst Group',
            'Value': row['worst_value']
        })
        plot_data.append({
            'Model': row['model'].replace('_', ' ').title(),
            'Metric': row['metric'].upper(),
            'Type': 'Best Group',
            'Value': row['best_value']
        })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create grouped bar plot
    metrics = plot_df['Metric'].unique()
    models = plot_df['Model'].unique()
    
    x = np.arange(len(metrics))
    width = 0.2
    
    for i, model in enumerate(models):
        for j, type_ in enumerate(['Worst Group', 'Best Group']):
            values = []
            for metric in metrics:
                val = plot_df[(plot_df['Model'] == model) & 
                             (plot_df['Metric'] == metric) & 
                             (plot_df['Type'] == type_)]['Value'].values
                values.append(val[0] if len(val) > 0 else 0)
            
            offset = (i * 2 + j - 1.5) * width
            color = 'salmon' if type_ == 'Worst Group' else 'lightgreen'
            label = f"{model} - {type_}" if i == 0 else ""
            
            ax.bar(x + offset, values, width, label=f"{model} - {type_}", 
                  alpha=0.8, color=color, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Metric', fontsize=11, fontweight='bold')
    ax.set_ylabel('Value', fontsize=11, fontweight='bold')
    ax.set_title('Worst-Group vs Best-Group Performance', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to {save_path}")
    
    return fig


def plot_disparity_ratios(
    disparity_df: pd.DataFrame,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot disparity ratios across groups.
    
    Args:
        disparity_df: DataFrame with disparity ratios
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Filter out reference groups for clearer visualization
    non_ref = disparity_df[disparity_df['group'] != disparity_df['reference_group']]
    
    # Plot 1: Sex-based disparities
    ax1 = axes[0]
    sex_data = non_ref[non_ref['stratification'] == 'Sex']
    
    if len(sex_data) > 0:
        pivot_sex = sex_data.pivot_table(
            values='disparity_ratio',
            index=['model', 'metric'],
            columns='group',
            aggfunc='first'
        )
        
        pivot_sex.plot(kind='bar', ax=ax1, width=0.7, alpha=0.8)
        ax1.axhline(y=1.0, color='red', linestyle='--', linewidth=2, 
                   label='Parity (ratio=1.0)')
        ax1.set_xlabel('Model & Metric', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Disparity Ratio', fontsize=11, fontweight='bold')
        ax1.set_title('Sex-Based Disparity Ratios', fontsize=12, fontweight='bold')
        ax1.legend(title='Group', loc='best')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Plot 2: Age-based disparities
    ax2 = axes[1]
    age_data = non_ref[non_ref['stratification'] == 'Age']
    
    if len(age_data) > 0:
        pivot_age = age_data.pivot_table(
            values='disparity_ratio',
            index=['model', 'metric'],
            columns='group',
            aggfunc='first'
        )
        
        pivot_age.plot(kind='bar', ax=ax2, width=0.7, alpha=0.8)
        ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, 
                   label='Parity (ratio=1.0)')
        ax2.set_xlabel('Model & Metric', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Disparity Ratio', fontsize=11, fontweight='bold')
        ax2.set_title('Age-Based Disparity Ratios', fontsize=12, fontweight='bold')
        ax2.legend(title='Group', loc='best')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to {save_path}")
    
    return fig


def plot_fairness_heatmap(
    results_df: pd.DataFrame,
    metric: str = 'fnr',
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot heatmap of metric values across groups.
    
    Args:
        results_df: DataFrame with evaluation results
        metric: Metric to visualize
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure
    """
    subgroup_data = results_df[results_df['stratification'] != 'Overall'].copy()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    models = subgroup_data['model'].unique()
    
    for idx, model in enumerate(models):
        ax = axes[idx]
        model_data = subgroup_data[subgroup_data['model'] == model]
        
        # Create pivot table
        pivot = model_data.pivot_table(
            values=metric,
            index='stratification',
            columns='group',
            aggfunc='first'
        )
        
        # Plot heatmap
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn_r' if metric in ['fnr', 'fpr'] else 'RdYlGn',
                   ax=ax, cbar_kws={'label': metric.upper()}, vmin=0, vmax=1)
        
        ax.set_title(f'{model.replace("_", " ").title()} - {metric.upper()}', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Demographic Group', fontsize=11, fontweight='bold')
        ax.set_ylabel('Stratification', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to {save_path}")
    
    return fig


def create_fairness_report_plots(
    results_df: pd.DataFrame,
    fairness_metrics: Dict[str, pd.DataFrame],
    output_dir: Path
) -> Dict[str, Path]:
    """
    Create all fairness report plots.
    
    Args:
        results_df: DataFrame with evaluation results
        fairness_metrics: Dictionary of fairness metrics
        output_dir: Directory to save plots
        
    Returns:
        Dictionary mapping plot names to file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plots = {}
    
    # 1. Subgroup performance
    print("\n1. Creating subgroup performance plot...")
    plots['subgroup_performance'] = output_dir / 'subgroup_performance.png'
    plot_subgroup_performance(results_df, save_path=plots['subgroup_performance'])
    plt.close()
    
    # 2. Performance gaps
    print("2. Creating performance gaps plot...")
    plots['performance_gaps'] = output_dir / 'performance_gaps.png'
    plot_performance_gaps(fairness_metrics['subgroup_gaps'], 
                         save_path=plots['performance_gaps'])
    plt.close()
    
    # 3. Worst-group comparison
    print("3. Creating worst-group comparison plot...")
    plots['worst_group'] = output_dir / 'worst_group_comparison.png'
    plot_worst_group_comparison(fairness_metrics['worst_group'], 
                                save_path=plots['worst_group'])
    plt.close()
    
    # 4. Disparity ratios
    print("4. Creating disparity ratios plot...")
    plots['disparity_ratios'] = output_dir / 'disparity_ratios.png'
    plot_disparity_ratios(fairness_metrics['disparity_ratios'], 
                         save_path=plots['disparity_ratios'])
    plt.close()
    
    # 5. Heatmaps for key metrics
    for metric in ['fnr', 'fpr', 'accuracy']:
        print(f"5. Creating {metric} heatmap...")
        plots[f'{metric}_heatmap'] = output_dir / f'{metric}_heatmap.png'
        plot_fairness_heatmap(results_df, metric=metric, 
                             save_path=plots[f'{metric}_heatmap'])
        plt.close()
    
    return plots


