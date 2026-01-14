"""
Fairness-Aware Model Implementation

Implements fairness-aware training strategies including subgroup reweighting
to reduce performance disparities across demographic groups.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from sklearn.base import BaseEstimator
import warnings
warnings.filterwarnings('ignore')


class SubgroupReweighter:
    """
    Implements subgroup reweighting strategies to reduce disparities.
    
    Assigns higher weights to samples from disadvantaged groups to
    improve their performance, particularly reducing false negative rates.
    """
    
    def __init__(
        self,
        strategy: str = 'inverse_fnr',
        baseline_metrics: Optional[pd.DataFrame] = None,
        alpha: float = 1.0
    ):
        """
        Initialize reweighter.
        
        Args:
            strategy: Reweighting strategy
                - 'inverse_fnr': Weight inversely proportional to group FNR
                - 'worst_group': Give extra weight to worst-performing groups
                - 'balanced': Balance positive/negative samples per group
                - 'hybrid': Combine multiple strategies
            baseline_metrics: DataFrame with baseline performance by group
            alpha: Strength of reweighting (0=no reweighting, 1=full)
        """
        self.strategy = strategy
        self.baseline_metrics = baseline_metrics
        self.alpha = alpha
        self.weights_map = {}
        
    def compute_group_weights(
        self,
        metadata: pd.DataFrame,
        y: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute weight multipliers for each demographic group.
        
        Args:
            metadata: Metadata with demographic information
            y: Labels
            
        Returns:
            Dictionary mapping group identifiers to weight multipliers
        """
        weights = {}
        
        if self.strategy == 'inverse_fnr' and self.baseline_metrics is not None:
            # Weight inversely proportional to FNR
            for _, row in self.baseline_metrics.iterrows():
                if row['stratification'] in ['Sex', 'Age']:
                    group_key = f"{row['stratification']}_{row['group']}"
                    fnr = row['fnr']
                    
                    # Higher FNR = higher weight
                    # Add small epsilon to avoid division by zero
                    if not np.isnan(fnr):
                        # Weight proportional to FNR (groups with high FNR get more weight)
                        weights[group_key] = 1.0 + self.alpha * fnr
                    else:
                        weights[group_key] = 1.0
        
        elif self.strategy == 'worst_group':
            # Give extra weight to worst-performing groups
            if self.baseline_metrics is not None:
                # Find worst FNR
                worst_fnr = self.baseline_metrics['fnr'].max()
                
                for _, row in self.baseline_metrics.iterrows():
                    if row['stratification'] in ['Sex', 'Age']:
                        group_key = f"{row['stratification']}_{row['group']}"
                        fnr = row['fnr']
                        
                        if not np.isnan(fnr):
                            # Groups closer to worst FNR get higher weight
                            weights[group_key] = 1.0 + self.alpha * (fnr / worst_fnr)
                        else:
                            weights[group_key] = 1.0
        
        elif self.strategy == 'balanced':
            # Balance positive and negative samples within each group
            for sex in metadata['sex'].unique():
                if pd.isna(sex):
                    continue
                    
                sex_mask = metadata['sex'] == sex
                group_key = f"Sex_{'Male' if sex == 1 else 'Female'}"
                
                # Balance within group
                n_pos = np.sum(y[sex_mask] == 1)
                n_neg = np.sum(y[sex_mask] == 0)
                
                if n_pos > 0 and n_neg > 0:
                    # Higher weight for minority class
                    pos_weight = n_neg / (n_pos + n_neg)
                    weights[group_key] = {'positive': pos_weight, 'negative': 1 - pos_weight}
                else:
                    weights[group_key] = {'positive': 0.5, 'negative': 0.5}
            
            # Same for age groups
            for age_group in ['<=40', '41-65', '>65']:
                age_mask = metadata['age_group'] == age_group
                group_key = f"Age_{age_group}"
                
                n_pos = np.sum(y[age_mask] == 1)
                n_neg = np.sum(y[age_mask] == 0)
                
                if n_pos > 0 and n_neg > 0:
                    pos_weight = n_neg / (n_pos + n_neg)
                    weights[group_key] = {'positive': pos_weight, 'negative': 1 - pos_weight}
                else:
                    weights[group_key] = {'positive': 0.5, 'negative': 0.5}
        
        elif self.strategy == 'hybrid':
            # Combine inverse_fnr and balanced
            weights = self.compute_group_weights(metadata, y)
            
        return weights
    
    def compute_sample_weights(
        self,
        metadata: pd.DataFrame,
        y: np.ndarray,
        fn_penalty: float = 2.0
    ) -> np.ndarray:
        """
        Compute sample-level weights based on demographics and label.
        
        Args:
            metadata: Metadata with demographic information
            y: Labels
            fn_penalty: Extra penalty weight for positive samples (to reduce FN)
            
        Returns:
            Array of sample weights
        """
        n_samples = len(y)
        weights = np.ones(n_samples)
        
        # Compute group weights
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
                        # Balanced strategy
                        label_key = 'positive' if y[idx] == 1 else 'negative'
                        sample_weight *= group_weights[sex_key][label_key]
                    else:
                        # Other strategies
                        sample_weight *= group_weights[sex_key]
            
            # Age-based weight
            age_group = metadata.iloc[idx]['age_group']
            if not pd.isna(age_group):
                age_key = f"Age_{age_group}"
                
                if age_key in group_weights:
                    if isinstance(group_weights[age_key], dict):
                        # Balanced strategy
                        label_key = 'positive' if y[idx] == 1 else 'negative'
                        sample_weight *= group_weights[age_key][label_key]
                    else:
                        # Other strategies
                        sample_weight *= group_weights[age_key]
            
            # Extra penalty for positive samples to reduce FN
            if y[idx] == 1:
                sample_weight *= fn_penalty
            
            weights[idx] = sample_weight
        
        # Normalize weights to have mean of 1
        weights = weights / weights.mean()
        
        return weights
    
    def get_class_weights(
        self,
        y: np.ndarray,
        fn_fp_ratio: float = 2.0
    ) -> Dict[int, float]:
        """
        Compute class weights with FN penalty.
        
        Args:
            y: Labels
            fn_fp_ratio: Ratio of FN cost to FP cost
            
        Returns:
            Dictionary mapping class to weight
        """
        n_samples = len(y)
        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == 0)
        
        if n_pos == 0 or n_neg == 0:
            return {0: 1.0, 1: 1.0}
        
        # Base balanced weights
        weight_neg = n_samples / (2 * n_neg)
        weight_pos = n_samples / (2 * n_pos)
        
        # Apply FN penalty (increase positive class weight)
        weight_pos *= fn_fp_ratio
        
        return {0: weight_neg, 1: weight_pos}


def print_weight_statistics(
    weights: np.ndarray,
    metadata: pd.DataFrame,
    y: np.ndarray
):
    """
    Print statistics about computed weights.
    
    Args:
        weights: Sample weights
        metadata: Metadata
        y: Labels
    """
    print("\nSample Weight Statistics:")
    print("=" * 60)
    
    print(f"\nOverall:")
    print(f"  Mean weight: {weights.mean():.4f}")
    print(f"  Std weight:  {weights.std():.4f}")
    print(f"  Min weight:  {weights.min():.4f}")
    print(f"  Max weight:  {weights.max():.4f}")
    
    print(f"\nBy Label:")
    for label in [0, 1]:
        label_weights = weights[y == label]
        label_name = "Positive" if label == 1 else "Negative"
        print(f"  {label_name}:")
        print(f"    Mean: {label_weights.mean():.4f}")
        print(f"    Std:  {label_weights.std():.4f}")
    
    print(f"\nBy Sex:")
    for sex in metadata['sex'].unique():
        if pd.isna(sex):
            continue
        sex_mask = metadata['sex'] == sex
        sex_weights = weights[sex_mask]
        sex_label = 'Male' if sex == 1 else 'Female'
        print(f"  {sex_label}:")
        print(f"    Mean: {sex_weights.mean():.4f}")
        print(f"    Positive samples: {np.sum((metadata['sex'] == sex) & (y == 1))}")
    
    print(f"\nBy Age Group:")
    for age_group in ['<=40', '41-65', '>65']:
        age_mask = metadata['age_group'] == age_group
        if np.sum(age_mask) > 0:
            age_weights = weights[age_mask]
            print(f"  {age_group}:")
            print(f"    Mean: {age_weights.mean():.4f}")
            print(f"    Positive samples: {np.sum((metadata['age_group'] == age_group) & (y == 1))}")
    
    # Highlight highest weighted groups
    print(f"\nHighest Weighted Groups:")
    for age_group in ['<=40', '41-65', '>65']:
        for sex in [0, 1]:
            mask = (metadata['age_group'] == age_group) & (metadata['sex'] == sex)
            if np.sum(mask) > 0:
                group_mean = weights[mask].mean()
                sex_label = 'Male' if sex == 1 else 'Female'
                print(f"  {age_group}, {sex_label}: {group_mean:.4f}")


