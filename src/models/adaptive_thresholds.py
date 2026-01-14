"""
Adaptive Decision Thresholds Module

Implements demographic-aware threshold optimization to minimize false negatives
for high-risk subgroups while maintaining overall model performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from sklearn.metrics import confusion_matrix, roc_curve, auc
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')


class AdaptiveThresholdOptimizer:
    """
    Optimizes classification thresholds for different demographic groups.
    
    Supports multiple optimization objectives:
    - Minimize FNR (maximize sensitivity)
    - Maximize F1 score
    - Constrained optimization (FNR ≤ target)
    - Cost-sensitive optimization
    """
    
    def __init__(
        self,
        objective: str = 'minimize_fnr',
        max_fpr: Optional[float] = None,
        fn_cost: float = 2.0,
        fp_cost: float = 1.0
    ):
        """
        Initialize threshold optimizer.
        
        Args:
            objective: Optimization objective
                - 'minimize_fnr': Minimize false negative rate
                - 'maximize_f1': Maximize F1 score
                - 'youden': Maximize Youden's J statistic (TPR - FPR)
                - 'cost': Minimize weighted cost
            max_fpr: Maximum acceptable FPR (constraint)
            fn_cost: Cost of false negative (relative to true positive)
            fp_cost: Cost of false positive (relative to true negative)
        """
        self.objective = objective
        self.max_fpr = max_fpr
        self.fn_cost = fn_cost
        self.fp_cost = fp_cost
        self.group_thresholds = {}
    
    def _compute_metrics_at_threshold(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: float
    ) -> Dict[str, float]:
        """
        Compute classification metrics at a specific threshold.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            threshold: Classification threshold
            
        Returns:
            Dictionary of metrics
        """
        y_pred = (y_prob >= threshold).astype(int)
        
        if len(np.unique(y_true)) < 2:
            return {
                'fnr': 0.0,
                'fpr': 0.0,
                'tpr': 1.0,
                'tnr': 1.0,
                'f1': 1.0,
                'precision': 1.0,
                'cost': 0.0
            }
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Rates
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # Precision and F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tpr
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Cost
        cost = (fn * self.fn_cost + fp * self.fp_cost) / len(y_true)
        
        return {
            'fnr': fnr,
            'fpr': fpr,
            'tpr': tpr,
            'tnr': tnr,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'cost': cost,
            'youden': tpr - fpr
        }
    
    def _objective_function(
        self,
        threshold: float,
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> float:
        """
        Objective function to minimize.
        
        Args:
            threshold: Classification threshold
            y_true: True labels
            y_prob: Predicted probabilities
            
        Returns:
            Objective value (lower is better)
        """
        metrics = self._compute_metrics_at_threshold(y_true, y_prob, threshold)
        
        # Check FPR constraint
        if self.max_fpr is not None and metrics['fpr'] > self.max_fpr:
            return np.inf  # Infeasible
        
        # Return objective based on strategy
        if self.objective == 'minimize_fnr':
            return metrics['fnr']
        elif self.objective == 'maximize_f1':
            return -metrics['f1']  # Negative for minimization
        elif self.objective == 'youden':
            return -metrics['youden']
        elif self.objective == 'cost':
            return metrics['cost']
        else:
            raise ValueError(f"Unknown objective: {self.objective}")
    
    def optimize_threshold(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        search_range: Tuple[float, float] = (0.0, 1.0)
    ) -> Tuple[float, Dict[str, float]]:
        """
        Find optimal threshold for a single group.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            search_range: Range to search for threshold
            
        Returns:
            Tuple of (optimal_threshold, metrics_at_threshold)
        """
        # Grid search for robust optimization
        thresholds = np.linspace(search_range[0], search_range[1], 201)
        objectives = []
        
        for t in thresholds:
            obj_val = self._objective_function(t, y_true, y_prob)
            objectives.append(obj_val)
        
        # Find best threshold
        valid_indices = np.where(np.isfinite(objectives))[0]
        
        if len(valid_indices) == 0:
            # No valid threshold found, use default
            optimal_threshold = 0.5
        else:
            best_idx = valid_indices[np.argmin(np.array(objectives)[valid_indices])]
            optimal_threshold = thresholds[best_idx]
        
        # Compute metrics at optimal threshold
        metrics = self._compute_metrics_at_threshold(y_true, y_prob, optimal_threshold)
        
        return optimal_threshold, metrics
    
    def optimize_group_thresholds(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        metadata: pd.DataFrame,
        groups: List[str] = ['sex', 'age_group']
    ) -> Dict[str, Dict]:
        """
        Optimize thresholds for each demographic group.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            metadata: Metadata with demographic information
            groups: List of grouping variables
            
        Returns:
            Dictionary mapping group identifiers to {threshold, metrics}
        """
        self.group_thresholds = {}
        
        # Overall threshold
        overall_threshold, overall_metrics = self.optimize_threshold(y_true, y_prob)
        self.group_thresholds['overall'] = {
            'threshold': overall_threshold,
            'metrics': overall_metrics,
            'n_samples': len(y_true),
            'n_positive': int(np.sum(y_true == 1))
        }
        
        # Group-specific thresholds
        for group_var in groups:
            if group_var not in metadata.columns:
                continue
            
            for group_val in metadata[group_var].unique():
                if pd.isna(group_val):
                    continue
                
                # Get samples for this group
                mask = metadata[group_var] == group_val
                
                if np.sum(mask) < 10:  # Skip if too few samples
                    continue
                
                y_group = y_true[mask]
                prob_group = y_prob[mask]
                
                # Skip if no positive samples
                if np.sum(y_group == 1) == 0:
                    continue
                
                # Optimize threshold for this group
                group_threshold, group_metrics = self.optimize_threshold(
                    y_group,
                    prob_group
                )
                
                # Create group key
                if group_var == 'sex':
                    group_key = f"sex_{'male' if group_val == 1 else 'female'}"
                elif group_var == 'age_group':
                    group_key = f"age_{group_val}"
                else:
                    group_key = f"{group_var}_{group_val}"
                
                self.group_thresholds[group_key] = {
                    'threshold': group_threshold,
                    'metrics': group_metrics,
                    'n_samples': int(np.sum(mask)),
                    'n_positive': int(np.sum(y_group == 1))
                }
        
        return self.group_thresholds


class AdaptiveThresholdPredictor:
    """
    Applies adaptive thresholds based on patient demographics.
    """
    
    def __init__(
        self,
        group_thresholds: Dict[str, Dict],
        default_threshold: float = 0.5,
        priority_order: List[str] = ['age_group', 'sex']
    ):
        """
        Initialize adaptive threshold predictor.
        
        Args:
            group_thresholds: Dictionary of group-specific thresholds
            default_threshold: Fallback threshold
            priority_order: Order of demographic factors to consider
        """
        self.group_thresholds = group_thresholds
        self.default_threshold = default_threshold
        self.priority_order = priority_order
    
    def get_threshold(
        self,
        sex: Optional[int] = None,
        age_group: Optional[str] = None
    ) -> float:
        """
        Get adaptive threshold for a patient.
        
        Args:
            sex: Patient sex (0=female, 1=male)
            age_group: Patient age group ('<=40', '41-65', '>65')
            
        Returns:
            Recommended threshold
        """
        # Try priority order
        for factor in self.priority_order:
            if factor == 'sex' and sex is not None:
                key = f"sex_{'male' if sex == 1 else 'female'}"
                if key in self.group_thresholds:
                    return self.group_thresholds[key]['threshold']
            
            elif factor == 'age_group' and age_group is not None:
                key = f"age_{age_group}"
                if key in self.group_thresholds:
                    return self.group_thresholds[key]['threshold']
        
        # Fall back to overall threshold
        if 'overall' in self.group_thresholds:
            return self.group_thresholds['overall']['threshold']
        
        return self.default_threshold
    
    def predict(
        self,
        y_prob: np.ndarray,
        metadata: pd.DataFrame
    ) -> np.ndarray:
        """
        Make predictions using adaptive thresholds.
        
        Args:
            y_prob: Predicted probabilities
            metadata: Metadata with demographics
            
        Returns:
            Binary predictions
        """
        predictions = np.zeros(len(y_prob), dtype=int)
        
        for i in range(len(y_prob)):
            # Get patient demographics
            sex = metadata.iloc[i]['sex'] if 'sex' in metadata.columns else None
            age_group = metadata.iloc[i]['age_group'] if 'age_group' in metadata.columns else None
            
            # Get adaptive threshold
            threshold = self.get_threshold(sex, age_group)
            
            # Make prediction
            predictions[i] = int(y_prob[i] >= threshold)
        
        return predictions
    
    def predict_with_explanation(
        self,
        y_prob: float,
        sex: Optional[int] = None,
        age_group: Optional[str] = None
    ) -> Tuple[int, float, str]:
        """
        Make prediction with explanation of threshold used.
        
        Args:
            y_prob: Predicted probability
            sex: Patient sex
            age_group: Patient age group
            
        Returns:
            Tuple of (prediction, threshold_used, explanation)
        """
        threshold = self.get_threshold(sex, age_group)
        prediction = int(y_prob >= threshold)
        
        # Build explanation
        if age_group and f"age_{age_group}" in self.group_thresholds:
            explanation = f"Age-specific threshold ({age_group}): {threshold:.3f}"
        elif sex is not None and f"sex_{'male' if sex == 1 else 'female'}" in self.group_thresholds:
            sex_label = 'male' if sex == 1 else 'female'
            explanation = f"Sex-specific threshold ({sex_label}): {threshold:.3f}"
        elif 'overall' in self.group_thresholds:
            explanation = f"Overall threshold: {threshold:.3f}"
        else:
            explanation = f"Default threshold: {threshold:.3f}"
        
        return prediction, threshold, explanation


def compare_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metadata: pd.DataFrame,
    fixed_threshold: float = 0.5,
    adaptive_thresholds: Optional[Dict[str, Dict]] = None
) -> pd.DataFrame:
    """
    Compare fixed vs adaptive threshold performance.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        metadata: Metadata with demographics
        fixed_threshold: Fixed threshold to compare
        adaptive_thresholds: Adaptive thresholds to use
        
    Returns:
        DataFrame with comparison metrics
    """
    from train_baseline import compute_metrics
    
    results = []
    
    # Fixed threshold predictions
    y_pred_fixed = (y_prob >= fixed_threshold).astype(int)
    
    # Adaptive threshold predictions
    if adaptive_thresholds:
        predictor = AdaptiveThresholdPredictor(adaptive_thresholds)
        y_pred_adaptive = predictor.predict(y_prob, metadata)
    else:
        y_pred_adaptive = y_pred_fixed
    
    # Overall comparison
    for method, y_pred in [('Fixed', y_pred_fixed), ('Adaptive', y_pred_adaptive)]:
        metrics = compute_metrics(y_true, y_pred, y_prob)
        metrics['method'] = method
        metrics['group'] = 'Overall'
        results.append(metrics)
    
    # Group-wise comparison
    for sex in metadata['sex'].unique():
        if pd.isna(sex):
            continue
        
        mask = metadata['sex'] == sex
        sex_label = 'Male' if sex == 1 else 'Female'
        
        for method, y_pred in [('Fixed', y_pred_fixed), ('Adaptive', y_pred_adaptive)]:
            metrics = compute_metrics(y_true[mask], y_pred[mask], y_prob[mask])
            metrics['method'] = method
            metrics['group'] = f'Sex: {sex_label}'
            results.append(metrics)
    
    # Age groups
    for age_group in ['<=40', '41-65', '>65']:
        mask = metadata['age_group'] == age_group
        
        if np.sum(mask) < 5:
            continue
        
        for method, y_pred in [('Fixed', y_pred_fixed), ('Adaptive', y_pred_adaptive)]:
            metrics = compute_metrics(y_true[mask], y_pred[mask], y_prob[mask])
            metrics['method'] = method
            metrics['group'] = f'Age: {age_group}'
            results.append(metrics)
    
    return pd.DataFrame(results)


