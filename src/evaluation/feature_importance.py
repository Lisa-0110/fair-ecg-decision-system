"""
Feature Importance Analysis Module

Analyzes and compares feature importance across baseline and fairness-aware models
to understand decision drivers and assess interpretability.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')


class FeatureImportanceAnalyzer:
    """
    Analyzes feature importance using multiple methods.
    """
    
    def __init__(self, model, feature_names: List[str]):
        """
        Initialize analyzer.
        
        Args:
            model: Trained classifier
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names
        self.n_features = len(feature_names)
    
    def get_coefficient_importance(self) -> pd.DataFrame:
        """
        Get feature importance from model coefficients (for linear models).
        
        Returns:
            DataFrame with feature importance scores
        """
        if not hasattr(self.model, 'coef_'):
            raise ValueError("Model does not have coefficients (not a linear model)")
        
        # Get coefficients (for binary classification, shape is [1, n_features])
        coef = self.model.coef_[0]
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': coef,
            'abs_coefficient': np.abs(coef)
        })
        
        # Sort by absolute value
        importance_df = importance_df.sort_values('abs_coefficient', ascending=False)
        importance_df['rank'] = range(1, len(importance_df) + 1)
        
        return importance_df
    
    def get_tree_importance(self) -> pd.DataFrame:
        """
        Get feature importance from tree-based models.
        
        Returns:
            DataFrame with feature importance scores
        """
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model does not have feature_importances_ (not a tree model)")
        
        importance = self.model.feature_importances_
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        importance_df['rank'] = range(1, len(importance_df) + 1)
        
        return importance_df
    
    def get_permutation_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_repeats: int = 10,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Compute permutation importance (model-agnostic).
        
        Args:
            X: Feature matrix
            y: True labels
            n_repeats: Number of permutations
            random_state: Random seed
            
        Returns:
            DataFrame with permutation importance
        """
        result = permutation_importance(
            self.model,
            X,
            y,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=-1
        )
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance_mean': result.importances_mean,
            'importance_std': result.importances_std
        })
        
        # Sort by mean importance
        importance_df = importance_df.sort_values('importance_mean', ascending=False)
        importance_df['rank'] = range(1, len(importance_df) + 1)
        
        return importance_df
    
    def categorize_features(self) -> Dict[str, List[str]]:
        """
        Categorize features by type.
        
        Returns:
            Dictionary mapping category to feature names
        """
        categories = {
            'Time Domain - Amplitude': [],
            'Time Domain - Statistical': [],
            'Time Domain - Derivative': [],
            'Frequency Domain - Power': [],
            'Frequency Domain - Spectral': [],
            'HRV - Time Domain': [],
            'HRV - Frequency Domain': [],
            'HRV - Nonlinear': [],
            'Heart Rate': [],
            'Entropy': [],
            'Quality': [],
            'Other': []
        }
        
        for feature in self.feature_names:
            # Time domain
            if any(x in feature for x in ['amplitude_max', 'amplitude_min', 'amplitude_range', 
                                          'amplitude_mean', 'amplitude_median', 'amplitude_mad']):
                categories['Time Domain - Amplitude'].append(feature)
            elif any(x in feature for x in ['variance', 'rms', 'skewness', 'kurtosis', 
                                            'q25', 'q50', 'q75', 'iqr', 'cv', 'std']):
                categories['Time Domain - Statistical'].append(feature)
            elif any(x in feature for x in ['first_deriv', 'second_deriv', 'zero_crossing', 
                                            'mean_crossing', 'peak_']):
                categories['Time Domain - Derivative'].append(feature)
            
            # Frequency domain
            elif any(x in feature for x in ['_power', 'lf_', 'hf_', 'qrs_power', 'twave_power']):
                categories['Frequency Domain - Power'].append(feature)
            elif any(x in feature for x in ['spectral_', 'dominant_freq', 'dominant_power']):
                categories['Frequency Domain - Spectral'].append(feature)
            
            # HRV
            elif any(x in feature for x in ['hrv_mean_rr', 'hrv_std_rr', 'hrv_median_rr', 
                                            'hrv_rmssd', 'hrv_sdsd', 'hrv_nn50', 'hrv_pnn50',
                                            'hrv_mad_rr', 'hrv_range_rr', 'hrv_iqr_rr', 'hrv_cv']):
                categories['HRV - Time Domain'].append(feature)
            elif any(x in feature for x in ['hrv_lf', 'hrv_hf']):
                categories['HRV - Frequency Domain'].append(feature)
            elif any(x in feature for x in ['hrv_sd1', 'hrv_sd2', 'hrv_sd_ratio', 
                                            'hrv_csi', 'hrv_cvi', 'hrv_csvi',
                                            'hrv_triangular', 'hrv_tinn']):
                categories['HRV - Nonlinear'].append(feature)
            
            # Heart rate
            elif any(x in feature for x in ['hr_mean', 'hr_min', 'hr_max', 'hr_std', 'hr_range']):
                categories['Heart Rate'].append(feature)
            
            # Entropy
            elif any(x in feature for x in ['entropy', 'mse_']):
                categories['Entropy'].append(feature)
            
            # Quality
            elif any(x in feature for x in ['is_good_quality', 'snr_db']):
                categories['Quality'].append(feature)
            
            else:
                categories['Other'].append(feature)
        
        # Remove empty categories
        categories = {k: v for k, v in categories.items() if len(v) > 0}
        
        return categories
    
    def aggregate_by_category(
        self,
        importance_df: pd.DataFrame,
        importance_col: str = 'abs_coefficient'
    ) -> pd.DataFrame:
        """
        Aggregate feature importance by category.
        
        Args:
            importance_df: DataFrame with feature importance
            importance_col: Column name for importance scores
            
        Returns:
            DataFrame with category-level importance
        """
        categories = self.categorize_features()
        
        category_importance = []
        
        for category, features in categories.items():
            # Get importance for features in this category
            mask = importance_df['feature'].isin(features)
            cat_features = importance_df[mask]
            
            if len(cat_features) > 0:
                category_importance.append({
                    'category': category,
                    'n_features': len(cat_features),
                    'total_importance': cat_features[importance_col].sum(),
                    'mean_importance': cat_features[importance_col].mean(),
                    'max_importance': cat_features[importance_col].max(),
                    'top_feature': cat_features.iloc[0]['feature'],
                    'top_feature_importance': cat_features.iloc[0][importance_col]
                })
        
        category_df = pd.DataFrame(category_importance)
        category_df = category_df.sort_values('total_importance', ascending=False)
        category_df['rank'] = range(1, len(category_df) + 1)
        
        return category_df


def compare_feature_importance(
    importance1: pd.DataFrame,
    importance2: pd.DataFrame,
    model1_name: str = 'Model 1',
    model2_name: str = 'Model 2',
    importance_col: str = 'abs_coefficient'
) -> pd.DataFrame:
    """
    Compare feature importance between two models.
    
    Args:
        importance1: Feature importance from model 1
        importance2: Feature importance from model 2
        model1_name: Name of model 1
        model2_name: Name of model 2
        importance_col: Column name for importance scores
        
    Returns:
        DataFrame with comparison
    """
    # Merge on feature name
    comparison = importance1[['feature', importance_col]].merge(
        importance2[['feature', importance_col]],
        on='feature',
        suffixes=(f'_{model1_name}', f'_{model2_name}')
    )
    
    col1 = f'{importance_col}_{model1_name}'
    col2 = f'{importance_col}_{model2_name}'
    
    # Compute differences
    comparison['difference'] = comparison[col2] - comparison[col1]
    comparison['abs_difference'] = np.abs(comparison['difference'])
    comparison['percent_change'] = (comparison['difference'] / (comparison[col1] + 1e-10)) * 100
    
    # Sort by absolute difference
    comparison = comparison.sort_values('abs_difference', ascending=False)
    comparison['rank_change'] = range(1, len(comparison) + 1)
    
    return comparison


def interpret_clinical_relevance(feature_name: str) -> Dict[str, str]:
    """
    Provide clinical interpretation for a feature.
    
    Args:
        feature_name: Name of the feature
        
    Returns:
        Dictionary with interpretation
    """
    interpretations = {
        # Amplitude features
        'amplitude_max': {
            'clinical_name': 'Peak Amplitude',
            'meaning': 'Maximum voltage deviation in ECG',
            'clinical_relevance': 'Indicates R-wave height; high values may suggest ventricular hypertrophy'
        },
        'amplitude_range': {
            'clinical_name': 'Amplitude Range',
            'meaning': 'Total voltage span of ECG signal',
            'clinical_relevance': 'Reflects overall cardiac electrical activity'
        },
        
        # HRV features
        'hrv_rmssd': {
            'clinical_name': 'RMSSD (HRV)',
            'meaning': 'Root mean square of successive RR interval differences',
            'clinical_relevance': 'Measures short-term heart rate variability; low values indicate poor autonomic function'
        },
        'hrv_sdnn': {
            'clinical_name': 'SDNN (HRV)',
            'meaning': 'Standard deviation of RR intervals',
            'clinical_relevance': 'Overall HRV measure; reduced in cardiac disease and aging'
        },
        
        # Heart rate
        'hr_mean': {
            'clinical_name': 'Mean Heart Rate',
            'meaning': 'Average heart rate during recording',
            'clinical_relevance': 'Abnormal values may indicate arrhythmia or cardiac dysfunction'
        },
        
        # Frequency domain
        'lf_hf_ratio': {
            'clinical_name': 'LF/HF Ratio',
            'meaning': 'Ratio of low to high frequency power',
            'clinical_relevance': 'Reflects autonomic balance; elevated in stress/cardiac disease'
        },
        
        # Entropy
        'sample_entropy': {
            'clinical_name': 'Sample Entropy',
            'meaning': 'Signal complexity/regularity measure',
            'clinical_relevance': 'Reduced entropy indicates more regular (potentially pathological) rhythm'
        },
        
        # Quality
        'snr_db': {
            'clinical_name': 'Signal-to-Noise Ratio',
            'meaning': 'Signal quality measure',
            'clinical_relevance': 'Low SNR indicates poor recording quality'
        }
    }
    
    # Try to match feature name
    for key, interpretation in interpretations.items():
        if key in feature_name:
            return interpretation
    
    # Generic interpretation
    return {
        'clinical_name': feature_name.replace('_', ' ').title(),
        'meaning': 'ECG-derived measurement',
        'clinical_relevance': 'Contributes to overall cardiac risk assessment'
    }


def assess_interpretability(
    importance_df: pd.DataFrame,
    top_k: int = 20
) -> Dict[str, any]:
    """
    Assess model interpretability based on feature importance.
    
    Args:
        importance_df: DataFrame with feature importance
        top_k: Number of top features to consider
        
    Returns:
        Dictionary with interpretability assessment
    """
    top_features = importance_df.head(top_k)
    
    # Get categories of top features
    analyzer = FeatureImportanceAnalyzer(None, importance_df['feature'].tolist())
    categories = analyzer.categorize_features()
    
    top_categories = {}
    for category, features in categories.items():
        n_in_top = sum(1 for f in features if f in top_features['feature'].values)
        if n_in_top > 0:
            top_categories[category] = n_in_top
    
    # Calculate concentration (Gini coefficient-like measure)
    importances = importance_df['abs_coefficient' if 'abs_coefficient' in importance_df.columns else 'importance'].values
    cumsum = np.cumsum(importances)
    total = cumsum[-1]
    
    # Percentage of importance in top K features
    top_k_importance = cumsum[top_k-1] / total if len(cumsum) >= top_k else 1.0
    
    assessment = {
        'top_k': top_k,
        'top_k_importance_fraction': top_k_importance,
        'top_categories': top_categories,
        'n_categories_in_top_k': len(top_categories),
        'concentration_score': top_k_importance,  # Higher = more concentrated
        'interpretability': 'High' if top_k_importance > 0.8 else ('Moderate' if top_k_importance > 0.6 else 'Low')
    }
    
    return assessment


