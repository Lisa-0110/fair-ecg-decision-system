"""
Uncertainty-Based Rejection Module

Implements confidence-based rejection mechanisms to defer uncertain predictions
to human clinicians, improving safety in clinical decision support systems.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List, Union
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')


class UncertaintyEstimator:
    """
    Estimates prediction uncertainty using multiple methods.
    """
    
    def __init__(self, method: str = 'probability'):
        """
        Initialize uncertainty estimator.
        
        Args:
            method: Uncertainty estimation method
                - 'probability': Use prediction probability (1 - max(p))
                - 'entropy': Shannon entropy of prediction distribution
                - 'margin': Distance from decision boundary
                - 'ensemble': Ensemble disagreement (if multiple models)
        """
        self.method = method
    
    def estimate(
        self,
        y_prob: np.ndarray,
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Estimate uncertainty for predictions.
        
        Args:
            y_prob: Predicted probabilities (for positive class)
            threshold: Classification threshold
            
        Returns:
            Array of uncertainty scores (higher = more uncertain)
        """
        if self.method == 'probability':
            # Uncertainty = distance from confident prediction (0 or 1)
            # Most uncertain at p=0.5, least uncertain at p=0 or p=1
            uncertainty = 1 - np.abs(2 * y_prob - 1)
            
        elif self.method == 'entropy':
            # Shannon entropy: -p*log(p) - (1-p)*log(1-p)
            p_pos = np.clip(y_prob, 1e-10, 1 - 1e-10)  # Avoid log(0)
            p_neg = 1 - p_pos
            uncertainty = -(p_pos * np.log2(p_pos) + p_neg * np.log2(p_neg))
            
        elif self.method == 'margin':
            # Distance from decision boundary
            # Uncertainty = how close to threshold
            uncertainty = 1 - np.abs(y_prob - threshold) / threshold
            
        else:
            raise ValueError(f"Unknown uncertainty method: {self.method}")
        
        return uncertainty
    
    def get_confidence(
        self,
        y_prob: np.ndarray,
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Get confidence scores (inverse of uncertainty).
        
        Args:
            y_prob: Predicted probabilities
            threshold: Classification threshold
            
        Returns:
            Array of confidence scores (higher = more confident)
        """
        uncertainty = self.estimate(y_prob, threshold)
        confidence = 1 - uncertainty
        return confidence


class RejectionOptimizer:
    """
    Optimizes rejection threshold to balance accuracy and coverage.
    """
    
    def __init__(
        self,
        target_accuracy: Optional[float] = None,
        target_coverage: Optional[float] = None,
        min_accuracy_gain: float = 0.05
    ):
        """
        Initialize rejection optimizer.
        
        Args:
            target_accuracy: Desired accuracy on accepted predictions
            target_coverage: Desired coverage (fraction not rejected)
            min_accuracy_gain: Minimum accuracy improvement to justify rejection
        """
        self.target_accuracy = target_accuracy
        self.target_coverage = target_coverage
        self.min_accuracy_gain = min_accuracy_gain
    
    def compute_metrics_with_rejection(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        confidence: np.ndarray,
        confidence_threshold: float
    ) -> Dict[str, float]:
        """
        Compute metrics with rejection at a specific confidence threshold.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities
            confidence: Confidence scores
            confidence_threshold: Minimum confidence to accept prediction
            
        Returns:
            Dictionary of metrics including rejection statistics
        """
        # Determine which predictions to accept/reject
        accept_mask = confidence >= confidence_threshold
        n_total = len(y_true)
        n_accepted = np.sum(accept_mask)
        n_rejected = n_total - n_accepted
        
        metrics = {
            'confidence_threshold': confidence_threshold,
            'n_total': n_total,
            'n_accepted': n_accepted,
            'n_rejected': n_rejected,
            'coverage': n_accepted / n_total if n_total > 0 else 0.0,
            'rejection_rate': n_rejected / n_total if n_total > 0 else 0.0,
        }
        
        if n_accepted == 0:
            # All predictions rejected
            metrics.update({
                'accuracy_accepted': 0.0,
                'fnr_accepted': 0.0,
                'fpr_accepted': 0.0,
                'accuracy_rejected': np.mean(y_true == y_pred),
            })
            return metrics
        
        # Metrics on accepted predictions
        y_true_acc = y_true[accept_mask]
        y_pred_acc = y_pred[accept_mask]
        
        accuracy_accepted = np.mean(y_true_acc == y_pred_acc)
        
        if len(np.unique(y_true_acc)) > 1:
            tn, fp, fn, tp = confusion_matrix(y_true_acc, y_pred_acc).ravel()
            fnr_accepted = fn / (fn + tp) if (fn + tp) > 0 else 0.0
            fpr_accepted = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        else:
            fnr_accepted = 0.0
            fpr_accepted = 0.0
        
        metrics['accuracy_accepted'] = accuracy_accepted
        metrics['fnr_accepted'] = fnr_accepted
        metrics['fpr_accepted'] = fpr_accepted
        
        # Metrics on rejected predictions (if we made them anyway)
        if n_rejected > 0:
            reject_mask = ~accept_mask
            y_true_rej = y_true[reject_mask]
            y_pred_rej = y_pred[reject_mask]
            accuracy_rejected = np.mean(y_true_rej == y_pred_rej)
            metrics['accuracy_rejected'] = accuracy_rejected
            metrics['accuracy_gain'] = accuracy_accepted - accuracy_rejected
        else:
            metrics['accuracy_rejected'] = 0.0
            metrics['accuracy_gain'] = 0.0
        
        # Overall accuracy (treating rejections as errors is conservative)
        # Alternative: could compute only on accepted + perfect oracle on rejected
        metrics['accuracy_overall_conservative'] = accuracy_accepted * metrics['coverage']
        
        return metrics
    
    def optimize_rejection_threshold(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        confidence: np.ndarray,
        search_range: Tuple[float, float] = (0.0, 1.0),
        n_points: int = 101
    ) -> Tuple[float, Dict[str, float]]:
        """
        Find optimal rejection threshold.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities
            confidence: Confidence scores
            search_range: Range to search for threshold
            n_points: Number of points to evaluate
            
        Returns:
            Tuple of (optimal_threshold, metrics_at_threshold)
        """
        thresholds = np.linspace(search_range[0], search_range[1], n_points)
        results = []
        
        for thresh in thresholds:
            metrics = self.compute_metrics_with_rejection(
                y_true, y_pred, y_prob, confidence, thresh
            )
            results.append(metrics)
        
        results_df = pd.DataFrame(results)
        
        # Select optimal threshold based on objectives
        if self.target_accuracy is not None:
            # Find minimum rejection rate that achieves target accuracy
            valid = results_df[results_df['accuracy_accepted'] >= self.target_accuracy]
            if len(valid) > 0:
                optimal_idx = valid['coverage'].idxmax()
            else:
                # Target not achievable, pick best accuracy
                optimal_idx = results_df['accuracy_accepted'].idxmax()
        
        elif self.target_coverage is not None:
            # Find best accuracy at target coverage
            valid = results_df[results_df['coverage'] >= self.target_coverage]
            if len(valid) > 0:
                optimal_idx = valid['accuracy_accepted'].idxmax()
            else:
                # Target not achievable, pick closest
                optimal_idx = (results_df['coverage'] - self.target_coverage).abs().idxmin()
        
        else:
            # Maximize accuracy gain while maintaining reasonable coverage (>50%)
            valid = results_df[
                (results_df['coverage'] >= 0.5) & 
                (results_df['accuracy_gain'] >= self.min_accuracy_gain)
            ]
            if len(valid) > 0:
                # Choose point with best accuracy-coverage tradeoff
                valid['score'] = valid['accuracy_accepted'] * valid['coverage']
                optimal_idx = valid['score'].idxmax()
            else:
                # No good rejection threshold, use 0 (accept all)
                optimal_idx = 0
        
        optimal_threshold = results_df.loc[optimal_idx, 'confidence_threshold']
        optimal_metrics = results_df.loc[optimal_idx].to_dict()
        
        return optimal_threshold, optimal_metrics


class UncertaintyRejectPredictor:
    """
    Predictor with uncertainty-based rejection option.
    """
    
    def __init__(
        self,
        model,
        uncertainty_estimator: UncertaintyEstimator,
        rejection_threshold: float,
        classification_threshold: float = 0.5
    ):
        """
        Initialize rejection predictor.
        
        Args:
            model: Trained classifier with predict_proba method
            uncertainty_estimator: Uncertainty estimation method
            rejection_threshold: Confidence threshold for acceptance
            classification_threshold: Threshold for positive/negative classification
        """
        self.model = model
        self.uncertainty_estimator = uncertainty_estimator
        self.rejection_threshold = rejection_threshold
        self.classification_threshold = classification_threshold
    
    def predict_with_rejection(
        self,
        X: np.ndarray,
        return_confidence: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Make predictions with rejection option.
        
        Args:
            X: Feature matrix
            return_confidence: Whether to return confidence scores
            
        Returns:
            If return_confidence=False:
                Array of predictions: 0, 1, or -1 (rejected)
            If return_confidence=True:
                Tuple of (predictions, probabilities, confidence_scores)
        """
        # Get probabilities
        y_prob = self.model.predict_proba(X)[:, 1]
        
        # Get confidence
        confidence = self.uncertainty_estimator.get_confidence(
            y_prob,
            self.classification_threshold
        )
        
        # Make predictions
        predictions = np.zeros(len(X), dtype=int)
        
        # Accept confident predictions
        accept_mask = confidence >= self.rejection_threshold
        y_pred = (y_prob >= self.classification_threshold).astype(int)
        predictions[accept_mask] = y_pred[accept_mask]
        
        # Reject uncertain predictions
        predictions[~accept_mask] = -1
        
        if return_confidence:
            return predictions, y_prob, confidence
        else:
            return predictions
    
    def predict_with_explanation(
        self,
        X: np.ndarray,
        metadata: Optional[pd.DataFrame] = None
    ) -> List[Dict]:
        """
        Make predictions with detailed explanations.
        
        Args:
            X: Feature matrix
            metadata: Optional metadata for context
            
        Returns:
            List of dictionaries with prediction details
        """
        predictions, y_prob, confidence = self.predict_with_rejection(
            X,
            return_confidence=True
        )
        
        results = []
        
        for i in range(len(X)):
            result = {
                'prediction': predictions[i],
                'probability': y_prob[i],
                'confidence': confidence[i],
                'classification_threshold': self.classification_threshold,
                'rejection_threshold': self.rejection_threshold,
            }
            
            # Interpretation
            if predictions[i] == -1:
                result['decision'] = 'REFER TO CLINICIAN'
                result['reason'] = f'Low confidence ({confidence[i]:.3f} < {self.rejection_threshold:.3f})'
                result['recommendation'] = 'Manual review required due to uncertain prediction'
            elif predictions[i] == 1:
                result['decision'] = 'POSITIVE (High Risk)'
                result['reason'] = f'High confidence ({confidence[i]:.3f} ≥ {self.rejection_threshold:.3f})'
                result['recommendation'] = 'Further testing/monitoring recommended'
            else:
                result['decision'] = 'NEGATIVE (Low Risk)'
                result['reason'] = f'High confidence ({confidence[i]:.3f} ≥ {self.rejection_threshold:.3f})'
                result['recommendation'] = 'Routine follow-up'
            
            # Add demographic context if available
            if metadata is not None and i < len(metadata):
                if 'age' in metadata.columns:
                    result['age'] = metadata.iloc[i]['age']
                if 'age_group' in metadata.columns:
                    result['age_group'] = metadata.iloc[i]['age_group']
                if 'sex' in metadata.columns:
                    sex_val = metadata.iloc[i]['sex']
                    result['sex'] = 'Male' if sex_val == 1 else 'Female'
            
            results.append(result)
        
        return results
    
    def get_rejection_statistics(
        self,
        X: np.ndarray,
        y_true: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Get statistics about rejections.
        
        Args:
            X: Feature matrix
            y_true: True labels (optional)
            
        Returns:
            Dictionary of rejection statistics
        """
        predictions, y_prob, confidence = self.predict_with_rejection(
            X,
            return_confidence=True
        )
        
        n_total = len(predictions)
        n_rejected = np.sum(predictions == -1)
        n_accepted = n_total - n_rejected
        
        stats = {
            'n_total': n_total,
            'n_accepted': n_accepted,
            'n_rejected': n_rejected,
            'rejection_rate': n_rejected / n_total,
            'coverage': n_accepted / n_total,
            'mean_confidence_all': confidence.mean(),
            'mean_confidence_accepted': confidence[predictions != -1].mean() if n_accepted > 0 else 0,
            'mean_confidence_rejected': confidence[predictions == -1].mean() if n_rejected > 0 else 0,
        }
        
        if y_true is not None:
            # Accuracy on accepted predictions
            accepted_mask = predictions != -1
            if np.sum(accepted_mask) > 0:
                y_true_acc = y_true[accepted_mask]
                y_pred_acc = predictions[accepted_mask]
                stats['accuracy_accepted'] = np.mean(y_true_acc == y_pred_acc)
                
                # Check if rejected predictions would have been wrong
                rejected_mask = predictions == -1
                if np.sum(rejected_mask) > 0:
                    y_true_rej = y_true[rejected_mask]
                    y_pred_rej = (y_prob[rejected_mask] >= self.classification_threshold).astype(int)
                    stats['accuracy_rejected_if_accepted'] = np.mean(y_true_rej == y_pred_rej)
        
        return stats


