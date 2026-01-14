"""
Time Domain Feature Extraction Module

Extracts time-domain features from ECG signals including amplitude,
morphological, statistical, and derivative features.
"""

import numpy as np
from typing import Dict, Optional
from scipy import stats


def extract_amplitude_features(ecg_signal: np.ndarray) -> Dict[str, float]:
    """
    Extract amplitude-based features from ECG signal.
    
    Args:
        ecg_signal: ECG signal (1D array)
        
    Returns:
        Dictionary of amplitude features
    """
    features = {}
    
    features['amplitude_max'] = float(np.max(ecg_signal))
    features['amplitude_min'] = float(np.min(ecg_signal))
    features['amplitude_range'] = float(np.ptp(ecg_signal))
    features['amplitude_mean'] = float(np.mean(ecg_signal))
    features['amplitude_std'] = float(np.std(ecg_signal))
    features['amplitude_median'] = float(np.median(ecg_signal))
    features['amplitude_mad'] = float(np.median(np.abs(ecg_signal - np.median(ecg_signal))))
    
    return features


def extract_statistical_features(ecg_signal: np.ndarray) -> Dict[str, float]:
    """
    Extract statistical features from ECG signal.
    
    Args:
        ecg_signal: ECG signal (1D array)
        
    Returns:
        Dictionary of statistical features
    """
    features = {}
    
    # Basic moments
    features['variance'] = float(np.var(ecg_signal))
    features['rms'] = float(np.sqrt(np.mean(ecg_signal ** 2)))
    
    # Higher order moments
    if len(ecg_signal) > 3:
        features['skewness'] = float(stats.skew(ecg_signal))
        features['kurtosis'] = float(stats.kurtosis(ecg_signal))
    else:
        features['skewness'] = 0.0
        features['kurtosis'] = 0.0
    
    # Percentiles
    features['q25'] = float(np.percentile(ecg_signal, 25))
    features['q50'] = float(np.percentile(ecg_signal, 50))
    features['q75'] = float(np.percentile(ecg_signal, 75))
    features['iqr'] = features['q75'] - features['q25']
    
    # Coefficient of variation
    mean_val = np.mean(ecg_signal)
    std_val = np.std(ecg_signal)
    if mean_val != 0:
        features['cv'] = std_val / abs(mean_val)
    else:
        features['cv'] = 0.0
    
    return features


def extract_derivative_features(ecg_signal: np.ndarray) -> Dict[str, float]:
    """
    Extract features from signal derivatives.
    
    Args:
        ecg_signal: ECG signal (1D array)
        
    Returns:
        Dictionary of derivative features
    """
    features = {}
    
    if len(ecg_signal) < 2:
        return {
            'first_deriv_mean': 0.0, 'first_deriv_std': 0.0, 'first_deriv_max': 0.0,
            'second_deriv_mean': 0.0, 'second_deriv_std': 0.0, 'second_deriv_max': 0.0
        }
    
    # First derivative
    first_deriv = np.diff(ecg_signal)
    features['first_deriv_mean'] = float(np.mean(np.abs(first_deriv)))
    features['first_deriv_std'] = float(np.std(first_deriv))
    features['first_deriv_max'] = float(np.max(np.abs(first_deriv)))
    
    # Second derivative
    if len(first_deriv) > 1:
        second_deriv = np.diff(first_deriv)
        features['second_deriv_mean'] = float(np.mean(np.abs(second_deriv)))
        features['second_deriv_std'] = float(np.std(second_deriv))
        features['second_deriv_max'] = float(np.max(np.abs(second_deriv)))
    else:
        features['second_deriv_mean'] = 0.0
        features['second_deriv_std'] = 0.0
        features['second_deriv_max'] = 0.0
    
    return features


def extract_zero_crossing_features(ecg_signal: np.ndarray) -> Dict[str, float]:
    """
    Extract zero-crossing features.
    
    Args:
        ecg_signal: ECG signal (1D array)
        
    Returns:
        Dictionary of zero-crossing features
    """
    features = {}
    
    # Zero crossings
    zero_crossings = np.where(np.diff(np.sign(ecg_signal)))[0]
    features['zero_crossing_count'] = len(zero_crossings)
    features['zero_crossing_rate'] = len(zero_crossings) / len(ecg_signal)
    
    # Mean crossing (signal crosses mean)
    mean_centered = ecg_signal - np.mean(ecg_signal)
    mean_crossings = np.where(np.diff(np.sign(mean_centered)))[0]
    features['mean_crossing_rate'] = len(mean_crossings) / len(ecg_signal)
    
    return features


def extract_peak_features(ecg_signal: np.ndarray) -> Dict[str, float]:
    """
    Extract peak-related features.
    
    Args:
        ecg_signal: ECG signal (1D array)
        
    Returns:
        Dictionary of peak features
    """
    from scipy.signal import find_peaks
    
    features = {}
    
    # Find peaks with minimal height threshold
    threshold = np.mean(ecg_signal) + 0.5 * np.std(ecg_signal)
    peaks, properties = find_peaks(ecg_signal, height=threshold, distance=20)
    
    features['peak_count'] = len(peaks)
    
    if len(peaks) > 0:
        features['peak_mean_height'] = float(np.mean(properties['peak_heights']))
        features['peak_std_height'] = float(np.std(properties['peak_heights']))
        features['peak_max_height'] = float(np.max(properties['peak_heights']))
        
        # Peak distances (if multiple peaks)
        if len(peaks) > 1:
            peak_distances = np.diff(peaks)
            features['peak_mean_distance'] = float(np.mean(peak_distances))
            features['peak_std_distance'] = float(np.std(peak_distances))
        else:
            features['peak_mean_distance'] = 0.0
            features['peak_std_distance'] = 0.0
    else:
        features['peak_mean_height'] = 0.0
        features['peak_std_height'] = 0.0
        features['peak_max_height'] = 0.0
        features['peak_mean_distance'] = 0.0
        features['peak_std_distance'] = 0.0
    
    return features


def extract_all_time_domain_features(ecg_signal: np.ndarray) -> Dict[str, float]:
    """
    Extract all time-domain features from ECG signal.
    
    Args:
        ecg_signal: ECG signal (1D array)
        
    Returns:
        Dictionary of all time-domain features
        
    Examples:
        >>> signal = np.sin(2 * np.pi * 2 * np.arange(1000) / 100)
        >>> features = extract_all_time_domain_features(signal)
        >>> 'amplitude_mean' in features
        True
    """
    features = {}
    
    features.update(extract_amplitude_features(ecg_signal))
    features.update(extract_statistical_features(ecg_signal))
    features.update(extract_derivative_features(ecg_signal))
    features.update(extract_zero_crossing_features(ecg_signal))
    features.update(extract_peak_features(ecg_signal))
    
    return features

