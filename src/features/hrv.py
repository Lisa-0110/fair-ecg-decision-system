"""
Heart Rate Variability (HRV) Feature Extraction Module

Extracts HRV features from RR intervals detected in ECG signals.
"""

import numpy as np
from scipy.signal import find_peaks
from typing import Dict, Optional, Tuple


def detect_r_peaks(
    ecg_signal: np.ndarray,
    fs: float = 100.0,
    min_distance: float = 0.4
) -> np.ndarray:
    """
    Detect R-peaks in ECG signal using simple peak detection.
    
    Args:
        ecg_signal: ECG signal (1D array)
        fs: Sampling frequency in Hz
        min_distance: Minimum distance between peaks in seconds
        
    Returns:
        Array of R-peak indices
    """
    min_samples = int(min_distance * fs)
    
    # Find peaks with minimum distance
    threshold = np.mean(ecg_signal) + 0.5 * np.std(ecg_signal)
    peaks, _ = find_peaks(ecg_signal, height=threshold, distance=min_samples)
    
    return peaks


def compute_rr_intervals(
    r_peaks: np.ndarray,
    fs: float = 100.0
) -> np.ndarray:
    """
    Compute RR intervals from R-peak locations.
    
    Args:
        r_peaks: Array of R-peak indices
        fs: Sampling frequency in Hz
        
    Returns:
        Array of RR intervals in milliseconds
    """
    if len(r_peaks) < 2:
        return np.array([])
    
    rr_samples = np.diff(r_peaks)
    rr_intervals = (rr_samples / fs) * 1000  # Convert to milliseconds
    
    return rr_intervals


def filter_rr_intervals(
    rr_intervals: np.ndarray,
    min_rr: float = 300.0,
    max_rr: float = 2000.0
) -> np.ndarray:
    """
    Filter out physiologically invalid RR intervals.
    
    Args:
        rr_intervals: RR intervals in milliseconds
        min_rr: Minimum valid RR interval (ms)
        max_rr: Maximum valid RR interval (ms)
        
    Returns:
        Filtered RR intervals
    """
    if len(rr_intervals) == 0:
        return rr_intervals
    
    valid_mask = (rr_intervals >= min_rr) & (rr_intervals <= max_rr)
    return rr_intervals[valid_mask]


def extract_time_domain_hrv(rr_intervals: np.ndarray) -> Dict[str, float]:
    """
    Extract time-domain HRV features.
    
    Args:
        rr_intervals: RR intervals in milliseconds
        
    Returns:
        Dictionary of time-domain HRV features
    """
    features = {}
    
    if len(rr_intervals) < 2:
        return {
            'hrv_mean_rr': 0.0, 'hrv_std_rr': 0.0, 'hrv_rmssd': 0.0,
            'hrv_sdsd': 0.0, 'hrv_nn50': 0.0, 'hrv_pnn50': 0.0,
            'hrv_cv': 0.0, 'hrv_median_rr': 0.0, 'hrv_mad_rr': 0.0,
            'hrv_range_rr': 0.0, 'hrv_iqr_rr': 0.0
        }
    
    # Basic statistics
    features['hrv_mean_rr'] = float(np.mean(rr_intervals))
    features['hrv_std_rr'] = float(np.std(rr_intervals))  # SDNN
    features['hrv_median_rr'] = float(np.median(rr_intervals))
    features['hrv_mad_rr'] = float(np.median(np.abs(rr_intervals - np.median(rr_intervals))))
    features['hrv_range_rr'] = float(np.ptp(rr_intervals))
    
    # IQR
    q75, q25 = np.percentile(rr_intervals, [75, 25])
    features['hrv_iqr_rr'] = float(q75 - q25)
    
    # Coefficient of variation
    if features['hrv_mean_rr'] > 0:
        features['hrv_cv'] = features['hrv_std_rr'] / features['hrv_mean_rr']
    else:
        features['hrv_cv'] = 0.0
    
    # Successive differences
    successive_diffs = np.diff(rr_intervals)
    
    # RMSSD - Root mean square of successive differences
    features['hrv_rmssd'] = float(np.sqrt(np.mean(successive_diffs ** 2)))
    
    # SDSD - Standard deviation of successive differences
    features['hrv_sdsd'] = float(np.std(successive_diffs))
    
    # NN50 and pNN50
    features['hrv_nn50'] = float(np.sum(np.abs(successive_diffs) > 50))
    if len(successive_diffs) > 0:
        features['hrv_pnn50'] = (features['hrv_nn50'] / len(successive_diffs)) * 100
    else:
        features['hrv_pnn50'] = 0.0
    
    return features


def extract_geometric_hrv(rr_intervals: np.ndarray) -> Dict[str, float]:
    """
    Extract geometric HRV features.
    
    Args:
        rr_intervals: RR intervals in milliseconds
        
    Returns:
        Dictionary of geometric HRV features
    """
    features = {}
    
    if len(rr_intervals) < 2:
        return {'hrv_triangular_index': 0.0, 'hrv_tinn': 0.0}
    
    # Triangular index (total NN intervals / height of histogram)
    if np.ptp(rr_intervals) > 0:
        bins = int(np.ptp(rr_intervals) / 7.8125)  # 7.8125 ms bins
        if bins > 0:
            hist, _ = np.histogram(rr_intervals, bins=bins)
            max_hist = np.max(hist)
            if max_hist > 0:
                features['hrv_triangular_index'] = float(len(rr_intervals) / max_hist)
            else:
                features['hrv_triangular_index'] = 0.0
        else:
            features['hrv_triangular_index'] = 0.0
    else:
        features['hrv_triangular_index'] = 0.0
    
    # TINN - Baseline width of RR interval histogram
    features['hrv_tinn'] = float(np.ptp(rr_intervals))
    
    return features


def extract_poincare_hrv(rr_intervals: np.ndarray) -> Dict[str, float]:
    """
    Extract Poincaré plot HRV features (non-linear).
    
    Args:
        rr_intervals: RR intervals in milliseconds
        
    Returns:
        Dictionary of Poincaré features
    """
    features = {}
    
    if len(rr_intervals) < 3:
        return {
            'hrv_sd1': 0.0, 'hrv_sd2': 0.0, 'hrv_sd_ratio': 0.0,
            'hrv_csi': 0.0, 'hrv_cvi': 0.0, 'hrv_csvi': 0.0
        }
    
    # Poincaré plot: RR(n) vs RR(n+1)
    rr1 = rr_intervals[:-1]
    rr2 = rr_intervals[1:]
    
    # SD1 - Standard deviation perpendicular to line of identity
    # Measures short-term variability
    features['hrv_sd1'] = float(np.std(rr2 - rr1) / np.sqrt(2))
    
    # SD2 - Standard deviation along line of identity
    # Measures long-term variability
    features['hrv_sd2'] = float(np.std(rr2 + rr1) / np.sqrt(2))
    
    # SD1/SD2 ratio
    if features['hrv_sd2'] > 0:
        features['hrv_sd_ratio'] = features['hrv_sd1'] / features['hrv_sd2']
    else:
        features['hrv_sd_ratio'] = 0.0
    
    # Cardiac Sympathetic Index
    features['hrv_csi'] = features['hrv_sd2'] / features['hrv_sd1'] if features['hrv_sd1'] > 0 else 0.0
    
    # Cardiac Vagal Index
    mean_rr = np.mean(rr_intervals)
    if mean_rr > 0:
        features['hrv_cvi'] = np.log10(features['hrv_sd1'] * features['hrv_sd2']) / mean_rr
    else:
        features['hrv_cvi'] = 0.0
    
    # Modified CSI
    features['hrv_csvi'] = features['hrv_csi'] * features['hrv_cvi']
    
    return features


def calculate_heart_rate_statistics(rr_intervals: np.ndarray) -> Dict[str, float]:
    """
    Calculate heart rate statistics from RR intervals.
    
    Args:
        rr_intervals: RR intervals in milliseconds
        
    Returns:
        Dictionary of heart rate statistics
    """
    features = {}
    
    if len(rr_intervals) == 0:
        return {
            'hr_mean': 0.0, 'hr_min': 0.0, 'hr_max': 0.0,
            'hr_std': 0.0, 'hr_range': 0.0
        }
    
    # Convert RR intervals to heart rate (bpm)
    heart_rates = 60000.0 / rr_intervals
    
    features['hr_mean'] = float(np.mean(heart_rates))
    features['hr_min'] = float(np.min(heart_rates))
    features['hr_max'] = float(np.max(heart_rates))
    features['hr_std'] = float(np.std(heart_rates))
    features['hr_range'] = float(np.ptp(heart_rates))
    
    return features


def extract_all_hrv_features(
    ecg_signal: np.ndarray,
    fs: float = 100.0
) -> Dict[str, float]:
    """
    Extract all HRV features from ECG signal.
    
    Detects R-peaks, computes RR intervals, and extracts HRV features.
    
    Args:
        ecg_signal: ECG signal (1D array)
        fs: Sampling frequency in Hz
        
    Returns:
        Dictionary of all HRV features
        
    Examples:
        >>> # Create synthetic ECG with regular peaks
        >>> signal = np.zeros(1000)
        >>> for i in range(0, 1000, 80):  # ~75 bpm
        ...     signal[i] = 1.0
        >>> features = extract_all_hrv_features(signal, fs=100)
        >>> features['hr_mean'] > 0
        True
    """
    # Detect R-peaks
    r_peaks = detect_r_peaks(ecg_signal, fs)
    
    # Compute RR intervals
    rr_intervals = compute_rr_intervals(r_peaks, fs)
    
    # Filter invalid intervals
    rr_intervals = filter_rr_intervals(rr_intervals)
    
    # Extract features
    features = {}
    features['rr_count'] = len(rr_intervals)
    features['r_peak_count'] = len(r_peaks)
    
    features.update(extract_time_domain_hrv(rr_intervals))
    features.update(extract_geometric_hrv(rr_intervals))
    features.update(extract_poincare_hrv(rr_intervals))
    features.update(calculate_heart_rate_statistics(rr_intervals))
    
    return features

