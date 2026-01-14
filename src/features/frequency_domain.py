"""
Frequency Domain Feature Extraction Module

Extracts frequency-domain features from ECG signals using FFT analysis.
"""

import numpy as np
from scipy import signal as scipy_signal
from typing import Dict, Tuple


def compute_power_spectrum(
    ecg_signal: np.ndarray,
    fs: float = 100.0,
    method: str = 'welch'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute power spectrum of ECG signal.
    
    Args:
        ecg_signal: ECG signal (1D array)
        fs: Sampling frequency in Hz
        method: 'welch' or 'fft'
        
    Returns:
        Tuple of (frequencies, power spectral density)
    """
    if method == 'welch':
        nperseg = min(256, len(ecg_signal))
        freqs, psd = scipy_signal.welch(ecg_signal, fs=fs, nperseg=nperseg)
    else:  # FFT
        fft = np.fft.fft(ecg_signal)
        freqs = np.fft.fftfreq(len(ecg_signal), 1/fs)
        psd = np.abs(fft) ** 2
        
        # Only positive frequencies
        positive_mask = freqs >= 0
        freqs = freqs[positive_mask]
        psd = psd[positive_mask]
    
    return freqs, psd


def extract_frequency_band_power(
    ecg_signal: np.ndarray,
    fs: float = 100.0
) -> Dict[str, float]:
    """
    Extract power in different frequency bands.
    
    Args:
        ecg_signal: ECG signal (1D array)
        fs: Sampling frequency in Hz
        
    Returns:
        Dictionary of frequency band powers
    """
    features = {}
    
    freqs, psd = compute_power_spectrum(ecg_signal, fs)
    
    # Define frequency bands relevant for ECG
    # VLF: Very Low Frequency (0.003-0.04 Hz) - not applicable for 10s signals
    # LF: Low Frequency (0.04-0.15 Hz)
    # HF: High Frequency (0.15-0.4 Hz)
    # QRS band: 5-15 Hz (main QRS complex energy)
    # T-wave band: 1-7 Hz
    
    # Total power
    features['total_power'] = float(np.sum(psd))
    
    # Low frequency band (0.04-0.15 Hz)
    lf_mask = (freqs >= 0.04) & (freqs < 0.15)
    features['lf_power'] = float(np.sum(psd[lf_mask])) if np.any(lf_mask) else 0.0
    
    # High frequency band (0.15-0.4 Hz)
    hf_mask = (freqs >= 0.15) & (freqs < 0.4)
    features['hf_power'] = float(np.sum(psd[hf_mask])) if np.any(hf_mask) else 0.0
    
    # QRS band (5-15 Hz)
    qrs_mask = (freqs >= 5) & (freqs <= 15)
    features['qrs_power'] = float(np.sum(psd[qrs_mask])) if np.any(qrs_mask) else 0.0
    
    # T-wave band (1-7 Hz)
    twave_mask = (freqs >= 1) & (freqs <= 7)
    features['twave_power'] = float(np.sum(psd[twave_mask])) if np.any(twave_mask) else 0.0
    
    # Clinical band (0.5-40 Hz)
    clinical_mask = (freqs >= 0.5) & (freqs <= 40)
    features['clinical_power'] = float(np.sum(psd[clinical_mask])) if np.any(clinical_mask) else 0.0
    
    # Normalized powers
    if features['total_power'] > 0:
        features['lf_norm'] = features['lf_power'] / features['total_power']
        features['hf_norm'] = features['hf_power'] / features['total_power']
        features['qrs_norm'] = features['qrs_power'] / features['total_power']
    else:
        features['lf_norm'] = 0.0
        features['hf_norm'] = 0.0
        features['qrs_norm'] = 0.0
    
    # LF/HF ratio
    features['lf_hf_ratio'] = features['lf_power'] / features['hf_power'] if features['hf_power'] > 0 else 0.0
    
    return features


def extract_spectral_features(
    ecg_signal: np.ndarray,
    fs: float = 100.0
) -> Dict[str, float]:
    """
    Extract spectral characteristics features.
    
    Args:
        ecg_signal: ECG signal (1D array)
        fs: Sampling frequency in Hz
        
    Returns:
        Dictionary of spectral features
    """
    features = {}
    
    freqs, psd = compute_power_spectrum(ecg_signal, fs)
    
    # Avoid division by zero
    psd_sum = np.sum(psd)
    if psd_sum == 0:
        return {
            'spectral_centroid': 0.0,
            'spectral_spread': 0.0,
            'spectral_entropy': 0.0,
            'spectral_flatness': 0.0,
            'spectral_rolloff': 0.0,
            'spectral_kurtosis': 0.0
        }
    
    # Spectral centroid (center of mass of spectrum)
    features['spectral_centroid'] = float(np.sum(freqs * psd) / psd_sum)
    
    # Spectral spread (spread around centroid)
    centroid = features['spectral_centroid']
    features['spectral_spread'] = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * psd) / psd_sum))
    
    # Spectral entropy
    psd_norm = psd / psd_sum
    psd_norm = psd_norm[psd_norm > 0]  # Remove zeros for log
    features['spectral_entropy'] = float(-np.sum(psd_norm * np.log2(psd_norm + 1e-10)))
    
    # Spectral flatness (geometric mean / arithmetic mean)
    psd_positive = psd[psd > 0]
    if len(psd_positive) > 0:
        geometric_mean = np.exp(np.mean(np.log(psd_positive + 1e-10)))
        arithmetic_mean = np.mean(psd)
        features['spectral_flatness'] = float(geometric_mean / (arithmetic_mean + 1e-10))
    else:
        features['spectral_flatness'] = 0.0
    
    # Spectral rolloff (frequency below which 85% of energy is contained)
    cumsum_psd = np.cumsum(psd)
    rolloff_threshold = 0.85 * cumsum_psd[-1]
    rolloff_idx = np.where(cumsum_psd >= rolloff_threshold)[0]
    features['spectral_rolloff'] = float(freqs[rolloff_idx[0]]) if len(rolloff_idx) > 0 else 0.0
    
    # Spectral kurtosis
    if len(psd) > 3:
        psd_centered = psd - np.mean(psd)
        psd_std = np.std(psd)
        if psd_std > 0:
            features['spectral_kurtosis'] = float(np.mean((psd_centered / psd_std) ** 4))
        else:
            features['spectral_kurtosis'] = 0.0
    else:
        features['spectral_kurtosis'] = 0.0
    
    return features


def extract_dominant_frequencies(
    ecg_signal: np.ndarray,
    fs: float = 100.0,
    n_peaks: int = 3
) -> Dict[str, float]:
    """
    Extract dominant frequency peaks.
    
    Args:
        ecg_signal: ECG signal (1D array)
        fs: Sampling frequency in Hz
        n_peaks: Number of dominant peaks to extract
        
    Returns:
        Dictionary of dominant frequency features
    """
    features = {}
    
    freqs, psd = compute_power_spectrum(ecg_signal, fs)
    
    # Find peaks in power spectrum
    peaks, properties = scipy_signal.find_peaks(psd, height=np.max(psd) * 0.1)
    
    if len(peaks) > 0:
        # Sort by height
        sorted_indices = np.argsort(properties['peak_heights'])[::-1]
        top_peaks = peaks[sorted_indices[:min(n_peaks, len(peaks))]]
        
        for i in range(n_peaks):
            if i < len(top_peaks):
                peak_idx = top_peaks[i]
                features[f'dominant_freq_{i+1}'] = float(freqs[peak_idx])
                features[f'dominant_power_{i+1}'] = float(psd[peak_idx])
            else:
                features[f'dominant_freq_{i+1}'] = 0.0
                features[f'dominant_power_{i+1}'] = 0.0
    else:
        for i in range(n_peaks):
            features[f'dominant_freq_{i+1}'] = 0.0
            features[f'dominant_power_{i+1}'] = 0.0
    
    return features


def extract_all_frequency_domain_features(
    ecg_signal: np.ndarray,
    fs: float = 100.0
) -> Dict[str, float]:
    """
    Extract all frequency-domain features from ECG signal.
    
    Args:
        ecg_signal: ECG signal (1D array)
        fs: Sampling frequency in Hz
        
    Returns:
        Dictionary of all frequency-domain features
        
    Examples:
        >>> signal = np.sin(2 * np.pi * 2 * np.arange(1000) / 100)
        >>> features = extract_all_frequency_domain_features(signal, fs=100)
        >>> 'total_power' in features
        True
    """
    features = {}
    
    features.update(extract_frequency_band_power(ecg_signal, fs))
    features.update(extract_spectral_features(ecg_signal, fs))
    features.update(extract_dominant_frequencies(ecg_signal, fs))
    
    return features

