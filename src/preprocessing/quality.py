"""
ECG Signal Quality Assessment Module

Provides functions for assessing ECG signal quality and detecting artifacts.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from scipy import stats


def detect_flatline(
    ecg_signal: np.ndarray,
    threshold: float = 0.001,
    window_size: int = 100,
    min_windows: int = 1
) -> bool:
    """
    Detect flat-line artifacts in ECG signal.
    
    A flat-line occurs when the signal has very low variance, indicating
    electrode disconnection or equipment failure.
    
    Args:
        ecg_signal: Input ECG signal
        threshold: Variance threshold below which signal is considered flat
        window_size: Size of analysis window in samples
        min_windows: Minimum number of flat windows to trigger detection
        
    Returns:
        True if flat-line detected, False otherwise
        
    Examples:
        >>> flat_signal = np.ones(1000)
        >>> detect_flatline(flat_signal)
        True
        >>> normal_signal = np.random.randn(1000)
        >>> detect_flatline(normal_signal)
        False
    """
    if len(ecg_signal) < window_size:
        return np.var(ecg_signal) < threshold
    
    flat_count = 0
    
    # Analyze signal in windows
    for i in range(0, len(ecg_signal) - window_size, window_size // 2):
        window = ecg_signal[i:i + window_size]
        if np.var(window) < threshold:
            flat_count += 1
            if flat_count >= min_windows:
                return True
    
    return False


def detect_saturation(
    ecg_signal: np.ndarray,
    saturation_threshold: float = 0.1
) -> bool:
    """
    Detect signal saturation (clipping).
    
    Saturation occurs when the signal amplitude exceeds the recording
    device's range, causing clipping at min/max values.
    
    Args:
        ecg_signal: Input ECG signal
        saturation_threshold: Fraction of samples allowed at min/max
                             (e.g., 0.1 = 10% of samples)
        
    Returns:
        True if saturation detected, False otherwise
        
    Examples:
        >>> saturated = np.clip(np.random.randn(1000), -1, 1)
        >>> saturated[0:200] = 1.0  # Force saturation
        >>> detect_saturation(saturated, saturation_threshold=0.1)
        True
    """
    if len(ecg_signal) == 0:
        return False
    
    # Check if signal range is too small (all same value)
    signal_range = np.ptp(ecg_signal)
    if signal_range < 1e-10:
        return True
    
    # Normalize to [0, 1]
    normalized = (ecg_signal - np.min(ecg_signal)) / signal_range
    
    # Count samples near min (0) or max (1)
    near_min = np.sum(normalized < 0.01) / len(normalized)
    near_max = np.sum(normalized > 0.99) / len(normalized)
    
    # Check if too many samples are saturated
    return (near_min > saturation_threshold) or (near_max > saturation_threshold)


def detect_excessive_noise(
    ecg_signal: np.ndarray,
    fs: float = 100.0,
    noise_threshold: float = 0.5
) -> bool:
    """
    Detect excessive noise in ECG signal.
    
    High-frequency noise can obscure clinically relevant features.
    This function checks if the high-frequency power is too large
    relative to the clinical frequency band.
    
    Args:
        ecg_signal: Input ECG signal
        fs: Sampling frequency in Hz
        noise_threshold: Ratio threshold (high_freq_power / clinical_power)
        
    Returns:
        True if excessive noise detected, False otherwise
        
    Examples:
        >>> clean_signal = np.sin(2*np.pi*2*np.arange(1000)/100)
        >>> detect_excessive_noise(clean_signal)
        False
        >>> noisy_signal = clean_signal + 0.5*np.random.randn(1000)
        >>> detect_excessive_noise(noisy_signal)
        True
    """
    if len(ecg_signal) < 10:
        return False
    
    # Compute power spectrum
    fft = np.fft.fft(ecg_signal)
    freqs = np.fft.fftfreq(len(ecg_signal), 1/fs)
    power_spectrum = np.abs(fft) ** 2
    
    # Only consider positive frequencies
    positive_freq_mask = freqs >= 0
    freqs = freqs[positive_freq_mask]
    power_spectrum = power_spectrum[positive_freq_mask]
    
    # Clinical frequency band (0.5-40 Hz)
    clinical_mask = (freqs >= 0.5) & (freqs <= 40)
    clinical_power = np.sum(power_spectrum[clinical_mask])
    
    # High-frequency noise band (40 Hz to Nyquist)
    noise_mask = (freqs > 40) & (freqs < fs / 2)
    if np.sum(noise_mask) == 0:
        return False
    
    noise_power = np.sum(power_spectrum[noise_mask])
    
    # Check if noise power is too high relative to clinical power
    if clinical_power == 0:
        return True
    
    noise_ratio = noise_power / clinical_power
    return noise_ratio > noise_threshold


def calculate_snr(
    ecg_signal: np.ndarray,
    fs: float = 100.0
) -> float:
    """
    Calculate signal-to-noise ratio (SNR).
    
    Estimates SNR by comparing power in clinical frequency band
    to power in high-frequency noise band.
    
    Args:
        ecg_signal: Input ECG signal
        fs: Sampling frequency in Hz
        
    Returns:
        SNR in dB
        
    Examples:
        >>> clean_signal = np.sin(2*np.pi*2*np.arange(1000)/100)
        >>> snr = calculate_snr(clean_signal)
        >>> snr > 20  # Clean signal should have high SNR
        True
    """
    # Compute power spectrum
    fft = np.fft.fft(ecg_signal)
    freqs = np.fft.fftfreq(len(ecg_signal), 1/fs)
    power_spectrum = np.abs(fft) ** 2
    
    # Only consider positive frequencies
    positive_freq_mask = freqs >= 0
    freqs = freqs[positive_freq_mask]
    power_spectrum = power_spectrum[positive_freq_mask]
    
    # Signal power (clinical band: 0.5-40 Hz)
    signal_mask = (freqs >= 0.5) & (freqs <= 40)
    signal_power = np.mean(power_spectrum[signal_mask])
    
    # Noise power (high-frequency band: 40 Hz to Nyquist)
    noise_mask = (freqs > 40) & (freqs < fs / 2)
    noise_power = np.mean(power_spectrum[noise_mask]) if np.any(noise_mask) else 1e-10
    
    # Calculate SNR in dB
    if noise_power > 0:
        snr = 10 * np.log10(signal_power / noise_power)
    else:
        snr = np.inf
    
    return snr


def calculate_quality_metrics(
    ecg_signal: np.ndarray,
    fs: float = 100.0
) -> Dict[str, float]:
    """
    Calculate comprehensive quality metrics for ECG signal.
    
    Args:
        ecg_signal: Input ECG signal
        fs: Sampling frequency in Hz
        
    Returns:
        Dictionary containing quality metrics:
            - snr: Signal-to-noise ratio in dB
            - variance: Signal variance
            - kurtosis: Signal kurtosis (measure of peakedness)
            - skewness: Signal skewness (measure of asymmetry)
            - range: Signal range (max - min)
            - has_flatline: Boolean indicating flatline detection
            - has_saturation: Boolean indicating saturation
            - has_excessive_noise: Boolean indicating excessive noise
            
    Examples:
        >>> signal = np.random.randn(1000)
        >>> metrics = calculate_quality_metrics(signal)
        >>> 'snr' in metrics and 'variance' in metrics
        True
    """
    metrics = {}
    
    # Basic statistics
    metrics['variance'] = float(np.var(ecg_signal))
    metrics['std'] = float(np.std(ecg_signal))
    metrics['range'] = float(np.ptp(ecg_signal))
    metrics['mean'] = float(np.mean(ecg_signal))
    
    # Shape metrics
    if len(ecg_signal) > 3:
        metrics['kurtosis'] = float(stats.kurtosis(ecg_signal))
        metrics['skewness'] = float(stats.skew(ecg_signal))
    else:
        metrics['kurtosis'] = 0.0
        metrics['skewness'] = 0.0
    
    # SNR
    metrics['snr_db'] = float(calculate_snr(ecg_signal, fs))
    
    # Artifact detection
    metrics['has_flatline'] = detect_flatline(ecg_signal)
    metrics['has_saturation'] = detect_saturation(ecg_signal)
    metrics['has_excessive_noise'] = detect_excessive_noise(ecg_signal, fs)
    
    return metrics


def is_good_quality(
    ecg_signal: np.ndarray,
    fs: float = 100.0,
    min_snr: float = 5.0,
    min_variance: float = 0.001,
    max_kurtosis: float = 10.0
) -> Tuple[bool, Dict[str, float]]:
    """
    Determine if ECG signal is of acceptable quality.
    
    Args:
        ecg_signal: Input ECG signal
        fs: Sampling frequency in Hz
        min_snr: Minimum acceptable SNR in dB
        min_variance: Minimum acceptable variance
        max_kurtosis: Maximum acceptable kurtosis
        
    Returns:
        Tuple of (is_good_quality: bool, metrics: dict)
        
    Examples:
        >>> good_signal = np.sin(2*np.pi*2*np.arange(1000)/100) + 0.01*np.random.randn(1000)
        >>> is_good, metrics = is_good_quality(good_signal)
        >>> is_good
        True
    """
    metrics = calculate_quality_metrics(ecg_signal, fs)
    
    # Check quality criteria
    quality_checks = {
        'no_flatline': not metrics['has_flatline'],
        'no_saturation': not metrics['has_saturation'],
        'no_excessive_noise': not metrics['has_excessive_noise'],
        'sufficient_variance': metrics['variance'] >= min_variance,
        'acceptable_kurtosis': abs(metrics['kurtosis']) <= max_kurtosis,
        'sufficient_snr': metrics['snr_db'] >= min_snr
    }
    
    # Add check results to metrics
    metrics['quality_checks'] = quality_checks
    
    # Signal is good quality if all checks pass
    is_good = all(quality_checks.values())
    
    return is_good, metrics

