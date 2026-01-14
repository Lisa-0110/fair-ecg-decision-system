"""
ECG Signal Filtering Module

Provides functions for filtering ECG signals to remove noise and artifacts
while preserving clinically relevant features.
"""

import numpy as np
from scipy import signal
from typing import Tuple, Optional


def bandpass_filter(
    ecg_signal: np.ndarray,
    lowcut: float = 0.5,
    highcut: float = 40.0,
    fs: float = 100.0,
    order: int = 4
) -> np.ndarray:
    """
    Apply Butterworth bandpass filter to ECG signal.
    
    The bandpass filter removes baseline wander (low frequencies) and 
    high-frequency noise while preserving the QRS complex and other 
    clinically relevant features.
    
    Args:
        ecg_signal: Input ECG signal (1D array for single lead, 2D for multi-lead)
        lowcut: Low cutoff frequency in Hz (default: 0.5 Hz for baseline wander)
        highcut: High cutoff frequency in Hz (default: 40 Hz for muscle noise)
        fs: Sampling frequency in Hz (default: 100 Hz for PTB-XL)
        order: Filter order (default: 4, higher = steeper cutoff)
        
    Returns:
        Filtered ECG signal with same shape as input
        
    Raises:
        ValueError: If cutoff frequencies are invalid
        
    Examples:
        >>> signal = np.random.randn(1000)
        >>> filtered = bandpass_filter(signal, fs=100)
        >>> filtered.shape
        (1000,)
    """
    # Validate inputs
    if lowcut <= 0:
        raise ValueError(f"Low cutoff must be positive, got {lowcut}")
    if highcut >= fs / 2:
        raise ValueError(f"High cutoff must be less than Nyquist frequency ({fs/2} Hz), got {highcut}")
    if lowcut >= highcut:
        raise ValueError(f"Low cutoff ({lowcut}) must be less than high cutoff ({highcut})")
    
    # Normalize cutoff frequencies to Nyquist frequency
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Design Butterworth bandpass filter
    b, a = signal.butter(order, [low, high], btype='band')
    
    # Apply zero-phase filtering (forward-backward)
    # Works with both 1D and 2D arrays
    if ecg_signal.ndim == 1:
        filtered_signal = signal.filtfilt(b, a, ecg_signal)
    elif ecg_signal.ndim == 2:
        # Apply to each lead separately
        filtered_signal = np.array([signal.filtfilt(b, a, lead) for lead in ecg_signal.T]).T
    else:
        raise ValueError(f"Expected 1D or 2D array, got {ecg_signal.ndim}D")
    
    return filtered_signal


def notch_filter(
    ecg_signal: np.ndarray,
    freq: float = 60.0,
    fs: float = 100.0,
    quality_factor: float = 30.0
) -> np.ndarray:
    """
    Apply notch filter to remove powerline interference.
    
    Removes narrow-band interference at a specific frequency (typically 50 or 60 Hz
    powerline noise) while minimally affecting adjacent frequencies.
    
    Args:
        ecg_signal: Input ECG signal
        freq: Frequency to remove in Hz (50 or 60 for powerline)
        fs: Sampling frequency in Hz
        quality_factor: Q factor determining notch width (higher = narrower)
        
    Returns:
        Filtered ECG signal
        
    Examples:
        >>> signal_with_noise = np.random.randn(1000) + np.sin(2*np.pi*60*np.arange(1000)/100)
        >>> clean = notch_filter(signal_with_noise, freq=60, fs=100)
    """
    if freq >= fs / 2:
        raise ValueError(f"Notch frequency must be less than Nyquist frequency ({fs/2} Hz)")
    
    # Design notch filter
    b, a = signal.iirnotch(freq, quality_factor, fs)
    
    # Apply filter
    if ecg_signal.ndim == 1:
        filtered_signal = signal.filtfilt(b, a, ecg_signal)
    elif ecg_signal.ndim == 2:
        filtered_signal = np.array([signal.filtfilt(b, a, lead) for lead in ecg_signal.T]).T
    else:
        raise ValueError(f"Expected 1D or 2D array, got {ecg_signal.ndim}D")
    
    return filtered_signal


def remove_baseline_wander(
    ecg_signal: np.ndarray,
    fs: float = 100.0,
    cutoff: float = 0.5
) -> np.ndarray:
    """
    Remove baseline wander using high-pass filter.
    
    Baseline wander is low-frequency drift caused by respiration,
    patient movement, or electrode contact issues.
    
    Args:
        ecg_signal: Input ECG signal
        fs: Sampling frequency in Hz
        cutoff: Cutoff frequency in Hz (typically 0.5 Hz)
        
    Returns:
        Signal with baseline wander removed
    """
    if cutoff >= fs / 2:
        raise ValueError(f"Cutoff must be less than Nyquist frequency ({fs/2} Hz)")
    
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    
    # Design high-pass filter
    b, a = signal.butter(1, normal_cutoff, btype='high')
    
    # Apply filter
    if ecg_signal.ndim == 1:
        filtered_signal = signal.filtfilt(b, a, ecg_signal)
    elif ecg_signal.ndim == 2:
        filtered_signal = np.array([signal.filtfilt(b, a, lead) for lead in ecg_signal.T]).T
    else:
        raise ValueError(f"Expected 1D or 2D array, got {ecg_signal.ndim}D")
    
    return filtered_signal


def preprocess_ecg(
    ecg_signal: np.ndarray,
    fs: float = 100.0,
    lowcut: float = 0.5,
    highcut: float = 40.0,
    apply_notch: bool = False,
    powerline_freq: float = 60.0
) -> np.ndarray:
    """
    Complete ECG preprocessing pipeline.
    
    Applies the full preprocessing chain: baseline removal, bandpass filtering,
    and optional notch filtering for powerline interference.
    
    Args:
        ecg_signal: Raw ECG signal
        fs: Sampling frequency in Hz
        lowcut: Low cutoff for bandpass filter in Hz
        highcut: High cutoff for bandpass filter in Hz
        apply_notch: Whether to apply notch filter
        powerline_freq: Powerline frequency (50 or 60 Hz)
        
    Returns:
        Preprocessed ECG signal
        
    Examples:
        >>> raw_ecg = np.random.randn(1000)
        >>> clean_ecg = preprocess_ecg(raw_ecg, fs=100)
    """
    # Apply bandpass filter (includes baseline removal)
    signal_filtered = bandpass_filter(ecg_signal, lowcut, highcut, fs)
    
    # Apply notch filter if requested
    if apply_notch and powerline_freq < fs / 2:
        signal_filtered = notch_filter(signal_filtered, freq=powerline_freq, fs=fs)
    
    return signal_filtered


def normalize_signal(
    ecg_signal: np.ndarray,
    method: str = 'zscore'
) -> np.ndarray:
    """
    Normalize ECG signal amplitude.
    
    Args:
        ecg_signal: Input ECG signal
        method: Normalization method
            - 'zscore': Zero mean, unit variance (z-score normalization)
            - 'minmax': Scale to [0, 1] range
            - 'robust': Use median and IQR (robust to outliers)
            
    Returns:
        Normalized ECG signal
        
    Examples:
        >>> signal = np.array([1, 2, 3, 4, 5])
        >>> normalized = normalize_signal(signal, method='zscore')
        >>> np.abs(normalized.mean()) < 1e-10  # approximately zero mean
        True
    """
    if method == 'zscore':
        # Z-score normalization
        mean = np.mean(ecg_signal, axis=0)
        std = np.std(ecg_signal, axis=0)
        # Avoid division by zero
        std = np.where(std == 0, 1, std)
        normalized = (ecg_signal - mean) / std
        
    elif method == 'minmax':
        # Min-max normalization to [0, 1]
        min_val = np.min(ecg_signal, axis=0)
        max_val = np.max(ecg_signal, axis=0)
        # Avoid division by zero
        range_val = max_val - min_val
        range_val = np.where(range_val == 0, 1, range_val)
        normalized = (ecg_signal - min_val) / range_val
        
    elif method == 'robust':
        # Robust normalization using median and IQR
        median = np.median(ecg_signal, axis=0)
        q75 = np.percentile(ecg_signal, 75, axis=0)
        q25 = np.percentile(ecg_signal, 25, axis=0)
        iqr = q75 - q25
        # Avoid division by zero
        iqr = np.where(iqr == 0, 1, iqr)
        normalized = (ecg_signal - median) / iqr
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized

