"""
Entropy Feature Extraction Module

Extracts entropy-based features from ECG signals for complexity analysis.
"""

import numpy as np
import math
from typing import Dict
from scipy.stats import entropy as scipy_entropy


def sample_entropy(
    signal: np.ndarray,
    m: int = 2,
    r: float = None
) -> float:
    """
    Calculate Sample Entropy of a signal.
    
    Sample Entropy measures the regularity and unpredictability of a time series.
    Lower values indicate more regular/predictable signals.
    
    Args:
        signal: Input signal
        m: Embedding dimension (pattern length)
        r: Tolerance (if None, uses 0.2 * std)
        
    Returns:
        Sample entropy value
    """
    if r is None:
        r = 0.2 * np.std(signal)
    
    N = len(signal)
    if N < m + 1:
        return 0.0
    
    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])
    
    def _phi(m_val):
        patterns = [[signal[j] for j in range(i, i + m_val)] 
                   for i in range(N - m_val + 1)]
        C = []
        for i, x_i in enumerate(patterns):
            matches = sum(1 for j, x_j in enumerate(patterns) 
                         if i != j and _maxdist(x_i, x_j) <= r)
            C.append(matches)
        return sum(C)
    
    phi_m = _phi(m)
    phi_m_plus = _phi(m + 1)
    
    if phi_m == 0 or phi_m_plus == 0:
        return 0.0
    
    return -np.log(phi_m_plus / phi_m)


def approximate_entropy(
    signal: np.ndarray,
    m: int = 2,
    r: float = None
) -> float:
    """
    Calculate Approximate Entropy of a signal.
    
    ApEn is a regularity statistic that quantifies the unpredictability
    of fluctuations in a time series.
    
    Args:
        signal: Input signal
        m: Embedding dimension
        r: Tolerance (if None, uses 0.2 * std)
        
    Returns:
        Approximate entropy value
    """
    if r is None:
        r = 0.2 * np.std(signal)
    
    N = len(signal)
    if N < m + 1:
        return 0.0
    
    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])
    
    def _phi(m_val):
        patterns = [[signal[j] for j in range(i, i + m_val)] 
                   for i in range(N - m_val + 1)]
        C = []
        for x_i in patterns:
            matches = sum(1 for x_j in patterns if _maxdist(x_i, x_j) <= r)
            C.append(matches / (N - m_val + 1))
        phi = sum([np.log(c) for c in C if c > 0]) / (N - m_val + 1)
        return phi
    
    return abs(_phi(m) - _phi(m + 1))


def shannon_entropy(signal: np.ndarray, bins: int = 50) -> float:
    """
    Calculate Shannon Entropy of a signal.
    
    Shannon entropy measures the average information content in a signal.
    
    Args:
        signal: Input signal
        bins: Number of bins for histogram
        
    Returns:
        Shannon entropy value
    """
    hist, _ = np.histogram(signal, bins=bins)
    hist = hist[hist > 0]  # Remove zero bins
    
    if len(hist) == 0:
        return 0.0
    
    prob = hist / np.sum(hist)
    return float(-np.sum(prob * np.log2(prob)))


def permutation_entropy(
    signal: np.ndarray,
    order: int = 3,
    delay: int = 1,
    normalize: bool = True
) -> float:
    """
    Calculate Permutation Entropy of a signal.
    
    Permutation entropy is a robust complexity measure based on ordinal patterns.
    
    Args:
        signal: Input signal
        order: Embedding dimension (order of permutation)
        delay: Time delay
        normalize: Whether to normalize by maximum entropy
        
    Returns:
        Permutation entropy value
    """
    N = len(signal)
    if N < (order - 1) * delay + 1:
        return 0.0
    
    # Create patterns
    n_patterns = N - (order - 1) * delay
    permutations = []
    
    for i in range(n_patterns):
        pattern = [signal[i + j * delay] for j in range(order)]
        sorted_indices = tuple(np.argsort(pattern))
        permutations.append(sorted_indices)
    
    # Count unique permutations
    unique_perms, counts = np.unique(permutations, return_counts=True, axis=0)
    probabilities = counts / n_patterns
    
    # Calculate entropy
    pe = float(-np.sum(probabilities * np.log2(probabilities)))
    
    # Normalize
    if normalize:
        max_entropy = np.log2(math.factorial(order))
        if max_entropy > 0:
            pe = pe / max_entropy
    
    return pe


def spectral_entropy(signal: np.ndarray, fs: float = 100.0) -> float:
    """
    Calculate Spectral Entropy of a signal.
    
    Spectral entropy measures the complexity of the power spectrum.
    
    Args:
        signal: Input signal
        fs: Sampling frequency in Hz
        
    Returns:
        Spectral entropy value
    """
    # Compute power spectrum
    fft = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), 1/fs)
    
    # Only positive frequencies
    positive_mask = freqs >= 0
    psd = np.abs(fft[positive_mask]) ** 2
    
    # Normalize to probability distribution
    psd_norm = psd / np.sum(psd)
    psd_norm = psd_norm[psd_norm > 0]  # Remove zeros
    
    if len(psd_norm) == 0:
        return 0.0
    
    # Calculate entropy
    return float(-np.sum(psd_norm * np.log2(psd_norm)))


def multiscale_entropy(
    signal: np.ndarray,
    max_scale: int = 3,
    m: int = 2
) -> Dict[str, float]:
    """
    Calculate Multiscale Entropy.
    
    MSE quantifies complexity across multiple time scales.
    
    Args:
        signal: Input signal
        max_scale: Maximum scale factor
        m: Embedding dimension for sample entropy
        
    Returns:
        Dictionary of entropy values at different scales
    """
    features = {}
    
    for scale in range(1, max_scale + 1):
        # Coarse-grain the signal
        if scale == 1:
            coarse_signal = signal
        else:
            n = len(signal) // scale
            if n < m + 1:
                features[f'mse_scale_{scale}'] = 0.0
                continue
            coarse_signal = np.array([
                np.mean(signal[i*scale:(i+1)*scale]) 
                for i in range(n)
            ])
        
        # Calculate sample entropy
        features[f'mse_scale_{scale}'] = sample_entropy(coarse_signal, m=m)
    
    # Calculate MSE complexity index (average across scales)
    mse_values = [features[f'mse_scale_{s}'] for s in range(1, max_scale + 1)]
    features['mse_mean'] = float(np.mean(mse_values))
    
    return features


def extract_all_entropy_features(signal: np.ndarray, fs: float = 100.0) -> Dict[str, float]:
    """
    Extract all entropy-based features from a signal.
    
    Args:
        signal: Input signal
        fs: Sampling frequency in Hz
        
    Returns:
        Dictionary of all entropy features
        
    Examples:
        >>> signal = np.random.randn(1000)
        >>> features = extract_all_entropy_features(signal, fs=100)
        >>> 'sample_entropy' in features
        True
    """
    features = {}
    
    # Basic entropy measures
    features['sample_entropy'] = sample_entropy(signal)
    features['approximate_entropy'] = approximate_entropy(signal)
    features['shannon_entropy'] = shannon_entropy(signal)
    features['permutation_entropy'] = permutation_entropy(signal)
    features['spectral_entropy'] = spectral_entropy(signal, fs)
    
    # Multiscale entropy
    features.update(multiscale_entropy(signal, max_scale=3))
    
    return features

