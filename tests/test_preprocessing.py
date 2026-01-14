"""
Unit Tests for ECG Preprocessing Module

Tests for filtering, normalization, and quality assessment functions.
"""

import numpy as np
import pytest
from src.preprocessing.filtering import (
    bandpass_filter,
    notch_filter,
    remove_baseline_wander,
    preprocess_ecg,
    normalize_signal
)
from src.preprocessing.quality import (
    detect_flatline,
    detect_saturation,
    detect_excessive_noise,
    calculate_snr,
    calculate_quality_metrics,
    is_good_quality
)


class TestBandpassFilter:
    """Tests for bandpass_filter function."""
    
    def test_bandpass_filter_shape_1d(self):
        """Test that output shape matches input for 1D signal."""
        signal = np.random.randn(1000)
        filtered = bandpass_filter(signal, fs=100)
        assert filtered.shape == signal.shape
    
    def test_bandpass_filter_shape_2d(self):
        """Test that output shape matches input for 2D signal (multi-lead)."""
        signal = np.random.randn(1000, 12)  # 12-lead ECG
        filtered = bandpass_filter(signal, fs=100)
        assert filtered.shape == signal.shape
    
    def test_bandpass_removes_low_freq(self):
        """Test that bandpass filter removes low-frequency components."""
        # Create signal with low-frequency component (0.1 Hz)
        t = np.arange(0, 10, 0.01)  # 10 seconds at 100 Hz
        low_freq = np.sin(2 * np.pi * 0.1 * t)
        signal = low_freq
        
        filtered = bandpass_filter(signal, lowcut=0.5, highcut=40, fs=100)
        
        # Filtered signal should have much lower amplitude
        assert np.std(filtered) < 0.5 * np.std(signal)
    
    def test_bandpass_preserves_ecg_freq(self):
        """Test that bandpass filter preserves ECG frequencies."""
        # Create signal in ECG range (2 Hz, typical heart rate)
        t = np.arange(0, 10, 0.01)
        ecg_freq = np.sin(2 * np.pi * 2 * t)
        
        filtered = bandpass_filter(ecg_freq, lowcut=0.5, highcut=40, fs=100)
        
        # Should preserve most of the signal
        correlation = np.corrcoef(ecg_freq, filtered)[0, 1]
        assert correlation > 0.9
    
    def test_bandpass_invalid_cutoff(self):
        """Test that invalid cutoff frequencies raise errors."""
        signal = np.random.randn(1000)
        
        with pytest.raises(ValueError):
            bandpass_filter(signal, lowcut=-1, highcut=40, fs=100)
        
        with pytest.raises(ValueError):
            bandpass_filter(signal, lowcut=0.5, highcut=60, fs=100)  # > Nyquist
        
        with pytest.raises(ValueError):
            bandpass_filter(signal, lowcut=40, highcut=0.5, fs=100)  # low > high


class TestNotchFilter:
    """Tests for notch_filter function."""
    
    def test_notch_removes_powerline(self):
        """Test that notch filter removes powerline interference."""
        # Create signal with 40 Hz interference (below Nyquist for fs=100)
        t = np.arange(0, 10, 0.01)
        clean_signal = np.sin(2 * np.pi * 2 * t)  # 2 Hz ECG-like
        powerline = 0.5 * np.sin(2 * np.pi * 40 * t)  # 40 Hz interference
        noisy_signal = clean_signal + powerline
        
        filtered = notch_filter(noisy_signal, freq=40, fs=100)
        
        # Filtered should be closer to clean signal
        error_before = np.mean((noisy_signal - clean_signal) ** 2)
        error_after = np.mean((filtered - clean_signal) ** 2)
        assert error_after < error_before


class TestNormalization:
    """Tests for normalize_signal function."""
    
    def test_zscore_normalization(self):
        """Test z-score normalization produces zero mean, unit variance."""
        signal = np.random.randn(1000) * 5 + 10
        normalized = normalize_signal(signal, method='zscore')
        
        assert np.abs(np.mean(normalized)) < 0.01
        assert np.abs(np.std(normalized) - 1.0) < 0.01
    
    def test_minmax_normalization(self):
        """Test min-max normalization produces [0, 1] range."""
        signal = np.random.randn(1000) * 5 + 10
        normalized = normalize_signal(signal, method='minmax')
        
        assert np.min(normalized) >= -0.01
        assert np.max(normalized) <= 1.01
        assert np.abs(np.min(normalized)) < 0.01
        assert np.abs(np.max(normalized) - 1.0) < 0.01
    
    def test_robust_normalization(self):
        """Test robust normalization using median and IQR."""
        signal = np.random.randn(1000)
        normalized = normalize_signal(signal, method='robust')
        
        # Should have median near 0
        assert np.abs(np.median(normalized)) < 0.1
    
    def test_normalization_2d(self):
        """Test normalization works with multi-lead signals."""
        signal = np.random.randn(1000, 12)
        normalized = normalize_signal(signal, method='zscore')
        
        assert normalized.shape == signal.shape


class TestFlatlineDetection:
    """Tests for flatline detection."""
    
    def test_detect_perfect_flatline(self):
        """Test detection of perfect flat-line (constant signal)."""
        flat_signal = np.ones(1000)
        assert detect_flatline(flat_signal) == True
    
    def test_detect_normal_signal(self):
        """Test that normal signal is not flagged as flat-line."""
        normal_signal = np.random.randn(1000)
        assert detect_flatline(normal_signal) == False
    
    def test_detect_partial_flatline(self):
        """Test detection of partial flat-line."""
        signal = np.random.randn(1000)
        signal[400:600] = 0.0  # Flat segment
        assert detect_flatline(signal, min_windows=1) == True


class TestSaturationDetection:
    """Tests for saturation detection."""
    
    def test_detect_saturation_clipped(self):
        """Test detection of clipped/saturated signal."""
        signal = np.random.randn(1000)
        saturated = np.clip(signal, -1, 1)
        saturated[0:150] = 1.0  # Force 15% at max
        
        assert detect_saturation(saturated, saturation_threshold=0.1) == True
    
    def test_no_saturation_normal(self):
        """Test that normal signal is not flagged as saturated."""
        signal = np.random.randn(1000)
        assert detect_saturation(signal) == False


class TestExcessiveNoiseDetection:
    """Tests for excessive noise detection."""
    
    def test_detect_excessive_noise(self):
        """Test detection of excessively noisy signal."""
        # Clean signal with primarily high-frequency content
        t = np.arange(0, 10, 0.01)
        clean = np.sin(2 * np.pi * 2 * t)
        
        # Add high-frequency noise (45 Hz, above clinical band)
        high_freq_noise = 3 * np.sin(2 * np.pi * 45 * t)
        noisy = clean + high_freq_noise
        
        assert detect_excessive_noise(noisy, fs=100, noise_threshold=0.5) == True
    
    def test_no_excessive_noise_clean(self):
        """Test that clean signal is not flagged as noisy."""
        t = np.arange(0, 10, 0.01)
        clean = np.sin(2 * np.pi * 2 * t) + 0.01 * np.random.randn(len(t))
        
        assert detect_excessive_noise(clean, fs=100) == False


class TestSNRCalculation:
    """Tests for SNR calculation."""
    
    def test_snr_clean_signal(self):
        """Test that clean signal has high SNR."""
        t = np.arange(0, 10, 0.01)
        clean = np.sin(2 * np.pi * 2 * t) + 0.01 * np.random.randn(len(t))
        
        snr = calculate_snr(clean, fs=100)
        assert snr > 10  # Should have SNR > 10 dB
    
    def test_snr_noisy_signal(self):
        """Test that noisy signal has low SNR."""
        t = np.arange(0, 10, 0.01)
        noisy = np.sin(2 * np.pi * 2 * t) + np.random.randn(len(t))
        
        snr = calculate_snr(noisy, fs=100)
        assert snr < 10  # Should have SNR < 10 dB


class TestQualityMetrics:
    """Tests for comprehensive quality metrics."""
    
    def test_calculate_quality_metrics(self):
        """Test that all quality metrics are calculated."""
        signal = np.random.randn(1000)
        metrics = calculate_quality_metrics(signal, fs=100)
        
        required_keys = ['snr_db', 'variance', 'kurtosis', 'skewness', 
                        'has_flatline', 'has_saturation', 'has_excessive_noise']
        
        for key in required_keys:
            assert key in metrics
    
    def test_is_good_quality_good_signal(self):
        """Test that good signal passes quality check."""
        # Create good ECG-like signal
        t = np.arange(0, 10, 0.01)
        signal = np.sin(2 * np.pi * 2 * t) + 0.05 * np.random.randn(len(t))
        
        is_good, metrics = is_good_quality(signal, fs=100)
        assert is_good == True
    
    def test_is_good_quality_bad_signal(self):
        """Test that bad signal fails quality check."""
        # Flat signal
        bad_signal = np.ones(1000)
        
        is_good, metrics = is_good_quality(bad_signal, fs=100)
        assert is_good == False


class TestPreprocessingPipeline:
    """Tests for complete preprocessing pipeline."""
    
    def test_preprocess_ecg_output_shape(self):
        """Test that preprocessing preserves signal shape."""
        signal = np.random.randn(1000)
        processed = preprocess_ecg(signal, fs=100)
        
        assert processed.shape == signal.shape
    
    def test_preprocess_ecg_with_notch(self):
        """Test preprocessing with notch filter."""
        signal = np.random.randn(1000)
        processed = preprocess_ecg(signal, fs=100, apply_notch=True, powerline_freq=60)
        
        assert processed.shape == signal.shape


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

