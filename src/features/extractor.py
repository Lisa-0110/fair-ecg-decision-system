"""
ECG Feature Extractor

Main module for extracting comprehensive ECG features and aligning them
with metadata in a structured DataFrame.
"""

import numpy as np
import pandas as pd
import wfdb
from pathlib import Path
from typing import Dict, List, Optional, Union
from tqdm import tqdm

from src.preprocessing.filtering import preprocess_ecg, normalize_signal
from src.preprocessing.quality import is_good_quality
from src.features.time_domain import extract_all_time_domain_features
from src.features.frequency_domain import extract_all_frequency_domain_features
from src.features.hrv import extract_all_hrv_features
from src.features.entropy import extract_all_entropy_features


class ECGFeatureExtractor:
    """
    Comprehensive ECG feature extraction with metadata alignment.
    """
    
    def __init__(
        self,
        fs: float = 100.0,
        apply_preprocessing: bool = True,
        apply_normalization: bool = True,
        quality_check: bool = True
    ):
        """
        Initialize feature extractor.
        
        Args:
            fs: Sampling frequency in Hz
            apply_preprocessing: Whether to preprocess signals
            apply_normalization: Whether to normalize signals
            quality_check: Whether to perform quality checks
        """
        self.fs = fs
        self.apply_preprocessing = apply_preprocessing
        self.apply_normalization = apply_normalization
        self.quality_check = quality_check
        
    def extract_features_from_signal(
        self,
        signal: np.ndarray,
        lead_name: str = 'Lead_II'
    ) -> Dict[str, float]:
        """
        Extract all features from a single ECG signal.
        
        Args:
            signal: ECG signal (1D array)
            lead_name: Name of the lead (for feature naming)
            
        Returns:
            Dictionary of features with lead-specific names
        """
        all_features = {}
        
        # Preprocess if requested
        if self.apply_preprocessing:
            signal = preprocess_ecg(signal, fs=self.fs)
        
        # Normalize if requested
        if self.apply_normalization:
            signal = normalize_signal(signal, method='zscore')
        
        # Quality check
        if self.quality_check:
            is_good, quality_metrics = is_good_quality(signal, fs=self.fs)
            all_features[f'{lead_name}_is_good_quality'] = float(is_good)
            all_features[f'{lead_name}_snr_db'] = quality_metrics['snr_db']
        
        # Extract features from each category
        time_features = extract_all_time_domain_features(signal)
        freq_features = extract_all_frequency_domain_features(signal, self.fs)
        hrv_features = extract_all_hrv_features(signal, self.fs)
        entropy_features = extract_all_entropy_features(signal, self.fs)
        
        # Add lead prefix to feature names
        for feature_dict in [time_features, freq_features, hrv_features, entropy_features]:
            for key, value in feature_dict.items():
                all_features[f'{lead_name}_{key}'] = value
        
        return all_features
    
    def extract_features_from_multilead(
        self,
        signal: np.ndarray,
        lead_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Extract features from multi-lead ECG signal.
        
        Args:
            signal: Multi-lead ECG signal (samples × leads)
            lead_names: List of lead names
            
        Returns:
            Dictionary of features from all leads
        """
        if signal.ndim == 1:
            # Single lead
            return self.extract_features_from_signal(signal, 'Lead')
        
        n_leads = signal.shape[1]
        if lead_names is None:
            lead_names = [f'Lead_{i}' for i in range(n_leads)]
        
        all_features = {}
        
        for i, lead_name in enumerate(lead_names):
            lead_signal = signal[:, i]
            lead_features = self.extract_features_from_signal(lead_signal, lead_name)
            all_features.update(lead_features)
        
        return all_features
    
    def extract_features_from_file(
        self,
        filepath: Union[str, Path],
        lead_idx: Optional[int] = None,
        lead_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Extract features from ECG file.
        
        Args:
            filepath: Path to ECG file (without extension)
            lead_idx: If specified, extract only this lead
            lead_names: Names of leads
            
        Returns:
            Dictionary of features
        """
        # Load signal
        signal, meta = wfdb.rdsamp(str(filepath))
        
        # Get lead names from metadata if not provided
        if lead_names is None and 'sig_name' in meta:
            lead_names = meta['sig_name']
        
        # Extract specific lead or all leads
        if lead_idx is not None:
            signal = signal[:, lead_idx]
            lead_name = lead_names[lead_idx] if lead_names else f'Lead_{lead_idx}'
            return self.extract_features_from_signal(signal, lead_name)
        else:
            return self.extract_features_from_multilead(signal, lead_names)
    
    def extract_features_batch(
        self,
        metadata_df: pd.DataFrame,
        data_path: Union[str, Path],
        lead_idx: Optional[int] = 1,  # Default to Lead II
        max_samples: Optional[int] = None,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Extract features from batch of ECG files with metadata.
        
        Args:
            metadata_df: DataFrame with ECG metadata (must have 'filename_lr' or 'filename_hr')
            data_path: Base path to ECG data
            lead_idx: Which lead to extract (None for all leads)
            max_samples: Maximum number of samples to process
            verbose: Whether to show progress bar
            
        Returns:
            DataFrame with features aligned with metadata
        """
        data_path = Path(data_path)
        
        # Limit samples if requested
        if max_samples is not None:
            metadata_df = metadata_df.head(max_samples)
        
        # Determine filename column
        if 'filename_lr' in metadata_df.columns:
            filename_col = 'filename_lr'
        elif 'filename_hr' in metadata_df.columns:
            filename_col = 'filename_hr'
        else:
            raise ValueError("Metadata must contain 'filename_lr' or 'filename_hr' column")
        
        features_list = []
        
        iterator = metadata_df.iterrows()
        if verbose:
            iterator = tqdm(iterator, total=len(metadata_df), desc="Extracting features")
        
        for idx, row in iterator:
            try:
                # Get filepath
                filepath = data_path / row[filename_col]
                
                # Extract features
                features = self.extract_features_from_file(filepath, lead_idx=lead_idx)
                
                # Add ECG ID
                features['ecg_id'] = idx
                
                features_list.append(features)
                
            except Exception as e:
                if verbose:
                    print(f"\nError processing ECG ID {idx}: {e}")
                continue
        
        # Create features DataFrame
        features_df = pd.DataFrame(features_list)
        
        # Set index to ecg_id
        if 'ecg_id' in features_df.columns:
            features_df = features_df.set_index('ecg_id')
        
        # Merge with metadata
        result_df = metadata_df.join(features_df, how='inner')
        
        return result_df


def extract_ecg_features(
    metadata_df: pd.DataFrame,
    data_path: Union[str, Path],
    lead_idx: int = 1,
    fs: float = 100.0,
    apply_preprocessing: bool = True,
    apply_normalization: bool = True,
    quality_check: bool = True,
    max_samples: Optional[int] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Convenience function to extract ECG features with metadata.
    
    Args:
        metadata_df: DataFrame with ECG metadata
        data_path: Path to ECG data directory
        lead_idx: Which lead to extract (1 = Lead II)
        fs: Sampling frequency in Hz
        apply_preprocessing: Apply bandpass filtering
        apply_normalization: Apply z-score normalization
        quality_check: Perform quality assessment
        max_samples: Maximum number of ECGs to process
        verbose: Show progress
        
    Returns:
        DataFrame with features and metadata
        
    Examples:
        >>> import pandas as pd
        >>> metadata = pd.read_csv('ptbxl_database.csv', index_col='ecg_id')
        >>> features_df = extract_ecg_features(
        ...     metadata.head(10),
        ...     'physionet.org/files/ptb-xl/1.0.3/',
        ...     lead_idx=1,
        ...     verbose=True
        ... )
        >>> print(f"Extracted {len(features_df.columns)} features")
    """
    extractor = ECGFeatureExtractor(
        fs=fs,
        apply_preprocessing=apply_preprocessing,
        apply_normalization=apply_normalization,
        quality_check=quality_check
    )
    
    return extractor.extract_features_batch(
        metadata_df=metadata_df,
        data_path=data_path,
        lead_idx=lead_idx,
        max_samples=max_samples,
        verbose=verbose
    )

