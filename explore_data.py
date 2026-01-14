"""
PTB-XL Dataset Exploration - Show Examples of Extractable Data

This script demonstrates what data can be extracted from the PTB-XL dataset.
"""

import pandas as pd
import numpy as np
import wfdb
import ast
from pathlib import Path


def explore_ptbxl_data():
    """Explore and display examples of PTB-XL data."""
    
    # Set path to dataset
    data_path = Path('physionet.org/files/ptb-xl/1.0.3/')
    
    print("=" * 80)
    print("PTB-XL DATASET EXPLORATION")
    print("=" * 80)
    
    # Load metadata
    print("\n1. LOADING METADATA...")
    df = pd.read_csv(data_path / 'ptbxl_database.csv', index_col='ecg_id')
    df['scp_codes'] = df.scp_codes.apply(lambda x: ast.literal_eval(x))
    
    print(f"✓ Total records: {len(df)}")
    print(f"✓ Unique patients: {df.patient_id.nunique()}")
    
    # Show metadata examples
    print("\n2. METADATA EXAMPLES (First 5 Records):")
    print("-" * 80)
    
    cols_to_show = ['patient_id', 'age', 'sex', 'height', 'weight', 
                    'recording_date', 'scp_codes', 'filename_lr']
    print(df[cols_to_show].head())
    
    # Sex distribution
    print("\n3. SEX DISTRIBUTION:")
    print("-" * 80)
    sex_counts = df['sex'].value_counts()
    print(f"Male (1):   {sex_counts.get(1, 0):5d} ({sex_counts.get(1, 0)/len(df)*100:.1f}%)")
    print(f"Female (0): {sex_counts.get(0, 0):5d} ({sex_counts.get(0, 0)/len(df)*100:.1f}%)")
    
    # Age distribution
    print("\n4. AGE DISTRIBUTION:")
    print("-" * 80)
    print(f"Mean age: {df['age'].mean():.1f} years")
    print(f"Min age:  {df['age'].min():.0f} years")
    print(f"Max age:  {df['age'].max():.0f} years")
    print(f"Median:   {df['age'].median():.0f} years")
    
    age_groups = pd.cut(df['age'], bins=[0, 40, 65, 120], labels=['≤40', '41-65', '>65'])
    print("\nAge groups:")
    for group, count in age_groups.value_counts().sort_index().items():
        print(f"  {group:6s}: {count:5d} ({count/len(df)*100:.1f}%)")
    
    # Load diagnostic classes
    agg_df = pd.read_csv(data_path / 'scp_statements.csv', index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]
    
    def aggregate_diagnostic(y_dict):
        tmp = []
        for key in y_dict.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))
    
    df['diagnostic_superclass'] = df.scp_codes.apply(aggregate_diagnostic)
    
    # Diagnostic distribution
    print("\n5. DIAGNOSTIC SUPERCLASS DISTRIBUTION:")
    print("-" * 80)
    diag_counts = {}
    for diag_list in df['diagnostic_superclass']:
        for diag in diag_list:
            diag_counts[diag] = diag_counts.get(diag, 0) + 1
    
    for diag, count in sorted(diag_counts.items(), key=lambda x: -x[1]):
        print(f"  {diag:6s}: {count:5d} ({count/len(df)*100:.1f}%)")
    
    # Top specific diagnoses
    print("\n6. TOP 10 SPECIFIC DIAGNOSES:")
    print("-" * 80)
    all_codes = {}
    for codes_dict in df['scp_codes']:
        for code, confidence in codes_dict.items():
            if code in agg_df.index:  # Only diagnostic codes
                all_codes[code] = all_codes.get(code, 0) + 1
    
    for code, count in sorted(all_codes.items(), key=lambda x: -x[1])[:10]:
        desc = agg_df.loc[code, 'description'] if code in agg_df.index else 'Unknown'
        superclass = agg_df.loc[code, 'diagnostic_class'] if code in agg_df.index else 'N/A'
        print(f"  {code:6s} ({superclass:4s}): {count:5d} - {desc}")
    
    # Load a few actual ECG signals
    print("\n7. LOADING ACTUAL ECG SIGNALS (3 Examples):")
    print("-" * 80)
    
    sample_records = df.head(3)
    
    for idx, (ecg_id, record) in enumerate(sample_records.iterrows()):
        print(f"\nRecord {idx+1} (ECG ID: {ecg_id}):")
        print(f"  Patient: {record['patient_id']}, Age: {record['age']}, Sex: {'Male' if record['sex']==1 else 'Female'}")
        print(f"  Diagnoses: {record['diagnostic_superclass']}")
        print(f"  SCP Codes: {list(record['scp_codes'].keys())}")
        
        # Load ECG signal
        filepath = data_path / record['filename_lr']
        signal, meta = wfdb.rdsamp(str(filepath))
        
        print(f"  Signal shape: {signal.shape} (samples × leads)")
        print(f"  Duration: {signal.shape[0] / 100:.1f} seconds @ 100 Hz")
        print(f"  Leads: {meta['sig_name']}")
        print(f"  Signal range: [{signal.min():.3f}, {signal.max():.3f}] mV")
        
        # Calculate some basic statistics per lead
        print(f"  Lead statistics (first 3 leads):")
        for i, lead in enumerate(meta['sig_name'][:3]):
            lead_data = signal[:, i]
            print(f"    {lead:4s}: mean={lead_data.mean():7.3f}, std={lead_data.std():.3f}, "
                  f"range=[{lead_data.min():.3f}, {lead_data.max():.3f}]")
    
    # Show what we can extract for ML
    print("\n8. EXTRACTABLE FEATURES FOR MACHINE LEARNING:")
    print("-" * 80)
    
    # Load one record for demonstration
    first_record = df.iloc[0]
    filepath = data_path / first_record['filename_lr']
    signal, meta = wfdb.rdsamp(str(filepath))
    
    print("\n✓ From Raw Signal:")
    print("  - 12-lead ECG waveforms (1000 samples × 12 leads)")
    print("  - Sampling rate: 100 Hz")
    print("  - Duration: 10 seconds")
    print("  - Amplitude values in mV")
    
    print("\n✓ Time-Domain Features (per lead/beat):")
    lead_0 = signal[:, 0]  # Lead I
    print(f"  - Mean amplitude: {lead_0.mean():.4f}")
    print(f"  - Standard deviation: {lead_0.std():.4f}")
    print(f"  - Min/Max: {lead_0.min():.4f} / {lead_0.max():.4f}")
    print(f"  - RMS: {np.sqrt(np.mean(lead_0**2)):.4f}")
    print(f"  - Variance: {lead_0.var():.4f}")
    
    print("\n✓ Frequency-Domain Features:")
    fft = np.fft.fft(lead_0)
    freqs = np.fft.fftfreq(len(lead_0), 1/100)
    power = np.abs(fft)**2
    print(f"  - Total power: {power.sum():.2e}")
    print(f"  - Dominant frequency: {abs(freqs[np.argmax(power[1:])+1]):.2f} Hz")
    
    print("\n✓ Statistical Features:")
    print(f"  - Skewness: {((lead_0 - lead_0.mean())/lead_0.std())**3}.mean() = ...")
    print(f"  - Kurtosis: {((lead_0 - lead_0.mean())/lead_0.std())**4}.mean() = ...")
    print(f"  - Percentiles: 25th, 50th, 75th")
    
    print("\n✓ Clinical Metadata:")
    print(f"  - Age: {first_record['age']}")
    print(f"  - Sex: {first_record['sex']}")
    print(f"  - Height: {first_record['height']} cm")
    print(f"  - Weight: {first_record['weight']} kg")
    
    print("\n✓ Diagnostic Labels:")
    print(f"  - Superclass: {first_record['diagnostic_superclass']}")
    print(f"  - SCP codes with confidence: {first_record['scp_codes']}")
    
    print("\n✓ Potential Derived Features:")
    print("  - Heart Rate Variability (HRV) metrics")
    print("  - R-peak detection and RR intervals")
    print("  - QRS complex duration")
    print("  - ST segment analysis")
    print("  - T-wave amplitude")
    print("  - P-wave detection")
    print("  - Entropy measures (Sample, Approximate, Permutation)")
    print("  - Wavelet coefficients")
    print("  - Morphological features")
    
    # Summary statistics
    print("\n9. DATASET SUMMARY:")
    print("-" * 80)
    print(f"✓ {len(df)} ECG records from {df.patient_id.nunique()} patients")
    print(f"✓ {len(df['sex'].unique())} sex categories")
    print(f"✓ Age range: {df['age'].min():.0f}-{df['age'].max():.0f} years")
    print(f"✓ {len(diag_counts)} diagnostic superclasses")
    print(f"✓ {len(all_codes)} unique diagnostic codes")
    print(f"✓ 12 leads per record")
    print(f"✓ 10 seconds @ 100 Hz = 1000 samples per lead")
    print(f"✓ Total data points per record: 12,000")
    
    print("\n" + "=" * 80)
    print("READY FOR ANALYSIS!")
    print("=" * 80)


if __name__ == "__main__":
    explore_ptbxl_data()

