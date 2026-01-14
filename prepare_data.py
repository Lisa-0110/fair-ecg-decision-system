"""
PTB-XL Dataset Loader and Splitter

This script loads the PTB-XL ECG dataset and creates train/validation/test splits
that are patient-independent and stratified by diagnostic superclass, sex, and age group.
"""

import numpy as np
import pandas as pd
import wfdb
import ast
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple, List
from collections import Counter


class PTBXLDataLoader:
    """Load and split PTB-XL dataset with multiple stratification."""
    
    def __init__(self, data_path: str = './data/raw/ptb-xl/', sampling_rate: int = 100):
        """
        Initialize PTB-XL data loader.
        
        Args:
            data_path: Path to PTB-XL dataset directory
            sampling_rate: Sampling rate (100 or 500 Hz)
        """
        self.data_path = Path(data_path)
        self.sampling_rate = sampling_rate
        self.metadata = None
        self.scp_statements = None
        
    def load_metadata(self) -> pd.DataFrame:
        """
        Load PTB-XL metadata.
        
        Returns:
            DataFrame with metadata
        """
        print("Loading PTB-XL metadata...")
        
        # Load database metadata
        self.metadata = pd.read_csv(self.data_path / 'ptbxl_database.csv', index_col='ecg_id')
        
        # Load SCP statements mapping
        agg_df = pd.read_csv(self.data_path / 'scp_statements.csv', index_col=0)
        agg_df = agg_df[agg_df.diagnostic == 1]
        
        # Convert scp_codes from string to dict
        self.metadata['scp_codes'] = self.metadata.scp_codes.apply(lambda x: ast.literal_eval(x))
        
        # Add diagnostic superclass
        def aggregate_diagnostic(y_dict):
            tmp = []
            for key in y_dict.keys():
                if key in agg_df.index:
                    tmp.append(agg_df.loc[key].diagnostic_class)
            return list(set(tmp))
        
        self.metadata['diagnostic_superclass'] = self.metadata.scp_codes.apply(aggregate_diagnostic)
        
        # Create age groups
        self.metadata['age_group'] = pd.cut(
            self.metadata['age'],
            bins=[0, 40, 65, 120],
            labels=['<=40', '41-65', '>65']
        )
        
        print(f"Loaded {len(self.metadata)} records")
        print(f"Unique patients: {self.metadata.patient_id.nunique()}")
        
        return self.metadata
    
    def load_raw_data(self, record_id: int, leads: List[str] = None) -> np.ndarray:
        """
        Load raw ECG data for a specific record.
        
        Args:
            record_id: ECG record ID
            leads: List of leads to load (None = all 12 leads)
            
        Returns:
            ECG signal data
        """
        if self.sampling_rate == 100:
            data_folder = self.data_path / 'records100'
        else:
            data_folder = self.data_path / 'records500'
        
        # Get filename from metadata
        filename = self.metadata.loc[record_id, 'filename_hr' if self.sampling_rate == 100 else 'filename_lr']
        
        # Load waveform
        record = wfdb.rdsamp(str(data_folder / filename))
        data = record[0]  # Signal data
        
        if leads is not None:
            # Select specific leads
            all_leads = record[1]['sig_name']
            lead_indices = [all_leads.index(lead) for lead in leads if lead in all_leads]
            data = data[:, lead_indices]
        
        return data
    
    def create_stratification_key(self, row: pd.Series) -> str:
        """
        Create stratification key combining multiple factors.
        
        Args:
            row: Metadata row
            
        Returns:
            Stratification key string
        """
        # Get primary diagnostic superclass (use first if multiple)
        if len(row['diagnostic_superclass']) > 0:
            diag = row['diagnostic_superclass'][0]
        else:
            diag = 'NORM'  # Default to normal if no diagnosis
        
        # Combine: diagnostic_sex_agegroup
        sex = row['sex']
        age_group = row['age_group']
        
        return f"{diag}_{sex}_{age_group}"
    
    def create_patient_independent_split(
        self,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create patient-independent train/val/test split with stratification.
        
        Args:
            test_size: Proportion of test set
            val_size: Proportion of validation set (from remaining after test)
            random_state: Random seed
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if self.metadata is None:
            self.load_metadata()
        
        print("\nCreating patient-independent stratified split...")
        
        # Filter out records with missing critical information
        valid_data = self.metadata[
            (self.metadata['diagnostic_superclass'].apply(len) > 0) &
            (self.metadata['sex'].notna()) &
            (self.metadata['age_group'].notna())
        ].copy()
        
        print(f"Valid records: {len(valid_data)}")
        
        # Group by patient to ensure patient independence
        patient_groups = valid_data.groupby('patient_id').agg({
            'diagnostic_superclass': lambda x: x.iloc[0],  # Use first record's diagnosis
            'sex': 'first',
            'age_group': 'first',
            'ecg_id': list  # Keep track of all ECG IDs for this patient
        }).reset_index()
        
        print(f"Unique patients: {len(patient_groups)}")
        
        # Create stratification key for each patient
        patient_groups['strat_key'] = patient_groups.apply(
            lambda row: f"{row['diagnostic_superclass'][0] if len(row['diagnostic_superclass']) > 0 else 'NORM'}_{row['sex']}_{row['age_group']}",
            axis=1
        )
        
        # Show stratification distribution
        print("\nStratification distribution:")
        strat_counts = Counter(patient_groups['strat_key'])
        for key, count in sorted(strat_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"  {key}: {count} patients")
        
        # Remove rare stratification groups (with < 3 patients) to avoid split issues
        min_patients = 3
        valid_strat_keys = [k for k, v in strat_counts.items() if v >= min_patients]
        patient_groups_filtered = patient_groups[patient_groups['strat_key'].isin(valid_strat_keys)]
        
        print(f"\nPatients after filtering rare groups: {len(patient_groups_filtered)}")
        
        # First split: separate test set
        train_val_patients, test_patients = train_test_split(
            patient_groups_filtered,
            test_size=test_size,
            stratify=patient_groups_filtered['strat_key'],
            random_state=random_state
        )
        
        # Second split: separate validation set from training
        val_size_adjusted = val_size / (1 - test_size)  # Adjust for already removed test set
        train_patients, val_patients = train_test_split(
            train_val_patients,
            test_size=val_size_adjusted,
            stratify=train_val_patients['strat_key'],
            random_state=random_state
        )
        
        # Get ECG IDs for each split
        train_ids = [ecg_id for ecg_list in train_patients['ecg_id'] for ecg_id in ecg_list]
        val_ids = [ecg_id for ecg_list in val_patients['ecg_id'] for ecg_id in ecg_list]
        test_ids = [ecg_id for ecg_list in test_patients['ecg_id'] for ecg_id in ecg_list]
        
        # Create split dataframes
        train_df = valid_data[valid_data.index.isin(train_ids)]
        val_df = valid_data[valid_data.index.isin(val_ids)]
        test_df = valid_data[valid_data.index.isin(test_ids)]
        
        # Verify patient independence
        train_patients_set = set(train_df['patient_id'])
        val_patients_set = set(val_df['patient_id'])
        test_patients_set = set(test_df['patient_id'])
        
        assert len(train_patients_set & val_patients_set) == 0, "Patient overlap between train and val!"
        assert len(train_patients_set & test_patients_set) == 0, "Patient overlap between train and test!"
        assert len(val_patients_set & test_patients_set) == 0, "Patient overlap between val and test!"
        
        print(f"\n✓ Split created successfully (patient-independent verified)")
        print(f"  Train: {len(train_df)} records from {len(train_patients_set)} patients")
        print(f"  Val:   {len(val_df)} records from {len(val_patients_set)} patients")
        print(f"  Test:  {len(test_df)} records from {len(test_patients_set)} patients")
        
        return train_df, val_df, test_df
    
    def print_split_statistics(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        """Print statistics about the splits."""
        print("\n" + "="*80)
        print("SPLIT STATISTICS")
        print("="*80)
        
        for split_name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
            print(f"\n{split_name} Set:")
            print(f"  Total records: {len(df)}")
            print(f"  Unique patients: {df['patient_id'].nunique()}")
            
            print(f"\n  Sex distribution:")
            sex_dist = df['sex'].value_counts()
            for sex, count in sex_dist.items():
                print(f"    {sex}: {count} ({count/len(df)*100:.1f}%)")
            
            print(f"\n  Age group distribution:")
            age_dist = df['age_group'].value_counts()
            for age, count in age_dist.items():
                print(f"    {age}: {count} ({count/len(df)*100:.1f}%)")
            
            print(f"\n  Top diagnostic superclasses:")
            # Count diagnostic superclasses
            diag_counts = {}
            for diag_list in df['diagnostic_superclass']:
                for diag in diag_list:
                    diag_counts[diag] = diag_counts.get(diag, 0) + 1
            
            for diag, count in sorted(diag_counts.items(), key=lambda x: -x[1])[:5]:
                print(f"    {diag}: {count} ({count/len(df)*100:.1f}%)")
    
    def save_splits(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        output_path: str = './data/processed/'
    ):
        """
        Save split indices and metadata.
        
        Args:
            train_df: Training set DataFrame
            val_df: Validation set DataFrame
            test_df: Test set DataFrame
            output_path: Output directory path
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving splits to {output_path}...")
        
        # Save indices
        split_indices = {
            'train_ids': train_df.index.tolist(),
            'val_ids': val_df.index.tolist(),
            'test_ids': test_df.index.tolist(),
            'train_patients': train_df['patient_id'].unique().tolist(),
            'val_patients': val_df['patient_id'].unique().tolist(),
            'test_patients': test_df['patient_id'].unique().tolist()
        }
        
        with open(output_path / 'split_indices.pkl', 'wb') as f:
            pickle.dump(split_indices, f)
        
        # Save metadata for each split
        train_df.to_csv(output_path / 'train_metadata.csv')
        val_df.to_csv(output_path / 'val_metadata.csv')
        test_df.to_csv(output_path / 'test_metadata.csv')
        
        # Save split configuration
        config = {
            'sampling_rate': self.sampling_rate,
            'train_size': len(train_df),
            'val_size': len(val_df),
            'test_size': len(test_df),
            'stratification': ['diagnostic_superclass', 'sex', 'age_group'],
            'patient_independent': True
        }
        
        with open(output_path / 'split_config.pkl', 'wb') as f:
            pickle.dump(config, f)
        
        print(f"✓ Saved split indices and metadata")
        print(f"  - split_indices.pkl")
        print(f"  - train_metadata.csv")
        print(f"  - val_metadata.csv")
        print(f"  - test_metadata.csv")
        print(f"  - split_config.pkl")


def main():
    """Main function to load and split PTB-XL dataset."""
    
    # Initialize loader
    loader = PTBXLDataLoader(
        data_path='./data/raw/ptb-xl/',
        sampling_rate=100  # Use 100 Hz version (faster to load)
    )
    
    # Load metadata
    loader.load_metadata()
    
    # Create stratified patient-independent split
    train_df, val_df, test_df = loader.create_patient_independent_split(
        test_size=0.2,
        val_size=0.1,
        random_state=42
    )
    
    # Print statistics
    loader.print_split_statistics(train_df, val_df, test_df)
    
    # Save splits
    loader.save_splits(train_df, val_df, test_df)
    
    print("\n" + "="*80)
    print("Dataset loading and splitting complete!")
    print("="*80)
    
    # Example: Load a single ECG record
    print("\nExample: Loading first training record...")
    first_record_id = train_df.index[0]
    ecg_data = loader.load_raw_data(first_record_id)
    print(f"Record ID: {first_record_id}")
    print(f"ECG data shape: {ecg_data.shape}")
    print(f"Duration: {ecg_data.shape[0] / loader.sampling_rate:.1f} seconds")
    print(f"Leads: 12-lead ECG")


if __name__ == "__main__":
    main()

