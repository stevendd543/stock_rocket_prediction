import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE

def apply_smote(X, y, sampling_strategy='auto', random_state = 42):
    """
    Apply SMOTE method for oversampling

    Parameters
    - X: Feature matrix
    - y: Label vector
    - sampling_strategy: Sampling strategy, 'auto' or a float (ratio of minority to majority class)
    - random_state: Random seed
    
    Returns
    - X_resampled: Oversampled feature matrix
    - y_resampled: Oversampled label vector
    """
    
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    print(f"Original shape: {X.shape}, Resampled shape: {X_resampled.shape}")
    return X_resampled, y_resampled


def filter_features_by_missing_rate(df, threshold, missing_report=None, return_type='df', label_name = '飆股'):
    """
    Filter features with missing rate smaller than the specified threshold
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame
    threshold : float
        Missing rate threshold, ranging from 0 to 100
    missing_report : str, optional
        File path of missing report, such as .csv, .txt
    return_type : {'list', 'df'}, default='list'
        Format of return value, must be either 'list' or 'df'
    
    Returns
    -------
    list or pandas.DataFrame
        If return_type='list': List of feature names with missing rate smaller than the threshold
        If return_type='df': DataFrame containing only features with missing rate smaller than the threshold
    """
    missing_rates = df.isnull().mean() * 100 # pandas Series
    
    features_to_keep = missing_rates[missing_rates <= threshold].index.tolist() # names of features

    if missing_report is not None:
        # Create a DataFrame with feature names and their missing rates
        missing_df = pd.DataFrame({
            'feature': missing_rates.index,
            'missing_rate': missing_rates.values
        })
        
        # Sort by missing rate
        missing_df = missing_df.sort_values('missing_rate', ascending=False)
        
        # Save to file
        if missing_report.endswith('.csv'):
            missing_df.to_csv(missing_report, index=False)
        else:
            missing_df.to_csv(missing_report, sep='\t', index=False)
    
    else:
        if return_type == 'list':
            return features_to_keep
        
        elif return_type == 'df':
            return df[features_to_keep] # copy-on-write
        
        else: 
            print("Error: Invalid return_type. Must be 'list' or 'df'.")
            return None

def impute_missing_values(df, method='mean'):
    """
    Impute missing values in the DataFrame using specified method
    
    Parameters
    -----------
    df : pandas.DataFrame
        Input DataFrame with missing values
    method : str, default='mean'
        Method to use for imputation. Options:
        - 'mean': Use mean for numeric columns
        - 'median': Use median for numeric columns
        - 'mode': Use mode for numeric and categorical columns
        - 'ffill': Forward fill (use previous value)
        - 'bfill': Backward fill (use next value)
        - 'zero': Fill with zeros
    
    Returns
    --------
    pandas.DataFrame
        DataFrame with imputed values
    """
    imputed_df = df.copy()
    
    initial_missing = imputed_df.isnull().sum().sum()
    
    if method == 'mean':
        numeric_cols = imputed_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            imputed_df[col] = imputed_df[col].fillna(imputed_df[col].mean())
        
    elif method == 'median':
        numeric_cols = imputed_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            imputed_df[col] = imputed_df[col].fillna(imputed_df[col].median())
    
    elif method == 'mode':
        for col in imputed_df.columns:
            if not imputed_df[col].isna().all():  # Skip if column is all NaN
                imputed_df[col] = imputed_df[col].fillna(imputed_df[col].mode()[0])
    
    elif method == 'ffill':
        imputed_df = imputed_df.fillna(method='ffill')
        
    elif method == 'bfill':
        imputed_df = imputed_df.fillna(method='bfill')
        
    elif method == 'zero':
        imputed_df = imputed_df.fillna(0)
        
    else:
        raise ValueError(f"Unknown imputation method: {method}")
    
    remaining_missing = imputed_df.isnull().sum().sum()
    
    if remaining_missing > 0:
        imputed_df = imputed_df.fillna(0)
        print(f"Warning: {remaining_missing} missing values couldn't be imputed with {method} method and were filled with 0")
    
    print(f"Imputation summary:")
    print(f"  - Initial missing values: {initial_missing}")
    print(f"  - Imputed values: {initial_missing - remaining_missing}")
    print(f"  - Remaining missing values: {imputed_df.isnull().sum().sum()}")
    
    return imputed_df

def prepare_data(data_path):
    """
    Read CSV file from file system
    
    Parameters
    ----------
    data_path : str
        Path to the CSV dataset file
        
    Returns
    -------
    pandas.DataFrame
        The loaded DataFrame containing all data from the CSV
    list
        List of column names from the DataFrame
    """
    df = pd.read_csv(data_path)
    df.replace([np.inf, -np.inf], np.nan)
    
    return df, df.columns.tolist()

if __name__ == "__main__":
    
    file_path = "split_file_21.csv"
    threshold = .0
    
    try:

        print(f"Reading data from {file_path}...")
        df = pd.read_csv(file_path)

        print(f"\nDataset shape: {df.shape}")
        print(f"Number of columns: {len(df.columns)}")

        features_list = filter_features_by_missing_rate(df, threshold)
        features_df = filter_features_by_missing_rate(df, threshold, return_type='df')

        print(f"TEST: Kept {len(features_list)} out of {len(df.columns)} features")
        # print(features_list)
        # print(features_df.head)

    except FileNotFoundError:
        print(f"Error: File{file_path} not found")
    except Exception as e:
        print(f"Error: {e}")