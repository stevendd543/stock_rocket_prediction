import pandas as pd
import numpy as np
from imblearn.over_sampling import *
def apply_oversampling(X, y, method='SMOTE', sampling_strategy='auto', random_state=42, **kwargs):
    """
    Apply various oversampling methods from imbalanced-learn library
    
    Parameters
    - X: Feature matrix
    - y: Label vector
    - method: Oversampling method to use ('SMOTE', 'SMOTENC', 'SMOTEN', 'ADASYN', 
              'BorderlineSMOTE', 'KMeansSMOTE', 'SVMSMOTE')
    - sampling_strategy: Sampling strategy, 'auto' or a float (ratio of minority to majority class)
    - random_state: Random seed
    - **kwargs: Additional parameters for specific oversampling methods
    
    Returns
    - X_resampled: Oversampled feature matrix
    - y_resampled: Oversampled label vector
    """
    
    # Dictionary mapping method names to their corresponding classes
    methods = {
        'SMOTE': SMOTE,
        # 'SMOTENC': SMOTENC,
        'SMOTEN': SMOTEN,
        'ADASYN': ADASYN,
        'BorderlineSMOTE': BorderlineSMOTE,
        'KMeansSMOTE': KMeansSMOTE,
        'SVMSMOTE': SVMSMOTE
    }
    
    # Check if the specified method exists
    if method not in methods:
        raise ValueError(f"Method '{method}' not supported. Choose from: {', '.join(methods.keys())}")
    
    # Handle special case for SMOTENC which requires categorical_features parameter
    # if method == 'SMOTENC' and 'categorical_features' not in kwargs:
    #     raise ValueError("SMOTENC requires 'categorical_features' parameter")
    
    # Initialize the oversampler
    oversampler = methods[method](sampling_strategy=sampling_strategy, random_state=random_state, **kwargs)

    # Apply oversampling
    X_resampled, y_resampled = oversampler.fit_resample(X, y)
    
    # Print statistics
    print(f"Method used: {method}")
    print(f"Original shape: {X.shape}, Resampled shape: {X_resampled.shape}")
    print(f"Original class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"Resampled class distribution: {dict(zip(*np.unique(y_resampled, return_counts=True)))}")
    
    return X_resampled, y_resampled

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
    print(f"Original amount of postive: {y.sum()}, Resampled amount of postive: {y_resampled.sum()}")
    return X_resampled, y_resampled


def filter_features_by_missing_rate(df=None, threshold=None, missing_report=None, return_type='df', label_name = '飆股'):
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

    if df is None and return_type == 'df':
        raise ValueError("Can't return datafram without origin dataframe")
    
    if missing_report is not None:
        
        if isinstance(missing_report, str):
            missing_df = pd.read_csv(missing_report)
        elif isinstance(missing_report, pd.DataFrame):
            missing_df = missing_report.copy()
        else:
            raise ValueError("missing_report must be a file path or DataFrame")
        

        if len(missing_df.columns) >= 3:

            standard_columns = ['feature', 'missing_count', 'missing_rate']
            missing_df.columns.values[0:3] = standard_columns
            
            missing_df = missing_df.iloc[:, 0:3]
            missing_df = missing_df.sort_values('missing_rate', ascending=True)
            
            
            missing_df = missing_df[missing_df.iloc[:, 1] <= threshold]

            feature_list = missing_df['feature'].tolist()

            missing_features = [feature for feature in feature_list if feature not in df.columns]

            if missing_features:
                print(f"Warning: the following features don't exist in df")
                for feature in missing_features:
                    print(f"  - {feature}")
                

            feature_list = [feature for feature in feature_list if feature in df.columns]
            

            return feature_list if return_type == 'list' else df[feature_list]
        else:
            raise ValueError("Input data must have at least 3 columns")
    
    else:
        missing_rates = df.isnull().mean() * 100 # pandas Series
        features_to_keep = missing_rates[missing_rates <= threshold].index.tolist() # names of features

        if return_type == 'list':
            return features_to_keep
        
        elif return_type == 'df':
            return df[features_to_keep] # copy-on-write
        
        else: 
            print("Error: Invalid return_type. Must be 'list' or 'df'.")
            return None
def drop_features_by_f1_change(fi_df, threshold=0, return_dropped=False):
    """
    Keep features with f1_change greater than the threshold value.    
    
    Parameters:
    -----------
    fi_df : pandas.DataFrame
        DataFrame containing feature evaluation results with columns including
        'model_name', 'removed_feature', 'f1_score', 'f1_change', etc.
    threshold : float, default=0
        The threshold for f1_change. Features with f1_change greater than this threshold
        will be kept. Negative values indicate features that, when removed,
        improved the F1 score.
    return_dropped : bool, default=False
        If True, returns the list of dropped features alongside the features to keep.
        
    Returns:
    --------
    if return_dropped=False:
        list: List of features to keep (those with f1_change > threshold)
    if return_dropped=True:
        tuple: (list of features to keep, list of features to drop)
    """
    # Ensure the DataFrame has the required columns
    required_columns = ['removed_feature', 'f1_change']
    for col in required_columns:
        if col not in fi_df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame")
    
    # Filter features to keep (those with f1_change greater than threshold)
    features_to_keep = fi_df[fi_df['f1_change'] > threshold]['removed_feature'].tolist()
    
    # Get all unique features
    all_features = fi_df['removed_feature'].unique().tolist()
    
    # Features to drop
    features_to_drop = [f for f in all_features if f not in features_to_keep]
    
    if return_dropped:
        return features_to_keep, features_to_drop
    else:
        return features_to_keep
    
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
    if initial_missing == 0: 
        print("Not found missing value")
        return df
    
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

def prepare_dataframe(data_path):
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
def read_missing_report(data_path):
    """
    Read CSV of missing value report 
    
    Columes name : 特徵名稱,缺失值數量,缺失比例(%)
    
    Parameters
    ----------
    data_path : str
        Path to the CSV report file
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing columns: 特徵名稱,缺失值數量,缺失比例(%)
    """
    df = pd.read_csv(data_path)
    
    required_columns = ["特徵名稱", "缺失值數量", "缺失比例(%)"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
            
    return df[required_columns]
    
    
if __name__ == "__main__":
    

    file_path = "missing_values_rocket_1_report.csv"
    threshold = .0

    ####################################################################################  
    # try:

    #     print(f"Reading data from {file_path}...")
    #     df = pd.read_csv(file_path)

    #     print(f"\nDataset shape: {df.shape}")
    #     print(f"Number of columns: {len(df.columns)}")

    #     features_list = filter_features_by_missing_rate(df, threshold)
    #     features_df = filter_features_by_missing_rate(df, threshold, return_type='df')

    #     print(f"TEST: Kept {len(features_list)} out of {len(df.columns)} features")
    #     # print(features_list)
    #     # print(features_df.head)

    # except FileNotFoundError:
    #     print(f"Error: File{file_path} not found")
    # except Exception as e:
    #     print(f"Error: {e}")
    ####################################################################################
    #               Selece features with missing value report                          #                
    ####################################################################################  
    df,_ = prepare_dataframe("data.csv")
    missing_report = read_missing_report(file_path)
    selected_features = filter_features_by_missing_rate(df=df, missing_report=missing_report, return_type='df', threshold=10)
    print(selected_features)