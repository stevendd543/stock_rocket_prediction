import pandas as pd



def filter_features_by_missing_rate(df, threshold, missing_report=None, return_type='list'):
    """
        Filter features with missing rate smaller than the specified threshold
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input DataFrame
        threshold : float
            Missing rate threshold, ranging from 0 to 100
        missing_report : string
            File path of missing report, such as .csv, .txt
        return_type : string
            Format of return value, must be either 'list' or 'df'

        Returns:
        --------
        (default) : list
            List of feature names with missing rate smaller than the threshold
        
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



if __name__ == "__main__":
    
    file_path = "dataset//train//training.csv"
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