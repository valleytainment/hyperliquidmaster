"""
Data Type Handling Utilities for Technical Indicators

This module provides utility functions to ensure proper data type handling
in technical indicator calculations, preventing method errors and ensuring
robust calculations.
"""

import numpy as np
import pandas as pd
from typing import Union, Any, List, Dict, Optional

def ensure_pandas_series(data: Any) -> pd.Series:
    """
    Ensure data is a pandas Series.
    
    Args:
        data: Data to convert to pandas Series
        
    Returns:
        Pandas Series
    """
    if isinstance(data, np.ndarray):
        return pd.Series(data)
    elif isinstance(data, pd.Series):
        return data
    elif isinstance(data, pd.DataFrame) and len(data.columns) == 1:
        return data.iloc[:, 0]
    elif isinstance(data, list):
        return pd.Series(data)
    elif isinstance(data, dict):
        return pd.Series(data)
    else:
        return pd.Series([data])

def ensure_pandas_dataframe(data: Any) -> pd.DataFrame:
    """
    Ensure data is a pandas DataFrame.
    
    Args:
        data: Data to convert to pandas DataFrame
        
    Returns:
        Pandas DataFrame
    """
    if isinstance(data, pd.DataFrame):
        return data
    elif isinstance(data, np.ndarray):
        if data.ndim == 1:
            return pd.DataFrame(data.reshape(-1, 1))
        else:
            return pd.DataFrame(data)
    elif isinstance(data, pd.Series):
        return pd.DataFrame(data)
    elif isinstance(data, list):
        if len(data) > 0 and isinstance(data[0], dict):
            return pd.DataFrame(data)
        else:
            return pd.DataFrame(data, columns=['value'])
    elif isinstance(data, dict):
        return pd.DataFrame(data, index=[0])
    else:
        return pd.DataFrame([data], columns=['value'])

def safe_shift(data: Any, periods: int = 1) -> pd.Series:
    """
    Safely shift data, ensuring it's a pandas Series first.
    
    Args:
        data: Data to shift
        periods: Number of periods to shift
        
    Returns:
        Shifted pandas Series
    """
    series = ensure_pandas_series(data)
    return series.shift(periods)

def safe_rolling(data: Any, window: int) -> pd.core.window.rolling.Rolling:
    """
    Safely create a rolling window, ensuring data is a pandas Series first.
    
    Args:
        data: Data to create rolling window for
        window: Rolling window size
        
    Returns:
        Rolling window object
    """
    series = ensure_pandas_series(data)
    return series.rolling(window)

def safe_ewm(data: Any, span: int) -> pd.core.window.ewm.ExponentialMovingWindow:
    """
    Safely create an exponential weighted window, ensuring data is a pandas Series first.
    
    Args:
        data: Data to create EWM window for
        span: EWM span
        
    Returns:
        EWM window object
    """
    series = ensure_pandas_series(data)
    return series.ewm(span=span)

def safe_diff(data: Any, periods: int = 1) -> pd.Series:
    """
    Safely calculate difference, ensuring data is a pandas Series first.
    
    Args:
        data: Data to calculate difference for
        periods: Number of periods to difference
        
    Returns:
        Differenced pandas Series
    """
    series = ensure_pandas_series(data)
    return series.diff(periods)

def safe_pct_change(data: Any, periods: int = 1) -> pd.Series:
    """
    Safely calculate percentage change, ensuring data is a pandas Series first.
    
    Args:
        data: Data to calculate percentage change for
        periods: Number of periods to calculate percentage change over
        
    Returns:
        Percentage change pandas Series
    """
    series = ensure_pandas_series(data)
    return series.pct_change(periods)

def extract_column(df: Any, column: str) -> pd.Series:
    """
    Safely extract a column from a DataFrame, with fallbacks for different data types.
    
    Args:
        df: DataFrame or dict-like object
        column: Column name to extract
        
    Returns:
        Extracted column as pandas Series
    """
    # Handle DataFrame
    if isinstance(df, pd.DataFrame):
        if column in df.columns:
            return df[column]
        else:
            # Try case-insensitive match
            for col in df.columns:
                if col.lower() == column.lower():
                    return df[col]
            # Return first column as fallback
            return df.iloc[:, 0]
    
    # Handle dict or dict-like
    elif isinstance(df, dict):
        if column in df:
            return ensure_pandas_series(df[column])
        else:
            # Try case-insensitive match
            for key in df:
                if isinstance(key, str) and key.lower() == column.lower():
                    return ensure_pandas_series(df[key])
            # Return first value as fallback
            if len(df) > 0:
                return ensure_pandas_series(next(iter(df.values())))
            else:
                return pd.Series()
    
    # Handle other types
    else:
        return ensure_pandas_series(df)

def get_ohlc(df: Any) -> Dict[str, pd.Series]:
    """
    Safely extract OHLC data from various input types.
    
    Args:
        df: Input data (DataFrame, dict, etc.)
        
    Returns:
        Dictionary with open, high, low, close as pandas Series
    """
    result = {
        'open': None,
        'high': None,
        'low': None,
        'close': None
    }
    
    # Handle DataFrame
    if isinstance(df, pd.DataFrame):
        # Try standard column names
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                result[col] = df[col]
            else:
                # Try case-insensitive match
                for df_col in df.columns:
                    if df_col.lower() == col.lower():
                        result[col] = df[df_col]
                        break
        
        # If close is still None, try 'price' column
        if result['close'] is None and 'price' in df.columns:
            result['close'] = df['price']
            
        # If any are still None, use first numeric column as fallback for all
        if any(v is None for v in result.values()):
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                fallback = df[numeric_cols[0]]
                for col in result:
                    if result[col] is None:
                        result[col] = fallback
    
    # Handle dict or dict-like
    elif isinstance(df, dict):
        # Try standard keys
        for col in ['open', 'high', 'low', 'close']:
            if col in df:
                result[col] = ensure_pandas_series(df[col])
            else:
                # Try case-insensitive match
                for key in df:
                    if isinstance(key, str) and key.lower() == col.lower():
                        result[col] = ensure_pandas_series(df[key])
                        break
        
        # If close is still None, try 'price' key
        if result['close'] is None and 'price' in df:
            result['close'] = ensure_pandas_series(df['price'])
            
        # If any are still None, use first value as fallback for all
        if any(v is None for v in result.values()) and len(df) > 0:
            fallback = ensure_pandas_series(next(iter(df.values())))
            for col in result:
                if result[col] is None:
                    result[col] = fallback
    
    # Handle other types (convert to Series and use for all)
    else:
        series = ensure_pandas_series(df)
        for col in result:
            result[col] = series
    
    # Final check - if any are still None, create empty Series
    for col in result:
        if result[col] is None:
            result[col] = pd.Series()
    
    return result

def handle_missing_values(data: Union[pd.Series, pd.DataFrame], method: str = 'ffill') -> Union[pd.Series, pd.DataFrame]:
    """
    Handle missing values in data.
    
    Args:
        data: Data to handle missing values for
        method: Method to use ('ffill', 'bfill', 'interpolate', 'drop', 'zero')
        
    Returns:
        Data with missing values handled
    """
    if method == 'ffill':
        return data.fillna(method='ffill')
    elif method == 'bfill':
        return data.fillna(method='bfill')
    elif method == 'interpolate':
        return data.interpolate()
    elif method == 'drop':
        return data.dropna()
    elif method == 'zero':
        return data.fillna(0)
    else:
        return data.fillna(method='ffill')  # Default to forward fill
