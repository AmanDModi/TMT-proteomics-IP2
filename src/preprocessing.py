"""
Data preprocessing utilities for proteomics analysis.

This module provides functions for cleaning, normalizing, and quality control
of proteomics data.
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, List, Tuple
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler

logger = logging.getLogger(__name__)


def preprocess_data(
    data: pd.DataFrame,
    remove_missing: bool = True,
    normalize: bool = True,
    normalize_method: str = 'quantile',
    log_transform: bool = True,
    **kwargs
) -> pd.DataFrame:
    """
    Comprehensive preprocessing pipeline for proteomics data.
    
    Parameters
    ----------
    data : pd.DataFrame
        Raw proteomics data
    remove_missing : bool, default True
        Whether to remove rows with missing values
    normalize : bool, default True
        Whether to normalize the data
    normalize_method : str, default 'quantile'
        Normalization method ('quantile', 'zscore', 'robust')
    log_transform : bool, default True
        Whether to log-transform intensity values
    **kwargs
        Additional arguments for specific preprocessing steps
        
    Returns
    -------
    pd.DataFrame
        Preprocessed data
    """
    processed_data = data.copy()
    
    # Step 1: Remove missing values
    if remove_missing:
        processed_data = remove_missing_values(processed_data, **kwargs)
    
    # Step 2: Log transform (if specified)
    if log_transform:
        processed_data = log_transform_intensities(processed_data, **kwargs)
    
    # Step 3: Normalize data
    if normalize:
        processed_data = normalize_data(processed_data, method=normalize_method, **kwargs)
    
    # Step 4: Quality control
    processed_data = quality_control(processed_data, **kwargs)
    
    logger.info(f"Preprocessing complete. Final shape: {processed_data.shape}")
    return processed_data


def remove_missing_values(
    data: pd.DataFrame,
    threshold: float = 0.5,
    axis: int = 0
) -> pd.DataFrame:
    """
    Remove rows or columns with too many missing values.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data
    threshold : float, default 0.5
        Maximum fraction of missing values allowed
    axis : int, default 0
        0 for rows, 1 for columns
        
    Returns
    -------
    pd.DataFrame
        Data with missing values removed
    """
    if axis == 0:
        # Remove rows with too many missing values
        missing_frac = data.isnull().sum(axis=1) / data.shape[1]
        data_cleaned = data[missing_frac <= threshold]
    else:
        # Remove columns with too many missing values
        missing_frac = data.isnull().sum(axis=0) / data.shape[0]
        data_cleaned = data.loc[:, missing_frac <= threshold]
    
    removed_count = data.shape[axis] - data_cleaned.shape[axis]
    logger.info(f"Removed {removed_count} {'rows' if axis == 0 else 'columns'} with >{threshold*100}% missing values")
    
    return data_cleaned


def log_transform_intensities(
    data: pd.DataFrame,
    intensity_columns: Optional[List[str]] = None,
    base: float = 2
) -> pd.DataFrame:
    """
    Apply log transformation to intensity columns.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data
    intensity_columns : list, optional
        Columns to transform. If None, auto-detect intensity columns
    base : float, default 2
        Base for logarithm
        
    Returns
    -------
    pd.DataFrame
        Data with log-transformed intensities
    """
    if intensity_columns is None:
        # Auto-detect intensity columns
        intensity_columns = [col for col in data.columns 
                           if any(keyword in col.lower() 
                                 for keyword in ['intensity', 'abundance', 'count'])]
    
    if not intensity_columns:
        logger.warning("No intensity columns detected for log transformation")
        return data
    
    data_transformed = data.copy()
    
    for col in intensity_columns:
        if col in data.columns:
            # Add small constant to avoid log(0)
            min_val = data[col].min()
            if min_val <= 0:
                offset = abs(min_val) + 1e-10
                data_transformed[col] = np.log(data[col] + offset) / np.log(base)
            else:
                data_transformed[col] = np.log(data[col]) / np.log(base)
    
    logger.info(f"Applied log{base} transformation to {len(intensity_columns)} columns")
    return data_transformed


def normalize_data(
    data: pd.DataFrame,
    method: str = 'quantile',
    sample_columns: Optional[List[str]] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Normalize proteomics data using various methods.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data
    method : str, default 'quantile'
        Normalization method ('quantile', 'zscore', 'robust', 'median')
    sample_columns : list, optional
        Columns to normalize. If None, use all numeric columns
        
    Returns
    -------
    pd.DataFrame
        Normalized data
    """
    if sample_columns is None:
        sample_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    data_normalized = data.copy()
    
    if method == 'quantile':
        data_normalized = quantile_normalize(data_normalized, sample_columns, **kwargs)
    elif method == 'zscore':
        data_normalized = zscore_normalize(data_normalized, sample_columns, **kwargs)
    elif method == 'robust':
        data_normalized = robust_normalize(data_normalized, sample_columns, **kwargs)
    elif method == 'median':
        data_normalized = median_normalize(data_normalized, sample_columns, **kwargs)
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    logger.info(f"Applied {method} normalization to {len(sample_columns)} columns")
    return data_normalized


def quantile_normalize(
    data: pd.DataFrame,
    sample_columns: List[str]
) -> pd.DataFrame:
    """
    Perform quantile normalization.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data
    sample_columns : list
        Columns to normalize
        
    Returns
    -------
    pd.DataFrame
        Quantile-normalized data
    """
    # Implementation of quantile normalization
    # This is a simplified version - for production, consider using specialized libraries
    data_normalized = data.copy()
    
    # Calculate quantiles for each sample
    quantiles = {}
    for col in sample_columns:
        if col in data.columns:
            quantiles[col] = data[col].quantile([0.25, 0.5, 0.75])
    
    # Apply normalization (simplified approach)
    for col in sample_columns:
        if col in data.columns:
            # Center around median
            median_val = data[col].median()
            data_normalized[col] = data[col] - median_val
    
    return data_normalized


def zscore_normalize(
    data: pd.DataFrame,
    sample_columns: List[str]
) -> pd.DataFrame:
    """
    Perform z-score normalization.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data
    sample_columns : list
        Columns to normalize
        
    Returns
    -------
    pd.DataFrame
        Z-score normalized data
    """
    data_normalized = data.copy()
    
    for col in sample_columns:
        if col in data.columns:
            mean_val = data[col].mean()
            std_val = data[col].std()
            if std_val > 0:
                data_normalized[col] = (data[col] - mean_val) / std_val
    
    return data_normalized


def robust_normalize(
    data: pd.DataFrame,
    sample_columns: List[str]
) -> pd.DataFrame:
    """
    Perform robust normalization using median and MAD.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data
    sample_columns : list
        Columns to normalize
        
    Returns
    -------
    pd.DataFrame
        Robustly normalized data
    """
    data_normalized = data.copy()
    
    for col in sample_columns:
        if col in data.columns:
            median_val = data[col].median()
            mad_val = stats.median_abs_deviation(data[col].dropna())
            if mad_val > 0:
                data_normalized[col] = (data[col] - median_val) / mad_val
    
    return data_normalized


def median_normalize(
    data: pd.DataFrame,
    sample_columns: List[str]
) -> pd.DataFrame:
    """
    Perform median normalization.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data
    sample_columns : list
        Columns to normalize
        
    Returns
    -------
    pd.DataFrame
        Median-normalized data
    """
    data_normalized = data.copy()
    
    for col in sample_columns:
        if col in data.columns:
            median_val = data[col].median()
            if median_val > 0:
                data_normalized[col] = data[col] / median_val
    
    return data_normalized


def quality_control(
    data: pd.DataFrame,
    intensity_threshold: Optional[float] = None,
    cv_threshold: float = 0.5,
    **kwargs
) -> pd.DataFrame:
    """
    Perform quality control on proteomics data.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data
    intensity_threshold : float, optional
        Minimum intensity threshold
    cv_threshold : float, default 0.5
        Maximum coefficient of variation threshold
        
    Returns
    -------
    pd.DataFrame
        Quality-controlled data
    """
    data_qc = data.copy()
    
    # Remove low-intensity measurements
    if intensity_threshold is not None:
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            data_qc = data_qc[data_qc[col] >= intensity_threshold]
    
    # Remove high CV measurements (if technical replicates exist)
    # This is a placeholder - would need replicate information
    
    logger.info(f"Quality control complete. Final shape: {data_qc.shape}")
    return data_qc 