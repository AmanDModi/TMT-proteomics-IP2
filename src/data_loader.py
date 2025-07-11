"""
Data loading utilities for proteomics analysis.

This module provides functions to load proteomics data from various formats
including CSV, Excel, and mzML files.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Optional, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_proteomics_data(
    file_path: Union[str, Path], 
    file_type: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Load proteomics data from various file formats.
    
    Parameters
    ----------
    file_path : str or Path
        Path to the data file
    file_type : str, optional
        Type of file ('csv', 'excel', 'mzml'). If None, inferred from extension
    **kwargs
        Additional arguments passed to pandas read functions
        
    Returns
    -------
    pd.DataFrame
        Loaded proteomics data
        
    Raises
    ------
    ValueError
        If file type is not supported or file doesn't exist
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Infer file type from extension if not provided
    if file_type is None:
        file_type = file_path.suffix.lower().lstrip('.')
    
    try:
        if file_type in ['csv', 'txt']:
            data = pd.read_csv(file_path, **kwargs)
        elif file_type in ['xlsx', 'xls', 'excel']:
            data = pd.read_excel(file_path, **kwargs)
        elif file_type == 'mzml':
            data = _load_mzml_data(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        logger.info(f"Successfully loaded data from {file_path}")
        logger.info(f"Data shape: {data.shape}")
        
        return data
        
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {str(e)}")
        raise


def _load_mzml_data(file_path: Path, **kwargs) -> pd.DataFrame:
    """
    Load data from mzML files (placeholder for future implementation).
    
    Parameters
    ----------
    file_path : Path
        Path to the mzML file
    **kwargs
        Additional arguments
        
    Returns
    -------
    pd.DataFrame
        Processed mzML data
    """
    # TODO: Implement mzML loading functionality
    # This would typically use pymzml library
    logger.warning("mzML loading not yet implemented. Returning empty DataFrame.")
    return pd.DataFrame()


def load_multiple_files(
    file_paths: list, 
    file_types: Optional[list] = None,
    **kwargs
) -> Dict[str, pd.DataFrame]:
    """
    Load multiple proteomics data files.
    
    Parameters
    ----------
    file_paths : list
        List of file paths to load
    file_types : list, optional
        List of file types corresponding to file_paths
    **kwargs
        Additional arguments passed to load_proteomics_data
        
    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary mapping file names to loaded data
    """
    if file_types is None:
        file_types = [None] * len(file_paths)
    
    if len(file_paths) != len(file_types):
        raise ValueError("file_paths and file_types must have the same length")
    
    data_dict = {}
    
    for file_path, file_type in zip(file_paths, file_types):
        try:
            data = load_proteomics_data(file_path, file_type, **kwargs)
            file_name = Path(file_path).stem
            data_dict[file_name] = data
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {str(e)}")
            continue
    
    return data_dict


def validate_proteomics_data(data: pd.DataFrame) -> bool:
    """
    Validate that the loaded data has expected proteomics columns.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data to validate
        
    Returns
    -------
    bool
        True if data appears to be valid proteomics data
    """
    # Common proteomics column names
    expected_columns = [
        'protein', 'peptide', 'sequence', 'mz', 'intensity',
        'retention_time', 'charge', 'mass', 'abundance'
    ]
    
    # Check if any expected columns are present
    found_columns = [col for col in expected_columns if col.lower() in data.columns.str.lower()]
    
    if len(found_columns) == 0:
        logger.warning("No common proteomics column names found in data")
        return False
    
    logger.info(f"Found proteomics columns: {found_columns}")
    return True 