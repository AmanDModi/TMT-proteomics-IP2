"""
Data processing utilities for proteomics analysis.

This module provides functions for cleaning and processing proteomics data
from CSV files, including duplicate removal, filtering, and log transformation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_proteomics_data(
    input_file: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    gene_column: str = 'GENE',  # Changed from accession_column
    ratio_column: str = 'NORM_RATIO_2_1'
) -> pd.DataFrame:
    """
    Process proteomics data from CSV file with comprehensive cleaning.
    
    Parameters
    ----------
    input_file : str or Path
        Path to the input CSV file
    output_file : str or Path, optional
        Path to save the processed data. If None, automatically generates output filename
        by appending '_output_df' to the input filename
    gene_column : str, default 'GENE'  # Changed from accession_column
        Name of the gene column for duplicate removal
    ratio_column : str, default 'NORM_RATIO_2_1'
        Name of the ratio column for log2 transformation
        
    Returns
    -------
    pd.DataFrame
        Processed proteomics data
        
    Raises
    ------
    FileNotFoundError
        If input file doesn't exist
    ValueError
        If required columns are not found
    """
    input_path = Path(input_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    logger.info(f"Loading data from {input_path}")
    
    # Load the CSV file
    try:
        data = pd.read_csv(input_path)
        logger.info(f"Loaded data with shape: {data.shape}")
    except Exception as e:
        logger.error(f"Error loading CSV file: {str(e)}")
        raise
    
    # Validate required columns
    required_columns = [gene_column, ratio_column]  # Changed from accession_column
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        raise ValueError(f"Required columns not found: {missing_columns}")
    
    # Process the data
    processed_data = data.copy()
    
    # Step 1: Remove rows containing 'Reverse' or 'contaminant' in ACCESSION
    rows_before_filter = len(processed_data)
    processed_data = filter_accession_contaminants(processed_data, 'ACCESSION')
    logger.info(f"Removed {rows_before_filter - len(processed_data)} contaminant/reverse rows")
    
    # Step 2: Remove duplicate GENE values (keep first occurrence) - MOVED HERE
    initial_rows = len(processed_data)
    processed_data = remove_duplicate_genes(processed_data, gene_column)
    logger.info(f"Removed {initial_rows - len(processed_data)} duplicate gene rows")
    
    # Step 3: Remove rows with 0 or 'infinity' values in ratio column
    rows_before_ratio_filter = len(processed_data)
    processed_data = filter_ratio_values(processed_data, ratio_column)
    logger.info(f"Removed {rows_before_ratio_filter - len(processed_data)} invalid ratio rows")
    
    # Step 4: Add log2 transformation column
    processed_data = add_log2_ratio_column(processed_data, ratio_column)
    logger.info(f"Added log2_norm_ratio column")
    
    # Generate output filename if not provided
    if output_file is None:
        # Get the input filename without extension
        input_stem = input_path.stem
        # Get the input directory
        input_dir = input_path.parent
        # Create output filename with '_output_df' appended
        output_filename = f"{input_stem}_output_df.csv"
        output_path = input_dir / output_filename
    else:
        output_path = Path(output_file)
    
    # Save processed data
    try:
        processed_data.to_csv(output_path, index=False)
        logger.info(f"Processed data saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving processed data: {str(e)}")
        raise
    
    logger.info(f"Data processing complete. Final shape: {processed_data.shape}")
    return processed_data


def remove_duplicate_genes(data: pd.DataFrame, gene_column: str) -> pd.DataFrame:
    """
    Remove duplicate rows based on GENE column, keeping the first occurrence.
    """
    data_cleaned = data.copy()
    
    # Clean the gene column
    data_cleaned[gene_column] = data_cleaned[gene_column].astype(str).str.strip()
    data_cleaned[gene_column] = data_cleaned[gene_column].replace(['', 'nan', 'NaN', 'None'], np.nan)
    
    # Count duplicates before removal
    duplicates = data_cleaned[gene_column].duplicated().sum()
    if duplicates > 0:
        logger.info(f"Found {duplicates} duplicate gene values")
        
        # Show some examples
        duplicate_genes = data_cleaned[data_cleaned[gene_column].duplicated()][gene_column].unique()
        logger.info(f"Sample duplicate genes: {duplicate_genes[:3].tolist()}")
    
    # Try standard drop_duplicates first
    result = data_cleaned.drop_duplicates(subset=[gene_column], keep='first')
    
    # Verify it worked
    remaining_duplicates = result[gene_column].duplicated().sum()
    if remaining_duplicates > 0:
        logger.warning(f"Standard drop_duplicates failed! {remaining_duplicates} duplicates remain.")
        logger.warning("Using manual duplicate removal...")
        
        # Use manual approach as fallback
        unique_genes = data_cleaned[gene_column].dropna().unique()
        keep_mask = pd.Series(False, index=data_cleaned.index)
        
        for gene in unique_genes:
            gene_indices = data_cleaned[data_cleaned[gene_column] == gene].index
            if len(gene_indices) > 0:
                keep_mask.loc[gene_indices[0]] = True
        
        # Keep NaN genes too
        nan_mask = data_cleaned[gene_column].isna()
        keep_mask = keep_mask | nan_mask
        
        result = data_cleaned[keep_mask].copy()
        
        # Final verification
        final_duplicates = result[gene_column].duplicated().sum()
        logger.info(f"Manual removal complete. Final duplicates: {final_duplicates}")
    
    return result


def filter_accession_contaminants(data: pd.DataFrame, accession_column: str) -> pd.DataFrame:
    """
    Remove rows where ACCESSION contains 'Reverse' or 'contaminant'.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data
    accession_column : str
        Name of the accession column
        
    Returns
    -------
    pd.DataFrame
        Data with contaminant/reverse rows removed
    """
    # Convert to string and check for contaminants (case-insensitive)
    accession_str = data[accession_column].astype(str).str.lower()
    
    # Create mask for rows to keep (not containing contaminants)
    mask = ~(accession_str.str.contains('reverse', na=False) | 
             accession_str.str.contains('contaminant', na=False))
    
    # Count filtered rows
    filtered_count = (~mask).sum()
    if filtered_count > 0:
        logger.info(f"Filtered out {filtered_count} rows containing 'reverse' or 'contaminant'")
    
    return data[mask]


def filter_ratio_values(data: pd.DataFrame, ratio_column: str) -> pd.DataFrame:
    """
    Remove rows where ratio column contains 0 or 'infinity' values.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data
    ratio_column : str
        Name of the ratio column
        
    Returns
    -------
    pd.DataFrame
        Data with invalid ratio values removed
    """
    # Convert to string for checking 'infinity'
    ratio_str = data[ratio_column].astype(str).str.lower()
    
    # Create mask for valid values (not 0 and not 'infinity')
    mask = (data[ratio_column] != 0) & (ratio_str != 'infinity')
    
    # Count filtered rows
    filtered_count = (~mask).sum()
    if filtered_count > 0:
        logger.info(f"Filtered out {filtered_count} rows with 0 or 'infinity' values")
    
    return data[mask]


def add_log2_ratio_column(data: pd.DataFrame, ratio_column: str) -> pd.DataFrame:
    """
    Add a new column 'log2_norm_ratio' with log2 transformation of ratio values.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data
    ratio_column : str
        Name of the ratio column
        
    Returns
    -------
    pd.DataFrame
        Data with new log2_norm_ratio column added
    """
    data_with_log2 = data.copy()
    
    # Calculate log2 of ratio values
    # Handle potential negative or zero values
    ratio_values = data[ratio_column].astype(float)
    
    # Check for negative values
    negative_count = (ratio_values < 0).sum()
    if negative_count > 0:
        logger.warning(f"Found {negative_count} negative values in ratio column. These will result in NaN in log2")
    
    # Calculate log2, handling negative values
    log2_values = np.log2(ratio_values)
    
    # Insert the new column after the ratio column
    ratio_col_idx = data.columns.get_loc(ratio_column)
    data_with_log2.insert(ratio_col_idx + 1, 'log2_norm_ratio', log2_values)
    
    # Log statistics
    valid_log2 = log2_values.dropna()
    if len(valid_log2) > 0:
        logger.info(f"Log2 ratio statistics - Mean: {valid_log2.mean():.3f}, "
                   f"Std: {valid_log2.std():.3f}, "
                   f"Range: [{valid_log2.min():.3f}, {valid_log2.max():.3f}]")
    
    return data_with_log2


def get_processing_summary(data: pd.DataFrame, processed_data: pd.DataFrame) -> dict:
    """
    Generate a summary of the processing steps.
    
    Parameters
    ----------
    data : pd.DataFrame
        Original data
    processed_data : pd.DataFrame
        Processed data
        
    Returns
    -------
    dict
        Summary statistics
    """
    summary = {
        'original_rows': len(data),
        'processed_rows': len(processed_data),
        'rows_removed': len(data) - len(processed_data),
        'removal_percentage': ((len(data) - len(processed_data)) / len(data)) * 100
    }
    
    return summary


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python data_processing.py <input_file> [output_file]")
        print("Note: If output_file is not specified, it will be automatically generated")
        print("      by appending '_output_df' to the input filename")
        print("      Duplicates are removed based on GENE column")  # Updated note
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        processed_data = process_proteomics_data(input_file, output_file)
        summary = get_processing_summary(pd.read_csv(input_file), processed_data)
        
        print("\nProcessing Summary:")
        print(f"Original rows: {summary['original_rows']}")
        print(f"Processed rows: {summary['processed_rows']}")
        print(f"Rows removed: {summary['rows_removed']}")
        print(f"Removal percentage: {summary['removal_percentage']:.2f}%")
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        sys.exit(1) 