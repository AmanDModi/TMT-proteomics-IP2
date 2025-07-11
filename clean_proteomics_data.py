#!/usr/bin/env python3
"""
Proteomics Data Cleaning Script

This script cleans proteomics data by:
1. Removing duplicate entries based on ACCESSION column
2. Removing rows with 0 or infinity values in NORM_RATIO_2_1 column
3. Adding log2 transformation of NORM_RATIO_2_1 column
4. Creating volcano plot visualization
5. Saving cleaned data to a new file

Usage:
    python clean_proteomics_data.py input_file.xlsx output_file.xlsx
    python clean_proteomics_data.py input_file.xlsx output_file.xlsx --verbose
    python clean_proteomics_data.py input_file.xlsx output_file.xlsx --plot
"""

import pandas as pd
import numpy as np
import argparse
import sys
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load Excel or CSV file and return DataFrame.
    
    Parameters
    ----------
    file_path : str
        Path to the file
        
    Returns
    -------
    pd.DataFrame
        Loaded data
    """
    file_path_obj = Path(file_path)
    ext = file_path_obj.suffix.lower()
    try:
        logger.info(f"Loading data from {file_path}")
        if ext in ['.xlsx', '.xls']:
            try:
                data = pd.read_excel(file_path, engine='openpyxl')
            except Exception as e1:
                logger.warning(f"Failed with openpyxl engine: {str(e1)}")
                try:
                    data = pd.read_excel(file_path, engine='xlrd')
                except Exception as e2:
                    logger.warning(f"Failed with xlrd engine: {str(e2)}")
                    data = pd.read_excel(file_path)
        elif ext == '.csv':
            data = pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
        logger.info(f"Successfully loaded data with shape: {data.shape}")
        logger.info(f"Columns found: {list(data.columns)}")
        return data
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {str(e)}")
        raise


def add_log2_column(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add log2 transformation of NORM_RATIO_2_1 column.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data
        
    Returns
    -------
    pd.DataFrame
        Data with new log2(norm_ratio) column
    """
    if 'NORM_RATIO_2_1' not in data.columns:
        logger.warning("NORM_RATIO_2_1 column not found, skipping log2 transformation")
        return data
    
    logger.info("Adding log2 transformation of NORM_RATIO_2_1 column...")
    
    # Calculate log2, handling negative values and zeros
    data['log2(norm_ratio)'] = np.log2(data['NORM_RATIO_2_1'])
    
    # Count any infinite or NaN values
    inf_count = np.isinf(data['log2(norm_ratio)']).sum()
    nan_count = data['log2(norm_ratio)'].isna().sum()
    
    if inf_count > 0 or nan_count > 0:
        logger.warning(f"Found {inf_count} infinite and {nan_count} NaN values in log2 transformation")
    
    logger.info(f"Added log2(norm_ratio) column with range: {data['log2(norm_ratio)'].min():.3f} to {data['log2(norm_ratio)'].max():.3f}")
    
    return data


def remove_contaminants_and_reverses(data: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows containing 'contaminant_' or 'Reverse_' in ACCESSION column.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data
        
    Returns
    -------
    pd.DataFrame
        Data with contaminant and reverse entries removed
    """
    if 'ACCESSION' not in data.columns:
        logger.warning("ACCESSION column not found, skipping contaminant/reverse removal")
        return data
    
    logger.info("Removing contaminant and reverse entries...")
    
    initial_rows = len(data)
    
    # Create mask to keep rows that don't contain contaminant_ or Reverse_
    mask = ~(data['ACCESSION'].str.contains('contaminant_', case=False, na=False) | 
             data['ACCESSION'].str.contains('Reverse_', case=False, na=False))
    
    # Count what will be removed
    contaminant_count = data['ACCESSION'].str.contains('contaminant_', case=False, na=False).sum()
    reverse_count = data['ACCESSION'].str.contains('Reverse_', case=False, na=False).sum()
    
    # Apply filter
    filtered_data = data[mask]
    
    removed_count = initial_rows - len(filtered_data)
    
    logger.info(f"Removed {contaminant_count} contaminant entries")
    logger.info(f"Removed {reverse_count} reverse entries")
    logger.info(f"Total rows removed: {removed_count}")
    
    return filtered_data


def create_volcano_plot(data: pd.DataFrame, output_path: str, pvalue_threshold: float = 0.05, fold_change_threshold: float = 1.0) -> None:
    """
    Create a volcano plot from proteomics data.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data containing log2(norm_ratio) and NORM_PVALUE_1 columns
    output_path : str
        Path for the plot output file
    pvalue_threshold : float
        P-value threshold for significance
    fold_change_threshold : float
        Log2 fold change threshold for significance
    """
    if 'log2(norm_ratio)' not in data.columns or 'NORM_PVALUE_1' not in data.columns:
        logger.warning("Required columns (log2(norm_ratio) or NORM_PVALUE_1) not found, skipping volcano plot")
        return
    
    logger.info("Creating volcano plot...")
    
    # Calculate -log10(p-value)
    data['neg_log10_pvalue'] = -np.log10(data['NORM_PVALUE_1'])
    
    # Define significance criteria
    significant = (data['NORM_PVALUE_1'] < pvalue_threshold) & \
                 (abs(data['log2(norm_ratio)']) >= fold_change_threshold)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot non-significant points
    ax.scatter(
        data.loc[~significant, 'log2(norm_ratio)'],
        data.loc[~significant, 'neg_log10_pvalue'],
        alpha=0.6, color='gray', s=20, label='Not significant'
    )
    
    # Plot significant points
    up_regulated = significant & (data['log2(norm_ratio)'] > 0)
    down_regulated = significant & (data['log2(norm_ratio)'] < 0)
    
    if up_regulated.any():
        ax.scatter(
            data.loc[up_regulated, 'log2(norm_ratio)'],
            data.loc[up_regulated, 'neg_log10_pvalue'],
            alpha=0.8, color='red', s=30, label='Up-regulated'
        )
    
    if down_regulated.any():
        ax.scatter(
            data.loc[down_regulated, 'log2(norm_ratio)'],
            data.loc[down_regulated, 'neg_log10_pvalue'],
            alpha=0.8, color='blue', s=30, label='Down-regulated'
        )
    
    # Add gene labels for significant points (without boxes)
    if 'GENE' in data.columns:
        # Label up-regulated genes
        if up_regulated.any():
            up_data = data[up_regulated]
            for idx, row in up_data.iterrows():
                gene_name = row['GENE'] if pd.notna(row['GENE']) else row['ACCESSION']
                ax.annotate(
                    gene_name,
                    (row['log2(norm_ratio)'], row['neg_log10_pvalue']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.8, color='red'
                )
        
        # Label down-regulated genes
        if down_regulated.any():
            down_data = data[down_regulated]
            for idx, row in down_data.iterrows():
                gene_name = row['GENE'] if pd.notna(row['GENE']) else row['ACCESSION']
                ax.annotate(
                    gene_name,
                    (row['log2(norm_ratio)'], row['neg_log10_pvalue']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.8, color='blue'
                )
    
    # Add threshold lines
    ax.axhline(-np.log10(pvalue_threshold), color='black', linestyle='--', alpha=0.5)
    ax.axvline(fold_change_threshold, color='black', linestyle='--', alpha=0.5)
    ax.axvline(-fold_change_threshold, color='black', linestyle='--', alpha=0.5)
    
    # Labels and title
    ax.set_xlabel('log2(Norm Ratio)')
    ax.set_ylabel('-log10(P-value)')
    ax.set_title('Volcano Plot')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save the plot
    plot_path = Path(output_path).with_suffix('.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Volcano plot saved to: {plot_path}")
    
    # Print summary statistics
    total_sig = significant.sum()
    up_sig = up_regulated.sum()
    down_sig = down_regulated.sum()
    
    logger.info(f"Volcano plot summary:")
    logger.info(f"  Total significant proteins: {total_sig}")
    logger.info(f"  Up-regulated: {up_sig}")
    logger.info(f"  Down-regulated: {down_sig}")


def clean_proteomics_data(data: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Clean proteomics data by removing duplicates and invalid values.
    
    Parameters
    ----------
    data : pd.DataFrame
        Raw proteomics data
    verbose : bool
        Whether to print detailed information about cleaning steps
        
    Returns
    -------
    pd.DataFrame
        Cleaned data
    """
    cleaned_data = data.copy()
    initial_rows = len(cleaned_data)
    
    # Step 1: Remove duplicates based on GENE column
    if 'GENE' in cleaned_data.columns:
        logger.info("Removing duplicates based on GENE column...")
        duplicates_before = cleaned_data.duplicated(subset=['GENE']).sum()
        cleaned_data = cleaned_data.drop_duplicates(subset=['GENE'], keep='first')
        duplicates_removed = initial_rows - len(cleaned_data)
        
        if verbose:
            logger.info(f"Found {duplicates_before} duplicate entries")
            logger.info(f"Removed {duplicates_removed} duplicate rows")
    else:
        logger.warning("GENE column not found in data")
    
    # Step 2: Remove contaminant and reverse entries
    cleaned_data = remove_contaminants_and_reverses(cleaned_data)
    
    # Step 3: Remove rows with 0 or infinity values in NORM_RATIO_2_1 column
    if 'NORM_RATIO_2_1' in cleaned_data.columns:
        logger.info("Removing rows with 0 or infinity values in NORM_RATIO_2_1 column...")
        
        # Count invalid values before removal
        zero_values = (cleaned_data['NORM_RATIO_2_1'] == 0).sum()
        inf_values = np.isinf(cleaned_data['NORM_RATIO_2_1']).sum()
        
        # Remove rows with 0 or infinity values
        mask = (cleaned_data['NORM_RATIO_2_1'] != 0) & (~np.isinf(cleaned_data['NORM_RATIO_2_1']))
        cleaned_data = cleaned_data[mask]
        
        invalid_removed = zero_values + inf_values
        
        if verbose:
            logger.info(f"Found {zero_values} rows with zero values")
            logger.info(f"Found {inf_values} rows with infinity values")
            logger.info(f"Removed {invalid_removed} rows with invalid values")
    else:
        logger.warning("NORM_RATIO_2_1 column not found in data")
    
    # Step 4: Add log2 transformation
    cleaned_data = add_log2_column(cleaned_data)
    
    # Summary
    final_rows = len(cleaned_data)
    total_removed = initial_rows - final_rows
    
    logger.info(f"Data cleaning complete:")
    logger.info(f"  Initial rows: {initial_rows}")
    logger.info(f"  Final rows: {final_rows}")
    logger.info(f"  Total rows removed: {total_removed}")
    logger.info(f"  Data retention: {final_rows/initial_rows*100:.1f}%")
    
    return cleaned_data


def save_cleaned_data(data: pd.DataFrame, output_path: str) -> None:
    """
    Save cleaned data to Excel or CSV file.
    
    Parameters
    ----------
    data : pd.DataFrame
        Cleaned data to save
    output_path : str
        Path for output file
    """
    try:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        ext = output_file.suffix.lower()
        if ext in ['.xlsx', '.xls']:
            data.to_excel(output_path, index=False)
        elif ext == '.csv':
            data.to_csv(output_path, index=False)
        else:
            raise ValueError(f"Unsupported output file extension: {ext}")
        logger.info(f"Cleaned data saved to: {output_path}")
    except Exception as e:
        logger.error(f"Error saving file {output_path}: {str(e)}")
        raise


def validate_input_file(file_path: str) -> bool:
    """
    Validate that input file exists and is an Excel or CSV file.
    
    Parameters
    ----------
    file_path : str
        Path to input file
        
    Returns
    -------
    bool
        True if file is valid
    """
    file_path = Path(file_path)
    if not file_path.exists():
        logger.error(f"Input file does not exist: {file_path}")
        return False
    if not file_path.suffix.lower() in ['.xlsx', '.xls', '.csv']:
        logger.error(f"Input file must be an Excel or CSV file (.xlsx, .xls, .csv): {file_path}")
        return False
    return True


def main():
    """Main function to run the data cleaning pipeline."""
    parser = argparse.ArgumentParser(
        description="Clean proteomics data by removing duplicates and invalid values",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python clean_proteomics_data.py input.xlsx output.xlsx
    python clean_proteomics_data.py input.csv output.csv
    python clean_proteomics_data.py input.csv output.xlsx --verbose --plot
        """
    )
    
    parser.add_argument(
        'input_file',
        help='Path to input Excel or CSV file'
    )
    
    parser.add_argument(
        'output_file',
        help='Path for output Excel or CSV file'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print detailed information about cleaning steps'
    )
    
    parser.add_argument(
        '--plot', '-p',
        action='store_true',
        help='Generate volcano plot'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not validate_input_file(args.input_file):
        sys.exit(1)
    
    try:
        # Load data
        data = load_data(args.input_file)
        
        # Clean data
        cleaned_data = clean_proteomics_data(data, verbose=args.verbose)
        
        # Save cleaned data
        save_cleaned_data(cleaned_data, args.output_file)
        
        # Create volcano plot if requested
        if args.plot:
            create_volcano_plot(cleaned_data, args.output_file)
        
        logger.info("Data cleaning completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during data cleaning: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 