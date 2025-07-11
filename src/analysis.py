"""
Statistical analysis functions for proteomics data.

This module provides functions for differential expression analysis,
statistical testing, and pathway analysis.
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, List, Tuple, Dict, Any
import logging
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def analyze_proteins(
    data: pd.DataFrame,
    group1: List[str],
    group2: List[str],
    analysis_type: str = 'differential_expression',
    **kwargs
) -> Dict[str, Any]:
    """
    Main analysis function for proteomics data.
    
    Parameters
    ----------
    data : pd.DataFrame
        Preprocessed proteomics data
    group1 : list
        Column names for first group (e.g., control)
    group2 : list
        Column names for second group (e.g., treatment)
    analysis_type : str, default 'differential_expression'
        Type of analysis to perform
    **kwargs
        Additional arguments for specific analysis types
        
    Returns
    -------
    dict
        Analysis results
    """
    if analysis_type == 'differential_expression':
        results = differential_expression_analysis(data, group1, group2, **kwargs)
    elif analysis_type == 'pca':
        results = pca_analysis(data, **kwargs)
    elif analysis_type == 'clustering':
        results = clustering_analysis(data, **kwargs)
    else:
        raise ValueError(f"Unknown analysis type: {analysis_type}")
    
    return results


def differential_expression_analysis(
    data: pd.DataFrame,
    group1: List[str],
    group2: List[str],
    test_method: str = 't_test',
    pvalue_threshold: float = 0.05,
    fold_change_threshold: float = 1.5,
    **kwargs
) -> Dict[str, Any]:
    """
    Perform differential expression analysis.
    
    Parameters
    ----------
    data : pd.DataFrame
        Preprocessed proteomics data
    group1 : list
        Column names for first group
    group2 : list
        Column names for second group
    test_method : str, default 't_test'
        Statistical test method ('t_test', 'wilcoxon', 'mann_whitney')
    pvalue_threshold : float, default 0.05
        P-value threshold for significance
    fold_change_threshold : float, default 1.5
        Fold change threshold for significance
    **kwargs
        Additional arguments
        
    Returns
    -------
    dict
        Differential expression results
    """
    # Validate input columns
    all_columns = group1 + group2
    missing_columns = [col for col in all_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in data: {missing_columns}")
    
    results = {
        'method': test_method,
        'group1': group1,
        'group2': group2,
        'pvalue_threshold': pvalue_threshold,
        'fold_change_threshold': fold_change_threshold,
        'results': []
    }
    
    # Get numeric columns for analysis
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    for protein in numeric_cols:
        if protein in all_columns:
            continue  # Skip group columns
        
        # Get group data
        group1_data = data.loc[:, group1].dropna()
        group2_data = data.loc[:, group2].dropna()
        
        if group1_data.empty or group2_data.empty:
            continue
        
        # Calculate statistics
        stats_result = _calculate_differential_stats(
            group1_data[protein], 
            group2_data[protein], 
            test_method
        )
        
        # Calculate fold change
        mean1 = group1_data[protein].mean()
        mean2 = group2_data[protein].mean()
        
        if mean1 > 0:
            fold_change = mean2 / mean1
            log2_fold_change = np.log2(fold_change)
        else:
            fold_change = np.nan
            log2_fold_change = np.nan
        
        # Determine significance
        significant = (stats_result['pvalue'] < pvalue_threshold and 
                     abs(fold_change) >= fold_change_threshold)
        
        # Determine regulation direction
        if significant:
            if fold_change > 1:
                regulation = 'up'
            else:
                regulation = 'down'
        else:
            regulation = 'ns'
        
        results['results'].append({
            'protein': protein,
            'mean_group1': mean1,
            'mean_group2': mean2,
            'fold_change': fold_change,
            'log2_fold_change': log2_fold_change,
            'pvalue': stats_result['pvalue'],
            'statistic': stats_result['statistic'],
            'significant': significant,
            'regulation': regulation
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results['results'])
    results['results_df'] = results_df
    
    # Summary statistics
    results['summary'] = {
        'total_proteins': len(results_df),
        'significant_proteins': len(results_df[results_df['significant']]),
        'up_regulated': len(results_df[results_df['regulation'] == 'up']),
        'down_regulated': len(results_df[results_df['regulation'] == 'down'])
    }
    
    logger.info(f"Differential expression analysis complete: {results['summary']['significant_proteins']} significant proteins found")
    
    return results


def _calculate_differential_stats(
    group1_data: pd.Series,
    group2_data: pd.Series,
    test_method: str
) -> Dict[str, float]:
    """
    Calculate statistical test results.
    
    Parameters
    ----------
    group1_data : pd.Series
        Data for first group
    group2_data : pd.Series
        Data for second group
    test_method : str
        Statistical test method
        
    Returns
    -------
    dict
        Test statistic and p-value
    """
    if test_method == 't_test':
        statistic, pvalue = stats.ttest_ind(group1_data, group2_data)
    elif test_method == 'wilcoxon':
        statistic, pvalue = stats.wilcoxon(group1_data, group2_data)
    elif test_method == 'mann_whitney':
        statistic, pvalue = stats.mannwhitneyu(group1_data, group2_data, alternative='two-sided')
    else:
        raise ValueError(f"Unknown test method: {test_method}")
    
    return {'statistic': statistic, 'pvalue': pvalue}


def pca_analysis(
    data: pd.DataFrame,
    n_components: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Perform Principal Component Analysis.
    
    Parameters
    ----------
    data : pd.DataFrame
        Preprocessed proteomics data
    n_components : int, optional
        Number of components to keep
    **kwargs
        Additional arguments
        
    Returns
    -------
    dict
        PCA results
    """
    # Select numeric columns
    numeric_data = data.select_dtypes(include=[np.number])
    
    if numeric_data.empty:
        raise ValueError("No numeric columns found for PCA")
    
    # Remove missing values
    numeric_data = numeric_data.dropna()
    
    if numeric_data.empty:
        raise ValueError("No data remaining after removing missing values")
    
    # Standardize data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(numeric_data)
    
    # Perform PCA
    if n_components is None:
        n_components = min(numeric_data.shape[0], numeric_data.shape[1])
    
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(data_scaled)
    
    # Calculate explained variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    results = {
        'pca_result': pca_result,
        'components': pca.components_,
        'explained_variance': explained_variance,
        'cumulative_variance': cumulative_variance,
        'feature_names': numeric_data.columns.tolist(),
        'n_components': n_components,
        'total_variance_explained': cumulative_variance[-1]
    }
    
    logger.info(f"PCA complete: {n_components} components explain {results['total_variance_explained']:.2%} of variance")
    
    return results


def clustering_analysis(
    data: pd.DataFrame,
    n_clusters: int = 3,
    method: str = 'kmeans',
    **kwargs
) -> Dict[str, Any]:
    """
    Perform clustering analysis.
    
    Parameters
    ----------
    data : pd.DataFrame
        Preprocessed proteomics data
    n_clusters : int, default 3
        Number of clusters
    method : str, default 'kmeans'
        Clustering method
    **kwargs
        Additional arguments
        
    Returns
    -------
    dict
        Clustering results
    """
    # Select numeric columns
    numeric_data = data.select_dtypes(include=[np.number])
    
    if numeric_data.empty:
        raise ValueError("No numeric columns found for clustering")
    
    # Remove missing values
    numeric_data = numeric_data.dropna()
    
    if numeric_data.empty:
        raise ValueError("No data remaining after removing missing values")
    
    # Standardize data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(numeric_data)
    
    if method == 'kmeans':
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = clusterer.fit_predict(data_scaled)
        cluster_centers = clusterer.cluster_centers_
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    
    results = {
        'cluster_labels': cluster_labels,
        'cluster_centers': cluster_centers,
        'n_clusters': n_clusters,
        'method': method,
        'feature_names': numeric_data.columns.tolist(),
        'data_scaled': data_scaled
    }
    
    # Add cluster information to original data
    data_with_clusters = data.copy()
    data_with_clusters['cluster'] = cluster_labels
    results['data_with_clusters'] = data_with_clusters
    
    logger.info(f"Clustering complete: {n_clusters} clusters identified")
    
    return results


def calculate_correlation_matrix(
    data: pd.DataFrame,
    method: str = 'pearson',
    **kwargs
) -> pd.DataFrame:
    """
    Calculate correlation matrix for proteomics data.
    
    Parameters
    ----------
    data : pd.DataFrame
        Preprocessed proteomics data
    method : str, default 'pearson'
        Correlation method ('pearson', 'spearman', 'kendall')
    **kwargs
        Additional arguments
        
    Returns
    -------
    pd.DataFrame
        Correlation matrix
    """
    # Select numeric columns
    numeric_data = data.select_dtypes(include=[np.number])
    
    if numeric_data.empty:
        raise ValueError("No numeric columns found for correlation analysis")
    
    # Calculate correlation matrix
    corr_matrix = numeric_data.corr(method=method)
    
    logger.info(f"Correlation matrix calculated using {method} method")
    
    return corr_matrix


def pathway_analysis(
    differential_results: Dict[str, Any],
    pathway_database: Optional[pd.DataFrame] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Perform pathway analysis on differential expression results.
    
    Parameters
    ----------
    differential_results : dict
        Results from differential expression analysis
    pathway_database : pd.DataFrame, optional
        Pathway database with protein-pathway mappings
    **kwargs
        Additional arguments
        
    Returns
    -------
    dict
        Pathway analysis results
    """
    # This is a placeholder for pathway analysis
    # In practice, you would integrate with pathway databases like KEGG, Reactome, etc.
    
    logger.warning("Pathway analysis not yet implemented")
    
    return {
        'message': 'Pathway analysis requires integration with pathway databases',
        'differential_results': differential_results
    } 