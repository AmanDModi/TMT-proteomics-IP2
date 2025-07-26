"""
Visualization functions for proteomics analysis.

This module provides functions for creating various plots commonly used
in proteomics data analysis including volcano plots, heatmaps, and PCA plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from typing import Union, Optional, List, Dict, Any, Tuple
import logging

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

logger = logging.getLogger(__name__)


def create_plots(
    data: Union[pd.DataFrame, Dict[str, Any]],
    plot_type: str = 'volcano',
    **kwargs
) -> Union[plt.Figure, go.Figure]:
    """
    Main plotting function for proteomics data.
    
    Parameters
    ----------
    data : pd.DataFrame or dict
        Data to plot (either raw data or analysis results)
    plot_type : str, default 'volcano'
        Type of plot to create
    **kwargs
        Additional arguments for specific plot types
        
    Returns
    -------
    matplotlib.Figure or plotly.graph_objects.Figure
        The created plot
    """
    if plot_type == 'volcano':
        return create_volcano_plot(data, **kwargs)
    elif plot_type == 'heatmap':
        return create_heatmap(data, **kwargs)
    elif plot_type == 'pca':
        return create_pca_plot(data, **kwargs)
    elif plot_type == 'correlation':
        return create_correlation_plot(data, **kwargs)
    elif plot_type == 'boxplot':
        return create_boxplot(data, **kwargs)
    elif plot_type == 'scatter':
        return create_scatter_plot(data, **kwargs)
    else:
        raise ValueError(f"Unknown plot type: {plot_type}")


def create_volcano_plot(
    differential_results: Union[Dict[str, Any], pd.DataFrame],
    pvalue_threshold: float = 0.05,
    fold_change_threshold: float = 1.5,
    interactive: bool = False,
    **kwargs
) -> Union[plt.Figure, go.Figure]:
    """
    Create a volcano plot from differential expression results.
    
    Parameters
    ----------
    differential_results : dict or pd.DataFrame
        Results from differential expression analysis (dict with 'results_df' key) 
        or DataFrame directly
    pvalue_threshold : float, default 0.05
        P-value threshold for significance
    fold_change_threshold : float, default 1.5
        Fold change threshold for significance
    interactive : bool, default False
        Whether to create an interactive plotly plot
    **kwargs
        Additional arguments
        
    Returns
    -------
    matplotlib.Figure or plotly.graph_objects.Figure
        Volcano plot
    """
    if isinstance(differential_results, dict):
        if 'results_df' not in differential_results:
            raise ValueError("Differential results must contain 'results_df'")
        results_df = differential_results['results_df']
    else:
        # Assume it's a DataFrame directly
        results_df = differential_results
    
    if interactive:
        return _create_interactive_volcano_plot(
            results_df, pvalue_threshold, fold_change_threshold, **kwargs
        )
    else:
        return _create_static_volcano_plot(
            results_df, pvalue_threshold, fold_change_threshold, **kwargs
        )


def _create_static_volcano_plot(
    results_df: pd.DataFrame,
    pvalue_threshold: float,
    fold_change_threshold: float,
    **kwargs
) -> plt.Figure:
    """Create a static volcano plot using matplotlib."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Rename columns to match expected format
    plot_df = results_df.copy()
    if 'NORM_PVALUE_1' in plot_df.columns:
        plot_df['pvalue'] = plot_df['NORM_PVALUE_1']
    if 'log2_norm_ratio' in plot_df.columns:
        plot_df['log2_fold_change'] = plot_df['log2_norm_ratio']
    
    # Calculate -log10(p-value)
    plot_df['neg_log10_pvalue'] = -np.log10(plot_df['pvalue'])
    
    # Calculate fold change from log2 fold change for significance criteria
    plot_df['fold_change'] = 2 ** plot_df['log2_fold_change']
    
    # Define significance criteria
    significant = (plot_df['pvalue'] < pvalue_threshold) & \
                 (abs(plot_df['fold_change']) >= fold_change_threshold)
    
    # Plot points
    ax.scatter(
        plot_df.loc[~significant, 'log2_fold_change'],
        plot_df.loc[~significant, 'neg_log10_pvalue'],
        alpha=0.6, color='gray', s=20, label='Not significant'
    )
    
    # Plot significant points
    up_regulated = significant & (plot_df['fold_change'] > 1)
    down_regulated = significant & (plot_df['fold_change'] < 1)
    
    if up_regulated.any():
        ax.scatter(
            plot_df.loc[up_regulated, 'log2_fold_change'],
            plot_df.loc[up_regulated, 'neg_log10_pvalue'],
            alpha=0.8, color='red', s=30, label='Up-regulated'
        )
    
    if down_regulated.any():
        ax.scatter(
            plot_df.loc[down_regulated, 'log2_fold_change'],
            plot_df.loc[down_regulated, 'neg_log10_pvalue'],
            alpha=0.8, color='blue', s=30, label='Down-regulated'
        )
    
    # Add threshold lines
    ax.axhline(-np.log10(pvalue_threshold), color='black', linestyle='--', alpha=0.5)
    ax.axvline(np.log2(fold_change_threshold), color='black', linestyle='--', alpha=0.5)
    ax.axvline(-np.log2(fold_change_threshold), color='black', linestyle='--', alpha=0.5)
    
    # Labels and title
    ax.set_xlabel('log2(Fold Change)')
    ax.set_ylabel('-log10(p-value)')
    ax.set_title('Volcano Plot')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def _create_interactive_volcano_plot(
    results_df: pd.DataFrame,
    pvalue_threshold: float,
    fold_change_threshold: float,
    **kwargs
) -> go.Figure:
    """Create an interactive volcano plot using plotly."""
    # Rename columns to match expected format
    plot_df = results_df.copy()
    if 'NORM_PVALUE_1' in plot_df.columns:
        plot_df['pvalue'] = plot_df['NORM_PVALUE_1']
    if 'log2_norm_ratio' in plot_df.columns:
        plot_df['log2_fold_change'] = plot_df['log2_norm_ratio']
    
    # Calculate -log10(p-value)
    plot_df['neg_log10_pvalue'] = -np.log10(plot_df['pvalue'])
    
    # Calculate fold change from log2 fold change for significance criteria
    plot_df['fold_change'] = 2 ** plot_df['log2_fold_change']
    
    # Define significance criteria
    significant = (plot_df['pvalue'] < pvalue_threshold) & \
                 (abs(plot_df['fold_change']) >= fold_change_threshold)
    
    # Create traces
    traces = []
    
    # Not significant
    not_sig = plot_df[~significant]
    if not not_sig.empty:
        traces.append(go.Scatter(
            x=not_sig['log2_fold_change'],
            y=not_sig['neg_log10_pvalue'],
            mode='markers',
            marker=dict(color='gray', size=5, opacity=0.6),
            name='Not significant',
            hovertemplate='<b>%{text}</b><br>log2(FC): %{x}<br>-log10(p): %{y}<extra></extra>',
            text=not_sig.get('ACCESSION', not_sig.get('protein', ''))
        ))
    
    # Up-regulated
    up_reg = plot_df[significant & (plot_df['fold_change'] > 1)]
    if not up_reg.empty:
        traces.append(go.Scatter(
            x=up_reg['log2_fold_change'],
            y=up_reg['neg_log10_pvalue'],
            mode='markers',
            marker=dict(color='red', size=8, opacity=0.8),
            name='Up-regulated',
            hovertemplate='<b>%{text}</b><br>log2(FC): %{x}<br>-log10(p): %{y}<extra></extra>',
            text=up_reg.get('ACCESSION', up_reg.get('protein', ''))
        ))
    
    # Down-regulated
    down_reg = plot_df[significant & (plot_df['fold_change'] < 1)]
    if not down_reg.empty:
        traces.append(go.Scatter(
            x=down_reg['log2_fold_change'],
            y=down_reg['neg_log10_pvalue'],
            mode='markers',
            marker=dict(color='blue', size=8, opacity=0.8),
            name='Down-regulated',
            hovertemplate='<b>%{text}</b><br>log2(FC): %{x}<br>-log10(p): %{y}<extra></extra>',
            text=down_reg.get('ACCESSION', down_reg.get('protein', ''))
        ))
    
    # Create figure
    fig = go.Figure(data=traces)
    
    # Add threshold lines
    fig.add_hline(y=-np.log10(pvalue_threshold), line_dash="dash", line_color="black")
    fig.add_vline(x=np.log2(fold_change_threshold), line_dash="dash", line_color="black")
    fig.add_vline(x=-np.log2(fold_change_threshold), line_dash="dash", line_color="black")
    
    # Update layout
    fig.update_layout(
        title="Volcano Plot",
        xaxis_title="log2(Fold Change)",
        yaxis_title="-log10(p-value)",
        hovermode='closest'
    )
    
    return fig


def create_heatmap(
    data: pd.DataFrame,
    method: str = 'correlation',
    interactive: bool = False,
    **kwargs
) -> Union[plt.Figure, go.Figure]:
    """
    Create a heatmap from proteomics data.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data to plot
    method : str, default 'correlation'
        Type of heatmap ('correlation', 'expression')
    interactive : bool, default False
        Whether to create an interactive plotly plot
    **kwargs
        Additional arguments
        
    Returns
    -------
    matplotlib.Figure or plotly.graph_objects.Figure
        Heatmap
    """
    if method == 'correlation':
        plot_data = data.corr()
        title = 'Correlation Heatmap'
    elif method == 'expression':
        plot_data = data
        title = 'Expression Heatmap'
    else:
        raise ValueError(f"Unknown heatmap method: {method}")
    
    if interactive:
        return _create_interactive_heatmap(plot_data, title, **kwargs)
    else:
        return _create_static_heatmap(plot_data, title, **kwargs)


def _create_static_heatmap(
    data: pd.DataFrame,
    title: str,
    **kwargs
) -> plt.Figure:
    """Create a static heatmap using seaborn."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(
        data,
        annot=False,
        cmap='RdBu_r',
        center=0,
        square=True,
        ax=ax,
        cbar_kws={'shrink': 0.8}
    )
    
    ax.set_title(title)
    plt.tight_layout()
    
    return fig


def _create_interactive_heatmap(
    data: pd.DataFrame,
    title: str,
    **kwargs
) -> go.Figure:
    """Create an interactive heatmap using plotly."""
    fig = go.Figure(data=go.Heatmap(
        z=data.values,
        x=data.columns,
        y=data.index,
        colorscale='RdBu',
        zmid=0
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Features",
        yaxis_title="Features"
    )
    
    return fig


def create_pca_plot(
    pca_results: Dict[str, Any],
    interactive: bool = False,
    **kwargs
) -> Union[plt.Figure, go.Figure]:
    """
    Create PCA plots from PCA analysis results.
    
    Parameters
    ----------
    pca_results : dict
        Results from PCA analysis
    interactive : bool, default False
        Whether to create an interactive plotly plot
    **kwargs
        Additional arguments
        
    Returns
    -------
    matplotlib.Figure or plotly.graph_objects.Figure
        PCA plot
    """
    if interactive:
        return _create_interactive_pca_plot(pca_results, **kwargs)
    else:
        return _create_static_pca_plot(pca_results, **kwargs)


def _create_static_pca_plot(
    pca_results: Dict[str, Any],
    **kwargs
) -> plt.Figure:
    """Create static PCA plots using matplotlib."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Scree plot
    explained_variance = pca_results['explained_variance']
    cumulative_variance = pca_results['cumulative_variance']
    
    ax1.plot(range(1, len(explained_variance) + 1), explained_variance, 'bo-')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_title('Scree Plot')
    ax1.grid(True, alpha=0.3)
    
    # Cumulative variance plot
    ax2.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'ro-')
    ax2.set_xlabel('Principal Component')
    ax2.set_ylabel('Cumulative Explained Variance Ratio')
    ax2.set_title('Cumulative Variance Plot')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def _create_interactive_pca_plot(
    pca_results: Dict[str, Any],
    **kwargs
) -> go.Figure:
    """Create interactive PCA plots using plotly."""
    # Create subplots
    fig = go.Figure()
    
    # Scree plot
    explained_variance = pca_results['explained_variance']
    cumulative_variance = pca_results['cumulative_variance']
    
    fig.add_trace(go.Scatter(
        x=list(range(1, len(explained_variance) + 1)),
        y=explained_variance,
        mode='lines+markers',
        name='Explained Variance',
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=list(range(1, len(cumulative_variance) + 1)),
        y=cumulative_variance,
        mode='lines+markers',
        name='Cumulative Variance',
        line=dict(color='red'),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title='PCA Analysis',
        xaxis_title='Principal Component',
        yaxis_title='Explained Variance Ratio',
        yaxis2=dict(
            title='Cumulative Explained Variance Ratio',
            overlaying='y',
            side='right'
        )
    )
    
    return fig


def create_correlation_plot(
    data: pd.DataFrame,
    method: str = 'pearson',
    interactive: bool = False,
    **kwargs
) -> Union[plt.Figure, go.Figure]:
    """
    Create a correlation plot.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data to plot
    method : str, default 'pearson'
        Correlation method
    interactive : bool, default False
        Whether to create an interactive plotly plot
    **kwargs
        Additional arguments
        
    Returns
    -------
    matplotlib.Figure or plotly.graph_objects.Figure
        Correlation plot
    """
    corr_matrix = data.corr(method=method)
    
    if interactive:
        return _create_interactive_heatmap(corr_matrix, f'{method.capitalize()} Correlation Matrix', **kwargs)
    else:
        return _create_static_heatmap(corr_matrix, f'{method.capitalize()} Correlation Matrix', **kwargs)


def create_boxplot(
    data: pd.DataFrame,
    group_column: Optional[str] = None,
    interactive: bool = False,
    **kwargs
) -> Union[plt.Figure, go.Figure]:
    """
    Create boxplots for proteomics data.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data to plot
    group_column : str, optional
        Column to group by
    interactive : bool, default False
        Whether to create an interactive plotly plot
    **kwargs
        Additional arguments
        
    Returns
    -------
    matplotlib.Figure or plotly.graph_objects.Figure
        Boxplot
    """
    if interactive:
        return _create_interactive_boxplot(data, group_column, **kwargs)
    else:
        return _create_static_boxplot(data, group_column, **kwargs)


def _create_static_boxplot(
    data: pd.DataFrame,
    group_column: Optional[str],
    **kwargs
) -> plt.Figure:
    """Create static boxplots using seaborn."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    if group_column and group_column in data.columns:
        # Melt data for grouped boxplot
        melted_data = data.melt(id_vars=[group_column], var_name='Protein', value_name='Expression')
        sns.boxplot(data=melted_data, x='Protein', y='Expression', hue=group_column, ax=ax)
    else:
        # Simple boxplot
        data.boxplot(ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    
    ax.set_title('Protein Expression Boxplot')
    plt.tight_layout()
    
    return fig


def _create_interactive_boxplot(
    data: pd.DataFrame,
    group_column: Optional[str],
    **kwargs
) -> go.Figure:
    """Create interactive boxplots using plotly."""
    if group_column and group_column in data.columns:
        # Grouped boxplot
        fig = px.box(data, x=group_column, y=data.columns[0], color=group_column)
    else:
        # Simple boxplot
        fig = px.box(data)
    
    fig.update_layout(title='Protein Expression Boxplot')
    return fig


def create_scatter_plot(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    interactive: bool = False,
    **kwargs
) -> Union[plt.Figure, go.Figure]:
    """
    Create a scatter plot.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data to plot
    x_column : str
        Column for x-axis
    y_column : str
        Column for y-axis
    interactive : bool, default False
        Whether to create an interactive plotly plot
    **kwargs
        Additional arguments
        
    Returns
    -------
    matplotlib.Figure or plotly.graph_objects.Figure
        Scatter plot
    """
    if interactive:
        return _create_interactive_scatter_plot(data, x_column, y_column, **kwargs)
    else:
        return _create_static_scatter_plot(data, x_column, y_column, **kwargs)


def _create_static_scatter_plot(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    **kwargs
) -> plt.Figure:
    """Create static scatter plot using matplotlib."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.scatter(data[x_column], data[y_column], alpha=0.6)
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    ax.set_title(f'{x_column} vs {y_column}')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def _create_interactive_scatter_plot(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    **kwargs
) -> go.Figure:
    """Create interactive scatter plot using plotly."""
    fig = px.scatter(data, x=x_column, y=y_column)
    fig.update_layout(title=f'{x_column} vs {y_column}')
    return fig 