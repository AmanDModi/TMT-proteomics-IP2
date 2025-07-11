"""
Proteomics Analysis Pipeline

A comprehensive Python package for analyzing raw proteomics data from mass spectrometry experiments.
"""

__version__ = "0.1.0"
__author__ = "Aman Modi"

from .data_loader import load_proteomics_data
from .preprocessing import preprocess_data
from .analysis import analyze_proteins
from .visualization import create_plots

__all__ = [
    "load_proteomics_data",
    "preprocess_data", 
    "analyze_proteins",
    "create_plots",
] 