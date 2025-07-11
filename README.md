# Proteomics Data Analysis Pipeline

A comprehensive Python pipeline for analyzing raw proteomics data from mass spectrometry experiments.

## Project Structure

```
proteomics_analysis_project/
├── src/                    # Source code
│   ├── __init__.py
│   ├── data_loader.py     # Data loading utilities
│   ├── preprocessing.py    # Data preprocessing functions
│   ├── analysis.py        # Core analysis functions
│   └── visualization.py   # Plotting and visualization
├── data/                  # Raw and processed data
│   ├── raw/              # Raw mass spec data
│   └── processed/        # Processed data files
├── docs/                 # Documentation
├── tests/                # Unit tests
├── notebooks/            # Jupyter notebooks for exploration
├── requirements.txt      # Python dependencies
├── setup.py             # Package setup
└── README.md           # This file
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/AmanDModi/TMT-proteomics-IP2.git
cd TMT-proteomics-IP2
```

2. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from src.data_loader import load_proteomics_data
from src.preprocessing import preprocess_data
from src.analysis import analyze_proteins
from src.visualization import create_plots

# Load your data
data = load_proteomics_data('data/raw/your_data.csv')

# Preprocess
processed_data = preprocess_data(data)

# Analyze
results = analyze_proteins(processed_data)

# Visualize
create_plots(results)
```

### Jupyter Notebooks

For interactive analysis, launch Jupyter:
```bash
jupyter notebook
```

## Features

- **Data Loading**: Support for various proteomics data formats (CSV, Excel, mzML)
- **Preprocessing**: Quality control, normalization, and filtering
- **Statistical Analysis**: Differential expression, pathway analysis
- **Visualization**: Volcano plots, heatmaps, PCA plots
- **Export**: Results export in multiple formats

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Contact

For questions or issues, please open an issue on GitHub.