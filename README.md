# Proteomics Data Analysis Pipeline

Pipeline for processing IP2 TMT data

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

