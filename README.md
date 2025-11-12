# 📊 Asset Returns Covariance Estimation & Comparison

An interactive Streamlit application for exploring and comparing different methods of estimating covariance matrices for a selected set of S&P 500 stock returns.

## Overview

This application allows users to visualize and compare various covariance estimation techniques on historical stock market data. It's particularly useful for understanding how different estimation methods perform and how they can be applied in portfolio analysis and risk management.

## Features

- **Multiple Estimation Methods:**
  - Sample Covariance (standard approach)
  - Ledoit-Wolf Shrinkage
  - Oracle Approximating Shrinkage (OAS)
  - Exponentially Weighted Moving Average (EWMA)

- **Interactive Visualization:**
  - Heatmaps for each covariance matrix
  - Side-by-side comparison of estimation methods
  - Real-time data fetching from Yahoo Finance

- **Customizable Parameters:**
  - Select from 32 popular S&P 500 stocks
  - Adjustable historical data range
  - Configurable EWMA decay factor (λ)
  - Frobenius norm difference metrics for comparison

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/sh-mukherjee/covariance_estimate.git
cd covariance_estimate
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

### Required Packages

```
streamlit
numpy
pandas
plotly
yfinance
scikit-learn
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser to the provided local URL (typically `http://localhost:8501`)

3. Use the sidebar to:
   - Select stock tickers from the S&P 500
   - Choose the historical data range (in years)
   - Adjust the EWMA decay factor

4. View the comparison of covariance estimation methods with interactive heatmaps and difference metrics

## Methodology

### Covariance Estimation Methods

**Sample Covariance**
- Standard unbiased estimator
- Can be unstable with limited data or high dimensionality

**Ledoit-Wolf Shrinkage**
- Shrinks sample covariance toward a structured target
- Optimal for high-dimensional settings with limited samples
- Requires n_samples ≥ n_features

**Oracle Approximating Shrinkage (OAS)**
- Alternative shrinkage estimator
- Often provides better performance than Ledoit-Wolf
- Requires n_samples ≥ 2

**EWMA Covariance**
- Assigns exponentially decreasing weights to older observations
- More responsive to recent market conditions
- Captures volatility clustering effectively
- λ parameter controls decay speed (typical values: 0.94-0.97)

### Performance Metrics

The **Frobenius Norm Difference** quantifies the discrepancy between each estimated covariance matrix and the reference (full historical) covariance matrix:

```
||A - B||_F = sqrt(sum of squared differences of all elements)
```

Lower values indicate better agreement with the reference matrix.

## Use Cases

- **Portfolio Optimization:** Understanding asset correlations for efficient frontier analysis
- **Risk Management:** Quantifying portfolio volatility and Value-at-Risk (VaR)
- **Trading Strategy Development:** Identifying covariance structure changes over time
- **Educational:** Learning about covariance estimation in financial applications

## Technical Notes

- Data is fetched from Yahoo Finance and cached for 1 hour
- Log returns are calculated as: `ln(P_t / P_{t-1})`
- All matrices use consistent color scaling for easy comparison
- Missing data is handled automatically with appropriate warnings

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).

## Author

**Sh. Mukherjee**

## Acknowledgments

- Data provided by Yahoo Finance via the `yfinance` library
- Shrinkage estimators implemented using scikit-learn
- Visualization powered by Plotly

## Contact

For questions or feedback, please open an issue on GitHub.

---

**Note:** This application is for educational and research purposes only. It should not be used as the sole basis for investment decisions.
