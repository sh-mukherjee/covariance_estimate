import streamlit as st
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import pandas as pd
from sklearn.covariance import LedoitWolf, OAS # Oracle Approximating Shrinkage
from datetime import datetime, timedelta

# --- App Configuration ---
st.set_page_config(layout="wide", page_title="Asset Covariance Estimation Demo")

# --- Predefined list of popular SP500 tickers ---
# A small subset for demonstration purposes.
SP500_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "PG",
    "JNJ", "UNH", "XOM", "CVX", "HD", "KO", "PEP", "DIS", "NFLX", "ADBE", "MS",
    "BAC", "WFC", "GS", "SCHW", "C", "AXP", "BLK", "SPG", "PLD", "EQIX", "AMT"
]

# --- Functions for Covariance Estimation ---

def calculate_ewma_covariance(returns, lambda_decay=0.94):
    """
    Calculates the Exponentially Weighted Moving Average (EWMA) covariance matrix.
    Args:
        returns (pd.DataFrame): DataFrame of daily log returns (rows are dates, columns are assets).
        lambda_decay (float): The decay factor (between 0 and 1). Higher lambda means slower decay.
    Returns:
        np.array: The EWMA covariance matrix.
    """
    if returns.empty:
        return np.array([[]])
    
    T, N = returns.shape # T = number of observations, N = number of assets
    
    # Demean returns for covariance calculation
    demeaned_returns = returns - returns.mean()

    # Initialize covariance matrix (often starts with simple sample covariance for the first few points,
    # or just zeros for simplicity in a recursive definition)
    # For a fixed window, we directly calculate the weighted sum of outer products.
    
    weights = np.array([lambda_decay**(T - 1 - i) for i in range(T)])
    # Normalize weights so they sum to 1 if you want unbiased.
    # However, for the formula Sigma_t = lambda * Sigma_{t-1} + (1-lambda) * r_t * r_t^T
    # the sum of weights is usually not 1 across the window, but the (1-lambda) factor ensures it.
    # For simplicity, we'll use the common weighted sum of outer products.
    weights = weights / np.sum(weights) # Normalize for proper weighted average of products
    
    ewma_cov = np.zeros((N, N))
    for i in range(T):
        rt_outer_product = np.outer(demeaned_returns.iloc[i], demeaned_returns.iloc[i])
        ewma_cov += weights[i] * rt_outer_product
        
    return ewma_cov



# --- App Title and Introduction ---
st.title("📊 Asset Returns Covariance Estimation & Comparison")
st.markdown("""
This application allows you to explore different methods for estimating the **covariance matrix of daily log returns**
for selected S&P 500 stocks.
""")
st.markdown("---")
st.markdown("""
### How to Interpret the Results:
* **Estimated Covariance Matrices:** These are calculated using the `n` most recent days of log returns.
* **Heatmaps:** Visualize the magnitudes of variances (diagonal) and covariances (off-diagonal).
* **Frobenius Norm Difference:** Quantifies the discrepancy between the estimated and sample covariance matrices.
* **Estimator Performance:**
    * **Ledoit-Wolf & OAS Shrinkage:** These are often more stable and accurate than sample covariance, especially in high-dimensional settings or with limited data, by "shrinking" the extreme values towards a more stable target.
    * **EWMA Covariance:** Particularly useful in finance as it assigns more weight to recent returns, making it more responsive to current market conditions and volatility clusters. The `lambda` parameter controls the decay speed.
""")

#st.markdown("---")

# --- Sidebar for User Inputs ---
st.sidebar.header("Data & Estimation Parameters")

selected_tickers = st.sidebar.multiselect(
    "Select S&P 500 Stock Tickers:",
    options=SP500_TICKERS,
    default=["AAPL", "MSFT", "GOOGL", "AMZN"], # Default selection
    help="Choose stocks for which to estimate covariance."
)

if not selected_tickers:
    st.warning("Please select at least one stock ticker to proceed.")
    st.stop()

# Date range selection for historical data
n_years =st.sidebar.slider(
    "Select Sample Size (number of years for estimation):",
    min_value=1,
    max_value=len(selected_tickers),
    value=1, # Default to approx 1 year of trading days
    step=1,
    help=f"Number of years of past data to use for estimation (up to {len(selected_tickers)})."
)
end_date = datetime.now()
start_date = end_date - timedelta(days=max(365, n_years*365)) # minimum period of 1 year


# --- Data Fetching ---
@st.cache_data(ttl=3600) # Cache data for 1 hour to avoid repeated downloads
def fetch_stock_data(tickers, start, end):
    """Fetches adjusted close prices for given tickers and calculates log returns."""
    st.sidebar.info(f"Fetching data for {len(tickers)} stocks from {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}...")
    try:
        data = yf.download(tickers, start=start, end=end, auto_adjust=False)['Adj Close']
        
        if isinstance(data, pd.Series): # Handle case of single ticker returning a Series
            data = data.to_frame(name=tickers[0])
        
        if data.empty:
            st.error("No data downloaded for selected tickers or date range. Please try different selections.")
            return None
        
        # Drop any columns (tickers) that have all NaN values
        data = data.dropna(axis=1, how='all')

        # Calculate daily log returns
        log_returns = np.log(data / data.shift(1)).dropna()
        
        if log_returns.empty:
            st.error("Not enough data to calculate log returns for all selected tickers. Please try different tickers or a wider date range.")
            return None

        return log_returns
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

log_returns_full = fetch_stock_data(selected_tickers, start_date, end_date)

if log_returns_full is None or log_returns_full.empty:
    st.stop() # Stop if no valid log returns are available

# Update num_features based on actual fetched data columns
num_features = log_returns_full.shape[1]
if num_features == 0:
    st.error("No valid stock data found for selected tickers after processing. Please try different selections.")
    st.stop()

# --- Define the "True" (Reference) Covariance Matrix ---
# We'll use the covariance of the full historical period's log returns as our reference.
true_covariance = log_returns_full.cov().values

# Determine global min/max for consistent color scaling in Plotly heatmaps
# We might set a more robust range if values vary wildly, but this is a good start.
matrix_min_val = np.min(true_covariance)
matrix_max_val = np.max(true_covariance)

# --- User Input for Sample Size (n_days) ---
max_days_available = log_returns_full.shape[0]
n_samples = max_days_available

# EWMA decay factor
ewma_lambda = st.sidebar.slider(
    "EWMA Decay Factor (Lambda λ):",
    min_value=0.01,
    max_value=0.99,
    value=0.94,
    step=0.01,
    help="Higher lambda means slower decay, giving more weight to older observations. Common values are 0.94-0.97."
)




# --- Main Content Area - Reference Covariance ---


st.markdown("---")

st.header(f"Sample and Estimated Covariance Matrices (n = {n_samples} recent days)")

# --- Prepare Data for Estimation ---
if n_samples == 0:
    st.warning("Please select a sample size greater than 0.")
    st.stop()

simulated_data = log_returns_full #.tail(n_samples)

# Ensure data is 2D, even if only one feature or one sample for covariance functions
if num_features == 1:
    simulated_data_for_cov = simulated_data.values.reshape(-1, 1)
else:
    simulated_data_for_cov = simulated_data.values

# --- Calculate Estimators and Metrics ---
estimation_results = {}

# 1. Sample Covariance
estimated_covariance_sample = np.cov(simulated_data_for_cov, rowvar=False)
frobenius_diff_sample = np.linalg.norm(true_covariance - estimated_covariance_sample, 'fro')
estimation_results['Sample Covariance'] = {
    'matrix': estimated_covariance_sample,
    'diff': frobenius_diff_sample
}

# 2. Ledoit-Wolf Shrinkage
if n_samples >= num_features or num_features == 1 and n_samples >= 1: # LedoitWolf can fail if n < p, or n=1 for p>1
    try:
        lw_estimator = LedoitWolf()
        lw_estimated_covariance = lw_estimator.fit(simulated_data_for_cov).covariance_
        frobenius_diff_lw = np.linalg.norm(true_covariance - lw_estimated_covariance, 'fro')
        estimation_results['Ledoit-Wolf Shrinkage'] = {
            'matrix': lw_estimated_covariance,
            'diff': frobenius_diff_lw
        }
    except Exception as e:
        st.warning(f"Ledoit-Wolf estimator could not be computed: {e}. (Often requires n_samples >= num_features if num_features > 1)")
        estimation_results['Ledoit-Wolf Shrinkage'] = {'matrix': np.full_like(true_covariance, np.nan), 'diff': np.nan}
else:
    st.warning(f"Ledoit-Wolf estimator not computed: Requires n_samples ({n_samples}) >= num_features ({num_features}) for multiple features.")
    estimation_results['Ledoit-Wolf Shrinkage'] = {'matrix': np.full_like(true_covariance, np.nan), 'diff': np.nan}


# 3. Oracle Approximating Shrinkage (OAS)
if n_samples >= 2: # OAS requires at least 2 samples
    try:
        oas_estimator = OAS()
        oas_estimated_covariance = oas_estimator.fit(simulated_data_for_cov).covariance_
        frobenius_diff_oas = np.linalg.norm(true_covariance - oas_estimated_covariance, 'fro')
        estimation_results['OAS Shrinkage'] = {
            'matrix': oas_estimated_covariance,
            'diff': frobenius_diff_oas
        }
    except Exception as e:
        st.warning(f"OAS estimator could not be computed: {e}. (Often requires n_samples >= 2)")
        estimation_results['OAS Shrinkage'] = {'matrix': np.full_like(true_covariance, np.nan), 'diff': np.nan}
else:
    st.warning(f"OAS estimator not computed: Requires n_samples ({n_samples}) >= 2.")
    estimation_results['OAS Shrinkage'] = {'matrix': np.full_like(true_covariance, np.nan), 'diff': np.nan}

# 4. EWMA Covariance
if n_samples >= 1: # EWMA can be computed from 1 sample, but meaningful from more
    ewma_estimated_covariance = calculate_ewma_covariance(simulated_data, ewma_lambda)
    frobenius_diff_ewma = np.linalg.norm(true_covariance - ewma_estimated_covariance, 'fro')
    estimation_results['EWMA Covariance'] = {
        'matrix': ewma_estimated_covariance,
        'diff': frobenius_diff_ewma
    }
else:
    st.warning("EWMA not computed: Requires n_samples >= 1.")
    estimation_results['EWMA Covariance'] = {'matrix': np.full_like(true_covariance, np.nan), 'diff': np.nan}


# --- Display Results in Columns ---
cols_per_row = 2
num_estimators = len(estimation_results)
rows = [st.columns(cols_per_row) for _ in range((num_estimators + cols_per_row - 1) // cols_per_row)]
col_idx = 0

for estimator_name, result in estimation_results.items():
    current_col = rows[col_idx // cols_per_row][col_idx % cols_per_row]
    
    with current_col:
        st.subheader(estimator_name)
        st.write(f"Calculated from {n_samples} recent days:")
        
        # Display DataFrame
        if not np.isnan(result['matrix']).all():
            st.dataframe(pd.DataFrame(result['matrix'], index=log_returns_full.columns, columns=log_returns_full.columns), use_container_width=True)
            
            # Plot Heatmap
            fig_est = go.Figure(data=go.Heatmap(
                z=result['matrix'],
                x=log_returns_full.columns.tolist(),
                y=log_returns_full.columns.tolist(),
                colorscale='Viridis',
                zmin=matrix_min_val,
                zmax=matrix_max_val,
                colorbar=dict(title='Covariance Value')
            ))
            fig_est.update_layout(
                title=f'{estimator_name} (n={n_samples} days)',
                xaxis_title='Stocks',
                yaxis_title='Stocks',
                height=450,
                width=500,
                margin=dict(l=50, r=50, t=50, b=50),
                yaxis=dict(autorange='reversed') # <--- Ensure diagonal runs top-left to bottom-right
            )
            st.plotly_chart(fig_est, use_container_width=False)
            
            if not np.isnan(result['diff']):
                st.metric(label=f"Frobenius Norm Difference ({estimator_name})", value=f"{result['diff']:.4f}")
            else:
                st.write("Difference not available.")
        else:
            st.write("Could not compute matrix for this estimator.")
            st.metric(label=f"Frobenius Norm Difference ({estimator_name})", value="N/A")
            
    col_idx += 1

#* **Reference Covariance:** This is calculated from all available daily log returns for your selected historical period and serves as our best estimate of the "true" underlying covariance.
#* **Sample Covariance:** Simple, but noisy for small `n`. Can be singular if `n < p` (number of features).
