## GMF Investments — Portfolio Optimization Strategy##

 ```text
portfolio-optimization/
│
├── src/
│   ├── init.py
│   ├── data_preprocessing.py     # Task 1: Ingestion, cleaning, and log return transforms
│   ├── forecasting_engine.py     # Task 2: Classical Time Series (ARIMA) modeling
│   ├── deep_learning_models.py   # Task 3: Deep Learning (PyTorch Stacked LSTM) engine
│   ├── portfolio_optimizer.py    # Task 4: MPT optimization (Max Sharpe / Efficient Frontier)
│   └── backtest_engine.py        # Task 5: Performance backtesting and risk analytics (VaR/Drawdown)
│
├── run_pipeline.py               # Master orchestration engine
├── requirements.txt              # Environment dependencies configuration
└── README.md                     # Platform overview and reproduction guide
```
---

## 🚀 Architectural Breakdown & Core Tasks

### Task 1: Data Ingestion & Preprocessing (`src/data_preprocessing.py`)
* **Ingestion:** Automatically streams multi-asset daily market data via `yfinance` integration across diverse asset classes (`TSLA` for high-beta equity, `SPY` for market equity, `BND` for fixed income).
* **Data Cleansing:** Implements robust forward-filling (`ffill()`) and backward-filling (`bfill()`) mechanics to eliminate data gaps resulting from non-aligned market holidays.
* **Stationarity Transformations:** Converts raw historical closing feeds into continuous log returns, guaranteeing structural covariance stationarity required for down-stream mathematical modeling:
    $$\\text{Log Return}_t = \\ln\\left(\\frac{P_t}{P_{t-1}}\\right)$$

![Exploratory Data Analysis Plots](notebooks/visuals/eda_comprehensive_plots.png)

### Task 2: Classical Econometric Forecasting (`src/forecasting_engine.py`)
* **Statistical Baseline:** Leverages an `ARIMA(1,0,1)` framework via `statsmodels` to model conditional mean asset returns.
* **Data Splitting:** Implements chronological out-of-sample partitioning (splitting at `2025-01-01`) to eliminate lookahead bias during performance validations.
* **Metrics Engine:** Evaluates prediction deviations via Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE).

### Task 3: Deep Learning Time Series Engine (`src/deep_learning_models.py`)
* **Network Architecture:** Implements a custom, stacked **Long Short-Term Memory (LSTM)** recurrent neural network using `PyTorch`, specifically optimized for high-frequency Python 3.14+ runtimes.
* **Regularization:** Integrates explicit `Dropout (0.2)` layers between memory layers to avoid overfitting on noisy financial dimensions.
* **Sequence Windows:** Uses sliding 30-day sequence lookback buffers normalized via a specialized scalar interval mapping ($[-1, 1]$) to predict next-day returns.

### Task 4: Modern Portfolio Theory Allocation (`src/portfolio_optimizer.py`)
* **Numerical Optimization:** Formulates a Sequential Least Squares Programming (`SLSQP`) bounded optimizer via `scipy.optimize` to locate the absolute mathematical **Maximum Sharpe Ratio Portfolio**.
* **Constraints:** Enforces an institutional long-only layout:
    $$\\sum w_i = 1.0 \\quad \\text{and} \\quad 0 \\le w_i \\le 1$$
* **Annualization Logic:** Scales daily mean arrays and non-linear asset covariance matrices to an exact 252-day business trading year.
![TSLA Future Horizon Forecast](notebooks/visuals/tsla_future_forecast_horizon.png)

![Asset Covariance Matrix Heatmap](notebooks/visuals/covariance_heatmap_matrix.png)

![MPT Efficient Frontier Boundary Curve](notebooks/visuals/efficient_frontier_curve.png)

### Task 5: Backtesting & Risk Budgeting Engine (`src/backtest_engine.py`)
* **Historical Simulation:** Generates continuous historical equity curves starting from a standard baseline capital position of \$10,000.
* **Benchmark Alignment:** Compares the strategy against a classic institutional passive **60/40 Benchmark** (60% SPY / 40% BND).
* **Risk Metrics Engine:** Computes non-linear parameters including:
    * **Maximum Drawdown (MDD):** Captures worst-case peak-to-trough systemic drops.
    * **99% Historical Value at Risk (VaR):** Outlines the maximum expected loss within a single business day at a 99% confidence threshold.

    ![Out of Sample Cumulative Growth Tracking](notebooks/visuals/backtest_cumulative_returns.png)

---

## 📈 Institutional Pipeline Output Summary

Executing the complete multi-stage pipeline yields the following performance log:

### 1. Target Forecast Performance Logs [Asset: TSLA]
| Metric | ARIMA Baseline | Deep Learning LSTM |
| :--- | :---: | :---: |
| **MAE** | 0.026475 | 0.026665 |
| **RMSE** | 0.035944 | 0.036089 |
| **MAPE (%)** | 115.4578% | 156.0502% |

*Note: High MAPE scores are standard when evaluating raw stationary return differentials near zero, rather than raw price paths.*

### 2. Maximum Sharpe Ratio Target Allocation
* **Asset Weights:**
    * `TSLA` (High Volatility Equity): **0.00%**
    * `SPY` (S&P 500 Market Tracking Index): **87.72%**
    * `BND` (Total Bond Market Fixed Income): **12.28%**
* **Expected Annualized Return:** 14.86%
* **Expected Annualized Volatility:** 19.95%
* **Optimized Portfolio Sharpe Ratio:** **0.6448**

### 3. Historical Backtest vs. Institutional Benchmark (60/40)
| Risk-Adjusted Metric | Optimized Strategy | Institutional 60/40 Benchmark |
| :--- | :---: | :---: |
| **Total Cumulative Growth (%)** | 460.81% | 846.71% |
| **Maximum Peak-to-Trough Drawdown** | **-36.40%** | -47.78% |
| **99% Daily Value at Risk (VaR)** | **-3.57%** | -5.13% |

> **Quantitative Insight:** By dropping the highly volatile asset (TSLA) and scaling into a risk-optimized frontier combination of SPY and BND, the **Optimized Strategy demonstrated vastly superior risk mitigation**, capping maximum downside drawdowns by **11.38%** and severely reducing daily tail-risk exposures relative to the benchmark.

---

## ⚙️ Installation & Replication Guide

To replicate these analytics locally, follow this exact environment setup:

### 1. Initialize Your Virtual Environment
```bash
# Clone or open the repository directory
cd portfolio-optimization

# Spin up a localized clean virtual environment
python -m venv .venv

# Activate the localized environment shell
# On Windows:
.\\.venv\\Scripts\\Activate.ps1
# On macOS/Linux:
source .venv/bin/activate
```
### 2. Install Project Dependencies
```bash
# Upgrade core packet tooling
python -m pip install --upgrade pip

# Ingest all specialized engineering packages (yfinance, torch, statsmodels, scipy, pandas, sklearn)
pip install -r requirements.txt
```
### 3. Execute the Full Quantitative Pipeline
```bash
python run_pipeline.py
```
### 🛠️ Git & Branching Strategy
The platform maintains clean separation across development cycles via isolated task streams:

main: Stable production-grade master branch.

task-3: Implements PyTorch deep learning networks.

task-4: Introduces MPT portfolio optimization layers.

task-5: Completes the historical backtesting engine and structural logging.

Developed as part of the GMF Quantitative Research Framework.