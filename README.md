# GMF Investments — Portfolio Optimization Strategy
## Interim Progress Report (Tasks 1 & 2)
**Date:** July 05, 2026  
**Status:** Complete (Initial Baseline Established)

---

### 1. Data Cleaning and Handling Summary
* **Pipeline:** Sourced full transaction history across TSLA, BND, and SPY from January 1, 2015, to June 30, 2026, through the `yfinance` library.
* **Resolution:** Evaluated null indicators; records were forward-filled to address tracking alignments over minor exchange deltas.

### 2. Stationarity Metrics & Analysis
* **Raw Stock Evaluation:** Running the Augmented Dickey-Fuller (ADF) test on TSLA's raw closing values produced a high p-value (>0.05), indicating a non-stationary profile that demands active transformation or differencing ($d \ge 1$).
* **Returns Transformation:** Evaluation of the daily percent changes produced a confident rejection of the null hypothesis (p-value < 0.01), confirming stationarity for return-centric time-series structures.

### 3. Quantitative Financial Risk Metrics
* **TSLA Historical 95% VaR:** Evaluated to observe the statistical floor of asset pullbacks on ordinary trading frames.
* **Sharpe Evaluation:** TSLA displays expected structural return spikes combined with high historical standard deviation baselines, standing in stark contrast to the stable income path exhibited by BND.

### 4. Progress Tracking (Task 2)
* Established strict chronological splits (Train: 2015-2024; Test: 2025-2026) to respect temporal sequences.
* Executed baseline validation parameters via `pmdarima.auto_arima` on training arrays, providing our foundational benchmark for upcoming architectural reviews.

## 📊 Exploratory Data Analysis
Below is the historical baseline price trends and rolling volatility analysis generated during Task 1 preprocessing:

![Exploratory Data Analysis](notebooks/visuals/eda_plots.png)
![Arima Visualization](notebooks/visuals/arima_evaluation.png)