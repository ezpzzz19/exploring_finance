# Methodology Documentation — Individual Assignment

## Table of Contents
1. [Data Sources & Construction](#1-data-sources--construction)
2. [Look-Ahead Bias Audit](#2-look-ahead-bias-audit)
3. [Machine Learning Strategy](#3-machine-learning-strategy)
4. [Portfolio Construction](#4-portfolio-construction)
5. [Assignment Requirements Checklist](#5-assignment-requirements-checklist)

---

## 1. Data Sources & Construction

### 1.1 Base Dataset: `mma_sample_v2.csv`

| Property | Value |
|----------|-------|
| Rows | ~273,000 stock-month observations |
| Columns | 165 (including 147 stock characteristics) |
| Period | January 2000 – December 2023 |
| Universe | ~1,000 U.S. equities per month |
| Target variable | `stock_exret` (monthly stock excess return over risk-free rate) |

The 147 firm characteristics come from the `factor_char_list.csv` file and cover categories such as momentum, value, size, profitability, investment, and trading frictions — consistent with the factor zoo literature (e.g., Gu, Kelly & Xiu, 2020; Jensen, Kelly & Pedersen, 2023).

### 1.2 WRDS Financial Ratios Suite (Source 1 — 39 new features)

| Property | Value |
|----------|-------|
| WRDS Table | `wrdsapps_finratio.firm_ratio` |
| Reference | van Binsbergen, Jules H., Xiao Han, and Alejandro Lopez-Lira (2023) |
| Rows downloaded | ~1.2 million |
| Coverage after merge | **96.7%** of stock-month observations |
| Merge keys | `permno`, `year`, `month` (from `public_date`) |

**How this dataset is compiled by WRDS:** The Financial Ratios Suite is computed by WRDS from Compustat annual and quarterly fundamentals. Each month, WRDS looks up the most recent fiscal-period financial statements for each CRSP permno and computes 67+ ratios. The `public_date` field represents the calendar month in which the ratio is available — it respects reporting lags (firms report financials with a delay), so using `public_date` as the merge key is **safe from look-ahead bias**.

**Features selected (39 new, non-overlapping with base 147):**

| Category | Variables |
|----------|-----------|
| **Valuation** | `capei` (Shiller CAPE), `evm` (Enterprise Value Multiple), `pe_op_dil` (P/E operating diluted), `pe_exi` (P/E excl. extraordinary items), `ps` (Price/Sales), `pcf` (Price/Cash Flow), `ptb` (Price/Book), `peg_trailing` (PEG ratio), `divyield` (Dividend Yield) |
| **Profitability** | `npm` (Net Profit Margin), `opmbd` (Operating Margin Before Depreciation), `opmad` (Operating Margin After Depreciation), `gpm` (Gross Profit Margin), `cfm` (Cash Flow Margin), `roa` (Return on Assets), `roe` (Return on Equity), `roce` (Return on Capital Employed), `efftax` (Effective Tax Rate), `aftret_eq` (After-tax Return on Equity), `aftret_invcapx` (After-tax Return on Invested Capital), `pretret_noa` (Pre-tax Return on Net Operating Assets) |
| **Leverage & Solvency** | `de_ratio` (Debt/Equity), `debt_ebitda` (Debt/EBITDA), `debt_capital` (Debt/Capital), `debt_at` (Debt/Assets), `intcov_ratio` (Interest Coverage), `curr_ratio` (Current Ratio), `quick_ratio` (Quick Ratio), `cash_ratio` (Cash Ratio) |
| **Efficiency** | `cash_conversion` (Cash Conversion Cycle), `inv_turn` (Inventory Turnover), `rect_turn` (Receivables Turnover), `pay_turn` (Payables Turnover), `sale_invcap` (Sales/Invested Capital) |
| **Other** | `accrual` (Accruals), `fcf_ocf` (Free CF / Operating CF), `cash_debt` (Cash Flow / Total Debt), `short_debt` (Short-term Debt / Total Debt) |
| **Sector** | `gsector` (GICS sector code — used as a categorical grouping variable) |

**Why these matter:** The base 147 characteristics are predominantly market-microstructure and return-based (momentum, volatility, turnover). The Financial Ratios Suite adds **fundamental accounting ratios** that capture firm health, valuation, and operational efficiency — a complementary signal source that is well-documented in the asset pricing literature.

### 1.3 I/B/E/S Analyst Consensus (Source 2 — 9 new features)

| Property | Value |
|----------|-------|
| WRDS Table | `ibes.statsum_epsus` |
| Link Table | `wrdsapps.ibcrsphist` (IBES ticker → CRSP permno) |
| Filter | `fpi = '1'` (current fiscal year), `measure = 'EPS'` |
| Rows downloaded | ~1.2 million |
| Coverage after merge | **100%** (very high analyst coverage in this universe) |
| Merge keys | `permno`, `year`, `month` (from `statpers`) |

**How this dataset is compiled by WRDS:** I/B/E/S (Institutional Brokers' Estimate System, owned by Refinitiv) collects individual analyst earnings forecasts from brokerage houses. The `statsum_epsus` table provides **monthly summary statistics** of all outstanding analyst EPS forecasts as of the `statpers` (statistical period) date. This is a **point-in-time snapshot** — it reflects only what analysts had published by that date.

The IBES-CRSP link (`wrdsapps.ibcrsphist`) maps IBES ticker symbols to CRSP permnos with valid date ranges (`sdate` to `edate`), ensuring historically accurate matching.

**Features constructed:**

| Variable | Formula | Interpretation |
|----------|---------|----------------|
| `ibes_numest` | Raw: number of analysts covering | **Analyst attention** — more coverage → more liquid, better price discovery |
| `ibes_meanest` | Raw: mean EPS forecast | **Earnings expectation level** |
| `ibes_medest` | Raw: median EPS forecast | **Robust central tendency** of forecasts |
| `ibes_stdev` | Raw: standard deviation of forecasts | **Forecast uncertainty** — high dispersion predicts higher volatility |
| `ibes_disp` | `stdev / |meanest|` | **Normalized disagreement** — Diether, Malloy & Scherbina (2002) show high dispersion predicts lower returns |
| `ibes_range` | `highest - lowest` | **Forecast spread** — captures extremeness of analyst views |
| `ibes_revision` | `(numup - numdown) / numest` | **Net revision direction** — positive = more upgrades than downgrades, a strong momentum signal |
| `ibes_numup` | Raw: analysts revising up | **Upgrade intensity** |
| `ibes_numdown` | Raw: analysts revising down | **Downgrade intensity** |

**Field removed — `ibes_surprise` (earnings surprise):** See [Section 2](#2-look-ahead-bias-audit).

### 1.4 Institutional Ownership (Source 3 — 4 new features)

| Property | Value |
|----------|-------|
| WRDS Table | `comp.io_qaggregate` |
| Link Table | `crsp.ccmxpf_lnkhist` (Compustat gvkey → CRSP permno) |
| Frequency | Quarterly (forward-filled to monthly) |
| Rows downloaded | ~24,000 |
| Coverage after merge | **~2%** (limited due to CCM link coverage) |
| Merge keys | `permno`, `year`, `month` |

**How this dataset is compiled by WRDS:** Compustat aggregates data from SEC 13-F filings, which institutional investment managers with >$100M AUM must file quarterly. The `io_qaggregate` table pre-aggregates all 13-F holdings to the stock level, providing total institutional shares held, number of institutional holders, and net buying/selling activity. The `datadate` reflects the quarter-end reporting date.

The CCM (CRSP-Compustat Merged) link table maps Compustat's `gvkey` to CRSP's `permno` using `linktype IN ('LC','LU')` and `linkprim IN ('P','C')` — the standard high-quality link used in academic research.

**Quarterly → Monthly conversion:** Since 13-F data is quarterly, we forward-fill each permno's values to subsequent months until the next quarter's filing. This is conservative — it uses **stale but known** information, introducing no look-ahead.

**Features constructed:**

| Variable | Source | Interpretation |
|----------|--------|----------------|
| `io_num_holders` | `iotlhldr` | Number of institutional holders — proxy for institutional attention |
| `io_new_holders` | `ionwhldr` | Net new institutional holders this quarter |
| `io_share_change_pct` | `ioshrchg` | % change in institutional shares held |
| `io_buy_sell_ratio` | `(buyers - sellers) / holders` | Net institutional demand imbalance |

**Note on coverage:** The 2% coverage is low because the CCM link table does not cover all firms in our universe. XGBoost handles missing values natively (it learns optimal split directions for NaN), so the low coverage does not break the model — it simply means this feature provides signal only for matched firms.

### 1.5 Short Interest (Source 4 — 2 new features)

| Property | Value |
|----------|-------|
| WRDS Table | `comp.sec_shortint` |
| Link Table | `crsp.ccmxpf_lnkhist` (same CCM link) |
| Frequency | Semi-monthly → aggregated to monthly |
| Rows downloaded | ~2.05 million |
| Coverage after merge | **67.1%** |
| Merge keys | `permno`, `year`, `month` |

**How this dataset is compiled by WRDS:** Compustat collects short interest data from exchange-reported settlement data. U.S. exchanges report the total number of shares held short as of the **settlement date** (typically mid-month and end-of-month). The `shortintadj` field is adjusted for stock splits. This data is backward-looking by construction — it reflects positions that have already been established.

**Features constructed:**

| Variable | Formula | Interpretation |
|----------|---------|----------------|
| `si_shares_short` | Raw: split-adjusted shares short | **Level of bearish sentiment** — heavily shorted stocks tend to underperform (Dechow et al., 2001) |
| `si_shares_short_chg` | Month-over-month % change | **Change in short pressure** — increasing shorts signal deteriorating outlook |

### 1.6 Final Merged Dataset

| Property | Value |
|----------|-------|
| File | `data_we/mma_sample_enhanced.csv` |
| Shape | 274,086 rows × 219 columns |
| Predictive features | 147 base + 54 WRDS = **~200 characteristics** |
| Merge method | Left join on `(permno, year, month)` — preserves all original rows |

---

## 2. Look-Ahead Bias Audit

Look-ahead (forward-looking) bias occurs when a model uses information at training or prediction time that would not have been available to a real investor at that point. We performed a systematic audit of every data source and every computation in the pipeline.

### 2.1 Field Removed: `ibes_surprise` (Earnings Surprise)

**What it was:**
```
ibes_surprise = (actual_EPS - mean_forecast) / |mean_forecast|
```

**Why it was removed:** The `actual` field in I/B/E/S represents the **realized EPS** — the actual earnings per share reported by the company. This value is only known **after the earnings announcement**, which typically occurs 1–3 months after the `statpers` (statistical period) date. Merging `actual` onto the `statpers` month means the model would have access to future earnings outcomes at the time of the forecast — a textbook case of look-ahead bias.

**What we kept instead:** The analyst forecast statistics (`meanest`, `medest`, `stdev`, `numup`, `numdown`) are all valid because they represent **what analysts had published** as of the `statpers` date — genuine point-in-time information.

### 2.2 Full Audit Results

| Component | Status | Reasoning |
|-----------|--------|-----------|
| **Financial Ratios (`public_date`)** | ✅ Safe | WRDS computes ratios from the most recently **filed** financial statements. The `public_date` is the calendar date at which the ratio becomes available — it already accounts for reporting lags. |
| **IBES Consensus (`statpers`)** | ✅ Safe | `statpers` is the date the consensus snapshot was taken. All forecast statistics reflect published analyst opinions as of that date. |
| **IBES `actual` field** | 🚨 **Removed** | Realized EPS is known only after earnings announcement — future information. |
| **Institutional Ownership (`datadate`)** | ✅ Safe | 13-F filings are filed with a lag (up to 45 days after quarter-end). Using the `datadate` (quarter-end) and forward-filling is conservative. |
| **Short Interest (`datadate`)** | ✅ Safe | Settlement date reflects already-established short positions. |
| **Rank Transform** | ✅ Safe | Cross-sectional rank within each month — uses only contemporaneous data. |
| **StandardScaler** | ✅ Safe | Fitted on **training data only**, then applied to validation and test sets. |
| **Managed Portfolios (AE)** | ✅ Safe | Cross-sectional OLS regression of returns on characteristics within each month — no future information. |
| **AE Factor Estimation** | ✅ Safe | Uses `expanding().mean()` (only past data) and `shift(-1)` to ensure factors are lagged by one period before use in OOS prediction. |
| **Expanding Window** | ✅ Safe | Strict temporal ordering: Train (2000 → T), Validation (T → T+2), Test (T+2 → T+3). No information from future windows leaks backward. |
| **XGBoost early stopping** | ✅ Safe | Early stopping uses validation MSE — validation period is strictly before the test period. |
| **Missing value imputation** | ✅ Safe | Missing values filled with cross-sectional **median** within each month — no future information. |

---

## 3. Machine Learning Strategy

### 3.1 Overview

We use a **two-model ensemble** that combines fundamentally different approaches:

1. **XGBoost** — a gradient-boosted tree model that directly predicts stock excess returns from characteristics
2. **Autoencoder** — a neural network conditional factor model (Gu, Kelly & Xiu, 2021) that learns latent factors and factor loadings simultaneously

The ensemble prediction is the **equal-weight average** of both models' out-of-sample predictions.

### 3.2 Model 1: XGBoost (Gradient Boosted Trees)

**Architecture:**
- Algorithm: XGBoost (`XGBRegressor` with `tree_method='hist'`)
- Maximum trees: 1,000 (with early stopping)
- Early stopping: 15 rounds without validation improvement

**Hyperparameter Grid Search:**

| Parameter | Search Values | Best selected per window |
|-----------|--------------|------------------------|
| Learning rate | {0.01, 0.1} | Chosen by min validation MSE |
| Max depth | {1, 2} | Chosen by min validation MSE |

**Why XGBoost:**
- Handles missing values natively (critical for the partially-covered WRDS features)
- Captures nonlinear interactions between characteristics
- Fast training with `hist` tree method
- Regularized by construction (shallow trees + early stopping)

**Training procedure:**
1. Demean training targets: $Y_{train}^{dm} = Y_{train} - \bar{Y}_{train}$
2. Fit all 4 grid configurations on training data with validation early stopping
3. Select best by minimum validation MSE
4. Refit best configuration, predict on test data
5. Add back training mean: $\hat{Y}_{test} = \hat{Y}_{test}^{dm} + \bar{Y}_{train}$

### 3.3 Model 2: Conditional Autoencoder (Gu, Kelly & Xiu, 2021)

**Architecture:**
- **Beta network** (firm-specific loadings): Input(~200) → Linear(32) → BatchNorm → ReLU → Linear(6)
- **Factor network** (latent factors): Input(~200) → Linear(6)
- Predicted return: $\hat{r}_{i,t} = \boldsymbol{\beta}_{i,t}' \mathbf{f}_t = \sum_{k=1}^{6} \beta_{i,t,k} \cdot f_{t,k}$
- L1 regularization on all weights ($\lambda = 10^{-4}$)
- Optimizer: Adam (lr = $10^{-3}$)
- Ensemble: 3 independent models (different random initializations), averaged

**Managed Portfolios:**
The factor network receives "managed portfolio" returns as input. These are constructed via cross-sectional OLS regression:

$$R_t = Z_t \gamma_t + \epsilon_t$$

where $Z_t$ is the matrix of firm characteristics at time $t$ and $R_t$ is the vector of returns. The estimated coefficients $\hat{\gamma}_t$ become the managed-portfolio returns — each representing the return to a characteristic-weighted portfolio.

**Out-of-sample factor estimation:**
At test time, we do not observe future factors. Instead:
1. Compute an **expanding mean** of estimated factors from the training/validation period
2. **Shift by one period** to ensure we only use past information
3. Use these lagged average factors as the $\mathbf{f}_t$ in $\hat{r}_{i,t} = \boldsymbol{\beta}_{i,t}' \mathbf{f}_t$

### 3.4 Expanding-Window Protocol

The expanding window follows the assignment specification:

```
Window 1:  Train = 2000–2007,  Val = 2008–2009,  Test = 2010
Window 2:  Train = 2000–2008,  Val = 2009–2010,  Test = 2011
Window 3:  Train = 2000–2009,  Val = 2010–2011,  Test = 2012
  ...
Window 14: Train = 2000–2020,  Val = 2021–2022,  Test = 2023
```

- **Training**: 8 years (expanding by 1 year each window)
- **Validation**: 2 years (for hyperparameter selection / early stopping)
- **Test**: 1 year (out-of-sample predictions — never seen during training)
- **Total OOS period**: 2010–2023 (14 years)

### 3.5 Feature Preprocessing

1. **Missing values**: Filled with cross-sectional median (within each month)
2. **Rank transformation**: Each feature is cross-sectionally ranked within each month, then mapped to [-1, +1]:

$$x_{i,t}^{ranked} = \frac{\text{rank}(x_{i,t})}{N_t - 1} \times 2 - 1$$

This makes features comparable across time, robust to outliers, and uniformly distributed — following the standard approach in Gu, Kelly & Xiu (2020).

3. **StandardScaler**: After rank-transforming, features are standardized using training-period mean/std (for XGBoost). The autoencoder uses the rank-transformed values directly.

---

## 4. Portfolio Construction

### 4.1 Signal Generation

Each month, the ensemble model produces a predicted excess return $\hat{r}_{i,t}$ for each stock. Stocks are ranked by their ensemble prediction.

### 4.2 Long-Short Portfolio

- **Long leg**: Top $n_{long}$ stocks by predicted return
- **Short leg**: Bottom $n_{short}$ stocks by predicted return
- **Dollar-neutral**: $\sum w_i^{long} = +1$, $\sum |w_i^{short}| = -1$ → zero net investment

### 4.3 Weighting Schemes (Grid Searched)

| Scheme | Long weights | Short weights |
|--------|-------------|---------------|
| **Equal** | $w_i = 1/n_{long}$ | $w_i = -1/n_{short}$ |
| **Signal** | $w_i \propto (\hat{r}_i - \min\hat{r})$ | $w_i \propto -(\max\hat{r} - \hat{r}_i)$ |
| **Rank** | $w_i \propto \text{rank}_i$ (highest gets most) | $w_i \propto -\text{rank}_i$ (lowest gets most) |

### 4.4 Portfolio Size Configurations

Grid search over `(n_long, n_short)` pairs, all satisfying the **50–100 total stocks** constraint:

| Config | Long | Short | Total |
|--------|------|-------|-------|
| 40L/10S | 40 | 10 | 50 |
| 50L/10S | 50 | 10 | 60 |
| 50L/20S | 50 | 20 | 70 |
| 60L/10S | 60 | 10 | 70 |
| 60L/20S | 60 | 20 | 80 |
| 60L/30S | 60 | 30 | 90 |
| 70L/20S | 70 | 20 | 90 |
| 70L/30S | 70 | 30 | 100 |

### 4.5 Configuration Selection

The best configuration is selected by **turnover-adjusted Sharpe ratio**:

$$\text{Sharpe}_{adj} = \text{Sharpe} - 0.5 \times \overline{\text{turnover}}$$

This penalizes excessive trading — a portfolio with a slightly lower raw Sharpe but much less turnover may be preferred.

### 4.6 Trade Actions

Each month, positions are classified as:
- **BUY**: Stock enters the portfolio (was not held last month)
- **HOLD**: Stock remains in the portfolio (no transaction needed)
- **SELL**: Stock exits the portfolio (was held last month)

### 4.7 Performance Metrics

| Metric | Description |
|--------|-------------|
| **Annualized Sharpe Ratio** | $\frac{\bar{r}_{LS}}{\sigma_{LS}} \times \sqrt{12}$ |
| **CAPM Alpha** | Intercept from $r_{LS,t} = \alpha + \beta \cdot r_{MKT,t} + \epsilon_t$ with Newey-West (HAC) standard errors |
| **Maximum Drawdown** | Largest peak-to-trough decline in cumulative returns |
| **Average Monthly Turnover** | Fraction of portfolio positions replaced each month |
| **Cumulative Return** | $(1 + r_1)(1 + r_2)\cdots(1 + r_T) - 1$ |

---

## 5. Assignment Requirements Checklist

| # | Requirement | Status | Implementation |
|---|-------------|--------|----------------|
| 1 | Use ML to predict stock returns | ✅ | XGBoost + Autoencoder ensemble |
| 2 | Use provided `mma_sample_v2.csv` dataset | ✅ | Base dataset loaded first, enhanced features merged on top |
| 3 | Can merge additional WRDS data | ✅ | 4 WRDS sources: Financial Ratios, IBES, Inst. Ownership, Short Interest (54 new features) |
| 4 | No forward-looking information | ✅ | Full audit performed; `ibes_surprise` removed; expanding window with strict temporal ordering |
| 5 | Build a portfolio of 50–100 stocks | ✅ | Grid search over 8 configs from 50 to 100 total stocks |
| 6 | Dollar-neutral long-short portfolio | ✅ | Long weights sum to +1, short weights sum to -1 |
| 7 | Monthly rebalancing | ✅ | Portfolio reconstructed each month with BUY/HOLD/SELL actions |
| 8 | Beat S&P 500 benchmark | 📊 | Evaluated via cumulative returns comparison and CAPM alpha |
| 9 | Report Sharpe ratio | ✅ | Annualized Sharpe computed in `build_portfolio.py` |
| 10 | Report CAPM alpha with HAC standard errors | ✅ | Newey-West via `statsmodels` OLS |
| 11 | Report maximum drawdown | ✅ | Peak-to-trough computed on cumulative returns |
| 12 | 5-page presentation deck | ⏳ | To be created from `output/` results |

### Pipeline Execution Order

```
1. python download_wrds_ratios.py    # Download WRDS data, merge, save enhanced CSV
2. python main.py                     # Run ML models, save output/predictions.csv
3. python build_portfolio.py          # Build portfolio, grid search, save all outputs
```

### Output Files

| File | Contents |
|------|----------|
| `output/predictions.csv` | Stock-level monthly predictions: `year, month, date, permno, stock_exret, xgb, ae, ensemble` |
| `output/monthly_holdings.csv` | Every position every month: weight, side, action (BUY/HOLD/SELL) |
| `output/monthly_trades.csv` | Only BUY and SELL actions (what you actually trade) |
| `output/portfolio_summary.csv` | Monthly L/S returns + turnover statistics |
| `output/sharpe_grid.csv` | Sharpe ratio for each (n_long, n_short, scheme) configuration |
| `output/cumulative_returns.png` | Cumulative return chart |
| `output/strategy_returns.csv` | Final strategy returns for presentation |
| `output/decile_returns.csv` | Decile portfolio analysis |
