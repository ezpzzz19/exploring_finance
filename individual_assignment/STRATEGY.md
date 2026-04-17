# Individual Assignment — Strategy Summary

## Winning Configuration

After grid searching **54 configs** (18 long-short/long-only pairs × 3 weighting
schemes), the turnover-adjusted Sharpe-optimal portfolio is:

| Param        | Value                          |
| ------------ | ------------------------------ |
| n_long       | **100**                        |
| n_short      | **0**  (long-only)             |
| Weighting    | **signal** (w ∝ predicted ret) |
| Rebalance    | Monthly                        |
| Universe     | US stocks above NYSE-median cap (~1k/month) |
| OOS window   | 2010-01 → 2023-12 (168 months) |

## Headline Performance vs S&P 500

| Metric                       | Strategy   | S&P 500    |
| ---------------------------- | ---------- | ---------- |
| Avg Annualized Return        | **15.98%** | 11.52%     |
| Annualized Std Dev           | 21.11%     | 14.82%     |
| Annualized Sharpe Ratio      | 0.757      | 0.777      |
| Annualized Alpha (CAPM)      | **+2.45%** | —          |
| Alpha t-stat                 | 0.91       | —          |
| CAPM Beta                    | 1.27       | 1.00       |
| Information Ratio            | 0.254      | —          |
| Maximum Drawdown             | 43.81%     | 28.46%     |
| Maximum 1-Month Loss         | −25.53%    | −12.51%    |
| Monthly Turnover             | 51.8%      | —          |
| Final $1 value (cumulative)  | **$6.76**  | $4.28      |

Win rate vs S&P 500 (yearly): **64%** (9 of 14 years).

## ML Pipeline (what produced `predictions.csv`)

Equal-weight ensemble of **3 models**:

1. **XGBoost** — gradient-boosted trees, depth 1–2, early-stopped, grid over
   learning rate {0.01, 0.1}.
2. **NN2** — feed-forward 32 → 16 MLP with ReLU + dropout 0.1 + L1 = 1e-4,
   3-bag ensemble, early-stopped.
3. **Conditional Autoencoder** (Gu-Kelly-Xiu 2021) — managed-portfolio
   factor model with 6 latent factors, 3-bag ensemble.

### Features (~4,000 per stock-month)

- **Base panel:** `data_we/mma_sample_v2.csv` — 147 firm characteristics
  from the JKP 2022 (Jensen-Kelly-Pedersen) Global Factor dataset, monthly,
  cross-sectionally rank-transformed to `[-1, 1]`.
- **WRDS augmentation (147 → ~200 firm features):** `download_wrds_ratios.py`
  pulls four additional data sources and merges them into
  `data_we/mma_sample_enhanced.csv`, all lagged 2 months to preserve the
  JKP point-in-time convention and avoid look-ahead:
  1. **Financial Ratios Suite** (van Binsbergen et al. 2023,
     `wrdsapps.firm_ratio`) — ~35 valuation / profitability / leverage /
     liquidity / efficiency ratios (e.g. `capei`, `pe_op_dil`, `ptb`,
     `roa`, `roe`, `gpm`, `de_ratio`, `intcov_ratio`, `curr_ratio`,
     `inv_turn`, `accrual`, `fcf_ocf`, …).
  2. **IBES Analyst Consensus** (`ibes.statsum_epsus`) — number of
     estimates, mean/median EPS forecast, dispersion (`ibes_disp`),
     forecast range, and up/down-revision ratio (`ibes_revision`).
     Forward-looking `actual` / `surprise` fields are deliberately
     dropped.
  3. **Institutional Ownership** (`comp.io_qaggregate`) — holder count,
     new-holder count, share-change %, and a buyer/seller imbalance ratio;
     quarterly data forward-filled to monthly.
  4. **Short Interest** (`comp.sec_shortint`) — shares short and MoM
     change in short interest.
- **17 macro predictors** from the Goyal-Welch-Zafirov 2024 archive
  (the "extra data" asked for by the assignment), loaded via
  `download_macro.py` into `data_we/macro_data_goyal_enhanced.csv`.
  Each value is extracted from the diagonal cell of the OOS forecast
  matrix (`[t, t]`), so only information available up to month *t* is
  used — validated by a 6-step no-look-ahead pipeline.
- 17 GWZ × 160 firm characteristics = **2,720 char × macro interactions**
  feeding each model directly, allowing the models to learn conditional
  factor exposures (e.g. "high momentum matters more when dividend yield is
  high").

### Training scheme — rolling expanding windows

- Train 8 years → Validate 2 years → Test 1 year.
- First window trains 2000–2007, validates 2008–2009, tests 2010.
- Each subsequent window rolls both train and val forward one year.
- 14 windows total → 2010–2023 full OOS span (158,980 predictions).

### OOS R² (benchmark = 0, higher = better)

| Model    | OOS R²       |
| -------- | ------------ |
| xgb      | −0.22%       |
| nn       | +0.29%       |
| ae       | +0.54%       |
| **ensemble** | **+0.81%**   |

GKX (2020) benchmark: ≈ 0.40%. Our ensemble roughly **doubles** it.

## Why Long-Only Beat Long-Short

The grid search revealed that **every long-only config beat every long-short
config** on adjusted Sharpe. Root cause:

- 2010–2023 was a secular bull market, so even the ML "bottom decile"
  returned **+9.9% / year** in absolute terms.
- In a dollar-neutral L/S, you're effectively **short a +10%/yr portfolio** —
  that bleed swamps the +2–3%/yr alpha on the short side.
- Removing the short leg recovers the mechanical market drift AND keeps the
  ML's stock-selection alpha on the long side.

## Key Risk Caveats

1. **High beta (1.27).** The long-only top-100 is tilted toward high-
   volatility mid-caps; outperformance is partly market exposure.
2. **Max DD 43.81%** exceeds S&P's 28.46% — concentrated 100-name book.
3. **Alpha t-stat = 0.91** — positive but not statistically significant
   over 168 months; more data would be needed to separate skill from beta.

## Outputs

All deliverables in `output_last_day/`:

- `predictions.csv` — model-level and ensemble predicted returns (168 mo × ~1k stocks)
- `monthly_holdings.csv` — every position, every month, with weight / side / action
- `monthly_trades.csv` — BUY / SELL trade log
- `portfolio_summary.csv` — monthly strategy / long / short / S&P returns
- `sharpe_grid.csv` — full grid-search results
- `cumulative_returns.png` — expanding Sharpe curve
- `oos_r2_overall.csv`, `oos_r2_by_year.csv` — OOS R² tables
- `xgb_feature_importance.csv` — top predictive features

5-slide deck content in `output_last_day/slide_deck/`:

- `slide1_executive_summary.txt`
- `slide2_strategy_description.txt`, `slide2_top10_holdings.csv`
- `slide3_methodology.txt`, `slide3_oos_r2_by_year.png`
- `slide4_performance_table.csv`, `slide4_performance_stats.txt`
- `slide5_discussion.txt`, `slide5_yearly_performance.csv`,
  `slide5_top10_profitable_longs.csv`, `slide5_worst10_longs.csv`
- PPT-ready charts: `ppt_cumulative_growth.png`, `ppt_rolling_12m_returns.png`,
  `ppt_annual_returns.png`, `ppt_drawdowns.png`, `ppt_risk_return.png`,
  `ppt_return_distributions.png`, `decile_returns.png`
