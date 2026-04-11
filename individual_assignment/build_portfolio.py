"""
build_portfolio.py - Portfolio Construction & Sharpe Optimization
================================================================
Reads predictions from output/predictions.csv (produced by main.py)
and builds the month-by-month portfolio with explicit holdings & weights.

Three weighting schemes to maximize Sharpe:
  1. Equal-weight      - 1/N in each long/short position
  2. Signal-weight     - weight proportional to |predicted return|
  3. Rank-weight       - weight proportional to cross-sectional rank

Also grid-searches over n_long / n_short to find the Sharpe-maximizing
portfolio size.

Outputs:
  output/monthly_holdings.csv  - every stock, every month, with weight & side
  output/portfolio_summary.csv - month-by-month returns for best config
  output/sharpe_grid.csv       - Sharpe for each (n_long, n_short, scheme)
  output/cumulative_returns.png - expanding Sharpe ratio plot
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from pandas.tseries.offsets import MonthBegin

warnings.filterwarnings("ignore")

# ===========================================================================
#  WEIGHTING SCHEMES
# ===========================================================================

def equal_weight(group, model, n_long, n_short):
    """Equal weight: 1/n_long for longs, -1/n_short for shorts."""
    sg = group.sort_values(model, ascending=False)
    top = sg.head(n_long).copy()
    bottom = sg.tail(n_short).copy()
    top["weight"] = 1.0 / n_long
    top["side"] = "long"
    bottom["weight"] = -1.0 / n_short
    bottom["side"] = "short"
    return pd.concat([top, bottom], ignore_index=True)


def signal_weight(group, model, n_long, n_short):
    """Weight proportional to |predicted return| signal strength."""
    sg = group.sort_values(model, ascending=False)
    top = sg.head(n_long).copy()
    bottom = sg.tail(n_short).copy()

    # Long weights: proportional to predicted return (higher = more weight)
    long_signals = top[model].values - top[model].values.min() + 1e-8
    long_signals = long_signals / long_signals.sum()
    top["weight"] = long_signals
    top["side"] = "long"

    # Short weights: proportional to how negative the prediction is
    short_signals = bottom[model].values.max() - bottom[model].values + 1e-8
    short_signals = short_signals / short_signals.sum()
    bottom["weight"] = -short_signals
    bottom["side"] = "short"

    return pd.concat([top, bottom], ignore_index=True)


def rank_weight(group, model, n_long, n_short):
    """Weight proportional to rank within the selected stocks."""
    sg = group.sort_values(model, ascending=False)
    top = sg.head(n_long).copy()
    bottom = sg.tail(n_short).copy()

    # Long: rank 1 gets highest weight
    long_ranks = np.arange(n_long, 0, -1, dtype=float)
    long_ranks = long_ranks / long_ranks.sum()
    top["weight"] = long_ranks
    top["side"] = "long"

    # Short: last rank gets highest (most negative) weight
    short_ranks = np.arange(1, n_short + 1, dtype=float)
    short_ranks = short_ranks / short_ranks.sum()
    bottom["weight"] = -short_ranks
    bottom["side"] = "short"

    return pd.concat([top, bottom], ignore_index=True)


WEIGHT_SCHEMES = {
    "equal": equal_weight,
    "signal": signal_weight,
    "rank": rank_weight,
}


# ===========================================================================
#  BUILD PORTFOLIO FOR ONE CONFIG
# ===========================================================================

def build_portfolio(pred, model, n_long, n_short, scheme="equal"):
    """
    Build month-by-month portfolio.
    Returns:
      holdings_df: every stock, every month, with weight, side, realized return
      strat_df:    monthly strategy returns
    """
    ret_var = "stock_exret"
    weight_fn = WEIGHT_SCHEMES[scheme]

    all_holdings = []
    monthly_returns = []

    for (yr, mo), group in pred.groupby(["year", "month"]):
        h = weight_fn(group, model, n_long, n_short)
        h = h[["year", "month", "date", "permno", model,
               ret_var, "weight", "side"]].copy()

        # Weighted return: sum(w_i * r_i) for longs, sum(|w_i| * r_i) for shorts
        long_h = h[h["side"] == "long"]
        short_h = h[h["side"] == "short"]

        long_ret = (long_h["weight"] * long_h[ret_var]).sum()
        short_ret = (short_h["weight"].abs() * short_h[ret_var]).sum()
        ls_ret = long_ret - short_ret

        monthly_returns.append({
            "year": yr, "month": mo,
            "long_ret": long_ret,
            "short_ret": short_ret,
            "ls_ret": ls_ret,
            "n_long": len(long_h),
            "n_short": len(short_h),
        })
        all_holdings.append(h)

    holdings_df = pd.concat(all_holdings, ignore_index=True)
    strat_df = pd.DataFrame(monthly_returns)
    return holdings_df, strat_df


# ===========================================================================
#  COMPUTE METRICS
# ===========================================================================

def compute_metrics(strat, mkt):
    """Compute Sharpe, alpha, drawdown, turnover-ready stats."""
    strat = strat.merge(mkt, on=["year", "month"], how="inner")
    strat["mkt_rf"] = strat["sp_ret"] - strat["rf"]

    T = len(strat)
    if T < 3:
        return {"sharpe": np.nan, "alpha": np.nan, "alpha_t": np.nan,
                "ir": np.nan, "max_dd": np.nan, "months": T}

    mr = strat["ls_ret"].mean()
    sd = strat["ls_ret"].std()
    sharpe = mr / sd * np.sqrt(12) if sd > 0 else 0.0

    nw = smf.ols(formula="ls_ret ~ mkt_rf", data=strat).fit(
        cov_type="HAC", cov_kwds={"maxlags": 6}, use_t=True)
    alpha = nw.params["Intercept"]
    alpha_t = nw.tvalues["Intercept"]
    beta = nw.params["mkt_rf"]
    ir = alpha / np.sqrt(nw.mse_resid) * np.sqrt(12)

    strat["log_ret"] = np.log(strat["ls_ret"] + 1)
    strat["cum_log"] = strat["log_ret"].cumsum()
    peak = strat["cum_log"].cummax()
    max_dd = (peak - strat["cum_log"]).max()

    return {
        "sharpe": sharpe,
        "mean_ret_monthly": mr,
        "std_ret_monthly": sd,
        "alpha_monthly": alpha,
        "alpha_annual": alpha * 12,
        "alpha_t": alpha_t,
        "beta": beta,
        "ir": ir,
        "max_dd": max_dd,
        "max_1m_loss": strat["ls_ret"].min(),
        "months": T,
    }


# ===========================================================================
#  GRID SEARCH FOR OPTIMAL SHARPE
# ===========================================================================

def grid_search_sharpe(pred, mkt, model="ensemble"):
    """
    Search over n_long, n_short, and weighting scheme
    to find the Sharpe-maximizing configuration.
    """
    print("\n" + "="*60)
    print("GRID SEARCH: Maximizing Sharpe Ratio")
    print("="*60)

    # Grid: number of long/short positions
    n_longs = [20, 30, 40, 50, 60, 80, 100]
    n_shorts = [10, 20, 30, 40, 50, 60]
    schemes = ["equal", "signal", "rank"]

    results = []
    best_sharpe = -999
    best_config = None

    total = len(n_longs) * len(n_shorts) * len(schemes)
    i = 0

    for scheme in schemes:
        for nl in n_longs:
            for ns in n_shorts:
                i += 1
                _, strat = build_portfolio(pred, model, nl, ns, scheme)
                metrics = compute_metrics(strat.copy(), mkt)
                sharpe = metrics["sharpe"]

                results.append({
                    "n_long": nl, "n_short": ns, "scheme": scheme,
                    "sharpe": sharpe,
                    "alpha_annual": metrics.get("alpha_annual", np.nan),
                    "alpha_t": metrics.get("alpha_t", np.nan),
                    "max_dd": metrics.get("max_dd", np.nan),
                })

                if not np.isnan(sharpe) and sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_config = (nl, ns, scheme)

                if i % 21 == 0:
                    print(f"  {i}/{total} configs tested...", flush=True)

    grid_df = pd.DataFrame(results)
    print(f"\n  All {total} configs tested.", flush=True)
    print(f"\n  BEST CONFIG: n_long={best_config[0]}, "
          f"n_short={best_config[1]}, scheme={best_config[2]}")
    print(f"  BEST SHARPE: {best_sharpe:.3f}")

    # Show top 10
    top10 = grid_df.nlargest(10, "sharpe")
    print(f"\n  Top 10 configurations:")
    print(top10.to_string(index=False))

    return grid_df, best_config


# ===========================================================================
#  FULL EVALUATION WITH BEST CONFIG
# ===========================================================================

def full_evaluation(pred, mkt, model, n_long, n_short, scheme):
    """Run full evaluation and save all outputs."""
    print(f"\n{'='*60}")
    print(f"FINAL PORTFOLIO  (n_long={n_long}, n_short={n_short}, "
          f"scheme={scheme})")
    print(f"{'='*60}")

    ret_var = "stock_exret"
    holdings, strat = build_portfolio(pred, model, n_long, n_short, scheme)
    strat_m = strat.merge(mkt, on=["year", "month"], how="inner")
    strat_m["mkt_rf"] = strat_m["sp_ret"] - strat_m["rf"]

    metrics = compute_metrics(strat.copy(), mkt)

    # Print results
    print(f"\n  Months: {metrics['months']}")
    print(f"  Mean monthly return:  {metrics['mean_ret_monthly']*100:.3f}%")
    print(f"  Std monthly return:   {metrics['std_ret_monthly']*100:.3f}%")
    print(f"  Annualized Sharpe:    {metrics['sharpe']:.3f}")
    print(f"\n  CAPM Alpha (monthly): {metrics['alpha_monthly']*100:.3f}%  "
          f"(t={metrics['alpha_t']:.2f})")
    print(f"  CAPM Alpha (annual):  {metrics['alpha_annual']*100:.3f}%")
    print(f"  CAPM Beta:            {metrics['beta']:.3f}")
    print(f"  Information Ratio:    {metrics['ir']:.3f}")
    print(f"\n  Max 1-month loss:     {metrics['max_1m_loss']*100:.3f}%")
    print(f"  Maximum Drawdown:     {metrics['max_dd']*100:.3f}%")

    # Turnover
    long_h = holdings[holdings["side"] == "long"]
    short_h = holdings[holdings["side"] == "short"]

    def calc_turnover(df):
        start = df[["permno", "date"]].sort_values(["date", "permno"])
        sc = start.groupby("date")["permno"].count().reset_index()
        end = df[["permno", "date"]].copy()
        end["date"] = end["date"] - MonthBegin(1)
        end = end.sort_values(["date", "permno"])
        remain = start.merge(end, on=["date", "permno"], how="inner")
        rc = remain.groupby("date")["permno"].count().reset_index()
        rc = rc.rename(columns={"permno": "remain"})
        m = sc.merge(rc, on="date", how="inner")
        m["turnover"] = (m["permno"] - m["remain"]) / m["permno"]
        return m["turnover"].mean()

    l_to = calc_turnover(long_h)
    s_to = calc_turnover(short_h)
    print(f"\n  Long turnover:        {l_to*100:.1f}%")
    print(f"  Short turnover:       {s_to*100:.1f}%")

    # Long-only
    print(f"\n  --- Long-only portfolio (top {n_long}) ---")
    lm = strat_m["long_ret"].mean()
    ls_std = strat_m["long_ret"].std()
    print(f"  Mean monthly return:  {lm*100:.3f}%")
    print(f"  Annualized Sharpe:    {lm/ls_std*np.sqrt(12):.3f}")

    # --- CAPM regression ---
    nw = smf.ols(formula="ls_ret ~ mkt_rf", data=strat_m).fit(
        cov_type="HAC", cov_kwds={"maxlags": 6}, use_t=True)
    print(f"\n{nw.summary()}")

    # --- Save outputs ---
    os.makedirs("output", exist_ok=True)

    # 1) Monthly holdings with weights
    holdings.to_csv("output/monthly_holdings.csv", index=False)
    print(f"\n  Holdings saved to output/monthly_holdings.csv")
    print(f"    Columns: year, month, date, permno, {model}, "
          f"{ret_var}, weight, side")

    # 2) Strategy returns
    strat_m.to_csv("output/portfolio_summary.csv", index=False)
    print(f"  Portfolio summary saved to output/portfolio_summary.csv")

    # 3) Print sample holdings for first month
    first_date = holdings["date"].min()
    first_month = holdings[holdings["date"] == first_date]
    longs = first_month[first_month["side"] == "long"].sort_values(
        "weight", ascending=False)
    shorts = first_month[first_month["side"] == "short"].sort_values(
        "weight", ascending=True)

    print(f"\n  --- Sample: {first_date} ---")
    print(f"  Top 5 LONG positions:")
    for _, row in longs.head(5).iterrows():
        print(f"    permno={int(row['permno']):>6d}  "
              f"weight={row['weight']:+.4f}  "
              f"pred={row[model]:+.6f}  "
              f"realized={row[ret_var]:+.6f}")
    print(f"  Top 5 SHORT positions:")
    for _, row in shorts.head(5).iterrows():
        print(f"    permno={int(row['permno']):>6d}  "
              f"weight={row['weight']:+.4f}  "
              f"pred={row[model]:+.6f}  "
              f"realized={row[ret_var]:+.6f}")

    # --- Expanding Sharpe ratio plot ---
    strat_m["date_plot"] = pd.to_datetime(
        strat_m["year"].astype(str) + "-"
        + strat_m["month"].astype(str) + "-01")

    def expanding_sharpe(series):
        sharpes = []
        for i in range(1, len(series) + 1):
            s = series.iloc[:i]
            if i < 2:
                sharpes.append(0.0)
            else:
                sharpes.append(s.mean() / s.std() * np.sqrt(12))
        return sharpes

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(strat_m["date_plot"], expanding_sharpe(strat_m["ls_ret"]),
            label="Long-Short", linewidth=2)
    ax.plot(strat_m["date_plot"], expanding_sharpe(strat_m["long_ret"]),
            label=f"Long Top {n_long}", linewidth=1.5)
    ax.plot(strat_m["date_plot"], expanding_sharpe(strat_m["sp_ret"]),
            label="S&P 500", linewidth=1.5, alpha=0.7)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.5)
    ax.set_title(f"Expanding Annualized Sharpe Ratio - {scheme}-weight "
                 f"(L{n_long}/S{n_short})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Annualized Sharpe Ratio")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("output/cumulative_returns.png", dpi=150)
    print(f"\n  Plot saved to output/cumulative_returns.png")

    return holdings, strat_m, metrics


# ===========================================================================
#  MAIN
# ===========================================================================

if __name__ == "__main__":
    print("="*60)
    print("Portfolio Construction & Sharpe Optimization")
    print("="*60)

    # Load predictions (produced by main.py)
    pred_path = os.path.join(os.path.dirname(__file__), "output", "predictions.csv")
    if not os.path.exists(pred_path):
        print(f"ERROR: {pred_path} not found.")
        print("Run main.py first to generate predictions.")
        sys.exit(1)

    pred = pd.read_csv(pred_path, parse_dates=["date"])
    print(f"Loaded {len(pred):,} predictions")
    print(f"  Models: {[c for c in pred.columns if c not in ['year','month','date','permno','stock_exret']]}")
    print(f"  Date range: {pred['date'].min()} to {pred['date'].max()}")

    mkt_path = os.path.join(os.path.dirname(__file__), "data", "mkt_ind.csv")
    mkt = pd.read_csv(mkt_path)

    model = "ensemble"

    # --- Step 1: Grid search for best Sharpe ---
    grid_df, best_config = grid_search_sharpe(pred, mkt, model=model)
    grid_df.to_csv(os.path.join(os.path.dirname(__file__),
                                "output", "sharpe_grid.csv"), index=False)

    # --- Step 2: Full evaluation with best config ---
    n_long, n_short, scheme = best_config
    holdings, strat, metrics = full_evaluation(
        pred, mkt, model, n_long, n_short, scheme)

    print(f"\n{'='*60}")
    print("Done!")
