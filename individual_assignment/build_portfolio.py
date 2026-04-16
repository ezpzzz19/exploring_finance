"""
build_portfolio.py - Portfolio Construction & Sharpe Optimization
================================================================
Reads predictions from output/predictions.csv (produced by main.py)
and builds the month-by-month portfolio with explicit holdings & weights.

HOW WEIGHTS WORK:
  Think of starting with $1 of capital each month:
    - Long side:  split $1 equally across n_long stocks → each gets $1/n_long
    - Short side: split $1 equally across n_short stocks → each gets -$1/n_short
  This is a dollar-neutral long-short portfolio (zero net market exposure).
  e.g. with 60 longs: each long stock = 1.67% of your capital per month.

MONTH-TO-MONTH TRADING:
  - BUY:  stock enters the portfolio this month (wasn't held last month)
  - HOLD: stock stays in the portfolio from last month (no transaction)
  - SELL: stock exits the portfolio this month (was held last month)
  To minimize transaction fees, prefer configs with low turnover.

Grid searches only over: (60L/10S) and (70L/20S) — three weighting schemes each.
Picks the config with the best turnover-adjusted Sharpe ratio.

Outputs:
  output/monthly_holdings.csv    - every stock, every month: weight, side, action
  output/monthly_trades.csv      - only BUY and SELL actions (what you actually trade)
  output/portfolio_summary.csv   - month-by-month returns
  output/sharpe_grid.csv         - Sharpe for each config tested
  output/cumulative_returns.png  - expanding Sharpe ratio plot
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

# Configs to try: (n_long, n_short)
# Assignment requires 50-100 stocks TOTAL (long + short combined)
GRID_CONFIGS = [(40, 10), (50, 10), (50, 20), (60, 10), (60, 20), (60, 30), (70, 20), (70, 30)]

# Weighting schemes to try
SCHEMES = ["equal", "signal", "rank"]

# Turnover penalty: sharpe_adjusted = sharpe - TURNOVER_PENALTY * avg_monthly_turnover
# e.g. 0.5 means a portfolio with 50% monthly turnover loses 0.25 from its Sharpe score
TURNOVER_PENALTY = 0.5


# ===========================================================================
#  WEIGHTING SCHEMES
# ===========================================================================

def equal_weight(group, model, n_long, n_short):
    """1/n_long for longs, -1/n_short for shorts — simplest and most common."""
    sg = group.sort_values(model, ascending=False)
    top = sg.head(n_long).copy()
    bottom = sg.tail(n_short).copy()
    top["weight"] = 1.0 / n_long
    top["side"] = "long"
    bottom["weight"] = -1.0 / n_short
    bottom["side"] = "short"
    return pd.concat([top, bottom], ignore_index=True)


def signal_weight(group, model, n_long, n_short):
    """Weight proportional to predicted return magnitude."""
    sg = group.sort_values(model, ascending=False)
    top = sg.head(n_long).copy()
    bottom = sg.tail(n_short).copy()

    long_signals = top[model].values - top[model].values.min() + 1e-8
    long_signals /= long_signals.sum()
    top["weight"] = long_signals
    top["side"] = "long"

    short_signals = bottom[model].values.max() - bottom[model].values + 1e-8
    short_signals /= short_signals.sum()
    bottom["weight"] = -short_signals
    bottom["side"] = "short"

    return pd.concat([top, bottom], ignore_index=True)


def rank_weight(group, model, n_long, n_short):
    """Weight proportional to cross-sectional rank."""
    sg = group.sort_values(model, ascending=False)
    top = sg.head(n_long).copy()
    bottom = sg.tail(n_short).copy()

    long_ranks = np.arange(n_long, 0, -1, dtype=float)
    long_ranks /= long_ranks.sum()
    top["weight"] = long_ranks
    top["side"] = "long"

    short_ranks = np.arange(1, n_short + 1, dtype=float)
    short_ranks /= short_ranks.sum()
    bottom["weight"] = -short_ranks
    bottom["side"] = "short"

    return pd.concat([top, bottom], ignore_index=True)


WEIGHT_SCHEMES = {"equal": equal_weight, "signal": signal_weight, "rank": rank_weight}


# ===========================================================================
#  BUILD PORTFOLIO + MONTH-TO-MONTH TRADE ACTIONS
# ===========================================================================

def build_portfolio(pred, model, n_long, n_short, scheme="equal"):
    """
    Build month-by-month portfolio.
    Adds an 'action' column: BUY / HOLD / SELL.
      - BUY:  new position this month (trade required → costs fees)
      - HOLD: carried over from last month (no trade needed)
      - SELL: exiting position this month (trade required → costs fees)

    Returns:
      holdings_df : all active positions each month (BUY + HOLD)
      trades_df   : only the BUY and SELL actions (what you actually execute)
      strat_df    : monthly L/S returns + turnover stats
    """
    ret_var = "stock_exret"
    weight_fn = WEIGHT_SCHEMES[scheme]

    all_holdings = []
    all_exits = []       # stocks that were held last month but not this month
    monthly_returns = []

    prev_long_set = set()
    prev_short_set = set()

    sorted_months = sorted(pred[["year", "month"]].drop_duplicates().itertuples(
        index=False, name=None))

    for (yr, mo) in sorted_months:
        group = pred[(pred["year"] == yr) & (pred["month"] == mo)]
        h = weight_fn(group, model, n_long, n_short)
        h = h[["year", "month", "date", "permno", model,
               ret_var, "weight", "side"]].copy()

        cur_long_set = set(h.loc[h["side"] == "long", "permno"])
        cur_short_set = set(h.loc[h["side"] == "short", "permno"])

        # Assign BUY vs HOLD
        def assign_action(row):
            prev = prev_long_set if row["side"] == "long" else prev_short_set
            return "HOLD" if row["permno"] in prev else "BUY"

        h["action"] = h.apply(assign_action, axis=1)

        # Record SELL actions for stocks that were held last month but exit now
        for permno in prev_long_set - cur_long_set:
            all_exits.append({"year": yr, "month": mo, "permno": permno,
                              "side": "long", "action": "SELL"})
        for permno in prev_short_set - cur_short_set:
            all_exits.append({"year": yr, "month": mo, "permno": permno,
                              "side": "short", "action": "SELL"})

        all_holdings.append(h)

        # Monthly return
        long_h = h[h["side"] == "long"]
        short_h = h[h["side"] == "short"]
        long_ret = (long_h["weight"] * long_h[ret_var]).sum()
        short_ret = (short_h["weight"].abs() * short_h[ret_var]).sum()

        n_buys_long = (h[(h["side"] == "long") & (h["action"] == "BUY")].shape[0])
        n_buys_short = (h[(h["side"] == "short") & (h["action"] == "BUY")].shape[0])
        n_sells_long = len(prev_long_set - cur_long_set)
        n_sells_short = len(prev_short_set - cur_short_set)

        turnover_long = n_buys_long / n_long if n_long > 0 else 0
        turnover_short = n_buys_short / n_short if n_short > 0 else 0

        monthly_returns.append({
            "year": yr, "month": mo,
            "long_ret": long_ret,
            "short_ret": short_ret,
            "ls_ret": long_ret - short_ret,
            "n_long": len(long_h),
            "n_short": len(short_h),
            "buys_long": n_buys_long,
            "buys_short": n_buys_short,
            "sells_long": n_sells_long,
            "sells_short": n_sells_short,
            "turnover_long": turnover_long,
            "turnover_short": turnover_short,
            "total_trades": n_buys_long + n_buys_short + n_sells_long + n_sells_short,
        })

        prev_long_set = cur_long_set
        prev_short_set = cur_short_set

    holdings_df = pd.concat(all_holdings, ignore_index=True)
    exits_df = pd.DataFrame(all_exits) if all_exits else pd.DataFrame(
        columns=["year", "month", "permno", "side", "action"])

    # Combine active holdings (BUY+HOLD) and exits (SELL) into one trades view
    buy_hold = holdings_df[["year", "month", "date", "permno",
                             model, ret_var, "weight", "side", "action"]].copy()
    sell_rows = exits_df.copy()
    if not sell_rows.empty:
        sell_rows["weight"] = 0.0
        sell_rows[model] = np.nan
        sell_rows[ret_var] = np.nan
        sell_rows["date"] = pd.NaT

    trades_df = pd.concat(
        [buy_hold[buy_hold["action"].isin(["BUY", "SELL"])],
         sell_rows[["year", "month", "date", "permno", model,
                    ret_var, "weight", "side", "action"]]
         if not sell_rows.empty else pd.DataFrame()],
        ignore_index=True
    ).sort_values(["year", "month", "side", "action", "permno"])

    strat_df = pd.DataFrame(monthly_returns)
    return holdings_df, trades_df, strat_df


# ===========================================================================
#  COMPUTE METRICS
# ===========================================================================

def compute_metrics(strat, mkt):
    """Compute Sharpe, alpha, drawdown."""
    strat = strat.merge(mkt, on=["year", "month"], how="inner")
    strat["mkt_rf"] = strat["sp_ret"] - strat["rf"]

    T = len(strat)
    if T < 3:
        return {"sharpe": np.nan, "sharpe_adj": np.nan,
                "alpha": np.nan, "alpha_t": np.nan,
                "ir": np.nan, "max_dd": np.nan, "months": T,
                "avg_turnover": np.nan}

    mr = strat["ls_ret"].mean()
    sd = strat["ls_ret"].std()
    sharpe = mr / sd * np.sqrt(12) if sd > 0 else 0.0

    avg_to = strat["turnover_long"].mean() if "turnover_long" in strat.columns else 0.0
    sharpe_adj = sharpe - TURNOVER_PENALTY * avg_to

    nw = smf.ols(formula="ls_ret ~ mkt_rf", data=strat).fit(
        cov_type="HAC", cov_kwds={"maxlags": 6}, use_t=True)
    alpha = nw.params["Intercept"]
    alpha_t = nw.tvalues["Intercept"]
    beta = nw.params["mkt_rf"]
    ir = alpha / np.sqrt(nw.mse_resid) * np.sqrt(12)

    strat["cum_log"] = np.log(strat["ls_ret"] + 1).cumsum()
    peak = strat["cum_log"].cummax()
    max_dd = (peak - strat["cum_log"]).max()

    return {
        "sharpe": sharpe,
        "sharpe_adj": sharpe_adj,
        "mean_ret_monthly": mr,
        "std_ret_monthly": sd,
        "alpha_monthly": alpha,
        "alpha_annual": alpha * 12,
        "alpha_t": alpha_t,
        "beta": beta,
        "ir": ir,
        "max_dd": max_dd,
        "max_1m_loss": strat["ls_ret"].min(),
        "avg_turnover_long": avg_to,
        "months": T,
    }


# ===========================================================================
#  GRID SEARCH
# ===========================================================================

def grid_search_sharpe(pred, mkt, model="ensemble"):
    """
    Try all combinations of GRID_CONFIGS x SCHEMES.
    Rank by turnover-adjusted Sharpe to reward low-churn portfolios.
    """
    print("\n" + "="*60)
    print("GRID SEARCH: Configs to test")
    print("="*60)

    configs = [(nl, ns, sc) for (nl, ns) in GRID_CONFIGS for sc in SCHEMES]
    print(f"  Testing {len(configs)} configs: "
          f"{GRID_CONFIGS} × {SCHEMES}\n")

    results = []
    best_score = -999
    best_config = None

    for nl, ns, scheme in configs:
        print(f"  Testing L{nl}/S{ns} {scheme}...", end=" ", flush=True)
        _, _, strat = build_portfolio(pred, model, nl, ns, scheme)
        metrics = compute_metrics(strat.copy(), mkt)
        sharpe = metrics["sharpe"]
        sharpe_adj = metrics["sharpe_adj"]
        avg_to = metrics.get("avg_turnover_long", np.nan)

        print(f"Sharpe={sharpe:.3f}  "
              f"Adj={sharpe_adj:.3f}  "
              f"Turnover={avg_to*100:.1f}%")

        results.append({
            "n_long": nl, "n_short": ns, "scheme": scheme,
            "sharpe": sharpe,
            "sharpe_adj": sharpe_adj,
            "avg_turnover_long": avg_to,
            "alpha_annual": metrics.get("alpha_annual", np.nan),
            "alpha_t": metrics.get("alpha_t", np.nan),
            "max_dd": metrics.get("max_dd", np.nan),
        })

        if not np.isnan(sharpe_adj) and sharpe_adj > best_score:
            best_score = sharpe_adj
            best_config = (nl, ns, scheme)

    grid_df = pd.DataFrame(results).sort_values("sharpe_adj", ascending=False)

    print(f"\n  All configs tested.")
    print(f"\n  Results (sorted by turnover-adjusted Sharpe):")
    print(grid_df.to_string(index=False))
    print(f"\n  BEST CONFIG: n_long={best_config[0]}, "
          f"n_short={best_config[1]}, scheme={best_config[2]}")
    print(f"  BEST ADJ. SHARPE: {best_score:.3f}")

    return grid_df, best_config


# ===========================================================================
#  FULL EVALUATION WITH BEST CONFIG
# ===========================================================================

def full_evaluation(pred, mkt, model, n_long, n_short, scheme):
    """Run full evaluation, print trade-level details, and save all outputs."""
    print(f"\n{'='*60}")
    print(f"FINAL PORTFOLIO  (n_long={n_long}, n_short={n_short}, scheme={scheme})")
    print(f"  Capital allocation per stock:")
    print(f"    Long  side: $1 / {n_long} = ${1/n_long:.4f} per stock  "
          f"(weight = {100/n_long:.2f}% each)")
    print(f"    Short side: $1 / {n_short} = ${1/n_short:.4f} per stock  "
          f"(weight = {100/n_short:.2f}% each)")
    print(f"{'='*60}")

    ret_var = "stock_exret"
    holdings, trades_df, strat = build_portfolio(pred, model, n_long, n_short, scheme)
    strat_m = strat.merge(mkt, on=["year", "month"], how="inner")
    strat_m["mkt_rf"] = strat_m["sp_ret"] - strat_m["rf"]
    metrics = compute_metrics(strat.copy(), mkt)

    # Print performance
    print(f"\n  Months: {metrics['months']}")
    print(f"  Mean monthly return:  {metrics['mean_ret_monthly']*100:.3f}%")
    print(f"  Std monthly return:   {metrics['std_ret_monthly']*100:.3f}%")
    print(f"  Annualized Sharpe:    {metrics['sharpe']:.3f}")
    print(f"  Adj. Sharpe (after turnover penalty): {metrics['sharpe_adj']:.3f}")
    print(f"\n  CAPM Alpha (monthly): {metrics['alpha_monthly']*100:.3f}%  "
          f"(t={metrics['alpha_t']:.2f})")
    print(f"  CAPM Alpha (annual):  {metrics['alpha_annual']*100:.3f}%")
    print(f"  CAPM Beta:            {metrics['beta']:.3f}")
    print(f"  Information Ratio:    {metrics['ir']:.3f}")
    print(f"\n  Max 1-month loss:     {metrics['max_1m_loss']*100:.3f}%")
    print(f"  Maximum Drawdown:     {metrics['max_dd']*100:.3f}%")

    # Turnover summary
    avg_to_l = strat["turnover_long"].mean()
    avg_to_s = strat["turnover_short"].mean()
    avg_trades = strat["total_trades"].mean()
    print(f"\n  --- Turnover (fraction of portfolio replaced each month) ---")
    print(f"  Long  side avg turnover:  {avg_to_l*100:.1f}%  "
          f"(~{avg_to_l*n_long:.1f} stocks swapped/month)")
    print(f"  Short side avg turnover:  {avg_to_s*100:.1f}%  "
          f"(~{avg_to_s*n_short:.1f} stocks swapped/month)")
    print(f"  Avg total trades/month:   {avg_trades:.1f}  "
          f"(BUY + SELL executions)")

    # CAPM regression
    nw = smf.ols(formula="ls_ret ~ mkt_rf", data=strat_m).fit(
        cov_type="HAC", cov_kwds={"maxlags": 6}, use_t=True)
    print(f"\n{nw.summary()}")

    # --- Print first two months in detail ---
    sorted_months = sorted(strat[["year", "month"]].drop_duplicates().itertuples(
        index=False, name=None))

    for idx, (yr, mo) in enumerate(sorted_months[:2]):
        month_h = holdings[(holdings["year"] == yr) & (holdings["month"] == mo)]
        month_t = trades_df[(trades_df["year"] == yr) & (trades_df["month"] == mo)]
        row = strat[(strat["year"] == yr) & (strat["month"] == mo)].iloc[0]
        date_str = month_h["date"].iloc[0] if len(month_h) > 0 else f"{yr}-{mo:02d}"

        print(f"\n  {'='*50}")
        print(f"  MONTH {idx+1}: {date_str}")
        print(f"  {'='*50}")
        print(f"  Portfolio return this month: {row['ls_ret']*100:+.3f}%  "
              f"(long={row['long_ret']*100:+.3f}%  short={row['short_ret']*100:+.3f}%)")
        print(f"  Trades: {int(row['buys_long'])} BUY long, "
              f"{int(row['sells_long'])} SELL long, "
              f"{int(row['buys_short'])} BUY short, "
              f"{int(row['sells_short'])} SELL short")

        longs = month_h[month_h["side"] == "long"].sort_values("weight", ascending=False)
        shorts = month_h[month_h["side"] == "short"].sort_values("weight")

        print(f"\n  LONG positions ({len(longs)} stocks, {100/n_long:.2f}% each):")
        print(f"  {'permno':>8}  {'action':>6}  {'weight':>8}  {'pred_ret':>10}  {'realized':>10}")
        for _, r in longs.head(10).iterrows():
            print(f"  {int(r['permno']):>8}  {r['action']:>6}  "
                  f"{r['weight']:>8.4f}  {r[model]:>+10.6f}  {r[ret_var]:>+10.6f}")
        if len(longs) > 10:
            print(f"  ... and {len(longs)-10} more long positions")

        print(f"\n  SHORT positions ({len(shorts)} stocks, {100/n_short:.2f}% each):")
        print(f"  {'permno':>8}  {'action':>6}  {'weight':>8}  {'pred_ret':>10}  {'realized':>10}")
        for _, r in shorts.head(10).iterrows():
            print(f"  {int(r['permno']):>8}  {r['action']:>6}  "
                  f"{r['weight']:>8.4f}  {r[model]:>+10.6f}  {r[ret_var]:>+10.6f}")

    # --- Save outputs ---
    os.makedirs("output", exist_ok=True)

    holdings.to_csv("output/monthly_holdings.csv", index=False)
    print(f"\n  Saved: output/monthly_holdings.csv  "
          f"({len(holdings):,} rows — all active positions each month)")
    print(f"    Columns: year, month, date, permno, {model}, "
          f"{ret_var}, weight, side, action")
    print(f"    action = BUY (new this month) | HOLD (carried over) | SELL (exiting)")

    trades_df.to_csv("output/monthly_trades.csv", index=False)
    n_buys = (trades_df["action"] == "BUY").sum()
    n_sells = (trades_df["action"] == "SELL").sum()
    print(f"  Saved: output/monthly_trades.csv  "
          f"({n_buys:,} BUYs + {n_sells:,} SELLs = {n_buys+n_sells:,} total trades)")

    strat_m.to_csv("output/portfolio_summary.csv", index=False)
    print(f"  Saved: output/portfolio_summary.csv")

    # --- Expanding Sharpe plot ---
    strat_m["date_plot"] = pd.to_datetime(
        strat_m["year"].astype(str) + "-" + strat_m["month"].astype(str) + "-01")

    def expanding_sharpe(series):
        out = []
        for i in range(1, len(series) + 1):
            s = series.iloc[:i]
            out.append(s.mean() / s.std() * np.sqrt(12) if i >= 2 else 0.0)
        return out

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(strat_m["date_plot"], expanding_sharpe(strat_m["ls_ret"]),
            label="Long-Short", linewidth=2)
    ax.plot(strat_m["date_plot"], expanding_sharpe(strat_m["long_ret"]),
            label=f"Long Top {n_long}", linewidth=1.5)
    ax.plot(strat_m["date_plot"], expanding_sharpe(strat_m["sp_ret"]),
            label="S&P 500", linewidth=1.5, alpha=0.7)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.5)
    ax.set_title(f"Expanding Annualized Sharpe — {scheme}-weight L{n_long}/S{n_short}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Annualized Sharpe Ratio")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("output/cumulative_returns.png", dpi=150)
    print(f"  Saved: output/cumulative_returns.png")

    return holdings, trades_df, strat_m, metrics


# ===========================================================================
#  MAIN
# ===========================================================================

if __name__ == "__main__":
    print("="*60)
    print("Portfolio Construction & Sharpe Optimization")
    print("="*60)

    pred_path = os.path.join(os.path.dirname(__file__), "output", "predictions.csv")
    if not os.path.exists(pred_path):
        print(f"ERROR: {pred_path} not found. Run main.py first.")
        sys.exit(1)

    pred = pd.read_csv(pred_path, parse_dates=["date"])
    print(f"Loaded {len(pred):,} predictions")
    print(f"  Date range: {pred['date'].min()} to {pred['date'].max()}")

    mkt_path = os.path.join(os.path.dirname(__file__), "data_we", "mkt_ind.csv")
    mkt = pd.read_csv(mkt_path)

    model = "ensemble"

    # Step 1: Grid search
    grid_df, best_config = grid_search_sharpe(pred, mkt, model=model)
    grid_df.to_csv(os.path.join(os.path.dirname(__file__),
                                "output", "sharpe_grid.csv"), index=False)

    # Step 2: Full evaluation with best config
    n_long, n_short, scheme = best_config
    holdings, trades_df, strat, metrics = full_evaluation(
        pred, mkt, model, n_long, n_short, scheme)

    print(f"\n{'='*60}")
    print("Done!")


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

    mkt_path = os.path.join(os.path.dirname(__file__), "data_we", "mkt_ind.csv")
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
