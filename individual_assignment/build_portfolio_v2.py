"""
build_portfolio_v2.py
=====================
Reads `output_last_day/predictions.csv` and produces every CSV / PNG needed
for the 5-slide deck for the WINNING configuration:

    n_long = 100, n_short = 0, weighting = signal (w ∝ max(ŷ,0)),
    rebalance = monthly, OOS = 2010-01 → 2023-12.

Run with:
    uv run python individual_assignment/build_portfolio_v2.py

Inputs
------
- output_last_day/predictions.csv  (year, month, date, permno,
                                    stock_exret, xgb, nn, ae, ensemble)
- data_we/mkt_ind.csv              (year, month, rf, sp_ret)
- data_we/mma_sample_v2.csv        (for permno → ticker / company name)
- output_last_day/oos_r2_by_year.csv        (optional, for slide 3)
- output_last_day/oos_r2_overall.csv        (optional, for slide 3)
- output_last_day/xgb_feature_importance.csv (optional, for slide 5)

Outputs  (all in output_last_day/)
---------------------------------
- monthly_holdings.csv       every position every month (weight / action)
- monthly_trades.csv         BUY / SELL log
- portfolio_summary.csv      monthly strategy + S&P returns + turnover
- portfolio_stats.csv        slide-4 performance table
- top10_holdings.csv         slide-2 top-10 avg holdings (with ticker / name)
- yearly_performance.csv     slide-5 year-by-year strategy vs S&P
- top10_profitable_longs.csv slide-5 best stock-picks
- worst10_longs.csv          slide-5 worst stock-picks
- decile_returns.csv         decile sort on signal (diagnostic)
- slide_deck/
    cumulative_returns.png   strategy vs S&P (slides 1 & 2)
    annual_returns.png       bar chart (slide 5)
    drawdowns.png            underwater plot (slide 4)
    rolling_12m.png          rolling 1-year returns
    decile_returns.png       ML signal efficacy
    top10_holdings.png       top 10 avg-weight stocks (slide 2)
    oos_r2_by_year.png       OOS R² per model per year (slide 3)  [if data]
    feature_importance.png   top-20 XGBoost features (slide 5)    [if data]
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# ────────────────────────────────────────────────────────────────────────────
#  Config
# ────────────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent
PRED_PATH = ROOT / "output_last_day" / "predictions.csv"
MKT_PATH  = ROOT / "data_we" / "mkt_ind.csv"
META_PATH = ROOT / "data_we" / "mma_sample_v2.csv"
R2_YR     = ROOT / "output_last_day" / "oos_r2_by_year.csv"
R2_ALL    = ROOT / "output_last_day" / "oos_r2_overall.csv"
FEAT_PATH = ROOT / "output_last_day" / "xgb_feature_importance.csv"
OUT_DIR   = ROOT / "output_last_day"
FIG_DIR   = OUT_DIR / "slide_deck"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SIGNAL        = "ensemble"
N_LONG        = 100
N_SHORT       = 0
WEIGHT_SCHEME = "signal"   # 'signal' (w ∝ max(ŷ,0)) or 'equal'

plt.rcParams.update({
    "figure.figsize": (10, 5.5),
    "figure.dpi": 110,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})


# ────────────────────────────────────────────────────────────────────────────
#  1. Load
# ────────────────────────────────────────────────────────────────────────────
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred = pd.read_csv(PRED_PATH, parse_dates=["date"])
    pred["permno"] = pred["permno"].astype(int)
    pred = pred.dropna(subset=[SIGNAL, "stock_exret"])
    pred = pred.sort_values(["year", "month", SIGNAL], ascending=[True, True, False])

    mkt = pd.read_csv(MKT_PATH)[["year", "month", "rf", "sp_ret"]]
    mkt["mkt_rf"] = mkt["sp_ret"] - mkt["rf"]

    meta = pd.DataFrame(columns=["permno", "ticker", "name"])
    if META_PATH.exists():
        # read just the columns we need to keep memory low
        tmp = pd.read_csv(META_PATH, usecols=lambda c: c.lower() in
                          {"permno", "stock_ticker", "comp_name"},
                          low_memory=False)
        tmp.columns = [c.lower() for c in tmp.columns]
        tmp = tmp.rename(columns={"stock_ticker": "ticker",
                                  "comp_name": "name"})
        tmp["permno"] = tmp["permno"].astype(int)
        # last observed ticker / name per permno
        meta = (tmp.dropna(subset=["permno"])
                   .groupby("permno").last().reset_index())
    return pred, mkt, meta


# ────────────────────────────────────────────────────────────────────────────
#  2. Build holdings
# ────────────────────────────────────────────────────────────────────────────
def build_holdings(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (yr, mo), g in pred.groupby(["year", "month"], sort=True):
        g = g.sort_values(SIGNAL, ascending=False)
        longs = g.head(N_LONG).copy()
        longs["side"] = "long"
        if WEIGHT_SCHEME == "signal":
            s = longs[SIGNAL].clip(lower=0)
            longs["weight"] = (
                (s / s.sum()).values if s.sum() > 0
                else np.full(len(longs), 1.0 / len(longs))
            )
        else:
            longs["weight"] = 1.0 / len(longs)
        rows.append(longs)

    hold = pd.concat(rows, ignore_index=True)
    return hold[["year", "month", "date", "permno",
                 SIGNAL, "stock_exret", "weight", "side"]]


# ────────────────────────────────────────────────────────────────────────────
#  3. Trades + turnover
# ────────────────────────────────────────────────────────────────────────────
def build_trades(hold: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    hold = hold.sort_values(["year", "month", "permno"]).reset_index(drop=True)
    months = (hold[["year", "month"]].drop_duplicates()
                                       .sort_values(["year", "month"])
                                       .reset_index(drop=True))
    prev: set[int] = set()
    trades, turns = [], []
    for _, r in months.iterrows():
        y, m = int(r.year), int(r.month)
        cur  = hold[(hold.year == y) & (hold.month == m)]
        cset = set(cur["permno"].tolist())
        buys, sells = cset - prev, prev - cset

        for p in cur["permno"]:
            row = cur[cur.permno == p].iloc[0]
            trades.append({"year": y, "month": m, "date": row["date"],
                           "permno": p, "weight": row["weight"],
                           "side": "long",
                           "action": "BUY" if p in buys else "HOLD"})
        for p in sells:
            trades.append({"year": y, "month": m,
                           "date": cur["date"].iloc[0], "permno": p,
                           "weight": 0.0, "side": "long", "action": "SELL"})
        turns.append({"year": y, "month": m,
                      "turnover": len(buys) / max(len(cset), 1),
                      "n_long": len(cset),
                      "n_buys": len(buys), "n_sells": len(sells)})
        prev = cset
    return pd.DataFrame(trades), pd.DataFrame(turns)


# ────────────────────────────────────────────────────────────────────────────
#  4. Monthly returns
# ────────────────────────────────────────────────────────────────────────────
def monthly_returns(hold: pd.DataFrame, turns: pd.DataFrame,
                    mkt: pd.DataFrame) -> pd.DataFrame:
    port = (hold.groupby(["year", "month"])
                 .apply(lambda d: float(np.sum(d["weight"] * d["stock_exret"])))
                 .rename("long_ret").reset_index())
    port["short_ret"] = 0.0
    port["strat_ret"] = port["long_ret"]
    port = port.merge(turns, on=["year", "month"], how="left")
    port = port.merge(mkt,   on=["year", "month"], how="left")
    return port


# ────────────────────────────────────────────────────────────────────────────
#  5. Performance stats (Slide 4)
# ────────────────────────────────────────────────────────────────────────────
def perf_stats(port: pd.DataFrame) -> pd.DataFrame:
    strat_ex  = port["strat_ret"]              # already excess
    strat_tot = strat_ex + port["rf"]
    sp_tot    = port["sp_ret"]
    sp_ex     = sp_tot - port["rf"]

    def _s(tot, ex, turnover=None, vs=None):
        cum = np.log1p(tot).cumsum()
        dd  = (cum.cummax() - cum).max()
        out = {
            "Ann. Return":     tot.mean() * 12,
            "Ann. Std Dev":    tot.std()  * np.sqrt(12),
            "Sharpe (ann.)":   ex.mean() / ex.std() * np.sqrt(12)
                               if ex.std() > 0 else np.nan,
            "Max Drawdown":    1 - np.exp(-dd),
            "Max 1-M Loss":    tot.min(),
        }
        if turnover is not None:
            out["Monthly Turnover"] = turnover
        if vs is not None:
            a = tot - vs
            out["Info Ratio"] = (a.mean() / a.std() * np.sqrt(12)
                                  if a.std() > 0 else np.nan)
        return out

    strat = _s(strat_tot, strat_ex,
               turnover=port["turnover"].mean(), vs=sp_tot)
    sp    = _s(sp_tot,    sp_ex)

    reg = smf.ols("strat_ex ~ mkt_rf",
                  data=port.assign(strat_ex=strat_ex)).fit(
        cov_type="HAC", cov_kwds={"maxlags": 3}, use_t=True)
    strat["Ann. Alpha"]   = reg.params["Intercept"] * 12
    strat["Alpha t-stat"] = reg.tvalues["Intercept"]
    strat["CAPM Beta"]    = reg.params["mkt_rf"]

    order = ["Ann. Return", "Ann. Std Dev", "Sharpe (ann.)", "Ann. Alpha",
             "Alpha t-stat", "CAPM Beta", "Info Ratio", "Max Drawdown",
             "Max 1-M Loss", "Monthly Turnover"]
    return (pd.DataFrame({"Strategy (100L/0S)": strat, "S&P 500": sp})
              .reindex(order))


# ────────────────────────────────────────────────────────────────────────────
#  6. Plots
# ────────────────────────────────────────────────────────────────────────────
def _dt(port): return pd.to_datetime(
    dict(year=port["year"], month=port["month"], day=1))


def plot_cumulative(port: pd.DataFrame):
    idx = _dt(port)
    cum_s = (1 + port["strat_ret"] + port["rf"]).cumprod()
    cum_m = (1 + port["sp_ret"]).cumprod()
    fig, ax = plt.subplots()
    ax.plot(idx, cum_s, lw=2.2,
            label=f"Strategy (100L/0S) → ${cum_s.iloc[-1]:.2f}")
    ax.plot(idx, cum_m, lw=2.0, linestyle="--", color="#888",
            label=f"S&P 500 → ${cum_m.iloc[-1]:.2f}")
    ax.set_title("Cumulative Growth of $1  (OOS 2010-01 → 2023-12)")
    ax.set_ylabel("Value of $1"); ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(FIG_DIR / "cumulative_returns.png")
    plt.close(fig)


def plot_annual(port: pd.DataFrame):
    strat_tot = port["strat_ret"] + port["rf"]
    ann = (port.assign(strat=strat_tot).groupby("year")
               .agg(Strategy=("strat",  lambda s: (1 + s).prod() - 1),
                    SP500=   ("sp_ret", lambda s: (1 + s).prod() - 1)))
    x, w = np.arange(len(ann)), 0.4
    fig, ax = plt.subplots()
    ax.bar(x - w/2, ann["Strategy"] * 100, w, label="Strategy", color="#1f77b4")
    ax.bar(x + w/2, ann["SP500"]    * 100, w, label="S&P 500", color="#888")
    ax.set_xticks(x); ax.set_xticklabels(ann.index, rotation=0)
    ax.set_ylabel("Annual Return (%)")
    ax.set_title("Annual Returns — Strategy vs S&P 500")
    ax.axhline(0, color="k", lw=0.8); ax.legend()
    fig.tight_layout(); fig.savefig(FIG_DIR / "annual_returns.png")
    plt.close(fig)
    return ann


def plot_drawdown(port: pd.DataFrame):
    idx = _dt(port)
    strat_tot = port["strat_ret"] + port["rf"]
    fig, ax = plt.subplots()
    for label, ret, color in [("Strategy", strat_tot, "#1f77b4"),
                              ("S&P 500", port["sp_ret"], "#888")]:
        cum = np.log1p(ret).cumsum()
        dd  = (np.exp(cum - cum.cummax()) - 1) * 100
        ax.plot(idx, dd, label=label, color=color, lw=1.8)
    ax.set_title("Drawdowns — Strategy vs S&P 500")
    ax.set_ylabel("Drawdown (%)"); ax.legend(loc="lower left")
    fig.tight_layout(); fig.savefig(FIG_DIR / "drawdowns.png"); plt.close(fig)


def plot_rolling_12m(port: pd.DataFrame):
    idx = _dt(port)
    strat_tot = port["strat_ret"] + port["rf"]
    rs = (1 + strat_tot).rolling(12).apply(np.prod) - 1
    rm = (1 + port["sp_ret"]).rolling(12).apply(np.prod) - 1
    fig, ax = plt.subplots()
    ax.plot(idx, rs * 100, lw=2, label="Strategy")
    ax.plot(idx, rm * 100, lw=1.8, color="#888", linestyle="--", label="S&P 500")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title("Rolling 12-Month Returns")
    ax.set_ylabel("12-M Return (%)"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG_DIR / "rolling_12m.png"); plt.close(fig)


def plot_deciles(pred: pd.DataFrame):
    p = pred.copy()
    p["decile"] = p.groupby(["year", "month"])[SIGNAL].transform(
        lambda s: pd.qcut(s.rank(method="first"), 10, labels=False)) + 1
    dec = p.groupby("decile")["stock_exret"].mean() * 12 * 100
    dec.to_csv(OUT_DIR / "decile_returns.csv")
    fig, ax = plt.subplots()
    ax.bar(dec.index, dec.values,
           color=["#d62728" if i < 5 else "#2ca02c" for i in range(10)])
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("Predicted-return decile (1 = lowest, 10 = highest)")
    ax.set_ylabel("Realised annualised excess return (%)")
    ax.set_title(f"Decile sort on {SIGNAL} prediction (OOS 2010–2023)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "decile_returns.png")
    plt.close(fig)


def top10_holdings(hold: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    n_months = hold[["year", "month"]].drop_duplicates().shape[0]
    agg = (hold.groupby("permno")
                .agg(avg_weight=("weight", "sum"),
                     months_held=("weight", "count"),
                     avg_pred=(SIGNAL, "mean"),
                     avg_realised=("stock_exret", "mean"))
                .assign(avg_weight=lambda d: d["avg_weight"] / n_months)
                .sort_values("avg_weight", ascending=False)
                .head(10).reset_index())
    if not meta.empty:
        agg = agg.merge(meta, on="permno", how="left")
        cols = ["permno", "ticker", "name", "avg_weight", "months_held",
                "avg_pred", "avg_realised"]
        agg = agg[cols]
    agg.to_csv(OUT_DIR / "top10_holdings.csv", index=False)

    label = (agg["ticker"].fillna(agg["permno"].astype(str))
             if "ticker" in agg.columns else agg["permno"].astype(str))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(label[::-1], agg["avg_weight"][::-1] * 100, color="#1f77b4")
    ax.set_xlabel("Average weight in portfolio (%)")
    ax.set_title("Top-10 Holdings (avg weight, OOS 2010–2023)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "top10_holdings.png")
    plt.close(fig)
    return agg


def slide5_names(hold: pd.DataFrame, meta: pd.DataFrame):
    """Per-stock total P&L contribution = sum(weight * stock_exret) over OOS."""
    contrib = (hold.assign(pnl=hold["weight"] * hold["stock_exret"])
                    .groupby("permno")
                    .agg(total_pnl_contrib=("pnl", "sum"),
                         months_held=("pnl", "count"),
                         avg_weight=("weight", "mean"),
                         avg_realised=("stock_exret", "mean"))
                    .reset_index())
    if not meta.empty:
        contrib = contrib.merge(meta, on="permno", how="left")

    top10  = contrib.nlargest(10, "total_pnl_contrib")
    worst10 = contrib.nsmallest(10, "total_pnl_contrib")
    top10.to_csv(OUT_DIR / "top10_profitable_longs.csv", index=False)
    worst10.to_csv(OUT_DIR / "worst10_longs.csv", index=False)
    return top10, worst10


def yearly_performance(port: pd.DataFrame) -> pd.DataFrame:
    strat_tot = port["strat_ret"] + port["rf"]
    yr = (port.assign(strat=strat_tot)
               .groupby("year")
               .agg(strategy_ret=("strat",  lambda s: (1 + s).prod() - 1),
                    sp500_ret  =("sp_ret", lambda s: (1 + s).prod() - 1),
                    strat_vol  =("strat",  lambda s: s.std() * np.sqrt(12)),
                    months     =("strat",  "count")))
    yr["excess_vs_sp"] = yr["strategy_ret"] - yr["sp500_ret"]
    yr["won"] = (yr["excess_vs_sp"] > 0).astype(int)
    yr.to_csv(OUT_DIR / "yearly_performance.csv")
    return yr


def plot_r2_by_year():
    if not R2_YR.exists():
        print("  (skip) oos_r2_by_year.csv not found")
        return
    df = pd.read_csv(R2_YR)
    fig, ax = plt.subplots()
    colors = {"xgb": "#d62728", "nn": "#2ca02c", "ae": "#ff7f0e",
              "ensemble": "#1f77b4"}
    for m, g in df.groupby("model"):
        g = g.sort_values("year")
        ax.plot(g["year"], g["oos_r2_pct"],
                marker="o", lw=2 if m == "ensemble" else 1.4,
                label=m, color=colors.get(m))
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("OOS R² (%)"); ax.set_xlabel("Year")
    ax.set_title("Out-of-Sample R² by Year (benchmark = 0)")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIG_DIR / "oos_r2_by_year.png")
    plt.close(fig)


def plot_feature_importance():
    if not FEAT_PATH.exists():
        print("  (skip) xgb_feature_importance.csv not found")
        return
    df = pd.read_csv(FEAT_PATH).head(20)[::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(df["feature"], df["importance"], color="#1f77b4")
    ax.set_xlabel("Avg XGBoost importance (across OOS windows)")
    ax.set_title("Top-20 Predictive Features")
    fig.tight_layout(); fig.savefig(FIG_DIR / "feature_importance.png")
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────────────
#  7. Main
# ────────────────────────────────────────────────────────────────────────────
def main():
    print("Loading predictions, market data and metadata …")
    pred, mkt, meta = load_data()
    print(f"  predictions: {pred.shape}  months: "
          f"{pred[['year', 'month']].drop_duplicates().shape[0]}  "
          f"ticker-meta rows: {len(meta)}")

    print("Building 100L / 0S signal-weighted holdings …")
    hold = build_holdings(pred)
    hold.to_csv(OUT_DIR / "monthly_holdings.csv", index=False)

    print("Building trade log and turnover …")
    trades, turns = build_trades(hold)
    trades.to_csv(OUT_DIR / "monthly_trades.csv", index=False)

    print("Computing monthly portfolio returns …")
    port = monthly_returns(hold, turns, mkt)
    port.to_csv(OUT_DIR / "portfolio_summary.csv", index=False)

    print("Performance statistics (slide 4) …")
    stats = perf_stats(port)
    stats.to_csv(OUT_DIR / "portfolio_stats.csv")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(stats)

    print("Slide-2 figures …")
    plot_cumulative(port)
    top = top10_holdings(hold, meta)
    print(top.to_string(index=False))

    print("Slide-3 figures …")
    plot_r2_by_year()

    print("Slide-4 figures …")
    plot_drawdown(port)
    plot_rolling_12m(port)

    print("Slide-5 figures and tables …")
    plot_annual(port)
    plot_deciles(pred)
    plot_feature_importance()
    yr = yearly_performance(port)
    print(yr.to_string())
    top10, worst10 = slide5_names(hold, meta)
    print("\nMost profitable longs (total P&L contribution):")
    print(top10.to_string(index=False))
    print("\nWorst longs:")
    print(worst10.to_string(index=False))

    print(f"\n✓ All outputs written to {OUT_DIR}")
    print(f"✓ Slide figures in       {FIG_DIR}")


if __name__ == "__main__":
    main()
