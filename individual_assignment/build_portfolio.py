"""
build_portfolio.py
==================
Build the winning portfolio (100 long / 0 short, signal-weighted) from
`output_last_day/predictions.csv` and produce every CSV / PNG needed for
the 5-slide deck.

Inputs
------
- output_last_day/predictions.csv   (year, month, date, permno,
                                     stock_exret, xgb, nn, ae, ensemble)
- data_we/mkt_ind.csv               (year, month, rf, sp_ret)

Outputs  (all written to output_last_day/)
-----------------------------------------
- monthly_holdings.csv        every position, every month (weight / action)
- monthly_trades.csv          BUY / SELL log
- portfolio_summary.csv       monthly strategy / S&P returns
- portfolio_stats.csv         slide-4 performance table
- top10_holdings.csv          slide-2 top-10 holdings (avg weight)
- decile_returns.csv          10-portfolio sort (diagnostic)
- slide_deck/
    cumulative_returns.png    strategy vs S&P (slide 2 / 4)
    annual_returns.png        bar chart (slide 5)
    drawdowns.png             underwater plot (slide 4)
    rolling_12m.png           rolling 1-year returns
    decile_returns.png        ML signal efficacy
    top10_holdings.png        top 10 avg-weight stocks
"""

from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# ────────────────────────────────────────────────────────────────────────────
#  Config
# ────────────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent
PRED_PATH     = ROOT / "output_last_day" / "predictions.csv"
MKT_PATH      = ROOT / "data_we" / "mkt_ind.csv"
OUT_DIR       = ROOT / "output_last_day"
FIG_DIR       = OUT_DIR / "slide_deck"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SIGNAL        = "ensemble"   # xgb / nn / ae / ensemble
N_LONG        = 100
N_SHORT       = 0
WEIGHT_SCHEME = "signal"     # 'signal' (w ∝ max(ŷ,0)) or 'equal'

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
#  1. Load data
# ────────────────────────────────────────────────────────────────────────────
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    pred = pd.read_csv(PRED_PATH, parse_dates=["date"])
    pred["permno"] = pred["permno"].astype(int)
    pred = pred.dropna(subset=[SIGNAL, "stock_exret"])
    pred = pred.sort_values(["year", "month", SIGNAL], ascending=[True, True, False])

    mkt = pd.read_csv(MKT_PATH)[["year", "month", "rf", "sp_ret"]]
    mkt["mkt_rf"] = mkt["sp_ret"] - mkt["rf"]
    return pred, mkt


# ────────────────────────────────────────────────────────────────────────────
#  2. Build monthly holdings (100L / 0S, signal-weighted)
# ────────────────────────────────────────────────────────────────────────────
def build_holdings(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (yr, mo), g in pred.groupby(["year", "month"], sort=True):
        g = g.sort_values(SIGNAL, ascending=False)
        longs = g.head(N_LONG).copy()
        longs["side"] = "long"

        if WEIGHT_SCHEME == "signal":
            s = longs[SIGNAL].clip(lower=0)
            if s.sum() <= 0:
                w = np.full(len(longs), 1.0 / len(longs))
            else:
                w = (s / s.sum()).values
            longs["weight"] = w
        else:  # equal
            longs["weight"] = 1.0 / len(longs)

        rows.append(longs)

    hold = pd.concat(rows, ignore_index=True)
    hold = hold[["year", "month", "date", "permno", SIGNAL,
                 "stock_exret", "weight", "side"]]
    return hold


# ────────────────────────────────────────────────────────────────────────────
#  3. Trade log + turnover
# ────────────────────────────────────────────────────────────────────────────
def build_trades(hold: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    hold = hold.sort_values(["year", "month", "permno"]).reset_index(drop=True)
    months = hold[["year", "month"]].drop_duplicates().sort_values(
        ["year", "month"]).reset_index(drop=True)

    prev_set: set[int] = set()
    trades, turns = [], []
    for _, r in months.iterrows():
        y, m = int(r.year), int(r.month)
        cur = hold[(hold.year == y) & (hold.month == m)]
        cur_set = set(cur["permno"].tolist())

        buys  = cur_set - prev_set
        sells = prev_set - cur_set

        for p in cur["permno"]:
            action = "BUY" if p in buys else "HOLD"
            row = cur[cur.permno == p].iloc[0]
            trades.append({
                "year": y, "month": m, "date": row["date"],
                "permno": p, "weight": row["weight"],
                "side": "long", "action": action,
            })
        for p in sells:
            trades.append({
                "year": y, "month": m, "date": cur["date"].iloc[0],
                "permno": p, "weight": 0.0, "side": "long", "action": "SELL",
            })

        n_long = max(len(cur_set), 1)
        turns.append({"year": y, "month": m,
                      "turnover": len(buys) / n_long,
                      "n_long": len(cur_set),
                      "n_buys": len(buys), "n_sells": len(sells)})
        prev_set = cur_set

    trade_log = pd.DataFrame(trades)
    turn_df   = pd.DataFrame(turns)
    return trade_log, turn_df


# ────────────────────────────────────────────────────────────────────────────
#  4. Monthly portfolio returns + merge market
# ────────────────────────────────────────────────────────────────────────────
def monthly_returns(hold: pd.DataFrame, turn_df: pd.DataFrame,
                    mkt: pd.DataFrame) -> pd.DataFrame:
    grp = hold.groupby(["year", "month"])
    port = grp.apply(lambda d: np.sum(d["weight"] * d["stock_exret"])) \
              .rename("long_ret").reset_index()
    port["short_ret"] = 0.0
    port["strat_ret"] = port["long_ret"]
    port = port.merge(turn_df, on=["year", "month"], how="left")
    port = port.merge(mkt, on=["year", "month"], how="left")
    return port


# ────────────────────────────────────────────────────────────────────────────
#  5. Performance statistics
# ────────────────────────────────────────────────────────────────────────────
def perf_stats(port: pd.DataFrame) -> pd.DataFrame:
    """One-row-per-series stats table: Strategy vs S&P 500."""
    # strategy returns are already excess (stock_exret = ret - rf)
    strat_ex = port["strat_ret"]
    strat_tot = strat_ex + port["rf"]
    sp_tot   = port["sp_ret"]
    sp_ex    = sp_tot - port["rf"]

    def _stats(tot, ex, turnover=None, vs=None):
        ann_ret = tot.mean() * 12
        ann_std = tot.std() * np.sqrt(12)
        sharpe  = ex.mean() / ex.std() * np.sqrt(12) if ex.std() > 0 else np.nan
        # drawdown on log total returns
        cum = np.log1p(tot).cumsum()
        dd = (cum.cummax() - cum).max()
        maxdd = 1 - np.exp(-dd)
        max_1m = tot.min()
        out = {
            "Ann. Return":      ann_ret,
            "Ann. Std Dev":     ann_std,
            "Sharpe (ann.)":    sharpe,
            "Max Drawdown":     maxdd,
            "Max 1-M Loss":     max_1m,
        }
        if turnover is not None:
            out["Monthly Turnover"] = turnover
        if vs is not None:
            active = tot - vs
            out["Info Ratio"] = active.mean() / active.std() * np.sqrt(12) \
                if active.std() > 0 else np.nan
        return out

    strat = _stats(strat_tot, strat_ex,
                   turnover=port["turnover"].mean(), vs=sp_tot)
    sp    = _stats(sp_tot,    sp_ex)

    # CAPM alpha for strategy (NW lag 3)
    reg = smf.ols("strat_ex ~ mkt_rf",
                  data=port.assign(strat_ex=strat_ex)).fit(
        cov_type="HAC", cov_kwds={"maxlags": 3}, use_t=True)
    strat["Ann. Alpha"]  = reg.params["Intercept"] * 12
    strat["Alpha t-stat"] = reg.tvalues["Intercept"]
    strat["CAPM Beta"]   = reg.params["mkt_rf"]

    stats = pd.DataFrame({"Strategy (100L/0S)": strat, "S&P 500": sp})
    order = ["Ann. Return", "Ann. Std Dev", "Sharpe (ann.)", "Ann. Alpha",
             "Alpha t-stat", "CAPM Beta", "Info Ratio", "Max Drawdown",
             "Max 1-M Loss", "Monthly Turnover"]
    return stats.reindex(order)


# ────────────────────────────────────────────────────────────────────────────
#  6. Slide-deck artefacts
# ────────────────────────────────────────────────────────────────────────────
def _dt_index(port: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.to_datetime(dict(year=port["year"], month=port["month"], day=1))


def plot_cumulative(port: pd.DataFrame):
    idx = _dt_index(port)
    strat_tot = port["strat_ret"] + port["rf"]
    sp_tot    = port["sp_ret"]
    cum_s = (1 + strat_tot).cumprod()
    cum_m = (1 + sp_tot).cumprod()

    fig, ax = plt.subplots()
    ax.plot(idx, cum_s, lw=2.2, label=f"Strategy (100L/0S) → ${cum_s.iloc[-1]:.2f}")
    ax.plot(idx, cum_m, lw=2.0, label=f"S&P 500 → ${cum_m.iloc[-1]:.2f}",
            color="#888", linestyle="--")
    ax.set_title("Cumulative Growth of $1  (OOS 2010-01 → 2023-12)")
    ax.set_ylabel("Value of $1")
    ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(FIG_DIR / "cumulative_returns.png"); plt.close(fig)


def plot_annual(port: pd.DataFrame):
    strat_tot = port["strat_ret"] + port["rf"]
    ann = port.assign(strat=strat_tot).groupby("year").agg(
        Strategy=("strat", lambda s: (1 + s).prod() - 1),
        SP500=("sp_ret", lambda s: (1 + s).prod() - 1))

    x = np.arange(len(ann))
    w = 0.4
    fig, ax = plt.subplots()
    ax.bar(x - w/2, ann["Strategy"] * 100, w, label="Strategy", color="#1f77b4")
    ax.bar(x + w/2, ann["SP500"]    * 100, w, label="S&P 500", color="#888")
    ax.set_xticks(x); ax.set_xticklabels(ann.index, rotation=0)
    ax.set_ylabel("Annual Return (%)")
    ax.set_title("Annual Returns — Strategy vs S&P 500")
    ax.axhline(0, color="k", lw=0.8); ax.legend()
    fig.tight_layout(); fig.savefig(FIG_DIR / "annual_returns.png"); plt.close(fig)


def plot_drawdown(port: pd.DataFrame):
    idx = _dt_index(port)
    strat_tot = port["strat_ret"] + port["rf"]
    for label, ret, color in [("Strategy", strat_tot, "#1f77b4"),
                              ("S&P 500", port["sp_ret"], "#888")]:
        cum = np.log1p(ret).cumsum()
        dd  = np.exp(cum - cum.cummax()) - 1
        plt.plot(idx, dd * 100, label=label, color=color, lw=1.8)
    plt.fill_between(idx, 0, 0)  # keep xlim
    plt.title("Drawdowns — Strategy vs S&P 500")
    plt.ylabel("Drawdown (%)"); plt.legend(loc="lower left")
    plt.tight_layout(); plt.savefig(FIG_DIR / "drawdowns.png"); plt.close()


def plot_rolling_12m(port: pd.DataFrame):
    idx = _dt_index(port)
    strat_tot = port["strat_ret"] + port["rf"]
    roll_s = (1 + strat_tot).rolling(12).apply(np.prod) - 1
    roll_m = (1 + port["sp_ret"]).rolling(12).apply(np.prod) - 1
    fig, ax = plt.subplots()
    ax.plot(idx, roll_s * 100, label="Strategy", lw=2)
    ax.plot(idx, roll_m * 100, label="S&P 500", lw=1.8, color="#888", linestyle="--")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title("Rolling 12-Month Returns")
    ax.set_ylabel("12-M Return (%)"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG_DIR / "rolling_12m.png"); plt.close(fig)


def plot_deciles(pred: pd.DataFrame):
    p = pred.copy()
    ranks = p.groupby(["year", "month"])[SIGNAL].transform(
        lambda s: pd.qcut(s.rank(method="first"), 10, labels=False))
    p["decile"] = ranks + 1
    dec = p.groupby("decile")["stock_exret"].mean() * 12 * 100
    dec.to_csv(OUT_DIR / "decile_returns.csv")
    fig, ax = plt.subplots()
    colors = ["#d62728" if i < 5 else "#2ca02c" for i in range(10)]
    ax.bar(dec.index, dec.values, color=colors)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("Predicted-return decile (1=lowest, 10=highest)")
    ax.set_ylabel("Realised annualised excess return (%)")
    ax.set_title(f"Decile sort on {SIGNAL} prediction (OOS 2010–2023)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "decile_returns.png"); plt.close(fig)


def top10_holdings(hold: pd.DataFrame) -> pd.DataFrame:
    # average weight per permno across all months it was held
    n_months = hold[["year", "month"]].drop_duplicates().shape[0]
    agg = (hold.groupby("permno")
                .agg(avg_weight=("weight", "sum"),      # will divide by n_months
                     months_held=("weight", "count"),
                     avg_pred=(SIGNAL, "mean"),
                     avg_realised=("stock_exret", "mean"))
                .assign(avg_weight=lambda d: d["avg_weight"] / n_months)
                .sort_values("avg_weight", ascending=False)
                .head(10)
                .reset_index())
    agg.to_csv(OUT_DIR / "top10_holdings.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(agg["permno"].astype(str)[::-1], agg["avg_weight"][::-1] * 100,
            color="#1f77b4")
    ax.set_xlabel("Average weight in portfolio (%)")
    ax.set_ylabel("PERMNO")
    ax.set_title("Top-10 Holdings (avg weight, OOS 2010–2023)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "top10_holdings.png"); plt.close(fig)
    return agg


# ────────────────────────────────────────────────────────────────────────────
#  7. Main
# ────────────────────────────────────────────────────────────────────────────
def main():
    print("Loading predictions and market data …")
    pred, mkt = load_data()
    print(f"  predictions: {pred.shape}  months: "
          f"{pred[['year', 'month']].drop_duplicates().shape[0]}")

    print("Building 100L / 0S signal-weighted holdings …")
    hold = build_holdings(pred)
    hold.to_csv(OUT_DIR / "monthly_holdings.csv", index=False)

    print("Building trade log and turnover …")
    trades, turns = build_trades(hold)
    trades.to_csv(OUT_DIR / "monthly_trades.csv", index=False)

    print("Computing monthly portfolio returns …")
    port = monthly_returns(hold, turns, mkt)
    port.to_csv(OUT_DIR / "portfolio_summary.csv", index=False)

    print("Computing performance statistics …")
    stats = perf_stats(port)
    stats.to_csv(OUT_DIR / "portfolio_stats.csv")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(stats)

    print("Producing slide-deck figures …")
    plot_cumulative(port)
    plot_annual(port)
    plot_drawdown(port)
    plot_rolling_12m(port)
    plot_deciles(pred)
    top = top10_holdings(hold)
    print("Top-10 holdings preview:")
    print(top.to_string(index=False))

    print(f"\n✓ All outputs written to {OUT_DIR}")
    print(f"✓ Slide figures in       {FIG_DIR}")


if __name__ == "__main__":
    main()
