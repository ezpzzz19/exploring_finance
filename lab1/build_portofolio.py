import os
import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — saves chart without blocking
import matplotlib.pyplot as plt

# ── paths ─────────────────────────────────────────────────────────────────────
work_dir  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
pred_path = os.path.join(work_dir, 'predicted', 'nn_all_stock_exret_0.csv')
msf_path  = os.path.join(work_dir, 'msf1.csv')
ff6_path  = os.path.join(work_dir, 'ff6_monthly_2023updated.csv')
spy_path  = os.path.join(work_dir, 'SPY returns.xlsx')

ret_var = 'stock_exret'
model   = 'nn2'           # OLS has a clean monotonic decile spread (Sharpe ~1.26 H-L)
                          # RF predictions are inverted: higher predicted = lower realized
                          # GBRT predictions are flat (no cross-sectional spread)
# ── 1. load predictions ───────────────────────────────────────────────────────
pred = pd.read_csv(pred_path, parse_dates=['date'])


# OOS R²
yreal  = pred[ret_var].values
ypred  = pred[model].values
oos_r2 = 1 - np.sum(np.square(yreal - ypred)) / np.sum(np.square(yreal))
print(f'OOS R² ({model}): {oos_r2 * 100:.4f}%')

# ── 2. merge with lagged market cap ───────────────────────────────────────────
msf = pd.read_csv(msf_path, low_memory=False)
msf = msf[['year', 'month', 'PERMNO', 'mve_m']].copy()
msf = msf.rename(columns={'PERMNO': 'permno'})

pred_mve = pred.merge(msf, how='inner', on=['year', 'month', 'permno'])
pred_mve = pred_mve[pred_mve['mve_m'] > 0]

# ── 3. sort stocks into deciles each month (professor's exact approach) ────────
#   rank() / (n+1) * 10  floors to 0–9, with 9 = top decile, 0 = bottom decile
#   Top decile (~50 stocks) = long leg; bottom decile = short leg → H-L portfolio
predicted        = pred_mve.groupby(['year', 'month'])[model]
pred_mve['rank'] = np.floor(
    predicted.transform(lambda s: s.rank()) * 10 /
    predicted.transform(lambda s: len(s) + 1)
)
pred_mve = pred_mve.sort_values(['year', 'month', 'rank', 'permno'])

# value-weighted return for each decile each month
# use pivot_table (not unstack) to avoid multi-level column issues
monthly_raw = (
    pred_mve
    .groupby(['year', 'month', 'rank'])
    .apply(lambda df: np.average(df[ret_var], weights=df['mve_m']))
    .rename('ret')
    .reset_index()
)
monthly_port = monthly_raw.pivot_table(index=['year', 'month'], columns='rank', values='ret')
monthly_port.columns = ['port_' + str(int(x) + 1) for x in monthly_port.columns]
monthly_port = monthly_port.dropna().reset_index()  # only drop months missing a full decile

# High-minus-Low: long top decile, short bottom decile
monthly_port['port_11'] = monthly_port['port_10'] - monthly_port['port_1']

# ── 3b. concentrated long/short portfolios ────────────────────────────────────
#   For each (N_LONG, N_SHORT) config:
#     Long:  top N_LONG stocks, market-cap weighted, notional = 1.0
#     Short: bottom N_SHORT stocks, market-cap weighted,
#            notional = N_SHORT / N_LONG  (proportional hedge ratio)
#   Net return = long_ret - (N_SHORT/N_LONG) * short_ret

CONFIGS = [
    (65, 35),   # 65L / 35S  → 54% short notional
    (60, 10),   # 60L / 10S  → 17% short notional
]

def build_ls_port(grp, n_long, n_short):
    grp = grp.sort_values(model, ascending=False).reset_index(drop=True)
    n   = len(grp)

    long_stocks  = grp.iloc[:n_long].copy()
    short_stocks = grp.iloc[max(0, n - n_short):].copy()

    long_ret   = np.average(long_stocks[ret_var],  weights=long_stocks['mve_m'])
    short_ret  = np.average(short_stocks[ret_var], weights=short_stocks['mve_m'])
    hedge_ratio = n_short / n_long          # proportional to how many stocks you short
    net_ret    = long_ret - hedge_ratio * short_ret

    return pd.Series({
        'long_ret':  long_ret,
        'short_ret': short_ret,
        'net_ret':   net_ret,
    })

ls_ports = {}
for (nl, ns) in CONFIGS:
    key = f'{nl}L{ns}S'
    port = (
        pred_mve.groupby(['year', 'month'])
        .apply(build_ls_port, n_long=nl, n_short=ns)
        .reset_index()
    )
    port = port.merge(monthly_port[['year', 'month']], how='inner', on=['year', 'month'])
    ls_ports[key] = port

# ── 3c. stock-level holdings for the 60L/10S portfolio ───────────────────────
NL_DETAIL, NS_DETAIL = 60, 10

def build_ls_holdings(grp, n_long, n_short):
    grp  = grp.sort_values(model, ascending=False).reset_index(drop=True)
    n    = len(grp)
    hedge_ratio = n_short / n_long

    long_stocks  = grp.iloc[:n_long].copy()
    short_stocks = grp.iloc[max(0, n - n_short):].copy()

    long_w_sum  = long_stocks['mve_m'].sum()
    short_w_sum = short_stocks['mve_m'].sum()

    long_stocks['leg']     = 'long'
    short_stocks['leg']    = 'short'
    long_stocks['weight']  = long_stocks['mve_m'] / long_w_sum
    short_stocks['weight'] = short_stocks['mve_m'] / short_w_sum * hedge_ratio

    # only select columns that are guaranteed inside the sub-group
    keep = ['date', 'permno', model, ret_var, 'mve_m', 'leg', 'weight']
    return pd.concat([long_stocks[keep], short_stocks[keep]], ignore_index=True)

holdings_60_10 = (
    pred_mve.groupby(['year', 'month'], group_keys=False)
    .apply(build_ls_holdings, n_long=NL_DETAIL, n_short=NS_DETAIL)
    .reset_index(drop=True)
)
# re-attach year / month from the date column
holdings_60_10['year']  = holdings_60_10['date'].dt.year
holdings_60_10['month'] = holdings_60_10['date'].dt.month


# ── 4. performance metrics (professor's exact formulas) ───────────────────────

# Sharpe ratio (annualised)
sharpe = monthly_port['port_11'].mean() / monthly_port['port_11'].std() * np.sqrt(12)
print(f'\nSharpe Ratio (H-L): {sharpe:.4f}')

# Max 1-month loss
max_1m_loss = monthly_port['port_11'].min()
print(f'Max 1-Month Loss:   {max_1m_loss * 100:.2f}%')

# Maximum drawdown (log cumulative, professor's formula)
monthly_port['log_port_11']        = np.log(monthly_port['port_11'] + 1)
monthly_port['cumsum_log_port_11'] = monthly_port['log_port_11'].cumsum()
rolling_peak                       = monthly_port['cumsum_log_port_11'].cummax()
drawdowns                          = rolling_peak - monthly_port['cumsum_log_port_11']
max_drawdown                       = drawdowns.max()
print(f'Maximum Drawdown:   {max_drawdown * 100:.2f}%')

# Average return and volatility (annualised)
avg_ret = monthly_port['port_11'].mean() * 12
vol     = monthly_port['port_11'].std() * np.sqrt(12)
print(f'Avg Annual Return:  {avg_ret * 100:.2f}%')
print(f'Annual Volatility:  {vol * 100:.2f}%')

# ── 5. FF6 alpha regression (Newey-West, 3 lags) ──────────────────────────────
ff6 = pd.read_csv(ff6_path)
monthly_port = monthly_port.merge(ff6, how='inner', on=['year', 'month'])

def ff6_alpha(ret_series, data):
    """Return (monthly alpha, t-stat, annualised alpha) via Newey-West HAC OLS."""
    df = data.copy()
    df['_ret'] = ret_series.values
    res = sm.ols(
        formula='_ret ~ Mkt_RF + SMB + HML + RMW + CMA + UMD',
        data=df
    ).fit(cov_type='HAC', cov_kwds={'maxlags': 3}, use_t=True)
    alpha_m  = res.params['Intercept']
    tstat    = res.tvalues['Intercept']
    alpha_a  = alpha_m * 12
    return alpha_m, tstat, alpha_a, res

# H-L portfolio
hl_alpha_m, hl_alpha_t, hl_alpha_a, nw_ols = ff6_alpha(monthly_port['port_11'], monthly_port)
print('\n--- FF6 Alpha Regression (H-L portfolio) ---')
print(nw_ols.summary())
print(f'H-L  Monthly Alpha: {hl_alpha_m*100:.4f}%  |  Annual Alpha: {hl_alpha_a*100:.2f}%  |  t-stat: {hl_alpha_t:.3f}')

# Top decile
top10_alpha_m, top10_alpha_t, top10_alpha_a, _ = ff6_alpha(monthly_port['port_10'], monthly_port)
print(f'Top Decile  Monthly Alpha: {top10_alpha_m*100:.4f}%  |  Annual Alpha: {top10_alpha_a*100:.2f}%  |  t-stat: {top10_alpha_t:.3f}')

# ── 6. turnover (professor's exact approach) ──────────────────────────────────
long_short_position = pred_mve[pred_mve['rank'].isin([0, 9])].copy()
sum_mve = long_short_position.groupby(['year', 'month', 'rank'])['mve_m']
long_short_position['start_weight'] = long_short_position['mve_m'] / sum_mve.transform('sum')
long_short_position['start_weight'] = np.where(
    long_short_position['rank'] == 0,
    long_short_position['start_weight'] * -1,   # short the bottom decile
    long_short_position['start_weight']
)
long_short_position['end_weight'] = (
    long_short_position['start_weight'] * (1 + long_short_position[ret_var])
)
end_weight_sum = long_short_position.groupby(['year', 'month', 'rank'])['end_weight']
long_short_position['end_weight1'] = (
    long_short_position['end_weight'] /
    end_weight_sum.transform('sum').transform('abs')
)

start_weight = long_short_position[['date', 'permno', 'start_weight']].copy()
end_weight   = long_short_position[['date', 'permno', 'end_weight1']].copy()
end_weight['date'] = end_weight['date'] + pd.DateOffset(months=1)

stock_weight = (
    start_weight
    .merge(end_weight, how='outer', on=['date', 'permno'])
    .sort_values(['date', 'permno'])
    .fillna(0)
)
stock_weight['weight_diff'] = np.abs(
    stock_weight['start_weight'] - stock_weight['end_weight1']
)
turnover     = stock_weight.groupby('date')['weight_diff'].sum().reset_index()
avg_turnover = turnover['weight_diff'].values[1:-1].mean()
print(f'\nAverage Monthly Turnover: {avg_turnover:.4f}')

# ── 7. SPY comparison ─────────────────────────────────────────────────────────
spy = pd.read_excel(spy_path, parse_dates=['date'])
spy['year']  = spy['date'].dt.year
spy['month'] = spy['date'].dt.month

monthly_port = monthly_port.merge(spy[['year', 'month', 'SPY_ret']],
                                  how='inner', on=['year', 'month'])
for key in ls_ports:
    ls_ports[key] = ls_ports[key].merge(spy[['year', 'month', 'SPY_ret']],
                                        how='inner', on=['year', 'month'])

spy_sharpe   = monthly_port['SPY_ret'].mean() / monthly_port['SPY_ret'].std() * np.sqrt(12)
spy_avg_ret  = monthly_port['SPY_ret'].mean() * 12
spy_vol      = monthly_port['SPY_ret'].std() * np.sqrt(12)
spy_max_loss = monthly_port['SPY_ret'].min()
spy_log      = np.log(monthly_port['SPY_ret'] + 1).cumsum()
spy_dd       = (spy_log.cummax() - spy_log).max()
_, _, spy_alpha_a, _ = ff6_alpha(monthly_port['SPY_ret'], monthly_port)

top10_sharpe   = monthly_port['port_10'].mean() / monthly_port['port_10'].std() * np.sqrt(12)
top10_avg_ret  = monthly_port['port_10'].mean() * 12
top10_vol      = monthly_port['port_10'].std() * np.sqrt(12)
top10_max_loss = monthly_port['port_10'].min()
top10_log      = np.log(monthly_port['port_10'] + 1).cumsum()
top10_dd       = (top10_log.cummax() - top10_log).max()

# compute metrics for each L/S config
def port_metrics(series, ff6_data):
    sharpe_ = series.mean() / series.std() * np.sqrt(12)
    ret_    = series.mean() * 12
    vol_    = series.std() * np.sqrt(12)
    loss_   = series.min()
    log_    = np.log(series + 1).cumsum()
    dd_     = (log_.cummax() - log_).max()
    _, _, alpha_a, _ = ff6_alpha(series, ff6_data)
    return dict(sharpe=sharpe_, ret=ret_, vol=vol_, loss=loss_, dd=dd_, alpha=alpha_a)

ls_metrics = {}
for key in ls_ports:
    merged = ls_ports[key].merge(monthly_port[['year', 'month', 'Mkt_RF', 'SMB', 'HML', 'RMW', 'CMA', 'UMD']],
                                 how='inner', on=['year', 'month'])
    ls_metrics[key] = port_metrics(merged['net_ret'], merged)

# ── print summary table ───────────────────────────────────────────────────────
col_w   = 12
keys    = list(ls_ports.keys())
n_extra = len(keys)
total_w = 28 + (3 + n_extra) * (col_w + 1)

header = f'{"Metric":<28}' + f'{"H-L Neutral":>{col_w}}' + f'{"Top Decile":>{col_w}}'
for k in keys:
    header += f'{k:>{col_w}}'
header += f'{"SPY":>{col_w}}'
print('\n' + '=' * total_w)
print(header)
print('=' * total_w)

def row(label, hl, top, ls_vals, spy_val, pct=False):
    fmt = lambda v: f'{v*100:>11.2f}%' if pct else f'{v:>12.4f}'
    line = f'{label:<28}' + fmt(hl) + fmt(top)
    for v in ls_vals:
        line += fmt(v)
    line += fmt(spy_val)
    print(line)

row('Sharpe Ratio (ann.)', sharpe,        top10_sharpe,   [ls_metrics[k]['sharpe'] for k in keys], spy_sharpe)
row('Avg Annual Return',   avg_ret,        top10_avg_ret,  [ls_metrics[k]['ret']    for k in keys], spy_avg_ret,  pct=True)
row('FF6 Alpha (ann.)',    hl_alpha_a,     top10_alpha_a,  [ls_metrics[k]['alpha']  for k in keys], spy_alpha_a,  pct=True)
row('Annual Volatility',   vol,            top10_vol,      [ls_metrics[k]['vol']    for k in keys], spy_vol,      pct=True)
row('Max Drawdown',        max_drawdown,   top10_dd,       [ls_metrics[k]['dd']     for k in keys], spy_dd,       pct=True)
row('Max 1-Month Loss',    max_1m_loss,    top10_max_loss, [ls_metrics[k]['loss']   for k in keys], spy_max_loss, pct=True)
print('=' * total_w)
print(f'OOS R² ({model}): {oos_r2*100:.4f}%   |   Avg Monthly Turnover: {avg_turnover:.4f}')
for nl, ns in CONFIGS:
    ratio = ns / nl * 100
    print(f'  {nl}L{ns}S: short notional = {ratio:.0f}% of long notional')

# ── 8. cumulative return chart ────────────────────────────────────────────────
monthly_port = monthly_port.sort_values(['year', 'month']).reset_index(drop=True)
monthly_port['date_plot'] = pd.to_datetime(
    monthly_port[['year', 'month']].assign(day=1)
)

colors  = ['purple', 'orange', 'red', 'brown']
styles  = [':', (0,(3,1,1,1)), '-.', (0,(5,1))]

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(monthly_port['date_plot'],
        np.exp(monthly_port['cumsum_log_port_11']) - 1,
        label=f'H-L Market-Neutral  Sharpe={sharpe:.2f}', linewidth=2)

top_decile_log = np.log(monthly_port['port_10'] + 1).cumsum()
ax.plot(monthly_port['date_plot'],
        np.exp(top_decile_log) - 1,
        label=f'Top Decile Long-Only  Sharpe={top10_sharpe:.2f}, Ret={top10_avg_ret*100:.1f}%',
        linewidth=2, linestyle='-.', color='green')

for i, key in enumerate(keys):
    p   = ls_ports[key].sort_values(['year', 'month']).reset_index(drop=True)
    m   = ls_metrics[key]
    dp  = pd.to_datetime(p[['year', 'month']].assign(day=1))
    ax.plot(dp,
            np.exp(np.log(p['net_ret'] + 1).cumsum()) - 1,
            label=f'{key}  Sharpe={m["sharpe"]:.2f}, Ret={m["ret"]*100:.1f}%',
            linewidth=2, linestyle=styles[i % len(styles)], color=colors[i % len(colors)])

ax.plot(monthly_port['date_plot'],
        np.exp(np.log(monthly_port['SPY_ret'] + 1).cumsum()) - 1,
        label=f'SPY  Sharpe={spy_sharpe:.2f}', linewidth=2, linestyle='--', color='gray')

ax.set_title(f'Cumulative OOS Returns — {model.upper()} Portfolio Comparison')
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative Return')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()

out_chart = os.path.join(work_dir, 'predicted', 'cumulative_returns.png')
plt.savefig(out_chart, dpi=150)
plt.show()
print(f'\nChart saved to {out_chart}')

# ── 9. save 60L/10S stock-level holdings (2020 only) ─────────────────────────
holdings_path = os.path.join(work_dir, 'predicted', 'holdings_60L10S.csv')
holdings_60_10 = (
    holdings_60_10[holdings_60_10['year'] == 2020]
    .sort_values(['month', 'leg', 'weight'], ascending=[True, True, False])
    .reset_index(drop=True)
)
holdings_60_10.to_csv(holdings_path, index=False)
print(f'Holdings saved to {holdings_path}')
print(f'  Columns: {list(holdings_60_10.columns)}')
print(f'  Rows:    {len(holdings_60_10):,}  '
      f'({holdings_60_10["month"].nunique()} months × ~{NL_DETAIL + NS_DETAIL} stocks)')
