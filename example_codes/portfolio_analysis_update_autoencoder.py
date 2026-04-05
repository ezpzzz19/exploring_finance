import os
import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
# import statsmodels.stats.sandwich_covariance as sw
# import statsmodels

########## Bottom-up SP500 Index ##########
# read predcited values
# ret_var, sp_on = 'stock_exret', 1
# pred_path = 'C:/MMA AI in Finance Course MAY 2024/LECTURES/WEEK1/predicted/linear_all_stock_exret_0.csv'
# pred = pd.read_csv(pred_path, parse_dates=['date'])

# # use CRSP monthly file for lagged market cap
# msf_path = 'C:/MMA AI in Finance Course MAY 2024/LECTURES/WEEK1/msf1.csv'
# msf = pd.read_csv(msf_path, parse_dates=['DATE', 'date1', 'ALTPRCDT'])
# msf = msf[['year', 'month', 'PERMNO', 'mve_m']]
# pred_mve = pred.merge(msf, how='inner', on=['year', 'month', 'PERMNO'])

# # calculate bottom-up forecasts for each model each month
# models = list(pred_mve.columns)[5:(pred_mve.shape[1] - 1)]
# monthly = pred_mve.groupby(['year', 'month']).apply(lambda df:
#                                                 pd.Series(np.average(df[models], weights=df['mve_m'], axis = 0), models))
# monthly = monthly.reset_index()

# # merge with real SP500 index return by year month
# sp_path = 'D:/Dropbox/Research/Option Returns ML/Predicted/crsp_sp.csv'
# sp = pd.read_csv(sp_path, parse_dates=['caldt'])
# sp['year'] = sp['caldt'].dt.year
# sp['month'] = sp['caldt'].dt.month
# monthly = monthly.merge(sp, how='inner', on=['year', 'month'])

# # check the correlation between constructed index and real index
# corr = monthly[['stock_exret', 'sprtrn']].corr().values
# print('Correlation:', corr[0, 1])

# # calculate the OOS R-squared for bottom-up forecasts
# ml_models = models[1:]
# yreal = monthly['sprtrn'].values
# for model_name in ml_models:
#     # OOS R2
#     ypred = monthly[model_name].values
#     r2 = 1 - np.sum(np.square((yreal - ypred))) / np.sum(np.square(yreal))
#     print(model_name, r2*100)

########## Machine Learning Portfolio Sorts ##########
# read predcited values
#ret_var, sp_on = 'stock_exret', 0
#pred_path = 'C:/MMA AI in Finance Course MAY 2024/LECTURES/WEEK1/predicted/linear_all_stock_exret_0.csv'
pred_path = os.path.dirname(os.path.abspath(__file__)) + '/auto_pred.csv'
pred = pd.read_csv(pred_path, parse_dates=['date'])
pred["year"] = pred["date"].dt.year
pred["month"] = pred["date"].dt.month
pred = pred.rename(columns={"permno": "PERMNO"})

# use CRSP monthly file for lagged market cap
msf_path = 'C:/MMA AI in Finance Course MAY 2024/LECTURES/WEEK1/msf1.csv'
msf = pd.read_csv(msf_path, parse_dates=['DATE', 'date1', 'ALTPRCDT'])
msf = msf[['year', 'month', 'PERMNO', 'mve_m']]
pred_mve = pred.merge(msf, how='inner', on=['year', 'month', 'PERMNO'])

# select model and variable set
#model = 'en'
#var_set = 'all'
model_var = 'ae'

# sort stocks into deciles each month and calculate portfolio return
predicted = pred_mve.groupby(['year', 'month'])[model_var]
pred_mve['rank'] = np.floor(predicted.transform(lambda s: s.rank()) * 10 / predicted.transform(lambda s: len(s) + 1))
pred_mve = pred_mve.sort_values(['year', 'month', 'rank', 'PERMNO'])
monthly_port = pred_mve.groupby(['year', 'month', 'rank']).apply(lambda df:
                                pd.Series(np.average(df['stock_exret'], weights=df['mve_m'], axis = 0)))
monthly_port = monthly_port.unstack().dropna().reset_index()
monthly_port.columns = ['year', 'month'] + ['port_' + str(x) for x in range (1, 11)]
monthly_port['port_11'] = monthly_port['port_10'] - monthly_port['port_1']

# Calculate the Sharpe ratio for H-L Portfolio
sharpe = monthly_port['port_11'].mean() / monthly_port['port_11'].std() * np.sqrt(12)
print('Sharpe Ratio:', sharpe)

# Calculate the FF5 + Mom Alpha for the H-L Portfolio
ff6_path = 'C:/MMA AI in Finance Course MAY 2024/LECTURES/WEEK1/ff6_monthly.csv'
ff6 = pd.read_csv(ff6_path)
monthly_port = monthly_port.merge(ff6, how='inner', on=['year', 'month'])
nw_ols = sm.ols(formula='port_11 ~ Mkt_RF + SMB + HML + RMW + CMA + UMD', 
                data=monthly_port).fit(cov_type='HAC', cov_kwds={'maxlags': 3}, use_t=True)
print(nw_ols.summary())

# Max one-month loss
max_1m_loss = monthly_port['port_11'].min()
print('Max 1-Month Loss:', max_1m_loss)

# Calculate Drawdown
monthly_port['log_port_11'] = np.log(monthly_port['port_11'] + 1)
monthly_port['cumsum_log_port_11'] = monthly_port['log_port_11'].cumsum(axis=0)
rolling_peak = monthly_port['cumsum_log_port_11'].cummax()
drawdowns = rolling_peak - monthly_port['cumsum_log_port_11']
max_drawdown = drawdowns.max()
print('Maximum Drawdown:', max_drawdown)

# Calculate Turnover
long_short_position = pred_mve[pred_mve['rank'].isin([1, 10])].copy()
sum_mve = long_short_position.groupby(['year', 'month', 'rank'])['mve_m']
long_short_position['start_weight'] = long_short_position['mve_m'] / sum_mve.transform('sum')
long_short_position['start_weight'] = np.where(long_short_position['rank'] == 1,
                                               long_short_position['start_weight'] * -1,
                                               long_short_position['start_weight'])
long_short_position['end_weight'] = long_short_position['start_weight'] * (1 + long_short_position['stock_exret'])
end_weight_sum = long_short_position.groupby(['year', 'month', 'rank'])['end_weight']
long_short_position['end_weight1'] = long_short_position['end_weight'] / end_weight_sum.transform('sum').transform('abs')

start_weight = long_short_position[['date', 'PERMNO', 'start_weight']].copy()
end_weight = long_short_position[['date', 'PERMNO', 'end_weight1']].copy()
end_weight['date'] = end_weight['date'] + pd.DateOffset(months=1)
stock_weight = start_weight.merge(end_weight, how='outer', on=['date', 'PERMNO']).sort_values(['date', 'PERMNO']).fillna(0)
stock_weight['weight_diff'] = np.abs(stock_weight['start_weight'] - stock_weight['end_weight1'])
turnover = stock_weight.groupby(['date'])['weight_diff'].sum().reset_index()
avg_turnover = turnover['weight_diff'].values[1:-1].mean()
print('Turnover:', avg_turnover)
