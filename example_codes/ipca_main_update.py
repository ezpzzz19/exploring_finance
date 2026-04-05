import pandas as pd
import numpy as np
from ipca_classes_update import IPCA_v0, IPCA_v1


if __name__ == '__main__':
    # read original dataset in the paper
    filepath = './kps_sample_2017.csv'
    data = pd.read_csv(filepath, parse_dates=['date'])
    # date needs to be the first day of the return month
    data['date'] = data['date'] + pd.offsets.MonthBegin(-1)
    data = data.dropna()  # require all characteristics to be non-missing
    # not enough cross-section before this date
    data = data[data['date'] >= pd.to_datetime('19640701', format='%Y%m%d')]

    # take the log of lme and at
    data['lme'] = np.log(data['lme'])
    data['at'] = np.log(data['at'])
    vars = list(data.columns[10:].sort_values())

    # calculate the excess return
    rf = pd.read_csv('./rf_1960.csv', parse_dates=['dateff'])
    rf['yy'] = rf['dateff'].dt.year
    rf['mm'] = rf['dateff'].dt.month
    data = data.merge(rf, on=['yy', 'mm'], how='inner')
    data['exret'] = data['ret'] - data['rf']

    # rank transform the data
    monthly = data.groupby('date')
    adj_data = pd.DataFrame()
    for date, monthly_raw in monthly:
        group = monthly_raw.copy()
        # rank transform each variable to [-0.5, 0.5]
        for var in vars:
            group[var] = group[var].rank(method='dense') - 1
            group_max = group[var].max()
            if group_max > 0:
                group[var] = (group[var] / group_max) - 0.5
            else:
                group[var] = 0  # in case of all missing values
                print('Warning:', date, var, 'set to zero.')

        # add the adjusted values
        adj_data = adj_data.append(group, ignore_index=True)

    # prepare the datasets (IPCA_v0 version)
    K = 6
    adj_data = adj_data.sort_values(['yy', 'mm', 'permno'])
    adj_data['constant'] = 1
    dates = adj_data['date'].unique()
    R0 = {date: adj_data.loc[adj_data['date'] == date, 'exret'] for date in dates}
    Z0 = {date: adj_data.loc[adj_data['date'] == date, vars + ['constant']] for date in dates}

    # IPCA_v0: no anomaly
    # ipca_00 = IPCA_v0(Z0, R=R0, K=K)
    # ipca_00.run_ipca(dispIters=False)
    # print(ipca_00.r2)

    # IPCA_v0: with anomaly
    gFac = pd.DataFrame(1, index=sorted(R0.keys()), columns=['anomaly']).T
    # ipca_01 = IPCA_v0(Z0, R=R0, K=K, gFac=gFac)
    # ipca_01.run_ipca(dispIters=False)
    # print(ipca_01.r2)

    # prepare the datasets (IPCA_v1 version)
    new_data = adj_data.set_index(['date', 'permno'])
    new_data = new_data[['exret'] + vars]

    # IPCA_v1: no anomoly
    # ipca_10 = IPCA_v1(new_data, return_column='exret', add_constant=True)
    # results_10 = ipca_10.fit(K=K)
    # results should be identical to above
    # print(results_10['rfits']['R2_Total'], '\n',
    #       results_10['rfits']['R2_Pred'], '\n',
    #       results_10['xfits']['R2_Total'], '\n',
    #       results_10['xfits']['R2_Pred'])

    # IPCA_v1: with anomoly
    # ipca_11 = IPCA_v1(new_data, return_column='exret', add_constant=True)
    # results_11 = ipca_11.fit(K=K, gFac=gFac)
    # results should be identical to above
    # print(results_11['rfits']['R2_Total'], '\n',
    #       results_11['rfits']['R2_Pred'], '\n',
    #       results_11['xfits']['R2_Total'], '\n',
    #       results_11['xfits']['R2_Pred'])

    # OOS can be done with IPCA_v1, e.g.
    ipca_12 = IPCA_v1(new_data, return_column='exret', add_constant=True)
    results_12 = ipca_12.fit(K=K, gFac=gFac, OOS=True, OOS_window='recursive', OOS_window_specs=120)
    print(results_12['rfits']['R2_Total'], '\n',
          results_12['rfits']['R2_Pred'], '\n',
          results_12['xfits']['R2_Total'], '\n',
          results_12['xfits']['R2_Pred'])
    results_12["rfits"]["Fits_Pred"].dropna().reset_index().to_csv('ipca_12_fits_pred.csv')
