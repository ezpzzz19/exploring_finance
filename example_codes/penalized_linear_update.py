import datetime
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error

if __name__ == '__main__':
    # for timing purpose
    print(datetime.datetime.now())

    # turn off Setting with Copy Warning
    pd.set_option('mode.chained_assignment', None)

    # set working directory
    work_dir = os.path.dirname(os.path.abspath(__file__))
    print(work_dir)

    # read list of predictors for stocks and options separately
    file_path = work_dir + 'stock_var_list.csv'
    stock_var = list(pd.read_csv(file_path)['variable'].values)
    file_path = work_dir + 'opt_var_list.csv'
    opt_var = list(pd.read_csv(file_path)['variable'].values)
    ind_list = ['ind' + str(i) for i in range(1, 69)]  # 68 industry dummies
    binary_vars = ['convind', 'divi', 'divo', 'rd', 'securedind', 'sin']

    # at this point, define the variables in use
    var_list = stock_var + opt_var  # stock_var, opt_var or stock_var + opt_var

    # loop over different specifications
    # [('dh_ret', 0), ('strad_ret', 0), ('dh_ret', 1), ('strad_ret', 1)]
    # [('stock_exret', 0), ('stock_exret', 1)]
    # [('log_opt_liq', 0), ('log_opt_liq', 1)]
    for ret_var, sp_only in [('stock_exret', 0)]:
        new_set = raw[raw[ret_var].notna()].copy()  # create a copy of the data

        if sp_only == 1:
            new_set = new_set[new_set['sp_ind'] == 1]  # run on SP500 constituents only

        if ret_var == 'strad_ret':
            new_set = new_set.rename(columns={'strad_baspread': 'opt_baspread'})
        else:
            new_set = new_set.rename(columns={'call_baspread': 'opt_baspread'})  # choose the right bid ask spread

        # transform each variable in each month
        monthly = new_set.groupby('date')
        adj_raw = pd.DataFrame()
        for date, monthly_raw in monthly:
            group = monthly_raw.copy()
            # rank transform each variable to [-1, 1]
            for var in var_list:
                var_median = group[var].median(skipna=True)
                group[var] = group[var].fillna(var_median)

                if var in binary_vars:
                    pass
                else:
                    group[var] = group[var].rank(method='dense') - 1
                    group_max = group[var].max()
                    if group_max > 0:
                        group[var] = (group[var] / group_max) * 2 - 1
                    else:
                        group[var] = 0  # in case of all missing values
                        print('Warning:', date, var, 'set to zero.')

            # add the adjusted values
            adj_raw = adj_raw.append(group, ignore_index=True)

        # create interactions with macro variables
        monthly_adj = adj_raw.groupby('date')
        data = pd.DataFrame()
        for date, group in monthly_adj:
            expand = group.copy()
            macro_values = macro[macro['date1'] == date]
            for macro_var in macro_list:
                macro_var_value = macro_values[macro_var].values[0]
                interaction = group[var_list] * macro_var_value
                int_col_names = [var_name + '_' + macro_var for var_name in var_list]
                col_dict = dict(zip(var_list, int_col_names))
                interaction = interaction.rename(columns=col_dict)
                expand = pd.concat([expand, interaction], axis=1)

            # add the expanded variables
            data = data.append(expand, ignore_index=True)

        # initialize for each run
        starting = pd.to_datetime('19960301', format='%Y%m%d')
        counter = 0
        pred_out = pd.DataFrame()
        # r2_out = pd.DataFrame()

        # get the list of all predictors
        full_var_list = var_list + [var_name + '_' + macro_var for macro_var in macro_list for var_name in var_list]
        final_col_names = full_var_list + ind_list

        # estimation with expanding window
        while (starting + pd.DateOffset(years=13 + counter)) <= pd.to_datetime('20200301', format='%Y%m%d'):
            cutoff = [starting, starting + pd.DateOffset(years=10 + counter),
                      starting + pd.DateOffset(years=12 + counter),
                      starting + pd.DateOffset(years=13 + counter)]

            # cut the sample into training, validation, and testing sets
            train = data[(data['date'] >= cutoff[0]) & (data['date'] < cutoff[1])]
            validate = data[(data['date'] >= cutoff[1]) & (data['date'] < cutoff[2])]
            test = data[(data['date'] >= cutoff[2]) & (data['date'] < cutoff[3])]

            # standardize the predictors (excluding industry dummies) for estimation
            scaler = StandardScaler().fit(train[full_var_list])
            train[full_var_list] = scaler.transform(train[full_var_list])
            validate[full_var_list] = scaler.transform(validate[full_var_list])
            test[full_var_list] = scaler.transform(test[full_var_list])

            # get Xs and Ys
            X_train = train[final_col_names].values
            Y_train = train[ret_var].values
            X_val = validate[final_col_names].values
            Y_val = validate[ret_var].values
            X_test = test[final_col_names].values
            Y_test = test[ret_var].values

            # de-mean Y
            Y_mean = np.mean(Y_train)
            Y_train_dm = Y_train - Y_mean

            # prepare output data
            reg_pred = test[['year', 'month', 'date', 'secid', 'PERMNO', ret_var]]

            # Linear Regression
            reg = LinearRegression(fit_intercept=False)
            reg.fit(X_train, Y_train_dm)
            x_pred = reg.predict(X_test) + Y_mean
            reg_pred['ols'] = x_pred

            # Lasso
            lambdas = np.arange(-1, 1.1, 0.1)
            val_mse = np.zeros(len(lambdas))
            for ind, i in enumerate(lambdas):
                reg = Lasso(alpha=(10 ** i), max_iter=100000, fit_intercept=False)
                reg.fit(X_train, Y_train_dm)
                val_mse[ind] = mean_squared_error(Y_val, reg.predict(X_val) + Y_mean)

            best_lambda = lambdas[np.argmin(val_mse)]
            reg = Lasso(alpha=(10 ** best_lambda), max_iter=100000, fit_intercept=False)
            reg.fit(X_train, Y_train_dm)
            x_pred = reg.predict(X_test) + Y_mean
            reg_pred['lasso'] = x_pred

            # Ridge
            lambdas = np.arange(-4, 4.1, 0.1)
            val_mse = np.zeros(len(lambdas))
            for ind, i in enumerate(lambdas):
                reg = Ridge(alpha=((10 ** i) * 0.5), fit_intercept=False)
                reg.fit(X_train, Y_train_dm)
                val_mse[ind] = mean_squared_error(Y_val, reg.predict(X_val) + Y_mean)

            best_lambda = lambdas[np.argmin(val_mse)]
            reg = Ridge(alpha=((10 ** best_lambda) * 0.5), fit_intercept=False)
            reg.fit(X_train, Y_train_dm)
            x_pred = reg.predict(X_test) + Y_mean
            reg_pred['ridge'] = x_pred

            # Elastic Net
            lambdas = np.arange(-2, 2.1, 0.1)
            val_mse = np.zeros(len(lambdas))
            for ind, i in enumerate(lambdas):
                reg = ElasticNet(alpha=(10 ** i), max_iter=100000, fit_intercept=False)
                reg.fit(X_train, Y_train_dm)
                val_mse[ind] = mean_squared_error(Y_val, reg.predict(X_val) + Y_mean)

            best_lambda = lambdas[np.argmin(val_mse)]
            reg = ElasticNet(alpha=(10 ** best_lambda), max_iter=100000, fit_intercept=False)
            reg.fit(X_train, Y_train_dm)
            x_pred = reg.predict(X_test) + Y_mean
            reg_pred['en'] = x_pred

            # add to the output data
            pred_out = pred_out.append(reg_pred, ignore_index=True)

            # go to the next year
            counter += 1

        # output the predicted value and coefficients to csv
        out_path = work_dir + 'predicted/linear_all_' + ret_var + '_' + str(sp_only) + '.csv'
        print(out_path)
        pred_out.to_csv(out_path)

        # print the OOS R2
        yreal = pred_out[ret_var].values
        for model_name in ['ols', 'lasso', 'ridge', 'en']:
            ypred = pred_out[model_name].values
            r2 = 1 - np.sum(np.square((yreal - ypred))) / np.sum(np.square(yreal))
            print(model_name, r2)

        # for timing purpose
        print(datetime.datetime.now())
