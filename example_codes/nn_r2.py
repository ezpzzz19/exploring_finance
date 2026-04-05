import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
import torch.optim as optim

seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


class NN1(nn.Module):
    def __init__(self, input_dim):
        super(NN1, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        out = self.fc2(x)
        out = torch.squeeze(out)  # get rid of the extra dimension
        return out


class NN2(nn.Module):
    def __init__(self, input_dim):
        super(NN2, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 16)
        self.bn2 = nn.BatchNorm1d(16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        out = self.fc3(x)
        out = torch.squeeze(out)
        return out


class NN3(nn.Module):
    def __init__(self, input_dim):
        super(NN3, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 16)
        self.bn2 = nn.BatchNorm1d(16)
        self.fc3 = nn.Linear(16, 8)
        self.bn3 = nn.BatchNorm1d(8)
        self.fc4 = nn.Linear(8, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        out = self.fc4(x)
        out = torch.squeeze(out)
        return out


class NN4(nn.Module):
    def __init__(self, input_dim):
        super(NN4, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 16)
        self.bn2 = nn.BatchNorm1d(16)
        self.fc3 = nn.Linear(16, 8)
        self.bn3 = nn.BatchNorm1d(8)
        self.fc4 = nn.Linear(8, 4)
        self.bn4 = nn.BatchNorm1d(4)
        self.fc5 = nn.Linear(4, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        out = self.fc5(x)
        out = torch.squeeze(out)
        return out


class NN5(nn.Module):
    def __init__(self, input_dim):
        super(NN5, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 16)
        self.bn2 = nn.BatchNorm1d(16)
        self.fc3 = nn.Linear(16, 8)
        self.bn3 = nn.BatchNorm1d(8)
        self.fc4 = nn.Linear(8, 4)
        self.bn4 = nn.BatchNorm1d(4)
        self.fc5 = nn.Linear(4, 2)
        self.bn5 = nn.BatchNorm1d(2)
        self.fc6 = nn.Linear(2, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        x = F.relu(self.bn5(self.fc5(x)))
        out = self.fc6(x)
        out = torch.squeeze(out)
        return out


def train_nn(model, criterion, train_loader, optimizer, device):
    # activate the training mode
    model.train()
    torch.set_grad_enabled(True)
    total_loss = 0

    # iteration over the mini-batches
    for batch_idx, (predictors, target) in enumerate(train_loader):
        # transfer the data on the chosen device
        predictors, target = predictors.to(device), target.to(device)

        # reinitialize the gradients to zero
        optimizer.zero_grad()

        # forward propagation on the data
        prediction = model(predictors)

        # compute the cost function w.r.t. the targets
        loss = criterion(prediction, target)

        # add l1 regularization to loss function
        l1_norm = torch.norm(model.fc1.weight, p=1)  # only apply l1 to input variables
        loss += 0.0001 * l1_norm  # lambda = 0.0001

        # execute the backpropagation
        loss.backward()

        # execute an optimization step
        optimizer.step()

        # accumulate the loss
        total_loss += loss.item() * len(predictors)

    # compute the average cost per epoch
    mean_loss = total_loss / len(train_loader.dataset)

    return mean_loss


def evaluate_nn(model, criterion, eval_loader, device):
    # activate the evaluation mode
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch_idx, (predictors, target) in enumerate(eval_loader):
            # transfer the data on the chosen device
            predictors, target = predictors.to(device), target.to(device)

            # forward propagation on the data
            prediction = model(predictors)

            # compute the cost function w.r.t. the targets
            loss = criterion(prediction, target)

            # accumulate the loss
            total_loss += loss.item() * len(predictors)

    # compute the average cost per epoch
    mean_loss = total_loss / len(eval_loader.dataset)

    return mean_loss


def learning_loop(ensemble, layers, input_size, output_size, batch_size, num_epoch, train_dataset, val_dataset,
                  patience, X_test_tensor, device):
    mean_pred = np.zeros((output_size, ensemble))
    for i in range(ensemble):
        # accumulators of average costs obtained per epoch
        train_losses = []
        val_losses = []

        # model definition
        if layers == 1:
            model = NN1(input_size)
        elif layers == 2:
            model = NN2(input_size)
        elif layers == 3:
            model = NN3(input_size)
        elif layers == 4:
            model = NN4(input_size)
        elif layers == 5:
            model = NN5(input_size)
        model = model.to(device)

        # loss function
        criterion = nn.MSELoss()

        # optimizer definition
        learning_rate = 0.01
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.955)  # 0.955^100 = 0.01, end lr = 0.0001

        # create mini-batches
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # learning loop
        output_name = 'best_model_r2_' + str(layers) + '_' + str(i) + '.pt'
        min_val_loss = 9999  # for early stopping, record the minimum validation loss
        num_no_improve = 0  # record the number of epochs with no improvements
        for epoch in range(num_epoch):
            # train the model
            train_loss = train_nn(model, criterion, train_loader, optimizer, device)

            # get the validation loss
            val_loss = evaluate_nn(model, criterion, val_loader, device)

            # Save the costs obtained
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # early stopping
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                num_no_improve = 0
                torch.save(model.state_dict(), output_name)  # save the best model
            else:
                num_no_improve += 1
            if num_no_improve >= patience:
                break

            # adapt the learning rate
            scheduler.step()

        model.load_state_dict(torch.load(output_name))  # load the best model
        model.eval()
        prediction = model(X_test_tensor)  # get the OOS prediction
        mean_pred[:, i] = prediction.cpu().data.numpy()

    mean_pred = mean_pred.mean(axis=1)  # compute the mean predictions across ensembles
    return mean_pred


def model_predictions(ensemble, layers, input_size, output_size, X_tensor, device):
    mean_pred = np.zeros((output_size, ensemble))
    for i in range(ensemble):
        if layers == 1:
            model = NN1(input_size)
        elif layers == 2:
            model = NN2(input_size)
        elif layers == 3:
            model = NN3(input_size)
        elif layers == 4:
            model = NN4(input_size)
        elif layers == 5:
            model = NN5(input_size)
        model = model.to(device)
        output_name = 'best_model_r2_' + str(layers) + '_' + str(i) + '.pt'
        model.load_state_dict(torch.load(output_name))  # load the best model
        model.eval()
        prediction = model(X_tensor)
        mean_pred[:, i] = prediction.cpu().data.numpy()

    mean_pred = mean_pred.mean(axis=1)
    return mean_pred


if __name__ == '__main__':
    # setup pytorch
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_gpu else "cpu")
    print("Torch version: ", torch.__version__)
    print("GPU Available: {}".format(use_gpu))

    # for timing purpose
    print(datetime.datetime.now())

    # turn off Setting with Copy Warning
    pd.set_option('mode.chained_assignment', None)

    # set working directory
    # work_dir = 'D:/Dropbox/Research/Option Returns ML/'
    work_dir = '/lustre03/project/6033558/option_ml_may_2021/'

    # read return data
    file_path = work_dir + 'ret_final_v5.csv'
    raw = pd.read_csv(file_path, parse_dates=['date1', 'DATE'], low_memory=False)
    raw = raw.drop(columns=['date1', 'DATE'])
    raw['day'] = 1
    raw['date'] = pd.to_datetime(raw[['year', 'month', 'day']])  # first day of the return month
    raw = raw.drop(columns=['day'])

    # read macro variables
    file_path = work_dir + 'macro_data_goyal.csv'
    macro = pd.read_csv(file_path, parse_dates=['date1'])  # date1 is hard coded to be first day of return month
    macro_list = list(macro.columns)[2:]

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
    for ret_var, sp_only in [('dh_ret', 0), ('strad_ret', 0), ('dh_ret', 1), ('strad_ret', 1)]:
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
        r2_out = pd.DataFrame()

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
            Y_val_dm = Y_val - Y_mean

            # prepare output data
            reg_r2 = pd.DataFrame({'date': [starting + pd.DateOffset(years=13 + counter)]})
            reg_pred = test[['year', 'month', 'date', 'secid', 'PERMNO', ret_var]]

            # load data into tensor datasets
            train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(Y_train_dm).float())
            val_dataset = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(Y_val_dm).float())
            X_test_tensor = torch.from_numpy(X_test).float()
            X_test_tensor = X_test_tensor.to(device)

            # define some general variables for nn
            batch_size = 10000
            num_epoch = 100
            patience = 5
            ensemble = 10
            input_size = len(final_col_names)
            output_size = len(Y_test)

            # train neural networks
            reg_pred['nn1'] = learning_loop(ensemble, 1, input_size, output_size, batch_size, num_epoch, train_dataset,
                                            val_dataset, patience, X_test_tensor, device) + Y_mean  # nn1
            reg_pred['nn2'] = learning_loop(ensemble, 2, input_size, output_size, batch_size, num_epoch, train_dataset,
                                            val_dataset, patience, X_test_tensor, device) + Y_mean  # nn2
            reg_pred['nn3'] = learning_loop(ensemble, 3, input_size, output_size, batch_size, num_epoch, train_dataset,
                                            val_dataset, patience, X_test_tensor, device) + Y_mean  # nn3
            reg_pred['nn4'] = learning_loop(ensemble, 4, input_size, output_size, batch_size, num_epoch, train_dataset,
                                            val_dataset, patience, X_test_tensor, device) + Y_mean  # nn4
            reg_pred['nn5'] = learning_loop(ensemble, 5, input_size, output_size, batch_size, num_epoch, train_dataset,
                                            val_dataset, patience, X_test_tensor, device) + Y_mean  # nn5

            # get the r2
            output_size = len(Y_train_dm)
            for layers in range(1, 6):
                X_train_tensor = torch.from_numpy(X_train).float()
                X_train_tensor = X_train_tensor.to(device)
                reg_r2['nn' + str(layers) + '_base'] = r2_score(Y_train_dm,
                                                                model_predictions(ensemble, layers, input_size, output_size,
                                                                                  X_train_tensor, device))

                for vnd, var_name in enumerate(var_list):
                    new_X_train = X_train.copy()
                    zero_ind = [vnd + len(var_list) * i for i in range(9)]
                    new_X_train[:, zero_ind] = 0
                    X_train_tensor = torch.from_numpy(new_X_train).float()
                    X_train_tensor = X_train_tensor.to(device)
                    reg_r2['nn' + str(layers) + '_' + var_name] = r2_score(Y_train_dm,
                                                                           model_predictions(ensemble, layers, input_size,
                                                                                             output_size,
                                                                                             X_train_tensor, device))

                new_X_train = X_train.copy()
                new_X_train[:, len(full_var_list):] = 0
                X_train_tensor = torch.from_numpy(new_X_train).float()
                X_train_tensor = X_train_tensor.to(device)
                reg_r2['nn' + str(layers) + '_sic2'] = r2_score(Y_train_dm,
                                                                model_predictions(ensemble, layers, input_size, output_size,
                                                                                  X_train_tensor, device))

            # add to the output data
            pred_out = pred_out.append(reg_pred, ignore_index=True)
            r2_out = r2_out.append(reg_r2, ignore_index=True)

            # go to the next year
            counter += 1

        # output the predicted value and coefficients to csv
        out_path = work_dir + 'r2/nn_all_' + ret_var + '_' + str(sp_only) + '.csv'
        print(out_path)
        r2_out.to_csv(out_path)

        out_path = work_dir + 'predicted/nn_all_' + ret_var + '_' + str(sp_only) + '.csv'
        print(out_path)
        pred_out.to_csv(out_path)

        # print the OOS R2
        yreal = pred_out[ret_var].values
        for model_name in ['nn1', 'nn2', 'nn3', 'nn4', 'nn5']:
            ypred = pred_out[model_name].values
            r2 = 1 - np.sum(np.square((yreal - ypred))) / np.sum(np.square(yreal))
            print(model_name, r2)

        # for timing purpose
        print(datetime.datetime.now())
