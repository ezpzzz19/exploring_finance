import numpy as np
import os
from sklearn.linear_model import LinearRegression
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset
import datetime
import gc  # garbage collect to reduce memory use
import pandas as pd

# setup pytorch
use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if use_gpu else "cpu")
print("Torch version: ", torch.__version__)
print("GPU Available: {}".format(use_gpu))

seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


class Autoencoder(nn.Module):
    def __init__(self, input_dim, factor_dim):
        super(Autoencoder, self).__init__()

        # Beta side (1 hidden layer as an example)
        self.beta_fc1 = nn.Linear(input_dim, 32, bias=False)
        self.beta_bn1 = nn.BatchNorm1d(32)
        self.beta_fc2 = nn.Linear(32, factor_dim, bias=False)

        # Factor side
        self.factor_fc1 = nn.Linear(input_dim, factor_dim, bias=False)

    def forward(self, firm_chars, portfolios, oos=False):
        # Beta side
        x = F.relu(self.beta_bn1(self.beta_fc1(firm_chars)))
        x = self.beta_fc2(x)
        x = torch.squeeze(x)

        # Factor side
        if not oos:
            y = self.factor_fc1(portfolios)
            y = torch.squeeze(y)
        else:
            y = portfolios

        # Dot product
        out = torch.sum(x * y, dim=1)

        return x, y, out


def train_ae(model, criterion, train_loader, optimizer, l1_lambda, device):
    # activate the training mode
    model.train()
    torch.set_grad_enabled(True)
    total_loss = 0

    # iteration over the mini-batches
    for batch_idx, (firm_chars, portfolios, target) in enumerate(train_loader):
        # transfer the data on the chosen device
        firm_chars, portfolios, target = (
            firm_chars.to(device),
            portfolios.to(device),
            target.to(device),
        )

        # reinitialize the gradients to zero
        optimizer.zero_grad()

        # forward propagation on the data
        beta, factor, prediction = model(firm_chars, portfolios)

        # compute the cost function w.r.t. the targets
        loss = criterion(prediction, target)

        # add l1 regularization to loss function
        l1_norm = (
            torch.linalg.norm(model.beta_fc1.weight, ord=1)
            + torch.linalg.norm(model.beta_fc2.weight, ord=1)
            + torch.linalg.norm(model.factor_fc1.weight, ord=1)
        )
        loss += l1_lambda * l1_norm

        # execute the backpropagation
        loss.backward()

        # execute an optimization step
        optimizer.step()

        # accumulate the loss
        total_loss += loss.item() * len(firm_chars)

    # compute the average cost per epoch
    mean_loss = total_loss / len(train_loader.dataset)

    return mean_loss


def evaluate_ae(model, criterion, eval_loader, device):
    # activate the evaluation mode
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch_idx, (firm_chars, portfolios, target) in enumerate(eval_loader):
            # transfer the data on the chosen device
            firm_chars, portfolios, target = (
                firm_chars.to(device),
                portfolios.to(device),
                target.to(device),
            )

            # forward propagation on the data
            beta, factor, prediction = model(firm_chars, portfolios)

            # compute the cost function w.r.t. the targets
            loss = criterion(prediction, target)

            # accumulate the loss
            total_loss += loss.item() * len(firm_chars)

    # compute the average cost per epoch
    mean_loss = total_loss / len(eval_loader.dataset)

    return mean_loss


def ae_learning_loop(
    ensemble,
    input_size,
    factor_size,
    batch_size,
    num_epoch,
    l1_lambda,
    learning_rate,
    train_dataset,
    val_dataset,
    test_dataset,
    patience,
    device,
):

    output_name = "best_model_ae.pt"
    mean_pred = np.zeros((test_dataset[0].shape[0], ensemble))

    for i in range(ensemble):
        print(f"Training ensemble {i+1}/{ensemble}...")
        # accumulators of average costs obtained per epoch
        train_losses = []
        val_losses = []

        # model definition
        model = Autoencoder(input_size, factor_size)
        model = model.to(device)

        # loss function
        criterion = nn.MSELoss()

        # optimizer definition
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # create mini-batches
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )

        # learning loop
        min_val_loss = 9999  # for early stopping, record the minimum validation loss
        num_no_improve = 0  # record the number of epochs with no improvements

        for epoch in range(num_epoch):
            print(f"Ensemble {i+1}/{ensemble}, Epoch {epoch+1}/{num_epoch}")
            # train the model
            train_loss = train_ae(
                model, criterion, train_loader, optimizer, l1_lambda, device
            )

            # get the validation loss
            val_loss = evaluate_ae(model, criterion, val_loader, device)

            # Save the costs obtained
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # early stopping
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                num_no_improve = 0

                # save the best model
                torch.save(model.state_dict(), output_name)
            else:
                num_no_improve += 1

            if num_no_improve >= patience:
                break

        model.load_state_dict(torch.load(output_name))  # load the best model
        model.eval()
        firm_chars, portfolios = test_dataset[2], test_dataset[3]
        _, factor, _ = model(firm_chars, portfolios)
        factor = factor.cpu().data.numpy()
        fac_list = [f"f{i}" for i in range(factor.shape[-1])]
        factor_df = pd.DataFrame(factor, columns=fac_list)
        factor_df = factor_df.expanding().mean()
        factor_df["ret_eom"] = test_dataset[4]
        factor_df["ret_eom"] = factor_df["ret_eom"].shift(-1)
        factor_df = factor_df.dropna()

        char_list = [f"z{i}" for i in range(test_dataset[0].shape[-1])]
        test_df = pd.DataFrame(test_dataset[0].cpu().numpy(), columns=char_list)
        test_df["ret_eom"] = test_dataset[1]
        test_df = test_df.merge(right=factor_df, on=["ret_eom"], how="inner")
        Z_test = torch.from_numpy(test_df[char_list].values).float().to(device)
        L_test = torch.from_numpy(test_df[fac_list].values).float().to(device)
        assert torch.all(test_dataset[0].cpu() == Z_test.cpu()), "Contact Chengyu for HELP"
        _, _, prediction = model(Z_test, L_test, oos=True)

        # get the OOS prediction
        mean_pred[:, i] = prediction.cpu().data.numpy()

    # compute the mean predictions across ensembles
    mean_pred = mean_pred.mean(axis=1)
    return mean_pred


if __name__ == "__main__":
    # for timing purpose
    print(datetime.datetime.now())

    # turn off Setting with Copy Warning
    pd.set_option("mode.chained_assignment", None)

    # read sample data
    file_path = "data/homework_sample_big.csv"
    raw = pd.read_csv(file_path, parse_dates=["date"], low_memory=False)
    raw = raw.drop(columns=["date"])
    raw["date"] = pd.to_datetime(
        raw["year"].astype(str) + "-" + raw["month"].astype(str) + "-01"
    )  # first day of the return month
    raw = raw[raw["stock_exret"].notna()].copy()
    gc.collect()

    # read list of predictors
    file_path = "data/factors_char_list.csv"
    var_list = list(pd.read_csv(file_path)["variable"].values)
    # remove one variable which would cause error
    if "zero_trades_126d" in var_list:
        var_list.remove("zero_trades_126d")
    binary_vars = ["convind", "divi", "divo", "rd", "securedind", "sin"]

    # transform each variable in each month
    monthly = raw.groupby("date")
    adj_raw_chunks = []
    for date, monthly_raw in monthly:
        print("Processing month: ", date)
        group = monthly_raw.copy()
        # rank transform each variable to [-1, 1]
        for var in var_list:
            var_median = group[var].median(skipna=True)
            group[var] = group[var].fillna(var_median)

            if var in binary_vars:
                pass
            else:
                group[var] = group[var].rank(method="dense") - 1
                group_max = group[var].max()
                if group_max > 0:
                    group[var] = (group[var] / group_max) * 2 - 1
                else:
                    group[var] = 0  # in case of all missing values
                    print("Warning:", date, var, "set to zero.")

        # add the adjusted values
        adj_raw_chunks.append(group)
        gc.collect()

    adj_raw = pd.concat(adj_raw_chunks, ignore_index=True)
    del adj_raw_chunks
    gc.collect()

    # add a constant to the list of characteristics
    adj_raw["constant"] = 1
    var_list.append("constant")
    port_list = [var + "_port" for var in var_list]

    # create managed portfolios for each month
    monthly_adj = adj_raw.groupby("date")
    data_chunks = []
    for date, group in monthly_adj:
        expand = group.copy()
        Zt = group[var_list].values
        Rt = group["stock_exret"].values
        Xt = LinearRegression(fit_intercept=False).fit(Zt, Rt).coef_
        X_port = pd.DataFrame(data=[Xt], columns=port_list)
        expand = pd.concat(
            [expand.reset_index(drop=True), X_port.reset_index(drop=True)], axis=1
        )
        expand[port_list] = expand[port_list].ffill()
        data_chunks.append(expand)
        gc.collect()

    data = pd.concat(data_chunks, ignore_index=True)
    del data_chunks, adj_raw
    gc.collect()

    # initialize
    starting = pd.to_datetime("20050101", format="%Y%m%d")
    counter = 0
    pred_out_chunks = []

    # estimation with expanding window
    while (starting + pd.DateOffset(years=8 + counter)) <= pd.to_datetime(
        "20240101", format="%Y%m%d"
    ):
        print("Processing year: ", starting.year + 6 + counter)
        cutoff = [
            starting,
            starting + pd.DateOffset(years=6 + counter),
            starting + pd.DateOffset(years=7 + counter),
            starting + pd.DateOffset(years=8 + counter),
        ]

        # cut the sample into training, validation, and testing sets
        train = data[(data["date"] >= cutoff[0]) & (data["date"] < cutoff[1])]
        validate = data[(data["date"] >= cutoff[1]) & (data["date"] < cutoff[2])]
        test = data[(data["date"] >= cutoff[2]) & (data["date"] < cutoff[3])]
        sample = data[(data["date"] >= cutoff[0]) & (data["date"] < cutoff[3])]
        sample = sample.groupby(["ret_eom"]).first().reset_index()

        # get Xs and Ys and Zs
        Z_train = train[var_list].values
        X_train = train[port_list].values
        Y_train = train["stock_exret"].values

        Z_val = validate[var_list].values
        X_val = validate[port_list].values
        Y_val = validate["stock_exret"].values

        Z_test = test[var_list].values
        Dates_test = test["ret_eom"].values
        Y_test = test["stock_exret"].values

        Dates_sample = sample["ret_eom"].values
        Z_sample = sample[var_list].values
        X_sample = sample[port_list].values

        # prepare output data
        pred_batch = test[["year", "month", "date", "permno", "stock_exret"]]

        # load data into tensor datasets (firm_chars, portfolios, target)
        train_dataset = TensorDataset(
            torch.from_numpy(Z_train).float(),
            torch.from_numpy(X_train).float(),
            torch.from_numpy(Y_train).float(),
        )
        val_dataset = TensorDataset(
            torch.from_numpy(Z_val).float(),
            torch.from_numpy(X_val).float(),
            torch.from_numpy(Y_val).float(),
        )
        Z_test = torch.from_numpy(Z_test).float()
        Z_test = Z_test.to(device)
        Z_sample = torch.from_numpy(Z_sample).float()
        Z_sample = Z_sample.to(device)
        X_sample = torch.from_numpy(X_sample).float()
        X_sample = X_sample.to(device)
        test_dataset = (Z_test, Dates_test, Z_sample, X_sample, Dates_sample)

        # define some general variables for nn
        batch_size = 1000
        num_epoch = 100
        patience = 5
        ensemble = 1 
        input_size = len(var_list)
        factor_size = 6  # could be a hyperparameter
        l1_lambda = 0.0001  # could be a hyperparameter
        learning_rate = 0.001  # could be a hyperparameter

        pred_batch["ae"] = ae_learning_loop(
            ensemble,
            input_size,
            factor_size,
            batch_size,
            num_epoch,
            l1_lambda,
            learning_rate,
            train_dataset,
            val_dataset,
            test_dataset,
            patience,
            device,
        )

        # add to the output data
        pred_out_chunks.append(pred_batch)

        # go to the next year
        counter += 1

        # for timing purpose
        print(datetime.datetime.now())

    pred_out = pd.concat(pred_out_chunks, ignore_index=True)
    del pred_out_chunks
    gc.collect()

    # print the OOS Total R2
    yreal = pred_out["stock_exret"].values
    ypred = pred_out["ae"].values
    r2 = 1 - np.sum(np.square((yreal - ypred))) / np.sum(np.square(yreal))
    print(r2)

    # write results to csv
    os.makedirs("data/predicted", exist_ok=True)
    pred_out.to_csv("data/predicted/ae_all_stock_exret_0.csv", index=False)
    print("Predictions saved to data/predicted/ae_all_stock_exret_0.csv")
