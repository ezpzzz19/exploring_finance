"""
Individual Assignment - Ensemble Alpha Strategy
================================================
Three-model ensemble for stock return prediction:
  1. IPCA         - conditional factor model, OOS predicted returns
  2. Autoencoder  - conditional factor model via managed portfolios
  3. NN3          - direct characteristics to return feedforward network

Each model produces its own OOS return predictions independently.
Final signal = equal-weight average of available model predictions.
Portfolio: Long-short top/bottom stocks, monthly rebalance.

No forward-looking information is used at any point.
"""

import datetime
import gc
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pandas.tseries.offsets import MonthBegin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset

from ipca_torch import IPCA

warnings.filterwarnings("ignore")
pd.set_option("mode.chained_assignment", None)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
USE_GPU = DEVICE.type == "cuda"
print(f"Device: {DEVICE}", flush=True)


# ===========================================================================
#  MODEL 1: NN3
# ===========================================================================

class NN3(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.fc4 = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        return self.fc4(x).squeeze(-1)


def _train_nn(model, criterion, loader, optimizer, device, l1_lambda=1e-4):
    model.train()
    total_loss = 0
    for X_b, Y_b in loader:
        X_b, Y_b = X_b.to(device), Y_b.to(device)
        optimizer.zero_grad()
        pred = model(X_b)
        loss = criterion(pred, Y_b)
        loss += l1_lambda * torch.norm(model.fc1.weight, p=1)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(X_b)
    return total_loss / len(loader.dataset)


def _eval_nn(model, criterion, loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_b, Y_b in loader:
            X_b, Y_b = X_b.to(device), Y_b.to(device)
            loss = criterion(model(X_b), Y_b)
            total_loss += loss.item() * len(X_b)
    return total_loss / len(loader.dataset)


def nn_predict(input_size, X_train, Y_train_dm, X_val, Y_val_dm, X_test,
               device, ensemble=3, epochs=100, patience=5, batch_size=2048):
    train_ds = TensorDataset(torch.from_numpy(X_train).float(),
                             torch.from_numpy(Y_train_dm).float())
    val_ds = TensorDataset(torch.from_numpy(X_val).float(),
                           torch.from_numpy(Y_val_dm).float())
    X_test_t = torch.from_numpy(X_test).float().to(device)
    all_preds = np.zeros((X_test.shape[0], ensemble))

    for i in range(ensemble):
        model = NN3(input_size).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.955)
        tr_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        va_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        best_val, no_improve, best_state = 1e9, 0, None
        for epoch in range(epochs):
            _train_nn(model, criterion, tr_loader, optimizer, device)
            vl = _eval_nn(model, criterion, va_loader, device)
            if vl < best_val:
                best_val, no_improve = vl, 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                no_improve += 1
            if no_improve >= patience:
                break
            scheduler.step()

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            all_preds[:, i] = model(X_test_t).cpu().numpy()
    return all_preds.mean(axis=1)


# ===========================================================================
#  MODEL 2: AUTOENCODER  (Gu, Kelly & Xiu 2021)
# ===========================================================================

class Autoencoder(nn.Module):
    def __init__(self, input_dim, factor_dim):
        super().__init__()
        self.beta_fc1 = nn.Linear(input_dim, 32, bias=False)
        self.beta_bn1 = nn.BatchNorm1d(32)
        self.beta_fc2 = nn.Linear(32, factor_dim, bias=False)
        self.factor_fc1 = nn.Linear(input_dim, factor_dim, bias=False)

    def forward(self, firm_chars, portfolios, oos=False):
        x = F.relu(self.beta_bn1(self.beta_fc1(firm_chars)))
        x = self.beta_fc2(x)
        x = torch.squeeze(x)
        if not oos:
            y = self.factor_fc1(portfolios)
            y = torch.squeeze(y)
        else:
            y = portfolios
        out = torch.sum(x * y, dim=1)
        return x, y, out


def _train_ae(model, criterion, loader, optimizer, l1_lambda, device):
    model.train()
    torch.set_grad_enabled(True)
    total_loss = 0
    for chars, ports, target in loader:
        chars, ports, target = chars.to(device), ports.to(device), target.to(device)
        optimizer.zero_grad()
        _, _, pred = model(chars, ports)
        loss = criterion(pred, target)
        l1 = (torch.norm(model.beta_fc1.weight, p=1)
              + torch.norm(model.beta_fc2.weight, p=1)
              + torch.norm(model.factor_fc1.weight, p=1))
        loss += l1_lambda * l1
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(chars)
    return total_loss / len(loader.dataset)


def _eval_ae(model, criterion, loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for chars, ports, target in loader:
            chars, ports, target = chars.to(device), ports.to(device), target.to(device)
            _, _, pred = model(chars, ports)
            loss = criterion(pred, target)
            total_loss += loss.item() * len(chars)
    return total_loss / len(loader.dataset)


def ae_predict(Z_train, X_train, Y_train,
               Z_val, X_val, Y_val,
               Z_test, Y_test, dates_test,
               Z_sample, X_sample, dates_sample,
               device, factor_size=6, ensemble=3,
               epochs=100, patience=5, batch_size=1000,
               l1_lambda=1e-4, lr=1e-3):
    input_size = Z_train.shape[1]
    train_ds = TensorDataset(torch.from_numpy(Z_train).float(),
                             torch.from_numpy(X_train).float(),
                             torch.from_numpy(Y_train).float())
    val_ds = TensorDataset(torch.from_numpy(Z_val).float(),
                           torch.from_numpy(X_val).float(),
                           torch.from_numpy(Y_val).float())
    Z_test_t = torch.from_numpy(Z_test).float().to(device)
    Z_sample_t = torch.from_numpy(Z_sample).float().to(device)
    X_sample_t = torch.from_numpy(X_sample).float().to(device)
    mean_pred = np.zeros((Z_test.shape[0], ensemble))

    for i in range(ensemble):
        model = Autoencoder(input_size, factor_size).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        tr_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        va_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        best_val, no_improve, best_state = 1e9, 0, None
        for epoch in range(epochs):
            _train_ae(model, criterion, tr_loader, optimizer, l1_lambda, device)
            vl = _eval_ae(model, criterion, va_loader, device)
            if vl < best_val:
                best_val, no_improve = vl, 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                no_improve += 1
            if no_improve >= patience:
                break

        model.load_state_dict(best_state)
        model.eval()

        with torch.no_grad():
            _, factor_sample, _ = model(Z_sample_t, X_sample_t)
        factor_np = factor_sample.cpu().numpy()
        fac_cols = [f"f{j}" for j in range(factor_np.shape[-1])]

        factor_df = pd.DataFrame(factor_np, columns=fac_cols)
        factor_df["ret_eom"] = dates_sample
        factor_df[fac_cols] = factor_df[fac_cols].expanding().mean()
        factor_df["ret_eom"] = factor_df["ret_eom"].shift(-1)
        factor_df = factor_df.dropna()

        test_df = pd.DataFrame({"ret_eom": dates_test})
        test_df = test_df.merge(factor_df, on="ret_eom", how="inner")

        if len(test_df) == 0:
            mean_pred[:, i] = 0.0
            continue

        L_test = torch.from_numpy(test_df[fac_cols].values).float().to(device)
        with torch.no_grad():
            _, _, prediction = model(Z_test_t[:len(L_test)], L_test, oos=True)
        mean_pred[:len(L_test), i] = prediction.cpu().numpy()

    return mean_pred.mean(axis=1)


# ===========================================================================
#  DATA LOADING & PREPROCESSING
# ===========================================================================

def load_and_preprocess(data_dir):
    print("Loading data...", flush=True)
    raw = pd.read_csv(os.path.join(data_dir, "mma_sample_v2.csv"), low_memory=False)
    stock_vars = list(
        pd.read_csv(os.path.join(data_dir, "factor_char_list.csv"))["variable"].values
    )
    mkt = pd.read_csv(os.path.join(data_dir, "mkt_ind.csv"))

    raw["date"] = pd.to_datetime(
        raw["year"].astype(str) + "-" + raw["month"].astype(str) + "-01"
    )
    raw = raw[raw["stock_exret"].notna()].copy()
    for var in stock_vars:
        raw[var] = raw[var].astype(float)

    print(f"  Raw data: {raw.shape[0]:,} rows, {raw['date'].nunique()} months", flush=True)

    print("Rank-transforming characteristics...", flush=True)
    chunks = []
    for date, group in raw.groupby("date"):
        g = group.copy()
        for var in stock_vars:
            med = g[var].median(skipna=True)
            g[var] = g[var].fillna(med)
            ranked = g[var].rank(method="dense") - 1
            rmax = ranked.max()
            if rmax > 0:
                g[var] = (ranked / rmax) * 2 - 1
            else:
                g[var] = 0.0
        chunks.append(g)
    data = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()

    print(f"  Preprocessed: {data.shape[0]:,} rows", flush=True)
    return data, stock_vars, mkt


# ===========================================================================
#  IPCA - OOS PREDICTED RETURNS
# ===========================================================================

def run_ipca_stage(data, stock_vars, K=6, oos_min_periods=96):
    print(f"\n{'='*60}", flush=True)
    print(f"IPCA  (K={K}, OOS min periods={oos_min_periods})", flush=True)
    print(f"{'='*60}", flush=True)

    ipca_data = data.set_index(["date", "permno"])[["stock_exret"] + stock_vars].copy()
    dates = ipca_data.index.get_level_values(0).unique().sort_values()
    gFac = pd.DataFrame(1.0, index=["anomaly"], columns=dates)

    model = IPCA(K=K, add_constant=True, device=str(DEVICE), max_iter=500, min_tol=1e-4)
    results = model.fit(
        ipca_data, return_col="stock_exret",
        OOS=True, OOS_min_periods=oos_min_periods,
        gFac=gFac, Beta_fit=True, R_fit=True,
        disp=True, disp_every=24,
    )

    beta_df = results["fittedBeta"]
    if results["rfits"]:
        print(f"  IPCA R2_Total: {results['rfits']['R2_Total']:.6f}", flush=True)
        print(f"  IPCA R2_Pred:  {results['rfits']['R2_Pred']:.6f}", flush=True)

    lambda_df = results["Lambda"]
    oos_dates = beta_df.index.get_level_values(0).unique().sort_values()
    parts = []
    for t in oos_dates:
        bt = beta_df.loc[t]
        lt = lambda_df[t].values
        parts.append(pd.DataFrame({"ipca_pred": bt.values @ lt}, index=bt.index))
    ipca_pred = pd.concat(parts, keys=oos_dates, names=["date", "permno"])

    print(f"  IPCA predictions: {len(ipca_pred):,} rows, "
          f"{ipca_pred.index.get_level_values(0).nunique()} months", flush=True)
    return ipca_pred


# ===========================================================================
#  EXPANDING-WINDOW ENSEMBLE  (NN3 + Autoencoder + IPCA)
# ===========================================================================

def run_ensemble_stage(data, stock_vars, ipca_pred):
    print("ENSEMBLE  (NN3 + Autoencoder + IPCA)", flush=True)

    ret_var = "stock_exret"

    # Managed portfolios for autoencoder
    print("  Computing managed portfolios...", flush=True)
    port_list = [v + "_port" for v in stock_vars]
    port_chunks = []
    for date, group in data.groupby("date"):
        g = group.copy()
        Zt = g[stock_vars].values
        Rt = g[ret_var].values
        coefs = LinearRegression(fit_intercept=False).fit(Zt, Rt).coef_
        for j, v in enumerate(port_list):
            g[v] = coefs[j]
        port_chunks.append(g)
    data_ae = pd.concat(port_chunks, ignore_index=True)
    del port_chunks
    gc.collect()
    print(f"  Managed portfolios: {len(port_list)}", flush=True)

    # Expanding window
    starting = pd.to_datetime("2000-01-01")
    counter = 0
    pred_out = pd.DataFrame()

    while (starting + pd.DateOffset(years=11 + counter)) <= pd.to_datetime("2024-01-01"):
        cutoff = [
            starting,
            starting + pd.DateOffset(years=8 + counter),
            starting + pd.DateOffset(years=10 + counter),
            starting + pd.DateOffset(years=11 + counter),
        ]
        test_year = cutoff[2].year
        print(f"\n  Window {counter+1}: train->{cutoff[1].year-1}, "
              f"val->{cutoff[2].year-1}, test={test_year}", flush=True)

        train = data_ae[(data_ae["date"] >= cutoff[0]) & (data_ae["date"] < cutoff[1])]
        validate = data_ae[(data_ae["date"] >= cutoff[1]) & (data_ae["date"] < cutoff[2])]
        test = data_ae[(data_ae["date"] >= cutoff[2]) & (data_ae["date"] < cutoff[3])]

        if len(test) == 0:
            print("    Skipping -- no test data", flush=True)
            counter += 1
            continue

        sample = data_ae[(data_ae["date"] >= cutoff[0]) & (data_ae["date"] < cutoff[3])]
        sample_unique = sample.groupby("ret_eom").first().reset_index()

        scaler = StandardScaler().fit(train[stock_vars])
        X_train_sc = scaler.transform(train[stock_vars])
        X_val_sc = scaler.transform(validate[stock_vars])
        X_test_sc = scaler.transform(test[stock_vars])
        Y_train = train[ret_var].values
        Y_val = validate[ret_var].values
        Y_test = test[ret_var].values

        print(f"    Shapes: train={X_train_sc.shape}, val={X_val_sc.shape}, test={X_test_sc.shape}", flush=True)

        Y_mean = np.mean(Y_train)
        Y_train_dm = Y_train - Y_mean
        Y_val_dm = Y_val - Y_mean

        reg_pred = test[["year", "month", "date", "permno", ret_var]].copy()

        # (A) NN3
        print("    NN3...", flush=True)
        nn_p = nn_predict(
            input_size=X_train_sc.shape[1],
            X_train=X_train_sc, Y_train_dm=Y_train_dm,
            X_val=X_val_sc, Y_val_dm=Y_val_dm,
            X_test=X_test_sc, device=DEVICE,
            ensemble=3, epochs=100, patience=5, batch_size=2048,
        ) + Y_mean
        reg_pred["nn3"] = nn_p
        print("    NN3 done", flush=True)

        # (B) Autoencoder
        print("    Autoencoder...", flush=True)
        ae_p = ae_predict(
            Z_train=train[stock_vars].values,
            X_train=train[port_list].values,
            Y_train=Y_train,
            Z_val=validate[stock_vars].values,
            X_val=validate[port_list].values,
            Y_val=Y_val,
            Z_test=test[stock_vars].values,
            Y_test=Y_test,
            dates_test=test["ret_eom"].values,
            Z_sample=sample_unique[stock_vars].values,
            X_sample=sample_unique[port_list].values,
            dates_sample=sample_unique["ret_eom"].values,
            device=DEVICE, factor_size=6, ensemble=3,
            epochs=100, patience=5, batch_size=1000,
            l1_lambda=1e-4, lr=1e-3,
        )
        reg_pred["ae"] = ae_p
        print("    AE done", flush=True)

        # (C) IPCA (pre-computed, just merge)
        test_idx = test.set_index(["date", "permno"]).index
        ipca_matched = ipca_pred.reindex(test_idx)
        reg_pred["ipca"] = ipca_matched["ipca_pred"].values
        n_ipca = reg_pred["ipca"].notna().sum()
        print(f"    IPCA: {n_ipca}/{len(reg_pred)} matched", flush=True)

        # Ensemble average
        model_cols = ["nn3", "ae", "ipca"]
        reg_pred["ensemble"] = reg_pred[model_cols].mean(axis=1, skipna=True)

        pred_out = pd.concat([pred_out, reg_pred], ignore_index=True)
        counter += 1
        gc.collect()

    # OOS R-squared
    print(f"\n{'-'*60}", flush=True)
    print("OOS R-squared (benchmark = 0):", flush=True)
    yreal = pred_out[ret_var].values
    for mn in ["nn3", "ae", "ipca", "ensemble"]:
        yp = pred_out[mn].values
        mask = ~np.isnan(yp)
        if mask.sum() == 0:
            print(f"  {mn:10s}  no predictions")
            continue
        r2 = 1 - np.sum((yreal[mask] - yp[mask])**2) / np.sum(yreal[mask]**2)
        print(f"  {mn:10s}  R2 = {r2:.6f}  ({r2*100:.4f}%)", flush=True)

    return pred_out


# ===========================================================================
#  PORTFOLIO CONSTRUCTION & EVALUATION
# ===========================================================================

def evaluate_portfolio(pred_out, mkt, model="ensemble", n_long=60, n_short=40):
    print(f"\n{'='*60}", flush=True)
    print(f"PORTFOLIO EVALUATION  (model={model}, long={n_long}, short={n_short})", flush=True)
    print(f"{'='*60}", flush=True)

    ret_var = "stock_exret"
    pred = pred_out[pred_out[model].notna()].copy()

    predicted = pred.groupby(["year", "month"])[model]
    pred["rank"] = np.floor(
        predicted.transform(lambda s: s.rank()) * 10
        / predicted.transform(lambda s: len(s) + 1)
    )
    pred = pred.sort_values(["year", "month", "rank", "permno"])

    monthly_port = (
        pred.groupby(["year", "month", "rank"])
        .apply(lambda df: np.mean(df[ret_var]))
        .unstack().dropna().reset_index()
    )
    monthly_port.columns = ["year", "month"] + [f"port_{x}" for x in range(1, 11)]
    monthly_port["port_LS"] = monthly_port["port_10"] - monthly_port["port_1"]

    ls_returns, long_holdings, short_holdings = [], [], []
    for (yr, mo), group in pred.groupby(["year", "month"]):
        sg = group.sort_values(model, ascending=False)
        top = sg.head(n_long)
        bottom = sg.tail(n_short)
        lr = top[ret_var].mean()
        sr_val = bottom[ret_var].mean()
        ls_returns.append({"year": yr, "month": mo,
                           "long_ret": lr, "short_ret": sr_val,
                           "ls_ret": lr - sr_val})
        long_holdings.append(top[["year", "month", "date", "permno"]])
        short_holdings.append(bottom[["year", "month", "date", "permno"]])

    strat = pd.DataFrame(ls_returns)
    strat = strat.merge(mkt, on=["year", "month"], how="inner")
    strat["mkt_rf"] = strat["sp_ret"] - strat["rf"]

    T = len(strat)
    mr = strat["ls_ret"].mean()
    sd = strat["ls_ret"].std()
    sharpe = mr / sd * np.sqrt(12)
    print(f"\n  Months: {T}", flush=True)
    print(f"  Mean monthly return:  {mr*100:.3f}%", flush=True)
    print(f"  Std monthly return:   {sd*100:.3f}%", flush=True)
    print(f"  Annualized Sharpe:    {sharpe:.3f}", flush=True)

    nw = smf.ols(formula="ls_ret ~ mkt_rf", data=strat).fit(
        cov_type="HAC", cov_kwds={"maxlags": 6}, use_t=True)
    alpha = nw.params["Intercept"]
    beta = nw.params["mkt_rf"]
    alpha_t = nw.tvalues["Intercept"]
    ir = alpha / np.sqrt(nw.mse_resid) * np.sqrt(12)
    print(f"\n  CAPM Alpha (monthly): {alpha*100:.3f}%  (t={alpha_t:.2f})", flush=True)
    print(f"  CAPM Alpha (annual):  {alpha*12*100:.3f}%", flush=True)
    print(f"  CAPM Beta:            {beta:.3f}", flush=True)
    print(f"  Information Ratio:    {ir:.3f}", flush=True)

    max_loss = strat["ls_ret"].min()
    print(f"\n  Max 1-month loss:     {max_loss*100:.3f}%", flush=True)

    strat["log_ret"] = np.log(strat["ls_ret"] + 1)
    strat["cum_log"] = strat["log_ret"].cumsum()
    peak = strat["cum_log"].cummax()
    max_dd = (peak - strat["cum_log"]).max()
    print(f"  Maximum Drawdown:     {max_dd*100:.3f}%", flush=True)

    long_df = pd.concat(long_holdings, ignore_index=True)
    short_df = pd.concat(short_holdings, ignore_index=True)

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

    l_to = calc_turnover(long_df)
    s_to = calc_turnover(short_df)
    print(f"\n  Long turnover:        {l_to*100:.1f}%", flush=True)
    print(f"  Short turnover:       {s_to*100:.1f}%", flush=True)

    print(f"\n  --- Long-only portfolio (top {n_long}) ---", flush=True)
    lm = strat["long_ret"].mean()
    ls_std = strat["long_ret"].std()
    print(f"  Mean monthly return:  {lm*100:.3f}%", flush=True)
    print(f"  Annualized Sharpe:    {lm/ls_std*np.sqrt(12):.3f}", flush=True)

    strat["date_plot"] = pd.to_datetime(
        strat["year"].astype(str) + "-" + strat["month"].astype(str) + "-01")

    # Rolling annualized Sharpe ratio (expanding window from start)
    def expanding_sharpe(series):
        """Compute expanding annualized Sharpe ratio at each point."""
        sharpes = []
        for i in range(1, len(series) + 1):
            s = series.iloc[:i]
            if i < 2:
                sharpes.append(0.0)
            else:
                sharpes.append(s.mean() / s.std() * np.sqrt(12))
        return pd.Series(sharpes, index=series.index)

    strat["sharpe_ls"] = expanding_sharpe(strat["ls_ret"]).values
    strat["sharpe_long"] = expanding_sharpe(strat["long_ret"]).values
    strat["sharpe_mkt"] = expanding_sharpe(strat["sp_ret"]).values

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(strat["date_plot"], strat["sharpe_ls"], label="Long-Short", linewidth=2)
    ax.plot(strat["date_plot"], strat["sharpe_long"], label=f"Long Top {n_long}", linewidth=1.5)
    ax.plot(strat["date_plot"], strat["sharpe_mkt"], label="S&P 500", linewidth=1.5, alpha=0.7)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.5)
    ax.set_title("Expanding Annualized Sharpe Ratio - Ensemble Alpha (NN3 + Autoencoder + IPCA)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Annualized Sharpe Ratio")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs("output", exist_ok=True)
    plt.savefig("output/cumulative_returns.png", dpi=150)
    print(f"\n  Plot saved to output/cumulative_returns.png", flush=True)

    strat.to_csv("output/strategy_returns.csv", index=False)
    monthly_port.to_csv("output/decile_returns.csv", index=False)
    print("  Strategy returns saved to output/strategy_returns.csv", flush=True)

    print(f"\n{nw.summary()}", flush=True)

    return strat, monthly_port


# ===========================================================================
#  MAIN
# ===========================================================================

if __name__ == "__main__":
    t_start = datetime.datetime.now()
    print(f"Start: {t_start}")
    print(f"{'='*60}\n")

    DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

    data, stock_vars, mkt = load_and_preprocess(DATA_DIR)

    # IPCA: OOS predicted returns (first 96 months train, OOS from Jan 2008)
    ipca_pred = run_ipca_stage(data, stock_vars, K=6, oos_min_periods=96)

    # Ensemble: NN3 + Autoencoder + IPCA
    pred_out = run_ensemble_stage(data, stock_vars, ipca_pred)

    os.makedirs("output", exist_ok=True)
    pred_out.to_csv("output/predictions.csv", index=False)
    print("\nPredictions saved to output/predictions.csv", flush=True)

    strat, deciles = evaluate_portfolio(pred_out, mkt, model="ensemble", n_long=60, n_short=40)

    t_end = datetime.datetime.now()
    print(f"\nTotal runtime: {t_end - t_start}")
    print("Done!")
