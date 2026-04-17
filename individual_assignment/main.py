"""
Individual Assignment - Ensemble Alpha Strategy
================================================
Two-model ensemble for stock return prediction:
  1. XGBoost      - gradient boosted trees for direct return prediction
  2. Autoencoder  - conditional factor model via managed portfolios

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

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset

warnings.filterwarnings("ignore")
pd.set_option("mode.chained_assignment", None)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

def _get_device():
    """
    Pick the best available device in priority order:
      1. CUDA  (NVIDIA GPU)
      2. MPS   (Apple Silicon)
      3. CPU   (fallback)
    Works on Lightning AI, local Mac, or any cloud GPU provider.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = _get_device()
print(f"Device: {DEVICE}", flush=True)


# ===========================================================================
#  MODEL 1: XGBoost (Gradient Boosted Trees)
# ===========================================================================

def xgb_predict(X_train, Y_train_dm, X_val, Y_val_dm, X_test):
    """
    XGBoost with hyperparameter grid search over learning rate and max_depth,
    following the tree.py pattern from the labs.
    Returns (predictions, feature_importances_dict).
    """
    # Grid search over learning rate (10^lambda) and tree depth
    lambdas = np.arange(-2, -0.9, 1)          # 0.01, 0.1
    nl = list(range(1, 3))                      # depth 1, 2
    n_trees = 1000
    val_mse = np.zeros((len(lambdas), len(nl)))

    for ind, i in enumerate(lambdas):
        for jnd, j in enumerate(nl):
            model = XGBRegressor(
                n_estimators=n_trees, max_depth=j,
                learning_rate=(10 ** i), tree_method='hist',
                reg_alpha=0, reg_lambda=0, random_state=SEED,
                callbacks=[xgb.callback.EarlyStopping(rounds=15, save_best=True)],
            )
            model.fit(
                X_train, Y_train_dm,
                eval_set=[(X_val, Y_val_dm)],
                verbose=False,
            )
            val_mse[ind, jnd] = mean_squared_error(
                Y_val_dm, model.predict(X_val)
            )

    # Refit best model
    best_i, best_j = divmod(val_mse.argmin(), val_mse.shape[1])
    best_lr = 10 ** lambdas[best_i]
    best_depth = nl[best_j]
    print(f"      XGB best: lr={best_lr:.4f}, depth={best_depth}", flush=True)

    model = XGBRegressor(
        n_estimators=n_trees, max_depth=best_depth,
        learning_rate=best_lr, tree_method='hist',
        reg_alpha=0, reg_lambda=0, random_state=SEED,
        callbacks=[xgb.callback.EarlyStopping(rounds=15, save_best=True)],
    )
    model.fit(
        X_train, Y_train_dm,
        eval_set=[(X_val, Y_val_dm)],
        verbose=False,
    )
    return model.predict(X_test), model.feature_importances_


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
#  MODEL 3: NN2  (2-hidden-layer MLP -- the lab1 Sharpe driver)
# ===========================================================================

class NN2(nn.Module):
    """Two-hidden-layer MLP with batch norm. L1 reg on fc1 added in training."""
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 16)
        self.bn2 = nn.BatchNorm1d(16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        return torch.squeeze(self.fc3(x))


def _train_nn(model, criterion, loader, optimizer, device, l1_lambda=1e-4):
    model.train()
    torch.set_grad_enabled(True)
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss = loss + l1_lambda * torch.norm(model.fc1.weight, p=1)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(x)
    return total_loss / len(loader.dataset)


def _eval_nn(model, criterion, loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            total_loss += loss.item() * len(x)
    return total_loss / len(loader.dataset)


def nn_predict(X_train, Y_train_dm, X_val, Y_val_dm, X_test,
               device, ensemble=3, epochs=100, patience=5,
               batch_size=10000, lr=0.01, l1_lambda=1e-4):
    """Train `ensemble` NN2 models; return mean OOS predictions (demeaned)."""
    input_dim = X_train.shape[1]
    train_ds = TensorDataset(
        torch.from_numpy(X_train.astype(np.float32)),
        torch.from_numpy(Y_train_dm.astype(np.float32)),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val.astype(np.float32)),
        torch.from_numpy(Y_val_dm.astype(np.float32)),
    )
    X_test_t = torch.from_numpy(X_test.astype(np.float32)).to(device)
    criterion = nn.MSELoss()
    mean_pred = np.zeros((X_test.shape[0], ensemble), dtype=np.float32)

    for i in range(ensemble):
        model = NN2(input_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        # decay lr: 0.01 * 0.955^100 ~= 1e-4 (matches lab1)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.955)
        tr_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        va_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        best_val, no_improve, best_state = float("inf"), 0, None
        for epoch in range(epochs):
            _train_nn(model, criterion, tr_loader, optimizer, device, l1_lambda)
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
            mean_pred[:, i] = model(X_test_t).cpu().numpy()
        del model, optimizer, scheduler
        gc.collect()

    return mean_pred.mean(axis=1)


# ===========================================================================
#  DATA LOADING & PREPROCESSING
# ===========================================================================

def load_and_preprocess(data_dir):
    print("Loading data...", flush=True)
    # Use enhanced dataset (with WRDS ratios) if available, else fallback
    enhanced_path = os.path.join(data_dir, "mma_sample_enhanced.csv")
    base_path = os.path.join(data_dir, "mma_sample_v2.csv")
    if os.path.exists(enhanced_path):
        print("  Using ENHANCED dataset (with WRDS financial ratios)", flush=True)
        raw = pd.read_csv(enhanced_path, parse_dates=["date"], low_memory=False)
    else:
        print("  Using base dataset (run download_wrds_ratios.py to enhance)", flush=True)
        raw = pd.read_csv(base_path, parse_dates=["date"], low_memory=False)

    all_vars = list(
        pd.read_csv(os.path.join(data_dir, "factor_char_list.csv"))["variable"].values
    )
    # Also include any WRDS columns that were merged in
    wrds_extra_cols = [
        # Financial Ratios Suite
        "capei", "evm", "pe_op_dil", "pe_exi", "ps", "pcf", "ptb",
        "peg_trailing", "divyield", "npm", "opmbd", "opmad", "gpm", "cfm",
        "roa", "roe", "roce", "efftax", "aftret_eq", "aftret_invcapx",
        "pretret_noa", "de_ratio", "debt_ebitda", "debt_capital", "debt_at",
        "intcov_ratio", "curr_ratio", "quick_ratio", "cash_ratio",
        "cash_conversion", "inv_turn", "rect_turn", "pay_turn", "sale_invcap",
        "accrual", "fcf_ocf", "cash_debt", "short_debt",
        # IBES Analyst Consensus
        "ibes_numest", "ibes_meanest", "ibes_medest", "ibes_stdev",
        "ibes_disp", "ibes_range", "ibes_revision",
        "ibes_numup", "ibes_numdown",
        # Institutional Ownership
        "io_num_holders", "io_new_holders", "io_share_change_pct",
        "io_buy_sell_ratio",
        # Short Interest
        "si_shares_short", "si_shares_short_chg",
    ]
    extra_vars = [c for c in wrds_extra_cols if c in raw.columns and c not in all_vars]
    all_vars = all_vars + extra_vars
    # Keep only predictors that actually exist in the data
    stock_vars = [v for v in all_vars if v in raw.columns]
    print(f"  Using {len(stock_vars)} characteristics ({len(extra_vars)} from WRDS)", flush=True)

    mkt = pd.read_csv(os.path.join(data_dir, "mkt_ind.csv"))

    # Ensure date column exists (use existing or construct from year/month)
    if "date" not in raw.columns or raw["date"].isna().all():
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

    # ── Build char × macro interactions (lab1-style) ──────────────────────────
    # Each firm-char multiplied by each macro variable for that month gives the
    # model access to regime-dependent anomaly behavior. Macros are constant in
    # the cross-section on any given date, so the only way a cross-sectional
    # model can learn regime effects is via these hand-crafted interactions.
    macro_path = os.path.join(data_dir, "macro_data_goyal_enhanced.csv")
    # Drop the 8 classic Goyal predictors (dp/ep/bm/ntis/tbl/tms/dfy/svar):
    # the professor-provided initial file only covers them through 2019-12,
    # so on any 2020-2023 test month they'd be mean-filled (= constant), and
    # 151 chars × 8 classics = 1208 near-constant interaction columns just
    # add noise and ~1.5 GB of RAM per window without contributing signal.
    # We keep only the GWZ 2024 monthly predictors, which cover through 2023.
    CLASSIC_MACROS = {"dp", "ep", "bm", "ntis", "tbl", "tms", "dfy", "svar"}
    interaction_vars = []
    if os.path.exists(macro_path):
        print("Building char × macro interactions...", flush=True)
        macro = pd.read_csv(macro_path, parse_dates=["date1"])
        macro_vars = [
            c for c in macro.columns
            if c not in ("yyyymm", "date1") and c not in CLASSIC_MACROS
        ]

        # align macro to the stock data's month-start dates
        macro_aligned = macro.rename(columns={"date1": "date"}).sort_values("date")
        # merge_asof = use the most recent macro row available on or before each stock date (no look-ahead)
        data = data.sort_values("date").reset_index(drop=True)
        data = pd.merge_asof(
            data, macro_aligned[["date"] + macro_vars],
            on="date", direction="backward",
        )
        # fill any remaining NaN macros with column mean (edge of series)
        for mv in macro_vars:
            data[mv] = data[mv].fillna(data[mv].mean())

        # vectorized interaction: for each macro var, multiply every char column
        # by the macro value on that row → len(stock_vars) new columns per macro
        char_mat = data[stock_vars].values
        new_cols = {}
        for mv in macro_vars:
            m_vec = data[mv].values.reshape(-1, 1)
            prod = char_mat * m_vec
            for j, sv in enumerate(stock_vars):
                new_cols[f"{sv}_{mv}"] = prod[:, j]
                interaction_vars.append(f"{sv}_{mv}")
        # single concat avoids frame fragmentation
        data = pd.concat(
            [data, pd.DataFrame(new_cols, index=data.index)], axis=1
        )
        del new_cols, char_mat
        gc.collect()
        print(f"  Added {len(interaction_vars)} interaction features "
              f"({len(stock_vars)} chars × {len(macro_vars)} macros)", flush=True)
    else:
        print("  macro_data_goyal_enhanced.csv not found — skipping interactions", flush=True)

    # Industry dummies (ind1..ind68) if present in the raw dataset — used by
    # tree/NN models but NOT rank-transformed or interacted.
    ind_list = [f"ind{i}" for i in range(1, 69) if f"ind{i}" in data.columns]
    if ind_list:
        print(f"  Found {len(ind_list)} industry dummies", flush=True)

    # Full feature list used by XGBoost + NN (AE still uses `stock_vars` only)
    feature_vars = stock_vars + interaction_vars + ind_list

    print(f"  Preprocessed: {data.shape[0]:,} rows, "
          f"{len(feature_vars)} total features", flush=True)
    return data, stock_vars, feature_vars, mkt


#  EXPANDING-WINDOW ENSEMBLE  (XGBoost + NN2 + Autoencoder)
def run_ensemble_stage(data, stock_vars, feature_vars):
    print("ENSEMBLE  (XGBoost + NN2 + Autoencoder)", flush=True)

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
    all_feat_imp = []

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

        # XGBoost + NN use the FULL feature matrix (chars + macro interactions + industry)
        # float32 halves memory vs the default float64 — at ~5k features and
        # ~170k training rows, each copy is ~6.9 GB in fp64, ~3.5 GB in fp32.
        scaler = StandardScaler().fit(train[feature_vars])
        X_train_sc = scaler.transform(train[feature_vars]).astype(np.float32, copy=False)
        X_val_sc = scaler.transform(validate[feature_vars]).astype(np.float32, copy=False)
        X_test_sc = scaler.transform(test[feature_vars]).astype(np.float32, copy=False)
        Y_train = train[ret_var].values.astype(np.float32, copy=False)
        Y_val = validate[ret_var].values.astype(np.float32, copy=False)
        Y_test = test[ret_var].values.astype(np.float32, copy=False)

        print(f"    Shapes: train={X_train_sc.shape}, val={X_val_sc.shape}, test={X_test_sc.shape}", flush=True)

        Y_mean = np.mean(Y_train)
        Y_train_dm = Y_train - Y_mean
        Y_val_dm = Y_val - Y_mean

        reg_pred = test[["year", "month", "date", "permno", ret_var]].copy()

        # (A) XGBoost
        print("    XGBoost...", flush=True)
        xgb_p, feat_imp = xgb_predict(
            X_train=X_train_sc, Y_train_dm=Y_train_dm,
            X_val=X_val_sc, Y_val_dm=Y_val_dm,
            X_test=X_test_sc,
        )
        xgb_p = xgb_p + Y_mean
        reg_pred["xgb"] = xgb_p
        all_feat_imp.append(feat_imp)
        print("    XGBoost done", flush=True)

        # (B) NN2 -- the lab1 Sharpe driver
        print("    NN2...", flush=True)
        nn_p = nn_predict(
            X_train=X_train_sc, Y_train_dm=Y_train_dm,
            X_val=X_val_sc, Y_val_dm=Y_val_dm,
            X_test=X_test_sc,
            device=DEVICE, ensemble=3,
            epochs=100, patience=5, batch_size=10000,
            lr=0.01, l1_lambda=1e-4,
        )
        reg_pred["nn"] = nn_p + Y_mean
        print("    NN2 done", flush=True)

        # (C) Autoencoder -- uses raw rank-transformed chars, NOT interactions
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

        # Ensemble average across all three models
        model_cols = ["xgb", "nn", "ae"]
        reg_pred["ensemble"] = reg_pred[model_cols].mean(axis=1, skipna=True)

        pred_out = pd.concat([pred_out, reg_pred], ignore_index=True)
        counter += 1

        # Explicit per-window cleanup: without this, window N's ~3.5 GB scaled
        # matrices linger through python's lazy collection while window N+1
        # allocates its own set and kicks us over the 62 GB cap.
        del X_train_sc, X_val_sc, X_test_sc
        del Y_train, Y_val, Y_test, Y_train_dm, Y_val_dm
        del train, validate, test, sample, sample_unique, reg_pred
        del scaler, xgb_p, nn_p, ae_p, feat_imp
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- Save XGBoost feature importances (Slide 3 & 5) ---
    if all_feat_imp:
        avg_imp = np.mean(np.vstack(all_feat_imp), axis=0)
        feat_df = pd.DataFrame({"feature": feature_vars, "importance": avg_imp})
        feat_df = feat_df.sort_values("importance", ascending=False).reset_index(drop=True)
        os.makedirs("individual_assignment/output_last_day", exist_ok=True)
        feat_df.to_csv("individual_assignment/output_last_day/xgb_feature_importance.csv", index=False)
        print(f"\n  Top 20 XGBoost features (avg across windows):", flush=True)
        for _, row in feat_df.head(20).iterrows():
            print(f"    {row['feature']:30s}  {row['importance']:.6f}", flush=True)

    # OOS R-squared
    print(f"\n{'-'*60}", flush=True)
    print("OOS R-squared (benchmark = 0):", flush=True)
    yreal = pred_out[ret_var].values
    r2_records = []
    for mn in ["xgb", "nn", "ae", "ensemble"]:
        yp = pred_out[mn].values
        mask = ~np.isnan(yp)
        if mask.sum() == 0:
            print(f"  {mn:10s}  no predictions")
            continue
        r2 = 1 - np.sum((yreal[mask] - yp[mask])**2) / np.sum(yreal[mask]**2)
        print(f"  {mn:10s}  R2 = {r2:.6f}  ({r2*100:.4f}%)", flush=True)
        r2_records.append({"model": mn, "oos_r2": r2, "oos_r2_pct": r2 * 100})

    # --- Save OOS R² by year for each model (Slide 3) ---
    r2_by_year = []
    for mn in ["xgb", "nn", "ae", "ensemble"]:
        yp = pred_out[mn].values
        for yr in sorted(pred_out["year"].unique()):
            mask_yr = (pred_out["year"].values == yr) & ~np.isnan(yp)
            if mask_yr.sum() == 0:
                continue
            r2_yr = 1 - np.sum((yreal[mask_yr] - yp[mask_yr])**2) / np.sum(yreal[mask_yr]**2)
            r2_by_year.append({"model": mn, "year": yr, "oos_r2": r2_yr, "oos_r2_pct": r2_yr * 100})

    os.makedirs("individual_assignment/output_last_day", exist_ok=True)
    pd.DataFrame(r2_records).to_csv("individual_assignment/output_last_day/oos_r2_overall.csv", index=False)
    pd.DataFrame(r2_by_year).to_csv("individual_assignment/output_last_day/oos_r2_by_year.csv", index=False)
    print("  Saved: individual_assignment/output_last_day/oos_r2_overall.csv, "
          "individual_assignment/output_last_day/oos_r2_by_year.csv", flush=True)

    return pred_out


# ===========================================================================
#  MAIN
# ===========================================================================

if __name__ == "__main__":
    t_start = datetime.datetime.now()
    print(f"Start: {t_start}")
    print(f"{'='*60}\n")

    DATA_DIR = os.path.join(os.path.dirname(__file__), "data_we")

    data, stock_vars, feature_vars, mkt = load_and_preprocess(DATA_DIR)

    # Ensemble: XGBoost + NN2 + Autoencoder
    pred_out = run_ensemble_stage(data, stock_vars, feature_vars)

    os.makedirs("individual_assignment/output_last_day", exist_ok=True)
    pred_out.to_csv("individual_assignment/output_last_day/predictions.csv", index=False)
    print("\nPredictions saved to individual_assignment/output_last_day/predictions.csv", flush=True)
    print("Now run build_portfolio.py to construct the trading strategy.", flush=True)

    t_end = datetime.datetime.now()
    print(f"\nTotal runtime: {t_end - t_start}")
    print("Done!")
