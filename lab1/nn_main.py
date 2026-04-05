import datetime
import gc
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import logging
from logging import getLogger

logging.basicConfig(level=logging.INFO)
logger = getLogger(__name__)

seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


# ── Model definitions (NN1 through NN5) ──────────────────────────────────────

class NN1(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        return torch.squeeze(self.fc2(x))


class NN2(nn.Module):
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

_NN_CLASSES = {1: NN1, 2: NN2}
    

# ── Training helpers ──────────────────────────────────────────────────────────

def train_nn(model, criterion, train_loader, optimizer, device):
    model.train()
    torch.set_grad_enabled(True)
    total_loss = 0
    for predictors, target in train_loader:
        predictors, target = predictors.to(device), target.to(device)
        optimizer.zero_grad()
        prediction = model(predictors)
        loss = criterion(prediction, target)
        # L1 regularization on input layer weights only
        loss += 0.0001 * torch.norm(model.fc1.weight, p=1)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(predictors)
    return total_loss / len(train_loader.dataset)


def evaluate_nn(model, criterion, eval_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for predictors, target in eval_loader:
            predictors, target = predictors.to(device), target.to(device)
            prediction = model(predictors)
            loss = criterion(prediction, target)
            total_loss += loss.item() * len(predictors)
    return total_loss / len(eval_loader.dataset)


def learning_loop(ensemble, layers, input_size, output_size, batch_size,
                  num_epoch, train_dataset, val_dataset, patience,
                  X_test_tensor, device, work_dir):
    """Train `ensemble` models of `layers` hidden layers; return mean OOS predictions."""
    mean_pred = np.zeros((output_size, ensemble))

    for i in range(ensemble):
        model = _NN_CLASSES[layers](input_size).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        # lr decays: 0.01 * 0.955^100 ≈ 0.0001
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.955)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False)

        ckpt_path = os.path.join(work_dir, f'best_model_{layers}_{i}.pt')
        min_val_loss = float('inf')
        num_no_improve = 0

        for epoch in range(num_epoch):
            train_nn(model, criterion, train_loader, optimizer, device)
            val_loss = evaluate_nn(model, criterion, val_loader, device)

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                num_no_improve = 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                num_no_improve += 1
            if num_no_improve >= patience:
                break

            scheduler.step()

        # weights_only=True required by modern PyTorch (suppresses FutureWarning)
        model.load_state_dict(torch.load(ckpt_path, weights_only=True))
        model.eval()
        with torch.no_grad():
            mean_pred[:, i] = model(X_test_tensor).cpu().numpy()

        # clean up checkpoint and model to save memory between ensemble members
        os.remove(ckpt_path)
        del model, optimizer, scheduler, train_loader, val_loader
        gc.collect()

    return mean_pred.mean(axis=1)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f'PyTorch {torch.__version__} | device: {device}')
    logger.info(datetime.datetime.now())

    pd.set_option('mode.chained_assignment', None)

    work_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

    # ── load data ─────────────────────────────────────────────────────────────
    file_path = os.path.join(work_dir, 'homework_sample_big.csv')
    raw = pd.read_csv(file_path, parse_dates=['date'], low_memory=False)
    raw = raw.drop(columns=['date'])
    # build date column without fragmenting: compute externally then assign once
    raw = raw.copy()  # defragment the DataFrame first
    raw['date'] = pd.to_datetime(
        raw['year'].astype(str) + '-' + raw['month'].astype(str) + '-01'
    )

    file_path = os.path.join(work_dir, 'macro_data_goyal.csv')
    macro = pd.read_csv(file_path, parse_dates=['date1'])
    macro_list = list(macro.columns)[2:]

    file_path = os.path.join(work_dir, 'factors_char_list.csv')
    var_list = [c for c in pd.read_csv(file_path)['variable'].values if c in raw.columns]
    ind_list = [c for c in ('ind' + str(i) for i in range(1, 69)) if c in raw.columns]

    for ret_var, sp_only in [('stock_exret', 0)]:
        new_set = raw[raw[ret_var].notna()].copy()

        if sp_only == 1 and 'sp_ind' in new_set.columns:
            new_set = new_set[new_set['sp_ind'] == 1]

        if ret_var == 'strad_ret' and 'strad_baspread' in new_set.columns:
            new_set = new_set.rename(columns={'strad_baspread': 'opt_baspread'})
        elif 'call_baspread' in new_set.columns:
            new_set = new_set.rename(columns={'call_baspread': 'opt_baspread'})

        # ── rank-transform each variable within each month ────────────────────
        adj_chunks = []
        for date, group in new_set.groupby('date'):
            group = group.copy()
            for var in var_list:
                if var not in group.columns:
                    continue
                group[var] = group[var].fillna(group[var].median(skipna=True))
                # rank to [-1, 1]
                group[var] = group[var].rank(method='dense') - 1
                mx = group[var].max()
                group[var] = (group[var] / mx * 2 - 1) if mx > 0 else 0.0
            adj_chunks.append(group)

        adj_raw = pd.concat(adj_chunks, ignore_index=True)
        del adj_chunks, new_set
        gc.collect()

        # ── create macro interactions ─────────────────────────────────────────
        data_chunks = []
        for date, group in adj_raw.groupby('date'):
            expand = group.copy()
            macro_values = macro[macro['date1'] == date]
            if macro_values.empty:
                macro_values = macro[macro['date1'] <= date].sort_values('date1').iloc[-1:]
            if macro_values.empty:
                macro_values = macro.iloc[:1]
            for macro_var in macro_list:
                macro_val = macro_values[macro_var].values[0]
                interaction = group[var_list] * macro_val
                interaction = interaction.rename(
                    columns={v: f'{v}_{macro_var}' for v in var_list})
                expand = pd.concat([expand, interaction], axis=1)
            data_chunks.append(expand)

        data = pd.concat(data_chunks, ignore_index=True)
        del data_chunks, adj_raw
        gc.collect()

        # ── rolling window setup ──────────────────────────────────────────────
        starting  = pd.to_datetime('20050101', format='%Y%m%d')
        end_date  = pd.to_datetime('20230101', format='%Y%m%d')
        counter   = 0
        pred_out  = pd.DataFrame()

        full_var_list  = var_list + [f'{v}_{m}' for m in macro_list for v in var_list]
        final_col_names = full_var_list + ind_list

        # assignment says ensemble=1 for NN; run only NN2 as required by Module 2
        batch_size = 10000
        num_epoch  = 100
        patience   = 5
        ensemble   = 1
        nn_layers  = [2]   # NN2 only; extend to [1,2,3,4,5] if desired

        while (starting + pd.DateOffset(years=13)) <= end_date:
            logger.info(f'Window starting {starting.date()} | counter={counter}')
            cutoff = [starting,
                      starting + pd.DateOffset(years=10),
                      starting + pd.DateOffset(years=12),
                      starting + pd.DateOffset(years=13)]

            train    = data[(data['date'] >= cutoff[0]) & (data['date'] < cutoff[1])].copy()
            validate = data[(data['date'] >= cutoff[1]) & (data['date'] < cutoff[2])].copy()
            test     = data[(data['date'] >= cutoff[2]) & (data['date'] < cutoff[3])].copy()

            scaler = StandardScaler().fit(train[full_var_list])
            train[full_var_list]    = scaler.transform(train[full_var_list])
            validate[full_var_list] = scaler.transform(validate[full_var_list])
            test[full_var_list]     = scaler.transform(test[full_var_list])

            X_train = train[final_col_names].values.astype(np.float32)
            Y_train = train[ret_var].values.astype(np.float32)
            X_val   = validate[final_col_names].values.astype(np.float32)
            Y_val   = validate[ret_var].values.astype(np.float32)
            X_test  = test[final_col_names].values.astype(np.float32)
            Y_test  = test[ret_var].values.astype(np.float32)

            del train, validate
            gc.collect()

            Y_mean     = float(Y_train.mean())
            Y_train_dm = Y_train - Y_mean
            Y_val_dm   = Y_val   - Y_mean

            # prepare output dataframe
            rid = ['year', 'month', 'date']
            if 'secid'  in test.columns: rid.append('secid')
            rid.append('PERMNO' if 'PERMNO' in test.columns else 'permno')
            rid.append(ret_var)
            reg_pred = test[rid].copy()
            del test
            gc.collect()

            # OLS baseline
            reg = LinearRegression(fit_intercept=False)
            reg.fit(X_train, Y_train_dm)
            reg_pred['ols'] = reg.predict(X_test) + Y_mean
            del reg

            # build tensor datasets once, reuse across NN depths
            train_dataset = TensorDataset(
                torch.from_numpy(X_train), torch.from_numpy(Y_train_dm))
            val_dataset = TensorDataset(
                torch.from_numpy(X_val), torch.from_numpy(Y_val_dm))
            X_test_tensor = torch.from_numpy(X_test).to(device)

            input_size  = len(final_col_names)
            output_size = len(Y_test)

            for layers in nn_layers:
                logger.info(f'Training NN{layers} (ensemble={ensemble})')
                preds = learning_loop(
                    ensemble, layers, input_size, output_size,
                    batch_size, num_epoch, train_dataset, val_dataset,
                    patience, X_test_tensor, device, work_dir,
                )
                reg_pred[f'nn{layers}'] = preds + Y_mean

            del train_dataset, val_dataset, X_test_tensor
            del X_train, Y_train, Y_train_dm, X_val, Y_val, Y_val_dm, X_test, Y_test
            gc.collect()

            pred_out = pd.concat([pred_out, reg_pred], ignore_index=True)

            starting += pd.DateOffset(years=1)
            counter  += 1

        # ── save predictions ──────────────────────────────────────────────────
        out_dir  = os.path.join(work_dir, 'predicted')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'nn_all_{ret_var}_{sp_only}.csv')
        pred_out.to_csv(out_path, index=False)
        logger.info(f'Saved: {out_path}')

        # ── OOS R² ───────────────────────────────────────────────────────────
        yreal = pred_out[ret_var].values
        for model_name in ['ols'] + [f'nn{l}' for l in nn_layers]:
            ypred = pred_out[model_name].values
            r2 = 1 - np.sum(np.square(yreal - ypred)) / np.sum(np.square(yreal))
            logger.info(f'{model_name} OOS R²: {r2:.6f}')

        # ── OOS R² per year ──────────────────────────────────────────────────
        r2_dateframe = pd.DataFrame(columns=['year', 'model', 'r2'])
        for model_name in ['ols'] + [f'nn{l}' for l in nn_layers]:
            logger.info(f'--- {model_name} OOS R² by year ---')
            for year, grp in pred_out.groupby('year'):
                yreal_y = grp[ret_var].values
                ypred_y = grp[model_name].values
                denom = np.sum(np.square(yreal_y))
                r2_y = 1 - np.sum(np.square(yreal_y - ypred_y)) / denom if denom > 0 else float('nan')
                r2_dateframe = pd.concat([r2_dateframe, pd.DataFrame({'year': year, 'model': model_name, 'r2': r2_y}, index=[0])], ignore_index=True)

        filename = os.path.join(out_dir, f'r2_nn_all_{ret_var}_{sp_only}.csv')
        r2_dateframe.to_csv(filename, index=False)
        logger.info(f'Saved: {filename}')

        # ── OOS R² per year chart ─────────────────────────────────────────────
        models_in_plot = ['ols'] + [f'nn{l}' for l in nn_layers]
        years = sorted(r2_dateframe['year'].unique())
        x = np.arange(len(years))
        bar_w = 0.8 / len(models_in_plot)
        colors_map = {'ols': 'steelblue', 'nn1': 'darkorange', 'nn2': 'seagreen',
                      'nn3': 'tomato', 'nn4': 'mediumpurple', 'nn5': 'sienna'}

        fig, ax = plt.subplots(figsize=(max(12, len(years) * 0.8), 5))
        for idx, m in enumerate(models_in_plot):
            vals = [
                r2_dateframe.loc[(r2_dateframe['year'] == y) & (r2_dateframe['model'] == m), 'r2'].values[0]
                for y in years
            ]
            offset = (idx - (len(models_in_plot) - 1) / 2) * bar_w
            bars = ax.bar(x + offset, vals, width=bar_w,
                          label=m.upper(), color=colors_map.get(m, None),
                          alpha=0.85, edgecolor='white', linewidth=0.5)

        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_xticks(x)
        ax.set_xticklabels(years, rotation=45, ha='right')
        ax.set_xlabel('Year')
        ax.set_ylabel('OOS R²')
        ax.set_title(f'Out-of-Sample R² by Year — {ret_var} (sp_only={sp_only})')
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        chart_path = os.path.join(out_dir, f'r2_by_year_{ret_var}_{sp_only}.png')
        plt.savefig(chart_path, dpi=150)
        plt.close(fig)
        logger.info(f'Saved chart: {chart_path}')

        logger.info(datetime.datetime.now())
