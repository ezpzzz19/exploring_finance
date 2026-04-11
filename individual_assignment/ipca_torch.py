"""
PyTorch-based IPCA (Instrumented Principal Component Analysis)
==============================================================
Re-implementation of Kelly, Pruitt & Su (JFE 2019) IPCA estimator using
PyTorch for all linear-algebra operations.  Runs on CPU today, moves to
GPU (CUDA / MPS) by simply changing `device`.

Follows the same Alternating Least Squares (ALS) algorithm as the
original scipy/cupy implementation in ipca_classes_update.py (IPCA_v1),
but every matrix op is a torch call.

Key outputs
-----------
- Gamma   (L x K)  : mapping from characteristics to factor loadings
- Factor  (K x T)  : latent factor returns
- Lambda  (K,)     : mean factor returns (constant risk price)
- fittedBeta (N_total x K) : time-varying betas per stock-month
- rfits   : fitted & predicted returns  +  OOS R²

Usage
-----
    from ipca_torch import IPCA
    model = IPCA(K=6, add_constant=True, device="cpu")
    results = model.fit(RZ, return_col="stock_exret",
                        OOS=True, OOS_min_periods=96)
"""

import numpy as np
import numpy.linalg as nla
import pandas as pd
import scipy.linalg as sla
import torch
from datetime import datetime
from typing import Optional, Dict, Any

# ── Linear-algebra helpers (NumPy-backed to avoid torch.linalg schema bugs) ──
# IPCA always runs on CPU with float64, so we round-trip through NumPy for the
# handful of factorisations that trigger crashes in torch ≤ 2.11 on macOS ARM.

def _np(t: torch.Tensor) -> np.ndarray:
    """Torch → NumPy (contiguous, float64)."""
    return t.detach().cpu().numpy()

def _pt(a: np.ndarray, device: torch.device) -> torch.Tensor:
    """NumPy → Torch (float64, given device)."""
    return torch.from_numpy(np.ascontiguousarray(a)).to(dtype=torch.float64, device=device)

def _lstsq(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Solve  A @ x = B  for x  (least-squares), via NumPy."""
    A_np, B_np = _np(A), _np(B)
    if B_np.ndim == 1:
        x, _, _, _ = nla.lstsq(A_np, B_np, rcond=None)
    else:
        x, _, _, _ = nla.lstsq(A_np, B_np, rcond=None)
    return _pt(x, A.device)


def _mldivide(denom: torch.Tensor, numer: torch.Tensor) -> torch.Tensor:
    """MATLAB-style  denom \\ numer."""
    return _lstsq(denom, numer)


def _mrdivide(numer: torch.Tensor, denom: torch.Tensor) -> torch.Tensor:
    """MATLAB-style  numer / denom  =  (denom' \\ numer')' ."""
    return _lstsq(denom.T, numer.T).T


def _cholesky_upper(A: torch.Tensor) -> torch.Tensor:
    """Upper-triangular Cholesky via SciPy (avoids torch schema bug)."""
    L = sla.cholesky(_np(A), lower=False)  # upper triangular
    return _pt(L, A.device)


def _svd(A: torch.Tensor, full_matrices: bool = False):
    """SVD via NumPy, returns (U, S, Vh) as torch tensors."""
    U, S, Vh = nla.svd(_np(A), full_matrices=full_matrices)
    dev = A.device
    return _pt(U, dev), _pt(S, dev), _pt(Vh, dev)


# ── Main class ───────────────────────────────────────────────────────────────

class IPCA:
    """
    PyTorch IPCA estimator with OOS expanding-window support.

    Parameters
    ----------
    K : int
        Number of latent factors to estimate.
    add_constant : bool
        If True, append a column of ones to characteristics (intercept in Gamma).
    device : str
        "cpu", "cuda", or "mps".
    min_tol : float
        Convergence tolerance for ALS.
    max_iter : int
        Maximum ALS iterations per window.
    """

    def __init__(
        self,
        K: int = 6,
        add_constant: bool = True,
        device: str = "cpu",
        min_tol: float = 1e-4,
        max_iter: int = 5000,
    ):
        self.K = K
        self.add_constant = add_constant
        # MPS doesn't support float64 — IPCA needs float64 for numerical
        # stability in ALS (Cholesky, lstsq).  Force CPU unless CUDA.
        if device == "mps" or (isinstance(device, torch.device) and device.type == "mps"):
            print("IPCA: MPS does not support float64 — using CPU instead.")
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        self.min_tol = min_tol
        self.max_iter = max_iter
        # will be populated by fit()
        self.char_names: list = []

    # ── Public API ───────────────────────────────────────────────────────

    def fit(
        self,
        RZ: pd.DataFrame,
        return_col: str = "stock_exret",
        OOS: bool = True,
        OOS_min_periods: int = 96,
        gFac: Optional[pd.DataFrame] = None,
        R_fit: bool = True,
        Beta_fit: bool = True,
        disp: bool = True,
        disp_every: int = 12,
    ) -> Dict[str, Any]:
        """
        Run IPCA estimation.

        Parameters
        ----------
        RZ : pd.DataFrame
            Multi-indexed (date, permno) DataFrame.  Must contain `return_col`
            and all characteristic columns.
        return_col : str
            Name of the return column.
        OOS : bool
            If True, use expanding-window out-of-sample estimation.
        OOS_min_periods : int
            Minimum number of months in the training window before first OOS
            prediction.
        gFac : pd.DataFrame or None
            Pre-specified factors (M x T).  Row index = factor names,
            columns = dates.  Pass a row of ones for an anomaly intercept.
        R_fit : bool
            Compute fitted returns.
        Beta_fit : bool
            Return the (stock x factor) betas.
        disp : bool
            Print progress.
        disp_every : int
            Print every N OOS months.

        Returns
        -------
        dict with keys:
            "Gamma", "Factor", "Lambda", "rfits", "fittedBeta", "numerical"
        """
        start_time = datetime.now()

        # ── Split R and Z ────────────────────────────────────────────────
        R_df = RZ[[return_col]].copy()
        Z_df = RZ.drop(columns=[return_col]).copy()
        self.char_names = list(Z_df.columns)
        dates = Z_df.index.get_level_values(0).unique().sort_values()
        T = len(dates)

        # ── Handle pre-specified factors ─────────────────────────────────
        has_gFac = gFac is not None and len(gFac) > 0
        M = gFac.shape[0] if has_gFac else 0
        KM = self.K + M
        if KM == 0:
            raise ValueError("K + M must be >= 1")

        F_names = [f"f{i}" for i in range(self.K)]
        G_names = list(gFac.index) if has_gFac else []
        Factor_names = F_names + G_names

        # ── Build X (managed portfolios) and W (char second moments) ─────
        if self.add_constant:
            char_list = self.char_names + ["Constant"]
        else:
            char_list = self.char_names
        L = len(char_list)

        # Pre-compute X_np (L, T) and W_np (L, L, T) and Nts
        X_np = np.zeros((L, T), dtype=np.float64)
        W_np = np.zeros((L, L, T), dtype=np.float64)
        Nts = np.zeros(T, dtype=np.float64)

        for ti, t in enumerate(dates):
            Zt = Z_df.loc[t].values.astype(np.float64)
            Rt = R_df.loc[t].values.astype(np.float64).ravel()
            if self.add_constant:
                ones = np.ones((Zt.shape[0], 1), dtype=np.float64)
                Zt = np.concatenate([Zt, ones], axis=1)
            Nt = Zt.shape[0]
            Nts[ti] = Nt
            X_np[:, ti] = Zt.T @ Rt / Nt
            W_np[:, :, ti] = Zt.T @ Zt / Nt

        # Move to torch
        X_t = torch.tensor(X_np, dtype=torch.float64, device=self.device)
        W_t = torch.tensor(W_np, dtype=torch.float64, device=self.device)
        Nts_t = torch.tensor(Nts, dtype=torch.float64, device=self.device)

        gFac_t = None
        if has_gFac:
            gFac_np = np.zeros((M, T), dtype=np.float64)
            for ti, t in enumerate(dates):
                gFac_np[:, ti] = gFac[t].values.astype(np.float64)
            gFac_t = torch.tensor(gFac_np, dtype=torch.float64, device=self.device)

        # ── Dispatch IS or OOS ───────────────────────────────────────────
        if not OOS:
            return self._fit_insample(
                X_t, W_t, Nts_t, gFac_t, dates, char_list, Factor_names,
                F_names, G_names, L, T, KM, M,
                Z_df, R_df, R_fit, Beta_fit, disp, start_time,
            )
        else:
            return self._fit_oos(
                X_t, W_t, Nts_t, gFac_t, dates, char_list, Factor_names,
                F_names, G_names, L, T, KM, M,
                Z_df, R_df, R_fit, Beta_fit,
                OOS_min_periods, disp, disp_every, start_time,
            )

    # ── In-sample estimation ─────────────────────────────────────────────

    def _fit_insample(
        self, X, W, Nts, gFac, dates, char_list, Factor_names,
        F_names, G_names, L, T, KM, M,
        Z_df, R_df, R_fit, Beta_fit, disp, start_time,
    ):
        K = self.K
        idx_all = list(range(T))

        Gamma, Factor = self._svd_initial(X, K, M, gFac, idx_all)

        tol, iters = float("inf"), 0
        while iters < self.max_iter and tol > self.min_tol:
            iters += 1
            Gamma1, Factor1 = self._als_step(
                Gamma.clone(), X, W, Nts, gFac, K, M, KM, L, idx_all
            )
            tol = max(
                (Gamma1 - Gamma).abs().max().item(),
                (Factor1 - Factor).abs().max().item(),
            )
            Gamma, Factor = Gamma1, Factor1

        if disp:
            print(f"IS IPCA converged in {iters} iters (tol={tol:.2e})")

        # Lambda = mean of Factor over time
        Lambda = Factor.mean(dim=1)  # (KM,)

        # ── Outputs ──────────────────────────────────────────────────────
        Gamma_np = Gamma.cpu().numpy()
        Factor_np = Factor.cpu().numpy()
        Lambda_np = Lambda.cpu().numpy()

        Gamma_df = pd.DataFrame(Gamma_np, index=char_list, columns=Factor_names)
        Factor_df = pd.DataFrame(Factor_np, index=Factor_names, columns=dates)
        Lambda_sr = pd.Series(Lambda_np, index=Factor_names)

        fittedR, fittedBeta = None, None
        if R_fit or Beta_fit:
            fittedR, fittedBeta = self._compute_fits(
                Gamma_np, Factor_np, Lambda_np, Z_df, R_df, dates,
                char_list, Factor_names, R_fit, Beta_fit,
            )

        return {
            "Gamma": Gamma_df,
            "Factor": Factor_df,
            "Lambda": Lambda_sr,
            "rfits": fittedR,
            "fittedBeta": fittedBeta,
            "numerical": {
                "iters": iters, "tol": tol,
                "time": (datetime.now() - start_time).total_seconds(),
            },
        }

    # ── Out-of-sample estimation (expanding window) ──────────────────────

    def _fit_oos(
        self, X, W, Nts, gFac, dates, char_list, Factor_names,
        F_names, G_names, L, T, KM, M,
        Z_df, R_df, R_fit, Beta_fit,
        OOS_min_periods, disp, disp_every, start_time,
    ):
        K = self.K

        # Initial guess from first OOS_min_periods months
        init_idx = list(range(OOS_min_periods))
        Gamma0, Factor0 = self._svd_initial(X, K, M, gFac, init_idx)

        # Storage
        all_Gamma = {}          # date -> (L, KM) numpy
        all_Factor = {}         # date -> (KM,) numpy
        all_Lambda = {}         # date -> (KM,) numpy
        all_Beta = {}           # date -> DataFrame(N_t x KM)
        all_fittedR_total = {}  # date -> Series(N_t)
        all_fittedR_pred = {}   # date -> Series(N_t)

        ct = 0
        for oos_ti in range(OOS_min_periods, T):
            t = dates[oos_ti]
            train_idx = list(range(oos_ti))  # expanding: 0 .. oos_ti-1

            # ALS on training data
            tol, iters = float("inf"), 0
            Gamma_iter = Gamma0.clone()
            Factor_iter = Factor0.clone() if Factor0.shape[1] == len(train_idx) else None
            if Factor_iter is None:
                Gamma_iter, Factor_iter = self._svd_initial(X, K, M, gFac, train_idx)

            while iters < self.max_iter and tol > self.min_tol:
                iters += 1
                Gamma1, Factor1 = self._als_step(
                    Gamma_iter.clone(), X, W, Nts, gFac, K, M, KM, L, train_idx
                )
                tol = max(
                    (Gamma1 - Gamma_iter).abs().max().item(),
                    (Factor1 - Factor_iter).abs().max().item(),
                )
                Gamma_iter, Factor_iter = Gamma1, Factor1

            Gamma_np = Gamma_iter.cpu().numpy()
            all_Gamma[t] = Gamma_np

            # ── NO FORWARD-LOOKING DATA ──────────────────────────────────
            # Gamma is estimated from returns & chars up to month t-1 only.
            # Z_{i,t} in our data is already lagged (chars known at t-1).
            #
            # For BETAS we only need:  beta_{i,t} = Z_{i,t} @ Gamma
            #   -> purely backward-looking ✓
            #
            # For PREDICTED RETURNS we use:
            #   E[r_{i,t+1}] = beta_{i,t} @ lambda
            #   where lambda = mean of training factors (months < t)
            #   -> purely backward-looking ✓
            #
            # We do NOT compute factor realization f_t here because
            # f_t = (Gamma' W_t Gamma)^{-1} Gamma' X_t  uses X_t which
            # contains month t returns (forward-looking for prediction).
            # Factor realizations are only useful for ex-post R² evaluation
            # and are computed separately below if needed.
            # ─────────────────────────────────────────────────────────────

            # Lambda = mean of TRAINING factors (backward-looking only)
            Lambda_np = Factor_iter.mean(dim=1).cpu().numpy()
            all_Lambda[t] = Lambda_np

            # Betas: beta_{i,t} = Z_{i,t} @ Gamma  (no future data)
            if R_fit or Beta_fit:
                Zt_raw = Z_df.loc[t].values.astype(np.float64)
                if self.add_constant:
                    Zt_aug = np.concatenate(
                        [Zt_raw, np.ones((Zt_raw.shape[0], 1))], axis=1
                    )
                else:
                    Zt_aug = Zt_raw

                Betat = Zt_aug @ Gamma_np  # (N_t, KM) — backward-looking

                if Beta_fit:
                    permnos = Z_df.loc[t].index
                    all_Beta[t] = pd.DataFrame(
                        Betat, index=permnos, columns=Factor_names
                    )

                if R_fit:
                    # Predicted return = beta @ lambda (backward-looking)
                    fitted_pred = Betat @ Lambda_np
                    permnos = R_df.loc[t].index
                    all_fittedR_pred[t] = pd.Series(fitted_pred, index=permnos)

                    # For ex-post R² we also need factor realization f_t
                    # This uses R_t (month t returns) — only for evaluation,
                    # NEVER fed back into prediction or beta computation
                    Wt = W[:, :, oos_ti]
                    xt = X[:, oos_ti]
                    if M > 0:
                        GammaF = Gamma_iter[:, :K]
                        GammaG = Gamma_iter[:, K:]
                        gf_t = gFac[:, oos_ti]
                        ft_latent = _mldivide(
                            GammaF.T @ Wt @ GammaF,
                            GammaF.T @ (xt - Wt @ GammaG @ gf_t),
                        )
                        ft_realized = torch.cat([ft_latent, gf_t])
                    else:
                        ft_realized = _mldivide(
                            Gamma_iter.T @ Wt @ Gamma_iter,
                            Gamma_iter.T @ xt,
                        )
                    fitted_total = Betat @ ft_realized.cpu().numpy()
                    all_fittedR_total[t] = pd.Series(fitted_total, index=permnos)
                    all_Factor[t] = ft_realized.cpu().numpy()

            # Warm-start next window
            # Expand Factor0 with the realized factor for warm-starting ALS
            # (this is fine: next iteration trains on data up to t, which
            #  includes R_t — and that window predicts t+1, not t)
            Gamma0 = Gamma_iter.clone()
            if R_fit and t in all_Factor:
                ft_ws = torch.tensor(all_Factor[t], dtype=torch.float64, device=self.device)
                Factor0 = torch.cat([Factor_iter, ft_ws.unsqueeze(1)], dim=1)
            else:
                # If we didn't compute factor realization, re-init next round
                Factor0 = Factor_iter

            ct += 1
            if disp and ct % disp_every == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                print(
                    f"  OOS month {ct}/{T - OOS_min_periods}  "
                    f"({t.strftime('%Y-%m')})  iters={iters}  "
                    f"tol={tol:.2e}  elapsed={elapsed:.0f}s"
                )

        # ── Assemble outputs ─────────────────────────────────────────────
        oos_dates = dates[OOS_min_periods:]

        # Factor DataFrame
        Factor_df = pd.DataFrame(
            {t: all_Factor[t] for t in oos_dates},
            index=Factor_names,
        )

        # Lambda DataFrame
        Lambda_df = pd.DataFrame(
            {t: all_Lambda[t] for t in oos_dates},
            index=Factor_names,
        )

        # Gamma — store the last one for convenience (or all)
        last_gamma = all_Gamma[oos_dates[-1]]
        Gamma_df = pd.DataFrame(last_gamma, index=char_list, columns=Factor_names)

        # fittedBeta
        fittedBeta = None
        if Beta_fit and all_Beta:
            pieces = []
            for t in oos_dates:
                if t in all_Beta:
                    df = all_Beta[t].copy()
                    df.index = pd.MultiIndex.from_product(
                        [[t], df.index], names=["date", "permno"]
                    )
                    pieces.append(df)
            fittedBeta = pd.concat(pieces)

        # fittedR
        fittedR = None
        if R_fit and all_fittedR_total:
            # Compute OOS R²
            real_parts, pred_parts, total_parts = [], [], []
            for t in oos_dates:
                if t in all_fittedR_total:
                    real_t = R_df.loc[t].values.ravel()
                    total_t = all_fittedR_total[t].values
                    pred_t = all_fittedR_pred[t].values
                    real_parts.append(real_t)
                    total_parts.append(total_t)
                    pred_parts.append(pred_t)

            real_all = np.concatenate(real_parts)
            total_all = np.concatenate(total_parts)
            pred_all = np.concatenate(pred_parts)

            sse_total = np.sum((real_all - total_all) ** 2)
            sse_pred = np.sum((real_all - pred_all) ** 2)
            sst = np.sum(real_all ** 2)  # benchmark = 0

            r2_total = 1.0 - sse_total / sst
            r2_pred = 1.0 - sse_pred / sst

            fittedR = {
                "R2_Total": r2_total,
                "R2_Pred": r2_pred,
            }

        elapsed = (datetime.now() - start_time).total_seconds()
        if disp:
            print(f"IPCA OOS done in {elapsed:.1f}s")
            if fittedR:
                print(f"  R2_Total = {fittedR['R2_Total']:.6f}")
                print(f"  R2_Pred  = {fittedR['R2_Pred']:.6f}")

        return {
            "Gamma": Gamma_df,
            "all_Gamma": all_Gamma,
            "Factor": Factor_df,
            "Lambda": Lambda_df,
            "rfits": fittedR,
            "fittedBeta": fittedBeta,
            "numerical": {
                "time": elapsed,
                "oos_dates": oos_dates,
            },
        }

    # ── Core ALS step (one iteration) ────────────────────────────────────

    def _als_step(
        self,
        Gamma: torch.Tensor,   # (L, KM)
        X: torch.Tensor,       # (L, T_full)
        W: torch.Tensor,       # (L, L, T_full)
        Nts: torch.Tensor,     # (T_full,)
        gFac: Optional[torch.Tensor],  # (M, T_full) or None
        K: int, M: int, KM: int, L: int,
        idx: list,              # which time indices to use
    ):
        """One ALS iteration.  Returns new (Gamma, Factor) tensors."""
        T2 = len(idx)

        # 1. Estimate latent factors (given Gamma)
        if K > 0:
            GammaF = Gamma[:, :K]
            FactorF = torch.zeros(K, T2, dtype=torch.float64, device=self.device)
            for ci, ti in enumerate(idx):
                Wt = W[:, :, ti]
                xt = X[:, ti]
                denom = GammaF.T @ Wt @ GammaF
                if M > 0:
                    GammaG = Gamma[:, K:]
                    gt = gFac[:, ti]
                    numer = GammaF.T @ (xt - Wt @ GammaG @ gt)
                else:
                    numer = GammaF.T @ xt
                FactorF[:, ci] = _mldivide(denom, numer)
        else:
            FactorF = None

        # Combine latent + pre-specified
        if K == KM:
            Factor = FactorF
        elif M == KM:
            Factor = gFac[:, idx]
        else:
            Factor = torch.cat([FactorF, gFac[:, idx]], dim=0)

        # 2. Estimate Gamma (given Factor)
        numer = torch.zeros(L * KM, dtype=torch.float64, device=self.device)
        denom = torch.zeros(L * KM, L * KM, dtype=torch.float64, device=self.device)
        for ci, ti in enumerate(idx):
            xt = X[:, ti]
            Wt = W[:, :, ti]
            ft = Factor[:, ci]
            nt = Nts[ti]
            numer += torch.kron(xt, ft) * nt
            denom += torch.kron(Wt, torch.outer(ft, ft)) * nt

        Gamma1_flat = _mldivide(denom, numer)
        Gamma1 = Gamma1_flat.reshape(L, KM)

        # 3. Normalization (PCA with positive-mean convention)
        if K > 0:
            Gamma1, Factor = self._normalize(Gamma1, Factor, K, M, KM, L)

        return Gamma1, Factor

    # ── SVD initial guess ────────────────────────────────────────────────

    def _svd_initial(
        self,
        X: torch.Tensor,       # (L, T_full)
        K: int, M: int,
        gFac: Optional[torch.Tensor],
        idx: list,
    ):
        """Compute SVD-based initial guess for Gamma and Factor."""
        Xsub = X[:, idx]  # (L, T2)
        if K > 0:
            U, S, Vh = _svd(Xsub, full_matrices=False)
            Gamma = U[:, :K]                          # (L, K)
            Factor = torch.diag(S[:K]) @ Vh[:K, :]    # (K, T2)

            # sign convention: positive mean
            signs = torch.sign(Factor.mean(dim=1))
            signs[signs == 0] = 1.0
            Factor = Factor * signs.unsqueeze(1)
            Gamma = Gamma * signs.unsqueeze(0)

        if M > 0 and K > 0:
            Gamma = torch.cat(
                [Gamma, torch.zeros(Gamma.shape[0], M, dtype=torch.float64, device=self.device)],
                dim=1,
            )
            Factor = torch.cat([Factor, gFac[:, idx]], dim=0)
        elif M > 0 and K == 0:
            Gamma = torch.zeros(X.shape[0], M, dtype=torch.float64, device=self.device)
            Factor = gFac[:, idx]

        return Gamma, Factor

    # ── Normalization ────────────────────────────────────────────────────

    def _normalize(self, Gamma, Factor, K, M, KM, L):
        """PCA normalization: Gamma orthonormal, Factor orthogonal, positive mean."""
        if K == KM:
            GammaF, FactorF = Gamma, Factor
            GammaG, FactorG = None, None
        else:
            GammaF = Gamma[:, :K]
            GammaG = Gamma[:, K:]
            FactorF = Factor[:K, :]
            FactorG = Factor[K:, :]

        # Cholesky of GammaF' GammaF → upper triangular via SciPy
        R1 = _cholesky_upper(GammaF.T @ GammaF)

        R2_mat = R1 @ FactorF @ FactorF.T @ R1.T
        U, _, _ = _svd(R2_mat)
        R2 = U

        GammaF = _mrdivide(GammaF, R1) @ R2
        FactorF = _mldivide(R2, R1 @ FactorF)

        # Sign convention
        signs = torch.sign(FactorF.mean(dim=1))
        signs[signs == 0] = 1.0
        FactorF = FactorF * signs.unsqueeze(1)
        GammaF = GammaF * signs.unsqueeze(0)

        # Orthogonality between GammaF and GammaG
        if M > 0:
            eye = torch.eye(L, dtype=torch.float64, device=self.device)
            GammaG = (eye - GammaF @ GammaF.T) @ GammaG
            FactorF = FactorF + (GammaF.T @ GammaG) @ FactorG
            # Re-apply sign convention
            signs = torch.sign(FactorF.mean(dim=1))
            signs[signs == 0] = 1.0
            FactorF = FactorF * signs.unsqueeze(1)
            GammaF = GammaF * signs.unsqueeze(0)

        if M > 0:
            Gamma_out = torch.cat([GammaF, GammaG], dim=1)
            Factor_out = torch.cat([FactorF, FactorG], dim=0)
        else:
            Gamma_out = GammaF
            Factor_out = FactorF

        return Gamma_out, Factor_out

    # ── Compute betas and fitted returns from numpy arrays ───────────────

    def _compute_fits(
        self, Gamma_np, Factor_np, Lambda_np,
        Z_df, R_df, dates, char_list, Factor_names,
        R_fit, Beta_fit,
    ):
        """Compute in-sample betas & fitted returns."""
        all_beta = []
        all_fitted_total = []
        all_fitted_pred = []
        all_real = []

        for ti, t in enumerate(dates):
            Zt_raw = Z_df.loc[t].values.astype(np.float64)
            if self.add_constant:
                Zt_aug = np.concatenate(
                    [Zt_raw, np.ones((Zt_raw.shape[0], 1))], axis=1
                )
            else:
                Zt_aug = Zt_raw

            Betat = Zt_aug @ Gamma_np  # (N_t, KM)
            ft = Factor_np[:, ti]

            if Beta_fit:
                permnos = Z_df.loc[t].index
                beta_df = pd.DataFrame(Betat, index=permnos, columns=Factor_names)
                beta_df.index = pd.MultiIndex.from_product(
                    [[t], permnos], names=["date", "permno"]
                )
                all_beta.append(beta_df)

            if R_fit:
                all_fitted_total.append(Betat @ ft)
                all_fitted_pred.append(Betat @ Lambda_np)
                all_real.append(R_df.loc[t].values.ravel())

        fittedBeta = pd.concat(all_beta) if Beta_fit else None

        fittedR = None
        if R_fit:
            real = np.concatenate(all_real)
            total = np.concatenate(all_fitted_total)
            pred = np.concatenate(all_fitted_pred)
            sst = np.sum(real ** 2)
            fittedR = {
                "R2_Total": 1.0 - np.sum((real - total) ** 2) / sst,
                "R2_Pred": 1.0 - np.sum((real - pred) ** 2) / sst,
            }

        return fittedR, fittedBeta
