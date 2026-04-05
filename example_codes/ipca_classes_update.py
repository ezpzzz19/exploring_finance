import pandas as pd
import numpy as np
import scipy.linalg as sla
import scipy.sparse.linalg as ssla
import copy
from datetime import datetime
from timeit import default_timer as timer

# matrix left/right division (following MATLAB function naming)
_mldivide = lambda denom, numer: sla.lstsq(np.array(denom), np.array(numer))[0]
_mrdivide = lambda numer, denom: (sla.lstsq(np.array(denom).T, np.array(numer).T)[0]).T


def _sign_convention(gamma, fac):
    """
    sign the latent factors to have positive mean, and sign gamma accordingly
    """
    sign_conv = fac.mean(axis=1).apply(lambda x: 1 if x >= 0 else -1)
    return gamma.mul(sign_conv.values, axis=1), fac.mul(sign_conv.values, axis=0)


def _calc_r2(r_act, r_fit):
    """
    compute r2 of fitted values vs actual
    """
    sumsq = lambda x: x.dot(x)
    sse = sum(sumsq(r_act[t] - r_fit[t]) for t in r_fit.keys())
    sst = sum(sumsq(r_act[t]) for t in r_fit.keys())
    return 1.0 - sse / sst


class IPCA_v0(object):
    def __init__(self, Z, R=None, X=None, K=0, gFac=None):
        """
        [Dimensions]
            N: the number of assets
            T: the number of time periods
            L: the number of characteristics
            K: the number of latent factors
            M: the number of pre-specified factors (plus anomaly term)

        [Inputs]
            Z (dict(T) of df(NxL)): characteristics; can be rank-demeaned
            R (dict(T) of srs(N); not needed for managed-ptf-only version): asset returns
            X (df(LxT); only needed for managed-ptf-only version): managed portfolio returns
            K (const; optional): number of latent factors
            gFac (df(MxT); optional): Anomaly term ([1,...,1]), or Pre-Specified Factors (i.e. returns of HML, SMB, etc.)

        * IPCA can be run with only K > 0 or only gFac
        * IMPORTANT: this structure and the code supposes that lagging has already occurred.
          i.e. If t is March 2003, monthly data, then R[t] are the returnss realized at the end of March 2003 during March 2003,
          and Z[t] are the characteristics known at the end of February 2003.

        [Transformed Inputs]
            N_valid (srs(T)): number of nonmissing obs each period, where a missing obs is any asset with missing return or any missing characteristic
            X (df(LxT)): managed portfolio returns: X[t] = Z[t][valid].T * R[t][valid] / N_valid[t]
            W (dict(T) of df(LxL)): characteristic second moments: W[t] = Z[t][valid].T * Z[t][valid].T / N_valid(t)

        [Outputs]
        calculated in run_ipca method:
            Gamma (df(Lx(K+M))): gamma estimate (fGamma for latent, gGamma for pre-specified)
            Fac (df((K+M)xT)): factor return estimate (fFac for latent, gFac for pre-specified)
            Lambd (srs(K+M)): mean of Fac (fLambd for latent, gLambd for pre-specified)
        calculated in fit method:
            fitvals (dict(4) of dict(T) of srs(N)): fitted values of asset returns; 4 versions: {constant risk price, dynamic risk price} x {assets, managed-ptfs}
            r2 (srs(4)): r-squared of the four versions of fitted values against actual values
        """
        # type of model
        self.X_only = True if R is None else False  # managed-ptf-only version
        self.has_latent = True if K else False
        self.has_prespec = True if (gFac is not None and len(gFac) > 0) else False

        # inputs
        self.Z, self.R, self.X = Z, R, X
        self.times = sorted(Z.keys())
        self.charas = Z[list(Z.keys())[0]].columns
        self.gFac = gFac if self.has_prespec else pd.DataFrame(columns=self.times)
        self.gLambd = self.gFac.mean(axis=1)
        self.fIdx, self.gIdx = [str(x) for x in range(1, K + 1)], list(self.gFac.index)
        self.K, self.M, self.L, self.T = (
            K,
            len(self.gIdx),
            len(self.charas),
            len(self.times),
        )

        # transformation inputs
        self.N_valid = pd.Series(index=self.times, dtype="float64")
        if not self.X_only:
            self.X = pd.DataFrame(index=self.charas, columns=self.times)
        self.W = {
            t: pd.DataFrame(index=self.charas, columns=self.charas) for t in self.times
        }
        for t in self.times:
            is_valid = pd.DataFrame(
                {"z": self.Z[t].notnull().all(axis=1), "r": self.R[t].notnull()}
            ).all(
                axis=1
            )  # not valid if ret or any charas are missing
            z_valid = self.Z[t].loc[is_valid.values, :]
            r_valid = self.R[t].loc[is_valid.values]
            self.N_valid[t] = (1.0 * is_valid).sum()
            if not self.X_only:
                self.X[t] = z_valid.T.dot(r_valid) / self.N_valid[t]
            self.W[t] = z_valid.T.dot(z_valid) / self.N_valid[t]

        # outputs
        self.Gamma, self.fGamma, self.gGamma = None, None, None
        self.Fac, self.fFac = None, None
        self.Lambd, self.fLambd = None, None
        self.fitvals, self.r2 = {}, pd.Series(dtype="float64")

    def run_ipca(self, fit=True, dispIters=False, MinTol=1e-6, MaxIter=5000):
        """
        Computes Gamma, Fac and Lambd

        [Inputs]
        fit (bool): whether to compute fitted returns and r-squared after params are estimated
        dispIters (bool): whether to display results of each iteration
        MinTol (float): tolerance for convergence
        MaxIter (int): max number of iterations

        [Outputs]
        Gamma (df(Lx(K+M))): gamma estimate (fGamma for latent, gGamma for pre-specified)
        Fac (df((K+M)xT)): factor return estimate (fFac for latent, gFac for pre-specified)
        Lambd (srs(K+M)): mean of Fac (fLambd for latent, gLambd for pre-specified)

        * When characteristics are rank-demeaned and returns are used in units (ie 0.01 is a 1% return),
          1e-6 tends to be a good convergence criterion.
          This is because the convergence of the algorithm mostly comes from GammaBeta being stable,
          and 1e-6 is small given that GammaBeta is always rotated to be orthonormal.
        """
        # initial guess
        Gamma0 = GammaDelta0 = pd.DataFrame(0.0, index=self.charas, columns=self.gIdx)
        if self.has_latent:
            svU, svS, svV = ssla.svds(self.X.values, self.K)
            svU, svS, svV = (
                np.fliplr(svU),
                svS[::-1],
                np.flipud(svV),
            )  # reverse order to match MATLAB svds output
            fFac0 = pd.DataFrame(
                np.diag(svS).dot(svV), index=self.fIdx, columns=self.times
            )  # first K PC of X
            GammaBeta0 = pd.DataFrame(
                svU, index=self.charas, columns=self.fIdx
            )  # first K eigvec of X
            GammaBeta0, fFac0 = _sign_convention(GammaBeta0, fFac0)
            Gamma0 = pd.concat([GammaBeta0, GammaDelta0], axis=1)

        # ALS estimate
        tol, iter = float("inf"), 0
        while iter < MaxIter and tol > MinTol:
            Gamma1, fFac1 = self._ipca_als_estimation(Gamma0)
            tol_Gamma = abs(Gamma1 - Gamma0).values.max()
            tol_fFac = abs(fFac1 - fFac0).values.max() if self.has_latent else None
            tol = max(tol_Gamma, tol_fFac)

            if dispIters:
                print(
                    "iter {}: tolGamma = {} and tolFac = {}".format(
                        iter, tol_Gamma, tol_fFac
                    )
                )

            Gamma0, fFac0 = Gamma1, fFac1
            iter += 1

        self.Gamma, self.fGamma, self.gGamma = (
            Gamma1,
            Gamma1[self.fIdx],
            Gamma1[self.gIdx],
        )
        self.Fac, self.fFac = pd.concat([fFac1, self.gFac]), fFac1
        self.Lambd, self.fLambd = self.Fac.mean(axis=1), self.fFac.mean(axis=1)

        if fit:  # default to automatically compute fitted values
            self.fit()

    def _ipca_als_estimation(self, Gamma0):
        """
        Runs one iteration of the alternating least squares estimation process

        [Inputs]
        Gamma0 (df(Lx(K+M))): previous iteration's Gamma estimate

        [Outputs]
        Gamma1 (df(Lx(K+M))): current iteration's Gamma estimate
        fFac1 (df(KxT)): current iteration's latent Factor estimate

        * Imposes identification assumption on Gamma1 and fFac1:
          Gamma1 is orthonormal matrix and fFac1 orthogonal with positive mean (taken across times)

        """
        # 1. estimate latent factor
        fFac1 = pd.DataFrame(index=self.fIdx, columns=self.times)
        if self.has_latent:
            GammaBeta0, GammaDelta0 = Gamma0[self.fIdx], Gamma0[self.gIdx]
            for t in self.times:
                numer = GammaBeta0.T.dot(self.X[t])
                if self.has_prespec:
                    numer -= (
                        GammaBeta0.T.dot(self.W[t]).dot(GammaDelta0).dot(self.gFac[t])
                    )
                denom = GammaBeta0.T.dot(self.W[t]).dot(GammaBeta0)
                fFac1[t] = pd.Series(_mldivide(denom, numer), index=self.fIdx)

        # 2. estimate gamma
        vec_len = self.L * (self.K + self.M)
        numer, denom = np.zeros(vec_len), np.zeros((vec_len, vec_len))
        for t in self.times:
            Fac = pd.concat([fFac1[t], self.gFac[t]])
            FacOutProd = np.outer(Fac, Fac)
            numer += np.kron(self.X[t], Fac) * self.N_valid[t]
            denom += (
                np.kron(self.W[t], FacOutProd) * self.N_valid[t]
            )  # this line takes most of the time
        Gamma1_tmp = np.reshape(_mldivide(denom, numer), (self.L, self.K + self.M))
        Gamma1 = pd.DataFrame(
            Gamma1_tmp, index=self.charas, columns=self.fIdx + self.gIdx
        )

        # 3. identification assumption
        if self.has_latent:  # GammaBeta orthonormal and fFac1 orthogonal
            GammaBeta1, GammaDelta1 = Gamma1[self.fIdx], Gamma1[self.gIdx]

            R1 = sla.cholesky(GammaBeta1.T.dot(GammaBeta1))
            R2, _, _ = sla.svd(R1.dot(fFac1).dot(fFac1.T).dot(R1.T))
            GammaBeta1 = pd.DataFrame(
                _mrdivide(GammaBeta1, R1).dot(R2), index=self.charas, columns=self.fIdx
            )
            fFac1 = pd.DataFrame(
                _mldivide(R2, R1.dot(fFac1)), index=self.fIdx, columns=self.times
            )
            GammaBeta1, fFac1 = _sign_convention(GammaBeta1, fFac1)

            if self.has_prespec:  # orthogonality between GammaBeta and GammaDelta
                GammaDelta1 = (np.identity(self.L) - GammaBeta1.dot(GammaBeta1.T)).dot(
                    GammaDelta1
                )
                fFac1 += GammaBeta1.T.dot(GammaDelta1).dot(
                    self.gFac
                )  # (K x M reg coef) * gFac
                GammaBeta1, fFac1 = _sign_convention(GammaBeta1, fFac1)

            Gamma1 = pd.concat([GammaBeta1, GammaDelta1], axis=1)
        return Gamma1, fFac1

    def fit(self):
        """
        Computes fitted values and their associated r-squared

        [Inputs]
        Assumes the run_ipca was already run

        [Outputs]
        fitvals (dict(4) of dict(T) of srs(N)): fitted values of asset returns; 4 versions: (constant vs dynamic risk prices) x (assets vs managed-ptfs)
        r2 (srs(4)): r-squared of the four versions of fitted values against actual values

        * Dynamic Risk Price -> F
          Constant Risk Price -> Lambda
        """
        if not self.X_only:
            self.fitvals["R_DRP"] = {
                t: self.Z[t].dot(self.Gamma).dot(self.Fac[t]) for t in self.times
            }
            self.fitvals["R_CRP"] = {
                t: self.Z[t].dot(self.Gamma).dot(self.Lambd) for t in self.times
            }
            self.r2["R_Tot"] = _calc_r2(self.R, self.fitvals["R_DRP"])
            self.r2["R_Prd"] = _calc_r2(self.R, self.fitvals["R_CRP"])

        self.fitvals["X_DRP"] = {
            t: self.W[t].dot(self.Gamma).dot(self.Fac[t]) for t in self.times
        }
        self.fitvals["X_CRP"] = {
            t: self.W[t].dot(self.Gamma).dot(self.Lambd) for t in self.times
        }
        self.r2["X_Tot"] = _calc_r2(self.X, self.fitvals["X_DRP"])
        self.r2["X_Prd"] = _calc_r2(self.X, self.fitvals["X_CRP"])


class IPCA_v1(object):
    def __init__(
        self, RZ=None, return_column=0, X=None, W=None, Nts=None, add_constant=True
    ):
        """
        [Inputs]
        RZ : df(TotalObs x L0+1) : returns and characteristics in a df with a 2-level MultiIndex index where Date is level 0 and
        AssetID is level 1. In a balanced dataset, TotalObs would equal the number of unique AssetIDs times the number
        of unique Dates, T; however, data is generally unbalanced and therefore the number of AssetIDs present varies
        over Dates. We split RZ into R df(TotalObs x 1) and Z df(TotalObs x L0). The
        code assumes that Z and R have been properly timed outside of IPCA, such that the rows of R with Date=d
        correspond to the rows of Z with Date=d according to the IPCA model in mind. E.g. the conditional APT of
        Kelly, Pruitt, Su (JFE 2019) says that the characteristics known at Date=d-1 determine the exposures associated
        with the returns realized at Date=d; hence, here we should have shifted the characteristics in Z relative to
        the returns in R. L0 is number of characteristics, and the unique CharNames are the 1-level column index of df Z.
            RZ is optional because we might use this code by passing in only the cross-sectional second moment df W,
            described below. E.g. when using the code for bootstrapped tests.

        return_column : int or str : which column of RZ, if given, is the return; all others are assumed to be chars.
            Default is 0: the first column. An int can be supplied to determine which column, or a column name str

        X : df(L x T) : Managed portfolio returns in a df with an index of CharNames (plus perhaps constant) and columns of the unique Dates
            X is optional because we might pass in R and Z directly and create X.
            See add_constant input description about dimension L.

        W : df(TL x L) : Cross-sectional second moments of characteristics, for the T unique Dates.
            W is optional because we might pass in Z directly and create W.
            See add_constant input description about dimension L.

        Nts : srs(T) : number of available assets for each Date
            Nts is optional because we can construct it from Z and R, if those are passed. Otherwise it is only
            useful if we are passing X and W, but not Z and R

        add_constant : bool : True (default) adds a constant to the calculation of X, W and it becomes the last char listed in
        in their indices. If true, then L = L0+1; if false, then L=L0

        [Transformed Inputs]
        X and W and Nts : df(L x T) and df(TL x L) and srs(T) : if Z,R input, these are created
        """

        # inputs
        self.RZ, self.X, self.W, self.Nts = RZ, X, W, Nts
        self.has_RZ = False if RZ is None else True
        self.has_X = False if X is None else True
        self.has_W = False if W is None else True

        # check for Nts if X,W passed but not RZ
        if self.has_X and self.has_W and not self.has_RZ and self.Nts == None:
            raise ValueError("RZ are not passed; X,W were passed with no Nts")

        if self.has_RZ:
            if isinstance(return_column, int):  # pick return column using iloc
                if return_column >= RZ.shape[1]:
                    raise ValueError(
                        "The return_column int is greater than the number of columns in RZ"
                    )
                self.R = RZ.iloc[:, return_column].to_frame()
            elif isinstance(return_column, str):  # pick return column using loc
                self.R = RZ.loc[:, return_column].to_frame()
            elif (
                type(return_column) is tuple
            ):  # pick return column using loc, with multiindex
                self.R = RZ.loc[:, return_column].to_frame()
            else:
                raise ValueError("Did not know how to pick return column")
            self.Z = RZ.drop(columns=self.R.columns)

        # create X and W if not provided
        if not self.has_X or not self.has_W:
            # the characteristics used for X, W (the index names of X df, column and index of each df in the W dict)
            if add_constant:
                charlist = list(self.Z.columns) + ["Constant"]
            else:
                charlist = list(self.Z.columns)

            # the unique dates used for X, W (the column names of X df, key names of W dict)
            datelist = self.Z.index.get_level_values(0).unique()

            if not self.has_X:
                self.X = pd.DataFrame(index=charlist, columns=datelist, data=np.nan)
            if not self.has_W:
                self.W = pd.DataFrame(
                    data=0.0,
                    index=pd.MultiIndex.from_product(
                        (datelist, charlist), names=["date", "char"]
                    ),
                    columns=charlist,
                )
            # ignoring a Nts if given
            self.Nts = pd.Series(index=datelist, data=np.nan)

            # create X and W
            for t in datelist:
                Zt = self.Z.loc[t, :].values
                if add_constant:
                    Zt = np.concatenate((Zt, np.ones((Zt.shape[0], 1))), axis=1)
                self.Nts[t] = Zt.shape[0]
                if not self.has_X:
                    self.X[t] = Zt.T.dot(self.R.loc[t]) / self.Nts[t]
                if not self.has_W:
                    self.W.loc[t] = Zt.T.dot(Zt) / self.Nts[t]

        # set properties
        self.Chars, self.Dates, self.L, self.add_constant = (
            self.X.index,
            self.X.columns,
            self.X.shape[0],
            add_constant,
        )
        # create ndarrays of X and W to speed up estimation
        self._X = self.X.values
        self._W = np.zeros((self.L, self.L, self.X.shape[1]), dtype="float32")
        ct = 0
        for t in self.Dates:
            self._W[:, :, ct] = self.W.loc[t].values
            ct += 1

    def fit(
        self,
        K=1,
        OOS=False,
        gFac=None,
        normalization_choice="PCA_positivemean",
        normalization_choice_specs=None,
        OOS_window="recursive",
        OOS_window_specs=60,
        factor_mean="constant",
        R_fit=True,
        Beta_fit=False,
        dispIters=False,
        minTol=1e-4,
        maxIters=5000,
        F_names=None,
        G_names=None,
        R2_bench="zero",
        dispItersInt=100,
    ):
        """
        [Inputs]

        self : initialized IPCA object

        K : int : number of latent factors. We define KM = K + M and KM must be at least 1

        OOS : bool : false (default); True selects out-of-sample estimation, using other options listed below with
        prefix "OOS_"; the default False does in-sample estimation

        gFac : df(M x T) :  pre-specified factors or an anomaly term which is constant for all Dates
        gFac is optional because we might include no pre-specified factors or anomaly term

        normalization_choice : str : the normalization scheme for Factors and Gamma
            'PCA_positivemean' (default) mimics the typical PCA normalization (as in Kelly, Pruitt, Su JFE 2019) where
                - Gamma is unitary matrix
                - Factors are orthogonal to each other
                - Factor means are non-negative

            'Identity' enforces that each of K of the L characteristics have a loading of 1 on one of the factors and
            loadings of 0 on the remaining factors. This must be accompanied by K-list in normalization_choice_specs
            which determines which characteristics define this normalization. The ordering of the normalized factors
            follows the ordering of characteristics in that K-list

        normalization_choice_specs : list of str : if normalization_choice='Identity', then this K-list defines which
        characteristics have unit loading on one factor and zero loading on all others, using the characteristic names

        OOS_window : str : determines training window choice
            'recursive' (default) is an expanding window using data from the beginning
            'rolling' is a rolling window of fixed length

        OOS_window_specs : int : number defining the OOS_window specification
            if OOS_window='recursive', this is the minimum number of time periods required to construct estimates
            if OOS_window='rolling', this is the window length
            Default is 60

        factor_mean : str : choice of factor mean to use
            'constant' (default) uses the mean of the Factors on the training data
            'VAR1' estimates a VAR(1) on the training data and uses it to construct conditional mean estimates for a
            time-varying lambda

        R_fit : bool : if True then calculate the R fitted values (only possible if Z and R were input). If
        estimation is out-of-sample, then these are the out-of-sample fits

        Beta_fit : bool : if True then return the Betas (only possible if Z was input). If estimation is
        out-of-sample, then these are the Betas used for the out-of-sample fits

        dispIters : bool : if True then iteration/convergence information is displayed

        dispItersInt : int : if dispIters is True, then how many iterations to wait until showing convergence info

        minTol : scalar : small number used to define convergence in ALS algorithm

        maxIters : scalar : large number that is the maximum number of iterations allowed

        F_names : list : list with K elements that name the latent factors

        G_names : list : list with M elements that name the prespecified factors

        R2_bench : str : what is the benchmark against which to calculate the R2
            'zero' (default) uses a constant 0 forecast for every asset and managed portfolio
            'mean' uses a mean of every asset and managed portfolio; if out-of-sample, this is a mean calculated
            from the training data

        [Outputs] dict with following keys

        xfits : dict(4) of 2 df(L x T) and 2 scalar :
            'Fits_Total' df(L x T) with factor-fits (using factor realization)
            'Fits_Pred' df(L x T) with lambda-fits (using lambda, the expectation of factor)
            'R2_Total' is R2 using 'Fits_Total'
            'R2_Pred' is R2 using 'Fits_Pred'

        Gamma : df(L x KM) or df(TL x KM) : if in-sample estimation, it is a single df. If out-of-sample
        estimation, then there is a separate Gamma for each training sample, indexed by the first element of the index
        The first columns are always the K estimated ones, with the remaining the gFac

        Factor : df(KM x T) : the in-sample factor estimates, or if out-of-sample the out-of-sample factor
        realizations, plus the prespecified factors
        The first rows are always the K estimated ones, with the remaining the gFac

        Lambda : dict(2) : 'estimate'  is srs(KM) or df(KM x T), 'VAR1' is df(KM x KM+1)  : 'estimate' has
        the main output: if factor_mean='constant' then a srs; if factor_mean='VAR' then a df with the
        conditional means of every factor realization. 'VAR1' is the VAR1 coefficients with constant first

        rfits : dict(2) of 1 df(TotalObs x 2) and 2 scalars:
            Is None if R_fit=False
            Otherwise is analogous to xfits dict described above

        fittedBeta : df(TotalObs x KM) : there is a beta for every Date,AssetID index row, and for each factor column
            Is None if Beta_return=False
        """

        fitstart = datetime.now()

        if gFac is None:
            has_gFac = False
            _gFac = None
        else:
            has_gFac = True
            _gFac = gFac.copy()

        if has_gFac:
            M = gFac.shape[0]
            if G_names is None:
                G_names = list(gFac.index)
        else:
            M = 0

        if M == 0:
            has_gFac = False

        KM = K + M

        # check for at least one factor
        if KM == 0:
            raise ValueError("K+M=0: there must be some factor used/estimated")

        if F_names is None and K > 0:
            F_names = list(range(K))
        if G_names is None and M > 0:
            G_names = gFac.index
        if K == KM:
            Factor_names = F_names
        elif M == KM:
            Factor_names = G_names
        else:
            Factor_names = F_names + G_names

        self.F_names, self.G_names, self.Factor_names = F_names, G_names, Factor_names

        if OOS:
            IS_or_OOS = "OOS"
        else:
            IS_or_OOS = "IS"

        # normalization_choice_specs turns into row indices for Gamma array
        if normalization_choice == "Identity":
            normalization_choice_specs = self._find_sublist(normalization_choice_specs)
            if np.min(normalization_choice_specs) == -1:
                raise ValueError("normalization problem")

        #######
        # In-sample (Full-sample) estimation
        #######
        if IS_or_OOS == "IS":
            Gamma0, Factor0 = self._svd_initial(K=K, M=M, gFac=gFac)
            tol, iters = float("inf"), 0
            timerstart = timer()
            while iters < maxIters and tol > minTol:
                iters += 1
                Gamma1, Factor1 = self._linear_als_estimation(
                    Gamma0=Gamma0.copy(),
                    gFac=_gFac,
                    K=K,
                    M=M,
                    KM=KM,
                    normalization_choice=normalization_choice,
                    normalization_choice_specs=normalization_choice_specs,
                )
                tolGam = np.max(np.abs(Gamma1 - Gamma0))
                tolFac = np.max(np.abs(Factor1 - Factor0))
                tol = np.max((tolGam, tolFac))
                if dispIters and iters % dispItersInt == 0:
                    print("iters %i: tol = %0.8f" % (iters, tol))
                # replace 0 with 1, for next iteration
                Gamma0, Factor0 = Gamma1.copy(), Factor1.copy()

            Gamma, Factor = Gamma0.copy(), Factor0.copy()
            numerical_stats = {
                "tol": tol,
                "minTol": minTol,
                "iters": iters,
                "maxIters": maxIters,
                "time": timer() - timerstart,
            }
            print(
                "ipca.fit finished estimation after %i seconds and %i iterations"
                % (numerical_stats["time"], numerical_stats["iters"])
            )

            Rdo, Betado, fittedX, fittedR, fittedBeta = self._setup_fits(
                R_fit, Beta_fit
            )

            Gamma = pd.DataFrame(data=Gamma, index=self.Chars, columns=Factor_names)
            if factor_mean == "constant":
                Lambda = pd.DataFrame(data=np.mean(Factor, axis=1), index=Factor_names)
                B = np.hstack(
                    (np.zeros((KM, KM)), np.mean(Factor, axis=1).reshape((-1, 1)))
                ).T
            elif factor_mean == "VAR1":
                Lambda = pd.DataFrame(
                    data=np.nan, index=Factor_names, columns=self.Dates
                )
                Lambda.iloc[:, 0] = np.mean(Factor, axis=1)  # for initial forecast
                B = self._VARB(X=Factor)
            Factor = pd.DataFrame(data=Factor, index=Factor_names, columns=self.Dates)

            for t in self.Dates:
                if factor_mean == "constant":
                    lamt = Lambda
                elif factor_mean == "VAR1":
                    if t == self.Dates[0]:
                        lamt = Lambda[t].values  # use unconditonal mean forecast
                    else:
                        lamt = B.T.dot(
                            np.hstack((Factor[t - 1].values, 1)).reshape((-1, 1))
                        )
                    Lambda[t] = lamt.reshape((-1, 1))
                fittedX["Fits_Total"][t] = (
                    self.W.loc[t].dot(Gamma).dot(Factor[t]).values.reshape((-1, 1))
                )
                fittedX["Fits_Pred"][t] = (
                    self.W.loc[t].dot(Gamma).dot(lamt).values.reshape((-1, 1))
                )
                if Rdo:  # have to construct beta if Rdo=True
                    if self.add_constant:
                        Betat = (
                            self.Z.loc[t].dot(Gamma.iloc[:-1, :]) + Gamma.iloc[-1, :]
                        )
                    else:
                        Betat = self.Z.loc[t].dot(Gamma)
                    fittedR["Fits_Total"].loc[t] = Betat.dot(Factor[t]).values.reshape(
                        (-1, 1)
                    )
                    fittedR["Fits_Pred"].loc[t] = Betat.dot(lamt).values.reshape(
                        (-1, 1)
                    )
                    if Betado:
                        fittedBeta.loc[t] = Betat.values
                elif Betado:  # could have Rdo=False and Betado=True
                    if self.add_constant:
                        Betat = (
                            self.Z.loc[t].dot(Gamma.iloc[:-1, :]) + Gamma.iloc[-1, :]
                        )
                    else:
                        Betat = self.Z.loc[t].dot(Gamma)
                    fittedBeta.loc[t] = Betat.values

            if R2_bench == "zero":
                benchR2 = None
            elif R2_bench == "mean":
                benchR2 = pd.DataFrame(
                    data=np.tile(self.X.mean(axis=1), self.X.shape[1]),
                    index=self.X.index,
                    columns=self.X.columns,
                )
            fittedX["R2_Total"], fittedX["R2_Pred"] = self._R2_calc(
                reals=self.X,
                fits_total=fittedX["Fits_Total"],
                fits_pred=fittedX["Fits_Pred"],
                benchR2=benchR2,
            )

            if Rdo:
                if R2_bench == "zero":
                    benchR2 = None
                elif R2_bench == "mean":
                    benchR2 = pd.DataFrame(
                        data=np.ones((self.R.shape[0], 1)) * self.R.mean().values,
                        index=self.R.index,
                        columns=self.R.columns,
                    )
                fittedR["R2_Total"], fittedR["R2_Pred"] = self._R2_calc(
                    reals=self.R,
                    fits_total=fittedR["Fits_Total"],
                    fits_pred=fittedR["Fits_Pred"],
                    benchR2=benchR2,
                )

        #######
        # Out of sample
        #######
        elif IS_or_OOS == "OOS":
            Gamma0, Factor0 = self._svd_initial(
                K=K, M=M, gFac=gFac, Dates=self.Dates[:OOS_window_specs]
            )
            Rdo, Betado, fittedX, fittedR, fittedBeta = self._setup_fits(
                R_fit, Beta_fit
            )

            Lambda = pd.DataFrame(data=np.nan, index=Factor_names, columns=self.Dates)
            Factor = Lambda.copy()
            Gamma = pd.DataFrame(
                data=np.nan,
                columns=Factor_names,
                index=pd.MultiIndex.from_product([self.Dates, self.Chars]),
            )
            if R2_bench == "mean":
                benchX = pd.DataFrame(data=0.0, columns=self.Dates, index=self.Chars)
            elif R2_bench == "zero":
                benchX = None

            if Rdo:
                if R2_bench == "mean":
                    benchR = pd.DataFrame(
                        data=0.0, index=self.R.index, columns=self.R.columns
                    )
                elif R2_bench == "zero":
                    benchR = None

            numerical_stats = {
                "minTol": minTol,
                "maxIters": maxIters,
                "tol": pd.DataFrame(data=np.nan, columns=self.Dates, index=[0]),
                "iters": pd.DataFrame(data=np.nan, columns=self.Dates, index=[0]),
                "time": pd.DataFrame(data=np.nan, columns=self.Dates, index=[0]),
            }
            ct = 0
            for t in self.Dates[OOS_window_specs:]:
                tol, iters = float("inf"), 0
                if OOS_window == "rolling":
                    datest = self.X.loc[
                        :,
                        (t - pd.DateOffset(months=OOS_window_specs)) : (
                            t - pd.DateOffset(months=1)
                        ),
                    ].columns  # dates through t-1, not including t
                else:
                    datest = self.X.loc[
                        :, : t - pd.DateOffset(months=1)
                    ].columns  # dates through t-1, not including t
                # Using datest below means that we estimate Gamma0, Factor0 on data through t-1, which include returns through t-1
                # and chars through t-2. Then the last Factor realization is for t-1
                timerstart = timer()
                while iters < maxIters and tol > minTol:
                    iters += 1
                    # for first t, Gamma0 will be from _svd_initial outside loop; for subsequent t, will be that last
                    #  Gamma0 obtained in the previous t's iterative "while" stmt
                    Gamma1, Factor1 = self._linear_als_estimation(
                        Gamma0=Gamma0.copy(),
                        gFac=gFac,  # make _gFac for ndarray-based
                        K=K,
                        M=M,
                        KM=KM,
                        normalization_choice=normalization_choice,
                        normalization_choice_specs=normalization_choice_specs,
                        Dates=datest,
                    )

                    tolGam = np.max(np.abs(Gamma1 - Gamma0))
                    tolFac = np.max(np.abs(Factor1 - Factor0))
                    tol = np.max((tolGam, tolFac))
                    if dispIters and iters % dispItersInt == 0:
                        print("iters {}: tol = {}".format(iters, tol))
                    # replace 0 with 1, for next iteration
                    Gamma0, Factor0 = Gamma1.copy(), Factor1.copy()

                numerical_stats["tol"][t] = tol
                numerical_stats["iters"][t] = iters
                numerical_stats["time"][t] = timer() - timerstart
                Gamma.loc[t] = Gamma0
                # "OOS" factor realization: use Gamma0 and W known at time t-1, but X known at time t
                if M == 0:
                    Factor[t] = _mldivide(
                        Gamma.loc[t].T.dot(self.W.loc[t]).dot(Gamma.loc[t]),
                        Gamma.loc[t].T.dot(self.X[t]),
                    )
                else:
                    tmp = _mldivide(
                        Gamma.loc[t, F_names]
                        .T.dot(self.W.loc[t])
                        .dot(Gamma.loc[t, F_names]),
                        Gamma.loc[t, F_names].T.dot(self.X[t]),
                    )
                    Factor[t].loc[F_names] = tmp
                    Factor[t].loc[G_names] = gFac[t].values
                fittedX["Fits_Total"][t] = (
                    self.W.loc[t]
                    .dot(Gamma.loc[t])
                    .dot(Factor[t])
                    .values.reshape((-1, 1))
                )
                # calculate Lambda and fill predictive fits
                if factor_mean == "constant":
                    Lambda[t] = Factor0.mean(
                        axis=1
                    )  # actually known before t; but we label to associate it with its realization
                    B = np.hstack(
                        (np.zeros((KM, KM)), Factor0.mean(axis=1).reshape((-1, 1)))
                    ).T
                elif factor_mean == "VAR1":
                    B = self._VARB(X=Factor0)
                    Lambda[t] = B.T.dot(np.hstack((Factor0[:, -1], 1)).reshape((-1, 1)))

                # predictive fittedX
                fittedX["Fits_Pred"][t] = (
                    self.W.loc[t]
                    .dot(Gamma.loc[t])
                    .dot(Lambda[t])
                    .values.reshape((-1, 1))
                )
                if R2_bench == "mean":
                    benchX[t] = self.X[:t].values.mean()

                # fittedR and fittedBeta
                if Rdo:
                    if self.add_constant:
                        Betat = self.Z.loc[t].dot(Gamma0[:-1, :]) + Gamma0[-1, :]
                    else:
                        Betat = self.Z.loc[t].dot(Gamma0)
                    fittedR["Fits_Total"].loc[t] = Betat.dot(
                        Factor[t].values
                    ).values.reshape((-1, 1))
                    fittedR["Fits_Pred"].loc[t] = Betat.dot(
                        Lambda[t].values
                    ).values.reshape((-1, 1))

                    if R2_bench == "mean":
                        benchR[t] = self.R.loc[:t].values.mean()
                    if Betado:
                        fittedBeta.loc[t] = Betat
                elif Betado:
                    if self.add_constant:
                        Betat = self.Z.loc[t].dot(Gamma0[:-1, :]) + Gamma0[-1, :]
                    else:
                        Betat = self.Z.loc[t].dot(Gamma0)
                    fittedBeta.loc[t] = Betat.values

                # Prepare for next period through loop
                if OOS_window == "rolling":
                    pass
                else:
                    Factor0 = np.concatenate(
                        (Factor0.copy(), np.zeros((Factor0.shape[0], 1))), axis=1
                    )

                ct += 1
                if dispIters and ct % 12 == 0:
                    print(
                        "%s is done and took %i iterations and %0.2f seconds"
                        % (t, numerical_stats["iters"][t], numerical_stats["time"][t])
                    )

            # R2s
            fittedX["R2_Total"], fittedX["R2_Pred"] = self._R2_calc(
                reals=self.X.loc[:, self.Dates[OOS_window_specs:]],
                fits_total=fittedX["Fits_Total"].loc[:, self.Dates[OOS_window_specs:]],
                fits_pred=fittedX["Fits_Pred"].loc[:, self.Dates[OOS_window_specs:]],
                benchR2=benchX,
            )
            if Rdo:
                fittedR["R2_Total"], fittedR["R2_Pred"] = self._R2_calc(
                    reals=self.R.loc[self.Dates[OOS_window_specs] :],
                    fits_total=fittedR["Fits_Total"].loc[
                        self.Dates[OOS_window_specs] :
                    ],
                    fits_pred=fittedR["Fits_Pred"].loc[self.Dates[OOS_window_specs] :],
                    benchR2=benchR,
                )
                fittedR["Fits_Pred"] = fittedR["Fits_Pred"].merge(
                    self.R.loc[self.Dates[OOS_window_specs] :].rename(columns={"exret": "real_exret"}),
                    on=["date", "permno"],
                    how="inner",
                )

        # Fill the Lambda dict
        Lambda_dict = {}
        Lambda_dict["estimate"] = Lambda
        Lambda_dict["VAR1"] = pd.DataFrame(
            data=B.T, index=Factor_names, columns=(Factor_names + ["cons"])
        )

        # numerical information
        fitend = datetime.now()
        numerical_stats["fit_start_time"], numerical_stats["fit_end_time"] = (
            fitstart,
            fitend,
        )
        return {
            "xfits": fittedX,
            "Gamma": Gamma,
            "Factor": Factor,
            "Lambda": Lambda_dict,
            "rfits": fittedR,
            "fittedBeta": fittedBeta,
            "numerical": numerical_stats,
        }

    def _linear_als_estimation(
        self,
        Gamma0,
        K,
        M,
        KM,
        normalization_choice,
        normalization_choice_specs,
        gFac,
        Dates=None,
    ):
        """
        Runs one iteration of the alternating least squares estimation process

        [Inputs]
        Gamma0 : df(L x KM) : previous iteration's Gamma estimate

        Dates : None, index : if None, uses self.Dates (full sample); otherwise uses the given Dates index which have
        T2 entries

        Other inputs of this particular fitted object, which should simply be passed

        [Outputs]
        Gamma1 : df(L x KM) : current iteration's Gamma estimate

        Factor : df(KM x T2) : current iteration's latent Factor estimate, or None
        """
        # dataframe-based, using self.X, self.W
        if Dates is None:
            Dates = self.Dates
        # # ndarray-based, using self._X, self._W
        # if Dates is None:
        #     Dates = range(0, len(self.Dates))

        if K == KM:
            GammaF = Gamma0
            GammaG = None
        elif M == KM:
            GammaF = None
            GammaG = Gamma0

        else:
            GammaF, GammaG = Gamma0[:, 0:K], Gamma0[:, K:]
        # 1. estimate latent factor, if K>0
        if K > 0:
            FactorF = np.zeros((K, len(Dates)))
            # dataframe-based
            ct = 0
            for t in Dates:
                if M > 0:
                    FactorF[:, ct] = _mldivide(
                        GammaF.T.dot(self.W.loc[t].values).dot(GammaF),
                        GammaF.T.dot(
                            self.X[t].values
                            - self.W.loc[t].values.dot(GammaG).dot(gFac[t].values)
                        ),
                    )
                else:
                    FactorF[:, ct] = _mldivide(
                        GammaF.T.dot(self.W.loc[t].values).dot(GammaF),
                        GammaF.T.dot(self.X[t].values),
                    )
                ct += 1
            # # ndarray-based
            # ct=0
            # for t in Dates:
            #     if M>0:
            #         FactorF[:, ct] = _mldivide(
            #             GammaF.T.dot(self._W[:, :, t]).dot(GammaF),
            #             GammaF.T.dot(self._X[:, t] - self._W[:, :, t].values.dot(GammaG).dot(gFac[:, t]))
            #         )
            #     else:
            #         FactorF[:, ct] = _mldivide(
            #             GammaF.T.dot(self._W[:, :, t]).dot(GammaF),
            #             GammaF.T.dot(self._X[:, t])
            #         )
            #     ct+=1
        else:
            FactorF = None
        # 2. estimate Gamma, for latent or prespecified factors which are present
        if K == KM:
            Factor = FactorF
        elif M == KM:
            # dataframe-based
            Factor = gFac[Dates].values
            # # ndarray-based
            # Factor = gFac[:, Dates]
        else:
            # dataframe-based
            Factor = np.concatenate((FactorF.copy(), gFac[Dates].values.copy()), axis=0)
            # # ndarray-based
            # Factor = np.concatenate((FactorF, gFac[:, Dates]), axis=0)

        numer, denom = np.zeros(self.L * KM), np.zeros((self.L * KM, self.L * KM))
        # dataframe-based
        ct = 0
        for t in Dates:
            numer += np.kron(self.X[t].values, Factor[:, ct]) * self.Nts[t]
            denom += (
                np.kron(self.W.loc[t].values, np.outer(Factor[:, ct], Factor[:, ct]))
                * self.Nts[t]
            )
            ct += 1
        # # ndarray-based
        # ct=0
        # for t in Dates:
        #     numer += np.kron(self._X[:, t], Factor[:, ct]) * self.Nts[t]
        #     denom += np.kron(self._W[:, :, t], np.outer(Factor[:, ct], Factor[:, ct])) * self.Nts[t]
        #     ct+=1

        Gamma1 = np.reshape(_mldivide(denom, numer), (self.L, KM))
        # 3. normalization
        if K > 0:
            Gamma1, Factor1 = self._normalization_choice(
                Gamma=Gamma1.copy(),
                Factor=Factor.copy(),
                K=K,
                M=M,
                KM=KM,
                normalization_choice=normalization_choice,
                normalization_choice_specs=normalization_choice_specs,
            )
        else:
            Factor1 = Factor.copy()
        return Gamma1, Factor1

    # calculate SVD for initial conditions for ALS algorithm
    def _svd_initial(self, K, M, gFac=None, Dates=None):
        """
        Dates : None, index : if None, uses self.Dates (full sample); otherwise uses the given Dates index

        gFac : None, (M x T)array :
        """
        if Dates is None:
            Dates = self.Dates

        (
            Gamma,
            Factor,
        ) = (
            None,
            None,
        )

        if K > 0:
            U, s, VT = sla.svd(self.X.loc[:, Dates].values, full_matrices=False)
            Gamma = U[:, 0:K]
            Factor = np.diag(s[0:K]).dot(VT[0:K, :])
        if M > 0 and K > 0:
            Gamma = np.concatenate((Gamma.copy(), np.zeros((self.L, M))), axis=1)
            Factor = np.concatenate((Factor.copy(), gFac[Dates].values), axis=0)
        elif M > 0 and K == 0:
            Gamma = np.zeros((self.L, M))
            Factor = gFac.values
        return Gamma, Factor

    # normalize Gamma, Factor
    def _normalization_choice(
        self, Gamma, Factor, K, M, KM, normalization_choice, normalization_choice_specs
    ):
        if M == KM:
            raise ValueError(
                "do not use _normalization_choice with no latent factors (M==KM)"
            )
        if normalization_choice == "PCA_positivemean":
            if K == KM:
                GammaF, FactorF = Gamma, Factor
            else:
                GammaF, GammaG = Gamma[:, 0:K], Gamma[:, K:]
                FactorF, FactorG = Factor[0:K, :], Factor[K:, :]
            R1 = sla.cholesky(GammaF.T.dot(GammaF))
            R2, _, _ = sla.svd(R1.dot(FactorF).dot(FactorF.T).dot(R1.T))
            GammaF = _mrdivide(GammaF, R1).dot(R2)
            FactorF = _mldivide(R2, R1.dot(FactorF))
            # sign convention: FactorF has positive mean
            sign_conv = np.sign(np.mean(FactorF, axis=1)).reshape((-1, 1))
            sign_conv[sign_conv == 0] = 1
            FactorF = FactorF * sign_conv
            GammaF = GammaF * sign_conv.T
            # orthogonality between GammaF and GammaG
            if M > 0:
                GammaG = (np.identity(self.L) - GammaF.dot(GammaF.T)).dot(GammaG)
                FactorF = FactorF + (GammaF.T.dot(GammaG)).dot(FactorG)
            # sign convention: FactorF has positive mean
            sign_conv = np.sign(np.mean(FactorF, axis=1)).reshape((-1, 1))
            sign_conv[sign_conv == 0] = 1
            FactorF = FactorF * sign_conv
            GammaF = GammaF * sign_conv.T
        elif normalization_choice == "Identity":
            if K == KM:
                GammaF, FactorF = Gamma, Factor
            else:
                GammaF, GammaG = Gamma[:, 0:K], Gamma[:, K:]
                FactorF, FactorG = Factor[0:K, :], Factor[K:, :]
            R = GammaF[normalization_choice_specs, :]
            GammaF = _mrdivide(GammaF, R)
            FactorF = R.dot(FactorF)
            # orthogonality between GammaF and GammaG
            if M > 0:
                GammaG = (np.identity(self.L) - GammaF.dot(GammaF.T)).dot(GammaG)
                FactorF = FactorF + (GammaF.T.dot(GammaG)).dot(FactorG)

        if M > 0:
            Gamma = np.concatenate((GammaF.copy(), GammaG.copy()), axis=1)
            Factor = np.concatenate((FactorF.copy(), FactorG.copy()), axis=0)
        else:
            Gamma = GammaF.copy()
            Factor = FactorF.copy()
        return Gamma, Factor

    # estimate a VAR1 coefficent matrix (including constant)
    def _VARB(self, X):
        Xtil = np.concatenate((X.copy(), np.ones((1, X.shape[1]))), axis=0)
        B = sla.lstsq(Xtil[:, :-1].T, X[:, 1:].T)[0]
        return B

    # internal R2 calculation code
    def _R2_calc(self, reals, fits_total, fits_pred, benchR2=None):
        if benchR2 is None:
            R2_Total = (
                1 - ((reals - fits_total) ** 2).values.sum() / (reals**2).values.sum()
            )
            R2_Pred = (
                1 - ((reals - fits_pred) ** 2).values.sum() / (reals**2).values.sum()
            )
        else:
            R2_Total = (
                1
                - ((reals - fits_total) ** 2).values.sum()
                / ((reals - benchR2) ** 2).values.sum()
            )
            R2_Pred = (
                1
                - ((reals - fits_pred) ** 2).values.sum()
                / ((reals - benchR2) ** 2).values.sum()
            )

        return R2_Total, R2_Pred

    # external R2 calculation code
    def R2_of_fits(
        self, results=None, date_range=None, benchR2=None, R2name=None, inplace=True
    ):
        """
        Calculates R2 over given datetime ranges

        [Inputs]

        results : dict : the output of an object output from ipca.IPCA.fit()

        date_range : datetime index : a choice of dates in the dfs within results over which to calculate the R2

        benchR2 : str, None : choice of benchmark; in the following the in-sample period is the one decided by date_range,
        but for out-of-sample we use data from the beginning of the available dates, even if this is prior to date_range (which
        will then be used to determine which periods' fits to put into the R2 calculation)
            - if None (default) then benchmark is taken as 0
            - if 'all_mean' then use in-sample mean of all observations
            - if 'individual_mean' then use in-sample mean of each cross-sectional unit's observations
            - if 'all_mean_recursive' then use recursively estimated out-of-sample mean of all observations
            - if 'individual_mean_recursive' then use recursively estimated out-of-sample mean of each cross-sectional unit's observations

        R2name : str, None : if None (default) then the suffix of the name of the R2s are given by the first and last date in date_range
        using strftime with the '%Y%m%d' format chosen (eg 'R2_Total_19650130_20161231')
            If a string is passed, that is instead the name of the R2 (eg 'R2_Total_mychosenstring')

        inplace : bool : if True (default) then the passed results dict is modified and None is passed out; if False, then the
        passed results dict is copied and then modified and passed out

        [Outputs]

        results : a dict with 'rfits' and 'xfits' keys of the results dict; if the the original ipca object had to R, then only 'xfits' is output
        if inplace=True (default) then the passed results dict is modified; if inplace=False then the passed results dict is copied and sent out

        """
        # error checking
        if results is None:
            raise ValueError("ipca.R2_of_fits requires a results dict to be passed")
        if date_range is None:
            raise ValueError(
                "ipca.R2_of_fits requires a date_range list or index to be passed"
            )

        # calculate the appropriate benchmarks
        if benchR2 is None:
            benchX = pd.DataFrame(data=0.0, index=self.X.index, columns=self.X.columns)
            if self.has_RZ:
                benchR = pd.DataFrame(
                    data=0.0, index=self.R.index, columns=self.R.columns
                )
        elif benchR2 == "all_mean":
            benchX = pd.DataFrame(
                data=np.mean(self.X.values), index=self.X.index, columns=self.X.columns
            )
            if self.has_RZ:
                benchR = pd.DataFrame(
                    data=np.mean(self.R.values),
                    index=self.R.index,
                    columns=self.R.columns,
                )
        elif benchR2 == "individual_mean":
            benchX = pd.DataFrame(
                data=np.tile(
                    self.X.T.loc[date_range].T.mean(axis=1).values.reshape((-1, 1)),
                    (1, self.X.shape[1]),
                ),
                index=self.X.index,
                columns=self.X.columns,
            )
            if self.has_RZ:
                benchR = (
                    self.R.loc[date_range]
                    .groupby(level=1)
                    .transform(lambda x: x.mean())
                )
        elif benchR2 == "all_mean_recursive":
            benchX = pd.DataFrame(data=0.0, index=self.X.index, columns=self.X.columns)
            for t in self.X.columns[1:]:
                benchX[t] = np.mean(self.X.T.loc[: t - 1].values)
            if self.has_RZ:
                benchR = pd.DataFrame(
                    data=0.0, index=self.R.index, columns=self.R.columns
                )
                for t in self.X.columns[1:]:
                    benchR.loc[t] = np.mean(self.R.loc[:t].values)
        elif benchR2 == "individual_mean_recursive":
            benchX = self.X.T.shift(1).expanding().mean().T
            if self.has_RZ:
                benchR = self.R.groupby(level=1).shift(1).expanding().mean()

        # calculate the R2s
        r2_x_t = (
            1
            - (
                (
                    self.X.T.loc[date_range]
                    - results["xfits"]["Fits_Total"].T.loc[date_range]
                )
                ** 2
            ).values.sum()
            / ((self.X.T.loc[date_range] - benchX.T.loc[date_range]) ** 2).values.sum()
        )
        r2_x_p = (
            1
            - (
                (
                    self.X.T.loc[date_range]
                    - results["xfits"]["Fits_Pred"].T.loc[date_range]
                )
                ** 2
            ).values.sum()
            / ((self.X.T.loc[date_range] - benchX.T.loc[date_range]) ** 2).values.sum()
        )
        if self.has_RZ:
            r2_r_t = (
                1
                - (
                    (
                        self.R.loc[date_range]
                        - results["rfits"]["Fits_Total"].loc[date_range]
                    )
                    ** 2
                ).values.sum()
                / ((self.R.loc[date_range] - benchR.loc[date_range]) ** 2).values.sum()
            )
            r2_r_p = (
                1
                - (
                    (
                        self.R.loc[date_range]
                        - results["rfits"]["Fits_Pred"].loc[date_range]
                    )
                    ** 2
                ).values.sum()
                / ((self.R.loc[date_range] - benchR.loc[date_range]) ** 2).values.sum()
            )

        # find the name
        if R2name is None:
            R2name = (
                date_range[0].strftime("%Y%m%d")
                + "-"
                + date_range[-1].strftime("%Y%m%d")
            )

        # inplace
        if inplace:
            results["rfits"]["R2_Total_" + R2name] = r2_r_t
            results["rfits"]["R2_Pred_" + R2name] = r2_r_p
            results["xfits"]["R2_Total_" + R2name] = r2_x_t
            results["xfits"]["R2_Pred_" + R2name] = r2_x_p
            newresults = None
        else:
            newresults = copy.deepcopy(results)
            newresults["rfits"]["R2_Total_" + R2name] = r2_r_t
            newresults["rfits"]["R2_Pred_" + R2name] = r2_r_p
            newresults["xfits"]["R2_Total_" + R2name] = r2_x_t
            newresults["xfits"]["R2_Pred_" + R2name] = r2_x_p

        return newresults

    # Set-up function using in both IS and OOS routines
    def _setup_fits(self, R_fit, Beta_return):
        Rdo = False
        if R_fit and not self.has_RZ:
            print(
                "ipca.fit() was given R_fit=True but is not given a RZ; None is being passed out as "
                '"fittedR"'
            )
            Rdo = False
        elif R_fit:
            Rdo = True

        Betado = False
        if Beta_return and not self.has_RZ:
            print(
                "ipca.fit() was given Beta_return=True but is not given a Z; None is being passed out as "
                '"fittedBeta"'
            )
            Betado = False
        elif Beta_return:
            Betado = True

        fittedX = {
            "Fits_Total": pd.DataFrame(np.nan, index=self.X.index, columns=self.Dates),
            "Fits_Pred": pd.DataFrame(np.nan, index=self.X.index, columns=self.Dates),
        }

        if Rdo:
            fittedR = {
                "Fits_Total": pd.DataFrame(
                    np.nan, index=self.R.index, columns=self.R.columns
                ),
                "Fits_Pred": pd.DataFrame(
                    np.nan, index=self.R.index, columns=self.R.columns
                ),
            }
        else:
            fittedR = None

        if Betado:
            fittedBeta = pd.DataFrame(
                np.nan, index=self.R.index, columns=self.Factor_names
            )
        else:
            fittedBeta = None

        return Rdo, Betado, fittedX, fittedR, fittedBeta

    # find sublist
    def _find_sublist(self, sub):
        pos = list()
        ct = 0
        for subj in sub:
            try:
                pos.append(list(self.Chars).index(subj))
            except:
                pos.append(-1)
                print("We did not find %s in the list of chars" % subj)
            ct += 1
        return pos
