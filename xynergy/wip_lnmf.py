import numpy as np
import scipy.linalg as la
import optuna
from .factor import _to_mat
from .fit import fit_individual_drugs
import polars as pl


# Function should pull out the max and min inhibition
#
# Unclear what the ic50 and slope should be - but per Kazi's email, let's go
# with an HSA-like approach and choose the parameters of the more potent one
def use_lnmf(x, dose_cols, response_col):
    # Limited to taking a single experiment (for now) since that's what _to_mat takes
    #

    # Currently just using this to brute force use lnmf

    # Ensure no replicates
    x = x.group_by(dose_cols).agg(pl.col(response_col).mean())
    dfcw = _to_mat(x, dose_cols, response_col)
    # Experiment cols is none because only one experiment
    max_val = x[response_col].max()
    min_val = x[response_col].min()

    fits = fit_individual_drugs(x, dose_cols, response_col, None)

    # As noted above, unclear which slope/ic50 to use - so in the style of HSA
    # we'll chose the params from the more potent one
    # FIXME This is gonna cause issues if the ic50s are in different units
    most_potent = fits.filter(pl.col("ic50") == pl.col("ic50").min())
    log10IC50 = np.log10(most_potent["ic50"][0])
    slope = most_potent["slope"][0]

    # I think kazi got the max and min values flipped. Need to make sure if that's true in all cases
    obj = FourParamLogisticNMFConsensus(dfcw / 100, min_val, slope, log10IC50, max_val)
    return obj.run_consensus()


class BaseFourParamLogisticNMF:
    """
    Base class providing 4-parameter logistic functionality and factor initialization.
    """

    def __init__(self, a, b, c, d):
        """
        Store the 4PL parameters
        a = max inhibition (ratio),
        b = Hill slope,
        c = EC50 molar,
        d = min inhibition (ratio)).

        Parameters
        ----------
        a, b, c, d : float
            The 4PL parameters.
        """
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def four_param_logistic(self, x):
        """
        4-parameter logistic function activation:
            f(z) = d + (a - d) / (1.0 + (z/c)**b).
        4-parameter logistic function inhibition:
            f(z) = d + (a - d) / (1.0 + (c/z)**b).
        """
        p = self.d + (self.a - self.d) / (1.0 + (self.c / x) ** self.b)

        return p

    def _4pl_derivative(self, z):
        """
        Derivative of the 4PL wrt z. If p = four_param_logistic(z):
          p = self.four_param_logistic(z)
          p'(z) = (b / (a - d)) * (p - a) * (d - p) * (1 / z)
        """
        p = self.four_param_logistic(z)
        delta = (self.b / (self.a - self.d)) * (p - self.a) * (self.d - p) * (1 / z)
        return delta

    @staticmethod
    def _initialize_factors(X, K):
        """
        Common partial SVD (or fallback to random) for factor initialization.
        Returns (W, H), both nonnegative, with columns of W normalized.

        Parameters
        ----------
        X : ndarray
            Data matrix.
        K : int
            Rank for factorization.

        Returns
        -------
        W, H : ndarray
            Factor matrices.
        """
        m, n = X.shape
        # Fill missing with 0 for SVD init
        X_filled = np.nan_to_num(X, nan=0.0)

        try:
            U, s, Vt = la.svd(X_filled, full_matrices=False)
            U = U[:, :K]
            s = s[:K]
            Vt = Vt[:K, :]
            W = np.abs(U)
            H = np.abs(np.diag(s) @ Vt)
        except Exception:
            # fallback to random init
            W = np.abs(np.random.rand(m, K))
            H = np.abs(np.random.rand(K, n))
        # NOTE KA: abs might not be the best fit - perhaps set to 0.0 instead?
        # Normalize columns of W
        col_norms = np.sqrt((W**2).sum(axis=0))
        col_norms[col_norms < 1e-9] = 1.0
        W /= col_norms
        H *= col_norms.reshape(-1, 1)
        return W, H


class FourParamLogisticNMF(BaseFourParamLogisticNMF):
    """
    Logistic Consensus Non-negative Matrix Factorization (LC-NMF) model with a 4-parameter
    logistic (4PL) function, using a cross-entropy-based objective for observed entries
    and partial SVD for initialization.

    The 4PL function is:
        f(z) = d + (a - d) / (1.0 + (c / z)**b).

    Cross-entropy form for observed entries:
        L_obs = alpha * sum( X_ij * log(X_hat_ij) + (1 - X_ij) * log(1 - X_hat_ij) ),
        where X_hat_ij = four_param_logistic( (WH)_ij ).

    Unobserved entries use a weaker penalty:
        L_unobs = beta * sum( log(1 - X_hat_ij) ).

    Regularization:
        lambda_W * ||W||^2 + lambda_H * ||H||^2.

    We invert the sign for gradient descent on the negative log-likelihood.

    Parameters
    ----------
    K : int
        Number of latent factors.
    a, b, c, d : float
        Parameters for the 4PL function (minimum asymptote, Hill slope, EC50, maximum asymptote).
    alpha : float, optional
        Weight for observed entries (default is 1.0).
    beta : float, optional
        Weight for unobserved entries (default is 0.01).
    lambda_W : float, optional
        Regularization parameter for W (default is 0.1).
    lambda_H : float, optional
        Regularization parameter for H (default is 0.1).
    max_iter : int, optional
        Maximum number of iterations for optimization (default is 500).
    eta : float, optional
        Base learning rate for gradient descent updates (default is 0.01).

    Methods
    -------
    fit_transform(X)
        Fits the LC-NMF model to input data X (with NaNs for missing)
        and returns the 4PL-transformed reconstruction.
    """

    def __init__(
        self,
        K,
        a,
        b,
        c,
        d,
        alpha=1.0,
        beta=0.01,
        lambda_W=0.1,
        lambda_H=0.1,
        max_iter=500,
        eta=0.01,
        gamma_bound=0.0,
    ):
        # Call the base class to store the 4PL parameters
        super().__init__(a, b, c, d)

        self.K = K
        self.a, self.b, self.c, self.d = a, b, c, d
        self.alpha, self.beta = alpha, beta
        self.lambda_W, self.lambda_H = lambda_W, lambda_H
        self.max_iter, self.eta = max_iter, eta
        self.gamma_bound = gamma_bound

        self.W_ = None
        self.H_ = None

    # ------------------------------
    # Main fit
    # ------------------------------
    def fit_transform(self, X):
        """
        Fit model to X and return the 4PL-transformed reconstruction.
        Missing entries in X are indicated by NaN.
        """
        # Build masks
        mask_obs = ~np.isnan(X)  # Observed data
        mask_unobs = np.isnan(X)  # Unobserved / missing data
        # Fill missing with 0 for gradient computations
        X_filled = np.nan_to_num(X, nan=0.0)

        # Initialize factors (via base class method)
        W, H = self._initialize_factors(X, self.K)
        prev_obj = np.inf

        for _iter in range(self.max_iter):
            # Update W
            W = self._update_W(X_filled, W, H, mask_obs, mask_unobs)
            # Update H
            H = self._update_H(X_filled, W, H, mask_obs, mask_unobs)

            # Compute objective
            obj = self._objective(X_filled, W, H, mask_obs, mask_unobs)
            if np.abs(prev_obj - obj) / max(1.0, np.abs(prev_obj)) < 1e-5:
                break
            prev_obj = obj

        self.W_, self.H_ = W, H
        return self.four_param_logistic(W @ H)

    # ------------------------------
    # Objective
    # ------------------------------
    def _objective(self, X, W, H, mask_obs, mask_unobs):
        """
        Negative log-likelihood (cross-entropy style) for observed data + log(1 - p) for unobserved,
        plus L2 regularization.
        We'll store it as a positive value, so we'll omit the minus sign in the code that
        performs gradient descent (the gradient steps are effectively minimizing 'obj').
        """

        WH = W @ H
        X_hat = self.four_param_logistic(WH)

        # cross-entropy for observed
        eps = 1e-12
        p = np.clip(X_hat, eps, 1 - eps)
        X_obs = X[mask_obs]

        # cross-entropy term: -[ x*log(p) + (1-x)*log(1-p ) ]
        ce_obs = -(X_obs * np.log(p[mask_obs]) + (1 - X_obs) * np.log(1 - p[mask_obs]))
        ce_obs_sum = ce_obs.sum()

        # unobserved term: -log(1 - p) => encourage p ~ 0 if unobserved
        unobs_term = -np.log(1 - p[mask_unobs] + eps)
        unobs_sum = unobs_term.sum()

        # regularization
        reg = self.lambda_W * (W**2).sum() + self.lambda_H * (H**2).sum()

        if self.gamma_bound > 0.0:
            over_1 = np.clip(X_hat - 1.0, a_min=0.0, a_max=None)  # values > 1
            under_0 = np.clip(-X_hat, a_min=0.0, a_max=None)  # values < 0
            penalty_out_of_bounds = (over_1.sum() + under_0.sum()) * self.gamma_bound
        else:
            penalty_out_of_bounds = 0.0
        # Weighted sum
        obj = (
            self.alpha * ce_obs_sum
            + self.beta * unobs_sum
            + reg
            + penalty_out_of_bounds
        )

        return obj

    # ------------------------------
    # Gradient Updates
    # ------------------------------
    def _update_W(self, X, W, H, mask_obs, mask_unobs):
        grad = self._grad_W(X, W, H, mask_obs, mask_unobs)
        W_new = W - self.eta * grad
        return np.maximum(0.0, W_new)

    def _update_H(self, X, W, H, mask_obs, mask_unobs):
        grad = self._grad_H(X, W, H, mask_obs, mask_unobs)
        H_new = H - self.eta * grad
        return np.maximum(0.0, H_new)

    def _grad_W(self, X, W, H, mask_obs, mask_unobs):
        """
        Gradient of the objective w.r.t. W using cross-entropy + 4PL derivative.
        """
        WH = W @ H
        X_hat = self.four_param_logistic(WH)
        dX_hat = self._4pl_derivative(WH)

        eps = 1e-12
        p = np.clip(X_hat, eps, 1 - eps)

        # For observed data:
        #   d/dz [ - (x ln(p) + (1 - x) ln(1 - p)) ] = (p - x) / [ p(1 - p) ] (due to negative sign)
        # Weighted by alpha, then multiplied by derivative of X_hat wlth respect to z => dX_hat.
        obs_factor = np.zeros_like(X_hat)
        X_obs = X[mask_obs]
        obs_factor[mask_obs] = (
            self.alpha * (p[mask_obs] - X_obs) / (p[mask_obs] * (1 - p[mask_obs]))
        )

        # For unobserved data:
        #   d/dz [ - beta ln(1 - p) ] => beta / (1 - p)
        unobs_factor = np.zeros_like(X_hat)
        unobs_factor[mask_unobs] = self.beta * (1.0 / (1 - p[mask_unobs]))

        total_factor = (obs_factor + unobs_factor) * dX_hat

        grad_W = total_factor @ H.T

        # L2 reg
        grad_W += 2.0 * self.lambda_W * W

        # penalty for out-of-bounds
        if self.gamma_bound > 0.0:
            above_mask = X_hat > 1.0
            below_mask = X_hat < 0.0
            penalty_factor = np.zeros_like(X_hat)

            penalty_factor[above_mask] = self.gamma_bound  # push down
            penalty_factor[below_mask] = -self.gamma_bound  # push up

            # multiply by logistic derivative
            penalty_factor *= dX_hat
            # chain rule into W
            grad_W += penalty_factor @ H.T

        return grad_W

    def _grad_H(self, X, W, H, mask_obs, mask_unobs):
        """
        Gradient of the objective w.r.t. H using cross-entropy + 4PL derivative.
        """
        WH = W @ H
        X_hat = self.four_param_logistic(WH)
        dX_hat = self._4pl_derivative(WH)

        eps = 1e-12
        p = np.clip(X_hat, eps, 1 - eps)

        obs_factor = np.zeros_like(X_hat)
        X_obs = X[mask_obs]
        obs_factor[mask_obs] = (
            self.alpha * (p[mask_obs] - X_obs) / (p[mask_obs] * (1 - p[mask_obs]))
        )

        unobs_factor = np.zeros_like(X_hat)
        unobs_factor[mask_unobs] = self.beta * (1.0 / (1 - p[mask_unobs]))

        total_factor = (obs_factor + unobs_factor) * dX_hat

        grad_H = W.T @ total_factor

        # L2 reg
        grad_H += 2.0 * self.lambda_H * H

        # penalty
        if self.gamma_bound > 0.0:
            above_mask = X_hat > 1.0
            below_mask = X_hat < 0.0
            penalty_factor = np.zeros_like(X_hat)

            penalty_factor[above_mask] = self.gamma_bound
            penalty_factor[below_mask] = -self.gamma_bound

            penalty_factor *= dX_hat
            grad_H += W.T @ penalty_factor

        return grad_H


class FourParamLogisticNMFConsensus:
    """
    A consensus-based wrapper class for tuning and running FourParamLogisticNMF
    using multiple parameter sets (via Optuna) and multiple runs.

    Parameters
    ----------
    a, b, c, d : float
        The 4PL parameters passed to FourParamLogisticNMF.
    max_iter : int
        Maximum number of iterations (passed on to the factorization).
    n_trials : int
        Number of Optuna trials to explore hyperparameters.
    top_n_params : int
        Number of best parameter sets to keep after tuning.

    Attributes
    ----------
    study : optuna.Study or None
        The Optuna study created when `tune` is called.
    top_params : list of dict or None
        The top parameter sets (hyperparameters) discovered by Optuna.

    Methods
    -------
    tune(X)
        Uses Optuna to find the `top_n_params` parameter sets with the lowest MSE on X.
    run_consensus(X, dfcw, id_vars_, value_name_, var_name_)
        Uses those parameter sets to factor X multiple times and merges the predictions
        into a final DataFrame with `inhibition_NMF`.
    """

    def __init__(
        self,
        dfcw,
        max_val,
        slope,
        log10IC50,
        min_val,
        max_iter=500,
        n_trials=100,
        top_n_params=10,
    ):
        self.X = dfcw  # make sure that we pass matrix before imputation
        self.a = min_val / 100
        self.b = slope
        self.c = 10**log10IC50
        self.d = max_val / 100
        self.max_iter = max_iter
        self.n_trials = n_trials
        self.top_n_params = top_n_params
        self.id_vars_ = ("dose_a",)
        self.var_name_ = ("dose_b",)
        self.value_name_ = "inhibition"

        self.study = None
        self.top_params = None

    def _objective(self, trial):
        """Objective function for Optuna to minimize mean squared error on observed entries."""
        # Hyperparameter search space:
        params = {
            "K": trial.suggest_int("K", 2, 4),
            "alpha": trial.suggest_float("alpha", 0.7, 1.0),
            "beta": trial.suggest_float("beta", 0.01, 0.1),
            "lambda_W": trial.suggest_float("lambda_W", 0.05, 0.3),
            "lambda_H": trial.suggest_float("lambda_H", 0.05, 0.3),
            "gamma_bound": trial.suggest_float("gamma_bound", 0.01, 1.0, log=True),
        }

        model = FourParamLogisticNMF(
            K=params["K"],
            a=self.a,
            b=self.b,
            c=self.c,
            d=self.d,
            alpha=params["alpha"],
            beta=params["beta"],
            lambda_W=params["lambda_W"],
            lambda_H=params["lambda_H"],
            max_iter=self.max_iter,
            gamma_bound=params["gamma_bound"],
        )

        pred = model.fit_transform(self.X)
        mask = ~np.isnan(self.X)
        loss = np.mean((self.X[mask] - pred[mask]) ** 2)
        return loss

    def tune(self):
        """
        Runs an Optuna study to find the best hyperparameters for FourParamLogisticNMF.

        Parameters
        ----------
        X : ndarray
            Input data matrix (with NaNs for missing entries).

        Returns
        -------
        top_params : list of dict
            The top parameter sets discovered by the study.
        """
        self.study = optuna.create_study(direction="minimize")
        self.study.optimize(
            lambda trial: self._objective(trial), n_trials=self.n_trials
        )

        # Sort trials by objective value (lower is better)
        best_trials = sorted(self.study.trials, key=lambda t: t.value)[
            : self.top_n_params
        ]
        self.top_params = [trial.params for trial in best_trials]
        return self.top_params

    def run_consensus(self):
        """
        Runs the factorization multiple times using the tuned hyperparameters,
        builds a 3D stack of predictions, then uses `bins_mode` to form a final
        consensus output.
        """
        # If no tuning done yet, tune automatically
        if self.top_params is None:
            self.tune()

        all_predictions = []
        for param_index, params in enumerate(self.top_params):
            # Try multiple runs per parameter set
            for run in range(10):
                np.random.seed(run)
                model = FourParamLogisticNMF(
                    K=params["K"],
                    a=self.a,
                    b=self.b,
                    c=self.c,
                    d=self.d,
                    alpha=params["alpha"],
                    beta=params["beta"],
                    lambda_W=params["lambda_W"],
                    lambda_H=params["lambda_H"],
                    max_iter=self.max_iter,
                )
                pred = model.fit_transform(self.X)
                all_predictions.append(pred)

        # Build 3D stack: shape (m, n, #runs*#top_params)
        prediction_stack = np.dstack(all_predictions)

        # Use bins_mode to build consensus
        dfcw_mfw = bins_mode(
            dfcw=self.X, prediction_stack=prediction_stack, num_bins=20
        )

        # Reshape for final insertion
        dfcw_mfm = dfcw_mfw.reset_index().melt(
            id_vars=[self.id_vars_],
            value_name=self.value_name_,
            var_name=self.var_name_,
        )
        dfcw_mfm = dfcw_mfm.sort_values(by=[self.id_vars_, self.var_name_]).reset_index(
            drop=True
        )

        return dfcw_mfm
