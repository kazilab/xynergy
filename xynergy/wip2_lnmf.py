# Based off Kazi's changed version of lnmf
import numpy as np
import scipy.linalg as la
import optuna
from xynergy.factor import _to_mat
import polars as pl


def use_lnmf(x, dose_cols, response_col):
    # Limited to taking a single experiment (for now) since that's what _to_mat takes
    #

    # Currently just using this to brute force use lnmf

    # Ensure no replicates
    x = x.group_by(dose_cols).agg(pl.col(response_col).mean())
    dfcw = _to_mat(x, dose_cols, response_col)
    obj = LogisticNMFConsensus(dfcw / 100)
    return obj.run_consensus()


class BaseLogisticNMF:
    """
    Base class providing logistic functionality and factor initialization.
    """

    def __init__(self):
        pass

    @staticmethod
    def logistic_function(z):
        """
        Standard sigmoid logistic function:
            f(z) = 1 / (1 + exp(-z)).
        """
        p = 1 / (1 + np.exp(-z))
        return p

    @staticmethod
    def logistic_derivative(z):
        """
        Derivative of the logistic function wrt z.
        p = logistic_function(z)
        p'(z) = p * (1 - p)
        """
        p = BaseLogisticNMF.logistic_function(z)
        delta = p * (1 - p)
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

        # Normalize columns of W
        col_norms = np.sqrt((W**2).sum(axis=0))
        col_norms[col_norms < 1e-9] = 1.0
        W /= col_norms
        H *= col_norms.reshape(-1, 1)

        return W, H


class LogisticNMF(BaseLogisticNMF):
    """
    Logistic Consensus Non-negative Matrix Factorization (LC-NMF) model, using a cross-entropy-based objective for observed entries
    and partial SVD for initialization.

    The logistic function is:
        f(z) = 1 / (1 + exp(-z)).

    Cross-entropy form for observed entries:
        L_obs = alpha * sum( X_ij * log(X_hat_ij) + (1 - X_ij) * log(1 - X_hat_ij) ),
        where X_hat_ij = logistic_function( (WH)_ij ).

    Unobserved entries use a weaker penalty:
        L_unobs = beta * sum( log(1 - X_hat_ij) ).

    Regularization:
        lambda_W * ||W||^2 + lambda_H * ||H||^2.

    We invert the sign for gradient descent on the negative log-likelihood.

    Parameters
    ----------
    K : int
        Number of latent factors.
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
        and returns the logistic-transformed reconstruction.
    """

    def __init__(
        self,
        K,
        alpha=1.0,
        beta=0.01,
        lambda_W=0.1,
        lambda_H=0.1,
        max_iter=500,
        eta=0.01,
        gamma_bound=0.0,
    ):
        super().__init__()
        self.K = K
        self.alpha, self.beta = alpha, beta
        self.lambda_W, self.lambda_H = lambda_W, lambda_H
        self.max_iter, self.eta = max_iter, eta
        self.gamma_bound = gamma_bound
        self.W_ = None
        self.H_ = None

    def fit_transform(self, X):
        """
        Fit model to X and return the logistic-transformed reconstruction.
        Missing entries in X are indicated by NaN.
        """
        mask_obs = ~np.isnan(X)  # Observed data
        mask_unobs = np.isnan(X)  # Unobserved / missing data
        X_filled = np.nan_to_num(X, nan=0.0)
        W, H = self._initialize_factors(X, self.K)
        prev_obj = np.inf

        for _iter in range(self.max_iter):
            W = self._update_W(X_filled, W, H, mask_obs, mask_unobs)
            H = self._update_H(X_filled, W, H, mask_obs, mask_unobs)
            obj = self._objective(X_filled, W, H, mask_obs, mask_unobs)
            if np.abs(prev_obj - obj) / max(1.0, np.abs(prev_obj)) < 1e-5:
                break
            prev_obj = obj

        self.W_, self.H_ = W, H
        return self.logistic_function(W @ H)

    def _objective(self, X, W, H, mask_obs, mask_unobs):
        WH = W @ H
        X_hat = WH
        eps = 1e-12
        p = np.clip(X_hat, eps, 1 - eps)
        X_obs = X[mask_obs]
        ce_obs = -(X_obs * np.log(p[mask_obs]) + (1 - X_obs) * np.log(1 - p[mask_obs]))
        ce_obs_sum = ce_obs.sum()
        unobs_term = -np.log(1 - p[mask_unobs] + eps)
        unobs_sum = unobs_term.sum()
        reg = self.lambda_W * (W**2).sum() + self.lambda_H * (H**2).sum()
        if self.gamma_bound > 0.0:
            over_1 = np.clip(X_hat - 1.0, a_min=0.0, a_max=None)
            under_0 = np.clip(-X_hat, a_min=0.0, a_max=None)
            penalty_out_of_bounds = (over_1.sum() + under_0.sum()) * self.gamma_bound
        else:
            penalty_out_of_bounds = 0.0
        obj = (
            self.alpha * ce_obs_sum
            + self.beta * unobs_sum
            + reg
            + penalty_out_of_bounds
        )
        return obj

    def _update_W(self, X, W, H, mask_obs, mask_unobs):
        grad = self._grad_W(X, W, H, mask_obs, mask_unobs)
        W_new = W - self.eta * grad
        return np.maximum(0.0, W_new)

    def _update_H(self, X, W, H, mask_obs, mask_unobs):
        grad = self._grad_H(X, W, H, mask_obs, mask_unobs)
        H_new = H - self.eta * grad
        return np.maximum(0.0, H_new)

    def _grad_W(self, X, W, H, mask_obs, mask_unobs):
        WH = W @ H
        X_hat = self.logistic_function(WH)
        dX_hat = self.logistic_derivative(WH)
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
        grad_W = total_factor @ H.T
        grad_W += 2.0 * self.lambda_W * W
        if self.gamma_bound > 0.0:
            above_mask = X_hat > 1.0
            below_mask = X_hat < 0.0
            penalty_factor = np.zeros_like(X_hat)
            penalty_factor[above_mask] = self.gamma_bound
            penalty_factor[below_mask] = -self.gamma_bound
            penalty_factor *= dX_hat
            grad_W += penalty_factor @ H.T
        return grad_W

    def _grad_H(self, X, W, H, mask_obs, mask_unobs):
        WH = W @ H
        X_hat = self.logistic_function(WH)
        dX_hat = self.logistic_derivative(WH)
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
        grad_H += 2.0 * self.lambda_H * H
        if self.gamma_bound > 0.0:
            above_mask = X_hat > 1.0
            below_mask = X_hat < 0.0
            penalty_factor = np.zeros_like(X_hat)
            penalty_factor[above_mask] = self.gamma_bound
            penalty_factor[below_mask] = -self.gamma_bound
            penalty_factor *= dX_hat
            grad_H += W.T @ penalty_factor
        return grad_H


class LogisticNMFConsensus:
    """
    A consensus-based wrapper class for tuning and running LogisticNMF
    using multiple parameter sets (via Optuna) and multiple runs.

    Parameters
    ----------
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
    run_consensus(X, dfcw, dfc, id_vars_, value_name_, var_name_)
        Uses those parameter sets to factor X multiple times and merges the predictions
        into a final DataFrame with `inhibition_NMF`.
    """

    def __init__(self, dfcw, max_iter=500, n_trials=100, top_n_params=10):
        self.X = dfcw  # matrix before imputation
        self.max_iter = max_iter
        self.n_trials = n_trials
        self.top_n_params = top_n_params
        self.id_vars_ = "conA"
        self.var_name_ = "conB"
        self.value_name_ = "inhibition"
        self.study = None
        self.top_params = None

    def _objective(self, trial):
        """Objective function for Optuna to minimize mean squared error on observed entries."""
        params = {
            "K": trial.suggest_int("K", 2, 4),
            "alpha": trial.suggest_float("alpha", 0.7, 1.0),
            "beta": trial.suggest_float("beta", 0.01, 0.1),
            "lambda_W": trial.suggest_float("lambda_W", 0.05, 0.3),
            "lambda_H": trial.suggest_float("lambda_H", 0.05, 0.3),
            "gamma_bound": trial.suggest_float("gamma_bound", 0.0001, 1.0, log=True),
        }
        model = LogisticNMF(
            K=params["K"],
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
        Runs an Optuna study to find the best hyperparameters for LogisticNMF.
        Returns the top parameter sets discovered by the study.
        """
        self.study = optuna.create_study(direction="minimize")
        self.study.optimize(
            lambda trial: self._objective(trial), n_trials=self.n_trials
        )
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

        Returns
        -------
        dfci : DataFrame
            The input `dfc` with an added column 'inhibition_NMF' after computing the consensus.
        """
        if self.top_params is None:
            self.tune()

        all_predictions = []
        for param_index, params in enumerate(self.top_params):
            for run in range(10):
                np.random.seed(run)
                model = LogisticNMF(
                    K=params["K"],
                    alpha=params["alpha"],
                    beta=params["beta"],
                    lambda_W=params["lambda_W"],
                    lambda_H=params["lambda_H"],
                    max_iter=self.max_iter,
                    gamma_bound=params["gamma_bound"],
                )
                pred = model.fit_transform(self.X)
                all_predictions.append(pred)

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
