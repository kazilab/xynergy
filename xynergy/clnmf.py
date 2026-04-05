# This is the same as lnmf, just needed a place to put the current lnmf code and
# a place to copy it to, bit by bit


def lnmf(x):
    # Decompose via SVD
    #
    pass


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

    # 4pl function
    # 4pl derivative

    # Factorize matrix using partial svd, but fill unknowns with 0. If fail, use
    # random matrices as decomp


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


    """

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
            W = self._update(X_filled, W, H, mask_obs, mask_unobs, "W")
            H = self._update(X_filled, W, H, mask_obs, mask_unobs, "H")

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
    def _update(self, X, W, H, mask_obs, mask_unobs, which):
        grad = self._grad(X, W, H, mask_obs, mask_unobs, which)
        V = W if which == "W" else H
        new = V - self.eta * grad
        return np.maximum(0.0, new)

    def _grad(self, X, W, H, mask_obs, mask_unobs, which):
        """
        Gradient of the objective w.r.t. VARIES using cross-entropy + 4PL derivative.
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

        grad_H = W.T @ total_factor  # DIFF
        grad_W = total_factor @ H.T

        # L2 reg
        grad_H += 2.0 * self.lambda_H * H  # DIFF
        grad_W += 2.0 * self.lambda_W * W
        # penalty
        if self.gamma_bound > 0.0:
            above_mask = X_hat > 1.0
            below_mask = X_hat < 0.0
            penalty_factor = np.zeros_like(X_hat)

            penalty_factor[above_mask] = self.gamma_bound
            penalty_factor[below_mask] = -self.gamma_bound

            penalty_factor *= dX_hat
            grad_H += W.T @ penalty_factor  # DIFF
            grad_W += penalty_factor @ H.T

        return grad_W if which == "W" else grad_H


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
    run_consensus(X, dfcw, dfc, id_vars_, value_name_, var_name_)
        Uses those parameter sets to factor X multiple times and merges the predictions
        into a final DataFrame with `inhibition_NMF`.
    """

    def __init__(
        self,
        dfc,
        dfcw,
        max_val,
        slope,
        log10IC50,
        min_val,
        max_iter=500,
        n_trials=100,
        top_n_params=10,
    ):
        self.dfc = dfc
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
            "gamma_bound": trial.suggest_float("gamma_bound", 0.0, 1.0, log=True),
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

        Returns
        -------
        dfci : DataFrame
            The input `dfc` with an added column 'inhibition_NMF' after computing the consensus.
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

        # Find mode
        # Wrangle and append to orignal data, return
