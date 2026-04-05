import numpy as np
import scipy.linalg as la
import polars as pl
from xynergy.util import venter

try:
    import optuna
except ModuleNotFoundError:
    optuna = None


def _lnmf(x, R, max_iter=500, n_trials=200, top_n=10):
    if optuna is None:
        raise ModuleNotFoundError(
            "optuna is required for LNMF factorization. Install optuna to use LNMF."
        )

    study = optuna.create_study()
    study.optimize(
        lambda trial: _optuna_objective(trial, x, R, max_iter), n_trials=n_trials
    )
    best_trials = sorted(study.trials, key=lambda t: t.value)[:top_n]
    best_params = [trial.params for trial in best_trials]
    best_param_fits = pl.DataFrame()

    for i, params in enumerate(best_params):
        predicted_matrix, _ = _fit(x, R, max_iter, params)
        out = pl.DataFrame(predicted_matrix).unpivot().drop("variable")
        out.columns = [str(i)]
        best_param_fits = pl.concat([best_param_fits, out], how="horizontal")

    modes = best_param_fits.transpose().select(pl.all().map_batches(venter)).transpose()
    return modes.to_numpy().reshape(x.shape).transpose()


# Returns fit and objective
def _fit(X, R, max_iter, args, eps=1e-12):
    X = np.clip(X / 100, eps, 1 - eps)  # Convert from range of 0-100 to 0-1
    X = np.log(X / (1 - X))  # Convert to log odds

    K = args.pop("K")
    W, H = _initialize_factors(X, R, K)

    args["X"] = X
    args["W"] = W
    args["H"] = H
    args["R"] = R

    predicted_matrix, obj = _descend(args, max_iter)
    return predicted_matrix, obj


def _optuna_objective(trial, X, R, max_iter, eps=1e-12):
    X = np.clip(X / 100, eps, 1 - eps)  # Convert from range of 0-100 to 0-1
    X = np.log(X / (1 - X))  # Convert to log odds

    K = trial.suggest_int("K", 2, 4)
    W, H = _initialize_factors(X, R, K)
    args = {
        "X": X,
        "W": W,
        "H": H,
        "R": R,
        "obs_penalty": trial.suggest_float("obs_penalty", 0.7, 1.0),
        "unobs_penalty": trial.suggest_float("unobs_penalty", 0.01, 0.1),
        "w_penalty": trial.suggest_float("w_penalty", 0.05, 0.3),
        "h_penalty": trial.suggest_float("h_penalty", 0.05, 0.3),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
    }
    _, obj = _descend(args, max_iter)
    return obj


def _descend(args, max_iter):
    prev_obj = np.inf
    for i in range(max_iter):
        W = _update(**args, respect_to="W")
        H = _update(**args, respect_to="H")
        if np.any(np.isinf(W)):
            raise ValueError("W contains infinite values during LNMF descent")
        args["W"] = W
        args["H"] = H
        obj = _objective(**args)
        if np.abs(prev_obj - obj) / max(1.0, np.abs(prev_obj)) < 1e-5:
            break
        prev_obj = obj

    predicted_matrix = _log(W @ H) * 100
    return predicted_matrix, obj


def _initialize_factors(X, R, K):
    # NOTE: What if we used the pre-imputed data here?

    # Fill missing with 0 for SVD init
    X = X * R

    U, s, Vt = la.svd(X, full_matrices=False)
    U = U[:, :K]
    s = s[:K]
    Vt = Vt[:K, :]
    W = U
    H = np.diag(s) @ Vt

    # NOTE: Kazi normalizes the columns of W, but I won't do that unless it
    # seems necessary

    return W, H


def _update(
    X,
    W,
    H,
    R,
    obs_penalty,
    unobs_penalty,
    w_penalty,
    h_penalty,
    learning_rate,
    respect_to,
    eps=1e-12,
):
    X_hat = W @ H
    Y = _log(X_hat)
    Y = np.clip(Y, eps, 1 - eps)
    dY = _log_derivative(X_hat)

    # Log-loss part
    penalties = obs_penalty * R + unobs_penalty * abs(R - 1)
    factor = penalties * dY * (Y - _log(X)) / (Y * (1 - Y))

    if respect_to == "H":
        ll_gradient = W.T @ factor
    else:
        ll_gradient = factor @ H.T

    # Regularization part
    if respect_to == "H":
        reg_gradient = 2 * h_penalty * H
    else:
        reg_gradient = 2 * w_penalty * W

    gradient = ll_gradient + reg_gradient

    if respect_to == "H":
        out = H - learning_rate * gradient
    else:
        out = W - learning_rate * gradient

    # In general, enforcing non-negativity resulted in worse RMSE with real
    # world data, so we're going to forgo that.

    return out


def _objective(
    X, W, H, R, obs_penalty, unobs_penalty, w_penalty, h_penalty, learning_rate
):
    """Objective function

    Parameters
    ----------
    X : np.array
        Matrix of responses, on a range of roughly 0-1 (that is, not from 0-100)
    W : np.array
    H : np.array
    R : np.array
        R is an indicator matrix. It indicates if a given datapoint was present
        in the original data set
    """
    obs = R
    unobs = abs(R - 1)

    # Reconstruct X_hat from factors
    X_hat = W @ H
    # NOTE: This step is to make it closer to Larsen and Clemmensen but is
    # different from Kazi's solution
    Y = _log(X_hat)

    # Log-loss
    log_loss = _log_loss(_log(X), Y)
    obs_ll = (obs_penalty * obs * log_loss).sum()
    unobs_ll = (unobs_penalty * unobs * log_loss).sum()

    # L2 regularization
    w_reg = w_penalty * (W**2).sum()
    h_reg = h_penalty * (H**2).sum()

    objective = obs_ll + unobs_ll + w_reg + h_reg
    return objective


def _log(x):
    return 1 / (1 + np.exp(-x))


def _log_derivative(x):
    return np.exp(-x) / ((1 + np.exp(-x)) ** 2)


def _log_loss(X, Y):
    return -(X * np.log(Y) + (1 - X) * np.log(1 - Y))
