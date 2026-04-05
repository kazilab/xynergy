# This function is my shot at implementing the Tome et al. matrix factorization. (https://doi.org/10.1007/s11045-013-0240-9)
# Starting with their 'M1' since it's a bit simpler
import numpy as np

df = np.array(range(1, 26)).reshape(5, 5)


# Tome et al define this as a maximization problem - if we remove the negative
# sign, we make it a minimization problem.
def loss(X, W, H):
    X_hat = W @ H
    return np.log(1 + np.exp(-(2 * X - 1) * X_hat)).sum()


def update_w(X, W, H, learning_rate):
    X_hat = W @ H
    grad = ((2 * X - 1) / (1 + np.exp((2 * X - 1) * X_hat))) * np.transpose(H)
    W = W + learning_rate * grad
    return W


def update_h(X, W, H, learning_rate):
    X_hat = W @ H
    grad = np.transpose(W) * ((2 * X - 1) / (1 + np.exp((2 * X - 1) * X_hat)))
    H = H + learning_rate * grad
    return H


def my_lnmf(
    df,
    experiment_cols,
    response_col,
    dose_cols,
    K=2,
    max_iter=1000,
    learning_rate=0.001,
):
    W, H = initialize_factors(X, K)
    prev_obj = np.inf

    for _ in range(max_iter):
        W = update_w(X, W, H, learning_rate)
        H = update_h(X, W, H, learning_rate)
        obj = loss(X, W, H)
        if np.abs(prev_obj - obj) / max(1.0, np.abs(prev_obj)) < 1e-5:
            break
        prev_obj = obj

    return W, H


from .dev import rm_off_axis
import polars as pl

data = pl.read_csv("../data/150_high_low_medium_zip.csv")
experiments = data["experiment_id"].unique().sort()
ex_1 = data.filter(pl.col("experiment_id") == experiments[4])
ex_1_min = rm_off_axis(
    ex_1, ["conA", "conB"], ["batch", "line", "drug_a", "drug_b", "experiment_id"]
)

my_lnmf(ex_1_min, ["conA", "conB"], "response", ["batch", "line", "drug_a", "drug_b", "experiment_id"])

# Will need to do this removing the off-axis ones to test imputation

df_summary = _factor_by_group(
    df_summary, dose_cols, response_col, experiment_cols, _rpca, "RPCA"
)
