# xynergy

<!-- PyPI version badge -->
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://xynergy.streamlit.app)
[![PyPI version](https://img.shields.io/pypi/v/xynergy.svg)](https://pypi.org/project/xynergy/)
[![Documentation Status](https://readthedocs.org/projects/xynergy/badge/?version=latest)](https://xynergy.readthedocs.io/en/latest/?badge=latest)
<!-- PyPI version badge -->
[![@KaziLab.se](https://img.shields.io/website?url=https://www.kazilab.se/)](https://www.kazilab.se/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-kazilab%2Fxynergy-181717?logo=github&logoColor=white)](https://github.com/kazilab/xynergy)
<!-- PyPI version badge -->

Fast and high throughput Drug Synergy Prediction from Minimal Combination Data via Radial Basis Function Surface Interpolation combined with NMF and XGBoost

## Workflow

A typical workflow will involve calling several functions in sequence:

1. `tidy()` - Coaxes your data into a format suitable for downstream analysis
2. `pre_impute()` - Fills in missing experimental points, largely to give full matrices for the next step to factorize
3. `matrix_factorize()` - Approximate the imputed matrix via various matrix factorization algorithms
4. `post_impute()` - Create a final imputed dataframe using matrix factorization results
5. `add_synergy()` - Add columns calculating the synergy at each point under various assumptions (Bliss, Loewe, HSA, and/or ZIP)

xynergy enforces relatively few constraints on the form of your data and in theory allows you to shim in your own data transformations between any of these steps (whether that is a good idea or not is not for me to decide).

An example workflow looks something like this:

``` python
import xynergy as xyn
import xynergy.example as ex

data = ex.load_example_data()

# Normalizes column names so we can use default arguments for downstream functions
clean_data = xyn.tidy(
    data,
    dose_cols=["dose_a", "dose_b"],
    response_col="response",
    experiment_cols=["experiment_source_id", "line", "drug_a", "drug_b", "pair_index"],
)
imputed = xyn.pre_impute(clean_data, method="XGBR")
factored = xyn.matrix_factorize(imputed, method=["SVD", "NMF"])
final = xyn.post_impute(factored)
with_synergy = xyn.add_synergy(final, method=["bliss", "zip"])

```

The bundled workbook lives at `xynergy/example_data/data.xlsx` and is loaded by
`xynergy.example.load_example_data()`. See the tutorial "Using Xynergy" in the
Sphinx documentation for a fuller walkthrough.

## Acknowledgments

This work stands on the shoulders of so many things. In addition to the software packages explicitly and implicitly used in this repo, some code/equations came from other places: 
- The cNMF-like algorithm that came from DECREASE (DOI: 10.1038/s42256-019-0122-4)
- The Venter-mode code that came from the R {modeest} package (DOI: 10.32614/CRAN.package.modeest)
- The closed-form Loewe additivity algorithm from 10.3389/fphar.2018.00031
