from importlib import import_module

from ._meta import (
    APP_NAME,
    APP_TAGLINE,
    CONTACT_EMAIL,
    COPYRIGHT_HOLDER,
    DEVELOPED_BY,
    PACKAGE_NAME,
    __version__,
)

_EXPORTS = {
    "add_synergy": ("xynergy.synergy", "add_synergy"),
    "add_reference": ("xynergy.reference", "add_reference"),
    "plot_response_landscape": ("xynergy.plot", "plot_response_landscape"),
    "pre_impute": ("xynergy.impute", "pre_impute"),
    "post_impute": ("xynergy.impute", "post_impute"),
    "add_uncombined_drug_responses": ("xynergy.fit", "add_uncombined_drug_responses"),
    "fit_individual_drugs": ("xynergy.fit", "fit_individual_drugs"),
    "fit_curve": ("xynergy.fit", "fit_curve"),
    "ll4": ("xynergy.fit", "ll4"),
    "inverse_ll4": ("xynergy.fit", "inverse_ll4"),
    "matrix_factorize": ("xynergy.factor", "matrix_factorize"),
    "mf_combination": ("xynergy.factor", "mf_combination"),
    "get_example_xynergy_kwargs": ("xynergy.example", "get_example_xynergy_kwargs"),
    "load_example_data": ("xynergy.example", "load_example_data"),
    "make_example_data": ("xynergy.example", "make_example_data"),
    "tidy": ("xynergy.tidy", "tidy"),
    "xynergy": ("xynergy.xynergy", "xynergy"),
    "quality_scores": ("xynergy.scores", "quality_scores"),
    "cal_auc_aac": ("xynergy.scores", "cal_auc_aac"),
    "xepto_score": ("xynergy.scores", "xepto_score"),
    "auc_from_params": ("xynergy.scores", "auc_from_params"),
    "dss": ("xynergy.scores", "dss"),
    "xeptosync": ("xynergy.scores", "xeptosync"),
    "unit_conversion": ("xynergy.util", "unit_conversion"),
    "outlier_remove": ("xynergy.util", "outlier_remove"),
    "remove_row_outliers": ("xynergy.util", "remove_row_outliers"),
    "xplot": ("xynergy.mpl_plots", "xplot"),
    "synergy_plots": ("xynergy.mpl_plots", "synergy_plots"),
    "synergy2plots": ("xynergy.mpl_plots", "synergy2plots"),
}

__all__ = sorted(
    [
        *_EXPORTS,
        "APP_NAME",
        "APP_TAGLINE",
        "CONTACT_EMAIL",
        "COPYRIGHT_HOLDER",
        "DEVELOPED_BY",
        "PACKAGE_NAME",
        "__version__",
    ]
)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
