import numpy as np
import polars as pl
import pytest

from xynergy import pre_impute
import xynergy.impute as impute_module


def _toy_df_with_missing():
    doses = [0.0, 1.0, 10.0, 100.0]
    single_resp = {0.0: 0.0, 1.0: 20.0, 10.0: 60.0, 100.0: 90.0}

    rows = []
    for dose_a in doses:
        for dose_b in doses:
            bliss = (
                single_resp[dose_a]
                + single_resp[dose_b]
                - single_resp[dose_a] * single_resp[dose_b] / 100
            )
            response = None if (dose_a, dose_b) == (1.0, 1.0) else bliss
            rows.append(
                {"dose_a": dose_a, "dose_b": dose_b, "response": response}
            )

    return pl.DataFrame(rows)


def _toy_df_with_missing_replicates():
    return pl.DataFrame(
        {
            "obs_id": list(range(8)),
            "dose_a": [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            "dose_b": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            "response": [0.0, 20.0, 30.0, None, 0.0, 20.0, 30.0, None],
        }
    )


def _toy_post_impute_df(response_col: str = "response"):
    return pl.DataFrame(
        {
            "experiment_id": [1, 1, 1, 1],
            "dose_a": [0.0, 1.0, 10.0, 100.0],
            "dose_b": [0.0, 1.0, 10.0, 100.0],
            response_col: [0.0, 20.0, 60.0, None],
            "resp_imputed_NMF": [0.0, 20.0, 60.0, 80.0],
        }
    )


def _toy_post_impute_df_two_observed(response_col: str = "response"):
    return pl.DataFrame(
        {
            "experiment_id": [1, 1, 1],
            "dose_a": [0.0, 1.0, 10.0],
            "dose_b": [0.0, 1.0, 10.0],
            response_col: [0.0, 20.0, None],
            "resp_imputed_NMF": [0.0, 20.0, 80.0],
        }
    )


class TestPreImpute:
    def test_combo_effect_target_adds_expected_columns(self):
        df = _toy_df_with_missing()
        out = pre_impute(
            df,
            method="IterativeImputer",
            target="combo_effect",
            reference_for_target="bliss",
        )

        assert "resp_imputed" in out.columns
        assert "combo_effect_imputed" in out.columns
        assert "response_imputed_from_effect" in out.columns
        assert "dose_a_resp" in out.columns
        assert "dose_b_resp" in out.columns

        reconstructed = (
            out["dose_a_resp"]
            + out["dose_b_resp"]
            - (out["dose_a_resp"] * out["dose_b_resp"] / 100)
            + out["combo_effect_imputed"]
        )

        assert np.allclose(reconstructed, out["response_imputed_from_effect"], equal_nan=True)
        assert np.allclose(out["resp_imputed"], out["response_imputed_from_effect"])

    def test_combo_effect_target_without_single_drug_features_still_works(self):
        df = _toy_df_with_missing()
        out = pre_impute(
            df,
            method="IterativeImputer",
            target="combo_effect",
            reference_for_target="hsa",
            use_single_drug_response_data=False,
        )

        assert "resp_imputed" in out.columns
        assert "combo_effect_imputed" in out.columns
        assert "response_imputed_from_effect" in out.columns
        assert "_target_reference" not in out.columns
        assert "_combo_effect_target" not in out.columns

    def test_bad_reference_for_combo_effect_raises(self):
        df = _toy_df_with_missing()
        try:
            pre_impute(
                df,
                method="IterativeImputer",
                target="combo_effect",
                reference_for_target="zip",
            )
            assert False, "Expected ValueError for invalid reference_for_target"
        except ValueError as e:
            assert "reference_for_target" in str(e)

    def test_ensemble_target_adds_component_columns_and_blends_predictions(self):
        df = _toy_df_with_missing()
        out = pre_impute(
            df,
            method="IterativeImputer",
            target="ensemble",
            ensemble_response_weight=0.25,
        )

        assert "resp_imputed" in out.columns
        assert "resp_imputed_response" in out.columns
        assert "resp_imputed_combo_effect" in out.columns
        assert "resp_imputed_ensemble" in out.columns
        assert "ensemble_response_weight" in out.columns
        assert "ensemble_combo_effect_weight" in out.columns

        reconstructed = (
            out["ensemble_response_weight"] * out["resp_imputed_response"]
            + out["ensemble_combo_effect_weight"] * out["resp_imputed_combo_effect"]
        )

        assert np.allclose(reconstructed, out["resp_imputed_ensemble"], equal_nan=True)
        assert np.allclose(out["resp_imputed"], out["resp_imputed_ensemble"], equal_nan=True)

    def test_ensemble_weight_out_of_bounds_raises(self):
        df = _toy_df_with_missing()
        try:
            pre_impute(
                df,
                method="IterativeImputer",
                target="ensemble",
                ensemble_response_weight=1.5,
            )
            assert False, "Expected ValueError for invalid ensemble_response_weight"
        except ValueError as e:
            assert "ensemble_response_weight" in str(e)

    def test_replicates_do_not_duplicate_rows_on_rejoin(self):
        df = _toy_df_with_missing_replicates()
        out = pre_impute(df, method="IterativeImputer")

        assert out.height == df.height
        assert out["obs_id"].n_unique() == df.height
        assert out["obs_id"].sort().to_list() == df["obs_id"].to_list()

    def test_gaussian_process_surface_pins_observed_values(self):
        df = _toy_df_with_missing()
        out = pre_impute(df, method="GaussianProcessSurface", target="response")

        observed_in = df.filter(pl.col("response").is_not_null()).sort(["dose_a", "dose_b"])
        observed_out = (
            out.filter(pl.col("response").is_not_null())
            .sort(["dose_a", "dose_b"])
        )

        assert np.allclose(observed_in["response"], observed_out["resp_imputed"])

    def test_gaussian_process_surface_combo_effect_reconstructs_bliss_case(self):
        df = _toy_df_with_missing()
        out = pre_impute(
            df,
            method="GaussianProcessSurface",
            target="combo_effect",
            reference_for_target="bliss",
        )

        missing = out.filter((pl.col("dose_a") == 1.0) & (pl.col("dose_b") == 1.0))

        assert missing.height == 1
        assert "combo_effect_imputed" in out.columns
        assert "response_imputed_from_effect" in out.columns
        assert abs(missing["combo_effect_imputed"][0]) < 1e-6
        assert abs(missing["resp_imputed"][0] - 36.0) < 1e-6


class TestPostImpute:
    def test_predefined_uses_n_estimators(self, monkeypatch):
        created = []

        class FakeXGBRegressor:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                created.append(self)

            def fit(self, X, y):
                return self

            def predict(self, X):
                return np.zeros(len(X), dtype=float)

        monkeypatch.setattr(impute_module, "XGBRegressor", FakeXGBRegressor)

        out = impute_module.post_impute(
            _toy_post_impute_df(),
            imputed_response_cols=["resp_imputed_NMF"],
            post_impute_tuning="Predefined",
            log="none",
        )

        assert out["response"].null_count() == 0
        assert created[0].kwargs["n_estimators"] == 50
        assert "n_estimator" not in created[0].kwargs

    @pytest.mark.parametrize(
        ("tuning", "search_attr"),
        [
            ("RandomizedSearchCV", "RandomizedSearchCV"),
            ("GridSearchCV", "GridSearchCV"),
        ],
    )
    def test_search_modes_use_response_col_and_n_estimators(
        self, monkeypatch, tuning, search_attr
    ):
        created = []

        class FakeXGBRegressor:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.fit_targets = None
                created.append(self)

            def fit(self, X, y):
                self.fit_targets = list(y)
                return self

            def predict(self, X):
                return np.full(len(X), 42.0, dtype=float)

        class FakeSearchCV:
            def __init__(self, estimator, **kwargs):
                self.estimator = estimator
                self.kwargs = kwargs
                self.cv_results_ = None

            def fit(self, X, y):
                self.cv_results_ = {
                    "rank_test_score": [1, 2],
                    "params": [
                        {"max_depth": 2, "n_estimators": 25},
                        {"max_depth": 3, "n_estimators": 50},
                    ],
                }
                return self

        monkeypatch.setattr(impute_module, "XGBRegressor", FakeXGBRegressor)
        monkeypatch.setattr(impute_module, search_attr, FakeSearchCV)

        out = impute_module.post_impute(
            _toy_post_impute_df(response_col="target"),
            response_col="target",
            imputed_response_cols=["resp_imputed_NMF"],
            post_impute_tuning=tuning,
            log="none",
        )

        assert out["target"].null_count() == 0
        retrained_models = created[1:]
        assert retrained_models
        assert all(model.fit_targets == [0.0, 20.0, 60.0] for model in retrained_models)
        assert all("n_estimators" in model.kwargs for model in retrained_models)

    def test_search_modes_reduce_cv_for_small_groups(self, monkeypatch):
        search_kwargs = {}

        class FakeXGBRegressor:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def fit(self, X, y):
                return self

            def predict(self, X):
                return np.full(len(X), 42.0, dtype=float)

        class FakeSearchCV:
            def __init__(self, estimator, **kwargs):
                search_kwargs.update(kwargs)
                self.cv_results_ = None

            def fit(self, X, y):
                self.cv_results_ = {
                    "rank_test_score": [1],
                    "params": [{"max_depth": 2, "n_estimators": 25}],
                }
                return self

        monkeypatch.setattr(impute_module, "XGBRegressor", FakeXGBRegressor)
        monkeypatch.setattr(impute_module, "RandomizedSearchCV", FakeSearchCV)

        out = impute_module.post_impute(
            _toy_post_impute_df_two_observed(),
            imputed_response_cols=["resp_imputed_NMF"],
            post_impute_tuning="RandomizedSearchCV",
            log="none",
        )

        assert out["response"].null_count() == 0
        assert search_kwargs["cv"] == 2
