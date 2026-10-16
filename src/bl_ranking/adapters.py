"""Thin subclasses of the researcher's classes. They change plumbing only — every override calls ``super()``
and no function body from the original modules is copied here.

Training (``TrainAdapter``):
- production mode trains on all rows by default (``holdout_days=0``); ``holdout_days=7`` is the original behaviour
- keeps the fitted models, test frames and row counts so the pipeline can log them
- turns the researcher's accuracy report into metrics (the report itself still goes to the original log file)
- caches the first-name → gender lookup (identical output, one NameDataset search per distinct name)
- keeps other root-logger handlers alive (the original clears them all)

Serving (``make_predict_adapter``):
- reuses models loaded once instead of loading (and refitting TabPFN) per request
- no log file per request; the global ``user_data`` the original reads is set under a lock
"""
from __future__ import annotations

import functools
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bl_ranking.original.bl_models_train import BLPayoutModelsFit
from bl_ranking.training.metrics import holdout_classification_metrics, payout_metrics

log = logging.getLogger(__name__)


def _preserving_root_handlers(setup):
    """Run the original logger setup without losing handlers other code installed on the root logger."""
    root = logging.getLogger()
    saved = list(root.handlers)
    logger = setup()
    for handler in saved:
        if handler not in root.handlers:
            root.addHandler(handler)
    return logger


class TrainAdapter(BLPayoutModelsFit):
    def __init__(self, input_path: str, input_file: str, output_predictors_path: str, train_test: bool,
                 holdout_days: int = 7):
        self.holdout_days = holdout_days
        self.metrics_: dict[str, float] = {}
        self.row_counts_: dict[str, int] = {}
        self._gender_cache: dict[str, tuple[str, float]] = {}
        self.model_tfm_ = self.tfm_context_ = self.cb_model_ = None
        self.x_test_all_days_ = self.y_test_all_days_ = None
        super().__init__(input_path, input_file, output_predictors_path, train_test)

    # --- plumbing -------------------------------------------------------------------------------------------
    def setup_bl_logger(self, log_dir):
        logger = _preserving_root_handlers(lambda: super(TrainAdapter, self).setup_bl_logger(log_dir))
        self.log_path_ = next(Path(h.baseFilename) for h in logger.handlers
                              if isinstance(h, logging.FileHandler)
                              and Path(h.baseFilename).parent == Path(log_dir).resolve())
        return logger

    def detect_gender_with_confidence(self, fname):
        if fname not in self._gender_cache:
            self._gender_cache[fname] = super().detect_gender_with_confidence(fname)
        return self._gender_cache[fname]

    def split_by_time(self, bl_data, days_for_test):
        return super().split_by_time(bl_data, days_for_test if self.train_test else self.holdout_days)

    # --- capture what the pipeline needs ------------------------------------------------------------------
    def import_preprocess(self):
        bl_data = super().import_preprocess()
        self.row_counts_["registered"] = len(bl_data)
        return bl_data

    def impute_survey_columns(self, bl_data):
        bl_data = super().impute_survey_columns(bl_data)
        self.row_counts_["after_survey_filter"] = len(bl_data)
        return bl_data

    def bl_preprocessing(self):
        x_train, y_train, x_test, y_test = super().bl_preprocessing()
        self.row_counts_.update(train=len(x_train), test=len(x_test))
        self.x_test_all_days_, self.y_test_all_days_ = x_test, y_test
        return x_train, y_train, x_test, y_test

    def catbosot_model_sold_to_client(self, x_train, y_train):
        self.cb_model_ = super().catbosot_model_sold_to_client(x_train, y_train)
        return self.cb_model_

    def tabpfn_regression_payout(self, x_train_payout, y_train_payout):
        self.model_tfm_, self.tfm_context_ = super().tabpfn_regression_payout(x_train_payout, y_train_payout)
        return self.model_tfm_, self.tfm_context_

    # --- researcher's accuracy report → metrics -------------------------------------------------------------
    def accuracy_classification_model(self, CB_model, y_test_all_days, x_test_all_days, target_col):
        super().accuracy_classification_model(CB_model, y_test_all_days, x_test_all_days, target_col)
        self.metrics_.update(holdout_classification_metrics(CB_model, x_test_all_days, y_test_all_days, target_col))

    def accuracy_cont_payout_prediction(self, model_tfm, x_test_payout, y_test_payout):
        try:
            super().accuracy_cont_payout_prediction(model_tfm, x_test_payout, y_test_payout)
        except ValueError as exc:  # sklearn on an empty threshold slice (possible with the fallback backend)
            log.warning("researcher payout report stopped early: %s", exc)
        # The original writes its predictions into the frame it was given (bl_models_train.py:302).
        self.metrics_.update(payout_metrics(y_test_payout["payout"], y_test_payout["pred_payout"]))


# ------------------------------------------------------------------------------------------------------------
# Serving
# ------------------------------------------------------------------------------------------------------------
# The predictor module builds its own NameDataset at import (~2.2 GB), so it is imported only when serving.
_USER_DATA_LOCK = threading.Lock()
_GENDER_CACHE: dict[str, tuple[str, float]] = {}
_GENDER_CACHE_MAX = 100_000


@dataclass
class LoadedModels:
    model_tfm: Any
    tfm_columns: list[str]
    cb_model: Any


def _predictor_module():
    from bl_ranking.original import bl_exp_payout_predictor

    return bl_exp_payout_predictor


@functools.cache
def _predict_adapter_cls():
    """The adapter class over the researcher's BLPayoutModelsPredict, created on first use (lazy import)."""
    module = _predictor_module()

    class PredictAdapter(module.BLPayoutModelsPredict):
        def __init__(self, predictors_path, user_data, loaded):
            # The original constructor opens a new log file per instance; on a request path we only need fields.
            self.predictors_path = predictors_path
            self.user_data = user_data
            self.user_data_file = user_data
            self.logger = logging.getLogger("bl_ranking.predictor")
            self._loaded = loaded

        def capture_warnings(self):
            return None

        def load_models(self):
            return self._loaded.model_tfm, self._loaded.tfm_columns, self._loaded.cb_model

        def import_preprocess(self):
            # The original reads the module-global `user_data` (bl_exp_payout_predictor.py:58). Holding the lock
            # while it builds its DataFrame copy keeps concurrent requests from seeing each other's data.
            with _USER_DATA_LOCK:
                module.user_data = self.user_data
                return super().import_preprocess()

        def detect_gender_with_confidence(self, fname):
            hit = _GENDER_CACHE.get(fname)
            if hit is None:
                hit = super().detect_gender_with_confidence(fname)
                if len(_GENDER_CACHE) < _GENDER_CACHE_MAX:
                    _GENDER_CACHE[fname] = hit
            return hit

    return PredictAdapter


def make_predict_adapter(predictors_path: str, user_data: dict, loaded: LoadedModels):
    return _predict_adapter_cls()(predictors_path, user_data, loaded)


def __getattr__(name: str):
    if name == "PredictAdapter":
        return _predict_adapter_cls()
    raise AttributeError(name)
