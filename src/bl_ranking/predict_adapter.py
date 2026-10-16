"""Serving-side adapter around the researcher's BLPayoutModelsPredict (plumbing only; every override calls super()).

- reuses models loaded once instead of loading (and refitting TabPFN) per request
- no log file per request; the module-global ``user_data`` the original reads is set under a lock
- the predictor module (and its NameDataset) is imported on first use, never by the trainer
"""
from __future__ import annotations

import functools
import logging
import threading
from dataclasses import dataclass
from typing import Any

# The predictor module builds its own NameDataset at import (~2.2 GB), so it is imported only when serving.
_USER_DATA_LOCK = threading.Lock()
_GENDER_CACHE: dict[str, tuple[str, float]] = {}
_GENDER_CACHE_MAX = 100_000


class InsufficientSurveyAnswers(Exception):
    """The original's survey filter (dropna thresh=5) removed the user; its next step would crash on the empty
    frame, so we stop here and the ranker answers with an empty ranking."""


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

        def impute_survey_columns(self, bl_data):
            bl_data = super().impute_survey_columns(bl_data)
            if bl_data.shape[0] == 0:
                raise InsufficientSurveyAnswers("fewer than 5 of 8 survey answers")
            return bl_data

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
