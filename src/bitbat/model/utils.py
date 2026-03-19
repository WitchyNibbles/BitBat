"""Shared utilities for model training and evaluation."""

from __future__ import annotations

import pandas as pd
import xgboost as xgb

from bitbat.model.train import DIRECTION_CLASSES


def is_classification_target(y: pd.Series) -> bool:
    """Return True if the target matches DIRECTION_CLASSES."""
    allowed = set(DIRECTION_CLASSES)
    values = {str(value) for value in y.dropna().unique()}
    return bool(values) and values.issubset(allowed)


def create_dmatrices(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_te: pd.DataFrame,
    y_te: pd.Series,
    feature_names: list[str],
    classification_mode: bool,
) -> tuple[xgb.DMatrix, xgb.DMatrix, pd.Series, pd.Series]:
    """Format splits into DMatrix objects and returned normalized labels.

    Returns:
        (dtrain, dtest, y_tr_labels, y_te_labels) for classification,
        (dtrain, dtest, y_tr_values, y_te_values) for regression.
    """
    if classification_mode:
        y_tr_labels = y_tr.astype(str)
        y_te_labels = y_te.astype(str)
        y_tr_values = y_tr_labels.map(DIRECTION_CLASSES).astype(int).to_numpy()
        y_te_values = y_te_labels.map(DIRECTION_CLASSES).astype(int).to_numpy()
        dtrain = xgb.DMatrix(X_tr, label=y_tr_values, feature_names=feature_names)
        dtest = xgb.DMatrix(X_te, label=y_te_values, feature_names=feature_names)
        return dtrain, dtest, y_tr_labels, y_te_labels
    else:
        y_tr_values = y_tr.astype("float64")
        y_te_values = y_te.astype("float64")
        dtrain = xgb.DMatrix(X_tr, label=y_tr_values.to_numpy(), feature_names=feature_names)
        dtest = xgb.DMatrix(X_te, label=y_te_values.to_numpy(), feature_names=feature_names)
        return dtrain, dtest, y_tr_values, y_te_values
