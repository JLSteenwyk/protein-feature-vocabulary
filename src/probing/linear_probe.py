"""Linear probing for biological properties.

Train simple linear classifiers/regressors on model activations
to measure what information is linearly accessible at each layer.
"""

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
)
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass


@dataclass
class ProbeResult:
    """Result of a single probing experiment."""
    property_name: str
    layer: int
    model_name: str
    accuracy: float | None = None
    f1: float | None = None
    auroc: float | None = None
    mse: float | None = None
    task_type: str = "classification"  # "classification" or "regression"

    def summary(self) -> str:
        if self.task_type == "classification":
            return (
                f"{self.model_name} layer {self.layer} | {self.property_name}: "
                f"acc={self.accuracy:.3f} f1={self.f1:.3f}"
            )
        return (
            f"{self.model_name} layer {self.layer} | {self.property_name}: "
            f"mse={self.mse:.4f}"
        )


def train_classification_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    property_name: str,
    layer: int,
    model_name: str,
    C: float = 1.0,
    max_iter: int = 1000,
) -> ProbeResult:
    """Train a logistic regression probe for classification.

    Args:
        X_train: (N, D) training activations
        y_train: (N,) training labels
        X_test: (M, D) test activations
        y_test: (M,) test labels
        property_name: name of the biological property
        layer: layer index
        model_name: model identifier
        C: inverse regularization strength
        max_iter: max iterations for solver

    Returns:
        ProbeResult with metrics
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    n_classes = len(np.unique(y_train))
    multi_class = "multinomial" if n_classes > 2 else "auto"

    clf = LogisticRegression(
        C=C,
        max_iter=max_iter,
        multi_class=multi_class,
        solver="lbfgs",
        n_jobs=-1,
    )
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    auroc = None
    if n_classes == 2:
        y_prob = clf.predict_proba(X_test_scaled)[:, 1]
        auroc = roc_auc_score(y_test, y_prob)

    return ProbeResult(
        property_name=property_name,
        layer=layer,
        model_name=model_name,
        accuracy=acc,
        f1=f1,
        auroc=auroc,
        task_type="classification",
    )


def train_regression_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    property_name: str,
    layer: int,
    model_name: str,
    alpha: float = 1.0,
) -> ProbeResult:
    """Train a Ridge regression probe.

    Args:
        X_train: (N, D) training activations
        y_train: (N,) training targets
        X_test: (M, D) test activations
        y_test: (M,) test targets
        property_name: name of the biological property
        layer: layer index
        model_name: model identifier
        alpha: regularization strength

    Returns:
        ProbeResult with metrics
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    reg = Ridge(alpha=alpha)
    reg.fit(X_train_scaled, y_train)
    y_pred = reg.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)

    return ProbeResult(
        property_name=property_name,
        layer=layer,
        model_name=model_name,
        mse=mse,
        task_type="regression",
    )
