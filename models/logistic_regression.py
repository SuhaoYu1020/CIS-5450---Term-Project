from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class LogisticRegressionConfig:
    """Hyper-parameters for the baseline logistic regression model."""

    C: float = 1.0
    penalty: str = "l2"
    solver: str = "lbfgs"  # 使用 lbfgs 以支持 warm_start 和进度条
    max_iter: int = 1000
    class_weight: Optional[Dict[int, float] | str] = "balanced"
    random_state: int = 42
    verbose: int = 0
    warm_start: bool = True  # 启用 warm_start 以支持进度条显示
    # Preprocessing
    impute_strategy: str = "median"
    with_mean: bool = True
    with_std: bool = True


class LogisticRegressionModel:
    """Encapsulated, reusable binary classification model based on logistic regression."""

    def __init__(self, config: Optional[LogisticRegressionConfig] = None) -> None:
        self.config = config or LogisticRegressionConfig()
        self.pipeline: Optional[Pipeline] = None
        self._build_pipeline()

    def _build_pipeline(self) -> None:
        """Construct the sklearn pipeline with imputation, scaling and classifier."""
        self.pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy=self.config.impute_strategy)),
                ("scaler", StandardScaler(with_mean=self.config.with_mean, with_std=self.config.with_std)),
                (
                    "clf",
                    LogisticRegression(
                        C=self.config.C,
                        penalty=self.config.penalty,
                        solver=self.config.solver,
                        max_iter=self.config.max_iter,
                        class_weight=self.config.class_weight,
                        random_state=self.config.random_state,
                        verbose=self.config.verbose,
                        warm_start=self.config.warm_start,
                    ),
                ),
            ]
        )

    # Public API
    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionModel":
        if self.pipeline is None:
            self._build_pipeline()
        self.pipeline.fit(X, y)
        return self

    def fit_with_progress(self, X: np.ndarray, y: np.ndarray, step: int = 50) -> "LogisticRegressionModel":
        """训练模型并使用 tqdm 显示进度条（通过 warm_start 模拟进度）。"""
        if self.pipeline is None:
            self._build_pipeline()
        
        try:
            from tqdm import tqdm  # type: ignore
        except Exception:
            tqdm = None
        
        assert self.pipeline is not None
        clf: LogisticRegression = self.pipeline.named_steps["clf"]  # type: ignore
        
        # 使用 warm_start 显示进度条（虽然不完美，但至少能看到训练进度）
        if self.config.warm_start and self.config.solver in ["lbfgs", "newton-cg", "sag", "saga"]:
            total_iter = int(self.config.max_iter)
            clf.set_params(warm_start=True)
            iters = list(range(step, total_iter + 1, step))
            if iters[-1] != total_iter:
                iters.append(total_iter)
            iterable = tqdm(iters, desc="Training", unit="iter") if tqdm else iters
            for m in iterable:
                clf.set_params(max_iter=int(m))
                self.pipeline.fit(X, y)
        else:
            # 不支持 warm_start，直接训练
            if tqdm:
                with tqdm(total=1, desc="Training") as pbar:
                    self.pipeline.fit(X, y)
                    pbar.update(1)
            else:
                self.pipeline.fit(X, y)
        
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("Model pipeline is not built. Call fit or load first.")
        return self.pipeline.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("Model pipeline is not built. Call fit or load first.")
        return self.pipeline.predict_proba(X)

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        """Return common binary classification metrics."""
        y_pred = self.predict(X)
        metrics: Dict[str, float] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }
        # roc_auc requires probabilities and both classes present
        try:
            proba = self.predict_proba(X)[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y_true, proba))
        except Exception:
            metrics["roc_auc"] = float("nan")
        return metrics

    def get_params(self) -> Dict[str, Any]:
        return asdict(self.config)

    # Persistence
    def save(self, path: str) -> None:
        if self.pipeline is None:
            raise RuntimeError("Cannot save an uninitialized model.")
        payload = {
            "config": self.get_params(),
            "pipeline": self.pipeline,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str) -> "LogisticRegressionModel":
        payload = joblib.load(path)
        config = LogisticRegressionConfig(**payload["config"])
        model = cls(config=config)
        model.pipeline = payload["pipeline"]
        return model


def build_model_from_params(params: Optional[Dict[str, Any]] = None) -> LogisticRegressionModel:
    """Factory to build a LogisticRegressionModel from a dict of params."""
    params = params or {}
    config = LogisticRegressionConfig(**params)
    return LogisticRegressionModel(config=config)


