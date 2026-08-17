"""
Trade-quality classifier.

Given the feature snapshot at a crossover, predict P(this round trip closes
profitable net of costs), and turn that into ACCEPT / AVOID plus a readable
explanation.

Two design decisions worth defending:

1. TIME-ORDERED SPLIT, NOT RANDOM. Trades are a time series. A random split
   lets the model learn from trades that happen after the ones it is evaluated
   on, which leaks future information and produces a validation score that will
   not survive contact with live data.

2. EXPLANATIONS ARE PER-INSTANCE, NOT JUST GLOBAL IMPORTANCE. Global importance
   tells you which features the model relies on overall; it says nothing about
   why THIS trade scored 0.71. We rank features by
   `importance x |z-score vs the training median|`, i.e. "features the model
   cares about, on which this trade is unusual", and use SHAP when installed.
   The heuristic is a documented approximation, not a causal attribution.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import ModelConfig, settings
from ..features import FEATURE_COLUMNS, FEATURE_LABELS, features_to_frame
from ..models import Decision, Prediction, Side
from ..utils import get_logger

log = get_logger(__name__)

MODEL_FORMAT_VERSION = 2


@dataclass
class TrainingReport:
    algorithm: str
    n_samples: int
    n_train: int
    n_test: int
    positive_rate: float
    train_accuracy: float
    test_accuracy: float
    precision: float
    recall: float
    roc_auc: float
    importances: list[tuple[str, float]] = field(default_factory=list)
    trained_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"algorithm      : {self.algorithm}",
            f"samples        : {self.n_samples} ({self.n_train} train / {self.n_test} test)",
            f"positive rate  : {self.positive_rate:.1%}",
            f"train accuracy : {self.train_accuracy:.3f}",
            f"test accuracy  : {self.test_accuracy:.3f}",
            f"precision      : {self.precision:.3f}",
            f"recall         : {self.recall:.3f}",
            f"roc auc        : {self.roc_auc:.3f}",
            "",
            "top features:",
        ]
        lines += [
            f"  {i:>2}. {FEATURE_LABELS.get(name, name):<34} {score:.4f}"
            for i, (name, score) in enumerate(self.importances[:10], 1)
        ]
        if self.warnings:
            lines += ["", "warnings:"] + [f"  ! {w}" for w in self.warnings]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "n_samples": self.n_samples,
            "positive_rate": self.positive_rate,
            "test_accuracy": self.test_accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "roc_auc": self.roc_auc,
            "trained_at": self.trained_at,
            "importances": self.importances,
            "warnings": self.warnings,
        }


class TradeClassifier:
    """Wraps the estimator plus everything needed to explain its output."""

    def __init__(self, cfg: ModelConfig | None = None) -> None:
        self.cfg = cfg or settings.model
        self.model = None
        self.feature_names: list[str] = list(FEATURE_COLUMNS)
        self.importances: dict[str, float] = {}
        self.medians: dict[str, float] = {}
        self.stds: dict[str, float] = {}
        self.report: TrainingReport | None = None
        self.version: str = "untrained"
        self._explainer = None

    # ------------------------------------------------------------------
    @property
    def is_trained(self) -> bool:
        return self.model is not None

    def _build_estimator(self):
        algo = self.cfg.algorithm.lower()

        if algo in {"xgboost", "xgb"}:
            try:
                from xgboost import XGBClassifier

                return XGBClassifier(
                    n_estimators=self.cfg.n_estimators,
                    max_depth=self.cfg.max_depth,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_lambda=1.0,
                    random_state=self.cfg.random_state,
                    eval_metric="logloss",
                    n_jobs=-1,
                )
            except ImportError:
                log.warning("xgboost not installed - falling back to RandomForest")

        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=self.cfg.n_estimators,
            max_depth=self.cfg.max_depth,
            min_samples_leaf=3,
            # Trade datasets are usually imbalanced; without this the model can
            # score well by predicting the majority class and never accepting.
            class_weight="balanced_subsample",
            random_state=self.cfg.random_state,
            n_jobs=-1,
        )

    # ------------------------------------------------------------------
    def train(self, dataset: pd.DataFrame, label_col: str = "label") -> TrainingReport:
        """
        Fit on a trade table produced by the simulator.

        `dataset` needs the label column plus the f_-prefixed feature columns
        (as emitted by Trade.to_dict) or the bare feature names.
        """
        from sklearn.metrics import (accuracy_score, precision_score,
                                     recall_score, roc_auc_score)

        if dataset is None or dataset.empty:
            raise ValueError("training dataset is empty")
        if label_col not in dataset.columns:
            raise ValueError(f"missing label column {label_col!r}")

        X = self._extract_features(dataset)
        y = dataset[label_col].astype(int).to_numpy()

        warnings: list[str] = []
        if len(X) < self.cfg.min_training_rows:
            warnings.append(
                f"only {len(X)} samples (recommended >= {self.cfg.min_training_rows}); "
                "scores below are optimistic and the model should be treated as a placeholder"
            )
        if len(np.unique(y)) < 2:
            raise ValueError(
                f"labels are all {int(y[0])} - a classifier needs both winners and losers. "
                "Widen the date range or lower costs."
            )

        positive_rate = float(y.mean())
        if positive_rate < 0.1 or positive_rate > 0.9:
            warnings.append(f"severe class imbalance (positive rate {positive_rate:.1%})")

        # Time-ordered split. See the module docstring for why this is not random.
        if "entry_time" in dataset.columns:
            order = pd.to_datetime(dataset["entry_time"], errors="coerce").argsort()
            X, y = X.iloc[order].reset_index(drop=True), y[order]
        split = max(1, int(len(X) * (1.0 - self.cfg.test_size)))
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y[:split], y[split:]

        if len(X_test) == 0 or len(np.unique(y_train)) < 2:
            X_train, y_train, X_test, y_test = X, y, X, y
            warnings.append("dataset too small to hold out a test set - scores are in-sample")

        self.model = self._build_estimator()
        self.model.fit(X_train, y_train)

        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        try:
            test_proba = self.model.predict_proba(X_test)[:, 1]
            auc = float(roc_auc_score(y_test, test_proba)) if len(np.unique(y_test)) > 1 else 0.5
        except (ValueError, AttributeError):
            auc = 0.5

        raw = getattr(self.model, "feature_importances_", np.zeros(len(self.feature_names)))
        self.importances = {n: float(v) for n, v in zip(self.feature_names, raw)}
        # Reference distribution for per-instance explanations.
        self.medians = X_train.median().to_dict()
        self.stds = X_train.std(ddof=0).replace(0.0, 1.0).to_dict()

        self.report = TrainingReport(
            algorithm=type(self.model).__name__,
            n_samples=len(X),
            n_train=len(X_train),
            n_test=len(X_test),
            positive_rate=positive_rate,
            train_accuracy=float(accuracy_score(y_train, train_pred)),
            test_accuracy=float(accuracy_score(y_test, test_pred)),
            precision=float(precision_score(y_test, test_pred, zero_division=0)),
            recall=float(recall_score(y_test, test_pred, zero_division=0)),
            roc_auc=auc,
            importances=sorted(self.importances.items(), key=lambda kv: kv[1], reverse=True),
            warnings=warnings,
        )
        self.version = f"{type(self.model).__name__}-{len(X)}-{self.report.trained_at}"
        self._explainer = None

        log.info("model trained: %s", self.report.summary().replace("\n", " | "))
        for w in warnings:
            log.warning("training warning: %s", w)
        return self.report

    # ------------------------------------------------------------------
    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Accept either f_-prefixed or bare feature columns."""
        prefixed = {f"f_{c}": c for c in FEATURE_COLUMNS if f"f_{c}" in df.columns}
        if prefixed:
            out = df[list(prefixed)].rename(columns=prefixed)
        else:
            present = [c for c in FEATURE_COLUMNS if c in df.columns]
            if not present:
                raise ValueError(
                    "no recognised feature columns; expected f_-prefixed names "
                    f"such as f_{FEATURE_COLUMNS[0]}"
                )
            out = df[present].copy()

        for col in FEATURE_COLUMNS:
            if col not in out.columns:
                out[col] = 0.0
        return (
            out[FEATURE_COLUMNS]
            .astype(float)
            .replace([np.inf, -np.inf], 0.0)
            .fillna(0.0)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    def predict(self, features: dict[str, float], symbol: str = "", side: Side = Side.BUY) -> Prediction:
        """Score one signal. Never raises - an unusable model yields UNKNOWN."""
        if not self.is_trained:
            return Prediction(
                symbol=symbol, side=side, probability=0.5, decision=Decision.UNKNOWN,
                explanation="no trained model available - run scripts/train_model.py",
                model_version=self.version,
            )

        X = features_to_frame([features])
        try:
            proba = float(self.model.predict_proba(X)[0, 1])
        except Exception as exc:  # noqa: BLE001
            log.error("prediction failed for %s: %s", symbol, exc)
            return Prediction(
                symbol=symbol, side=side, probability=0.5, decision=Decision.UNKNOWN,
                explanation=f"scoring error: {exc}", model_version=self.version,
            )

        decision = Decision.ACCEPT if proba >= self.cfg.accept_threshold else Decision.AVOID
        contributions = self._contributions(X, features)
        return Prediction(
            symbol=symbol, side=side, probability=proba, decision=decision,
            explanation=self._render_explanation(proba, decision, contributions),
            contributions=contributions, model_version=self.version,
        )

    def predict_batch(self, rows: list[tuple[str, Side, dict[str, float]]]) -> list[Prediction]:
        return [self.predict(f, symbol=s, side=sd) for s, sd, f in rows]

    # ------------------------------------------------------------------
    def _contributions(self, X: pd.DataFrame, features: dict[str, float]) -> list[tuple[str, float]]:
        """
        Per-instance feature attribution.

        SHAP if available (a principled additive attribution); otherwise the
        importance x deviation heuristic described in the module docstring.
        """
        try:
            import shap  # type: ignore

            if self._explainer is None:
                self._explainer = shap.TreeExplainer(self.model)
            values = self._explainer.shap_values(X)
            if isinstance(values, list):        # older API: one array per class
                values = values[1]
            arr = np.asarray(values).reshape(-1)[: len(self.feature_names)]
            pairs = list(zip(self.feature_names, (float(v) for v in arr)))
            return sorted(pairs, key=lambda kv: abs(kv[1]), reverse=True)[: self.cfg.top_explanations]
        except Exception:  # noqa: BLE001 - shap absent or unhappy with the model
            pass

        scored: list[tuple[str, float]] = []
        for name in self.feature_names:
            value = float(features.get(name, 0.0))
            median = float(self.medians.get(name, 0.0))
            std = float(self.stds.get(name, 1.0)) or 1.0
            z = (value - median) / std
            scored.append((name, float(self.importances.get(name, 0.0) * z)))
        return sorted(scored, key=lambda kv: abs(kv[1]), reverse=True)[: self.cfg.top_explanations]

    def _render_explanation(
        self, proba: float, decision: Decision, contributions: list[tuple[str, float]]
    ) -> str:
        if not contributions:
            return f"{decision.value} at {proba:.0%} confidence."
        parts = [
            f"{FEATURE_LABELS.get(n, n)} {'supports' if v > 0 else 'weighs against'}"
            for n, v in contributions
        ]
        return (
            f"{decision.value} ({proba:.0%} win probability). "
            f"Main drivers: {'; '.join(parts)}."
        )

    # ------------------------------------------------------------------
    def save(self, path: Path | None = None) -> Path:
        import joblib

        path = Path(path or settings.paths.model)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "format_version": MODEL_FORMAT_VERSION,
                "model": self.model,
                "feature_names": self.feature_names,
                "importances": self.importances,
                "medians": self.medians,
                "stds": self.stds,
                "version": self.version,
                "report": self.report.to_dict() if self.report else None,
                "accept_threshold": self.cfg.accept_threshold,
            },
            path,
        )
        if self.report:
            path.with_suffix(".report.json").write_text(
                json.dumps(self.report.to_dict(), indent=2), encoding="utf-8"
            )
        log.info("model saved to %s", path)
        return path

    @classmethod
    def load(cls, path: Path | None = None, cfg: ModelConfig | None = None) -> "TradeClassifier":
        import joblib

        path = Path(path or settings.paths.model)
        clf = cls(cfg)
        if not path.exists():
            log.warning("no model at %s - predictions will be UNKNOWN", path)
            return clf

        try:
            blob = joblib.load(path)
        except Exception as exc:  # noqa: BLE001
            log.error("could not load model from %s: %s", path, exc)
            return clf

        fmt = blob.get("format_version", 1)
        if fmt != MODEL_FORMAT_VERSION:
            log.warning(
                "model file format v%s != expected v%s - retrain to avoid surprises",
                fmt, MODEL_FORMAT_VERSION,
            )

        clf.model = blob.get("model")
        clf.feature_names = blob.get("feature_names", list(FEATURE_COLUMNS))
        clf.importances = blob.get("importances", {})
        clf.medians = blob.get("medians", {})
        clf.stds = blob.get("stds", {})
        clf.version = blob.get("version", "unknown")
        log.info("model loaded: %s", clf.version)
        return clf
