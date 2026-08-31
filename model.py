"""Forecast model and walk-forward evaluation.

Accuracy is the wrong headline metric for a probability forecast -- a model
that says 51% for every game the favourite wins scores the same accuracy as
one that says 90% and is right. So the backtest reports log loss and Brier
score first, and shows a calibration table so you can see whether "70%"
actually means 70%.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_estimator(cfg: dict):
    kind = cfg.get("type", "logistic")
    if kind == "logistic":
        base = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0 / max(cfg.get("l2", 1.0), 1e-9), max_iter=1000),
        )
    elif kind == "gradient_boosting":
        base = HistGradientBoostingClassifier(
            max_depth=3,
            learning_rate=0.05,
            max_iter=300,
            l2_regularization=cfg.get("l2", 1.0),
            early_stopping=True,
        )
    else:
        raise ValueError(f"unknown model type: {kind!r}")

    method = cfg.get("calibration", "isotonic")
    if method in ("isotonic", "sigmoid"):
        return CalibratedClassifierCV(base, method=method, cv=3)
    return base


@dataclass
class Scores:
    n: int
    log_loss: float
    brier: float
    accuracy: float
    baseline_log_loss: float

    @property
    def skill(self) -> float:
        """Fraction of log loss removed versus always predicting 50/50."""
        return 1.0 - self.log_loss / self.baseline_log_loss

    def __str__(self) -> str:
        return (
            f"n={self.n}  log_loss={self.log_loss:.4f}  brier={self.brier:.4f}  "
            f"acc={self.accuracy:.3f}  skill={self.skill:+.1%}"
        )


def score(y: np.ndarray, p: np.ndarray) -> Scores:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    ll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return Scores(
        n=len(y),
        log_loss=ll,
        brier=float(np.mean((p - y) ** 2)),
        accuracy=float(np.mean((p > 0.5) == (y == 1))),
        baseline_log_loss=float(np.log(2)),
    )


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = idx == b
        if not mask.any():
            continue
        rows.append(
            {
                "bucket": f"{edges[b]:.0%}-{edges[b + 1]:.0%}",
                "games": int(mask.sum()),
                "predicted": float(p[mask].mean()),
                "actual": float(y[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


def walk_forward(
    X: np.ndarray,
    y: np.ndarray,
    index: pd.DataFrame,
    cfg: dict,
    model_cfg: dict,
) -> tuple[Scores, pd.DataFrame]:
    """Refit periodically on everything before each window. No peeking."""
    dates = pd.to_datetime(index["date"])
    test_start = pd.Timestamp(cfg.get("test_start", "2024-01-01"))
    step = pd.Timedelta(days=int(cfg.get("refit_every_days", 30)))
    min_train = int(cfg.get("min_train_games", 2000))

    preds = np.full(len(y), np.nan)
    cursor = test_start
    end = dates.max()

    while cursor <= end:
        train_mask = (dates < cursor).to_numpy()
        test_mask = ((dates >= cursor) & (dates < cursor + step)).to_numpy()
        cursor += step
        if test_mask.sum() == 0 or train_mask.sum() < min_train:
            continue
        if len(np.unique(y[train_mask])) < 2:
            continue
        est = make_estimator(model_cfg)
        est.fit(X[train_mask], y[train_mask])
        preds[test_mask] = est.predict_proba(X[test_mask])[:, 1]

    evaluated = ~np.isnan(preds)
    if not evaluated.any():
        raise RuntimeError(
            "backtest produced no predictions -- check test_start and min_train_games"
        )

    out = index.loc[evaluated].copy()
    out["prediction"] = preds[evaluated]
    out["actual"] = y[evaluated]
    return score(y[evaluated], preds[evaluated]), out


def ablation(
    games: pd.DataFrame,
    enabled: list[str],
    rating_cfg: dict,
    backtest_cfg: dict,
    model_cfg: dict,
    stat_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Drop each feature in turn and re-run. Shows what each one is worth.

    Run this after adding a variable. A feature that does not move log loss
    is costing you complexity for nothing.
    """
    from .pipeline import build_features

    rows = []
    full_X, y, index, _ = build_features(games, enabled, rating_cfg, stat_columns)
    full, _ = walk_forward(full_X, y, index, backtest_cfg, model_cfg)
    rows.append({"removed": "(nothing)", "log_loss": full.log_loss, "delta": 0.0})

    for name in enabled:
        subset = [f for f in enabled if f != name]
        if not subset:
            continue
        X, y2, idx2, _ = build_features(games, subset, rating_cfg, stat_columns)
        s, _ = walk_forward(X, y2, idx2, backtest_cfg, model_cfg)
        rows.append(
            {
                "removed": name,
                "log_loss": s.log_loss,
                "delta": s.log_loss - full.log_loss,
            }
        )

    return pd.DataFrame(rows).sort_values("delta", ascending=False)
