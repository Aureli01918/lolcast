"""Self-calibration from live results.

The model already relearns ratings from new games every run. This is the
other half: checking whether its stated confidence matches reality, and
correcting it if not.

If matches you called at 70% only win 58% of the time, the fix is not to
change the ratings — it is to squash the whole probability scale toward
the middle. That is what this does, fitted on the ledger's graded live
forecasts rather than on the backtest.

Three guardrails, because a correction fitted on too little data is worse
than no correction at all:

1. **Minimum sample.** Below `min_matches`, returns the identity function
   and says so. Twenty matches of bad luck should not reshape the model.
2. **Shrinkage.** The correction is blended with "change nothing",
   weighted by how much data supports it. A hundred matches moves it part
   of the way; a thousand moves it most of the way.
3. **Monotone by construction.** Fitting a single scalar on the log-odds
   scale cannot reorder matches — a team the model preferred is still
   preferred afterwards. Only the confidence changes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class Calibration:
    """A learned confidence correction."""

    temperature: float      # >1 = model was overconfident, pull toward 50%
    matches: int
    active: bool
    reason: str

    def apply(self, p):
        scalar = np.isscalar(p)
        arr = np.atleast_1d(np.asarray(p, dtype=float))
        if not self.active:
            out = arr
        else:
            out = _sigmoid(_logit(arr) / self.temperature)
        return float(out[0]) if scalar else out

    def __str__(self) -> str:
        if not self.active:
            return f"calibration off ({self.reason}, {self.matches} matches)"
        direction = "overconfident" if self.temperature > 1 else "underconfident"
        return (f"calibration on: temperature {self.temperature:.3f} "
                f"({direction}) from {self.matches} live matches")


IDENTITY = Calibration(1.0, 0, False, "no data")


def fit(
    predictions: np.ndarray,
    outcomes: np.ndarray,
    min_matches: int = 300,
    shrink_at: int = 1000,
    max_temperature: float = 3.0,
) -> Calibration:
    """Fit a temperature on live graded forecasts.

    `shrink_at` is the sample size at which the correction is trusted
    almost fully; below it, the fit is blended toward doing nothing.
    """
    predictions = np.asarray(predictions, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    n = len(predictions)

    if n < min_matches:
        return Calibration(1.0, n, False,
                           f"needs {min_matches} graded matches")
    if len(np.unique(outcomes)) < 2:
        return Calibration(1.0, n, False, "all outcomes identical")

    z = _logit(predictions)

    def loss(temp: float) -> float:
        p = np.clip(_sigmoid(z / temp), 1e-9, 1 - 1e-9)
        return float(-np.mean(outcomes * np.log(p) + (1 - outcomes) * np.log(1 - p)))

    # Golden-section search over a bounded range. One parameter, smooth
    # objective -- no need for anything heavier.
    lo, hi = 1.0 / max_temperature, max_temperature
    phi = (np.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - phi * (b - a), a + phi * (b - a)
    for _ in range(60):
        if loss(c) < loss(d):
            b, d = d, c
            c = b - phi * (b - a)
        else:
            a, c = c, d
            d = a + phi * (b - a)
    raw = (a + b) / 2

    # Only accept a correction that actually improves the live score.
    if loss(raw) >= loss(1.0) - 1e-6:
        return Calibration(1.0, n, False, "no improvement over uncorrected")

    weight = min(1.0, n / float(shrink_at))
    shrunk = 1.0 + (raw - 1.0) * weight
    return Calibration(round(shrunk, 4), n, True, "fitted on live results")


def fit_from_ledger(ledger_df, source: str, cfg: dict) -> Calibration:
    from .ledger import graded_pairs

    if not cfg.get("enabled", False):
        return Calibration(1.0, 0, False, "disabled in config")
    p, y = graded_pairs(ledger_df, source, cfg.get("min_lead_hours", 1.0))
    if len(p) == 0:
        return IDENTITY
    return fit(
        p, y,
        min_matches=int(cfg.get("min_matches", 300)),
        shrink_at=int(cfg.get("shrink_at", 1000)),
        max_temperature=float(cfg.get("max_temperature", 3.0)),
    )
