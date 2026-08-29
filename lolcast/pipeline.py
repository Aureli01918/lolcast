"""Turns a chronological list of games into a model-ready feature table.

The whole pipeline is one forward pass over history:

    for each game, oldest first:
        1. compute features from the state so far   <- prediction time
        2. record them
        3. feed the result into the state           <- learning

Because step 3 always follows step 1, a game can never inform its own
features. Adding a variable means adding a function in features.py; the
loop below does not change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import FeatureContext, build_row
from .ratings import RatingSystem, fit_side_advantage

# Columns every loader must produce, one row per *game* (not per series).
REQUIRED = ["date", "league", "blue", "red", "blue_win"]
OPTIONAL = ["patch", "best_of", "blue_region", "red_region", "gameid"]


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"games frame is missing required columns: {missing}")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    for col in OPTIONAL:
        if col not in df.columns:
            df[col] = None
    df["best_of"] = pd.to_numeric(df["best_of"], errors="coerce").fillna(1).astype(int)
    df["blue_win"] = df["blue_win"].astype(int)
    return df.sort_values("date").reset_index(drop=True)


def build_features(
    games: pd.DataFrame,
    enabled: list[str],
    rating_cfg: dict,
    stat_columns: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, RatingSystem]:
    """Return (X, y, index_frame, fitted_rating_system).

    `index_frame` carries date/league/team names so the backtest can split
    by time and the dashboard can label rows.
    """
    games = normalise(games)
    stat_columns = stat_columns or []

    base_kwargs = {
        "initial": rating_cfg.get("initial", 1500.0),
        "k": rating_cfg.get("k", 24.0),
        "scale": rating_cfg.get("scale", 400.0),
        "split_regression": rating_cfg.get("split_regression", 0.25),
        "split_gap_days": rating_cfg.get("split_gap_days", 45.0),
    }

    side_cfg = rating_cfg.get("side_advantage", 0.0)
    if side_cfg == "auto":
        side_adv = fit_side_advantage(games, base_kwargs)
    else:
        side_adv = float(side_cfg)

    ratings = RatingSystem(side_advantage=side_adv, **base_kwargs)

    rows, labels, index = [], [], []

    for game in games.itertuples(index=False):
        blue_state = ratings.get(game.blue, game.date)
        red_state = ratings.get(game.red, game.date)

        ctx = FeatureContext(
            blue=game.blue,
            red=game.red,
            date=game.date,
            league=game.league,
            patch=game.patch,
            best_of=game.best_of,
            blue_state=blue_state,
            red_state=red_state,
            ratings=ratings,
            meta={"blue_region": game.blue_region, "red_region": game.red_region},
        )

        rows.append(build_row(ctx, enabled))
        labels.append(game.blue_win)
        index.append(
            {
                "date": game.date,
                "league": game.league,
                "blue": game.blue,
                "red": game.red,
                "blue_elo": blue_state.elo,
                "red_elo": red_state.elo,
                "elo_prob": ratings.expected(game.blue, game.red),
            }
        )

        stats = None
        if stat_columns:
            stats = {
                "blue": {c: getattr(game, c, None) for c in stat_columns},
                "red": {c: _flip(c, getattr(game, c, None)) for c in stat_columns},
            }
            stats["blue"]["patch"] = game.patch
            stats["red"]["patch"] = game.patch

        ratings.update(game.blue, game.red, int(game.blue_win), game.date, stats)

    X = np.asarray(rows, dtype=float)
    y = np.asarray(labels, dtype=int)
    return X, y, pd.DataFrame(index), ratings


def upcoming_row(
    ratings: RatingSystem,
    enabled: list[str],
    blue: str,
    red: str,
    date,
    league: str = "",
    patch: str | None = None,
    best_of: int = 1,
    meta: dict | None = None,
) -> list[float]:
    """Feature row for a game that has not been played.

    Uses the same code path as training, so a feature can never behave
    differently at prediction time than it did during the backtest.
    """
    ctx = FeatureContext(
        blue=blue,
        red=red,
        date=pd.Timestamp(date).tz_localize(None) if pd.Timestamp(date).tzinfo else pd.Timestamp(date),
        league=league,
        patch=patch,
        best_of=best_of,
        blue_state=ratings.get(blue),
        red_state=ratings.get(red),
        ratings=ratings,
        meta=meta or {},
    )
    return build_row(ctx, enabled)


def _flip(column: str, value):
    """Differential stats are stored from blue's view; negate for red."""
    if value is None:
        return None
    return -value if "diff" in column else value
