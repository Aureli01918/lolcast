"""End-to-end check on synthetic games with known team strengths.

Real data lives behind a network fetch, so this proves the machinery works
without one: teams are given hidden skill values, games are simulated from
them, and the model should recover most of that signal.

Run with: python tests/test_synthetic.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lolcast.model import calibration_table, score, walk_forward  # noqa: E402
from lolcast.pipeline import build_features, upcoming_row  # noqa: E402
from lolcast.ratings import series_win_probability  # noqa: E402

RATING_CFG = {
    "initial": 1500.0,
    "k": 24.0,
    "scale": 400.0,
    "split_regression": 0.25,
    "split_gap_days": 45,
    "side_advantage": "auto",
}
FEATURES = ["elo_diff", "is_blue_side", "form_diff_10", "rest_days_diff",
            "h2h_recent", "games_played_diff", "cross_region"]


def simulate(n_teams=24, n_games=9000, seed=7):
    rng = np.random.default_rng(seed)
    teams = [f"Team {i:02d}" for i in range(n_teams)]
    regions = {t: ["KR", "CN", "EU", "AM"][i % 4] for i, t in enumerate(teams)}
    skill = {t: rng.normal(0, 220) for t in teams}
    side_bonus = 55.0  # true blue-side edge, in Elo points

    start = pd.Timestamp("2021-01-01")
    rows = []
    for g in range(n_games):
        blue, red = rng.choice(teams, size=2, replace=False)
        # Skill drifts slowly so ratings have something to track.
        for t in (blue, red):
            skill[t] += rng.normal(0, 4)
        margin = skill[blue] + side_bonus - skill[red]
        p = 1 / (1 + 10 ** (-margin / 400))
        rows.append(
            {
                "date": start + pd.Timedelta(hours=6 * g),
                "league": regions[blue],
                "patch": "14.1",
                "blue": blue,
                "red": red,
                "blue_win": int(rng.random() < p),
                "blue_region": regions[blue],
                "red_region": regions[red],
                "best_of": 1,
                "true_prob": p,
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    games = simulate()
    print(f"Simulated {len(games):,} games over "
          f"{(games['date'].max() - games['date'].min()).days} days")

    X, y, index, ratings = build_features(games, FEATURES, RATING_CFG)
    assert X.shape == (len(games), len(FEATURES)), X.shape
    assert not np.isnan(X).any(), "features contain NaN"
    print(f"Feature matrix: {X.shape[0]:,} x {X.shape[1]}")
    print(f"Fitted blue-side advantage: {ratings.side_advantage:+.1f} Elo "
          f"(true value 55.0)")

    backtest_cfg = {"test_start": "2023-01-01", "refit_every_days": 30,
                    "min_train_games": 1500}
    model_cfg = {"type": "logistic", "calibration": "isotonic", "l2": 1.0}

    result, detail = walk_forward(X, y, index, backtest_cfg, model_cfg)
    elo_only = score(detail["actual"].to_numpy(), detail["elo_prob"].to_numpy())

    # The oracle: what a model that knew every hidden skill value would score.
    truth = games.loc[detail.index, "true_prob"].to_numpy()
    oracle = score(detail["actual"].to_numpy(), truth)

    print(f"\nElo only  : {elo_only}")
    print(f"Full model: {result}")
    print(f"Oracle     : {oracle}   <- ceiling given the noise")

    print("\nCalibration")
    print(calibration_table(detail["actual"].to_numpy(),
                            detail["prediction"].to_numpy()).to_string(index=False))

    # A forecast on a game that has not happened yet.
    row = upcoming_row(ratings, FEATURES, "Team 00", "Team 01",
                       pd.Timestamp("2026-01-01"))
    print(f"\nUpcoming-row features: {[round(v, 3) for v in row]}")

    p = 0.62
    print(f"Series conversion at p={p}: "
          f"Bo1 {series_win_probability(p, 1):.3f}  "
          f"Bo3 {series_win_probability(p, 3):.3f}  "
          f"Bo5 {series_win_probability(p, 5):.3f}")

    assert result.log_loss < elo_only.log_loss + 0.01, "model is worse than raw Elo"
    assert result.skill > 0.05, "model has almost no skill over a coin flip"
    assert oracle.log_loss < result.log_loss, "model beat the oracle -- leakage"
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
