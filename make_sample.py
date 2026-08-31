"""Write docs/data.json from simulated games.

Lets you open the dashboard and check the layout before downloading any
real data. `python -m lolcast predict` overwrites it with the real thing.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lolcast import ledger, selfcal  # noqa: E402
from lolcast.cli import _confidence  # noqa: E402
from lolcast.model import calibration_table, make_estimator, walk_forward  # noqa: E402
from lolcast.pipeline import build_features, upcoming_row  # noqa: E402
from lolcast import series  # noqa: E402
from lolcast.ratings import series_win_probability  # noqa: E402
from test_synthetic import FEATURES, RATING_CFG, simulate  # noqa: E402

NAMES = ["Gen.G", "T1", "Hanwha Life", "Dplus KIA", "Bilibili Gaming",
         "Top Esports", "JD Gaming", "Weibo Gaming", "G2 Esports",
         "Karmine Corp", "Fnatic", "Movistar KOI", "FlyQuest", "Cloud9",
         "100 Thieves", "paiN Gaming", "CTBC Flying Oyster", "PSG Talon",
         "Team Vitality", "Anyone's Legend", "Nongshim RedForce", "BNK FearX",
         "Team Liquid", "Shopify Rebellion"]
EVENTS = ["LCK 2026 Summer", "LPL 2026 Summer", "LEC 2026 Summer",
          "LTA North 2026 Split 3", "LCP 2026 Summer"]
ROUNDS = ["Week 4", "Week 5", "Quarterfinals", "Group Stage"]


def main() -> int:
    games = simulate(n_teams=len(NAMES), n_games=9000)
    rename = {f"Team {i:02d}": NAMES[i] for i in range(len(NAMES))}
    games["blue"] = games["blue"].map(rename)
    games["red"] = games["red"].map(rename)

    X, y, index, ratings = build_features(games, FEATURES, RATING_CFG)
    estimator = make_estimator({"type": "logistic", "calibration": "isotonic"})
    estimator.fit(X, y)

    result, detail = walk_forward(
        X, y, index,
        {"test_start": "2023-01-01", "refit_every_days": 30, "min_train_games": 1500},
        {"type": "logistic", "calibration": "isotonic"},
    )

    rng = np.random.default_rng(3)
    now = datetime.now(timezone.utc)

    # A few weeks of graded live forecasts, so the scoreboard and the
    # self-calibration status both have something real to show.
    book = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "sample_ledger.csv")
    if os.path.exists(book):
        os.remove(book)
    for i in range(420):
        kickoff = now - timedelta(days=90) + timedelta(hours=5 * i)
        true_p = float(rng.uniform(0.25, 0.85))
        mine = float(np.clip(true_p + rng.normal(0, 0.07), 0.02, 0.98))
        market = float(np.clip(true_p + rng.normal(0, 0.04), 0.02, 0.98))
        quotes = {"lolcast": mine, "elo": float(np.clip(true_p + rng.normal(0, 0.10), .02, .98))}
        if rng.random() < 0.7:                       # market skips some games
            quotes["polymarket"] = market
        ledger.append(book, ledger.snapshot_rows(
            {"key": f"H{i}", "kickoff": kickoff, "team1": "A", "team2": "B",
             "event": "Sample", "best_of": 3},
            quotes, kickoff - timedelta(hours=6)))
        ledger.apply_results(book, {f"H{i}": int(rng.random() < true_p)})

    history = ledger.load(book)
    live_scores, _ = ledger.scoreboard(history, 1.0, None)
    cal = selfcal.fit_from_ledger(history, "lolcast",
                                  {"enabled": True, "min_matches": 300})
    matches = []
    for i in range(11):
        blue, red = rng.choice(NAMES, size=2, replace=False)
        when = now + timedelta(hours=float(rng.uniform(2, 5 * 24)))
        best_of = int(rng.choice([1, 3, 3, 5]))
        row = upcoming_row(ratings, FEATURES, blue, red, pd.Timestamp(when), best_of=best_of)
        p_game = float(estimator.predict_proba(np.array([row]))[0, 1])
        p_elo_game = ratings.expected(blue, red)
        p_blue = float(np.clip(p_game + 0.05, 0.02, 0.98))
        p_red = float(np.clip(p_game - 0.05, 0.02, 0.98))
        lines = series.scoreline_distribution(p_blue, p_red, best_of,
                                              side_choice_rate=0.93)
        p_series = lines.series_win()
        b, r = ratings.get(blue), ratings.get(red)
        has_market = rng.random() < 0.75
        market = (float(np.clip(p_series + rng.normal(0, 0.05), 0.03, 0.97))
                  if has_market else None)
        matches.append({
            "id": f"sample-{i}",
            "date": when.isoformat(),
            "event": str(rng.choice(EVENTS)),
            "round": str(rng.choice(ROUNDS)),
            "bestOf": best_of,
            "blue": {"name": blue, "elo": round(b.elo), "form": round(b.form(10), 2),
                     "games": b.games},
            "red": {"name": red, "elo": round(r.elo), "form": round(r.form(10), 2),
                    "games": r.games},
            "gameProb": round(p_game, 4),
            "seriesProb": round(cal.apply(p_series), 4),
            "scorelines": {k: round(v, 4) for k, v in lines.as_dict().items()},
            "sweep": {"team1": round(lines.sweep("a"), 4),
                      "team2": round(lines.sweep("b"), 4)},
            "seriesProbUncalibrated": round(p_series, 4),
            "sources": {"polymarket": round(market, 4) if market else None},
            "eloProb": round(series_win_probability(p_elo_game, best_of), 4),
            "eloGameProb": round(p_elo_game, 4),
            "confidence": _confidence(b.games, r.games),
        })
    matches.sort(key=lambda m: m["date"])

    payload = {
        "generated": now.isoformat(),
        "sample": True,
        "matches": matches,
        "unmatchedTeams": [],
        "sources": [{"name": "polymarket", "label": "Polymarket",
                     "colour": "#5FD3A6", "error": None}],
        "selfLabels": {"model": "lolcast", "baseline": "elo"},
        "calibration": {"active": cal.active, "temperature": cal.temperature,
                        "matches": cal.matches, "reason": cal.reason,
                        "text": str(cal)},
        "sweepLive": [{"source": "lolcast", "logLoss": 0.412, "brier": 0.128,
                       "matches": 180},
                      {"source": "elo", "logLoss": 0.437, "brier": 0.139,
                       "matches": 180}],
        "series": {"sideChoiceRate": 0.94, "sidePairs": 4120,
                   "repeatRate": 0.516, "repeatPairs": 4120},
        "recent": [
            {"id": f"past-{i}", "date": (now - timedelta(days=i + 1)).isoformat(),
             "event": ev, "bestOf": bo, "team1": a, "team2": b,
             "team1Score": s1, "team2Score": s2, "team1Won": int(s1 > s2),
             "forecast": f}
            for i, (a, b, s1, s2, f, bo, ev) in enumerate([
                ("Gen.G", "T1", 2, 1, 0.58, 3, "LCK 2026 Summer"),
                ("G2 Esports", "Fnatic", 0, 2, 0.61, 3, "LEC 2026 Summer"),
                ("JD Gaming", "Weibo Gaming", 3, 0, 0.72, 5, "LPL 2026 Summer"),
            ])],
        "liveCalibration": ledger.live_calibration(history, "lolcast", bins=5),
        "divergence": ledger.divergence(history, "lolcast", 0.5540),
        "live": [{"source": s.source, "logLoss": s.log_loss, "brier": s.brier,
                  "accuracy": s.accuracy, "matches": s.common,
                  "coverage": round(s.coverage, 3)} for s in live_scores],
        "refresh": {"proxyUrl": "",
                    "actionsUrl": "https://github.com/you/lolcast/actions"},
        "model": {
            "features": FEATURES,
            "type": "logistic",
            "trainingGames": int(len(y)),
            "logLoss": round(result.log_loss, 4),
            "brier": round(result.brier, 4),
            "accuracy": round(result.accuracy, 4),
            "skill": round(result.skill, 4),
            "backtestGames": result.n,
            "calibration": calibration_table(
                detail["actual"].to_numpy(), detail["prediction"].to_numpy(), bins=5
            ).to_dict("records"),
        },
    }

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "docs", "data.json")
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Wrote sample forecast for {len(matches)} matches to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
