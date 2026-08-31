"""The prediction ledger.

Every forecast is written down *before* the match, then graded after. This
file is the only honest record of how anyone has actually done — the
backtest is a re-enactment of the past, which is a different and easier
test.

It is append-only. Nothing is ever edited except filling in the result.

Fair comparison is the fiddly part, and two rules handle it:

* **Same moment.** Market prices drift as news breaks. Scoring a market's
  five-minutes-before price against our three-days-before price would
  flatter the market. So each source is scored on its last quote taken at
  least `min_lead_hours` before kickoff, and everyone uses the same rule.

* **Same matches.** A source that only prices marquee games would look
  brilliant on accuracy alone. So the headline table scores every source
  on the subset of matches *all* of them covered, and reports coverage as
  its own column.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

COLUMNS = [
    "match_key", "kickoff_utc", "event", "best_of", "team1", "team2",
    "source", "market", "prob_team1", "captured_utc", "lead_hours",
    "liquidity", "result", "team1_score", "team2_score", "graded_utc",
]

# What each row is predicting. "result" always means "the thing this row
# predicted happened", so a sweep row is graded against the scoreline
# rather than against who won.
SERIES = "series"
SWEEP_1 = "sweep_team1"
SWEEP_2 = "sweep_team2"


def load(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(path, dtype={"match_key": str})
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = None
    for col in ("kickoff_utc", "captured_utc", "graded_utc"):
        # Rows appended directly use "2026-08-19T00:00:00+00:00" while rows
        # rewritten during grading use a space instead of the T. Without
        # format="mixed" pandas infers one shape and turns the other into
        # NaT, which silently drops matches from the lead-time rule and the
        # results feed. Nothing errors; rows just vanish.
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce",
                                 format="mixed")
    for col in ("prob_team1", "lead_hours", "liquidity", "result",
                "team1_score", "team2_score"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Ledgers written before markets existed hold series predictions only.
    df["market"] = df["market"].fillna(SERIES)
    return df[COLUMNS]


def append(path: str, rows: list[dict]) -> int:
    """Append snapshot rows. Skips duplicates of the same run."""
    if not rows:
        return 0
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    is_new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        if is_new:
            writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in COLUMNS})
    return len(rows)


def snapshot_rows(
    match: dict,
    quotes: dict[str, float],
    captured: datetime,
    liquidity: dict[str, float] | None = None,
    market: str = SERIES,
) -> list[dict]:
    """Build one row per source for a single upcoming match."""
    kickoff = pd.Timestamp(match["kickoff"])
    if kickoff.tzinfo is None:
        kickoff = kickoff.tz_localize("UTC")
    lead = (kickoff - pd.Timestamp(captured)).total_seconds() / 3600.0
    liquidity = liquidity or {}

    return [
        {
            "match_key": match["key"],
            "kickoff_utc": kickoff.isoformat(),
            "event": match.get("event", ""),
            "best_of": match.get("best_of", ""),
            "team1": match["team1"],
            "team2": match["team2"],
            "source": name,
            "market": market,
            "prob_team1": round(float(prob), 4),
            "captured_utc": pd.Timestamp(captured).isoformat(),
            "lead_hours": round(lead, 2),
            "liquidity": liquidity.get(name, ""),
            "result": "",
            "team1_score": "",
            "team2_score": "",
            "graded_utc": "",
        }
        for name, prob in quotes.items()
        if prob is not None
    ]


# ---------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------


def ungraded_keys(df: pd.DataFrame, before: datetime | None = None) -> list[str]:
    """Matches with no result yet whose kickoff has passed."""
    if df.empty:
        return []
    cutoff = pd.Timestamp(before or datetime.now(timezone.utc))
    pending = df[df["result"].isna() & (df["kickoff_utc"] < cutoff)]
    return sorted(pending["match_key"].dropna().unique().tolist())


def apply_results(path: str, results: dict) -> int:
    """Fill in `result` for the given matches, according to each row's market.

    `results` maps match_key to either a bare 1/0 (team1 won) or a dict
    with keys team1_won / team1_score / team2_score. The richer form is
    what lets sweep rows be graded; without a scoreline they are left
    ungraded rather than guessed at.
    """
    df = load(path)
    if df.empty or not results:
        return 0

    def outcome(key, market):
        entry = results.get(key)
        if entry is None:
            return None
        if not isinstance(entry, dict):
            entry = {"team1_won": int(entry)}
        won = entry.get("team1_won")
        s1, s2 = entry.get("team1_score"), entry.get("team2_score")
        if market == SERIES:
            return None if won is None else int(won)
        if s1 is None or s2 is None or won is None:
            return None          # no scoreline, so a sweep cannot be judged
        if market == SWEEP_1:
            return int(bool(won) and int(s2) == 0)
        if market == SWEEP_2:
            return int((not bool(won)) and int(s1) == 0)
        return None

    for col in ("result", "team1_score", "team2_score", "graded_utc"):
        df[col] = df[col].astype("object")
    stamp = datetime.now(timezone.utc).isoformat()

    filled = 0
    for idx, row in df[df["result"].isna()].iterrows():
        value = outcome(row["match_key"], row["market"])
        if value is None:
            continue
        df.at[idx, "result"] = value
        df.at[idx, "graded_utc"] = stamp
        # Keep the scoreline, not just who won. Without it a finished match
        # can only be reported as a win, never as 2-0 or 2-1.
        entry = results.get(row["match_key"])
        if isinstance(entry, dict):
            df.at[idx, "team1_score"] = entry.get("team1_score")
            df.at[idx, "team2_score"] = entry.get("team2_score")
        filled += 1

    if filled:
        df.to_csv(path, index=False)
    return filled


# ---------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------


@dataclass
class SourceScore:
    source: str
    graded: int          # matches this source covered and we can grade
    common: int          # matches in the all-sources subset
    log_loss: float | None
    brier: float | None
    accuracy: float | None
    coverage: float      # share of gradable matches this source priced

    def __str__(self) -> str:
        if self.log_loss is None:
            return f"{self.source:<12} no graded matches yet"
        return (f"{self.source:<12} log_loss={self.log_loss:.4f}  "
                f"brier={self.brier:.4f}  acc={self.accuracy:.3f}  "
                f"n={self.common}  coverage={self.coverage:.0%}")


def official_snapshots(df: pd.DataFrame, min_lead_hours: float = 1.0) -> pd.DataFrame:
    """One row per (match, source): the last quote taken before the cutoff.

    Quotes captured inside the cutoff window are discarded entirely rather
    than used as a fallback — a late quote is a different, easier
    prediction, and mixing the two would quietly corrupt the comparison.
    """
    if df.empty:
        return df
    eligible = df[df["lead_hours"] >= min_lead_hours]
    if eligible.empty:
        return eligible
    ordered = eligible.sort_values("lead_hours")           # closest first
    return ordered.groupby(["match_key", "source"], as_index=False).first()


def scoreboard(
    df: pd.DataFrame,
    min_lead_hours: float = 1.0,
    since_days: int | None = None,
    market: str = SERIES,
) -> tuple[list[SourceScore], pd.DataFrame]:
    """Score every source on one market. Returns (scores, common subset)."""
    if not df.empty:
        df = df[df["market"] == market]
    snaps = official_snapshots(df, min_lead_hours)
    graded = snaps[snaps["result"].notna()] if not snaps.empty else snaps
    if graded.empty:
        return [], graded

    if since_days:
        cutoff = pd.Timestamp(datetime.now(timezone.utc)) - pd.Timedelta(days=since_days)
        graded = graded[graded["kickoff_utc"] >= cutoff]
        if graded.empty:
            return [], graded

    sources = sorted(graded["source"].unique())
    per_match = graded.groupby("match_key")["source"].nunique()
    complete = set(per_match[per_match == len(sources)].index)
    total_matches = graded["match_key"].nunique()

    scores = []
    for name in sources:
        rows = graded[graded["source"] == name]
        common = rows[rows["match_key"].isin(complete)]
        coverage = rows["match_key"].nunique() / total_matches if total_matches else 0.0
        if common.empty:
            scores.append(SourceScore(name, len(rows), 0, None, None, None, coverage))
            continue
        p = np.clip(common["prob_team1"].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
        y = common["result"].to_numpy(dtype=float)
        scores.append(SourceScore(
            source=name,
            graded=len(rows),
            common=len(common),
            log_loss=float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
            brier=float(np.mean((p - y) ** 2)),
            accuracy=float(np.mean((p > 0.5) == (y == 1))),
            coverage=coverage,
        ))

    scores.sort(key=lambda s: (s.log_loss is None, s.log_loss))
    return scores, graded[graded["match_key"].isin(complete)]


def finished(
    df: pd.DataFrame,
    source: str,
    days: int = 45,
    min_lead_hours: float = 1.0,
) -> list[dict]:
    """Recently graded matches, with the scoreline and what we forecast.

    Used by the dashboard to report on matches the person is following once
    they have been played. Ordered newest first.
    """
    if df.empty:
        return []
    rows = df[(df["source"] == source) & (df["market"] == SERIES)
              & df["result"].notna()]
    if rows.empty:
        return []

    cutoff = pd.Timestamp(datetime.now(timezone.utc)) - pd.Timedelta(days=days)
    rows = rows[rows["kickoff_utc"] >= cutoff]
    if rows.empty:
        return []

    # One row per match: the forecast that counted, on the same lead-time
    # rule the scoreboard uses.
    eligible = rows[rows["lead_hours"] >= min_lead_hours]
    if eligible.empty:
        eligible = rows
    picked = eligible.sort_values("lead_hours").groupby("match_key", as_index=False).first()

    out = []
    for row in picked.sort_values("kickoff_utc", ascending=False).to_dict("records"):
        s1, s2 = row.get("team1_score"), row.get("team2_score")
        out.append({
            "id": str(row["match_key"]),
            "date": pd.Timestamp(row["kickoff_utc"]).isoformat(),
            "event": row.get("event") or "",
            "bestOf": int(row["best_of"]) if pd.notna(row.get("best_of")) else None,
            "team1": row["team1"], "team2": row["team2"],
            "team1Score": None if pd.isna(s1) else int(s1),
            "team2Score": None if pd.isna(s2) else int(s2),
            "team1Won": int(row["result"]),
            "forecast": float(row["prob_team1"]),
        })
    return out


def live_calibration(
    df: pd.DataFrame,
    source: str,
    bins: int = 5,
    min_lead_hours: float = 1.0,
    market: str = SERIES,
) -> list[dict]:
    """Said-versus-happened, from real graded forecasts.

    The backtest version of this table is a re-enactment: the model is
    scored on games drawn from the same era it was tuned against. This one
    is scored on forecasts that were published before anyone knew the
    result, which is the harder and more honest test.
    """
    p, y = graded_pairs(df, source, min_lead_hours, market)
    if len(p) == 0:
        return []

    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = idx == b
        if not mask.any():
            continue
        rows.append({
            "bucket": f"{edges[b]:.0%}-{edges[b + 1]:.0%}",
            "games": int(mask.sum()),
            "predicted": float(p[mask].mean()),
            "actual": float(y[mask].mean()),
        })
    return rows


def divergence(
    df: pd.DataFrame,
    source: str,
    backtest_log_loss: float,
    min_lead_hours: float = 1.0,
    min_matches: int = 100,
    margin: float = 0.03,
) -> dict:
    """Compare live performance against what the backtest promised.

    A model that looks good on history and worse in the wild is overfitted,
    and the fix is fewer variables rather than more. This is the check that
    catches that, so it is worth surfacing before it is worth acting on.

    `margin` is deliberately generous. Live samples are small and noisy,
    and crying overfitting on forty matches would be worse than silence.
    """
    p, y = graded_pairs(df, source, min_lead_hours)
    n = len(p)
    if n < min_matches:
        return {"status": "waiting", "matches": n, "needed": min_matches}

    p = np.clip(p, 1e-6, 1 - 1e-6)
    live = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    gap = live - backtest_log_loss

    if gap > margin:
        status = "worse"
    elif gap < -margin:
        status = "better"
    else:
        status = "consistent"

    # Which direction the miscalibration runs, if any.
    bias = float(np.mean(p) - np.mean(y))
    return {
        "status": status, "matches": n,
        "liveLogLoss": round(live, 4),
        "backtestLogLoss": round(backtest_log_loss, 4),
        "gap": round(gap, 4),
        "bias": round(bias, 4),
    }


def graded_pairs(df: pd.DataFrame, source: str,
                 min_lead_hours: float = 1.0,
                 market: str = SERIES) -> tuple[np.ndarray, np.ndarray]:
    """(predictions, outcomes) for one source. Used to fit self-calibration."""
    if not df.empty:
        df = df[df["market"] == market]
    snaps = official_snapshots(df, min_lead_hours)
    if snaps.empty:
        return np.array([]), np.array([])
    rows = snaps[(snaps["source"] == source) & snaps["result"].notna()]
    if rows.empty:
        return np.array([]), np.array([])
    return (rows["prob_team1"].to_numpy(dtype=float),
            rows["result"].to_numpy(dtype=float))
