"""Measuring roster churn, before deciding whether to model it.

The open question is not "would player ratings be nice" but "does our team
rating actually go stale when a lineup changes, and by how much". That is
measurable on data we can already reach, and it is much cheaper to measure
than to model.

Two numbers come out:

  1. How often a major-league team fields a different five than last game.
  2. Whether our forecasts are measurably worse in the games after a change.

If churn is rare and the forecast damage is small, player ratings are not
worth the tenfold increase in data collection, and the honest answer is to
stop here. This module exists to make that call on evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# ScoreboardPlayers holds one row per player per game. Field names are
# tried in descending order of usefulness: if the wiki rejects one, we fall
# back rather than failing the whole audit.
FIELD_SETS = [
    ["Link", "Team", "GameId", "DateTime_UTC", "OverviewPage", "IngameRole"],
    ["Link", "Team", "GameId", "DateTime_UTC", "OverviewPage"],
    ["Link", "Team", "GameId", "DateTime_UTC"],
    ["Link", "Team", "GameId"],
]


def lineups_from_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Collapse player rows into one lineup per team per game.

    Returns columns: gameid, team, date, lineup (a frozenset of players),
    size (how many players we actually saw).
    """
    if rows.empty:
        return pd.DataFrame(columns=["gameid", "team", "date", "lineup", "size"])

    frame = rows.dropna(subset=["gameid", "team", "player"]).copy()
    if "date" in frame:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")

    grouped = frame.groupby(["gameid", "team"], as_index=False).agg(
        date=("date", "min"),
        lineup=("player", lambda s: frozenset(str(x) for x in s)),
    )
    grouped["size"] = grouped["lineup"].map(len)
    return grouped.sort_values("date").reset_index(drop=True)


def annotate_changes(lineups: pd.DataFrame, expect: int = 5) -> pd.DataFrame:
    """For each team-game, how much the five changed since that team's last game.

    Incomplete lineups are dropped rather than treated as changes. A game
    where the wiki only recorded four players would otherwise look like a
    substitution, which is exactly the kind of quiet error that makes an
    audit worse than useless.
    """
    if lineups.empty:
        return lineups.assign(changed=[], swaps=[], games_since_change=[])

    usable = lineups[lineups["size"] == expect].copy()
    out = []
    for team, group in usable.sort_values("date").groupby("team"):
        previous = None
        since = np.nan
        for row in group.to_dict("records"):
            if previous is None:
                row["swaps"] = 0
                row["changed"] = False
                since = np.nan
            else:
                swaps = len(row["lineup"] - previous)
                row["swaps"] = swaps
                row["changed"] = swaps > 0
                since = 0 if swaps > 0 else (0 if np.isnan(since) else since + 1)
            row["games_since_change"] = since
            previous = row["lineup"]
            out.append(row)

    result = pd.DataFrame(out)
    return result.sort_values("date").reset_index(drop=True)


@dataclass
class ChurnReport:
    team_games: int
    teams: int
    changed_games: int
    swap_histogram: dict
    dropped_incomplete: int

    @property
    def change_rate(self) -> float:
        return self.changed_games / self.team_games if self.team_games else 0.0

    def __str__(self) -> str:
        lines = [
            f"Team-games examined : {self.team_games:,} across {self.teams} teams",
            f"Lineup differed     : {self.changed_games:,} "
            f"({self.change_rate:.1%} of team-games)",
            f"Dropped (incomplete): {self.dropped_incomplete:,}",
            "Players swapped since previous game:",
        ]
        for swaps in sorted(self.swap_histogram):
            count = self.swap_histogram[swaps]
            share = count / self.team_games if self.team_games else 0
            lines.append(f"  {swaps} changed: {count:,} ({share:.1%})")
        return "\n".join(lines)


def churn_report(lineups: pd.DataFrame, annotated: pd.DataFrame) -> ChurnReport:
    if annotated.empty:
        return ChurnReport(0, 0, 0, {}, len(lineups))
    # The first game of each team has no predecessor and cannot be judged.
    judged = annotated.groupby("team").apply(
        lambda g: g.iloc[1:], include_groups=False
    ).reset_index(drop=True)
    if judged.empty:
        return ChurnReport(0, annotated["team"].nunique(), 0, {}, 0)
    return ChurnReport(
        team_games=len(judged),
        teams=int(annotated["team"].nunique()),
        changed_games=int(judged["changed"].sum()),
        swap_histogram=judged["swaps"].value_counts().to_dict(),
        dropped_incomplete=int(len(lineups) - len(annotated)),
    )


# ---------------------------------------------------------------------
# Does churn actually hurt the forecast?
# ---------------------------------------------------------------------


def flag_recent_change(
    detail: pd.DataFrame,
    annotated: pd.DataFrame,
    window: int = 3,
) -> pd.DataFrame:
    """Mark each backtested game by whether either side changed recently.

    `detail` is the walk-forward output: date, blue, red, prediction, actual.
    """
    if annotated.empty or detail.empty:
        return detail.assign(recent_change=False, matched=False)

    recent = annotated[
        annotated["changed"]
        | (annotated["games_since_change"].fillna(99) < window)
    ]
    keys = {(str(t), pd.Timestamp(d).floor("D"))
            for t, d in zip(recent["team"], recent["date"])}
    known_teams = set(annotated["team"].astype(str))

    out = detail.copy()
    out["_day"] = pd.to_datetime(out["date"]).dt.floor("D")
    out["matched"] = (out["blue"].astype(str).isin(known_teams)
                      & out["red"].astype(str).isin(known_teams))
    out["recent_change"] = [
        ((str(b), d) in keys) or ((str(r), d) in keys)
        for b, r, d in zip(out["blue"], out["red"], out["_day"])
    ]
    return out.drop(columns="_day")


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def compare_groups(flagged: pd.DataFrame, draws: int = 2000, seed: int = 0) -> dict:
    """Log loss with and without a recent roster change, plus a spread.

    The bootstrap interval matters more than the point estimate here. With
    a few hundred games in the changed group, a gap of 0.02 is entirely
    consistent with noise, and reading it as a finding would send us off
    building player ratings for nothing.
    """
    usable = flagged[flagged["matched"]]
    changed = usable[usable["recent_change"]]
    stable = usable[~usable["recent_change"]]
    if len(changed) < 30 or len(stable) < 30:
        return {"enough_data": False,
                "changed_n": len(changed), "stable_n": len(stable)}

    def arrays(frame):
        return (frame["actual"].to_numpy(dtype=float),
                frame["prediction"].to_numpy(dtype=float))

    y_c, p_c = arrays(changed)
    y_s, p_s = arrays(stable)
    observed = _log_loss(y_c, p_c) - _log_loss(y_s, p_s)

    rng = np.random.default_rng(seed)
    diffs = np.empty(draws)
    for i in range(draws):
        ic = rng.integers(0, len(y_c), len(y_c))
        is_ = rng.integers(0, len(y_s), len(y_s))
        diffs[i] = _log_loss(y_c[ic], p_c[ic]) - _log_loss(y_s[is_], p_s[is_])
    low, high = np.percentile(diffs, [2.5, 97.5])

    return {
        "enough_data": True,
        "changed_n": len(changed), "stable_n": len(stable),
        "changed_log_loss": _log_loss(y_c, p_c),
        "stable_log_loss": _log_loss(y_s, p_s),
        "difference": observed,
        "ci_low": float(low), "ci_high": float(high),
        # Zero inside the interval means we cannot distinguish the two.
        "significant": bool(low > 0 or high < 0),
    }


def verdict(report: ChurnReport, comparison: dict, window: int) -> str:
    if not comparison.get("enough_data"):
        return ("Not enough matched games to judge. Widen the audit window "
                "or include more leagues, then re-run.")

    diff = comparison["difference"]
    if not comparison["significant"]:
        return (
            f"No measurable damage. Games within {window} of a lineup change "
            f"score {diff:+.4f} log loss versus stable ones, and the interval "
            f"[{comparison['ci_low']:+.4f}, {comparison['ci_high']:+.4f}] "
            f"includes zero. On this evidence, player ratings are not worth "
            f"the tenfold data cost. Revisit if churn rises."
        )
    if diff > 0:
        return (
            f"Roster changes do hurt. Forecasts are {diff:+.4f} log loss worse "
            f"after a change ({comparison['ci_low']:+.4f} to "
            f"{comparison['ci_high']:+.4f}), on {comparison['changed_n']:,} "
            f"games. Worth handling -- start by widening uncertainty after a "
            f"change before building full player ratings."
        )
    return (
        f"Forecasts are better after roster changes ({diff:+.4f}), which is "
        f"the opposite of the hypothesis. Likely churn concentrates in "
        f"mismatched games that are easy to call. Do not build player "
        f"ratings on this evidence."
    )
