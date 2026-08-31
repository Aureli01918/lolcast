"""Team rating engine.

Ratings are the backbone of the forecast. Everything else in features.py is
a correction on top of the rating difference.

Design note: ratings are updated in strict chronological order and every
feature reads the rating *before* the game it is describing. That is what
keeps the backtest honest -- there is no way for a result to leak into its
own prediction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass
class TeamState:
    """Everything we know about one team, as of a moment in time."""

    name: str
    elo: float
    games: int = 0
    last_game: datetime | None = None
    # Most recent results, newest last. 1 = win, 0 = loss.
    results: list[int] = field(default_factory=list)
    # Rolling per-game stats, newest last. Used by optional features.
    stat_history: list[dict] = field(default_factory=list)

    def form(self, n: int) -> float:
        """Win rate over the last n games. 0.5 when there is no history."""
        recent = self.results[-n:]
        return sum(recent) / len(recent) if recent else 0.5

    def rest_days(self, now: datetime) -> float:
        if self.last_game is None:
            return 7.0
        return (now - self.last_game).total_seconds() / 86400.0


class RatingSystem:
    """Elo with split regression and an optional fitted side advantage."""

    def __init__(
        self,
        initial: float = 1500.0,
        k: float = 24.0,
        scale: float = 400.0,
        split_regression: float = 0.25,
        split_gap_days: float = 45.0,
        side_advantage: float = 0.0,
    ):
        self.initial = initial
        self.k = k
        self.scale = scale
        self.split_regression = split_regression
        self.split_gap = timedelta(days=split_gap_days)
        self.side_advantage = side_advantage
        self.teams: dict[str, TeamState] = {}
        # Head-to-head log: (team_a, team_b) sorted -> list of (date, a_won)
        self.h2h: dict[tuple[str, str], list[tuple[datetime, int]]] = {}

    # -- access -------------------------------------------------------

    def get(self, name: str, now: datetime | None = None) -> TeamState:
        team = self.teams.get(name)
        if team is None:
            team = TeamState(name=name, elo=self.initial)
            self.teams[name] = team
        if now is not None:
            self._apply_split_regression(team, now)
        return team

    def _apply_split_regression(self, team: TeamState, now: datetime) -> None:
        """Pull a team back toward average after a long layoff.

        Rosters change between splits, so a stale rating overstates what we
        know. This is applied lazily, the first time we see the team again.
        """
        if team.last_game is None or self.split_regression <= 0:
            return
        if now - team.last_game < self.split_gap:
            return
        team.elo += (self.initial - team.elo) * self.split_regression
        team.last_game = now  # only regress once per gap

    # -- prediction ---------------------------------------------------

    def expected(self, blue: str, red: str, now: datetime | None = None) -> float:
        """P(blue wins this single game), from ratings and side alone."""
        b = self.get(blue, now).elo + self.side_advantage
        r = self.get(red, now).elo
        return 1.0 / (1.0 + 10 ** ((r - b) / self.scale))

    # -- update -------------------------------------------------------

    def update(
        self,
        blue: str,
        red: str,
        blue_won: int,
        date: datetime,
        stats: dict | None = None,
    ) -> None:
        b = self.get(blue, date)
        r = self.get(red, date)

        exp_b = 1.0 / (1.0 + 10 ** ((r.elo - (b.elo + self.side_advantage)) / self.scale))
        delta = self.k * (blue_won - exp_b)
        b.elo += delta
        r.elo -= delta

        for team, won in ((b, blue_won), (r, 1 - blue_won)):
            team.games += 1
            team.last_game = date
            team.results.append(won)
            if len(team.results) > 100:
                team.results.pop(0)
            if stats is not None:
                side = "blue" if team is b else "red"
                team.stat_history.append(stats.get(side, {}))
                if len(team.stat_history) > 100:
                    team.stat_history.pop(0)

        key = tuple(sorted((blue, red)))
        a_won = blue_won if key[0] == blue else 1 - blue_won
        self.h2h.setdefault(key, []).append((date, a_won))

    def head_to_head(self, a: str, b: str, n: int = 6) -> tuple[float, int]:
        """(win rate of `a` vs `b` over last n meetings, number of meetings)."""
        key = tuple(sorted((a, b)))
        log = self.h2h.get(key, [])[-n:]
        if not log:
            return 0.5, 0
        if key[0] == a:
            wins = sum(w for _, w in log)
        else:
            wins = sum(1 - w for _, w in log)
        return wins / len(log), len(log)


def fit_side_advantage(games, rating_kwargs: dict, iterations: int = 12) -> float:
    """Fit the blue-side edge in Elo points.

    The obvious estimator -- turn the overall blue win rate into Elo -- is
    biased low. The logistic curve flattens away from 50%, so lopsided
    matchups dilute the side effect, and you recover well under the true
    value.

    Instead, bisect on the value that makes the mean prediction residual
    zero once ratings have run. Costs a handful of extra passes over
    history, which is seconds. Set `side_advantage` to a number in
    config.yaml to skip it.

    `games` is a dataframe with blue / red / blue_win / date columns.
    """
    if len(games) == 0:
        return 0.0

    def mean_residual(side_adv: float) -> float:
        system = RatingSystem(side_advantage=side_adv, **rating_kwargs)
        total = 0.0
        for row in games.itertuples(index=False):
            total += row.blue_win - system.expected(row.blue, row.red, row.date)
            system.update(row.blue, row.red, int(row.blue_win), row.date)
        return total / len(games)

    lo, hi = -200.0, 200.0
    f_lo = mean_residual(lo)
    if f_lo < 0:
        return lo
    if mean_residual(hi) > 0:
        return hi

    for _ in range(iterations):
        mid = (lo + hi) / 2
        if mean_residual(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def series_win_probability(p_game: float, best_of: int) -> float:
    """Convert a per-game probability into a series probability.

    Upcoming matches are Bo1/Bo3/Bo5, so this is what actually gets shown.
    Assumes games are independent, which slightly understates the favourite
    in long series but is close enough and easy to reason about.
    """
    if best_of <= 1:
        return p_game
    need = best_of // 2 + 1
    total = 0.0
    for losses in range(need):
        games = need + losses
        ways = math.comb(games - 1, losses)
        total += ways * (p_game ** need) * ((1 - p_game) ** losses)
    return total
