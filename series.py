"""Scoreline probabilities for a series.

The existing series maths treats each game as an independent coin flip at
the same probability. That is wrong in a specific, systematic way: sides
are not fixed across a series. The loser of a game picks side for the next
one, and picks blue. So winning a game hands your opponent the side
advantage for the following game.

The practical consequence is that a sweep is less likely than squaring the
per-game probability suggests. A 65% favourite does not have a 42% chance
of going 2-0; the side swap pulls it toward 39%.

This walks the series game by game, tracking the score and who lost last,
and returns the full distribution over final scorelines. The series
probability falls out as a sum, and the sweep is a single cell.

Nothing here assumes the side convention holds perfectly.
`side_choice_rate` is measured from history rather than hard-coded, so if
the rulebook differs from what we expect the model still matches reality.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass
class Scorelines:
    """Distribution over final scores, from team A's point of view."""

    best_of: int
    # (a_wins, b_wins) -> probability
    outcomes: dict[tuple[int, int], float]

    @property
    def needed(self) -> int:
        return self.best_of // 2 + 1

    def series_win(self) -> float:
        """P(A wins the series)."""
        return sum(p for (a, _), p in self.outcomes.items() if a == self.needed)

    def sweep(self, team: str = "a") -> float:
        """P(a clean sweep): 2-0 in a Bo3, 3-0 in a Bo5, and so on."""
        need = self.needed
        target = (need, 0) if team == "a" else (0, need)
        return self.outcomes.get(target, 0.0)

    def as_dict(self) -> dict[str, float]:
        """Scorelines keyed as '2-0', '2-1', ... from A's side."""
        return {f"{a}-{b}": p for (a, b), p in sorted(
            self.outcomes.items(), key=lambda kv: (-kv[0][0], kv[0][1])
        )}

    def total(self) -> float:
        return sum(self.outcomes.values())


def scoreline_distribution(
    p_blue: float,
    p_red: float,
    best_of: int,
    side_choice_rate: float = 1.0,
    first_game_blue: float = 0.5,
) -> Scorelines:
    """Full scoreline distribution for team A.

    p_blue / p_red
        A's chance of winning one game, from each side.
    side_choice_rate
        How often the previous game's loser ends up on blue. Measured from
        history by `fit_side_choice`. At 0.5 side selection is effectively
        random and this reduces to the naive model.
    first_game_blue
        A's chance of being on blue in game one. Schedules do not say who
        starts where, so 0.5 is the honest default.
    """
    if best_of < 1:
        raise ValueError("best_of must be at least 1")
    need = best_of // 2 + 1

    # state: (a_wins, b_wins, who_lost_the_last_game) -> probability
    states: dict[tuple[int, int, str | None], float] = {(0, 0, None): 1.0}
    finished: dict[tuple[int, int], float] = defaultdict(float)

    for _ in range(best_of):
        nxt: dict[tuple[int, int, str | None], float] = defaultdict(float)
        for (a_wins, b_wins, last_loser), prob in states.items():
            if prob <= 0:
                continue

            if last_loser is None:
                a_on_blue = first_game_blue
            elif last_loser == "a":
                a_on_blue = side_choice_rate
            else:
                a_on_blue = 1.0 - side_choice_rate

            # Mixing over sides here is exact: the side only affects this
            # game's outcome, and the next game's sides depend solely on
            # who loses this one.
            p_a = a_on_blue * p_blue + (1.0 - a_on_blue) * p_red

            for winner, weight in (("a", p_a), ("b", 1.0 - p_a)):
                na = a_wins + (1 if winner == "a" else 0)
                nb = b_wins + (1 if winner == "b" else 0)
                if na == need or nb == need:
                    finished[(na, nb)] += prob * weight
                else:
                    nxt[(na, nb, "b" if winner == "a" else "a")] += prob * weight
        states = nxt
        if not states:
            break

    return Scorelines(best_of=best_of, outcomes=dict(finished))


def fit_side_choice(games, min_series: int = 50) -> tuple[float, int]:
    """Measure how often the previous game's loser appears on blue.

    Series are reconstructed from the history itself: games between the
    same two teams on the same day, in time order. That avoids needing a
    match identifier column, so it works on history that has already been
    downloaded.

    Returns (rate, number of game pairs measured). Falls back to 1.0 when
    there is too little evidence to measure anything.
    """
    import pandas as pd

    if games is None or len(games) == 0:
        return 1.0, 0

    frame = games.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date", "blue", "red", "blue_win"])
    frame["_day"] = frame["date"].dt.floor("D")
    frame["_pair"] = [
        "|".join(sorted((str(b), str(r))))
        for b, r in zip(frame["blue"], frame["red"])
    ]

    agree = total = 0
    for _, series in frame.sort_values("date").groupby(["_day", "_pair"]):
        if len(series) < 2:
            continue
        rows = series.to_dict("records")
        for prev, curr in zip(rows, rows[1:]):
            loser = prev["red"] if prev["blue_win"] == 1 else prev["blue"]
            total += 1
            if curr["blue"] == loser:
                agree += 1

    if total < min_series:
        return 1.0, total
    return agree / total, total


def fit_game_correlation(games, min_series: int = 50) -> tuple[float, int]:
    """How often the winner of a game also wins the next one in the series.

    Compared against 50%, this is a crude read on whether momentum exists
    beyond what ratings and sides already explain. Reported rather than
    modelled: encoding an effect this weakly evidenced would be worse than
    leaving it out.
    """
    import pandas as pd

    if games is None or len(games) == 0:
        return 0.5, 0

    frame = games.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date", "blue", "red", "blue_win"])
    frame["_day"] = frame["date"].dt.floor("D")
    frame["_pair"] = [
        "|".join(sorted((str(b), str(r))))
        for b, r in zip(frame["blue"], frame["red"])
    ]

    repeats = total = 0
    for _, series in frame.sort_values("date").groupby(["_day", "_pair"]):
        if len(series) < 2:
            continue
        rows = series.to_dict("records")
        for prev, curr in zip(rows, rows[1:]):
            prev_winner = prev["blue"] if prev["blue_win"] == 1 else prev["red"]
            curr_winner = curr["blue"] if curr["blue_win"] == 1 else curr["red"]
            total += 1
            if prev_winner == curr_winner:
                repeats += 1

    if total < min_series:
        return 0.5, total
    return repeats / total, total
