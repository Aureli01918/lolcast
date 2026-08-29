"""Feature registry.

This is where you add new variables.

To add one:

    @feature("my_variable")
    def my_variable(ctx):
        return some_number

Then list "my_variable" under features.enabled in config.yaml and run
`python -m lolcast backtest` to see whether it earned its place.

A feature receives a FeatureContext describing the game about to be played.
It may only read state from *before* that game. The context deliberately
does not expose the result, so leakage is hard to write by accident.

Every feature must return a single float.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable

from .ratings import RatingSystem, TeamState


@dataclass
class FeatureContext:
    """What a feature is allowed to know before a game."""

    blue: str
    red: str
    date: datetime
    league: str
    patch: str | None
    best_of: int
    blue_state: TeamState
    red_state: TeamState
    ratings: RatingSystem
    # Free-form extras from the source row (region, event, etc). Use this
    # rather than adding parameters when you need something new.
    meta: dict


FeatureFn = Callable[[FeatureContext], float]
REGISTRY: dict[str, FeatureFn] = {}


def feature(name: str) -> Callable[[FeatureFn], FeatureFn]:
    def register(fn: FeatureFn) -> FeatureFn:
        if name in REGISTRY:
            raise ValueError(f"feature {name!r} is already registered")
        REGISTRY[name] = fn
        return fn

    return register


def build_row(ctx: FeatureContext, enabled: list[str]) -> list[float]:
    missing = [n for n in enabled if n not in REGISTRY]
    if missing:
        raise KeyError(
            f"config lists unknown features: {missing}. "
            f"Registered: {sorted(REGISTRY)}"
        )
    return [float(REGISTRY[name](ctx)) for name in enabled]


# ---------------------------------------------------------------------
# Core features
# ---------------------------------------------------------------------


@feature("elo_diff")
def elo_diff(ctx: FeatureContext) -> float:
    """Rating gap, scaled so a value of 1.0 is roughly one 400-point gap."""
    return (ctx.blue_state.elo - ctx.red_state.elo) / 400.0


@feature("is_blue_side")
def is_blue_side(ctx: FeatureContext) -> float:
    """Constant 1. Lets the model fit its own side coefficient."""
    return 1.0


@feature("form_diff_10")
def form_diff_10(ctx: FeatureContext) -> float:
    """Win-rate gap over each team's last 10 games.

    Partly redundant with Elo, which is the point: if it adds nothing the
    backtest will show a coefficient near zero.
    """
    return ctx.blue_state.form(10) - ctx.red_state.form(10)


@feature("rest_days_diff")
def rest_days_diff(ctx: FeatureContext) -> float:
    """Difference in days since each team last played, capped at 14."""
    b = min(ctx.blue_state.rest_days(ctx.date), 14.0)
    r = min(ctx.red_state.rest_days(ctx.date), 14.0)
    return (b - r) / 14.0


@feature("h2h_recent")
def h2h_recent(ctx: FeatureContext) -> float:
    """Recent head-to-head record, shrunk toward neutral when sparse."""
    rate, n = ctx.ratings.head_to_head(ctx.blue, ctx.red, n=6)
    shrink = n / (n + 4.0)
    return (rate - 0.5) * 2.0 * shrink


@feature("games_played_diff")
def games_played_diff(ctx: FeatureContext) -> float:
    """How settled each rating is. Proxy for confidence, not for skill."""
    b = min(ctx.blue_state.games, 60) / 60.0
    r = min(ctx.red_state.games, 60) / 60.0
    return b - r


@feature("cross_region")
def cross_region(ctx: FeatureContext) -> float:
    """1 when the two teams come from different regions.

    Cross-region ratings are the least reliable part of any Elo system,
    so the model gets a chance to shrink its confidence on these games.
    """
    b = ctx.meta.get("blue_region")
    r = ctx.meta.get("red_region")
    return 1.0 if (b and r and b != r) else 0.0


# ---------------------------------------------------------------------
# Optional features: registered but off by default.
# Turn one on in config.yaml, run the backtest, keep it if it helps.
# ---------------------------------------------------------------------


def _stat_mean(state: TeamState, key: str, n: int, default: float = 0.0) -> float:
    vals = [s.get(key) for s in state.stat_history[-n:] if s.get(key) is not None]
    return sum(vals) / len(vals) if vals else default


@feature("gold_diff_15_form")
def gold_diff_15_form(ctx: FeatureContext) -> float:
    """Average gold difference at 15 minutes over the last 10 games.

    Early-game strength is more stable than win rate, so this sometimes
    beats form as a short-horizon signal.
    """
    b = _stat_mean(ctx.blue_state, "golddiffat15", 10)
    r = _stat_mean(ctx.red_state, "golddiffat15", 10)
    return (b - r) / 1000.0


@feature("roster_stability")
def roster_stability(ctx: FeatureContext) -> float:
    """Share of the last 10 games played by the current five starters.

    Requires meta["blue_roster"] / meta["red_roster"] to be populated by
    the loader. Returns 0 when unavailable.
    """
    b = ctx.meta.get("blue_roster_stability")
    r = ctx.meta.get("red_roster_stability")
    if b is None or r is None:
        return 0.0
    return float(b) - float(r)


@feature("patch_recency")
def patch_recency(ctx: FeatureContext) -> float:
    """Games played on the current patch, as a fraction of 10.

    Early on a new patch, ratings carry stale information about the meta.
    """
    patch = ctx.patch
    if not patch:
        return 1.0
    seen = sum(
        1 for s in ctx.blue_state.stat_history[-10:] if s.get("patch") == patch
    )
    return seen / 10.0
