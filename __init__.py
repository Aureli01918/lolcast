"""Comparison sources.

A "source" is anything that publishes a win probability before a match:
a prediction market, a bookmaker, another model, or one of our own
baselines. Each one lives in its own file and registers itself here.

To add a source, write a file in this folder:

    @source("my_site")
    def fetch(matches, config):
        return [Quote(match_key=..., prob_team1=..., ...), ...]

It then appears automatically in the ledger, the accuracy scoreboard, and
on every bar in the dashboard. No other file changes.

The one rule: a source must return the probability that **Team 1 wins the
series**, using the same team-1/team-2 orientation as the match passed in.
Getting this backwards silently inverts everything, so `Quote` asks you to
name the team you are quoting and the loader checks it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

# ---------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------

REGISTRY: dict[str, "SourceSpec"] = {}


@dataclass
class Quote:
    """One source's opinion on one match, at one moment."""

    match_key: str
    quoted_team: str      # which team prob_quoted refers to
    prob_quoted: float
    captured: datetime
    # Optional context the scoreboard can show.
    liquidity: float | None = None
    note: str | None = None

    def prob_for(self, team1: str, team2: str) -> float | None:
        """Re-orient to 'probability team1 wins'."""
        if same_team(self.quoted_team, team1):
            return self.prob_quoted
        if same_team(self.quoted_team, team2):
            return 1.0 - self.prob_quoted
        return None


@dataclass
class SourceSpec:
    name: str
    label: str
    colour: str
    fetch: Callable
    needs_key: bool = False


def source(name: str, label: str | None = None, colour: str = "#9AA7AD",
           needs_key: bool = False):
    def register(fn):
        REGISTRY[name] = SourceSpec(
            name=name, label=label or name, colour=colour,
            fetch=fn, needs_key=needs_key,
        )
        return fn

    return register


def enabled_sources(config: dict) -> list[SourceSpec]:
    names = config.get("sources", {}).get("enabled", [])
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        raise KeyError(f"config lists unknown sources: {unknown}. "
                       f"Registered: {sorted(REGISTRY)}")
    return [REGISTRY[n] for n in names]


# ---------------------------------------------------------------------
# Odds maths
# ---------------------------------------------------------------------


def implied_from_decimal(odds: float) -> float:
    """Decimal odds to a raw implied probability. 2.50 -> 0.40."""
    if odds <= 1.0:
        raise ValueError(f"decimal odds must exceed 1.0, got {odds}")
    return 1.0 / odds


def remove_margin(p1: float, p2: float) -> tuple[float, float]:
    """Strip the built-in margin from a two-way market.

    A bookmaker's two sides deliberately sum to more than 100% -- that
    surplus is their cut, called the overround. Quoting those raw numbers
    as probabilities would make every source look overconfident. Scaling
    both sides so they sum to 1 is the standard fix.

    Prediction-market prices have the same issue for a different reason:
    the gap between the best buy and sell price.
    """
    total = p1 + p2
    if total <= 0:
        raise ValueError("probabilities must be positive")
    return p1 / total, p2 / total


# ---------------------------------------------------------------------
# Team-name matching
# ---------------------------------------------------------------------
# Sources spell teams their own way: "Gen.G" / "GenG" / "Gen G Esports".
# Matching is deliberately conservative -- an unmatched quote is dropped
# and reported, because a wrong match corrupts the accuracy record and
# you would never notice.

_NOISE = {"esports", "esport", "gaming", "team", "club", "the"}
# Stripped from the ends of the run-together form, so "TopEsports" and
# "Top Esports" resolve to the same thing.
_AFFIXES = ("esports", "esport", "gaming", "team", "club")


def canonical(name: str) -> str:
    cleaned = re.sub(r"[^a-z0-9 ]", " ", (name or "").lower())
    words = [w for w in cleaned.split() if w and w not in _NOISE]
    return " ".join(words) or re.sub(r"[^a-z0-9 ]", " ", (name or "").lower()).strip()


def compact(name: str) -> str:
    """Spacing-insensitive form: 'Gen.G', 'Gen G' and 'GenG' all collapse."""
    text = re.sub(r"[^a-z0-9]", "", (name or "").lower())
    changed = True
    while changed:
        changed = False
        for affix in _AFFIXES:
            for stripped in (text[: -len(affix)] if text.endswith(affix) else None,
                             text[len(affix):] if text.startswith(affix) else None):
                if stripped:            # never strip down to nothing
                    text, changed = stripped, True
                    break
            if changed:
                break
    return text


def same_team(a: str, b: str) -> bool:
    ca, cb = canonical(a), canonical(b)
    if not ca or not cb:
        return False
    if ca == cb or compact(a) == compact(b):
        return True
    # One name extending the other, e.g. "t1" vs "t1 challengers".
    return ca.startswith(cb + " ") or cb.startswith(ca + " ")


def match_pair(quote_a: str, quote_b: str, team1: str, team2: str) -> bool:
    """True when a source's two teams are our two teams, either way round."""
    return (
        (same_team(quote_a, team1) and same_team(quote_b, team2))
        or (same_team(quote_a, team2) and same_team(quote_b, team1))
    )


# Import adapters so their decorators run. Keep at the bottom: they import
# names defined above.
from . import bookmaker, polymarket  # noqa: E402,F401
