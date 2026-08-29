"""Bookmaker odds — a working template, switched off by default.

There is no free, reliable bookmaker feed for LoL esports. Add this only
once Polymarket's coverage proves too thin for the leagues you follow;
otherwise you would be paying for a second opinion you do not need.

To turn it on:
  1. Sign up with an odds provider that covers LoL esports and get a key.
  2. Put the key in an environment variable, not in a file. In GitHub:
     Settings -> Secrets and variables -> Actions -> New secret,
     named ODDS_API_KEY.
  3. Fill in ENDPOINT and the parsing in `_parse` below to match your
     provider's response shape.
  4. Add "bookmaker" to sources.enabled in config.yaml.

The odds maths is already done: decimal odds are converted and the
bookmaker's margin removed, so what lands in the ledger is a fair
probability comparable to everyone else's.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import requests

from . import Quote, implied_from_decimal, match_pair, remove_margin, source

ENDPOINT = ""      # e.g. "https://api.example.com/v1/esports/lol/odds"
KEY_ENV = "ODDS_API_KEY"


@source("bookmaker", label="Bookmakers", colour="#E0A458", needs_key=True)
def fetch(matches, config) -> list[Quote]:
    key = os.environ.get(KEY_ENV)
    if not key or not ENDPOINT:
        # Silent no-op rather than a crash: the source is configured but
        # not yet set up, and a missing comparison should never stop the
        # forecast from being produced.
        return []

    response = requests.get(
        ENDPOINT,
        params={"apiKey": key, "regions": "eu", "markets": "h2h",
                "oddsFormat": "decimal"},
        headers={"User-Agent": config["data"]["user_agent"]},
        timeout=30,
    )
    response.raise_for_status()

    now = datetime.now(timezone.utc)
    quotes = []
    for offer in _parse(response.json()):
        hit = next(
            (m for m in matches
             if match_pair(offer["team_a"], offer["team_b"],
                           m["team1"], m["team2"])),
            None,
        )
        if hit is None:
            continue
        raw_a = implied_from_decimal(offer["odds_a"])
        raw_b = implied_from_decimal(offer["odds_b"])
        fair_a, _ = remove_margin(raw_a, raw_b)
        quotes.append(Quote(
            match_key=hit["key"],
            quoted_team=offer["team_a"],
            prob_quoted=round(fair_a, 4),
            captured=now,
            note=offer.get("book"),
        ))
    return quotes


def _parse(payload) -> list[dict]:
    """Reshape your provider's response into a flat list.

    Expected keys per item: team_a, team_b, odds_a, odds_b, book.
    Most providers nest odds under a bookmakers -> markets -> outcomes
    chain; if several books are returned, average the fair probabilities
    rather than picking one, since a consensus line is a stronger
    benchmark than any single book.
    """
    return []
