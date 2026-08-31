"""Polymarket prediction-market prices.

Public read-only API, no key and no signup. Prices are already
probabilities: a team trading at 63c means the market thinks there is a
63% chance they win.

Two traps this handles:

1. `outcomes` and `outcomePrices` arrive as JSON-encoded *strings*, not
   arrays. Indexing them directly gives you a bracket character.
2. The two sides rarely sum to exactly 1.00, because of the gap between
   the best buy and sell price. We rescale so they do -- otherwise the
   market looks systematically overconfident against our model.

Market questions look like:
    "LoL: Top Esports vs EDward Gaming (BO3) - LPL Group Ascend"
so team names are parsed out of the question text.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

import requests

from . import Quote, match_pair, remove_margin, source

GAMMA = "https://gamma-api.polymarket.com"
TAG_SLUG = "league-of-legends"

# "LoL: A vs B (BO3) - Event"  ->  captures A and B
QUESTION = re.compile(
    r"^\s*(?:lol|league of legends)\s*[:\-]\s*(.+?)\s+vs\.?\s+(.+?)\s*(?:\(|-|$)",
    re.IGNORECASE,
)


def parse_question(question: str) -> tuple[str, str] | None:
    hit = QUESTION.match(question or "")
    if hit:
        return hit.group(1).strip(), hit.group(2).strip()
    # Fall back to a bare "A vs B" with no prefix.
    plain = re.match(r"^\s*(.+?)\s+vs\.?\s+(.+?)\s*(?:\(|-|$)", question or "",
                     re.IGNORECASE)
    return (plain.group(1).strip(), plain.group(2).strip()) if plain else None


def fetch_events(user_agent: str, limit: int = 100, max_pages: int = 5) -> list[dict]:
    """Page through active LoL events."""
    events, offset = [], 0
    for _ in range(max_pages):
        response = requests.get(
            f"{GAMMA}/events",
            params={
                "tag_slug": TAG_SLUG,
                "active": "true",
                "closed": "false",
                "limit": limit,
                "offset": offset,
            },
            headers={"User-Agent": user_agent},
            timeout=30,
        )
        response.raise_for_status()
        batch = response.json()
        if isinstance(batch, dict):          # some responses wrap the list
            batch = batch.get("data", [])
        if not batch:
            break
        events.extend(batch)
        if len(batch) < limit:
            break
        offset += limit
    return events


def _decode(value):
    """Gamma returns these fields as JSON strings. Decode defensively."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return value


def collect_markets(events: list[dict]) -> list[dict]:
    """Flatten to two-outcome match markets with usable prices."""
    out = []
    for event in events:
        for market in event.get("markets", []) or []:
            if market.get("closed") or market.get("active") is False:
                continue
            outcomes = _decode(market.get("outcomes"))
            prices = _decode(market.get("outcomePrices"))
            if not outcomes or not prices or len(outcomes) != 2 or len(prices) != 2:
                continue
            teams = parse_question(market.get("question", ""))
            if not teams:
                continue
            try:
                p1, p2 = float(prices[0]), float(prices[1])
            except (TypeError, ValueError):
                continue
            if p1 <= 0 or p2 <= 0:
                continue
            out.append({
                "teams": teams,
                "outcomes": [str(o) for o in outcomes],
                "prices": (p1, p2),
                "liquidity": _as_float(market.get("liquidityNum")
                                       or market.get("liquidity")),
                "volume": _as_float(market.get("volumeNum") or market.get("volume")),
                "question": market.get("question"),
            })
    return out


def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@source("polymarket", label="Polymarket", colour="#5FD3A6")
def fetch(matches, config) -> list[Quote]:
    """`matches` is a list of dicts with key / team1 / team2."""
    user_agent = config["data"]["user_agent"]
    settings = config.get("sources", {}).get("polymarket", {})
    min_liquidity = float(settings.get("min_liquidity", 0) or 0)

    markets = collect_markets(fetch_events(user_agent))
    now = datetime.now(timezone.utc)
    quotes = []

    for match in matches:
        team1, team2 = match["team1"], match["team2"]
        hit = next(
            (m for m in markets if match_pair(m["teams"][0], m["teams"][1],
                                              team1, team2)),
            None,
        )
        if hit is None:
            continue
        if min_liquidity and (hit["liquidity"] or 0) < min_liquidity:
            continue

        # The outcome labels, not the question order, say which price is
        # which -- do not assume they line up.
        fair = remove_margin(hit["prices"][0], hit["prices"][1])
        pairs = list(zip(hit["outcomes"], fair))
        chosen = next(
            (p for label, p in pairs
             if _looks_like(label, team1) or _looks_like(label, team2)),
            None,
        )
        chosen_label = next(
            (label for label, _ in pairs
             if _looks_like(label, team1) or _looks_like(label, team2)),
            None,
        )
        if chosen is None:
            # Labels were "Yes"/"No" or similar; fall back to question order.
            chosen_label, chosen = hit["teams"][0], fair[0]

        quotes.append(Quote(
            match_key=match["key"],
            quoted_team=chosen_label,
            prob_quoted=round(chosen, 4),
            captured=now,
            liquidity=hit["liquidity"],
            note=hit["question"],
        ))

    return quotes


def _looks_like(label: str, team: str) -> bool:
    from . import same_team
    return same_team(label, team)
