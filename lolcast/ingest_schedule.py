"""Fetch upcoming matches from Leaguepedia's Cargo API.

Leaguepedia rate-limits unauthenticated clients to roughly one request per
minute, so every response is cached to disk and reused until it goes stale.
One request per run is plenty: the schedule does not change by the second.

If you start hitting the limit, create a wiki account and use `mwrogue`
(pip install mwrogue) for an authenticated client with a higher ceiling.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

FIELDS = [
    "MatchSchedule.Team1",
    "MatchSchedule.Team2",
    "MatchSchedule.DateTime_UTC",
    "MatchSchedule.BestOf",
    "MatchSchedule.OverviewPage",
    "MatchSchedule.Tab",
    "MatchSchedule.Stream",
    "MatchSchedule.MatchId",
]


def fetch(
    api: str,
    user_agent: str,
    horizon_days: int,
    cache_dir: str,
    cache_minutes: int = 60,
    limit: int = 200,
) -> pd.DataFrame:
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "schedule.json")

    payload = _read_cache(cache_path, cache_minutes)
    if payload is None:
        until = datetime.now(timezone.utc) + timedelta(days=horizon_days)
        params = {
            "action": "cargoquery",
            "format": "json",
            "tables": "MatchSchedule",
            "fields": ",".join(FIELDS),
            "where": (
                "MatchSchedule.DateTime_UTC > NOW() "
                f'AND MatchSchedule.DateTime_UTC < "{until:%Y-%m-%d %H:%M:%S}" '
                "AND MatchSchedule.Winner IS NULL"
            ),
            "order_by": "MatchSchedule.DateTime_UTC ASC",
            "limit": limit,
        }
        response = requests.get(
            api, params=params, headers={"User-Agent": user_agent}, timeout=30
        )
        response.raise_for_status()
        payload = response.json()
        if "error" in payload:
            raise RuntimeError(f"Leaguepedia API error: {payload['error']}")
        _write_cache(cache_path, payload)

    rows = []
    for item in payload.get("cargoquery", []):
        row = item.get("title", {})
        if not row.get("Team1") or not row.get("Team2"):
            continue  # bracket slot with no team assigned yet
        rows.append(
            {
                "match_id": row.get("MatchId"),
                "date": row.get("DateTime UTC") or row.get("DateTime_UTC"),
                "blue": row["Team1"],
                "red": row["Team2"],
                "best_of": int(row.get("BestOf") or 1),
                "event": row.get("OverviewPage"),
                "round": row.get("Tab"),
                "stream": row.get("Stream"),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=["match_id", "date", "blue", "red", "best_of", "event", "round", "stream"]
        )

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def fetch_results(
    api: str,
    user_agent: str,
    match_ids: list[str],
    batch_size: int = 40,
) -> dict[str, int]:
    """Look up finished matches by MatchId. Returns {match_id: 1 if Team1 won}.

    Results come from the same table the schedule came from, so a series
    graded here lines up exactly with the series that was forecast. Pulling
    results from the game-level history instead would mean reconstructing
    series boundaries, which is a needless source of quiet errors.
    """
    out: dict[str, int] = {}
    for start in range(0, len(match_ids), batch_size):
        chunk = match_ids[start:start + batch_size]
        quoted = ",".join(f'"{mid}"' for mid in chunk)
        params = {
            "action": "cargoquery",
            "format": "json",
            "tables": "MatchSchedule",
            "fields": "MatchSchedule.MatchId,MatchSchedule.Winner,"
                      "MatchSchedule.Team1Score,MatchSchedule.Team2Score",
            "where": f"MatchSchedule.MatchId IN ({quoted}) "
                     "AND MatchSchedule.Winner IS NOT NULL",
            "limit": batch_size,
        }
        response = requests.get(
            api, params=params, headers={"User-Agent": user_agent}, timeout=30
        )
        response.raise_for_status()
        payload = response.json()
        if "error" in payload:
            raise RuntimeError(f"Leaguepedia API error: {payload['error']}")

        for item in payload.get("cargoquery", []):
            row = item.get("title", {})
            match_id, winner = row.get("MatchId"), row.get("Winner")
            if not match_id or winner in (None, "", "0"):
                continue
            try:
                out[match_id] = 1 if int(winner) == 1 else 0
            except (TypeError, ValueError):
                continue

        # Respect the roughly one-request-per-minute unauthenticated limit.
        if start + batch_size < len(match_ids):
            time.sleep(61)

    return out


GAME_FIELDS = [
    "ScoreboardGames.GameId",
    "ScoreboardGames.OverviewPage",
    "ScoreboardGames.Team1",
    "ScoreboardGames.Team2",
    "ScoreboardGames.Winner",
    "ScoreboardGames.DateTime_UTC",
    "ScoreboardGames.Patch",
]


GAME_COLUMNS = ["gameid", "date", "league", "patch", "blue", "red",
                "blue_win", "best_of"]


def _league_clause(prefixes: list[str] | None) -> str:
    """Restrict to the leagues we care about, server-side.

    Filtering here rather than after download is the difference between a
    bootstrap that takes an hour and one that takes five, because the wiki
    covers hundreds of minor tournaments we never rate.
    """
    if not prefixes:
        return ""
    terms = " OR ".join(
        f'ScoreboardGames.OverviewPage LIKE "{p}/%"' for p in prefixes
    )
    return f" AND ({terms})"


def fetch_games(
    api: str,
    user_agent: str,
    start: datetime,
    end: datetime,
    league_prefixes: list[str] | None = None,
    page_size: int = 500,
    max_pages: int = 40,
    pause: float = 61.0,
) -> pd.DataFrame:
    """Finished games in a date window, shaped like the historical CSVs.

    Team1 is the blue side on Leaguepedia scoreboards.
    """
    rows, offset = [], 0

    for page in range(max_pages):
        params = {
            "action": "cargoquery",
            "format": "json",
            "tables": "ScoreboardGames",
            "fields": ",".join(GAME_FIELDS),
            "where": (
                f'ScoreboardGames.DateTime_UTC >= "{start:%Y-%m-%d %H:%M:%S}" '
                f'AND ScoreboardGames.DateTime_UTC < "{end:%Y-%m-%d %H:%M:%S}" '
                "AND ScoreboardGames.Winner IS NOT NULL"
                + _league_clause(league_prefixes)
            ),
            "order_by": "ScoreboardGames.DateTime_UTC ASC",
            "limit": page_size,
            "offset": offset,
        }
        response = requests.get(
            api, params=params, headers={"User-Agent": user_agent}, timeout=60
        )
        response.raise_for_status()
        payload = response.json()
        if "error" in payload:
            raise RuntimeError(f"Leaguepedia API error: {payload['error']}")

        batch = payload.get("cargoquery", [])
        for item in batch:
            row = item.get("title", {})
            winner = row.get("Winner")
            if not row.get("Team1") or not row.get("Team2") or winner in (None, ""):
                continue
            try:
                blue_win = 1 if int(winner) == 1 else 0
            except (TypeError, ValueError):
                continue
            rows.append({
                "gameid": row.get("GameId"),
                "date": row.get("DateTime UTC") or row.get("DateTime_UTC"),
                "league": _league_of(row.get("OverviewPage")),
                "patch": row.get("Patch"),
                "blue": row["Team1"],
                "red": row["Team2"],
                "blue_win": blue_win,
                "best_of": 1,
            })

        if len(batch) < page_size:
            break
        offset += page_size
        print(f"    {len(rows):,} games so far")
        time.sleep(pause)      # unauthenticated limit is ~1 request/minute

    if not rows:
        return pd.DataFrame(columns=GAME_COLUMNS)

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date"])
    df["date"] = df["date"].dt.tz_localize(None)
    return df.sort_values("date").reset_index(drop=True)


def fetch_recent_games(
    api: str,
    user_agent: str,
    days: int = 45,
    league_prefixes: list[str] | None = None,
) -> pd.DataFrame:
    """Just the last few weeks. Keeps ratings current between runs."""
    now = datetime.now(timezone.utc)
    return fetch_games(api, user_agent, now - timedelta(days=days),
                       now + timedelta(days=1), league_prefixes, max_pages=8)


def _league_of(overview_page: str | None) -> str:
    """OverviewPage looks like 'LCK/2026 Season/Summer Season'."""
    if not overview_page:
        return "Unknown"
    return str(overview_page).split("/")[0].strip()


def _read_cache(path: str, minutes: int):
    if minutes <= 0 or not os.path.exists(path):
        return None
    if time.time() - os.path.getmtime(path) > minutes * 60:
        return None
    with open(path) as fh:
        return json.load(fh)


def _write_cache(path: str, payload: dict) -> None:
    with open(path, "w") as fh:
        json.dump(payload, fh)


# ---------------------------------------------------------------------
# Name matching
# ---------------------------------------------------------------------
# Leaguepedia and Oracle's Elixir do not always spell teams the same way.
# Unmatched teams are reported rather than silently rated at 1500, because
# a silent default looks exactly like a real 50/50 forecast.

ALIASES = {
    # "Leaguepedia name": "Oracle's Elixir name",
    "Bilibili Gaming": "Bilibili Gaming",
    "Dplus KIA": "Dplus KIA",
}


def resolve(name: str, known: set[str]) -> str | None:
    if name in known:
        return name
    alias = ALIASES.get(name)
    if alias and alias in known:
        return alias
    lowered = {k.lower(): k for k in known}
    if name.lower() in lowered:
        return lowered[name.lower()]
    return None
