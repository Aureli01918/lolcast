"""Command line interface.

    python -m lolcast fetch-history     download Oracle's Elixir CSVs
    python -m lolcast backtest          score the model on held-out history
    python -m lolcast ablation          measure what each feature is worth
    python -m lolcast grade             fill in results for finished matches
    python -m lolcast predict           forecast, snapshot, write the dashboard
    python -m lolcast scoreboard        how everyone is actually doing
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml

from . import ingest_oracle, ingest_schedule, ledger, selfcal, series, sources
from .model import calibration_table, make_estimator, score, walk_forward
from .pipeline import build_features, upcoming_row
from .ratings import series_win_probability

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Our own entries in the ledger. "elo" is the ratings-only baseline: free
# to compute, and it puts "are my extra features earning their keep" in
# the same table as every external source.
SELF = "lolcast"
BASELINE = "elo"


def load_config(path: str = "config.yaml") -> dict:
    with open(os.path.join(ROOT, path)) as fh:
        return yaml.safe_load(fh)


def _path(*parts: str) -> str:
    return os.path.join(ROOT, *parts)


def ledger_path(cfg: dict) -> str:
    return _path(cfg.get("ledger", {}).get("path", "data/ledger.csv"))


def load_games(cfg: dict) -> pd.DataFrame:
    """Historical CSVs, topped up with recent games from Leaguepedia.

    The CSVs cover years but come from a source that fails often on CI.
    The top-up covers the last few weeks from the same place the schedule
    comes from, so ratings stay current even when a CSV refresh fails.
    """
    frames = []

    # Leaguepedia-built history: the primary source. Small enough to live
    # in the repo, and from the one place that has never rate-limited us.
    path = history_path(cfg)
    if os.path.exists(path):
        built = pd.read_csv(path)
        built["date"] = pd.to_datetime(built["date"], errors="coerce")
        frames.append(built.dropna(subset=["date"]))
        print(f"History: {len(frames[-1]):,} games from {os.path.basename(path)}")

    # Oracle's Elixir CSVs, if any are present. Optional, and only needed
    # for the stat-based features that are off by default.
    try:
        csvs = ingest_oracle.load(_path(cfg["data"]["raw_dir"]),
                                  cfg["data"]["years"], cfg["leagues"]["rate"])
        frames.append(csvs)
        print(f"History: {len(csvs):,} games from Oracle's Elixir CSVs")
    except FileNotFoundError:
        pass

    if not frames:
        raise FileNotFoundError(
            "No match history. Run `python -m lolcast bootstrap` to build it "
            "from Leaguepedia."
        )
    games = _merge_games(frames)

    days = int(cfg["data"].get("topup_days", 0) or 0)
    if days <= 0:
        return games

    try:
        recent = ingest_schedule.fetch_recent_games(
            cfg["data"]["leaguepedia_api"], cfg["data"]["user_agent"], days,
            cfg["leagues"].get("leaguepedia_prefixes"),
        )
    except Exception as exc:                                   # noqa: BLE001
        print(f"Top-up skipped ({type(exc).__name__}: {exc})")
        return games

    if recent.empty:
        return games

    # No league filter here: the query already restricted to the configured
    # Leaguepedia prefixes, and those page names ("Worlds", "LTA North")
    # differ from Oracle's codes ("WLDs", "LTA N"), so filtering against the
    # Oracle list would silently discard international and Americas games.
    recent = _align_team_names(recent, games)
    home = ingest_oracle.team_regions(games) if not games.empty else {}
    recent["blue_region"] = recent["blue"].map(home).fillna("INTL")
    recent["red_region"] = recent["red"].map(home).fillna("INTL")

    combined = _merge_games([games, recent])
    print(f"Top-up: {len(recent)} recent games, {len(combined) - len(games)} new")
    return combined


def _merge_games(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate sources and drop repeats.

    A game counted twice moves both teams' ratings by a double step and
    quietly inflates confidence, so this matters more than it looks.
    Matching is on the day plus the team pair, because the two sources
    give different game ids for the same game.
    """
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame()
    prepared = []
    for rank, frame in enumerate(frames):
        frame = frame.copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.dropna(subset=["date", "blue", "red", "blue_win"])
        frame["_day"] = frame["date"].dt.floor("D")
        frame["_pair"] = [
            "|".join(sorted((str(b), str(r))))
            for b, r in zip(frame["blue"], frame["red"])
        ]
        # Number each game *within its own source*. A Bo3 is three real
        # games between the same pair on the same day, so they must all
        # survive -- but the same three arriving from a second source must
        # not add three more.
        frame["_n"] = frame.sort_values("date").groupby(["_day", "_pair"]).cumcount()
        frame["_src"] = rank
        prepared.append(frame)

    combined = pd.concat(prepared, ignore_index=True).sort_values(["_src", "date"])
    combined = combined.drop_duplicates(subset=["_day", "_pair", "_n"], keep="first")
    combined = combined.drop(columns="_src")
    return (combined.drop(columns=["_day", "_pair", "_n"])
            .sort_values("date").reset_index(drop=True))


def _align_team_names(recent: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Map Leaguepedia spellings onto the names used in the CSVs.

    Without this a team can end up with two separate ratings -- one under
    each spelling -- which looks like nothing is wrong but halves the
    history behind every forecast it appears in.
    """
    if games.empty:
        return recent
    known = set(games["blue"]).union(games["red"])
    cache: dict[str, str] = {}

    def resolve(name: str) -> str:
        if name in cache:
            return cache[name]
        match = name if name in known else next(
            (k for k in known if sources.same_team(name, k)), name
        )
        cache[name] = match
        return match

    recent = recent.copy()
    recent["blue"] = recent["blue"].map(resolve)
    recent["red"] = recent["red"].map(resolve)
    return recent


def _built(cfg: dict, games: pd.DataFrame):
    return build_features(games, cfg["features"]["enabled"], cfg["ratings"],
                          ingest_oracle.STAT_COLUMNS)


# -- commands ---------------------------------------------------------


def cmd_fetch_history(cfg: dict, args) -> None:
    ingest_oracle.download(cfg["data"]["oracle_drive_folder"],
                           _path(cfg["data"]["raw_dir"]),
                           cfg["data"]["years"])


def history_path(cfg: dict) -> str:
    return _path(cfg["data"].get("history_path", "data/history.csv"))


def cmd_bootstrap(cfg: dict, args) -> None:
    """Build the historical game table from Leaguepedia, once.

    Slow on purpose: the wiki rate-limits anonymous callers to roughly one
    request a minute, so this walks year by year and sleeps between pages.
    Expect the better part of an hour. It only ever needs to run once --
    afterwards the file is committed and the daily top-up keeps it current.
    """
    # Building five years takes roughly sixty requests. Anonymous callers
    # get about one a minute *per IP*, and CI runners share IPs, so this
    # reliably fails part-way through. Better to say so in five seconds
    # than after fifteen minutes of backing off.
    if not ingest_schedule.logged_in(cfg["data"]["user_agent"]):
        raise SystemExit(
            "Bootstrap needs a Leaguepedia login.\n\n"
            "  1. Create a free account at lol.fandom.com\n"
            "  2. Go to lol.fandom.com/wiki/Special:BotPasswords and make a\n"
            "     credential with 'Basic rights' and 'High-volume editing'\n"
            "  3. In this repository: Settings -> Secrets and variables ->\n"
            "     Actions -> New repository secret. Add WIKI_USERNAME (the\n"
            "     full name@label form) and WIKI_PASSWORD.\n\n"
            "Then re-run this workflow. Secrets added while a run is in\n"
            "progress do not apply to that run."
        )

    out = history_path(cfg)
    existing = pd.read_csv(out) if os.path.exists(out) else pd.DataFrame()
    if not existing.empty and not args.force:
        print(f"{out} already has {len(existing):,} games. Use --force to rebuild.")
        return

    prefixes = cfg["leagues"].get("leaguepedia_prefixes")
    years = cfg["data"]["years"]
    frames = []
    for year in years:
        print(f"Year {year}")
        frames.append(ingest_schedule.fetch_games(
            cfg["data"]["leaguepedia_api"], cfg["data"]["user_agent"],
            datetime(year, 1, 1, tzinfo=timezone.utc),
            datetime(year + 1, 1, 1, tzinfo=timezone.utc),
            prefixes,
        ))
        print(f"  {len(frames[-1]):,} games")

    games = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if games.empty:
        raise RuntimeError("Leaguepedia returned no games. Check league prefixes.")

    games = games.drop_duplicates(subset=["gameid"]).sort_values("date")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    games.to_csv(out, index=False)
    size = os.path.getsize(out) / 1e6
    print(f"\nWrote {len(games):,} games to {out} ({size:.1f} MB)")
    print(f"Leagues: {sorted(games['league'].unique())}")


def cmd_backtest(cfg: dict, args) -> None:
    games = load_games(cfg)
    print(f"Loaded {len(games):,} games, {games['date'].min():%Y-%m-%d} to "
          f"{games['date'].max():%Y-%m-%d}")

    X, y, index, _ = _built(cfg, games)
    result, detail = walk_forward(X, y, index, cfg["backtest"], cfg["model"])
    elo_only = score(detail["actual"].to_numpy(), detail["elo_prob"].to_numpy())

    print(f"\nElo only  : {elo_only}")
    print(f"Full model: {result}")
    print("\nCalibration")
    print(calibration_table(detail["actual"].to_numpy(),
                            detail["prediction"].to_numpy()).to_string(index=False))

    if args.by_league:
        print("\nBy league")
        for league, group in detail.groupby("league"):
            if len(group) < 100:
                continue
            print(f"  {league:<8} "
                  f"{score(group['actual'].to_numpy(), group['prediction'].to_numpy())}")


def cmd_ablation(cfg: dict, args) -> None:
    from .model import ablation

    table = ablation(load_games(cfg), cfg["features"]["enabled"], cfg["ratings"],
                     cfg["backtest"], cfg["model"], ingest_oracle.STAT_COLUMNS)
    print("\nLog loss when each feature is removed.")
    print("A positive delta means the feature was pulling its weight.\n")
    print(table.to_string(index=False))


def cmd_grade(cfg: dict, args) -> None:
    path = ledger_path(cfg)
    pending = ledger.ungraded_keys(ledger.load(path))
    if not pending:
        print("Nothing to grade.")
        return

    print(f"{len(pending)} finished match(es) awaiting a result.")
    results = ingest_schedule.fetch_results(
        cfg["data"]["leaguepedia_api"], cfg["data"]["user_agent"], pending
    )
    filled = ledger.apply_results(path, results)
    print(f"Graded {len(results)} match(es), {filled} ledger row(s) updated.")

    still_open = set(pending) - set(results)
    if still_open:
        print(f"{len(still_open)} still unresolved (postponed, cancelled, or "
              f"result not yet published).")


def cmd_scoreboard(cfg: dict, args) -> None:
    cfg_l = cfg.get("ledger", {})
    book = ledger.load(ledger_path(cfg))
    scores, _ = ledger.scoreboard(book, cfg_l.get("min_lead_hours", 1.0), args.days)
    if not scores:
        print("No graded matches yet. Run `predict` regularly, then `grade` "
              "once matches have finished.")
        return

    window = f"last {args.days} days" if args.days else "all time"
    print(f"\nLive accuracy ({window}), scored only on matches every source "
          f"covered:\n")
    for s in scores:
        print(f"  {s}")
    print(f"\n{selfcal.fit_from_ledger(book, SELF, cfg.get('self_calibration', {}))}")


def cmd_predict(cfg: dict, args) -> None:
    now = datetime.now(timezone.utc)
    cfg_l = cfg.get("ledger", {})

    games = load_games(cfg)
    X, y, index, ratings = _built(cfg, games)
    estimator = make_estimator(cfg["model"])
    estimator.fit(X, y)

    # Confidence correction learned from our own live track record.
    cal = selfcal.fit_from_ledger(ledger.load(ledger_path(cfg)), SELF,
                                  cfg.get("self_calibration", {}))
    print(cal)

    # If the schedule source is down, keep whatever the dashboard is
    # already showing rather than replacing it with an empty page. A
    # stale forecast is more useful than no forecast.
    out_path = _path("docs", "data.json")
    try:
        schedule = ingest_schedule.fetch(
            cfg["data"]["leaguepedia_api"], cfg["data"]["user_agent"],
            cfg["dashboard"]["horizon_days"], _path(cfg["data"]["cache_dir"]),
            cfg["data"]["schedule_cache_minutes"],
        )
    except Exception as exc:                                   # noqa: BLE001
        print(f"Could not reach Leaguepedia: {exc}")
        if os.path.exists(out_path):
            print("Keeping the existing dashboard. Nothing was overwritten.")
            return
        raise

    known = set(ratings.teams)
    enabled = cfg["features"]["enabled"]
    resolved, unmatched = [], set()

    for row in schedule.itertuples(index=False):
        team1 = ingest_schedule.resolve(row.blue, known)
        team2 = ingest_schedule.resolve(row.red, known)
        if team1 is None or team2 is None:
            unmatched.update(n for n, r in ((row.blue, team1), (row.red, team2))
                             if r is None)
            continue
        resolved.append({
            "key": str(row.match_id), "kickoff": row.date,
            "team1": team1, "team2": team2,
            "display1": row.blue, "display2": row.red,
            "event": row.event, "round": row.round,
            "best_of": int(row.best_of),
        })

    # -- ask every configured source what it thinks ------------------
    specs = sources.enabled_sources(cfg)
    quotes: dict[str, dict[str, float]] = {}
    liquidity: dict[str, dict[str, float]] = {}
    source_errors: dict[str, str] = {}
    lookup = {m["key"]: m for m in resolved}

    for spec in specs:
        try:
            got = spec.fetch(resolved, cfg)
        except Exception as exc:                       # noqa: BLE001
            # One flaky source must never stop the forecast going out.
            source_errors[spec.name] = f"{type(exc).__name__}: {exc}"
            print(f"  {spec.name}: unavailable ({exc})")
            continue

        by_match, liq = {}, {}
        for quote in got:
            match = lookup.get(quote.match_key)
            if match is None:
                continue
            p = quote.prob_for(match["team1"], match["team2"])
            if p is None:
                continue
            by_match[quote.match_key] = p
            if quote.liquidity is not None:
                liq[quote.match_key] = quote.liquidity
        quotes[spec.name] = by_match
        liquidity[spec.name] = liq
        print(f"  {spec.name}: priced {len(by_match)} of {len(resolved)} matches")

    # -- forecast, snapshot, assemble --------------------------------
    matches, ledger_rows = [], []

    # How reliably the previous game's loser ends up on blue, measured from
    # this history rather than assumed from the rulebook.
    side_rate, side_n = series.fit_side_choice(games)
    repeat_rate, repeat_n = series.fit_game_correlation(games)
    print(f"Side choice: loser takes blue {side_rate:.1%} of the time "
          f"({side_n:,} game pairs)")
    print(f"Game repeat: previous winner wins the next {repeat_rate:.1%} "
          f"({repeat_n:,} pairs; 50% would mean no momentum)")

    for match in resolved:
        # Ask the model twice, once with each team on blue. The side is not
        # known in advance, and the difference between the two is exactly
        # what makes a sweep less likely than squaring one number.
        f_blue = upcoming_row(ratings, enabled, match["team1"], match["team2"],
                              match["kickoff"], best_of=match["best_of"])
        f_red = upcoming_row(ratings, enabled, match["team2"], match["team1"],
                             match["kickoff"], best_of=match["best_of"])
        p_blue = float(estimator.predict_proba(np.array([f_blue]))[0, 1])
        p_red = 1.0 - float(estimator.predict_proba(np.array([f_red]))[0, 1])
        p_game = 0.5 * (p_blue + p_red)

        lines = series.scoreline_distribution(
            cal.apply(p_blue), cal.apply(p_red), match["best_of"],
            side_choice_rate=side_rate,
        )
        p_series = lines.series_win()
        p_raw = series.scoreline_distribution(
            p_blue, p_red, match["best_of"], side_choice_rate=side_rate
        ).series_win()

        p_elo_blue = ratings.expected(match["team1"], match["team2"])
        p_elo_red = 1.0 - ratings.expected(match["team2"], match["team1"])
        p_elo_game = 0.5 * (p_elo_blue + p_elo_red)
        elo_lines = series.scoreline_distribution(
            p_elo_blue, p_elo_red, match["best_of"], side_choice_rate=side_rate
        )
        p_elo = elo_lines.series_win()

        others = {spec.name: quotes.get(spec.name, {}).get(match["key"])
                  for spec in specs}

        state1, state2 = ratings.get(match["team1"]), ratings.get(match["team2"])
        matches.append({
            "id": match["key"],
            "date": match["kickoff"].isoformat(),
            "event": match["event"], "round": match["round"],
            "bestOf": match["best_of"],
            "blue": {"name": match["display1"], "elo": round(state1.elo),
                     "form": round(state1.form(10), 2), "games": state1.games},
            "red": {"name": match["display2"], "elo": round(state2.elo),
                    "form": round(state2.form(10), 2), "games": state2.games},
            "gameProb": round(p_game, 4),
            "gameProbBlue": round(p_blue, 4),
            "gameProbRed": round(p_red, 4),
            "seriesProb": round(p_series, 4),
            "seriesProbUncalibrated": round(p_raw, 4),
            "eloProb": round(p_elo, 4),
            "eloGameProb": round(p_elo_game, 4),
            "scorelines": {k: round(v, 4) for k, v in lines.as_dict().items()},
            "sweep": {"team1": round(lines.sweep("a"), 4),
                      "team2": round(lines.sweep("b"), 4)},
            "sources": {k: (round(v, 4) if v is not None else None)
                        for k, v in others.items()},
            "confidence": _confidence(state1.games, state2.games),
        })

        row_quotes = {SELF: p_series, BASELINE: p_elo}
        row_quotes.update({k: v for k, v in others.items() if v is not None})
        ledger_rows.extend(ledger.snapshot_rows(
            match, row_quotes, now,
            liquidity={k: liquidity.get(k, {}).get(match["key"]) for k in liquidity},
        ))

        # Sweep forecasts are recorded separately. Even with perfect
        # game-level calibration these can be systematically wrong, because
        # they depend on how games within a series correlate -- and that
        # error is invisible in series accuracy, where the winner is still
        # called correctly.
        if match["best_of"] > 1:
            for market, prob, elo_prob in (
                (ledger.SWEEP_1, lines.sweep("a"), elo_lines.sweep("a")),
                (ledger.SWEEP_2, lines.sweep("b"), elo_lines.sweep("b")),
            ):
                ledger_rows.extend(ledger.snapshot_rows(
                    match, {SELF: prob, BASELINE: elo_prob}, now, market=market,
                ))

    written = ledger.append(ledger_path(cfg), ledger_rows)

    # -- scores for the dashboard ------------------------------------
    result, detail = walk_forward(X, y, index, cfg["backtest"], cfg["model"])
    book = ledger.load(ledger_path(cfg))
    live_scores, _ = ledger.scoreboard(book, cfg_l.get("min_lead_hours", 1.0),
                                       cfg_l.get("scoreboard_days"))
    sweep_scores, _ = ledger.scoreboard(book, cfg_l.get("min_lead_hours", 1.0),
                                        cfg_l.get("scoreboard_days"),
                                        market=ledger.SWEEP_1)

    output = {
        "generated": now.isoformat(),
        "matches": matches,
        "unmatchedTeams": sorted(unmatched),
        "sources": [{"name": s.name, "label": s.label, "colour": s.colour,
                     "error": source_errors.get(s.name)} for s in specs],
        "selfLabels": {"model": SELF, "baseline": BASELINE},
        "calibration": {"active": cal.active, "temperature": cal.temperature,
                        "matches": cal.matches, "reason": cal.reason,
                        "text": str(cal)},
        "live": [{"source": s.source, "logLoss": s.log_loss, "brier": s.brier,
                  "accuracy": s.accuracy, "matches": s.common,
                  "coverage": round(s.coverage, 3)} for s in live_scores],
        "sweepLive": [{"source": s.source, "logLoss": s.log_loss,
                       "brier": s.brier, "matches": s.common}
                      for s in sweep_scores],
        "series": {"sideChoiceRate": round(side_rate, 4), "sidePairs": side_n,
                   "repeatRate": round(repeat_rate, 4), "repeatPairs": repeat_n},
        "model": {
            "features": enabled, "type": cfg["model"]["type"],
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
        "refresh": cfg.get("dashboard", {}).get("refresh", {}),
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(output, fh, indent=2)

    print(f"\nWrote {len(matches)} matches to {out_path}")
    print(f"Appended {written} row(s) to the ledger")
    if unmatched:
        print("Teams with no rating (add them to ALIASES in ingest_schedule.py):")
        for name in sorted(unmatched):
            print(f"  {name}")


def _confidence(games1: int, games2: int) -> str:
    fewest = min(games1, games2)
    if fewest < 10:
        return "low"
    if fewest < 40:
        return "medium"
    return "high"


COMMANDS = {
    "bootstrap": cmd_bootstrap,
    "fetch-history": cmd_fetch_history,
    "backtest": cmd_backtest,
    "ablation": cmd_ablation,
    "grade": cmd_grade,
    "predict": cmd_predict,
    "scoreboard": cmd_scoreboard,
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="lolcast")
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--by-league", action="store_true",
                        help="backtest only: break scores down per league")
    parser.add_argument("--days", type=int, default=None,
                        help="scoreboard only: limit to the last N days")
    parser.add_argument("--force", action="store_true",
                        help="bootstrap only: rebuild even if history exists")
    args = parser.parse_args(argv)
    COMMANDS[args.command](load_config(args.config), args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
