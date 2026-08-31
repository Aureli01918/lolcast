"""Checks for the ledger, source matching and self-calibration.

These run offline. The point is to verify the parts that would fail
silently: a comparison that quietly flatters one source, a calibration
that fires on too little data, or a market price attached to the wrong
team.
"""

from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lolcast import ledger, selfcal  # noqa: E402
from lolcast.sources import (  # noqa: E402
    Quote, implied_from_decimal, match_pair, remove_margin, same_team,
)
from lolcast.sources.polymarket import parse_question  # noqa: E402

NOW = datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc)


def test_name_matching():
    assert same_team("Gen.G", "Gen G")
    assert same_team("T1", "T1 Esports")
    assert same_team("Top Esports", "TopEsports")
    assert not same_team("Gen.G", "T1")
    # Different teams that share a word must not collide.
    assert not same_team("Team Liquid", "Team Vitality")
    assert match_pair("Gen.G", "T1", "T1", "Gen G")
    print("name matching ok")


def test_question_parsing():
    cases = [
        ("LoL: Top Esports vs EDward Gaming (BO3) - LPL Group Ascend",
         ("Top Esports", "EDward Gaming")),
        ("LoL: Hanwha Life Esports Challengers vs BNK FearX Youth (BO3) - LCK CL",
         ("Hanwha Life Esports Challengers", "BNK FearX Youth")),
        ("Gen.G vs T1", ("Gen.G", "T1")),
    ]
    for question, expected in cases:
        got = parse_question(question)
        assert got == expected, f"{question!r} -> {got!r}, wanted {expected!r}"
    print("question parsing ok")


def test_odds_maths():
    assert abs(implied_from_decimal(2.5) - 0.4) < 1e-9
    # A bookmaker's two sides sum to more than 1; the margin must go.
    a, b = remove_margin(0.55, 0.52)
    assert abs(a + b - 1.0) < 1e-9
    assert a > b
    print(f"odds maths ok (0.55/0.52 -> {a:.4f}/{b:.4f})")


def test_quote_orientation():
    q = Quote("m1", quoted_team="T1", prob_quoted=0.7, captured=NOW)
    assert abs(q.prob_for("T1", "Gen.G") - 0.7) < 1e-9
    assert abs(q.prob_for("Gen.G", "T1") - 0.3) < 1e-9
    assert q.prob_for("Fnatic", "G2") is None      # refuses to guess
    print("quote orientation ok")


def _write_ledger(path, rows):
    ledger.append(path, rows)


def test_ledger_roundtrip_and_fairness():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ledger.csv")

        match = {"key": "M1", "kickoff": NOW + timedelta(hours=24),
                 "team1": "Gen.G", "team2": "T1", "event": "LCK", "best_of": 3}

        # Three snapshots of the same match at different lead times.
        for hours_before, mine in ((24, 0.60), (6, 0.62), (0.5, 0.95)):
            captured = match["kickoff"] - timedelta(hours=hours_before)
            _write_ledger(path, ledger.snapshot_rows(
                match, {"lolcast": mine, "polymarket": 0.55}, captured))

        df = ledger.load(path)
        assert len(df) == 6

        snaps = ledger.official_snapshots(df, min_lead_hours=1.0)
        assert len(snaps) == 2, snaps
        mine = snaps[snaps["source"] == "lolcast"]["prob_team1"].iloc[0]
        # Must pick the 6h quote, not the 0.5h one that saw late news.
        assert abs(mine - 0.62) < 1e-9, mine

        assert ledger.ungraded_keys(df, before=NOW + timedelta(days=2)) == ["M1"]
        filled = ledger.apply_results(path, {"M1": 1})
        assert filled == 6, filled
        assert ledger.load(path)["result"].notna().all()
        print("ledger snapshot, lead-time rule and grading ok")


def test_scoreboard_uses_common_subset():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ledger.csv")
        rng = np.random.default_rng(1)

        # Our model covers 40 matches. The market only covers the 20 most
        # lopsided ones, where it is nearly always right. If the scoreboard
        # compared raw accuracy it would look far better than it is.
        for i in range(40):
            lopsided = i < 20
            true_p = 0.9 if lopsided else 0.52
            won = int(rng.random() < true_p)
            kickoff = NOW - timedelta(days=40 - i)
            quotes = {"lolcast": true_p}
            if lopsided:
                quotes["polymarket"] = true_p
            _write_ledger(path, ledger.snapshot_rows(
                {"key": f"M{i}", "kickoff": kickoff, "team1": "A", "team2": "B",
                 "event": "E", "best_of": 3},
                quotes, kickoff - timedelta(hours=5)))
            ledger.apply_results(path, {f"M{i}": won})

        scores, common = ledger.scoreboard(ledger.load(path), min_lead_hours=1.0)
        by_name = {s.source: s for s in scores}
        assert by_name["polymarket"].coverage == 0.5, by_name["polymarket"].coverage
        assert by_name["lolcast"].coverage == 1.0
        # Both scored on the same 20 matches, so log loss must be equal.
        assert by_name["lolcast"].common == by_name["polymarket"].common == 20
        assert abs(by_name["lolcast"].log_loss - by_name["polymarket"].log_loss) < 1e-9
        print("scoreboard common-subset rule ok "
              f"(coverage {by_name['polymarket'].coverage:.0%} reported separately)")


def test_self_calibration():
    rng = np.random.default_rng(5)

    # An overconfident model: it says 0.9 when the truth is 0.75.
    true_p = rng.uniform(0.3, 0.9, 4000)
    stated = 1 / (1 + np.exp(-np.log(true_p / (1 - true_p)) * 1.6))  # stretched
    outcomes = (rng.random(4000) < true_p).astype(float)

    small = selfcal.fit(stated[:50], outcomes[:50])
    assert not small.active, "fired on 50 matches; guardrail failed"

    cal = selfcal.fit(stated, outcomes)
    assert cal.active, cal
    assert cal.temperature > 1.0, f"should soften, got {cal.temperature}"

    def ll(p):
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return -np.mean(outcomes * np.log(p) + (1 - outcomes) * np.log(1 - p))

    before, after = ll(stated), ll(cal.apply(stated))
    assert after < before, (before, after)

    # Ordering must survive: it changes confidence, not who is favoured.
    sample = np.array([0.2, 0.4, 0.5, 0.6, 0.85])
    adjusted = cal.apply(sample)
    assert np.all(np.diff(adjusted) > 0)
    assert abs(cal.apply(0.5) - 0.5) < 1e-9, "50/50 must stay 50/50"

    print(f"self-calibration ok: {cal}")
    print(f"  live log loss {before:.4f} -> {after:.4f}")


def test_calibration_declines_when_already_good():
    rng = np.random.default_rng(9)
    p = rng.uniform(0.2, 0.8, 3000)
    y = (rng.random(3000) < p).astype(float)
    cal = selfcal.fit(p, y)
    assert abs(cal.temperature - 1.0) < 0.12, cal
    print(f"well-calibrated input left nearly alone (temperature {cal.temperature})")


def test_scoreline_model():
    from lolcast.series import scoreline_distribution
    from lolcast.ratings import series_win_probability

    for bo in (1, 3, 5):
        d = scoreline_distribution(0.62, 0.55, bo)
        assert abs(d.total() - 1) < 1e-9
        # With no side gap it must reduce to the old independent-games maths.
        for p in (0.35, 0.5, 0.7):
            assert abs(scoreline_distribution(p, p, bo).series_win()
                       - series_win_probability(p, bo)) < 1e-9

    # The point of the whole exercise: a sweep is rarer than p squared,
    # because winning game one gives the opponent blue for game two.
    d = scoreline_distribution(0.70, 0.60, 3)
    naive = (0.65) ** 2
    assert d.sweep("a") < naive
    print(f"scoreline model ok: P(2-0)={d.sweep('a'):.4f} vs naive {naive:.4f}")


def test_sweep_grading():
    import tempfile
    from lolcast import ledger as L

    match = {"key": "M1", "kickoff": NOW + timedelta(hours=24), "team1": "A",
             "team2": "B", "event": "E", "best_of": 3}
    cap = match["kickoff"] - timedelta(hours=5)
    cases = [
        ({"team1_won": 1, "team1_score": 2, "team2_score": 0}, (1, 1, 0)),
        ({"team1_won": 1, "team1_score": 2, "team2_score": 1}, (1, 0, 0)),
        ({"team1_won": 0, "team1_score": 0, "team2_score": 2}, (0, 0, 1)),
    ]
    for outcome, (series_r, s1, s2) in cases:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "l.csv")
            for market in (L.SERIES, L.SWEEP_1, L.SWEEP_2):
                L.append(path, L.snapshot_rows(match, {"lolcast": 0.5}, cap,
                                               market=market))
            L.apply_results(path, {"M1": outcome})
            got = dict(zip(L.load(path)["market"], L.load(path)["result"]))
            assert (got[L.SERIES], got[L.SWEEP_1], got[L.SWEEP_2]) == (series_r, s1, s2), got
    print("sweep grading ok: judged on scoreline, not on who won")


def test_roster_audit():
    """The audit must find a planted effect and refuse to invent one."""
    from lolcast import rosters

    teams = [f"T{i}" for i in range(10)]
    pool = {t: [f"{t}_p{j}" for j in range(8)] for t in teams}

    def world(noise, seed, n_games=3600):
        rng = np.random.default_rng(seed)
        lineup = {t: set(pool[t][:5]) for t in teams}
        unstable = {t: -1 for t in teams}
        prows, detail = [], []
        day = pd.Timestamp("2025-01-01")
        for g in range(n_games):
            day += pd.Timedelta(hours=8)
            a, b = rng.choice(teams, 2, replace=False)
            for t in (a, b):
                if rng.random() < 0.06:
                    out = rng.choice(sorted(lineup[t]))
                    inn = rng.choice([q for q in pool[t] if q not in lineup[t]])
                    lineup[t] = (lineup[t] - {out}) | {inn}
                    unstable[t] = g + 3
                for q in sorted(lineup[t]):
                    prows.append({"player": q, "team": t, "gameid": f"g{g}",
                                  "date": day, "event": "LCK/2025", "role": None})
            true_p = float(rng.uniform(0.3, 0.8))
            hit = g <= unstable[a] or g <= unstable[b]
            ap = float(np.clip(true_p + (rng.normal(0, noise) if hit else 0),
                               0.02, 0.98))
            detail.append({"date": day, "blue": a, "red": b,
                           "prediction": true_p, "actual": int(rng.random() < ap)})
        return pd.DataFrame(prows), pd.DataFrame(detail)

    def run(noise, seed):
        prows, detail = world(noise, seed)
        lineups = rosters.lineups_from_rows(prows)
        annotated = rosters.annotate_changes(lineups)
        report = rosters.churn_report(lineups, annotated)
        flagged = rosters.flag_recent_change(detail, annotated, 3)
        return report, rosters.compare_groups(flagged)

    # A large planted disruption: the audit resolves roughly 0.025 log
    # loss and up at these sample sizes, so the test uses a clear one.
    report, big = run(1.2, 1)
    assert 0.03 < report.change_rate < 0.12, report.change_rate
    assert big["significant"] and big["difference"] > 0, big

    # No planted effect: must not be called significant.
    _, none = run(0.0, 2)
    assert not none["significant"], none
    print(f"roster audit ok: detects a real effect ({big['difference']:+.4f}), "
          f"declines a fake one ({none['difference']:+.4f}, interval spans zero)")


def test_lineup_parsing():
    """Incomplete lineups must be dropped, not read as substitutions."""
    from lolcast import rosters

    rows = pd.DataFrame([
        *[{"player": f"p{i}", "team": "A", "gameid": "g1",
           "date": pd.Timestamp("2026-01-01")} for i in range(5)],
        # Only four players recorded: a wiki gap, not a roster change.
        *[{"player": f"p{i}", "team": "A", "gameid": "g2",
           "date": pd.Timestamp("2026-01-02")} for i in range(4)],
        *[{"player": f"p{i}", "team": "A", "gameid": "g3",
           "date": pd.Timestamp("2026-01-03")} for i in range(5)],
    ])
    lineups = rosters.lineups_from_rows(rows)
    assert list(lineups["size"]) == [5, 4, 5]
    annotated = rosters.annotate_changes(lineups)
    assert len(annotated) == 2, "incomplete game was not dropped"
    assert not annotated["changed"].any(), "gap was misread as a substitution"
    print("lineup parsing ok: incomplete records dropped, not counted as changes")


def test_live_calibration_and_divergence():
    """Divergence must stay quiet on thin data and speak up on real decay."""
    import tempfile
    from lolcast import ledger as L

    def build(path, n, noise, seed=0):
        rng = np.random.default_rng(seed)
        for i in range(n):
            kick = NOW - timedelta(days=n - i)
            true_p = float(rng.uniform(0.25, 0.85))
            said = float(np.clip(true_p + rng.normal(0, noise), 0.02, 0.98))
            L.append(path, L.snapshot_rows(
                {"key": f"M{i}", "kickoff": kick, "team1": "A", "team2": "B",
                 "event": "E", "best_of": 3},
                {"lolcast": said}, kick - timedelta(hours=5)))
            L.apply_results(path, {f"M{i}": {
                "team1_won": int(rng.random() < true_p),
                "team1_score": 2, "team2_score": 1}})

    expected = {40: "waiting", 500: "consistent"}
    for n, status in expected.items():
        with tempfile.TemporaryDirectory() as t:
            path = os.path.join(t, "l.csv")
            build(path, n, 0.05)
            d = L.divergence(L.load(path), "lolcast", backtest_log_loss=0.60)
            assert d["status"] == status, (n, d)

    with tempfile.TemporaryDirectory() as t:
        path = os.path.join(t, "l.csv")
        build(path, 500, 0.30)
        book = L.load(path)
        d = L.divergence(book, "lolcast", backtest_log_loss=0.60)
        assert d["status"] == "worse" and d["gap"] > 0.03, d
        rows = L.live_calibration(book, "lolcast", bins=5)
        assert rows and all(0 <= r["actual"] <= 1 for r in rows)
        assert sum(r["games"] for r in rows) == 500
    print("live calibration ok: silent under 100 matches, flags real decay")


def test_finished_report():
    """Finished matches must report the scoreline, not just the winner."""
    import tempfile
    from lolcast import ledger as L

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "l.csv")
        cases = [("M0", 2, 0, 1), ("M1", 1, 2, 0), ("M2", 3, 2, 1)]
        for i, (key, s1, s2, won) in enumerate(cases):
            kick = NOW - timedelta(days=i + 1)
            L.append(path, L.snapshot_rows(
                {"key": key, "kickoff": kick, "team1": "Gen.G", "team2": "T1",
                 "event": "LCK", "best_of": 3},
                {"lolcast": 0.64}, kick - timedelta(hours=5)))
            L.apply_results(path, {key: {"team1_won": won, "team1_score": s1,
                                         "team2_score": s2}})

        rows = L.finished(L.load(path), "lolcast", days=365)
        assert len(rows) == 3, rows
        by_id = {r["id"]: r for r in rows}
        assert (by_id["M0"]["team1Score"], by_id["M0"]["team2Score"]) == (2, 0)
        assert by_id["M1"]["team1Won"] == 0
        # Newest first, so a followed match reads as a feed.
        assert [r["id"] for r in rows] == ["M0", "M1", "M2"]
        assert all(r["forecast"] == 0.64 for r in rows)

    # A win recorded without a scoreline must not invent one.
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "l.csv")
        kick = NOW - timedelta(days=1)
        L.append(path, L.snapshot_rows(
            {"key": "X", "kickoff": kick, "team1": "A", "team2": "B",
             "event": "E", "best_of": 3}, {"lolcast": 0.5},
            kick - timedelta(hours=5)))
        L.apply_results(path, {"X": 1})
        row = L.finished(L.load(path), "lolcast", days=365)[0]
        assert row["team1Score"] is None and row["team1Won"] == 1, row
    print("finished report ok: scorelines kept, never fabricated")


if __name__ == "__main__":
    test_name_matching()
    test_question_parsing()
    test_odds_maths()
    test_quote_orientation()
    test_ledger_roundtrip_and_fairness()
    test_scoreboard_uses_common_subset()
    test_self_calibration()
    test_calibration_declines_when_already_good()
    test_scoreline_model()
    test_sweep_grading()
    test_lineup_parsing()
    test_roster_audit()
    test_live_calibration_and_divergence()
    test_finished_report()
    print("\nAll checks passed.")
