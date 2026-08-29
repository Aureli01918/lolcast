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
    print("\nAll checks passed.")
