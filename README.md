# lolcast

Upcoming pro League of Legends matches with a forecast probability, built
from historical results and designed so you can keep adding variables to it.

Runs as a daily GitHub Action that writes a JSON file; a static page reads
that file. No server, no database, works the same on a phone and a laptop.

---

## What it does

| Layer | File | What lives here |
|---|---|---|
| History | `lolcast/ingest_oracle.py` | Oracle's Elixir CSVs, 2014→now, one row per game |
| Schedule | `lolcast/ingest_schedule.py` | Leaguepedia Cargo API: upcoming matches and results |
| Ratings | `lolcast/ratings.py` | Elo with split regression and a fitted side advantage |
| **Variables** | **`lolcast/features.py`** | **The part you extend** |
| Forecast | `lolcast/model.py` | Calibrated classifier + walk-forward backtest |
| **Sources** | **`lolcast/sources/`** | **Other forecasters to compare against** |
| Track record | `lolcast/ledger.py` | Every forecast, written down before the match |
| Self-correction | `lolcast/selfcal.py` | Confidence adjusted from the live record |
| Display | `docs/index.html` | Static dashboard |

## Setup

```bash
git clone <your-repo> && cd lolcast
pip install -r requirements.txt

python -m lolcast fetch-history   # ~200MB of CSVs, once
python -m lolcast backtest        # how good is it, on held-out history
python -m lolcast predict         # forecast + snapshot + docs/data.json
python -m lolcast grade           # after matches finish, record who won
python -m lolcast scoreboard      # you vs the market, on real forecasts
```

Open `docs/index.html` in a browser. A sample `data.json` is committed so
the page renders before your first real run.

To put it on your phone: push to GitHub, then Settings → Pages → Source:
GitHub Actions. The workflow in `.github/workflows/update.yml` refreshes it
every morning. Add the resulting URL to your home screen.

## How it keeps itself honest

Two different things are often confused here, and this does both.

**Relearning.** Every run rebuilds ratings from the full history including
yesterday's results, then refits. That happens automatically and always
has.

**Self-correction.** Separately, it checks whether its confidence matches
reality. If matches it calls at 70% only win 58%, the fix isn't to change
the ratings — it's to squash the whole scale toward the middle. That
correction is fitted on the ledger, not the backtest.

Three guardrails stop it chasing noise:

- Does nothing below 300 graded matches. A bad fortnight is not a signal.
- Blends toward "change nothing", weighted by sample size.
- Fits one number on the log-odds scale, so it cannot reorder matches. A
  team the model favoured is still favoured; only the confidence moves.

It also refuses to apply a correction that doesn't improve the live score.

## The ledger

`data/ledger.csv` is the only honest record of performance. The backtest
is a re-enactment of the past, which is an easier test than predicting
Thursday's game on Wednesday.

Every run appends one row per match per source, **before** kickoff.
Nothing is ever overwritten except filling in the result. The workflow
commits it back to the repo, so it survives.

Two rules make the comparison fair, and both matter more than they sound:

- **Same moment.** Market prices move as lineups and news land. Scoring a
  market's five-minutes-before price against your three-days-before price
  would flatter the market badly. Everyone is scored on their last quote
  taken at least `min_lead_hours` before kickoff. Later quotes are
  discarded, not used as a fallback.
- **Same matches.** A source that only prices marquee games would look
  brilliant on accuracy alone. The headline table scores everyone on the
  subset all of them covered, and reports coverage as its own column.

## Comparison sources

Configured under `sources.enabled` in `config.yaml`.

| Source | Cost | Notes |
|---|---|---|
| `polymarket` | free | Public API, no key. Real money behind the prices. Covers LCK, LPL, LEC, LCS, LCP and more. |
| `elo` | free | Your own ratings without the extra features. Always on. |
| `bookmaker` | paid | Template only. Fill in an endpoint and add a key. |

Adding one is a file in `lolcast/sources/`:

```python
@source("my_site", label="My Site", colour="#8899FF")
def fetch(matches, config):
    return [Quote(match_key=m["key"], quoted_team="Gen.G",
                  prob_quoted=0.61, captured=now) for m in matches]
```

It then appears in the ledger, the scoreboard, and on every bar. One rule:
say which team you're quoting. `Quote` re-orients from there, and refuses
to guess if the name doesn't match either side — a silently inverted
probability is the worst possible bug here, because everything still looks
fine.

A source that throws is logged and skipped. A flaky website must never
stop the forecast going out.

## The Update button

The dashboard is a static page, so opening it cannot recalculate anything
— it shows whatever was last built. There's a daily scheduled run, plus
two ways to force a refresh:

- **No setup:** put your Actions URL in `dashboard.refresh.actionsUrl`.
  The button opens that page and you press "Run workflow". Two taps.
- **Real button:** deploy `worker/update-proxy.js` to Cloudflare (free)
  and put its URL in `dashboard.refresh.proxyUrl`. The button then starts
  the rebuild and waits for it. Setup is in the file's header comment.

The proxy exists because triggering GitHub Actions needs an access token,
and a token in a public web page is a token anyone can use. The worker
holds it instead.

## Adding a variable

This is the part the whole design is arranged around. Write a function:

```python
@feature("bot_lane_synergy")
def bot_lane_synergy(ctx):
    """Games this ADC and support have played together."""
    return ctx.meta.get("blue_duo_games", 0) - ctx.meta.get("red_duo_games", 0)
```

List it under `features.enabled` in `config.yaml`, then find out whether it
was worth it:

```bash
python -m lolcast ablation
```

That drops each feature in turn and reports how much log loss gets worse
without it. A feature with a delta near zero is not earning its place — it
is adding variance and slowing you down.

**Features may only read the past.** A `FeatureContext` deliberately does
not expose the result of the game being predicted, and the pipeline
computes features before it updates ratings. That is what makes the
backtest number trustworthy. If you find yourself reaching around the
context to look up a result, stop — that is leakage, and it produces
models that look excellent and predict nothing.

## Reading the numbers

**Accuracy is the least useful metric here** and it is the one everybody
quotes. A model that says "51% Gen.G" and one that says "93% Gen.G" score
identical accuracy when Gen.G wins. The backtest leads with log loss and
Brier score instead, and prints a calibration table so you can check
whether games you called at 70% actually happen about 70% of the time.

`skill` is the share of log loss removed compared to predicting 50/50 on
everything. That is the honest headline.

Realistically: ratings alone get you most of the way. Public work on pro
LoL tends to land somewhere in the 60s for accuracy, and betting markets on
major-region matches are hard to beat. Expect your added variables to move
log loss by small amounts, and expect several of them to move it the wrong
way. The ablation command exists so you find that out rather than assume.

## What to expect against the market

You will probably lose to Polymarket on LCK and LPL. Deep markets on major
regions are genuinely hard to beat, and anyone selling you otherwise is
selling something. You may well beat it on smaller leagues where few
people are trading.

That is the useful part. "Did log loss improve a bit" is a weak test for a
new variable. "Did the gap to the market close" is a sharp one, and it
tells you which leagues your model is worth anything in.

Give it a few weeks before reading anything into the live table. Under a
hundred graded matches, the differences are mostly noise.

## Known rough edges

- **Team name matching.** Leaguepedia and Oracle's Elixir spell some teams
  differently. Unmatched teams are skipped and reported rather than
  silently rated at 1500, since a default rating looks exactly like a
  genuine 50/50 call. Add fixes to `ALIASES` in `ingest_schedule.py`.
- **Roster changes.** Ratings track teams, not players. A team that swaps
  three starters keeps its rating. `split_regression` in `config.yaml`
  blunts this across splits but not mid-split.
- **Leaguepedia rate limits.** Roughly one request per minute
  unauthenticated. Responses are cached for an hour. For heavier use,
  register an account and switch to `mwrogue`.
- **Series independence.** Bo3/Bo5 probabilities assume games in a series
  are independent, which slightly understates the favourite.
- **Market coverage is uneven.** Polymarket prices marquee matches
  heavily and minor-league games thinly. The `min_liquidity` setting drops
  markets with almost no money in them, since scoring against noise
  flatters you.
- **Name matching across sources.** Handled for spacing and suffixes
  ("Gen.G" / "Gen G" / "GenG", "Top Esports" / "TopEsports"), but a
  genuinely different name is dropped rather than guessed at. Wrong
  matches corrupt the record invisibly; missing ones are just missing.
- **2026 draft data.** Oracle's Elixir has flagged that champion select
  data for some 2026 games is incorrect pending a fix, so treat any
  draft-based feature you add with suspicion for that period.

## Data sources

- [Oracle's Elixir](https://oracleselixir.com/tools/downloads) — free for
  non-commercial use. Verify the current Drive folder ID there if the
  download breaks.
- [Leaguepedia](https://lol.fandom.com/wiki/Special:CargoTables) — CC BY-SA.

## Tests

```bash
python tests/test_synthetic.py    # end-to-end on simulated games
python tests/test_tracking.py     # ledger, sources, self-calibration
python tests/make_sample.py       # regenerate the demo data.json
```

The synthetic test gives teams hidden skill values, simulates games from
them, and checks the model recovers most of that signal without beating the
theoretical ceiling — which would mean leakage.
