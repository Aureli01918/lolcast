"""Load historical games from Oracle's Elixir.

Oracle's Elixir publishes one CSV per year to a public Google Drive folder,
refreshed daily. Each game occupies twelve rows: five players per side plus
one summary row per team. We keep the two team rows and pivot them into a
single row per game.

The data is free for non-commercial use. Be polite: download once a day,
not once a run.
"""

from __future__ import annotations

import glob
import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import date, timedelta

import pandas as pd
import requests

# Team summary rows use these participant ids.
BLUE_TEAM_ID, RED_TEAM_ID = 100, 200

# Per-team stats carried into TeamState.stat_history for optional features.
STAT_COLUMNS = [
    "golddiffat15",
    "xpdiffat15",
    "csdiffat15",
    "firstdragon",
    "firstbaron",
    "gamelength",
]


# Oracle's Elixir publishes the same CSVs to its own S3 bucket as well as
# to Google Drive. Prefer S3: Drive enforces a per-file download quota that
# is shared across everyone using the same IP, and CI runners share IPs
# with thousands of other jobs, so Drive fails there constantly.
S3_BUCKET = "https://oracleselixir-downloadable-match-data.s3-us-west-2.amazonaws.com"
# Two naming conventions exist. The plain one is what the site serves now;
# the dated one was used historically and those keys get deleted when the
# file is rebuilt, which is why guessing datestamps mostly returns 404.
PLAIN_PATTERN = "{year}_LoL_esports_match_data_from_OraclesElixir.csv"
DATED_PATTERN = "{year}_LoL_esports_match_data_from_OraclesElixir_{stamp}.csv"
# The trailing group allows for compressed uploads: GitHub's web uploader
# caps files at 25MB and these CSVs run to 100MB, but they compress about
# eightfold. pandas reads .zip/.gz/.bz2/.xz transparently.
# Windows' "Send to compressed folder" drops the .csv and produces
# NAME.zip; macOS' Compress keeps it and produces NAME.csv.zip. Accept both.
SUFFIXES = (".csv", ".csv.zip", ".csv.gz", ".csv.bz2", ".csv.xz",
            ".zip", ".gz", ".bz2", ".xz")
KEY_RE = re.compile(
    r"(\d{4})_LoL_esports_match_data_from_OraclesElixir(?:_(\d{8}))?"
    r"(?:\.csv)?(?:\.(?:zip|gz|bz2|xz))?$", re.I
)


def _data_files(directory: str, year: str = "*") -> list[str]:
    """Every match-data file for a year, compressed or not.

    Filenames are checked against KEY_RE rather than trusted from the glob,
    so an unrelated archive that happens to sit in the folder is ignored
    instead of being fed to the CSV parser.
    """
    paths: list[str] = []
    for suffix in SUFFIXES:
        paths.extend(glob.glob(os.path.join(directory, f"*{year}*{suffix}")))
    return sorted({p for p in paths if KEY_RE.search(os.path.basename(p))})


def _existing_years(dest: str) -> set[int]:
    found = set()
    for path in _data_files(dest):
        hit = KEY_RE.search(os.path.basename(path))
        # Compressed files are legitimately much smaller, so the sanity
        # floor has to be lower for them than for raw CSVs.
        floor = 100_000 if path.lower().endswith(".csv") else 10_000
        if hit and os.path.getsize(path) > floor:
            found.add(int(hit.group(1)))
    return found


def _s3_listing(session: requests.Session) -> dict[int, str]:
    """Ask the bucket what it holds. One request, exact filenames.

    Returns {year: newest key}. Empty dict if listing is not permitted,
    which is common for public-read buckets.
    """
    newest: dict[int, tuple[str, str]] = {}
    token = None
    for _ in range(10):
        params = {"list-type": "2", "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        response = session.get(S3_BUCKET, params=params, timeout=60)
        if response.status_code != 200:
            return {}
        try:
            root = ET.fromstring(response.content)
        except ET.ParseError:
            return {}
        ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
        for item in root.findall(".//s3:Contents/s3:Key", ns) or root.iter("Key"):
            key = (item.text or "").strip()
            hit = KEY_RE.search(key)
            if not hit:
                continue
            year, stamp = int(hit.group(1)), (hit.group(2) or "00000000")
            if year not in newest or stamp > newest[year][1]:
                newest[year] = (key, stamp)
        more = root.find(".//s3:IsTruncated", ns)
        token_el = root.find(".//s3:NextContinuationToken", ns)
        if more is None or (more.text or "").lower() != "true" or token_el is None:
            break
        token = token_el.text
    return {year: key for year, (key, _) in newest.items()}


def _candidate_keys(year: int, today: date) -> list[str]:
    """Filenames to try when the bucket won't list itself.

    The undated name comes first because it is what the site serves today,
    and it costs a single request. The dated names are a fallback for older
    files: the current year is rebuilt daily so recent dates are likely,
    while a finished year stops updating in the months after it ends.
    """
    keys = [PLAIN_PATTERN.format(year=year)]
    if year >= today.year:
        days = [today - timedelta(days=n) for n in range(0, 30)]
    else:
        end = date(year + 1, 3, 31)
        days = [end - timedelta(days=n) for n in range(0, 90)]
    keys += [DATED_PATTERN.format(year=year, stamp=d.strftime("%Y%m%d"))
             for d in days]
    return keys


def _stream_to_disk(session: requests.Session, url: str, path: str) -> bool:
    with session.get(url, stream=True, timeout=180) as response:
        if response.status_code != 200:
            return False
        tmp = path + ".part"
        size = 0
        with open(tmp, "wb") as fh:
            for chunk in response.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                size += len(chunk)
        if size < 100_000:                      # an error page, not a dataset
            os.remove(tmp)
            return False
        os.replace(tmp, path)
        print(f"    {os.path.basename(path)}  {size / 1e6:.1f} MB")
        return True


def download_from_s3(dest: str, years: list[int]) -> set[int]:
    """Fetch each wanted year from S3. Returns the years now on disk."""
    os.makedirs(dest, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "lolcast/0.1 (personal match tracker)"

    have = _existing_years(dest)
    listing = _s3_listing(session)
    if listing:
        print(f"  bucket listing returned {len(listing)} year(s)")

    today = date.today()
    for year in years:
        # The current year changes daily, so always refresh it. Finished
        # years never change, so skip them once downloaded.
        if year in have and year < today.year:
            print(f"  {year}: already downloaded")
            continue

        keys = [listing[year]] if year in listing else _candidate_keys(year, today)
        for key in keys:
            try:
                if _stream_to_disk(session, f"{S3_BUCKET}/{key}",
                                   os.path.join(dest, os.path.basename(key))):
                    have.add(year)
                    break
            except requests.RequestException as exc:
                print(f"  {year}: {type(exc).__name__}, trying next")
        else:
            print(f"  {year}: not found on S3")

    return have


def download_from_drive(folder_id: str, dest: str, attempts: int = 3) -> None:
    """Last resort. Google Drive enforces a shared download quota, so this
    fails often from CI even though it works fine from a home connection."""
    import gdown

    os.makedirs(dest, exist_ok=True)
    for attempt in range(1, attempts + 1):
        try:
            gdown.download_folder(
                f"https://drive.google.com/drive/folders/{folder_id}",
                output=dest, quiet=False, use_cookies=False,
            )
            return
        except Exception as exc:                            # noqa: BLE001
            print(f"  Drive attempt {attempt}/{attempts} failed: {exc}")
            if attempt < attempts:
                time.sleep(20 * attempt)


def download(folder_id: str, dest: str, years: list[int] | None = None) -> None:
    """Get the yearly CSVs, trying the reliable route first.

    Missing a year is not fatal. Fewer years means slightly weaker early
    ratings, not a broken system, and stopping the whole build because one
    file was unavailable would be the wrong trade.
    """
    years = years or [date.today().year]
    print("Fetching match history from S3")
    have = download_from_s3(dest, years)

    missing = [y for y in years if y not in have]
    if missing:
        print(f"S3 missing {missing}; falling back to Google Drive")
        download_from_drive(folder_id, dest)
        have = _existing_years(dest)
        missing = [y for y in years if y not in have]

    if not have:
        raise RuntimeError(
            "No match data could be downloaded from either source. "
            "Download the CSVs by hand from https://oracleselixir.com/tools/downloads "
            f"and put them in {dest}."
        )
    if missing:
        print(f"Warning: continuing without {missing}. Ratings will lean on "
              f"{sorted(have)}.")
    print(f"Match data available for {sorted(have)}")


def load(raw_dir: str, years: list[int], leagues: list[str] | None = None) -> pd.DataFrame:
    """Read the CSVs and return one normalised row per game."""
    paths = []
    for year in years:
        paths.extend(_data_files(raw_dir, str(year)))
    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError(
            f"No Oracle's Elixir CSVs found in {raw_dir!r}. "
            f"Run `python -m lolcast fetch-history` first."
        )

    frames = []
    for path in paths:
        # compression="infer" handles .zip/.gz/.bz2/.xz from the extension.
        frames.append(pd.read_csv(path, low_memory=False, compression="infer"))
        print(f"  read {os.path.basename(path)}: {len(frames[-1]):,} rows")
    raw = pd.concat(frames, ignore_index=True)

    teams = raw[raw["participantid"].isin([BLUE_TEAM_ID, RED_TEAM_ID])].copy()
    if leagues:
        teams = teams[teams["league"].isin(leagues)]

    # Drop games with known-incomplete data rather than modelling noise.
    if "datacompleteness" in teams.columns:
        teams = teams[teams["datacompleteness"] == "complete"]

    keep = ["gameid", "date", "league", "patch", "side", "teamname", "result"]
    keep += [c for c in STAT_COLUMNS if c in teams.columns]
    teams = teams[[c for c in keep if c in teams.columns]]

    blue = teams[teams["side"] == "Blue"].set_index("gameid")
    red = teams[teams["side"] == "Red"].set_index("gameid")
    common = blue.index.intersection(red.index)
    blue, red = blue.loc[common], red.loc[common]

    out = pd.DataFrame(
        {
            "gameid": common,
            "date": blue["date"].to_numpy(),
            "league": blue["league"].to_numpy(),
            "patch": blue["patch"].astype(str).to_numpy(),
            "blue": blue["teamname"].to_numpy(),
            "red": red["teamname"].to_numpy(),
            "blue_win": blue["result"].to_numpy(),
        }
    )
    for col in STAT_COLUMNS:
        if col in blue.columns:
            out[col] = blue[col].to_numpy()

    # A team's region comes from the domestic league it plays in most, not
    # from the league of the current game. Otherwise both sides of a Worlds
    # game would look like they share a region.
    home = team_regions(out)
    out["blue_region"] = out["blue"].map(home).fillna("INTL")
    out["red_region"] = out["red"].map(home).fillna("INTL")
    out["best_of"] = 1  # historical rows are individual games

    out = out.dropna(subset=["blue", "red", "blue_win", "date"])
    return out.sort_values("date").reset_index(drop=True)


# Keys cover both naming schemes: Oracle's Elixir league codes and
# Leaguepedia page prefixes, which differ for several leagues.
REGIONS = {
    "LCK": "KR", "LCKC": "KR",
    "LPL": "CN", "LDL": "CN",
    "LEC": "EU", "EM": "EU", "LFL": "EU", "PRM": "EU", "NLC": "EU",
    "LTA N": "AM", "LTA S": "AM", "LCS": "AM", "CBLOL": "AM",
    "LCP": "APAC", "PCS": "APAC", "VCS": "APAC", "LJL": "APAC",
    # Leaguepedia spellings
    "LTA North": "AM", "LTA South": "AM",
    "Worlds": "INTL", "WLDs": "INTL", "First Stand": "INTL", "FST": "INTL",
}


def region_of(league: str) -> str:
    """Region for a league. International events are marked INTL so that
    cross_region fires on them."""
    return REGIONS.get(league, "INTL")


def team_regions(games: pd.DataFrame) -> dict[str, str]:
    """Map each team to the region of the domestic league it plays most."""
    long = pd.concat(
        [
            games[["blue", "league"]].rename(columns={"blue": "team"}),
            games[["red", "league"]].rename(columns={"red": "team"}),
        ]
    )
    long["region"] = long["league"].map(region_of)
    domestic = long[long["region"] != "INTL"]
    if domestic.empty:
        return {}
    counts = domestic.groupby(["team", "region"]).size()
    return counts.groupby(level=0).idxmax().map(lambda pair: pair[1]).to_dict()
