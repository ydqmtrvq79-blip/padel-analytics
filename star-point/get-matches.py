import os
import sys
import time
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

# Configure basic logging
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Required environment variables
REQUIRED_ENVS = [
    "PADEL_API_TOKEN",
    "POSTGRES_USER",
    "POSTGRES_PWD",
    "POSTGRES_HOST",
    "POSTGRES_DB",
]

missing = [v for v in REQUIRED_ENVS if not os.environ.get(v)]
if missing:
    logger.error("Missing required environment variables: %s", ", ".join(missing))
    sys.exit(2)  # non-zero so schedulers know it failed

# API connection
API_URL = "https://padelapi.org/api/matches/"
API_TOKEN = os.environ["PADEL_API_TOKEN"]

# Postgres configuration
username = os.environ["POSTGRES_USER"]
password = os.environ["POSTGRES_PWD"]
host = os.environ["POSTGRES_HOST"]
database = os.environ["POSTGRES_DB"]

# Optional configuration
MAX_RETRIES = int(os.environ.get("PADEL_MAX_RETRIES", "2"))
REQUEST_TIMEOUT = int(os.environ.get("PADEL_REQUEST_TIMEOUT", "20"))  # seconds
INCREMENTAL_MATCHES = int(os.environ.get("INCREMENTAL_MATCHES", "1"))  # default to True

logger.info("INCREMENTAL_MATCHES(s)=%s", os.environ["INCREMENTAL_MATCHES"])
logger.info("INCREMENTAL_MATCHES(d)=%d", os.environ["INCREMENTAL_MATCHES"])

headers = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Accept": "application/json"
}

# Only matches from yesterday
if INCREMENTAL_MATCHES == 1:
    yesterday = (pd.Timestamp("today") - pd.Timedelta(1, "D")).date()
    params = {
        "before_date": yesterday,
        "after_date": yesterday
    }
else:
    params = {}

# Postgres configuration
username = os.environ["POSTGRES_USER"]
password = os.environ["POSTGRES_PWD"]
host = os.environ["POSTGRES_HOST"]
database = os.environ["POSTGRES_DB"]

# Force retries only for certain response codes
session = requests.Session()
retries = Retry(
    total=MAX_RETRIES,
    status_forcelist=[408, 429, 500, 502, 503, 504]
)
adapter = HTTPAdapter(max_retries=retries)
session.mount("https://", adapter)
session.mount("http://", adapter)

# Function to extract the players names and sides from the nested structure
def extract_team_sides(players: dict) -> dict:
    out = {
        "team_1_backhand": None,
        "team_1_drive": None,
        "team_2_backhand": None,
        "team_2_drive": None,
    }

    if not isinstance(players, dict):
        return out

    for team_key in ("team_1", "team_2"):
        team_players = players.get(team_key, []) or []

        # First pass: use explicit side if present
        for player in team_players:
            side = (player.get("side") or "").lower()
            if side in ("backhand", "drive"):
                out[f"{team_key}_{side}"] = player.get("name")

        # Second pass (fallback): assign by order if still missing
        if out[f"{team_key}_backhand"] is None and len(team_players) >= 1:
            out[f"{team_key}_backhand"] = team_players[0].get("name")

        if out[f"{team_key}_drive"] is None and len(team_players) >= 2:
            out[f"{team_key}_drive"] = team_players[1].get("name")

    return out


def parse_matches(df: pd.DataFrame) -> pd.DataFrame:
    # Flatten everything on first level (excluding players info)
    players_cols = df["players"].apply(extract_team_sides).apply(pd.Series)

    df_flat = pd.concat(
        [df.drop(columns=["players"]), players_cols],
        axis=1
    )

    # Keep only relevant fields
    return df_flat[
        [
            "id",
            "played_at",
            "category",
            "round_name",
            "team_1_backhand",
            "team_1_drive",
            "team_2_backhand",
            "team_2_drive",
            "score",
            "winner",
            "duration"
        ]
    ]

def fetch_matches():
    try:
        logger.info("Requesting %s", API_URL)
        resp = session.get(API_URL, headers=headers, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()  # raises HTTPError for 4xx/5xx
    except requests.exceptions.HTTPError as e:
        # Non-2xx response
        logger.error("HTTP error fetching matches: %s (status %s)", e, getattr(e.response, "status_code", ""))
        raise
    except requests.exceptions.RequestException as e:
        logger.error("Network error fetching matches: %s", e)
        raise

    try:
        payload = resp.json()
    except ValueError:
        logger.error("Response is not valid JSON")
        raise

    if "data" not in payload:
        logger.error("API response missing 'data' key")
        raise RuntimeError("API response missing 'data'")

    df_raw_data = pd.json_normalize(payload["data"], max_level=0)
    df_matches = pd.DataFrame()
    if len(df_raw_data) > 0:
        df_matches = parse_matches(df_raw_data)
        logger.info("Fetched %d match records", len(df_matches))

    return df_matches

def store_matches(df):
    engine_url = f"postgresql+psycopg2://{username}:{password}@{host}/{database}"
    logger.info("Connecting to database %s", database)
    try:
        engine = create_engine(engine_url, pool_pre_ping=True)
        with engine.begin() as conn:
            if_exists = "append" if INCREMENTAL_MATCHES == 1 else "replace"
            table_name = "matches"
            logger.info("Writing DataFrame to table '%s' ('%s')", table_name, if_exists)
            # method='multi' can speed up bulk inserts; adjust chunksize if needed.
            df.to_sql(table_name, conn, if_exists=if_exists, index=False, method="multi")
    except SQLAlchemyError as e:
        logger.exception("Database error while writing matches: %s", e)
        raise
    except Exception as e:
        logger.exception("Unexpected error while writing to DB: %s", e)
        raise
    finally:
        try:
            engine.dispose()
        except Exception:
            pass

def main():
    start = time.time()
    try:
        df_matches = fetch_matches()

        if df_matches.empty:
            logger.warning("No new matches found")
        else:
            store_matches(df_matches)
    except Exception as e:
        logger.error("Matches script failed: %s", e)
        # non-zero exit so scheduler detects failure
        sys.exit(1)

    elapsed = time.time() - start
    logger.info("Matches script completed successfully in %.2f seconds", elapsed)
    sys.exit(0)

if __name__ == "__main__":
    main()