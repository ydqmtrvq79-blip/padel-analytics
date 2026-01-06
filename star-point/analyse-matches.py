import os
import sys
import time
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import logging

# Configure basic logging
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Required environment variables
REQUIRED_ENVS = [
    "POSTGRES_USER",
    "POSTGRES_PWD",
    "POSTGRES_HOST",
    "POSTGRES_DB",
]

missing = [v for v in REQUIRED_ENVS if not os.environ.get(v)]
if missing:
    logger.error("Missing required environment variables: %s", ", ".join(missing))
    sys.exit(2)  # non-zero so schedulers know it failed

# Postgres configuration
username = os.environ["POSTGRES_USER"]
password = os.environ["POSTGRES_PWD"]
host = os.environ["POSTGRES_HOST"]
database = os.environ["POSTGRES_DB"]

def read_matches(conn):
    try:
        # Read from Postgres DB
        df_matches = pd.read_sql_query(text("SELECT * FROM matches"), conn)
        logger.info("Found %d matches", len(df_matches))
    except Exception as e:
        logger.error("Error reading matches from DB: %s", e,)
        raise

    return df_matches

def transform(df):
    # Example transformation: calculate match duration in minutes
    df["duration_minutes"] = (
        df["duration"]
          .dropna()
          .str.split(":", expand=True)
          .astype(int)
          .pipe(lambda x: x[0] * 60 + x[1])
    )
    return df

def create_summary(df):
    # Example summary: number of matches and average duration by category and day
    df_summary = df \
        .groupby(['category','played_at']) \
        .agg(
            match_count=('played_at', 'count'),
            avg_duration_minutes=('duration_minutes', 'mean')
        ).reset_index()
    
    df_summary['days_since_last_match'] = (
        pd.Timestamp("today").normalize() - pd.to_datetime(df_summary['played_at']).dt.normalize()
    ).dt.days

    logger.info("Summary created with %d rows", len(df_summary))
    return df_summary

def store_summary(df, conn):
    
    try:
        table_name = "match_summary"
        if_exists = "replace"
        logger.info("Writing DataFrame to table '%s' (%s)", table_name, if_exists)
        # method='multi' can speed up bulk inserts; adjust chunksize if needed.
        df.to_sql(table_name, conn, if_exists=if_exists, index=False, method="multi")
    except Exception as e:
        logger.exception("Unexpected error while writing to DB: %s", e)
        raise

def main():
    start = time.time()
    try:
        engine_url = f"postgresql+psycopg2://{username}:{password}@{host}/{database}"
        logger.info("Connecting to database %s", database)
        engine = create_engine(engine_url, pool_pre_ping=True)
        with engine.begin() as conn:
            df_raw_matches = read_matches(conn)

            if df_raw_matches.empty:
                logger.warning("No matches found")
            else:
                df_transformed = transform(df_raw_matches)
                df_summary = create_summary(df_transformed)
                store_summary(df_summary, conn)
        engine.dispose()
    except Exception as e:
        logger.error("Matches script failed: %s", e)
        # non-zero exit so scheduler detects failure
        sys.exit(1)

    elapsed = time.time() - start
    logger.info("Analysis script completed successfully in %.2f seconds", elapsed)
    sys.exit(0)

if __name__ == "__main__":
    main()