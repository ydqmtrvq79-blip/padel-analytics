import os
import sys
import time
import pandas as pd
import numpy as np
import logging
from utils.db_postgres import read_db_table, write_db


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

# Optional configuration
INCREMENTAL_MATCHES = int(os.environ.get("INCREMENTAL_MATCHES", "0"))  # default to False (full refresh)

# Postgres configuration
username = os.environ["POSTGRES_USER"]
password = os.environ["POSTGRES_PWD"]
host = os.environ["POSTGRES_HOST"]
database = os.environ["POSTGRES_DB"]

def get_bronze_data(table_name):
    df = read_db_table(table_name, schema="bronze")
    return df


def transform_matches(df_raw_data):
    df_matches = df_raw_data
    df_matches["duration_minutes"] = (
        df_raw_data["duration"]
          .str.split(":", expand=True)
          .astype(float)
          .pipe(lambda x: x[0] * 60 + x[1])
    )
    df_matches["created_at"] = pd.Timestamp("now")
    return df_matches.drop(columns=["duration"])


def add_set_scores(df):
    # 0) Deterministic point order
    df = df.sort_values(["match_id", "set_number", "game_number", "point_number"])

    # 1) Last point row per set (per match)
    last_row_per_set = (
        df.groupby(["match_id", "set_number"], as_index=False)
        .tail(1)
        .loc[:, ["match_id", "set_number", "game_score_start", "point_score_start"]]
        .copy()
    )

    # 2) Parse game score "6-4" -> g1, g2
    g = last_row_per_set["game_score_start"].str.split("-", expand=True)
    last_row_per_set["g1"] = pd.to_numeric(g[0], errors="coerce")
    last_row_per_set["g2"] = pd.to_numeric(g[1], errors="coerce")

    # 3) Parse point score (tiebreak points) "10-8" -> tb1, tb2
    tb = last_row_per_set["point_score_start"].str.split(r"[:\-]", expand=True)
    last_row_per_set["tb1"] = pd.to_numeric(tb[0], errors="coerce")
    last_row_per_set["tb2"] = pd.to_numeric(tb[1], errors="coerce")

    # 4) Decide set winner
    normal = last_row_per_set["g1"] != last_row_per_set["g2"]

    # normal sets: winner by games
    last_row_per_set["set_win_team_1"] = np.where(
        normal, (last_row_per_set["g1"] > last_row_per_set["g2"]).astype(int), np.nan
    )
    last_row_per_set["set_win_team_2"] = np.where(
        normal, (last_row_per_set["g2"] > last_row_per_set["g1"]).astype(int), np.nan
    )

    # tiebreak sets (games tied, e.g. 6-6): winner by tiebreak points on the last row
    tb_set = ~normal

    # Prefer "terminal" TB score (>=7 and lead by 2). As we have TB score at point-start,
    # it will not be terminal; then we fall back to "leader" on last row.
    tb_terminal = (
        last_row_per_set["tb1"].notna() &
        last_row_per_set["tb2"].notna() &
        (last_row_per_set[["tb1", "tb2"]].max(axis=1) >= 7) &
        ((last_row_per_set["tb1"] - last_row_per_set["tb2"]).abs() >= 2)
    )
    tb_leader = (
        last_row_per_set["tb1"].notna() &
        last_row_per_set["tb2"].notna() &
        (last_row_per_set["tb1"] != last_row_per_set["tb2"])
    )

    use_tb = tb_set & (tb_terminal | tb_leader)

    last_row_per_set.loc[use_tb, "set_win_team_1"] = (last_row_per_set.loc[use_tb, "tb1"] > last_row_per_set.loc[use_tb, "tb2"]).astype(int)
    last_row_per_set.loc[use_tb, "set_win_team_2"] = (last_row_per_set.loc[use_tb, "tb2"] > last_row_per_set.loc[use_tb, "tb1"]).astype(int)

    # If still unknown (live/incomplete), treat as not yet won
    last_row_per_set["set_win_team_1"] = last_row_per_set["set_win_team_1"].fillna(0).astype(int)
    last_row_per_set["set_win_team_2"] = last_row_per_set["set_win_team_2"].fillna(0).astype(int)

    # 5) Set score BEFORE current set (shift)
    last_row_per_set = last_row_per_set.sort_values(["match_id", "set_number"])

    last_row_per_set["set_score_team_1"] = (
        last_row_per_set.groupby("match_id")["set_win_team_1"]
        .cumsum()
        .shift(1)
        .fillna(0)
        .astype(int)
    )
    last_row_per_set["set_score_team_2"] = (
        last_row_per_set.groupby("match_id")["set_win_team_2"]
        .cumsum()
        .shift(1)
        .fillna(0)
        .astype(int)
    )

    # 6) Merge to every point row
    df = df.merge(
        last_row_per_set[["match_id", "set_number", "set_score_team_1", "set_score_team_2"]],
        on=["match_id", "set_number"],
        how="left",
    )
    df["created_at"] = pd.Timestamp("now")
    
    return df


def add_game_and_point_scores(df):
    df = (
        df
        .assign(
            # --- Game score: always "X-Y" (sometimes with spaces) ---
            game_score_team_1=lambda d: pd.to_numeric(
                d["game_score_start"].astype(str).str.extract(r"^\s*(\d+)\s*-\s*(\d+)\s*$")[0],
                errors="coerce",
            ),
            game_score_team_2=lambda d: pd.to_numeric(
                d["game_score_start"].astype(str).str.extract(r"^\s*(\d+)\s*-\s*(\d+)\s*$")[1],
                errors="coerce",
            ),

            # --- Point score: either "X:Y" (normal) or "X-Y" (tiebreak), can include "A" ---
            point_score_team_1=lambda d: d["point_score_start"]
                .astype(str)
                .str.extract(r"^\s*([^:\-]+)\s*[:\-]\s*([^:\-]+)\s*$")[0]
                .replace({"nan": pd.NA}),
            point_score_team_2=lambda d: d["point_score_start"]
                .astype(str)
                .str.extract(r"^\s*([^:\-]+)\s*[:\-]\s*([^:\-]+)\s*$")[1]
                .replace({"nan": pd.NA}),
        )
        .drop(columns=["game_score_start", "point_score_start"])
    )
    return df


def tag_tie_break_and_deuce_points(df):
    df["is_tiebreak"] = (
        (df["game_score_team_1"] == 6) &
        (df["game_score_team_2"] == 6)
    )#.astype(int)

    df["is_deuce"] = df.apply(
        lambda row: True if (row["point_score_team_1"] == "40" and row["point_score_team_2"] == "40")
         or (row["is_tiebreak"] == True and pd.to_numeric(row["point_score_team_1"]) >= 6 and row["point_score_team_1"] == row["point_score_team_2"]) else False,
        axis=1
    )

    return df


def tag_key_points(df):
    # Tag game points first, then set points, and finally match points
    SETS_TO_WIN = 2  # best-of-3; set to 3 for best-of-5

    df = df.sort_values(["match_id", "set_number", "game_number", "point_number"]).copy()

    # --- helpers: parse normal point values (0/15/30/40/A) ---
    point_rank = {"0": 0, "15": 1, "30": 2, "40": 3, "A": 4}

    p1r = df["point_score_team_1"].map(point_rank)
    p2r = df["point_score_team_2"].map(point_rank)

    # --- parse tiebreak numeric points (e.g. "7", "5") ---
    tb1 = pd.to_numeric(df["point_score_team_1"].where(df["point_score_team_1"].str.fullmatch(r"\d+")), errors="coerce")
    tb2 = pd.to_numeric(df["point_score_team_2"].where(df["point_score_team_2"].str.fullmatch(r"\d+")), errors="coerce")

    # Identify tiebreak points (your data uses numeric scores in TB)
    is_tb_point = (
        (df["game_score_team_1"] == 6) &
        (df["game_score_team_2"] == 6)
    )

    # --- determine game leader (1 or 2). ties -> 0 (no leader) ---
    leader = np.select(
        [
            is_tb_point & (tb1 > tb2),
            is_tb_point & (tb2 > tb1),
            (~is_tb_point) & (p1r > p2r),
            (~is_tb_point) & (p2r > p1r),
        ],
        [1, 2, 1, 2],
        default=0
    )

    # --- is this point a GAME POINT for the current leader? ---
    # Normal game point logic:
    # leader has A -> wins game if wins point
    # leader has 40 and opponent <= 30 -> wins game if wins point
    gp_normal_team1 = (p1r == 4) | ((p1r == 3) & (p2r <= 2))
    gp_normal_team2 = (p2r == 4) | ((p2r == 3) & (p1r <= 2))
    is_game_point_normal = np.where(leader == 1, gp_normal_team1,
                            np.where(leader == 2, gp_normal_team2, False))
    
    # Tiebreak game point logic:
    # if leader wins THIS point, do they reach >=7 with lead >=2?
    # (using start-of-point score, so check (l+1, o))
    l_tb = np.where(leader == 1, tb1, np.where(leader == 2, tb2, np.nan))
    o_tb = np.where(leader == 1, tb2, np.where(leader == 2, tb1, np.nan))
    is_game_point_tb = is_tb_point & (leader != 0) & ((l_tb + 1 >= 7) & ((l_tb + 1) - o_tb >= 2))

    is_game_point = np.where(is_tb_point, is_game_point_tb, is_game_point_normal)
    df["is_game_point"] = (leader != 0) & is_game_point

    # --- would winning the GAME clinch the SET? (set point) ---
    g1 = df["game_score_team_1"]
    g2 = df["game_score_team_2"]

    leader_games = np.where(leader == 1, g1, np.where(leader == 2, g2, np.nan))
    other_games  = np.where(leader == 1, g2, np.where(leader == 2, g1, np.nan))

    leader_games_if_win = leader_games + 1
    # Standard set win condition: reach 6 with 2-game lead, OR reach 7 (incl 7-5, 7-6 TB set)
    clinch_set_if_game_won = (leader_games_if_win >= 6) & (
        ((leader_games_if_win - other_games) >= 2) | (leader_games_if_win == 7)
    )

    df["is_set_point"] = (leader != 0) & is_game_point & clinch_set_if_game_won

    # --- would clinching the SET also clinch the MATCH? (match point) ---
    sets1 = df["set_score_team_1"]
    sets2 = df["set_score_team_2"]

    leader_sets_before = np.where(leader == 1, sets1, np.where(leader == 2, sets2, np.nan))
    df["is_match_point"] = df["is_set_point"] & (leader_sets_before + 1 >= SETS_TO_WIN)

    return df


def transform_scores(df_raw_data):
    df_scores = df_raw_data  # Placeholder for actual transformation
    df_scores = add_set_scores(df_scores)
    df_scores = add_game_and_point_scores(df_scores)
    df_scores = tag_tie_break_and_deuce_points(df_scores)
    df_scores = tag_key_points(df_scores)
    return df_scores


def store_silver_data(df, table_name):
    if_exists = "append" if INCREMENTAL_MATCHES == 1 else "replace"
    write_db(df, table_name, schema="silver", if_exists=if_exists)


def main():
    start = time.time()
    try:
        df_raw_matches = get_bronze_data(table_name="fact_match")

        if df_raw_matches.empty:
            logger.warning("No matches found in bronze layer.")
        else:
            df_trx_matches = transform_matches(df_raw_matches)
            store_silver_data(df_trx_matches, table_name="fact_match")

        df_raw_scores = get_bronze_data(table_name="fact_point")

        if df_raw_scores.empty:
            logger.warning("No point scores found in bronze layer.")
        else:
            df_trx_scores = transform_scores(df_raw_scores)
            store_silver_data(df_trx_scores, table_name="fact_point")

    except Exception as e:
        logger.error("Match transformation script failed: %s", e)
        # non-zero exit so scheduler detects failure
        sys.exit(1)

    elapsed = time.time() - start
    logger.info("Processed %d matches, %.1fK points, %.1f%% deuce, %.1f%% tie-break, %.1f%% game points, %.1f%% set points, %.1f%% match points", len(df_trx_matches), round(len(df_trx_scores)/1000,1), round(100 * df_trx_scores["is_deuce"].mean(),1), round(100 * df_trx_scores["is_tiebreak"].mean(),1), round(100 * df_trx_scores["is_game_point"].mean(),1), round(100 * df_trx_scores["is_set_point"].mean(),1), round(100 * df_trx_scores["is_match_point"].mean(),1))
    logger.info("Match transformation script completed successfully in %.2f seconds", elapsed)
    sys.exit(0)

if __name__ == "__main__":
    main()