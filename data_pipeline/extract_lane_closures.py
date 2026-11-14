import requests
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import os
import time
from requests.exceptions import RequestException
# ----------------- CONFIG ----------------- #

PEMS_BASE_URL = "https://pems.dot.ca.gov"
LOGIN_URL = f"{PEMS_BASE_URL}/?dnode=login"

# 🔁 Change this to switch districts (3,4,5,6,7,8,10,11,12)
DISTRICT_ID = int(os.getenv("PEMS_DISTRICT_ID", "12"))

# LCS URL (uses DISTRICT_ID)
LCS_URL = (
    f"{PEMS_BASE_URL}/?dnode=District&content=lcs&tab=lcs_list&district_id={DISTRICT_ID}"
)

# PeMS credentials are read from the PEMS_USERNAME / PEMS_PASSWORD environment variables.
# ❌ Do NOT push these to git.
PEMS_USERNAME = os.environ.get("PEMS_USERNAME", "")
PEMS_PASSWORD = os.environ.get("PEMS_PASSWORD", "")

SCRIPT_DIR = Path(__file__).resolve().parent

# Base output folder depending on district
def district_code(d):
    return f"{int(d):02d}"   # 3→"03", 10→"10"

DIST_CODE = district_code(DISTRICT_ID)

# New output directory: data/lane_closures/pemsXX/
OUTPUT_DIR = SCRIPT_DIR / "data" / "lane_closures" / f"pems{DIST_CODE}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Per-district date ranges (month/year) for lane closures
# Format: district_id: (start_year, start_month, start_day)
# End is shared: 2024-03-22 23:59 for all.
DISTRICT_START_DATES = {
    3:  (2001, 3, 4),   # PEMS03_Agg 03/2001 - 03/2024, start 2001-03-04
    4:  (2002, 6, 1),   # PEMS04_Agg 06/2002 - 03/2024
    5:  (2012, 3, 1),   # PEMS05_Agg 03/2012 - 03/2024
    6:  (2009, 12, 1),  # PEMS06_Agg 12/2009 - 03/2024
    7:  (2002, 6, 1),   # PEMS07_Agg 06/2002 - 03/2024
    8:  (2001, 3, 1),   # PEMS08_Agg 03/2001 - 03/2024
    10: (2007, 6, 1),   # PEMS10_Agg 06/2007 - 03/2024
    11: (2002, 9, 1),   # PEMS11_Agg 09/2002 - 03/2024
    12: (2002, 1, 1),   # PEMS12_Agg 01/2002 - 03/2024
}

# Common end date for all districts
GLOBAL_END_COMMON = datetime(2024, 3, 22, 23, 59)

# Max window length – keep under "3 months" by using 90 days
MAX_WINDOW_DAYS = 90

SKIP_EXISTING = True   # don't re-download chunks that are already on disk
MAX_RETRIES   = 3      # how many times to retry a failed chunk
RETRY_SLEEP   = 5      # seconds to wait between retries

def get_date_range_for_district(district_id: int) -> tuple[datetime, datetime]:
    """
    Return (global_start, global_end) for the given district.
    Start date comes from DISTRICT_START_DATES.
    End date is fixed to 2024-03-22 23:59 for all.
    """
    if district_id not in DISTRICT_START_DATES:
        raise ValueError(f"No start date configured for district {district_id}")

    sy, sm, sd = DISTRICT_START_DATES[district_id]
    global_start = datetime(sy, sm, sd, 0, 0)
    global_end = GLOBAL_END_COMMON
    return global_start, global_end


# ----------------- BASIC LOGIN ----------------- #

def create_session() -> requests.Session:
    s = requests.Session()
    print("Session created:", s)
    return s


def login(session: requests.Session) -> None:
    payload = {
        "username": PEMS_USERNAME,
        "password": PEMS_PASSWORD,
        "login": "Login",
    }

    print(f"\n[LOGIN] POST {LOGIN_URL}")
    resp = session.post(LOGIN_URL, data=payload)

    print("[LOGIN] HTTP status:", resp.status_code)
    print("[LOGIN] Response length:", len(resp.text))

    print("\n[LOGIN] Response snippet (first 300 chars):\n")
    print(resp.text[:300])
    print("\n--- END LOGIN SNIPPET ---\n")


# ----------------- LCS QUERY BUILDING ----------------- #

def build_lcs_query(start_dt: datetime, end_dt: datetime) -> dict:
    # Convert datetimes to UNIX seconds (UTC)
    s_unix = int(start_dt.replace(tzinfo=timezone.utc).timestamp())
    e_unix = int(end_dt.replace(tzinfo=timezone.utc).timestamp())

    params = {
        "report_form": "1",
        "dnode": "District",
        "content": "lcs",
        "tab": "lcs_list",

        "district_id": str(DISTRICT_ID),

        "lcs_date_type": "0",

        "s_time_id": str(s_unix),
        "e_time_id": str(e_unix),

        "s_time_id_f": "",
        "e_time_id_f": "",

        "q": "cnt",
        "lcs_grace_min": "10",

        "lcs_filter_mode": "none",

        "sort_col": "0",
        "sort_dir": "1",

        "export": "xls",
    }

    return params


def download_lcs_range(
    session: requests.Session,
    start_dt: datetime,
    end_dt: datetime,
    out_path: Path,
) -> None:
    params = build_lcs_query(start_dt, end_dt)

    print(f"[LCS] GET {LCS_URL}")
    print("[LCS] Query params:", params)

    resp = session.get(LCS_URL, params=params)

    print("[LCS] HTTP status:", resp.status_code)
    print("[LCS] Content-Type:", resp.headers.get("Content-Type"))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(resp.content)
    print(f"[LCS] Saved to {out_path}")

    snippet = resp.content[:300]
    try:
        txt = snippet.decode("utf-8", errors="replace")
    except Exception:
        txt = str(snippet)

    print("\n[LCS] Preview (first 300 bytes as text):\n")
    print(txt)
    print("\n--- END PREVIEW ---\n")


# ----------------- CHUNK ITERATOR ----------------- #

def iter_date_chunks(
    global_start: datetime,
    global_end: datetime,
    max_days: int,
):
    cur_start = global_start

    while cur_start <= global_end:
        cur_end = cur_start + timedelta(days=max_days)
        if cur_end > global_end:
            cur_end = global_end

        yield cur_start, cur_end

        # move to next chunk; +1 second to avoid overlap
        cur_start = cur_end + timedelta(seconds=1)


# ----------------- FULL RANGE DOWNLOAD ----------------- #

def download_full_range_in_chunks(
    session: requests.Session,
    global_start: datetime,
    global_end: datetime,
) -> Path:
    """
    Downloads chunks into pems_lane_closures/chunks.
    Returns the chunks directory path.

    - Same signature as before.
    - Same chunk_dir = OUTPUT_DIR / "chunks".
    - Adds:
        * resume support via SKIP_EXISTING
        * retry logic per chunk
    """
    chunk_dir = OUTPUT_DIR / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    for idx, (start_dt, end_dt) in enumerate(
        iter_date_chunks(global_start, global_end, MAX_WINDOW_DAYS),
        start=1,
    ):
        start_str = start_dt.strftime("%Y-%m-%d_%H%M")
        end_str   = end_dt.strftime("%Y-%m-%d_%H%M")
        out_file  = chunk_dir / f"lane_closures_dist{DISTRICT_ID}_{start_str}_to_{end_str}.xls"

        # ✅ Resume logic: skip if this file already exists
        if SKIP_EXISTING and out_file.exists():
            print(f"[SKIP] Chunk {idx} already exists: {out_file.name}")
            continue

        print(f"\n===== CHUNK {idx} :: {start_dt}  →  {end_dt} =====")

        # ✅ Retry logic
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                print(f"[TRY] Downloading chunk {idx}, attempt {attempt}/{MAX_RETRIES}")
                download_lcs_range(session, start_dt, end_dt, out_file)
                print(f"[OK] Chunk {idx} saved to {out_file}")
                break  # success → leave retry loop

            except RequestException as e:
                print(f"[WARN] Network error on chunk {idx}: {e}")
                if attempt == MAX_RETRIES:
                    print(f"[FAIL] Giving up on chunk {idx} after {MAX_RETRIES} attempts.")
                else:
                    print(f"[WAIT] Sleeping {RETRY_SLEEP}s before retry...")
                    time.sleep(RETRY_SLEEP)

            except Exception as e:
                # Any other unexpected error → log and move on
                print(f"[ERROR] Unexpected error on chunk {idx}: {e}")
                break

    return chunk_dir



# ----------------- MERGE + CLEANUP ----------------- #

def merge_chunks(
    chunks_dir: Path,
    global_start: datetime,
    global_end: datetime,
) -> None:
    """
    Merge all .xls files in chunks_dir into a single CSV in OUTPUT_DIR,
    then delete all chunk files and remove the chunks_dir.
    """
    excel_files = sorted(chunks_dir.glob("*.xls"))

    if not excel_files:
        print(f"No .xls files found in {chunks_dir}")
        return

    print(f"Found {len(excel_files)} chunk files to merge.")

    dfs = []
    for f in excel_files:
        print(f"Reading {f.name} ...")
        df = pd.read_excel(f)
        df["source_file"] = f.name
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)

    # Optional: drop exact duplicate rows
    # merged = merged.drop_duplicates()

    start_str = global_start.strftime("%Y-%m-%d")
    end_str = global_end.strftime("%Y-%m-%d")

    out_csv = OUTPUT_DIR / f"pems{DIST_CODE}_lane_closures_{start_str}_to_{end_str}.csv"
    merged.to_csv(out_csv, index=False)
    print(f"Saved merged CSV to {out_csv}")

    # 🔥 Delete chunk files
    print("Deleting chunk .xls files...")
    for f in excel_files:
        try:
            f.unlink()
            print(f"Deleted {f}")
        except Exception as e:
            print(f"Could not delete {f}: {e}")

    # 🔥 Try to remove the (now empty) chunk directory
    try:
        chunks_dir.rmdir()
        print(f"Removed empty directory {chunks_dir}")
    except OSError:
        print(f"Directory {chunks_dir} not empty or cannot be removed (might contain other files).")


# ----------------- OPTIONAL: ONE-DAY TEST ----------------- #

def test_one_day(session: requests.Session, global_start: datetime):
    start_dt = global_start
    end_dt = start_dt + timedelta(days=1)

    test_dir = OUTPUT_DIR / "test"
    out_file = test_dir / f"lane_closures_dist{DISTRICT_ID}_{start_dt.strftime('%Y-%m-%d')}.xls"
    download_lcs_range(session, start_dt, end_dt, out_file)


# ----------------- MAIN ----------------- #

def main():
    # Get district-specific date range
    global_start, global_end = get_date_range_for_district(DISTRICT_ID)
    print(f"[INFO] District {DISTRICT_ID} date range: {global_start} → {global_end}")

    session = create_session()
    login(session)

    # 🔍 Optional:
    # test_one_day(session, global_start); return

    chunks_dir = download_full_range_in_chunks(session, global_start, global_end)
    merge_chunks(chunks_dir, global_start, global_end)


if __name__ == "__main__":
    main()
