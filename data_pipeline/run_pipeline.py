import os
import re
import subprocess

# districts to run
#districts = ["8"]
districts = ["04","05","06","08","10","11","12"]

# python executable (the one that works)
PYTHON = r"C:\Users\erays\AppData\Local\Programs\Python\Python313\python.exe"


def run_script(script, args=None, capture=False):
    cmd = [PYTHON, script]
    if args:
        cmd += args

    if capture:
        result = subprocess.run(cmd, text=True, capture_output=True)
        print(result.stdout)  # show output
        if result.stderr:
            print(result.stderr)
        return result.stdout
    else:
        subprocess.run(cmd)


def extract_date(output):
    matches = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", output)
    if not matches:
        return None
    return matches[-1]


for d in districts:

    print("=========================================")
    print(f"Processing District {d}")
    print("=========================================")

    # set env variable
    os.environ["PEMS_DISTRICT_ID"] = str(int(d))

    # 1. create DESPT input
    run_script("create_DESPT_input.py")

    # 2. find best window (capture output but still print it)
    output = run_script("find_unseen_window.py", capture=True)

    date = extract_date(output)

    if not date:
        print("ERROR: Could not extract date")
        exit()

    print(f"Best date: {date}")

    # 3. cut dataset using date
    run_script("cut_save_data.py", [date])

    # 4. convert to csv
    run_script("h5_2_csv.py")

    # 5. fetch lat/lon
    run_script("get_station_latlon.py")

    # 6. build osrm adjacency
    run_script("build_osrm_adj.py")

    # 7. build STGCN input
    run_script("build_STGCN_input.py")

    print(f"DONE District {d}")