# districts to run
$districts = @("03")
#$districts = @("03","04","05","06","10","11","12")

# force correct python (the one that works)
$python = "C:\Users\erays\AppData\Local\Programs\Python\Python313\python.exe"

foreach ($d in $districts) {

    Write-Host "========================================="
    Write-Host "Processing District $d"
    Write-Host "========================================="

    # set env variable
    $env:PEMS_DISTRICT_ID = [int]$d

    #& $python create_DESPT_input.py 2>&1 | Tee-Object

    # 2. find best window (SHOW + CAPTURE)
    $output = & $python find_unseen_window.py 2>&1 | Tee-Object -Variable output

    # extract date (YYYY-MM-DD)
    $date = ($output | Select-String "^\d{4}-\d{2}-\d{2}$" | Select-Object -Last 1).Matches.Value

    if (-not $date) {
        Write-Host "ERROR: Could not extract date"
        exit
    }

    Write-Host "Best date: $date"

    # 3. cut dataset using date
    & $python cut_save_data.py $date 2>&1 | Tee-Object

    # 4. convert to csv
    & $python h5_2_csv.py 2>&1 | Tee-Object

    # 5. fetch lat/lon
    & $python fetch_station_latlon.py 2>&1 | Tee-Object

    # 6. build osrm adjacency
    & $python build_osrm_adj.py 2>&1 | Tee-Object

    # 7. build STGCN input
    & $python build_STGCN_input.py 2>&1 | Tee-Object

    Write-Host "DONE District $d"
}