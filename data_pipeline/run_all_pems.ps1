$python = "C:/Users/erays/AppData/Local/Programs/Python/Python313/python.exe"
$root   = "C:/Users/erays/Desktop/pemsdata"

Set-Location $root

# 1) Extract lane closures for 08, 10, 11
$extractDistricts = 8, 10, 11
foreach ($d in $extractDistricts) {
    Write-Host "=== EXTRACT: District $d ==="
    $env:PEMS_DISTRICT_ID = $d
    & $python "$root/extract_lane_closures.py"
}

# 2) Match closures for 05, 06, 08, 10, 11
$matchDistricts = 5, 6, 8, 10, 11
foreach ($d in $matchDistricts) {
    Write-Host "=== MATCH: District $d ==="
    $env:PEMS_DISTRICT_ID = $d
    & $python "$root/match_closure_with_sensors.py"
}
