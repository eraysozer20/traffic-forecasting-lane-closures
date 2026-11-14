$python = "C:/Users/erays/AppData/Local/Programs/Python/Python313/python.exe"
$root   = "C:/Users/erays/Desktop/pemsdata"

Set-Location $root

# --- District sets ---
$extract = @(11)
$match   = @(11)

# --- Small function to run Python with district variable ---
function Run-ForDistrict([string]$script, [int]$d) {
    Write-Host "`n=== $script : District $d ==="
    $env:PEMS_DISTRICT_ID = $d
    & $python "$root/$script"
}

# --- 1) Extract lane closures ---
foreach ($d in $extract) {
    Run-ForDistrict "analyze_closure_impacts.py" $d
}

# --- 2) Match closures with sensors ---
foreach ($d in $match) {
    Run-ForDistrict "plot_closure_speed_histogram.py" $d
}

Write-Host "`nAll tasks completed."
