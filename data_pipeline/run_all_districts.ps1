# -----------------------------------------
# Run closure scripts for multiple districts
# -----------------------------------------

# List of districts to process
$districts = @(3, 4, 5, 6, 8, 10, 11)

# Paths to your scripts (relative to this .ps1 file)
$hist_script     = "plot_closure_speed_histogram.py"
$analysis_script = "analyze_sensor_closure_patterns.py"

foreach ($d in $districts) {

    Write-Host "============================================="
    Write-Host " Running district $d ..."
    Write-Host "============================================="

    # Set environment variable for this session
    $env:PEMS_DISTRICT_ID = "$d"

    # --- Run histogram script ---
    Write-Host "Running: $hist_script"
    python $hist_script
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error in $hist_script for district $d" -ForegroundColor Red
        continue
    }

    # --- Run sensor closure analysis ---
    Write-Host "Running: $analysis_script"
    python $analysis_script
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error in $analysis_script for district $d" -ForegroundColor Red
        continue
    }

    Write-Host "Done district $d" -ForegroundColor Green
    Write-Host ""
}

Write-Host "============================================="
Write-Host " All districts processed."
Write-Host "============================================="
