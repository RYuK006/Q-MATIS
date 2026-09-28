$ErrorActionPreference = 'SilentlyContinue'

$python = ".\.venv312\Scripts\python.exe"

# We run 20 trials
$successCount = 0
$failCount = 0

for ($trial = 1; $trial -le 20; $trial++) {
    Write-Host "--- Trial $trial ---"
    
    # Reset DB
    Remove-Item "results\candidates.db" -Force
    Remove-Item "chunk_*.csv" -Force
    
    # Initialize DB with exactly 5 chunks
    & $python -c "
import sqlite3
import os
os.makedirs('results', exist_ok=True)
conn = sqlite3.connect('results/candidates.db')
c = conn.cursor()
c.execute('CREATE TABLE work_queue (formula TEXT PRIMARY KEY, chunk_id INTEGER, status TEXT DEFAULT ''pending'', claimed_at TIMESTAMP)')
for i in range(5):
    c.execute('INSERT INTO work_queue (formula, chunk_id, status) VALUES (?, ?, ?)', (f'formula_{i}', i, 'pending'))
conn.commit()
conn.close()
"
    
    # Launch 10 concurrent workers
    $jobs = @()
    for ($w = 0; $w -lt 10; $w++) {
        $jobs += Start-Job -ScriptBlock {
            Set-Location "C:\Users\Aaron\Desktop\Discovery"
            .\.venv312\Scripts\python.exe scripts\claim_chunk.py
        }
    }
    
    # Wait for all
    Wait-Job -Job $jobs | Out-Null
    
    # Collect output
    $outputs = @()
    foreach ($j in $jobs) {
        $out = Receive-Job -Job $j
        if ($out) {
            $outputs += $out -join " "
        }
    }
    Remove-Job -Job $jobs
    
    # Validate exactly 5 chunks were claimed and 5 processes got "No pending chunks"
    $claims = ($outputs | Where-Object { $_ -match "Claimed chunk" }).Count
    $empty = ($outputs | Where-Object { $_ -match "No pending chunks" }).Count
    
    Write-Host "Claims: $claims, Empty: $empty"
    
    if ($claims -eq 5 -and $empty -eq 5) {
        $successCount++
    } else {
        $failCount++
        Write-Host "FAILED TRIAL OUTPUTS:"
        $outputs | ForEach-Object { Write-Host $_ }
    }
}

Write-Host "`n=== Test Complete ==="
Write-Host "Successful trials: $successCount / 20"
Write-Host "Failed trials: $failCount / 20"
