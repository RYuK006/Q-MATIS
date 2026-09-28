Remove-Item -Path "results\candidates.db" -ErrorAction SilentlyContinue
.\.venv312\Scripts\python.exe scripts\generate_candidates.py | Out-Null

Write-Host "Running two claim_chunk processes concurrently..."

# Start both as background jobs
$job1 = Start-Job -ScriptBlock { 
    Set-Location -Path "C:\Users\Aaron\Desktop\Discovery"
    .\.venv312\Scripts\python.exe scripts\claim_chunk.py 
}
$job2 = Start-Job -ScriptBlock { 
    Set-Location -Path "C:\Users\Aaron\Desktop\Discovery"
    .\.venv312\Scripts\python.exe scripts\claim_chunk.py 
}

# Wait for both to finish
Wait-Job -Job $job1, $job2 | Out-Null

# Receive their output
Write-Host "--- Output from Process 1 ---"
Receive-Job -Job $job1

Write-Host "--- Output from Process 2 ---"
Receive-Job -Job $job2

Remove-Job -Job $job1, $job2
