Remove-Item -Path "results\candidates.db" -ErrorAction SilentlyContinue
Remove-Item -Path "res_chunk_*.db" -ErrorAction SilentlyContinue
Remove-Item -Path "chunk_*.csv" -ErrorAction SilentlyContinue

Write-Host "--- 1. Generating Candidates & Initializing DB ---"
.\.venv312\Scripts\python.exe scripts\generate_candidates.py

Write-Host "`n--- 2. Worker 1 Claims Chunk ---"
.\.venv312\Scripts\python.exe scripts\claim_chunk.py

Write-Host "`n--- 3. Worker 1 Scores Chunk ---"
.\.venv312\Scripts\python.exe scripts\batch_scorer.py --input chunk_0.csv --output res_chunk_0.db

Write-Host "`n--- 4. Worker 1 Submits Results ---"
.\.venv312\Scripts\python.exe scripts\submit_results.py res_chunk_0.db

Write-Host "`n--- 5. Worker 2 Claims Next Chunk ---"
.\.venv312\Scripts\python.exe scripts\claim_chunk.py

Write-Host "`n--- 6. Worker 2 Scores Next Chunk ---"
.\.venv312\Scripts\python.exe scripts\batch_scorer.py --input chunk_1.csv --output res_chunk_1.db

Write-Host "`n--- 7. Worker 2 Submits Results ---"
.\.venv312\Scripts\python.exe scripts\submit_results.py res_chunk_1.db

Write-Host "`n--- 8. Verification ---"
.\.venv312\Scripts\python.exe -c "
import sqlite3
conn = sqlite3.connect('results/candidates.db')
c = conn.cursor()
c.execute('SELECT status, COUNT(*) FROM work_queue GROUP BY status')
print('Work Queue Status:', c.fetchall())
c.execute('SELECT COUNT(*) FROM predictions')
print('Total Predictions:', c.fetchone()[0])
conn.close()
"
