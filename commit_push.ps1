$status = git status --porcelain
foreach ($line in $status) {
    # Skip ignored files or empty lines
    if ([string]::IsNullOrWhiteSpace($line)) { continue }
    
    $file = $line.Substring(3).Trim()
    
    # We ignore the very large things via gitignore, but let's double check
    if ($file -match '\.(parquet|joblib|bz2|gz|csv|db|txt)$') {
        continue
    }

    if ($file -notmatch '\.(py|md)$') {
        continue
    }

    Write-Host "Committing $file"
    git add $file
    git commit -m "Update $file for MACE generative materials pilot"
}

git push origin main
