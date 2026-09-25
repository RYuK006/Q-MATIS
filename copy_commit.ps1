$source = "C:\Users\Aaron\Desktop\Q-MATIS H1 001"
$dest = "C:\Users\Aaron\Desktop\Discovery\Testing_Q-MATIS"

New-Item -ItemType Directory -Force -Path $dest

$files = Get-ChildItem -Path $source -Recurse -File | Where-Object {
    $_.FullName -notmatch "\\\.venv" -and 
    $_.FullName -notmatch "\\__pycache__" -and 
    $_.FullName -notmatch "\\\.git"
}

foreach ($file in $files) {
    $relativePath = $file.FullName.Substring($source.Length + 1)
    $destPath = Join-Path $dest $relativePath
    $destDir = Split-Path $destPath -Parent
    
    if (-not (Test-Path $destDir)) {
        New-Item -ItemType Directory -Force -Path $destDir | Out-Null
    }
    
    Copy-Item -Path $file.FullName -Destination $destPath -Force
    
    git add $destPath
    $msg = "Add $relativePath to Testing_Q-MATIS"
    git commit -m "$msg" | Out-Null
}

git push
