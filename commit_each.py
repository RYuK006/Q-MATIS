import subprocess
import os

def commit_and_push():
    status_res = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True)
    lines = [l.strip() for l in status_res.stdout.splitlines() if l.strip()]
    
    print(f"Total items to commit: {len(lines)}")
    committed_count = 0
    
    for l in lines:
        status_code = l[:2].strip()
        f = l[2:].strip().strip('"')
        
        # Determine descriptive commit message
        if f.endswith('.py'):
            msg = f"Update Python module {f}"
        elif f.endswith('.md'):
            msg = f"Update documentation in {f}"
        elif f.endswith('.json'):
            msg = f"Update configuration/data {f}"
        elif f.endswith('.csv'):
            msg = f"Update data table {f}"
        elif f.endswith('.txt'):
            msg = f"Update log/text reference {f}"
        elif f == '.gitignore':
            msg = "Update .gitignore for BETE-NET and local artifacts"
        else:
            msg = f"Add/Update {f}"
            
        print(f"[{committed_count+1}/{len(lines)}] Staging and committing: {f}")
        add_res = subprocess.run(['git', 'add', f], capture_output=True, text=True)
        if add_res.returncode != 0:
            print(f"  Error adding {f}: {add_res.stderr}")
            continue
            
        commit_res = subprocess.run(['git', 'commit', '-m', msg], capture_output=True, text=True)
        if commit_res.returncode == 0:
            print(f"  Committed: {msg}")
            committed_count += 1
        else:
            print(f"  Commit output/warning: {commit_res.stdout or commit_res.stderr}")
            
    print(f"\nSuccessfully created {committed_count} individual commits.")
    
    print("Pushing to origin main...")
    push_res = subprocess.run(['git', 'push', 'origin', 'main'], capture_output=True, text=True)
    print("Push stdout:", push_res.stdout)
    print("Push stderr:", push_res.stderr)
    if push_res.returncode == 0:
        print("Git push completed successfully!")
    else:
        print(f"Git push failed with code {push_res.returncode}")

if __name__ == '__main__':
    commit_and_push()
