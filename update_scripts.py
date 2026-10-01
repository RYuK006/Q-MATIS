import os
import glob

count = 0
for f in glob.glob('*.py') + glob.glob('scripts/*.py'):
    with open(f, 'r', encoding='utf-8') as file:
        content = file.read()
    if 'joblib.load(download_if_missing())' in content:
        if 'from download_model import download_if_missing' not in content:
            content = 'from download_model import download_if_missing\n' + content
        content = content.replace('joblib.load(download_if_missing())', 'joblib.load(download_if_missing())')
        with open(f, 'w', encoding='utf-8') as file:
            file.write(content)
        count += 1
print(f'Updated {count} files')
