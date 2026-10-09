import json

with open('structures/mp-aaabfqpo.cif', 'r') as f:
    cif1 = f.read()

with open('structures/mp-aaacpwsj.cif', 'r') as f:
    cif2 = f.read()

with open('bete_net_colab.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        if "%%writefile BETE-NET/run_inference.py\n" in source:
            cif1_escaped = cif1.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
            cif2_escaped = cif2.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
            
            injection = [
                "import os\n",
                "os.makedirs('structures', exist_ok=True)\n",
                f"with open('structures/mp-aaabfqpo.cif', 'w') as f: f.write(\"{cif1_escaped}\")\n",
                f"with open('structures/mp-aaacpwsj.cif', 'w') as f: f.write(\"{cif2_escaped}\")\n\n"
            ]
            
            # Find the import os line and insert after it
            for i, line in enumerate(source):
                if line.startswith('import json'):
                    new_source = source[:i+1] + injection + source[i+1:]
                    cell['source'] = new_source
                    break

with open('bete_net_colab.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
