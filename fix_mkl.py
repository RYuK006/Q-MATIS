import json

with open('bete_net_colab.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        for i, line in enumerate(source):
            if line.startswith("!conda create -n bete_net"):
                source[i] = line.replace("cudatoolkit=11.3", "cudatoolkit=11.3 \"mkl<2024\"")

with open('bete_net_colab.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
