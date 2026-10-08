import json
with open('bete_net_colab.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        src = "".join(cell['source'])
        if '%%writefile BETE-NET/run_inference.py' in src:
            cell['source'] = [
                "!wget -q https://raw.githubusercontent.com/RYuK006/Q-MATIS/main/bete_inference.py -O BETE-NET/bete_inference.py\n",
                "print('Downloaded fixed inference script!')"
            ]
        elif 'conda run -n bete_net python run_inference.py' in src:
            cell['source'] = [
                "# Run the fixed inference script in the bete_net environment\n",
                "!cd BETE-NET && conda run -n bete_net python bete_inference.py"
            ]

with open('bete_net_colab.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
