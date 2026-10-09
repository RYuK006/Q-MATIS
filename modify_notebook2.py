import json

with open('bete_net_colab.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_cells = []
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and "conda install" in "".join(cell['source']):
        source = cell['source']
        new_source = []
        for line in source:
            if line.startswith("!conda install"):
                new_source.append("# Remove condacolab's pinned specs to avoid SpecsConfigurationConflictError\n")
                new_source.append("!rm -f /usr/local/conda-meta/pinned\n")
            new_source.append(line)
        cell['source'] = new_source
        
        # Also, the user wants to test the pytorch version. Let's add a cell immediately after this one to verify.
        new_cells.append(cell)
        verification_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import torch\n",
                "print(\"PyTorch Version:\", torch.__version__)\n",
                "print(\"CUDA Version:\", torch.version.cuda)\n",
                "assert torch.__version__.startswith('1.10'), 'PyTorch version is not 1.10.0!'\n"
            ]
        }
        new_cells.append(verification_cell)
    else:
        new_cells.append(cell)

nb['cells'] = new_cells

with open('bete_net_colab.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
