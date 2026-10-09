import json

with open('bete_net_colab.ipynb', 'r') as f:
    nb = json.load(f)

# The cells we want to modify are the last 3 cells (indices 5, 6, 7 typically)
# Let's just recreate the last cells

# First 4 cells are setup
new_cells = nb['cells'][:5]

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "!git clone https://github.com/henniggroup/BETE-NET.git\n",
        "!git clone https://github.com/RYuK006/Q-MATIS.git\n",
        "import os\n",
        "os.makedirs('BETE-NET/structures', exist_ok=True)\n",
        "!cp Q-MATIS/cifs_for_colab/*.cif BETE-NET/structures/\n",
        "!cp Q-MATIS/bete_batch_run.py BETE-NET/bete_batch_run.py\n",
        "print('Prepared structures and script!')"
    ]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Run the batch inference script in the bete_net environment\n",
        "!cd BETE-NET && conda run -n bete_net python bete_batch_run.py"
    ]
})

nb['cells'] = new_cells

with open('bete_net_colab.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("Modified bete_net_colab.ipynb")
