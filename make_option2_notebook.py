import json

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# BETE-NET 2-Structure Test (Subprocess Method)\n",
            "This notebook uses condacolab to install Miniforge, then explicitly creates a separate conda environment `bete_net` to completely bypass Colab's base environment restrictions and `LD_LIBRARY_PATH` reset bugs."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Install condacolab (This will restart the kernel when finished. DO NOT use 'Run All' across this cell!)\n",
            "!pip install -q condacolab\n",
            "import condacolab\n",
            "condacolab.install()"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Create the isolated bete_net environment (avoids Condacolab base pin conflicts)\n",
            "!conda create -n bete_net python=3.9 pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge -y"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# VERIFICATION STEP (CRITICAL): Ensure the correct versions are active in the environment\n",
            "!conda run -n bete_net python -c \"import torch; print('PyTorch Version:', torch.__version__); print('CUDA Version:', torch.version.cuda); assert torch.__version__.startswith('1.10'), 'Version Mismatch!'\""
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Install remaining dependencies inside the isolated environment\n",
            "!conda run -n bete_net pip install ase==3.22.0 pymatgen==2024.8.9 scikit-learn==1.6.1\n",
            "!conda run -n bete_net pip install e3nn==0.4.2 torch-cluster==1.5.9 torch-scatter==2.0.9 torch-sparse==0.6.12 torch-spline-conv==1.2.1 torch-geometric==2.0.2 -f https://pytorch-geometric.com/whl/torch-1.10.0+cu113.html"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "!git clone https://github.com/henniggroup/BETE-NET.git\n",
            "import os\n",
            "os.makedirs('BETE-NET/structures', exist_ok=True)\n",
            "print(\"Please upload mp-aaabfqpo.cif and mp-aaacpwsj.cif to the /content/BETE-NET/structures folder on the left pane.\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "%%writefile BETE-NET/run_inference.py\n",
            "import os\n",
            "import json\n",
            "import time\n",
            "import torch\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import ase.io\n",
            "import torch_geometric as tg\n",
            "from notebooks.utils.data import build_data, get_target, get_neighbors\n",
            "from notebooks.utils.training import get_model\n",
            "\n",
            "original_df = pd.read_json('database.json')\n",
            "train_num_neighbors = get_neighbors(original_df, original_df.index).mean()\n",
            "print(\"Training set num_neighbors:\", train_num_neighbors)\n",
            "\n",
            "test_candidates = [\"mp-aaabfqpo\", \"mp-aaacpwsj\"]\n",
            "data_dicts = []\n",
            "for idx in test_candidates:\n",
            "    data_dicts.append({\"index\": idx, \"target\": np.zeros(51).tolist()})\n",
            "df = pd.DataFrame(data_dicts)\n",
            "df.to_json('database.json')\n",
            "df.set_index('index', inplace=True)\n",
            "\n",
            "structures = []\n",
            "for index, row in df.iterrows():\n",
            "    structures.append(ase.io.read(f'structures/{index}.cif'))\n",
            "df['structure'] = structures\n",
            "\n",
            "df['data'] = df.apply(build_data, embed_ph_dos=False, embed_e_dos=False, fine=False, r_max=4, axis=1)\n",
            "\n",
            "device = \"cuda:0\" if torch.cuda.is_available() else \"cpu\"\n",
            "out_dim = 51\n",
            "in_dim = len(df.iloc[0].data.x[0])\n",
            "em_dim = 64\n",
            "\n",
            "init_dict_base = dict(in_dim=118, em_dim=em_dim, irreps_in=str(em_dim)+\"x0e\", \n",
            "    irreps_out=str(out_dim)+\"x0e\", irreps_node_attr=str(em_dim)+\"x0e\", \n",
            "    layers=2, mul=32, lmax=1, max_radius=4, \n",
            "    num_neighbors=train_num_neighbors, reduce_output=True, p=0.0)\n",
            "\n",
            "dataloader = tg.loader.DataLoader(df['data'].values, batch_size=900)\n",
            "start_time = time.time()\n",
            "for k in range(100):\n",
            "    name = f\"model_cso_{k}.pt\"\n",
            "    run_name = f'CSO/{name}'\n",
            "    model, opt, scheduler = get_model(init_dict_base, device=device)\n",
            "    model.load_state_dict(torch.load(run_name, map_location=device))\n",
            "    model.pool = True\n",
            "    model.to(device)\n",
            "    model.eval()\n",
            "    df[f'pred_{k}'] = np.empty((len(df), 1)).tolist()\n",
            "    with torch.no_grad():\n",
            "        for i, d in enumerate(dataloader):\n",
            "            d.to(device)\n",
            "            output = model(d)\n",
            "            df[f'pred_{k}'] = [val for val in output.cpu().numpy()]\n",
            "print(f\"Ensemble inference took {time.time() - start_time:.2f} seconds\")\n",
            "\n",
            "Freq_final = np.arange(0.25, 101, 2)\n",
            "def cal_lamb(freq_w, alpha_F):\n",
            "    lambdaF = 0\n",
            "    try:\n",
            "        for i in range(1, len(freq_w)):\n",
            "            dw = freq_w[i] - freq_w[i-1]\n",
            "            w = freq_w[i]\n",
            "            alpha_F_w = alpha_F[i]\n",
            "            lambdaF = lambdaF + ((alpha_F_w/w)*dw)\n",
            "        return 2*lambdaF\n",
            "    except:\n",
            "        return np.nan\n",
            "def cal_w_log(freq_w, alpha_F, lamb):\n",
            "    w_logF = 0\n",
            "    try:\n",
            "        for i in range(1, len(freq_w)):\n",
            "            dw = freq_w[i] - freq_w[i-1]\n",
            "            w_logF = w_logF + (alpha_F[i]*np.log(freq_w[i])*dw/freq_w[i])\n",
            "        return np.exp(2*w_logF/lamb)\n",
            "    except:\n",
            "        return np.nan\n",
            "def cal_tc(lamb, omega_log, mu=0.09):\n",
            "    frac = -1.04*(1+lamb)/(lamb-mu*(1+0.62*lamb))\n",
            "    return (omega_log/1.2)*np.exp(frac)\n",
            "folds = range(100)\n",
            "def get_avg(row):\n",
            "    pred = np.zeros(51)\n",
            "    for i in folds:\n",
            "        pred += row[f'pred_{i}']\n",
            "    return pred / len(folds)\n",
            "df['pred_avg'] = df.apply(get_avg, axis=1)\n",
            "df['lamb_pred'] = df.apply(lambda row: cal_lamb(Freq_final, row['pred_avg']), axis=1)\n",
            "df['wlog_pred'] = df.apply(lambda row: cal_w_log(Freq_final, row['pred_avg'], row['lamb_pred']) / 0.08617, axis=1)\n",
            "df['Tc_pred'] = df.apply(lambda row: cal_tc(row['lamb_pred'], row['wlog_pred']), axis=1)\n",
            "for index, row in df.iterrows():\n",
            "    print(f\"\\nCandidate {index}: lambda = {row['lamb_pred']:.3f}, w_log = {row['wlog_pred']:.2f} K, Tc = {row['Tc_pred']:.3f} K\")\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Run the inference script in the bete_net environment\n",
            "!cd BETE-NET && conda run -n bete_net python run_inference.py"
        ]
    }
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open('bete_net_colab.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
