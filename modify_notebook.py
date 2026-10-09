import json

with open('bete_net_colab.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_cell_source = [
    "# Calculate Tc from the ensemble predictions\n",
    "import numpy as np\n",
    "\n",
    "Freq_final = np.arange(0.25, 101, 2)\n",
    "\n",
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
    "\n",
    "def cal_w_log(freq_w, alpha_F, lamb):\n",
    "    w_logF = 0\n",
    "    try:\n",
    "        for i in range(1, len(freq_w)):\n",
    "            dw = freq_w[i] - freq_w[i-1]\n",
    "            w_logF = w_logF + (alpha_F[i]*np.log(freq_w[i])*dw/freq_w[i])\n",
    "        return np.exp(2*w_logF/lamb)\n",
    "    except:\n",
    "        return np.nan\n",
    "\n",
    "def cal_tc(lamb, omega_log, mu=0.09):\n",
    "    frac = -1.04*(1+lamb)/(lamb-mu*(1+0.62*lamb))\n",
    "    return (omega_log/1.2)*np.exp(frac)\n",
    "\n",
    "# Average the 100 predictions for each row\n",
    "folds = range(100)\n",
    "def get_avg(row):\n",
    "    pred = np.zeros(51)\n",
    "    for i in folds:\n",
    "        pred += row[f'pred_{i}']\n",
    "    return pred / len(folds)\n",
    "\n",
    "df['pred_avg'] = df.apply(get_avg, axis=1)\n",
    "\n",
    "# Calculate lambda, omega_log, and Tc\n",
    "df['lamb_pred'] = df.apply(lambda row: cal_lamb(Freq_final, row['pred_avg']), axis=1)\n",
    "df['wlog_pred'] = df.apply(lambda row: cal_w_log(Freq_final, row['pred_avg'], row['lamb_pred']) / 0.08617, axis=1)\n",
    "df['Tc_pred'] = df.apply(lambda row: cal_tc(row['lamb_pred'], row['wlog_pred']), axis=1)\n",
    "\n",
    "# Print the results explicitly\n",
    "for index, row in df.iterrows():\n",
    "    print(f\"Candidate {index}: lambda = {row['lamb_pred']:.3f}, w_log = {row['wlog_pred']:.2f} K, Tc = {row['Tc_pred']:.3f} K\")\n"
]

new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": new_cell_source
}

nb['cells'].append(new_cell)

with open('bete_net_colab.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
