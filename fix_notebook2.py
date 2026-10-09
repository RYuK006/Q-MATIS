import json

with open('bete_net_colab.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        for i, line in enumerate(source):
            if "scikit-learn==1.6.1" in line and "numpy<2" not in line:
                source[i] = line.replace("scikit-learn==1.6.1", "scikit-learn==1.6.1 \"numpy<2\"")

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        if "%%writefile BETE-NET/run_inference.py\n" in source:
            new_source = []
            skip = False
            for line in source:
                if line.startswith("original_df = pd.read_json"):
                    skip = True
                if skip and line.startswith("print(\"Training set num_neighbors:"):
                    skip = False
                    continue
                if not skip:
                    new_source.append(line)
            
            # Inject the dynamic num_neighbors computation after build_data
            for i, line in enumerate(new_source):
                if line.startswith("df['data'] = df.apply(build_data"):
                    new_source.insert(i+1, "train_num_neighbors = get_neighbors(df, df.index).mean()\n")
                    new_source.insert(i+2, "print(\"Computed test set num_neighbors:\", train_num_neighbors)\n")
                    break
            cell['source'] = new_source

with open('bete_net_colab.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
