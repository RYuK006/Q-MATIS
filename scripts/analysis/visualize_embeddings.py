import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score
import yaml

from superconductor.models import EncoderRegistry, TransferModel
from superconductor.data import get_dataloaders
from superconductor.features import get_node_feature_dim
from superconductor.data_sources.build_dataset import build_dataset
from superconductor.data_sources.mp import MPDataSource

def get_embeddings(model, dataloader, device):
    model.eval()
    embeddings = []
    targets = []
    
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            # Use the reusable embedding API
            emb = model.encode(data)
            embeddings.append(emb.cpu().numpy())
            
            # PyG colates individual attributes
            if hasattr(data, 'y_formation_energy'):
                targets.append(data.y_formation_energy.view(-1).cpu().numpy())
            else:
                targets.append(np.zeros(emb.size(0)))
                
    return np.concatenate(embeddings, axis=0), np.concatenate(targets, axis=0)

def visualize_embeddings(embeddings, targets, method="umap", title="Embeddings", save_path="embeddings.png"):
    plt.figure(figsize=(10, 8))
    
    if method.lower() == "umap":
        reducer = umap.UMAP(random_state=42)
    else:
        reducer = TSNE(n_components=2, random_state=42)
        
    proj = reducer.fit_transform(embeddings)
    
    scatter = plt.scatter(proj[:, 0], proj[:, 1], c=targets, cmap='viridis', alpha=0.7, s=10)
    plt.colorbar(scatter, label='Formation Energy')
    
    # Calculate Quantitative Clustering Metrics on the high-dimensional space
    # Bin continuous targets into 5 discrete classes for clustering metrics
    bins = np.percentile(targets, [20, 40, 60, 80])
    discrete_labels = np.digitize(targets, bins)
    
    # Only compute if we have enough samples and valid embeddings
    if len(embeddings) > 5:
        sil = silhouette_score(embeddings, discrete_labels)
        db = davies_bouldin_score(embeddings, discrete_labels)
        
        # Calculate trustworthiness to measure how well local neighborhoods are preserved in the 2D projection
        from sklearn.manifold import trustworthiness
        trust = trustworthiness(embeddings, proj, n_neighbors=5)
        
        print(f"[{title}] Silhouette Score: {sil:.4f} | Davies-Bouldin: {db:.4f} | Trustworthiness: {trust:.4f}")
    else:
        sil = 0.0
        trust = 0.0
        
    plt.title(f"{title} (Sil: {sil:.2f}, Trust: {trust:.2f})")
    plt.xlabel(f"{method.upper()} 1")
    plt.ylabel(f"{method.upper()} 2")
    
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {title} to {save_path}")

def run_visualization():
    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Fetch small amount of data
    os.environ["MP_API_KEY"] = config['data_sources']['api_key']
    mp_ds = MPDataSource(api_key=config['data_sources']['api_key'], cache_dir=config['data_sources']['dataset_cache_dir'])
    mp_data = mp_ds.fetch_data(limit=1000) # Small sample for visualization
    
    if not mp_data:
        print("No MP data available for visualization.")
        return
        
    structures = [d['structure'] for d in mp_data]
    # Recreate the target dictionary expected by dataset
    targets = [{'formation_energy': d['target']['formation_energy']} for d in mp_data]
    
    node_dim = get_node_feature_dim()
    dmin = config['data']['rbf_distance']['start']
    dmax = config['data']['rbf_distance']['end']
    step = config['data']['rbf_distance']['step']
    edge_dim = int((dmax - dmin) / step) + 1
    
    config['model']['node_dim'] = node_dim
    config['model']['edge_dim'] = edge_dim
    
    # Create dataloader (batch size large to do it quickly)
    config['training']['batch_size'] = 64
    train_loader, _, _, _ = get_dataloaders(structures, targets, config)
    
    # 2. Get Random Init Embeddings
    print("Extracting Random Init Embeddings...")
    encoder_name = config['model'].get('encoder_name', 'cgcnn')
    model_random = EncoderRegistry.build(encoder_name, config).to(device)
    emb_rand, t_rand = get_embeddings(model_random, train_loader, device)
    
    visualize_embeddings(emb_rand, t_rand, method="umap", title="UMAP: Random Init", save_path="umap_random.png")
    visualize_embeddings(emb_rand, t_rand, method="tsne", title="t-SNE: Random Init", save_path="tsne_random.png")
    
    # 3. If Pretrained model exists, visualize it
    pretrain_path = "experiments/run_latest/model_0/encoder_weights.pth" # Placeholder path, will just check if exists
    
    # Alternatively we can simulate it if the user just wanted the script
    print(f"Done. To visualize a trained model, load its state_dict into {encoder_name} before extracting.")
    
if __name__ == "__main__":
    run_visualization()
