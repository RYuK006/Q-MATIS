# Model Architecture

Q-MATIS heavily relies on Graph Neural Networks (GNNs) because standard ML architectures fail to capture periodic non-Euclidean structures.

## 1. CGCNN (Crystal Graph Convolutional Neural Networks)
Our baseline GNN. Uses simple atomic node passing with bond length edges.

## 2. ALIGNN (Atomistic Line Graph Neural Network)
Our native PyTorch Geometric adaptation of ALIGNN. It constructs a Line Graph from the original graph to capture multi-body bond angles implicitly through edge-to-edge message passing.

## 3. Deep Ensembles
We aggregate predictions from multiple parallel networks (each with random initialization seeds) to quantify epistemic uncertainty (PICP, MPIW).
