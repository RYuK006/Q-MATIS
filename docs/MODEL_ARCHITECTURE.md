# Model Architecture: Towards a Universal Encoder

Q-MATIS relies on Graph Neural Networks (GNNs) because standard ML architectures fail to capture periodic non-Euclidean structures. Our long-term objective is a single, unified Foundation Encoder for all crystalline matter.

## 1. Stepping Stones: CGCNN and ALIGNN
Our current implementations serve as foundations:
- **CGCNN (Crystal Graph Convolutional Neural Networks)**: Uses simple atomic node passing with bond length edges.
- **ALIGNN (Atomistic Line Graph Neural Network)**: Constructs a Line Graph from the original graph to capture multi-body bond angles implicitly through edge-to-edge message passing.

## 2. Universal Property Prediction Heads
The central encoder outputs a high-dimensional latent embedding that is fed into specialized, multi-domain property prediction heads (e.g., Formation Energy, Tc, Band Gap, Catalytic Activity). 

## 3. Deep Ensembles & Uncertainty
We aggregate predictions from multiple parallel networks to quantify epistemic uncertainty (PICP, MPIW). This uncertainty metric is permanently logged in the Scientific Memory Engine alongside the prediction, ensuring all AI hypotheses are bounded by statistical confidence limits.
