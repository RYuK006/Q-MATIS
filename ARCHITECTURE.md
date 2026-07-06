# Architecture of the Superconductor Discovery AI

This document details the architectural decisions and internal workings of the GNN-based superconductor discovery pipeline.

## 1. Graph Representation

Traditional machine learning models struggle with crystalline materials because they are invariant to rotations, translations, and permutations of the unit cell. Graph Neural Networks (GNNs) naturally respect these symmetries.

### Node Features (Atoms)
Each atom in the crystal is a node. The features are extracted using `pymatgen` and include:
- **Atomic Number ($Z$)**
- **Atomic Mass**
- **Electronegativity**
- **Group & Period** in the periodic table
- **Valence Electrons**

These features provide a robust baseline of the atom's chemical identity and reactivity.

### Edge Features (Bonds)
Edges represent spatial proximity rather than strict chemical bonds. 
- A cutoff radius (default: $4.0 \text{ \AA}$) is defined.
- If the distance between two atoms is less than the cutoff, an edge is formed.
- The edge feature is simply the 3D physical distance between the atoms.

## 2. Graph Neural Network (GNN) Model

The model is heavily inspired by the **Crystal Graph Convolutional Neural Network (CGCNN)** architecture.

### Message Passing (`CGCNNLayer`)
The network contains 3 to 5 message-passing layers. In each layer:
1. An atom receives "messages" from all its connected neighbors.
2. The message is a learned function of the target atom's features, the neighbor's features, and the distance between them (edge feature).
3. The atom updates its own state by aggregating these messages.

This allows the network to learn complex multi-body interactions and local chemical environments beyond just pairwise bonds.

### Global Pooling & Output
Because a crystal can have an arbitrary number of atoms in its unit cell, the final atomic features are aggregated into a single, fixed-length "crystal vector" using **Global Mean Pooling**.

A fully-connected Feed Forward Network (Regression Head) takes this crystal vector and outputs a single continuous variable: the predicted Critical Temperature ($T_c$) in Kelvin.

## 3. Training and Loss

- **Loss Function**: High-$T_c$ superconductors are exceptionally rare and appear as extreme outliers. A standard Mean Squared Error (MSE) heavily penalizes outliers. Instead, we use **Huber Loss**, which behaves quadratically for small errors and linearly for large errors, ensuring robust learning without ignoring the most critical discoveries.
- **Evaluation**: The model splits the dataset into training and validation sets, tracking both Root Mean Squared Error (RMSE) and $R^2$ scores to measure generalization.

## 4. Active Learning & DFT Handoff

The ultimate goal of this pipeline is discovery, handled by the Active Learning Loop.

1. **Candidate Generation**: The system takes known stable structures and performs specific ionic substitutions (e.g., swapping Y for Ba) to generate theoretically novel materials.
2. **Prediction**: The GNN evaluates the new structure.
3. **Validation Handoff**: If the predicted $T_c$ exceeds a user-defined threshold, the system flags the material. It generates standard input files (like a VASP POSCAR) to be passed to Density Functional Theory (DFT) software. This external step is crucial to verify if the theoretically generated lattice is dynamically stable and to compute actual electron-phonon coupling constants.
