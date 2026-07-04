# Data Pipeline

Q-MATIS standardizes crystal topologies using the DataOrchestrator.

1. **Materials Project**: Supplies relaxed 3D CIFs and Formation Energy.
2. **SuperCon**: Supplies empirical Tc measurements.
3. **SQLite Caching**: Ensures reproducible dataset loading.

## Graph Construction
Nodes are mapped using 92-dimensional physicochemical descriptors.
Edges encode spatial distances via a Gaussian RBF expansion from 0 to 8 Ångströms.
