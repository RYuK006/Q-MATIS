# The Universal Materials Lake

The Materials Lake is the central, unified data repository for Q-MATIS. It is intended to become a universal repository containing every known property of every material.

## Design and Architecture

The Materials Lake is not a collection of fragmented CSVs. It is a highly structured, relational repository built around the concept of a `MaterialEntity`. New information is continually appended rather than replacing existing values.

## What it Contains

The Materials Lake aggregates data across every dimension of materials science:

### Properties
- Structural
- Chemical
- Electronic
- Mechanical
- Magnetic
- Thermal
- Optical
- Transport
- Topological
- Phonon
- Elastic
- Thermodynamic

### Sources
- Experimental
- Computational
- Prediction

### Metadata & Provenance
- Provenance
- Embeddings
- Decision history
- Lineage
- Model history
- Version history
- Confidence
- Uncertainty
- Future measurements

## Entity Lifecycle & Versioning

1. **Instantiation:** A material is either imported from a public database (like Materials Project or OQMD) or generated via substitution/mutation. It is instantly assigned a persistent UUID.
2. **Filtering:** The Candidate Generation Engine applies domain-knowledge filters. If the material fails, the exact failure reason and intermediate mathematical values are written to the `physics_audits` table.
3. **Prediction:** A neural network predicts properties for the material. The exact prediction, alongside uncertainty estimates and a link to the model checkpoint that generated it, are logged in `predictions`.
4. **Validation:** If the material proceeds to DFT validation or physical synthesis, those hard ground-truth results are appended as new `properties`.

At no point in this lifecycle is a previous state overwritten. The Materials Lake continuously grows, serving as the raw training data for the next generation of Foundation Models.
