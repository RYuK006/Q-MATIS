# Scientific Memory Engine

The Scientific Memory Engine is the core infrastructure of Q-MATIS. It is an **append-only scientific ledger** designed to build the world's scientific memory for materials.

## Append-Only Philosophy

Every object and event in Q-MATIS is versioned. 
**Nothing is overwritten. Nothing is deleted. Everything becomes permanent scientific history.**

When an experiment generates a new structure, when a model predicts a property, or when a filter rejects a candidate, that data is locked into the ledger.

## What is Preserved?

The Scientific Memory Engine explicitly stores:
- generated materials
- imported materials
- crystal structures
- crystal graphs
- embeddings
- predicted properties
- experimentally measured properties
- DFT calculations
- literature references
- checkpoints
- configurations
- experiment metadata
- model versions
- training metrics
- uncertainty estimates
- accepted candidates
- rejected candidates
- rejection reasons
- physics filter decisions
- candidate lineage
- parent-child mutations
- timestamps
- hardware metadata
- software versions
- git commits

By rigorously storing every aspect of the research process, **every scientific decision remains reproducible forever.**

## How it Works

Instead of relying on fragmented CSV files or overwritten JSON state, the Scientific Memory Engine is built on top of the Materials Knowledge Graph, a highly relational graph-database schema (backed by SQLite/Parquet) that links every generated material back to the exact experiment, model, and Git commit that produced it.

Future generations of Universal Materials Foundation Models will be trained not just on the final successful outcomes, but on the massive volume of intermediate predictions, failures, and substitutions that led there.
