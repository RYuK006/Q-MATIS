# Contributing to Q-MATIS

Thank you for your interest in contributing to Q-MATIS! We are building the world's scientific memory for materials, and we welcome contributions from the global community.

## Our Philosophy

Before contributing, please read our [Project Philosophy](PROJECT_PHILOSOPHY.md). 

When writing code or designing schemas for Q-MATIS, remember:
1. **Preserve Reproducibility:** Every scientific decision must remain reproducible forever. Tie computations to exact commits, configurations, and random seeds.
2. **Append-Only Data Practices:** We never delete or overwrite data. The `MaterialsLake` is an append-only ledger. If a prediction is updated, append a new prediction event rather than overwriting the old one. Negative results and failures must be logged, as they are essential training data for future foundation models.

## How to Contribute

1. **Fork the Repository:** Create your own fork and work on a feature branch.
2. **Adhere to the Architecture:** All new modules must integrate seamlessly with the `ResearchExecutionEngine` and `MaterialsLake`. Do not create isolated workflows that bypass the core orchestration and provenance tracking.
3. **Write Tests:** Ensure all tests pass. We run strict `PRAGMA foreign_keys = ON;` in SQLite to ensure data integrity; your tests must properly insert parent entities (Experiments, Checkpoints, Materials) to satisfy these constraints.
4. **Submit a Pull Request:** Describe the scientific rationale and architectural impact of your changes.

Together, we can build permanent scientific infrastructure capable of supporting future AI systems and materials discovery across every scientific discipline.
