# Training Pipeline

## Transfer Learning
By freezing or partially unfreezing the encoder, we map robust structural representations from massive generic databases (like the Materials Project) onto our scarce Tc labels.

## Multi-Task Learning
Q-MATIS jointly optimizes for Tc and Formation Energy, sharing the latent representation. Formation Energy acts as an auxiliary regularizer, smoothing the loss landscape.

## Experiment Tracking
All hyperparameters, seeds, and hardware constraints are tracked in metadata.json alongside learning_curves.png.
