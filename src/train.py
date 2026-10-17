"""Model training for Early Tremor.

Trains baseline models (logistic regression, LightGBM) and the neural
network early-warning model on the processed data, then saves the trained
artefacts to models/.

All randomness uses SEED = 42.
"""

SEED = 42
