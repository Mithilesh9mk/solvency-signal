"""Model evaluation for Early Tremor.

Scores trained models on held-out data with metrics suited to heavy class
imbalance (PR AUC, ROC AUC, recall at fixed false positive rate, Brier
score) and writes tables and figures to reports/.

All randomness uses SEED = 42.
"""

SEED = 42
