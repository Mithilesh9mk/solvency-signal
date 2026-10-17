"""Data preprocessing for Early Tremor.

Loads the raw bankruptcy datasets from data/raw (read only), cleans them,
handles missing values, builds features and writes the result to
data/processed. Raw files are never modified.

All randomness uses SEED = 42.
"""

SEED = 42
