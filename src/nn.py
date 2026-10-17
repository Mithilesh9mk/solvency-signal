"""Neural network early-warning model for Early Tremor.

A small multilayer perceptron in PyTorch, wrapped with the scikit-learn
fit / predict_proba interface so it plugs into the same training and
evaluation code as the baselines.

Design choices:
- Inputs are median imputed with missing indicators, then each column is
  mapped to a normal distribution by rank (QuantileTransformer). The Polish
  ratios have extreme tails; rank scaling lifted 5-fold train CV PR AUC
  from 0.709 (clip and standard scale, scaling="clip") to 0.758. Chosen on
  train CV only, see reports/models_val.md.
- Early stopping uses an inner holdout carved from the training data
  (stratified 15%, seed 42), never the validation split. The preprocessor
  is fitted on the inner training part only.
- Plain binary cross-entropy, no class reweighting, so probabilities stay
  comparable with the baselines' Brier scores.
- CPU, one thread, torch deterministic algorithms and seeded data order,
  so the same seed gives identical predictions.

All randomness uses SEED = 42.
"""

import copy

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer
from torch import nn

from src.preprocess import build_preprocessor

SEED = 42


class MLP(nn.Module):
    def __init__(self, n_in: int, hidden: tuple, dropout: float):
        super().__init__()
        layers, prev = [], n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class TremorNet(BaseEstimator, ClassifierMixin):
    """MLP classifier with inner-holdout early stopping."""

    def __init__(self, hidden=(64, 32), dropout=0.2, lr=1e-3, weight_decay=1e-4,
                 batch_size=128, max_epochs=300, patience=25, holdout=0.15, scaling="quantile",
                 seed=SEED):
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.holdout = holdout
        self.scaling = scaling
        self.seed = seed

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)
        y = np.asarray(y, dtype=np.float32)
        X_in, X_ho, y_in, y_ho = train_test_split(
            X, y, test_size=self.holdout, stratify=y, random_state=self.seed
        )
        self.pre_ = self._preprocessor().fit(X_in)
        t_in = torch.tensor(self.pre_.transform(X_in), dtype=torch.float32)
        t_ho = torch.tensor(self.pre_.transform(X_ho), dtype=torch.float32)
        y_in_t, y_ho_t = torch.tensor(y_in), torch.tensor(y_ho)

        self.model_ = MLP(t_in.shape[1], tuple(self.hidden), self.dropout)
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()
        gen = torch.Generator().manual_seed(self.seed)

        best, best_state, wait = np.inf, None, 0
        self.history_ = []
        for epoch in range(self.max_epochs):
            self.model_.train()
            for idx in torch.randperm(len(t_in), generator=gen).split(self.batch_size):
                opt.zero_grad()
                loss_fn(self.model_(t_in[idx]), y_in_t[idx]).backward()
                opt.step()
            self.model_.eval()
            with torch.no_grad():
                ho_loss = loss_fn(self.model_(t_ho), y_ho_t).item()
            self.history_.append(ho_loss)
            if ho_loss < best - 1e-5:
                best, best_state, wait = ho_loss, copy.deepcopy(self.model_.state_dict()), 0
                self.best_epoch_ = epoch + 1
            else:
                wait += 1
                if wait >= self.patience:
                    break
        self.model_.load_state_dict(best_state)
        self.model_.eval()
        self.classes_ = np.array([0, 1])
        return self

    def _preprocessor(self):
        """'clip': shared clip, impute, scale. 'quantile': impute, then map each
        column to a normal distribution by rank, which tames heavy tails."""
        if self.scaling == "clip":
            return build_preprocessor()
        if self.scaling == "quantile":
            return Pipeline([
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("quantile", QuantileTransformer(n_quantiles=1000, output_distribution="normal",
                                                 random_state=self.seed)),
            ])
        raise ValueError(f"unknown scaling: {self.scaling}")

    def predict_proba(self, X):
        t = torch.tensor(self.pre_.transform(X), dtype=torch.float32)
        with torch.no_grad():
            p = torch.sigmoid(self.model_(t)).numpy().astype(float)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
