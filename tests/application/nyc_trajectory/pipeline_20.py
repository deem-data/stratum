import numpy as np
import skrub
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from skorch import NeuralNetBinaryClassifier
from skorch.callbacks import LRScheduler

from common import dense_scaled, features_v15, load_xy

DESCRIPTION = ("model family: MLP (skorch, GPU) on pipeline_15 features, dense scaled "
               "input; 3x hidden + BatchNorm + SiLU + dropout, AdamW, cosine LR "
               "(fused sweep: width x epochs)")
PARENT = "pipeline_15"


class MLP(nn.Module):
    def __init__(self, n_features, hidden=256, n_layers=3, dropout=0.1):
        super().__init__()
        layers, d = [], n_features
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.BatchNorm1d(hidden), nn.SiLU(),
                       nn.Dropout(dropout)]
            d = hidden
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, 1)

    def forward(self, x):
        return self.head(self.body(x)).squeeze(-1)


class SkorchMLP(ClassifierMixin, BaseEstimator):
    """Binary MLP; module sized from the data inside fit (per fold)."""

    def __init__(self, hidden=256, n_layers=3, dropout=0.1, lr=1e-3, max_epochs=15,
                 batch_size=4096, weight_decay=1e-5, random_state=0):
        self.hidden, self.n_layers, self.dropout = hidden, n_layers, dropout
        self.lr, self.max_epochs, self.batch_size = lr, max_epochs, batch_size
        self.weight_decay, self.random_state = weight_decay, random_state

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        self.classes_ = np.array([0, 1])
        self.net_ = NeuralNetBinaryClassifier(
            module=MLP, module__n_features=X.shape[1], module__hidden=self.hidden,
            module__n_layers=self.n_layers, module__dropout=self.dropout,
            max_epochs=self.max_epochs, lr=self.lr, batch_size=self.batch_size,
            optimizer=torch.optim.AdamW, optimizer__weight_decay=self.weight_decay,
            callbacks=[LRScheduler(policy=torch.optim.lr_scheduler.CosineAnnealingLR,
                                   T_max=self.max_epochs)],
            iterator_train__shuffle=True, device="cpu", train_split=None, verbose=0)
        self.net_.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.net_.predict_proba(np.array(X, dtype=np.float32))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


X, y, viol = load_xy()
feats = features_v15(X, viol)
model = SkorchMLP(hidden=skrub.choose_from([256, 512], name="hidden"),
                  max_epochs=skrub.choose_from([1, 3], name="epochs"))
pred = dense_scaled(feats).skb.apply(model, y=y)
