"""Forecasting models. Every model predicts log realised variance of the next period and
returns a LEVEL forecast exp(pred) * smearing, where the smearing factor (Duan 1983) is
estimated on training data only:
    - HAR (OLS, 4 parameters): in-sample residuals.
    - GBM / LSTM: residuals on a chronological 20% validation tail of the training window,
      which is also used for early stopping / epoch selection (no peeking at the test block).
GARCH models forecast the conditional variance of the next daily return directly.
Hyper-parameters are fixed a priori and were never tuned on the evaluation period.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import torch
from arch import arch_model
from sklearn.ensemble import HistGradientBoostingRegressor
from torch import nn

from .config import SEED


def smear(resid: np.ndarray) -> float:
    return float(np.mean(np.exp(resid)))


# ---------------------------------------------------------------- HAR
def har_fit_predict(Xtr, ytr, Xte):
    A = np.c_[np.ones(len(Xtr)), Xtr]
    beta, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    s = smear(ytr - A @ beta)
    return np.exp(np.c_[np.ones(len(Xte)), Xte] @ beta) * s


def naive_fit_predict(Xtr, ytr, Xte):
    """Random-walk benchmark: tomorrow's RV = today's RV (first column must be lrv)."""
    return np.exp(Xte[:, 0])


# ---------------------------------------------------------------- GBM
GBM_PARAMS = dict(max_iter=600, learning_rate=0.03, max_leaf_nodes=8, min_samples_leaf=30,
                  l2_regularization=5.0, early_stopping=False, random_state=SEED)


def _split_tail(n, frac=0.2):
    return int(n * (1 - frac))


def gbm_fit_predict(Xtr, ytr, Xte):
    """Gradient boosted trees (sklearn HistGradientBoosting; LightGBM-style histogram trees)."""
    k = _split_tail(len(Xtr))
    m = HistGradientBoostingRegressor(**GBM_PARAMS).fit(Xtr[:k], ytr[:k])
    val_mse = [np.mean((ytr[k:] - p) ** 2) for p in m.staged_predict(Xtr[k:])]
    best = int(np.argmin(val_mse)) + 1
    s = smear(ytr[k:] - list(m.staged_predict(Xtr[k:]))[best - 1])
    final = HistGradientBoostingRegressor(**{**GBM_PARAMS, "max_iter": max(best, 20)}).fit(Xtr, ytr)
    return np.exp(final.predict(Xte)) * s


# ---------------------------------------------------------------- LSTM
class _Net(nn.Module):
    def __init__(self, d, h=16):
        super().__init__()
        self.lstm = nn.LSTM(d, h, batch_first=True)
        self.drop = nn.Dropout(0.2)
        self.head = nn.Linear(h, 1)

    def forward(self, x):
        o, _ = self.lstm(x)
        return self.head(self.drop(o[:, -1])).squeeze(-1)


def _train_lstm(Xs, ys, Xv, yv, epochs, seed, patience=None):
    torch.manual_seed(seed)
    net = _Net(Xs.shape[-1])
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
    Xs_t, ys_t = torch.tensor(Xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32)
    best, best_ep, best_state, bad = np.inf, 0, None, 0
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Xs_t), generator=g)
        for i in range(0, len(perm), 64):
            b = perm[i:i + 64]
            opt.zero_grad()
            loss = nn.functional.mse_loss(net(Xs_t[b]), ys_t[b])
            loss.backward()
            opt.step()
        if Xv is not None:
            net.eval()
            with torch.no_grad():
                v = nn.functional.mse_loss(net(torch.tensor(Xv, dtype=torch.float32)),
                                           torch.tensor(yv, dtype=torch.float32)).item()
            if v < best - 1e-5:
                best, best_ep, bad = v, ep + 1, 0
                best_state = {k: t.clone() for k, t in net.state_dict().items()}
            else:
                bad += 1
                if patience and bad >= patience:
                    break
    return net, best_ep, best_state


def _predict(net, X):
    net.eval()
    with torch.no_grad():
        return net(torch.tensor(X, dtype=torch.float32)).numpy()


def lstm_fit_predict(Xtr, ytr, Xte, n_seeds=3, max_epochs=40):
    """Xtr/Xte are 3-D (rows, seq_len, features). Standardisation uses training rows only."""
    flat = Xtr.reshape(-1, Xtr.shape[-1])
    mu, sd = flat.mean(0), flat.std(0) + 1e-8
    ym, ys = ytr.mean(), ytr.std()
    f = lambda X: (X - mu) / sd
    Xtr_s, Xte_s, y_s = f(Xtr), f(Xte), (ytr - ym) / ys
    k = _split_tail(len(Xtr_s))
    preds, sm = [], []
    for seed in range(n_seeds):
        net, best_ep, state = _train_lstm(Xtr_s[:k], y_s[:k], Xtr_s[k:], y_s[k:], max_epochs,
                                          SEED + seed, patience=6)
        net.load_state_dict(state)
        sm.append(smear(ytr[k:] - (_predict(net, Xtr_s[k:]) * ys + ym)))
        full, _, _ = _train_lstm(Xtr_s, y_s, None, None, max(best_ep, 3), SEED + seed)
        preds.append(_predict(full, Xte_s) * ys + ym)
    return np.exp(np.mean(preds, axis=0)) * float(np.mean(sm))


# ---------------------------------------------------------------- GARCH
def garch_forecast(ret_pct: np.ndarray, n_train: int, gjr: bool) -> np.ndarray:
    """Conditional variance forecast of day i+1's return, for every i >= n_train - 1 ... end.

    Parameters are estimated by MLE (Student-t innovations) on ret_pct[:n_train] only.
    The variance recursion is then run forward with FIXED parameters; the forecast issued at
    index i uses returns up to and including i. Returns array aligned with ret_pct (NaN
    before n_train); units: percent^2.
    """
    tr = ret_pct[:n_train]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = arch_model(tr, mean="Constant", vol="GARCH", p=1, o=1 if gjr else 0, q=1,
                         dist="t", rescale=False).fit(disp="off", show_warning=False)
    p = res.params
    mu, om, al, be = p["mu"], p["omega"], p["alpha[1]"], p["beta[1]"]
    ga = p["gamma[1]"] if gjr else 0.0
    eps = ret_pct - mu
    s2 = np.empty(len(ret_pct) + 1)
    s2[0] = np.var(eps[:n_train])
    out = np.full(len(ret_pct), np.nan)
    for i in range(len(ret_pct)):
        e2 = eps[i] ** 2
        s2[i + 1] = om + (al + ga * (eps[i] < 0)) * e2 + be * s2[i]
        out[i] = s2[i + 1]
    out[: n_train - 1] = np.nan
    return out
