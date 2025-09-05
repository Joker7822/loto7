
# bnn_diffusion_es.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Callable, Tuple, Any, Dict
import numpy as np
from early_stopping import EarlyStopping, EarlyStopConfig

def _bnn_val_metric(model, guide, predict_fn: callable, X_val: np.ndarray, y_val: np.ndarray,
                    samples: int = 30, metric: str = "rmse") -> float:
    preds = predict_fn(model, guide, X_val, samples=samples)
    preds = np.asarray(preds, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.float32)
    if metric == "mae":
        return float(np.abs(preds - y_val).mean())
    return float(np.sqrt(np.mean((preds - y_val) ** 2)))

def fit_bnn_with_early_stopping_stepwise(model, guide, step_fn: callable, predict_fn: callable,
                                         X_train, y_train, X_val, y_val,
                                         *, max_epochs=200, patience=15, metric="rmse", samples=30):
    es = EarlyStopping(EarlyStopConfig(patience=patience, min_delta=1e-4, mode="min", restore_best=True))
    for epoch in range(1, max_epochs+1):
        step_fn(model, guide, X_train, y_train)
        val = _bnn_val_metric(model, guide, predict_fn, X_val, y_val, samples=samples, metric=metric)
        print(f"[BNN][Epoch {epoch}] val_{metric}={val:.5f}")
        if es.step(val, model):
            print(f"[BNN] Early stopped. Best val_{metric}={es.best:.5f}")
            break
    return model, guide

def fit_bnn_with_early_stopping_shotgun(train_fn: callable, predict_fn: callable,
                                        X_train, y_train, X_val, y_val, input_size,
                                        *, rounds=50, patience=8, metric="rmse", samples=30):
    es = EarlyStopping(EarlyStopConfig(patience=patience, min_delta=1e-4, mode="min", restore_best=False))
    best_score = np.inf; best_pack = None
    for r in range(1, rounds+1):
        model, guide = train_fn(X_train, y_train, input_size)
        score = _bnn_val_metric(model, guide, predict_fn, X_val, y_val, samples=samples, metric=metric)
        print(f"[BNN][Round {r}] val_{metric}={score:.5f}")
        if score + 1e-12 < best_score:
            best_score = score; best_pack = (model, guide); es.counter = 0
        else:
            es.counter += 1
            if es.counter >= es.cfg.patience:
                print(f"[BNN] Early stopped (shotgun). Best val_{metric}={best_score:.5f}")
                break
    return best_pack if best_pack is not None else train_fn(X_train, y_train, input_size)

def train_diffusion_with_es(train_fn: callable, sample_fn: callable, real_data_bin: np.ndarray,
                            *, max_epochs=5000, patience=200, validate_every=100, proxy_metric="hist_kl"):
    from inspect import signature
    sig = signature(train_fn)
    if "callback" not in sig.parameters:
        print("[Diffusion] callback 未対応: 通常学習にフォールバック")
        return train_fn(real_data_bin)
    es = EarlyStopping(EarlyStopConfig(patience=patience, min_delta=1e-4, mode="min", restore_best=True))
    real_hist = real_data_bin.sum(axis=0).astype(np.float64); real_hist /= max(real_hist.sum(), 1.0)
    def proxy_score(model) -> float:
        try:
            gen = sample_fn(model, dim=real_data_bin.shape[1], num_samples=256)
            if isinstance(gen, tuple): gen = gen[0]
            gen = np.asarray(gen, dtype=np.float64)
            if gen.ndim == 1: gen = gen.reshape(1, -1)
            bin_gen = np.zeros_like(gen); idx = np.argsort(gen, axis=1)[:, -7:]
            for i in range(gen.shape[0]): bin_gen[i, idx[i]] = 1.0
            gh = bin_gen.sum(axis=0); gh /= max(gh.sum(), 1.0)
            eps = 1e-9
            return float(np.sum(real_hist * (np.log(real_hist + eps) - np.log(gh + eps))))
        except Exception as e:
            print(f"[Diffusion] proxy_score 失敗: {e}")
            return 1e9
    def cb(epoch: int, model, losses: Dict[str, float]):
        if epoch % validate_every != 0: return False
        sc = proxy_score(model); print(f"[Diffusion][Epoch {epoch}] proxy({proxy_metric})={sc:.6f}")
        return es.step(sc, model)
    return train_fn(real_data_bin, max_epochs=max_epochs, callback=cb)
