
# bnn_diffusion_es.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Callable, Optional, Tuple, Any, Dict
import numpy as np
import torch
from early_stopping import EarlyStopping, EarlyStopConfig

"""
本ファイルは、既存の bnn_module / diffusion_module に “改変なし or 最小改変” で
早期終了(early stopping)を組み込むためのアダプタです。

■ BNN について
- 既存の bnn_module.train_bayesian_regression(...) が「エポック数指定・継続学習」
  に対応していない場合があるため、以下の2モードで動作します。

  1) ステップ学習コールバックが使える（推奨）
     - ユーザーが bnn_module に `train_step(model, guide, X, y)` のような
       「1ステップ更新」関数を追加している前提。
     - 本アダプタは train/val で指標を監視し、patience ベースで停止します。

  2) ショットガン再学習（フォールバック）
     - どうしてもステップ更新が提供されない場合、
       `train_bayesian_regression` を “round” ごとに呼び直し、
       val 指標が改善したモデル/ガイドを保持します（非効率ですが安全）。

■ Diffusion について
- diffusion_module.train_diffusion_ddpm(...) が `callback` 引数を受け取れる設計なら
  そのまま渡してバリデーション指標を監視します。
- 受け取れない場合は、単発学習にフォールバックします。
"""

# -------------------------
# BNN: 予測ヘルパ
# -------------------------
def _bnn_val_metric(model, guide, predict_fn: Callable, X_val: np.ndarray, y_val: np.ndarray,
                    samples: int = 30, metric: str = "rmse") -> float:
    """
    BNN のバリデーション指標を算出。predict_fn は (model, guide, X, samples) -> np.ndarray( N,7 )
    metric: "rmse" | "mae"
    """
    preds = predict_fn(model, guide, X_val, samples=samples)  # shape (N,7) 想定
    preds = np.asarray(preds, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.float32)
    if metric == "mae":
        return float(np.abs(preds - y_val).mean())
    # default rmse
    return float(np.sqrt(np.mean((preds - y_val) ** 2)))

# -------------------------
# BNN: 早期終了アダプタ（推奨: step_fn あり）
# -------------------------
def fit_bnn_with_early_stopping_stepwise(model, guide,
                                         step_fn: Callable[[Any, Any, np.ndarray, np.ndarray], None],
                                         predict_fn: Callable,
                                         X_train: np.ndarray, y_train: np.ndarray,
                                         X_val: np.ndarray, y_val: np.ndarray,
                                         *, max_epochs: int = 200, patience: int = 15,
                                         metric: str = "rmse", samples: int = 30) -> Tuple[Any, Any]:
    """
    step_fn: 1エポック or 1ステップの更新関数 (model, guide, X_train, y_train) -> None
    """
    es = EarlyStopping(EarlyStopConfig(patience=patience, min_delta=1e-4, mode="min", restore_best=True))

    best = None
    for epoch in range(1, max_epochs + 1):
        step_fn(model, guide, X_train, y_train)  # 1更新
        val = _bnn_val_metric(model, guide, predict_fn, X_val, y_val, samples=samples, metric=metric)
        print(f"[BNN][Epoch {epoch}] val_{metric}={val:.5f}")
        if es.step(val, model):  # model の state_dict 復元は効くが、guide の復元手段は bnn_module 側次第
            print(f"[BNN] Early stopped. Best val_{metric}={es.best:.5f}")
            break
        best = val
    return model, guide

# -------------------------
# BNN: 早期終了アダプタ（フォールバック）
# -------------------------
def fit_bnn_with_early_stopping_shotgun(train_fn: Callable,
                                        predict_fn: Callable,
                                        X_train: np.ndarray, y_train: np.ndarray,
                                        X_val: np.ndarray, y_val: np.ndarray,
                                        input_size: int,
                                        *, rounds: int = 50, patience: int = 8,
                                        metric: str = "rmse", samples: int = 30) -> Tuple[Any, Any]:
    """
    train_fn: bnn_module.train_bayesian_regression(X_train, y_train, input_size) -> (model, guide)
    ※ 各 round ごとに “ゼロから” 学習し直す方式（非効率）。
    """
    es = EarlyStopping(EarlyStopConfig(patience=patience, min_delta=1e-4, mode="min", restore_best=False))

    best_score = np.inf
    best_pack = None
    for r in range(1, rounds + 1):
        model, guide = train_fn(X_train, y_train, input_size)
        score = _bnn_val_metric(model, guide, predict_fn, X_val, y_val, samples=samples, metric=metric)
        print(f"[BNN][Round {r}] val_{metric}={score:.5f}")
        if score + 1e-12 < best_score:
            best_score = score
            best_pack = (model, guide)
            es.counter = 0
        else:
            es.counter += 1
            if es.counter >= es.cfg.patience:
                print(f"[BNN] Early stopped (shotgun). Best val_{metric}={best_score:.5f}")
                break
    return best_pack if best_pack is not None else train_fn(X_train, y_train, input_size)

# -------------------------
# Diffusion: 早期終了アダプタ
# -------------------------
def train_diffusion_with_es(train_fn: Callable,    # diffusion_module.train_diffusion_ddpm
                            sample_fn: Callable,   # diffusion_module.sample_diffusion_ddpm
                            real_data_bin: np.ndarray,
                            *, max_epochs: int = 5000, patience: int = 200,
                            validate_every: int = 100,
                            proxy_metric: str = "hist_kl") -> Tuple[Any, Any, Any]:
    """
    train_fn に callback(epoch, model, losses) を渡せる場合、そこから early stop 監視。
    もし受け取れない場合は、単発学習（元の train_fn を1回）へフォールバック。

    proxy_metric:
        - "hist_kl": 生成サンプルのヒストグラム vs 実データヒストの KL を指標にする（低いほど良い）
    """
    # 1) train_fn が callback を受け取れるか判定
    from inspect import signature
    sig = signature(train_fn)
    if "callback" not in sig.parameters:
        # フォールバック（従来通り）
        print("[Diffusion] train_fn に callback がありません。通常学習にフォールバックします。")
        return train_fn(real_data_bin)

    # 2) 監視セットアップ
    es = EarlyStopping(EarlyStopConfig(patience=patience, min_delta=1e-4, mode="min", restore_best=True))

    # 実データの数字ヒストグラム（37次元）
    real_hist = real_data_bin.sum(axis=0).astype(np.float64)
    real_hist = real_hist / max(real_hist.sum(), 1.0)

    def proxy_score(model) -> float:
        # 生成サンプルから分布を推定し、ヒストグラム距離で評価
        try:
            gen = sample_fn(model, dim=real_data_bin.shape[1], num_samples=256)  # (256,37) 連続 or 0/1
            if isinstance(gen, tuple):
                gen = gen[0]
            gen = np.asarray(gen, dtype=np.float64)
            if gen.ndim == 1:
                gen = gen.reshape(1, -1)
            # 上位7つを1にする方式で 0/1 化
            bin_gen = np.zeros_like(gen)
            idx = np.argsort(gen, axis=1)[:, -7:]
            for i in range(gen.shape[0]):
                bin_gen[i, idx[i]] = 1.0
            gh = bin_gen.sum(axis=0)
            gh = gh / max(gh.sum(), 1.0)
            # KL(Real || Gen) の近似（ゼロ割回避）
            eps = 1e-9
            score = float(np.sum(real_hist * (np.log(real_hist + eps) - np.log(gh + eps))))
            return score
        except Exception as e:
            print(f"[Diffusion] proxy_score 失敗: {e}")
            return 1e9  # 悪いスコア

    # 3) コールバック
    def cb(epoch: int, model, losses: Dict[str, float]):
        if epoch % validate_every != 0:
            return False  # 継続
        sc = proxy_score(model)
        print(f"[Diffusion][Epoch {epoch}] proxy({proxy_metric})={sc:.6f}")
        return es.step(sc, model)  # True なら停止

    # 4) 学習実行（train_fn は (real_data_bin, max_epochs, callback) を受ける想定）
    model, betas, alphas_cumprod = train_fn(real_data_bin, max_epochs=max_epochs, callback=cb)

    return model, betas, alphas_cumprod
