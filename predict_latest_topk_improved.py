
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# predict_latest_topk_improved.py
# - Apply diversity optimization after stacking (_stable_diverse_selection)
# - Z-score normalization across model match rates to align scales
# - Constraint score weight configurable via --cweight (default 0.10)
# - Keeps I/O compatible with original script

import os
import json
import argparse
import random
import numpy as np
import pandas as pd
from datetime import datetime

from lottery_prediction import (
    LotoPredictor,
    preprocess_data,
    create_advanced_features,
    save_predictions_to_csv,
)

# Diversity selector (provided in lottery_prediction.py)
try:
    from lottery_prediction import _stable_diverse_selection as stable_diverse_selection
    _HAVE_DIVERSE = True
except Exception as e:
    print("[WARN] Could not import diversity selection:", e)
    _HAVE_DIVERSE = False

# Optional constraint score
try:
    from limit_break_predictor import ConstraintConfig, constraint_score
    _HAVE_CONSTRAINT = True
except Exception:
    _HAVE_CONSTRAINT = False

def _ensure_df_with_feature_names(X: np.ndarray, feature_names):
    df = pd.DataFrame(X)
    if feature_names:
        for name in feature_names:
            if name not in df.columns:
                df[name] = 0.0
        df = df[feature_names]
    return df

def _to_top7_numbers_from_vector(vec37: np.ndarray) -> np.ndarray:
    idx = np.argsort(vec37)[-7:] + 1
    return np.sort(idx.astype(int))

def _similarity(candidate: np.ndarray, pred: np.ndarray) -> float:
    return len(set(candidate.tolist()) & set(pred.tolist())) / 7.0

def _normalize_weights(weights: dict, keys: list) -> dict:
    if not weights:
        return {k: 1.0/len(keys) for k in keys}
    w = {k: float(weights.get(k, 0.0)) for k in keys}
    s = sum(v for v in w.values() if np.isfinite(v))
    if s <= 0:
        return {k: 1.0/len(keys) for k in keys}
    return {k: v/s for k, v in w.items()}

def _load_stack_weights(path: str) -> dict:
    if not os.path.exists(path):
        print(f"[WARN] stacking.json not found: {path} -> using equal weights")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        params = json.load(f)
    if isinstance(params, dict):
        return params
    if isinstance(params, list):
        kv = {}
        for it in params:
            if isinstance(it, dict):
                m = it.get("model"); w = it.get("weight")
                if m is not None and w is not None:
                    kv[m] = float(w)
        if kv:
            return kv
    print("[WARN] Unknown stacking.json format -> using equal weights")
    return {}

def _zscore(x: np.ndarray) -> np.ndarray:
    mu = np.nanmean(x); sd = np.nanstd(x)
    if not np.isfinite(sd) or sd <= 1e-9:
        return np.zeros_like(x)
    return (x - mu) / sd

def build_latest_predictions_per_model(
    predictor: LotoPredictor,
    full_df: pd.DataFrame,
    X_latest: np.ndarray,
    num_samples: int = 200,
    noise_level: float = 0.01,
) -> dict:
    preds = {}
    N = num_samples
    X_batch = np.repeat(X_latest.reshape(1, -1), N, axis=0).astype(np.float32)
    try:
        from lottery_prediction import add_noise_to_features as _add_noise
        X_batch = _add_noise(X_batch, noise_level=noise_level)
    except Exception:
        X_batch = X_batch + np.random.normal(0.0, noise_level, size=X_batch.shape).astype(np.float32)

    # LSTM (ONNX)
    try:
        lstm_out = predictor.predict_with_onnx(X_batch.astype(np.float32))
        if lstm_out is not None and np.array(lstm_out).shape[0] == N:
            rows = [np.clip(np.sort(np.round(r).astype(int)), 1, 37) for r in np.array(lstm_out)]
            preds["lstm"] = rows
    except Exception as e:
        print("[WARN] Skip LSTM:", e)

    # AutoGluon
    try:
        X_df = _ensure_df_with_feature_names(X_batch, predictor.feature_names)
        cols = []
        for j in range(7):
            model_j = predictor.regression_models[j]
            if model_j is None:
                raise RuntimeError(f"AutoGluon model pos{j} not trained")
            col = model_j.predict(X_df)
            cols.append(np.asarray(col).reshape(-1, 1))
        automl = np.hstack(cols)
        preds["automl"] = [np.clip(np.sort(np.round(r).astype(int)), 1, 37) for r in automl]
    except Exception as e:
        print("[WARN] Skip AutoGluon:", e)

    # Set Transformer
    try:
        if predictor.set_transformer_model is not None:
            from lottery_prediction import predict_with_set_transformer
            st_out = predict_with_set_transformer(predictor.set_transformer_model, X_batch)
            preds["set_transformer"] = [np.clip(np.sort(np.round(r).astype(int)), 1, 37) for r in st_out]
    except Exception as e:
        print("[WARN] Skip SetTransformer:", e)

    # TabNet
    try:
        if hasattr(predictor, "tabnet_model") and predictor.tabnet_model is not None:
            from tabnet_module import predict_tabnet
            tn = predict_tabnet(predictor.tabnet_model, X_batch)
            preds["tabnet"] = [np.clip(np.sort(np.round(r).astype(int)), 1, 37) for r in tn]
    except Exception as e:
        print("[WARN] Skip TabNet:", e)

    # BNN
    try:
        if hasattr(predictor, "bnn_model") and predictor.bnn_model is not None:
            from bnn_module import predict_bayesian_regression
            bnn_raw = predict_bayesian_regression(predictor.bnn_model, predictor.bnn_guide, X_batch, samples=1)
            bnn_arr = np.array(bnn_raw)
            if bnn_arr.ndim == 3:
                bnn_arr = bnn_arr.mean(axis=0)
            preds["bnn"] = [np.clip(np.sort(np.round(r).astype(int)), 1, 37) for r in bnn_arr]
    except Exception as e:
        print("[WARN] Skip BNN:", e)

    # GAN
    try:
        if predictor.gan_model is not None:
            gan_mat = predictor.gan_model.generate_samples(N)
            rows = []
            for i in range(N):
                v = np.asarray(gan_mat[i], dtype=float)
                v = v / max(1e-9, v.sum())
                rows.append(_to_top7_numbers_from_vector(v))
            preds["gan"] = rows
    except Exception as e:
        print("[WARN] Skip GAN:", e)

    # PPO
    try:
        if predictor.ppo_model is not None:
            import numpy as _np
            rows = []
            for i in range(N):
                obs = _np.zeros(37, dtype=_np.float32)
                action, _ = predictor.ppo_model.predict(obs, deterministic=False)
                rows.append(_to_top7_numbers_from_vector(action))
            preds["ppo"] = rows
    except Exception as e:
        print("[WARN] Skip PPO:", e)

    # Diffusion
    try:
        if predictor.diffusion_model is not None:
            from diffusion_module import sample_diffusion_ddpm
            samples = sample_diffusion_ddpm(
                predictor.diffusion_model,
                predictor.diffusion_betas,
                predictor.diffusion_alphas_cumprod,
                dim=37,
                num_samples=N
            )
            rows = []
            for i in range(N):
                v = np.asarray(samples[i], dtype=float)
                rows.append(_to_top7_numbers_from_vector(v))
            preds["diffusion"] = rows
    except Exception as e:
        print("[WARN] Skip Diffusion:", e)

    # GNN
    try:
        if predictor.gnn_model is not None:
            from gnn_core import build_cooccurrence_graph
            graph = build_cooccurrence_graph(full_df)
            import torch
            predictor.gnn_model.eval()
            with torch.no_grad():
                base = predictor.gnn_model(graph.x, graph.edge_index).squeeze().cpu().numpy()
            rows = []
            rng = np.random.default_rng(123)
            for i in range(N):
                v = base + rng.normal(0, 1e-3, size=base.shape)
                rows.append(_to_top7_numbers_from_vector(v))
            preds["gnn"] = rows
    except Exception as e:
        print("[WARN] Skip GNN:", e)

    # keep only 7-length sets
    for k, v in list(preds.items()):
        preds[k] = [np.array(x, dtype=int) for x in v if isinstance(x, (list, np.ndarray)) and len(x) == 7]
        if not preds[k]:
            preds.pop(k, None)
    return preds

def rank_candidates_with_stacking(
    candidates: list,
    per_model_preds: dict,
    stack_weights: dict,
    use_constraints=True,
    constraint_weight=0.10,
    normalize_mode="zscore",
):
    model_keys = list(per_model_preds.keys())
    weights = _normalize_weights(stack_weights, model_keys)

    # matrix of best match per (candidate, model)
    best_mat = np.zeros((len(candidates), len(model_keys)), dtype=float)
    for mi, m in enumerate(model_keys):
        plist = per_model_preds[m]
        for ci, cand in enumerate(candidates):
            best = 0.0
            for p in plist:
                s = _similarity(cand, p)
                if s > best:
                    best = s
                    if best == 1.0:
                        break
            best_mat[ci, mi] = best

    # normalize across models
    if normalize_mode == "zscore":
        norm_mat = np.vstack([_zscore(best_mat[:, j]) for j in range(best_mat.shape[1])]).T
    elif normalize_mode == "minmax":
        norm_mat = np.zeros_like(best_mat)
        for j in range(best_mat.shape[1]):
            col = best_mat[:, j]
            mn, mx = np.min(col), np.max(col)
            norm_mat[:, j] = (col - mn) / (mx - mn + 1e-9)
    else:
        norm_mat = best_mat

    results = []
    for ci, cand in enumerate(candidates):
        parts = {model_keys[j]: float(best_mat[ci, j]) for j in range(len(model_keys))}
        score = 0.0
        for j, m in enumerate(model_keys):
            score += weights[m] * float(norm_mat[ci, j])

        if use_constraints and _HAVE_CONSTRAINT:
            try:
                cscore = float(constraint_score(cand.tolist(), ConstraintConfig()))
                score += float(constraint_weight) * cscore
                parts["constraint"] = cscore
            except Exception:
                pass

        results.append((cand, float(score), parts))

    results.sort(key=lambda x: x[1], reverse=True)
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="loto7_prediction_evaluation_with_bonus.csv", help="path to history CSV")
    ap.add_argument("--models_dir", default="models/full", help="trained models directory")
    ap.add_argument("--stacking", default="optuna_results/stacking.json", help="stacking weights JSON")
    ap.add_argument("--samples", type=int, default=200, help="num samples per model")
    ap.add_argument("--topk", type=int, default=30, help="top-K before diversity selection")
    ap.add_argument("--noise", type=float, default=0.01, help="feature noise for deterministic models")
    ap.add_argument("--out", default="predictions_topK_improved.csv", help="analysis CSV output")
    ap.add_argument("--cweight", type=float, default=0.10, help="constraint score weight (default 0.10)")
    ap.add_argument("--norm", choices=["zscore", "minmax", "none"], default="zscore", help="normalization mode")
    ap.add_argument("--post_diverse_k", type=int, default=30, help="final K after diversity selection")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"{args.csv} not found.")

    df = pd.read_csv(args.csv, encoding="utf-8-sig")
    X_all, y_all, scaler = preprocess_data(df)
    if X_all is None:
        raise RuntimeError("preprocess failed.")
    X_latest = X_all[-1]

    predictor = LotoPredictor(input_size=X_all.shape[1], hidden_size=128, output_size=7)
    predictor.load_saved_models(args.models_dir)

    per_model_preds = build_latest_predictions_per_model(
        predictor=predictor,
        full_df=df,
        X_latest=X_latest,
        num_samples=args.samples,
        noise_level=args.noise
    )
    if not per_model_preds:
        raise RuntimeError("no predictions from any model.")

    pool = []
    for lst in per_model_preds.values():
        pool.extend([tuple(x.tolist()) for x in lst])
    uniq = sorted(set(pool))
    candidates = [np.array(x, dtype=int) for x in uniq]
    print(f"[INFO] unique candidates: {len(candidates)}")

    stack_weights = _load_stack_weights(args.stacking)
    ranked = rank_candidates_with_stacking(
        candidates,
        per_model_preds,
        stack_weights,
        use_constraints=True,
        constraint_weight=args.cweight,
        normalize_mode=args.norm,
    )

    # diversity selection
    topk = ranked[: args.topk]
    numbers_only = [cand.tolist() for cand, sc, _ in topk]
    confidence_scores = [float(sc) for _, sc, _ in topk]
    if _HAVE_DIVERSE and len(numbers_only) > 0:
        numbers_only = stable_diverse_selection(
            numbers_only, confidence_scores, df, k=min(args.post_diverse_k, len(numbers_only)),
            lambda_div=0.6, temperature=0.35
        )
        # rebuild ordering
        recon = []
        seen = set()
        for nums in numbers_only:
            tpl = tuple(nums)
            if tpl in seen:
                continue
            seen.add(tpl)
            for cand, sc, parts in topk:
                if tuple(cand.tolist()) == tpl:
                    recon.append((cand, sc, parts))
                    break
        topk = recon

    # outputs
    for i, (cand, sc, parts) in enumerate(topk, 1):
        print(f"[{i:02d}] {cand.tolist()}  score={sc:.4f}  parts={{{', '.join(f'{k}:{v:.3f}' for k,v in parts.items())}}}")

    rows = []
    for rank, (cand, sc, parts) in enumerate(topk, 1):
        row = {"rank": rank, "score": float(sc), "numbers": " ".join(map(str, cand.tolist()))}
        row.update({f"s_{k}": float(v) for k, v in parts.items()})
        rows.append(row)
    pd.DataFrame(rows).to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"[INFO] saved analysis CSV: {args.out}")

    drawing_date = datetime.now().strftime("%Y-%m-%d")
    save_list = [(cand.tolist(), float(sc)) for cand, sc, _ in topk[:5]]
    try:
        save_predictions_to_csv(save_list, drawing_date=drawing_date, filename="loto7_predictions_improved.csv")
        print("[INFO] saved standard CSV: loto7_predictions_improved.csv")
    except Exception as e:
        print("[WARN] failed to save standard CSV:", e)

if __name__ == "__main__":
    random.seed(42); np.random.seed(42)
    main()
