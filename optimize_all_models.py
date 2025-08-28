
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_latest_topk.py
- 最新抽せん日に対して候補を大量生成（全モデル）
- Optunaで得たスタッキング重みでスコアリングして上位K組を出力
- 結果をCSV保存（標準フォーマット + 解析用）
"""
import os
import json
import argparse
import random
import numpy as np
import pandas as pd
from datetime import datetime

# ===== 既存プロジェクトの関数/クラス =====
from lottery_prediction import (
    LotoPredictor,
    preprocess_data,
    create_advanced_features,
    save_predictions_to_csv,
)

# オプション: 制約スコア（あれば使う）
try:
    from limit_break_predictor import ConstraintConfig, constraint_score
    _HAVE_CONSTRAINT = True
except Exception:
    _HAVE_CONSTRAINT = False

# オプション: GNN/拡散
def _try_import_diffusion():
    try:
        from diffusion_module import sample_diffusion_ddpm  # type: ignore
        return sample_diffusion_ddpm
    except Exception:
        return None

def _try_import_gnn():
    try:
        from gnn_core import build_cooccurrence_graph  # type: ignore
        return build_cooccurrence_graph
    except Exception:
        return None

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
    """候補と予測の類似度（一致率）"""
    return len(set(candidate.tolist()) & set(pred.tolist())) / 7.0

def _load_stack_weights(path: str) -> dict:
    if not os.path.exists(path):
        print(f"[WARN] stacking.json が見つかりません: {path} → 等重みを使用")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        params = json.load(f)
    # 期待: {"lstm": 0.2, "automl": 0.3, ...} のような辞書
    if isinstance(params, dict):
        return params
    # 形式が異なる場合に備え、平坦化や既定値を返す
    if isinstance(params, list):
        # [{"model":"lstm","weight":0.2}, ...] 形式などの簡易対応
        kv = {}
        for it in params:
            if isinstance(it, dict):
                m = it.get("model")
                w = it.get("weight")
                if m is not None and w is not None:
                    kv[m] = float(w)
        if kv:
            return kv
    print("[WARN] stacking.json の形式が不明のため、等重みを使用します。")
    return {}

def _normalize_weights(weights: dict, keys: list) -> dict:
    if not weights:
        return {k: 1.0/len(keys) for k in keys}
    w = {k: float(weights.get(k, 0.0)) for k in keys}
    s = sum(v for v in w.values() if np.isfinite(v))
    if s <= 0:
        return {k: 1.0/len(keys) for k in keys}
    return {k: v/s for k, v in w.items()}

def build_latest_predictions_per_model(
    predictor: LotoPredictor,
    full_df: pd.DataFrame,
    X_latest: np.ndarray,
    num_samples: int = 100,
    noise_level: float = 0.01,
) -> dict:
    """
    各モデルから最新1レコードのために num_samples 件の「予測 7個セット」を生成。
    - 決定論的モデル（LSTM/AutoGluon/SetTransformer/TabNet/BNN）は
      最新特徴量を複製して微小ノイズを加えて多様性を確保。
    - 生成系（GAN/PPO/Diffusion/GNN）はサンプルを直接生成。
    戻り値: {model_name: [np.array([7numbers]), ...] }
    """
    preds = {}
    N = num_samples

    # === 決定論的系は特徴量を N 個複製＋微小ノイズ ===
    X_batch = np.repeat(X_latest.reshape(1, -1), N, axis=0).astype(np.float32)
    try:
        from lottery_prediction import add_noise_to_features as _add_noise
        X_batch = _add_noise(X_batch, noise_level=noise_level)
    except Exception:
        # 簡易ノイズ
        X_batch = X_batch + np.random.normal(0.0, noise_level, size=X_batch.shape).astype(np.float32)

    # LSTM (ONNX)
    try:
        lstm_out = predictor.predict_with_onnx(X_batch.astype(np.float32))
        if lstm_out is not None and np.array(lstm_out).shape[0] == N:
            rows = []
            for r in np.array(lstm_out):
                rows.append(np.clip(np.sort(np.round(r).astype(int)), 1, 37))
            preds["lstm"] = rows
    except Exception as e:
        print("[WARN] LSTM 予測スキップ:", e)

    # AutoGluon
    try:
        X_df = _ensure_df_with_feature_names(X_batch, predictor.feature_names)
        cols = []
        for j in range(7):
            model_j = predictor.regression_models[j]
            if model_j is None:
                raise RuntimeError(f"AutoGluon model pos{j} 未学習")
            col = model_j.predict(X_df)
            cols.append(np.asarray(col).reshape(-1, 1))
        automl = np.hstack(cols)
        preds["automl"] = [np.clip(np.sort(np.round(r).astype(int)), 1, 37) for r in automl]
    except Exception as e:
        print("[WARN] AutoGluon 予測スキップ:", e)

    # Set Transformer
    try:
        if predictor.set_transformer_model is not None:
            from lottery_prediction import predict_with_set_transformer
            st_out = predict_with_set_transformer(predictor.set_transformer_model, X_batch)
            preds["set_transformer"] = [np.clip(np.sort(np.round(r).astype(int)), 1, 37) for r in st_out]
    except Exception as e:
        print("[WARN] SetTransformer 予測スキップ:", e)

    # TabNet
    try:
        if hasattr(predictor, "tabnet_model") and predictor.tabnet_model is not None:
            from tabnet_module import predict_tabnet
            tn = predict_tabnet(predictor.tabnet_model, X_batch)
            preds["tabnet"] = [np.clip(np.sort(np.round(r).astype(int)), 1, 37) for r in tn]
    except Exception as e:
        print("[WARN] TabNet 予測スキップ:", e)

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
        print("[WARN] BNN 予測スキップ:", e)

    # === 生成系 ===
    # GAN
    try:
        if predictor.gan_model is not None:
            gan_mat = predictor.gan_model.generate_samples(N)  # N×37
            rows = []
            for i in range(N):
                v = np.asarray(gan_mat[i], dtype=float)
                v = v / max(1e-9, v.sum())
                rows.append(_to_top7_numbers_from_vector(v))
            preds["gan"] = rows
    except Exception as e:
        print("[WARN] GAN 予測スキップ:", e)

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
        print("[WARN] PPO 予測スキップ:", e)

    # Diffusion
    try:
        if predictor.diffusion_model is not None:
            sample_diffusion_ddpm = _try_import_diffusion()
            if sample_diffusion_ddpm is not None:
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
        print("[WARN] Diffusion 予測スキップ:", e)

    # GNN
    try:
        if predictor.gnn_model is not None:
            build_cooccurrence_graph = _try_import_gnn()
            if build_cooccurrence_graph is not None:
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
        print("[WARN] GNN 予測スキップ:", e)

    # 7個セットのみを残す
    for k, v in list(preds.items()):
        preds[k] = [np.array(x, dtype=int) for x in v if isinstance(x, (list, np.ndarray)) and len(x) == 7]
        if not preds[k]:
            preds.pop(k, None)
    return preds

def rank_candidates_with_stacking(candidates: list, per_model_preds: dict, stack_weights: dict, use_constraints=True):
    """
    candidates: [np.array([7nums]), ...]  ← ユニーク候補集合
    per_model_preds: {model: [np.array([7nums]), ...]}
    stack_weights: {model: weight}
    戻り値: [(candidate, score, detail_dict), ...] を score 降順
    """
    model_keys = list(per_model_preds.keys())
    weights = _normalize_weights(stack_weights, model_keys)

    # 事前に各モデルごとの「候補との最大一致率」を高速に計算しやすい形に
    model_best_sim = {}
    for m, plist in per_model_preds.items():
        # 各候補とそのモデル予測群の最大一致率を後で求める
        model_best_sim[m] = plist

    results = []
    for cand in candidates:
        # スタッキングスコア
        score = 0.0
        parts = {}
        for m in model_keys:
            best = 0.0
            for p in model_best_sim[m]:
                s = _similarity(cand, p)
                if s > best:
                    best = s
                    if best == 1.0:
                        break
            parts[m] = best
            score += weights[m] * best

        # オプション: 制約スコアで微調整（0.00〜0.05加点）
        if use_constraints and _HAVE_CONSTRAINT:
            try:
                cscore = constraint_score(cand.tolist(), ConstraintConfig())
                score += 0.05 * cscore
                parts["constraint"] = cscore
            except Exception:
                pass

        results.append((cand, float(score), parts))

    results.sort(key=lambda x: x[1], reverse=True)
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="loto7_prediction_evaluation_with_bonus.csv", help="履歴CSVパス")
    ap.add_argument("--models_dir", default="models/full", help="学習済みモデルのディレクトリ")
    ap.add_argument("--stacking", default="optuna_results/stacking.json", help="スタッキング重みJSON")
    ap.add_argument("--samples", type=int, default=200, help="候補生成サンプル数（モデルごと）")
    ap.add_argument("--topk", type=int, default=30, help="出力する上位K組")
    ap.add_argument("--noise", type=float, default=0.01, help="決定論的モデルへの特徴ノイズ量")
    ap.add_argument("--out", default="predictions_topK.csv", help="候補の一覧CSV（解析用）")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"{args.csv} が見つかりません。")

    # ==== データ読み込み＆前処理 ====
    df = pd.read_csv(args.csv, encoding="utf-8-sig")
    X_all, y_all, scaler = preprocess_data(df)
    if X_all is None:
        raise RuntimeError("前処理に失敗しました。")

    # 最新行の特徴量
    X_latest = X_all[-1]

    # ==== 予測器のロード ====
    # すでに optimize_all_models_full.py で学習＆保存済みが前提
    predictor = LotoPredictor(input_size=X_all.shape[1], hidden_size=128, output_size=7)
    predictor.load_saved_models(args.models_dir)

    # ==== モデル別に候補生成 ====
    per_model_preds = build_latest_predictions_per_model(
        predictor=predictor,
        full_df=df,
        X_latest=X_latest,
        num_samples=args.samples,
        noise_level=args.noise
    )
    if not per_model_preds:
        raise RuntimeError("候補生成に失敗：どのモデルからも予測が得られませんでした。")

    # ==== 候補集合（ユニーク化） ====
    pool = []
    for lst in per_model_preds.values():
        pool.extend([tuple(x.tolist()) for x in lst])
    uniq = sorted(set(pool))
    candidates = [np.array(x, dtype=int) for x in uniq]
    print(f"[INFO] 候補ユニーク数: {len(candidates)}")

    # ==== スタッキング重み読み込み ====
    stack_weights = _load_stack_weights(args.stacking)
    ranked = rank_candidates_with_stacking(candidates, per_model_preds, stack_weights, use_constraints=True)

    # ==== 上位K出力 ====
    topk = ranked[: args.topk]
    for i, (cand, sc, parts) in enumerate(topk, 1):
        print(f"[{i:02d}] {cand.tolist()}  score={sc:.4f}  parts={{{', '.join(f'{k}:{v:.3f}' for k,v in parts.items())}}}")

    # 解析用CSV
    rows = []
    for rank, (cand, sc, parts) in enumerate(topk, 1):
        row = {"rank": rank, "score": sc, "numbers": " ".join(map(str, cand.tolist()))}
        row.update({f"s_{k}": float(v) for k, v in parts.items()})
        rows.append(row)
    pd.DataFrame(rows).to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"[INFO] 解析用CSVを保存: {args.out}")

    # 標準フォーマット保存（上位5だけ & 信頼度=スコア）
    drawing_date = datetime.now().strftime("%Y-%m-%d")
    save_list = [(cand.tolist(), float(sc)) for cand, sc, _ in topk[:5]]
    try:
        save_predictions_to_csv(save_list, drawing_date=drawing_date, filename="loto7_predictions.csv")
        print("[INFO] 標準フォーマットCSVを出力: loto7_predictions.csv")
    except Exception as e:
        print("[WARN] 標準フォーマットCSV保存でエラー:", e)

if __name__ == "__main__":
    random.seed(42); np.random.seed(42)
    main()
