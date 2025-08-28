
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optimize_all_models_full.py
- Optunaで各モデルのパラメータ最適化（既存ロジック流用）
- すべての学習済みモデル（LSTM/AutoGluon/SetTransformer/TabNet/BNN/GAN/PPO/Diffusion/GNN）の
  "本物の予測" を集約してスタッキング最適化を実行（ダミー完全排除）
"""
import os
import json
import time
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

# === 既存最適化ステップの関数（元ファイルにある前提） ===
# - AutoGluon/TabNet/Diffusion/Stacking最適化は既存の関数を呼び出す
try:
    from autogluon.tabular import TabularPredictor  # noqa: F401
    import optuna  # noqa: F401
except Exception:
    pass

try:
    from stacking_optuna import optimize_stacking
except Exception as e:
    raise RuntimeError("stacking_optuna.optimize_stacking を import できません。") from e

# === 本物の学習・予測器 ===
from lottery_prediction import (
    LotoPredictor,
    preprocess_data,
    create_advanced_features,
)

# Optional dependencies（存在すれば使用）
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

def _to_top7_numbers_from_vector(vec37: np.ndarray) -> np.ndarray:
    """37次元スコア/確率ベクトルから上位7番号(1..37)を返す"""
    idx = np.argsort(vec37)[-7:] + 1
    return np.sort(idx.astype(int))

def _ensure_df_with_feature_names(X: np.ndarray, feature_names):
    df = pd.DataFrame(X)
    if feature_names:
        # 予測器が覚えている列名の順序に合わせる（不足は0で埋める）
        for name in feature_names:
            if name not in df.columns:
                df[name] = 0.0
        df = df[feature_names]
    return df

def build_all_train_predictions(predictor: LotoPredictor, X_train: np.ndarray, train_df_for_graph: pd.DataFrame):
    """
    各モデルの「本物の予測」を N×7 配列で返す dict を構築
    - 生成モデル（GAN/PPO/Diffusion/GNN）は N 件ぶん生成して N×7 へ整形
    """
    N = X_train.shape[0]
    preds = {}

    # LSTM (ONNX) — そのまま数値出力（浮動小数可）
    try:
        lstm = predictor.predict_with_onnx(X_train.astype(np.float32))
        if lstm is not None and np.array(lstm).shape[0] == N:
            preds["lstm"] = np.array(lstm, dtype=float)
    except Exception as e:
        print("[WARN] LSTM予測スキップ:", e)

    # AutoGluon — 各ポジションの回帰出力を横結合
    try:
        X_df = _ensure_df_with_feature_names(X_train, predictor.feature_names)
        ag_cols = []
        for j in range(7):
            model_j = predictor.regression_models[j]
            if model_j is None:
                raise RuntimeError(f"AutoGluon model pos{j} が未学習です")
            col = model_j.predict(X_df)
            ag_cols.append(np.asarray(col).reshape(-1, 1))
        automl = np.hstack(ag_cols)
        preds["automl"] = automl
    except Exception as e:
        print("[WARN] AutoGluon予測スキップ:", e)

    # Set Transformer — そのまま N×7
    try:
        if predictor.set_transformer_model is not None:
            from lottery_prediction import predict_with_set_transformer  # 再利用
            st_out = predict_with_set_transformer(predictor.set_transformer_model, X_train)
            if st_out is not None and np.array(st_out).shape == (N, 7):
                preds["set_transformer"] = np.array(st_out, dtype=float)
    except Exception as e:
        print("[WARN] SetTransformer予測スキップ:", e)

    # TabNet — 学習済みなら N×7
    try:
        if hasattr(predictor, "tabnet_model") and predictor.tabnet_model is not None:
            from tabnet_module import predict_tabnet
            tn = predict_tabnet(predictor.tabnet_model, X_train)
            if tn is not None and np.array(tn).shape == (N, 7):
                preds["tabnet"] = np.array(tn, dtype=float)
    except Exception as e:
        print("[WARN] TabNet予測スキップ:", e)

    # BNN — サンプリング予測を平均化（or 1サンプル）→ N×7
    try:
        if hasattr(predictor, "bnn_model") and predictor.bnn_model is not None:
            from bnn_module import predict_bayesian_regression
            bnn_raw = predict_bayesian_regression(predictor.bnn_model, predictor.bnn_guide, X_train, samples=1)
            bnn_arr = np.array(bnn_raw)
            # 期待 shape: (N, 7) or (1, N, 7) → N×7に整形
            if bnn_arr.ndim == 3:
                bnn_arr = bnn_arr.mean(axis=0)
            if bnn_arr.shape == (N, 7):
                preds["bnn"] = bnn_arr.astype(float)
    except Exception as e:
        print("[WARN] BNN予測スキップ:", e)

    # GAN — N本生成し、各行の上位7本を整数化
    try:
        if predictor.gan_model is not None:
            gan_mat = predictor.gan_model.generate_samples(N)  # N×37 の [0..1]
            rows = []
            for i in range(N):
                v = np.asarray(gan_mat[i], dtype=float)
                # 温度スケーリング＆確率化（ゼロ割回避）
                v = v / max(1e-9, v.sum())
                rows.append(_to_top7_numbers_from_vector(v))
            preds["gan"] = np.vstack(rows)
    except Exception as e:
        print("[WARN] GAN予測スキップ:", e)

    # PPO — 各サンプルごとに確率ベクトル(=action)から上位7
    try:
        if predictor.ppo_model is not None:
            import numpy as _np
            rows = []
            for i in range(N):
                obs = _np.zeros(37, dtype=_np.float32)
                # 多様性のため deterministic=False
                action, _ = predictor.ppo_model.predict(obs, deterministic=False)
                action = _np.asarray(action, dtype=float)
                rows.append(_to_top7_numbers_from_vector(action))
            preds["ppo"] = np.vstack(rows)
    except Exception as e:
        print("[WARN] PPO予測スキップ:", e)

    # Diffusion — Nサンプル生成し、各行の上位7
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
                preds["diffusion"] = np.vstack(rows)
    except Exception as e:
        print("[WARN] Diffusion予測スキップ:", e)

    # GNN — 1本のスコアを作り、微小ノイズでN本に拡張
    try:
        if predictor.gnn_model is not None:
            build_cooccurrence_graph = _try_import_gnn()
            if build_cooccurrence_graph is not None:
                graph = build_cooccurrence_graph(train_df_for_graph)
                predictor.gnn_model.eval()
                import torch
                with torch.no_grad():
                    base = predictor.gnn_model(graph.x, graph.edge_index).squeeze().cpu().numpy()
                rows = []
                rng = np.random.default_rng(42)
                for i in range(N):
                    v = base + rng.normal(0, 1e-3, size=base.shape)
                    rows.append(_to_top7_numbers_from_vector(v))
                preds["gnn"] = np.vstack(rows)
    except Exception as e:
        print("[WARN] GNN予測スキップ:", e)

    # すべて N×7 のみ採用
    preds = {k: v for k, v in preds.items() if isinstance(v, np.ndarray) and v.shape == (N, 7)}
    if not preds:
        raise RuntimeError("有効なモデル予測が一つも作れませんでした。学習が完了しているか確認してください。")
    return preds

def main():
    # === データ読み込み ===
    csv_path = "loto7_prediction_evaluation_with_bonus.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} が見つかりません。")

    data = pd.read_csv(csv_path, encoding="utf-8-sig")

    # === 特徴量生成＆スケーリング（学習前に1回だけ） ===
    X_all, y_all, _ = preprocess_data(data)
    if X_all is None or y_all is None:
        raise RuntimeError("前処理に失敗しました（X or y が None）。")

    # 学習/評価分割（スタッキング用に学習部分の予測を作る）
    X_train, _, y_train, _ = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
    print(f"[INFO] Train size: {X_train.shape}, y: {y_train.shape}")

    # === すべてのモデルを本物学習 ===
    # 既存実装の train_model は内部で多種モデルを学習・保存まで行う
    predictor = LotoPredictor(input_size=X_train.shape[1], hidden_size=128, output_size=7)
    predictor.train_model(data, model_dir="models/full")

    # === 学習データに対する「本物の予測」を集める ===
    # GNNや共起用に DataFrame も渡す（抽せん日/本数字を含む必要あり）
    train_preds = build_all_train_predictions(predictor, X_train, data)

    # === スタッキング重み最適化 ===
    RESULT_DIR = "optuna_results"
    os.makedirs(RESULT_DIR, exist_ok=True)
    stack_params = optimize_stacking(train_preds, y_train.tolist())
    with open(os.path.join(RESULT_DIR, "stacking.json"), "w", encoding="utf-8") as f:
        json.dump(stack_params, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Stacking → {stack_params}")

if __name__ == "__main__":
    # 乱数の固定（再現性のため）
    random.seed(42); np.random.seed(42)
    main()
