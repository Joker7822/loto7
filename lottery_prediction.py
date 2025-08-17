
# === Optunaで最適化されたパラメータの読み込み ===
import os
import json

optuna_dir = "optuna_results"
optimized_params = {}

if os.path.exists(optuna_dir):
    for filename in os.listdir(optuna_dir):
        if filename.endswith(".json"):
            with open(os.path.join(optuna_dir, filename), "r") as f:
                optimized_params[filename.replace(".json", "")] = json.load(f)
    print(f"[INFO] Optuna最適化パラメータを適用しました: {list(optimized_params.keys())}")
else:
    print("[INFO] 最適化パラメータディレクトリが見つかりませんでした。デフォルト設定で実行します。")


import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

def stacking_predict_block(self, X, base_dict):
    import numpy as np
    if not getattr(self, "meta_models", None) or any(m is None for m in self.meta_models):
        mats = list(base_dict.values())
        return np.mean(mats, axis=0)
    N = next(iter(base_dict.values())).shape[0]
    out = np.zeros((N, 7), dtype=float)
    for i in range(7):
        entry = self.meta_models[i]
        if entry is None:
            mats = list(base_dict.values())
            out[:, i] = np.mean(mats, axis=0)[:, i]
            continue
        model, keys = entry
        cols = [base_dict[k][:, i].reshape(-1, 1) for k in keys]
        Xi = np.hstack(cols)
        out[:, i] = model.predict(Xi)
    return out
import stat
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor, RandomForestRegressor, StackingRegressor,
    GradientBoostingClassifier, RandomForestClassifier, HistGradientBoostingClassifier,
    AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier
)
from sklearn.linear_model import Ridge, LogisticRegression, RidgeClassifier, Perceptron, PassiveAggressiveClassifier, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import StackingRegressor
from statsmodels.tsa.arima.model import ARIMA
from stable_baselines3 import PPO
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna
import matplotlib
matplotlib.use('Agg')  # ← ★ この行を先に追加！
import matplotlib.pyplot as plt
import aiohttp
import asyncio
import warnings
import re
import platform
import gym
import sys
import os
import random
from sklearn.metrics import precision_score, recall_score, f1_score
from neuralforecast.models import TFT
from neuralforecast import NeuralForecast
import onnxruntime
import streamlit as st
from autogluon.tabular import TabularPredictor
import torch.backends.cudnn
from datetime import datetime 
from itertools import combinations
import shutil
import traceback
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from gym.utils import seeding
import time
import subprocess

# Windows環境のイベントループポリシーを設定
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

warnings.filterwarnings("ignore")


def _save_all_models_no_self(predictor, model_dir):
    os.makedirs(model_dir, exist_ok=True)
    try:
        def _save(path, save_fn):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            save_fn(path)
            print(f"[INFO] 保存完了: {path}")

        # LSTM ONNX (if produced externally)
        if os.path.exists("lstm_model.onnx"):
            dst = os.path.join(model_dir, "lstm_model.onnx")
            shutil.copyfile("lstm_model.onnx", dst)
            print(f"[INFO] 保存完了: {dst}")

        # GAN
        try:
            g = getattr(predictor, "gan_model", None)
            if g is not None:
                _save(os.path.join(model_dir, "gan_model.pth"), lambda p: torch.save(g.state_dict(), p))
        except Exception as e:
            print("[WARN] GAN保存失敗:", e)

        # PPO
        try:
            ppo = getattr(predictor, "ppo_model", None)
            if ppo is not None:
                out = os.path.join(model_dir, "ppo_model.zip")
                ppo.save(out)
                print(f"[INFO] 保存完了: {out}")
        except Exception as e:
            print("[WARN] PPO保存失敗:", e)

        # Diffusion
        try:
            dm = getattr(predictor, "diffusion_model", None)
            if dm is not None:
                _save(os.path.join(model_dir, "diffusion_model.pth"), lambda p: torch.save(dm.state_dict(), p))
        except Exception as e:
            print("[WARN] Diffusion保存失敗:", e)

        # GNN
        try:
            gnn = getattr(predictor, "gnn_model", None)
            if gnn is not None:
                _save(os.path.join(model_dir, "gnn_model.pth"), lambda p: torch.save(gnn.state_dict(), p))
        except Exception as e:
            print("[WARN] GNN保存失敗:", e)

        # TabNet
        try:
            tabnet = getattr(predictor, "tabnet_model", None)
            if tabnet is not None:
                from tabnet_module import save_tabnet_model
                save_tabnet_model(tabnet, os.path.join(model_dir, "tabnet_model"))
                print("[INFO] 保存完了: TabNet")
        except Exception as e:
            print("[WARN] TabNet保存失敗:", e)

        # BNN
        try:
            bnn = getattr(predictor, "bnn_model", None)
            bnn_guide = getattr(predictor, "bnn_guide", None)
            if bnn is not None and bnn_guide is not None:
                from bnn_module import save_bayesian_model
                save_bayesian_model(bnn, bnn_guide, os.path.join(model_dir, "bnn_model"))
                print("[INFO] 保存完了: BNN")
        except Exception as e:
            print("[WARN] BNN保存失敗:", e)

        # AutoGluon per position
        try:
            for j in range(7):
                model_path = f"autogluon_model_pos{j}"
                dest = os.path.join(model_dir, f"autogluon_model_pos{j}")
                if os.path.isdir(model_path):
                    if os.path.isdir(dest):
                        shutil.rmtree(dest)
                    shutil.copytree(model_path, dest)
            print("[INFO] 保存完了: AutoGluon")
        except Exception as e:
            print("[WARN] AutoGluon保存失敗:", e)

        print(f"[INFO] モデルを保存しました → {model_dir}")
        git_commit_and_push(model_dir, "Save trained models")
    except Exception as e:
        print(f"[WARNING] モデル保存中にエラー発生: {e}")
        traceback.print_exc()

def set_global_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def git_commit_and_push(file_path, message):
    try:
        subprocess.run(["git", "add", file_path], check=True)
        diff = subprocess.run(["git", "diff", "--cached", "--quiet"])
        if diff.returncode != 0:
            subprocess.run(["git", "config", "--global", "user.name", "github-actions"], check=True)
            subprocess.run(["git", "config", "--global", "user.email", "github-actions@github.com"], check=True)
            subprocess.run(["git", "commit", "-m", message], check=True)
            subprocess.run(["git", "push"], check=True)
        else:
            print(f"[INFO] No changes in {file_path}")
    except Exception as e:
        print(f"[WARNING] Git commit/push failed: {e}")

def get_valid_num_heads(embed_dim, max_heads=8):
    for h in reversed(range(1, max_heads + 1)):
        if embed_dim % h == 0:
            return h
    return 1

def add_noise_to_features(X, noise_level=0.02):
    import numpy as np
    noise = np.random.normal(loc=0.0, scale=noise_level, size=X.shape)
    return X + noise
    
class LotoGAN(nn.Module):
    def __init__(self, noise_dim=100):
        super(LotoGAN, self).__init__()
        self.generator = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 37),
            nn.Sigmoid()
        )
        self.discriminator = nn.Sequential(
            nn.Linear(37, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.noise_dim = noise_dim

    def generate_samples(self, num_samples):
        noise = torch.randn(num_samples, self.noise_dim)
        with torch.no_grad():
            samples = self.generator(noise)
        return samples.numpy()

def create_advanced_features(dataframe):

    def convert_to_number_list(x):
        if isinstance(x, str):
            cleaned = x.strip("[]").replace(",", " ").replace("'", "").replace('"', "")
            return [int(n) for n in cleaned.split() if n.isdigit()]
        return x if isinstance(x, list) else [0]

    dataframe['本数字'] = dataframe['本数字'].apply(convert_to_number_list)
    dataframe['ボーナス数字'] = dataframe['ボーナス数字'].apply(convert_to_number_list)
    dataframe['抽せん日'] = pd.to_datetime(dataframe['抽せん日'])

    valid_mask = (dataframe['本数字'].apply(len) == 7) & (dataframe['ボーナス数字'].apply(len) == 2)
    dataframe = dataframe[valid_mask].copy()

    nums_array = np.vstack(dataframe['本数字'].values)
    sorted_nums = np.sort(nums_array, axis=1)
    diffs = np.diff(sorted_nums, axis=1)

    features = pd.DataFrame(index=dataframe.index)
    features['奇数比'] = (nums_array % 2 != 0).sum(axis=1) / nums_array.shape[1]
    features['本数字合計'] = nums_array.sum(axis=1)
    features['レンジ'] = nums_array.max(axis=1) - nums_array.min(axis=1)
    features['標準偏差'] = np.std(nums_array, axis=1)
    features['曜日'] = dataframe['抽せん日'].dt.dayofweek
    features['月'] = dataframe['抽せん日'].dt.month
    features['年'] = dataframe['抽せん日'].dt.year
    features['連番数'] = (diffs == 1).sum(axis=1)
    features['最小間隔'] = diffs.min(axis=1)
    features['最大間隔'] = diffs.max(axis=1)
    features['数字平均'] = nums_array.mean(axis=1)
    features['偶数比'] = (nums_array % 2 == 0).sum(axis=1) / nums_array.shape[1]
    features['中央値'] = np.median(nums_array, axis=1)

    # 出現間隔平均（ループ）
    last_seen = {}
    gaps = []
    for idx, nums in dataframe['本数字'].items():
        gap = [idx - last_seen.get(n, idx) for n in nums]
        gaps.append(np.mean(gap))
        for n in nums:
            last_seen[n] = idx
    features['出現間隔平均'] = gaps

    # 出現頻度スコア（1回のみ算出し使い回す）
    all_numbers = [num for nums in dataframe['本数字'] for num in nums]
    all_freq = pd.Series(all_numbers).value_counts().to_dict()
    features['出現頻度スコア'] = dataframe['本数字'].apply(lambda nums: sum(all_freq.get(n, 0) for n in nums) / len(nums))

    # ペア・トリプル頻度も先にまとめて構築（重複処理を排除）
    pair_freq, triple_freq = {}, {}
    for nums in dataframe['本数字']:
        for pair in combinations(sorted(nums), 2):
            pair_freq[pair] = pair_freq.get(pair, 0) + 1
        for triple in combinations(sorted(nums), 3):
            triple_freq[triple] = triple_freq.get(triple, 0) + 1

    features['ペア出現頻度'] = dataframe['本数字'].apply(
        lambda nums: sum(pair_freq.get(tuple(sorted((nums[i], nums[j]))), 0) for i in range(len(nums)) for j in range(i+1, len(nums)))
    )
    features['トリプル出現頻度'] = dataframe['本数字'].apply(
        lambda nums: sum(triple_freq.get(tuple(sorted((nums[i], nums[j], nums[k]))), 0) for i in range(len(nums)) for j in range(i+1, len(nums)) for k in range(j+1, len(nums)))
    )

    # 直近5回出現率
    past_5_counts = {}
    for i, row in dataframe.iterrows():
        nums = row['本数字']
        recent = dataframe[dataframe['抽せん日'] < row['抽せん日']].tail(5)
        recent_nums = [num for nums in recent['本数字'] for num in nums]
        count = sum(n in recent_nums for n in nums)
        past_5_counts[i] = count / len(nums)
    features['直近5回出現率'] = features.index.map(past_5_counts)

    return pd.concat([dataframe, features], axis=1)

def preprocess_data(data):
    """データの前処理: 特徴量の作成 & スケーリング"""
    
    # 特徴量作成
    processed_data = create_advanced_features(data)

    if processed_data.empty:
        print("エラー: 特徴量生成後のデータが空です。データのフォーマットを確認してください。")
        return None, None, None

    print("=== 特徴量作成後のデータ ===")
    print(processed_data.head())

    # 数値特徴量の選択
    numeric_features = processed_data.select_dtypes(include=[np.number]).columns
    X = processed_data[numeric_features].fillna(0)  # 欠損値を0で埋める

    print(f"数値特徴量の数: {len(numeric_features)}, サンプル数: {X.shape[0]}")

    if X.empty:
        print("エラー: 数値特徴量が作成されず、データが空になっています。")
        return None, None, None

    # スケーリング
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    print("=== スケーリング後のデータ ===")
    print(X_scaled[:5])  # 最初の5件を表示

    # 目標変数の準備
    try:
        y = np.array([list(map(int, nums)) for nums in processed_data['本数字']])
    except Exception as e:
        print(f"エラー: 目標変数の作成時に問題が発生しました: {e}")
        return None, None, None

    return X_scaled, y, scaler

def convert_numbers_to_binary_vectors(data):
    """
    本数字を0/1ベクトル化する
    例：[1,5,7,22,28,30,36] → [1,0,0,0,1,0,1, ..., 0,1]
    """
    vectors = []
    for numbers in data['本数字']:
        vec = np.zeros(37)
        for n in numbers:
            if 1 <= n <= 37:
                vec[n-1] = 1
        vectors.append(vec)
    return np.array(vectors)

def calculate_prediction_errors(predictions, actual_numbers):
    """予測値と実際の当選結果の誤差を計算し、特徴量として保存"""
    errors = []
    for pred, actual in zip(predictions, actual_numbers):
        pred_numbers = set(pred[0])
        actual_numbers = set(actual)
        error_count = len(actual_numbers - pred_numbers)
        errors.append(error_count)
    
    return np.mean(errors)

def save_self_predictions(predictions, file_path="self_predictions.csv", max_records=100):
    """予測結果をCSVに保存し、保存件数を最大max_recordsに制限し、世代ファイルも保存"""
    rows = []
    for numbers, confidence in predictions:
        rows.append(numbers.tolist())

    # 既存データを読み込む
    if os.path.exists(file_path):
        existing = pd.read_csv(file_path, header=None).values.tolist()
        rows = existing + rows

    # 最新max_records件だけ残す
    rows = rows[-max_records:]

    df = pd.DataFrame(rows)

    # --- メイン保存 ---
    df.to_csv(file_path, index=False, header=False)
    print(f"[INFO] 自己予測結果を {file_path} に保存しました（最大{max_records}件）")

    # --- 🔥 世代ファイル保存も専用フォルダに変更 ---
    gen_dir = "self_predictions_gen"
    os.makedirs(gen_dir, exist_ok=True)  # フォルダがなければ作成

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    generation_file = os.path.join(gen_dir, f"self_predictions_gen_{timestamp}.csv")
    df.to_csv(generation_file, index=False, header=False)
    print(f"[INFO] 世代別に自己予測も保存しました: {generation_file}")

def load_self_predictions(file_path="self_predictions.csv", min_match_threshold=3, true_data=None):
    if not os.path.exists(file_path):
        print(f"[INFO] 自己予測ファイル {file_path} が見つかりません。")
        return None

    try:
        # 🔥 高速版に置き換え！
        df = pd.read_csv(file_path, header=None)
        df = df.dropna()
        df = df[df.apply(lambda row: all(1 <= x <= 37 for x in row), axis=1)]
        numbers_list = df.values.tolist()

        if true_data is not None:
            scores = evaluate_self_predictions(numbers_list, true_data)
            filtered_rows = [r for r, s in zip(numbers_list, scores) if s >= min_match_threshold]
            print(f"[INFO] 一致数{min_match_threshold}以上の自己予測データ: {len(filtered_rows)}件")
            return filtered_rows
        else:
            return numbers_list

    except Exception as e:
        print(f"[ERROR] 自己予測データ読み込みエラー: {e}")
        return None

def evaluate_self_predictions(self_predictions, true_data):
    """
    自己予測リストと本物データを比較して一致数を評価
    :param self_predictions: [[5,12,17,22,30,34,37], ...]
    :param true_data: 過去の本物本数字データ（data['本数字'].tolist()）
    :return: 各自己予測に対応する最大一致数リスト
    """
    scores = []
    true_sets = [set(nums) for nums in true_data]

    for pred in self_predictions:
        pred_set = set(pred)
        max_match = 0
        for true_set in true_sets:
            match = len(pred_set & true_set)
            if match > max_match:
                max_match = match
        scores.append(max_match)

    return scores

def update_features_based_on_results(data, accuracy_results):
    """過去の予測結果と実際の結果の比較から特徴量を更新"""
    
    for result in accuracy_results:
        event_date = result["抽せん日"]
        max_matches = result["最高一致数"]
        avg_matches = result["平均一致数"]
        confidence_avg = result["信頼度平均"]

        # 過去のデータに予測精度を組み込む
        data.loc[data["抽せん日"] == event_date, "過去の最大一致数"] = max_matches
        data.loc[data["抽せん日"] == event_date, "過去の平均一致数"] = avg_matches
        data.loc[data["抽せん日"] == event_date, "過去の予測信頼度"] = confidence_avg

    # 特徴量がない場合は0で埋める
    data["過去の最大一致数"] = data["過去の最大一致数"].fillna(0)
    data["過去の平均一致数"] = data["過去の平均一致数"].fillna(0)
    data["過去の予測信頼度"] = data["過去の予測信頼度"].fillna(0)

    return data

class LotoLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LotoLSTM, self).__init__()
        self.lstm = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(hidden_size * 2, 1)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden*2]
        attn_weights = torch.softmax(self.attn(lstm_out).squeeze(-1), dim=1)  # [batch, seq_len]
        context = torch.sum(lstm_out * attn_weights.unsqueeze(-1), dim=1)  # [batch, hidden*2]
        out = self.fc(context)
        return out

def train_lstm_model(X_train, y_train, input_size, device):
    
    torch.backends.cudnn.benchmark = True  # ★これを追加
    
    model = LotoLSTM(input_size=input_size, hidden_size=128, output_size=7).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=True, pin_memory=True, num_workers=2)  # ★変更

    scaler = torch.cuda.amp.GradScaler()  # ★Mixed Precision追加

    model.train()
    for epoch in range(50):
        total_loss = 0
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():  # ★ここもMixed Precision
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        print(f"[LSTM] Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

    # ONNXエクスポート
    dummy_input = torch.randn(1, 1, input_size).to(device)
    torch.onnx.export(
        model,
        dummy_input,
        "lstm_model.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=12
    )
    print("[INFO] LSTM モデルのトレーニングが完了")
    return model

def extract_high_accuracy_combinations(evaluation_df, threshold=5):
    high_matches = evaluation_df[evaluation_df["本数字一致数"] >= threshold]
    return high_matches

def convert_hit_combos_to_training_data(hit_combos, original_data):
    temp_df = original_data.copy()
    new_rows = []
    for _, row in hit_combos.iterrows():
        temp = {
            "抽せん日": row["抽せん日"],
            "本数字": row["予測番号"],
            "ボーナス数字": row["当選ボーナス"]
        }
        new_rows.append(temp)
    if not new_rows:
        return None, None
    temp_df = pd.DataFrame(new_rows)
    return preprocess_data(temp_df)[:2]

# === 🔧 Set Transformer モデル ===
class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=dim_in, num_heads=num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(dim_in)
        self.ff = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.ReLU(),
            nn.Linear(dim_out, dim_out)
        )
        self.ln2 = nn.LayerNorm(dim_out)

    def forward(self, x):
        h, _ = self.mha(x, x, x)
        x = self.ln1(x + h)
        h = self.ff(x)
        return self.ln2(x + h)

class SetTransformerRegressor(nn.Module):
    def __init__(self, input_dim, output_dim, num_sabs=2):
        super().__init__()
        num_heads = get_valid_num_heads(input_dim)  # ← 動的に決定！
        self.encoders = nn.ModuleList([
            SAB(input_dim, input_dim, num_heads) for _ in range(num_sabs)
        ])
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        for sab in self.encoders:
            x = sab(x)
        x = x.mean(dim=1)
        return self.fc(x)

def train_set_transformer_model(X, y, input_dim, output_dim=7, epochs=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SetTransformerRegressor(input_dim, output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            batch_X = batch_X.unsqueeze(1)  # [batch, set_size=1, input_dim]
            optimizer.zero_grad()
            out = model(batch_X)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[SetTransformer] Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

    return model

def predict_with_set_transformer(model, X):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        inputs = torch.tensor(X, dtype=torch.float32).to(device).unsqueeze(1)
        outputs = model(inputs).cpu().numpy()
    return outputs

class LotoPredictor:
    def __init__(self, input_size, hidden_size, output_size):
        print("[INFO] モデルを初期化")
        self.lstm_model = LotoLSTM(input_size, hidden_size, output_size)
        self.regression_models = [None] * 7
        self.scaler = None
        self.onnx_session = None
        self.gan_model = None
        self.ppo_model = None
        self.feature_names = None  # AutoGluon用に使用する特徴量名
        self.set_transformer_model = None
        self.diffusion_model = None
        self.diffusion_betas = None
        self.diffusion_alphas_cumprod = None
        self.regression_models = [None] * 7
        self.meta_models = [None] * 7  # [STACKING-INIT]

        
        # --- GANモデルロード（存在すれば） ---
        if os.path.exists("gan_model.pth"):
            self.gan_model = LotoGAN()
            self.gan_model.load_state_dict(torch.load("gan_model.pth"))
            self.gan_model.eval()
            print("[INFO] GANモデルをロードしました")

        # --- PPOモデルロード（存在すれば） ---
        if os.path.exists("ppo_model.zip"):
            self.ppo_model = PPO.load("ppo_model")
            print("[INFO] PPOモデルをロードしました")

    def load_onnx_model(self, onnx_path="lstm_model.onnx"):
        print("[INFO] ONNX モデルを読み込みます")
        self.onnx_session = onnxruntime.InferenceSession(
            onnx_path,
            providers=['CPUExecutionProvider']
        )

    def predict_with_onnx(self, X):
        if self.onnx_session is None:
            print("[ERROR] ONNX モデルがロードされていません")
            return None

        input_name = self.onnx_session.get_inputs()[0].name
        output = self.onnx_session.run(None, {input_name: X.astype(np.float32)})
        return output[0]

    def train_model(self, data, accuracy_results=None, model_dir="models/tmp"):
        os.makedirs(model_dir, exist_ok=True)
        set_global_seed(42)
        print("[INFO] データ前処理を開始")

        data["抽せん日"] = pd.to_datetime(data["抽せん日"], errors='coerce')
        latest_valid_date = data["抽せん日"].max()
        data = data[data["抽せん日"] <= latest_valid_date]
        print(f"[INFO] 未来データ除外済: {latest_valid_date.date()} 以前 {len(data)}件")

        true_numbers = data['本数字'].tolist()
        self_data = load_self_predictions(file_path="self_predictions.csv", min_match_threshold=6, true_data=true_numbers)
        high_match_combos = extract_high_match_patterns(data, min_match=6)

        if self_data or high_match_combos:
            print("[INFO] 過去の高一致自己予測＋高一致本物データを追加します")
            new_rows = []
            for nums in (self_data or []):
                new_rows.append({
                    '抽せん日': pd.Timestamp.now(),
                    '回号': 9999,
                    '本数字': nums,
                    'ボーナス数字': [0, 0]
                })
            for nums in (high_match_combos or []):
                new_rows.append({
                    '抽せん日': pd.Timestamp.now(),
                    '回号': 9999,
                    '本数字': nums,
                    'ボーナス数字': [0, 0]
                })
            if new_rows:
                new_data = pd.DataFrame(new_rows)
                data = pd.concat([data, new_data], ignore_index=True)

        X, y, self.scaler = preprocess_data(data)
        if X is None or y is None:
            print("[ERROR] 前処理後のデータが空です")
            return

        self.feature_names = [str(i) for i in range(X.shape[1])]
        processed_df = create_advanced_features(data)
        important_features = extract_strong_features(accuracy_results, processed_df) if accuracy_results else []
        print(f"[INFO] 強調対象の特徴量: {important_features}")
        X = reinforce_features(X, self.feature_names, important_features, multiplier=1.5)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        input_size = X_train.shape[1]
        print("[INFO] LSTM モデルの訓練開始")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_train_tensor = torch.tensor(X_train.reshape(-1, 1, input_size), dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
        self.lstm_model = train_lstm_model(X_train_tensor, y_train_tensor, input_size, device=device)

        dummy_input = torch.randn(1, 1, input_size)
        lstm_path = os.path.join(model_dir, "lstm_model.onnx")
        torch.onnx.export(self.lstm_model, dummy_input, lstm_path,
                        input_names=["input"], output_names=["output"],
                        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
                        opset_version=12)
        self.load_onnx_model(lstm_path)

        print("[INFO] Set Transformer モデルの学習を開始")
        self.set_transformer_model = train_set_transformer_model(X_train, y_train, input_size)

        print("[INFO] Diffusionモデルを訓練中")
        from diffusion_module import train_diffusion_ddpm
        real_data_bin = convert_numbers_to_binary_vectors(data)
        self.diffusion_model, self.diffusion_betas, self.diffusion_alphas_cumprod = train_diffusion_ddpm(real_data_bin)
        torch.save(self.diffusion_model.state_dict(), os.path.join(model_dir, "diffusion_model.pth"))

        print("[INFO] GNNモデルを訓練中")
        from gnn_core import LotoGNN, build_cooccurrence_graph
        graph_data = build_cooccurrence_graph(data)
        self.gnn_model = LotoGNN()
        optimizer = torch.optim.Adam(self.gnn_model.parameters(), lr=0.01)
        for epoch in range(200):
            self.gnn_model.train()
            out = self.gnn_model(graph_data.x, graph_data.edge_index)
            loss = torch.nn.functional.mse_loss(out, graph_data.x)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if epoch % 50 == 0:
                print(f"[GNN] Epoch {epoch} Loss: {loss.item():.4f}")
        torch.save(self.gnn_model.state_dict(), os.path.join(model_dir, "gnn_model.pth"))

        print("[INFO] TabNet モデルを訓練中")
        from tabnet_module import train_tabnet, save_tabnet_model
        self.tabnet_model = train_tabnet(X_train, y_train)
        save_tabnet_model(self.tabnet_model, os.path.join(model_dir, "tabnet_model"))

        print("[INFO] BNN モデルを訓練中")
        from bnn_module import train_bayesian_regression, save_bayesian_model
        self.bnn_model, self.bnn_guide = train_bayesian_regression(X_train, y_train, input_size)
        save_bayesian_model(self.bnn_model, self.bnn_guide, os.path.join(model_dir, "bnn_model"))

        print("[INFO] AutoGluon モデルを訓練中")
        for i in range(7):
            df_train = pd.DataFrame(X_train)
            df_train['target'] = y_train[:, i]
            ag_path = os.path.join(model_dir, f"autogluon_model_pos{i}")
            predictor = TabularPredictor(label='target', path=ag_path, verbosity=0).fit(
                df_train,
                excluded_model_types=['KNN', 'NN_TORCH'],
                hyperparameters={
                    'GBM': {'device': 'CPU', 'num_boost_round': 300},
                    'XGB': {'tree_method': 'hist', 'n_estimators': 300},
                    'CAT': {'task_type': 'CPU', 'iterations': 300},
                    'RF': {'n_estimators': 200}
                },
                num_gpus=1,
                ag_args_fit={'random_seed': 42}
            )
            self.regression_models[i] = predictor
            print(f"[DEBUG] AutoGluon モデル {i+1}/7 完了")

        print("[INFO] GAN モデルを訓練中")
        real_data_tensor = torch.tensor(real_data_bin, dtype=torch.float32)
        gan = LotoGAN()
        optimizer_G = optim.Adam(gan.generator.parameters(), lr=0.001)
        optimizer_D = optim.Adam(gan.discriminator.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        for epoch in range(3000):
            set_global_seed(10000 + epoch)
            idx = np.random.randint(0, real_data_tensor.size(0), 32)
            real_batch = real_data_tensor[idx]
            noise = torch.randn(32, gan.noise_dim)
            fake_batch = gan.generator(noise)
            real_labels = torch.ones(32, 1)
            fake_labels = torch.zeros(32, 1)

            optimizer_D.zero_grad()
            d_loss = (criterion(gan.discriminator(real_batch), real_labels) +
                    criterion(gan.discriminator(fake_batch), fake_labels)) / 2
            d_loss.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()
            fake_batch = gan.generator(torch.randn(32, gan.noise_dim))
            g_loss = criterion(gan.discriminator(fake_batch), real_labels)
            g_loss.backward()
            optimizer_G.step()

            if (epoch + 1) % 500 == 0:
                print(f"[GAN] Epoch {epoch+1} D: {d_loss.item():.4f} G: {g_loss.item():.4f}")
        self.gan_model = gan
        torch.save(self.gan_model.state_dict(), os.path.join(model_dir, "gan_model.pth"))

        print("[INFO] PPO モデルを訓練中")
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from reinforcement_env import LotoEnv  # あなたの LotoEnv が定義されているファイル名に合わせてください

        historical_numbers = [n for nums in data['本数字'].tolist() for n in nums]
        env = DummyVecEnv([lambda: LotoEnv(historical_numbers)])

        self.ppo_model = PPO("MlpPolicy", env, seed=42, verbose=0)
        self.ppo_model.learn(total_timesteps=50000)
        
        # [STACKING-TRAIN] メタスタッキング学習（ホールドアウトで学習）
        try:
            self._train_meta_models(X_test, y_test)
        except Exception as e:
            print(f"[STACK] メタ学習失敗: {e}")

self.ppo_model.save(os.path.join(model_dir, "ppo_model.zip"))

        print("[INFO] 全モデルの訓練と保存が完了しました")

    def load_saved_models(self, model_dir):

        # LSTM
        onnx_path = os.path.join(model_dir, "lstm_model.onnx")
        if os.path.exists(onnx_path):
            self.load_onnx_model(onnx_path)
            print("[INFO] LSTM (ONNX) モデルをロードしました")

        # GAN
        gan_path = os.path.join(model_dir, "gan_model.pth")
        if os.path.exists(gan_path):
            from gnn_core import LotoGAN
            self.gan_model = LotoGAN()
            self.gan_model.load_state_dict(torch.load(gan_path))
            self.gan_model.eval()
            print("[INFO] GANモデルをロードしました")

        # PPO
        ppo_path = os.path.join(model_dir, "ppo_model.zip")
        if os.path.exists(ppo_path):
            self.ppo_model = PPO.load(ppo_path)
            print("[INFO] PPOモデルをロードしました")

        # Diffusion
        diff_path = os.path.join(model_dir, "diffusion_model.pth")
        if os.path.exists(diff_path):
            from diffusion_module import DiffusionModel, get_diffusion_constants
            self.diffusion_model = DiffusionModel()
            self.diffusion_model.load_state_dict(torch.load(diff_path))
            self.diffusion_model.eval()
            self.diffusion_betas, self.diffusion_alphas_cumprod = get_diffusion_constants()
            print("[INFO] Diffusion モデルをロードしました")

        # GNN
        gnn_path = os.path.join(model_dir, "gnn_model.pth")
        if os.path.exists(gnn_path):
            from gnn_core import LotoGNN
            self.gnn_model = LotoGNN()
            self.gnn_model.load_state_dict(torch.load(gnn_path))
            self.gnn_model.eval()
            print("[INFO] GNN モデルをロードしました")

        # TabNet
        tabnet_path = os.path.join(model_dir, "tabnet_model")
        if os.path.exists(tabnet_path):
            from tabnet_module import load_tabnet_model
            self.tabnet_model = load_tabnet_model(tabnet_path)
            print("[INFO] TabNet モデルをロードしました")

        # BNN
        bnn_path = os.path.join(model_dir, "bnn_model")
        if os.path.exists(bnn_path):
            from bnn_module import load_bayesian_model
            self.bnn_model, self.bnn_guide = load_bayesian_model(bnn_path)
            print("[INFO] BNN モデルをロードしました")

        # AutoGluon
        for j in range(7):
            ag_path = os.path.join(model_dir, f"autogluon_model_pos{j}")
            if os.path.exists(ag_path):
                from autogluon.tabular import TabularPredictor
                self.regression_models[j] = TabularPredictor.load(ag_path)
                print(f"[INFO] AutoGluon モデル {j} をロードしました")

    def predict(self, latest_data, num_candidates=50):
        print(f"[INFO] 予測を開始（候補数: {num_candidates}）")
        X, _, _ = preprocess_data(latest_data)

        if X is None or len(X) == 0:
            print("[ERROR] 予測用データが空です")
            return
    # ===== [STACKING-METHODS] メタスタッキング補助 =====
    def _collect_base_preds(self, X):
        import numpy as np
        preds = {}
        try:
            if getattr(self, "regression_models", None) and self.regression_models[0] is not None:
                import pandas as pd
                X_df = pd.DataFrame(X)
                ag_mat = np.column_stack([self.regression_models[j].predict(X_df) for j in range(7)])
                preds["ag"] = ag_mat.astype(float)
        except Exception as e:
            print(f"[STACK] AutoGluon予測取得に失敗: {e}")
        try:
            if getattr(self, "lstm_model", None) is not None:
                import torch
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.lstm_model.to(device)
                self.lstm_model.eval()
                X_tensor = torch.tensor(X.reshape(-1, 1, X.shape[1]), dtype=torch.float32).to(device)
                with torch.no_grad():
                    lstm_mat = self.lstm_model(X_tensor).detach().cpu().numpy()
                preds["lstm"] = lstm_mat.astype(float)
        except Exception as e:
            print(f"[STACK] LSTM予測取得に失敗: {e}")
        try:
            if getattr(self, "set_transformer_model", None) is not None:
                st_mat = predict_with_set_transformer(self.set_transformer_model, X)
                preds["st"] = st_mat.astype(float)
        except Exception as e:
            print(f"[STACK] SetTransformer予測取得に失敗: {e}")
        try:
            if getattr(self, "tabnet_model", None) is not None:
                from tabnet_module import predict_tabnet
                tb_mat = predict_tabnet(self.tabnet_model, X)
                preds["tabnet"] = tb_mat.astype(float)
        except Exception as e:
            print(f"[STACK] TabNet予測取得に失敗: {e}")
        try:
            if getattr(self, "bnn_model", None) is not None and getattr(self, "bnn_guide", None) is not None:
                from bnn_module import predict_bayesian_regression
                bnn_mat = predict_bayesian_regression(self.bnn_model, self.bnn_guide, X, samples=1)
                import numpy as np
                bnn_mat = np.asarray(bnn_mat)
                if bnn_mat.ndim == 3:
                    bnn_mat = bnn_mat[0]
                preds["bnn"] = bnn_mat.astype(float)
        except Exception as e:
            print(f"[STACK] BNN予測取得に失敗: {e}")
        return preds

    def _train_meta_models(self, X_test, y_test):
        import numpy as np
        from sklearn.linear_model import Ridge
        base = self._collect_base_preds(X_test)
        if not base:
            print("[STACK] ベース予測が得られないため、メタ学習をスキップします。")
            self.meta_models = [None] * 7
            return
        keys = list(base.keys())
        N = X_test.shape[0]
        self.meta_models = [None] * 7
        for i in range(7):
            cols = []
            for k in keys:
                mat = base[k]
                if mat.shape != (N, 7):
                    raise ValueError(f"[STACK] 形状不一致: {k} has {mat.shape}, expected {(N, 7)}")
                cols.append(mat[:, i].reshape(-1, 1))
            Xi = np.hstack(cols)
            yi = y_test[:, i].astype(float)
            model = Ridge(alpha=1.0)
            model.fit(Xi, yi)
            self.meta_models[i] = (model, keys)
        print(f"[STACK] メタモデル学習完了 (#base={len(keys)}) → keys={keys}")
     None, None

        print(f"[DEBUG] 予測用データの shape: {X.shape}")

        freq_score = calculate_number_frequencies(latest_data)
        cycle_score = calculate_number_cycle_score(latest_data)
        all_predictions = []

        def append_prediction(numbers, base_confidence=0.8):
            numbers = [int(n) for n in numbers]  # ← 安全キャスト
            score = sum(freq_score.get(n, 0) for n in numbers) - sum(cycle_score.get(n, 0) for n in numbers)
            confidence = base_confidence + (score / 500.0)
            all_predictions.append((numbers, confidence))

        try:
            X_df = pd.DataFrame(X)

            if self.feature_names:
                for name in self.feature_names:
                    if name not in X_df.columns:
                        X_df[name] = 0.0
                X_df = X_df[self.feature_names]
                X = X_df.values
            else:
                print("[WARNING] self.feature_names が未定義です")

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            for i in range(num_candidates):
                set_global_seed(100 + i)

                ml_predictions = np.array([
                    self.regression_models[j].predict(X_df) for j in range(7)
                ]).T

                self.lstm_model.to(device)
                self.lstm_model.eval()
                X_tensor = torch.tensor(X.reshape(-1, 1, X.shape[1]), dtype=torch.float32).to(device)
                with torch.no_grad():
                    lstm_predictions = self.lstm_model(X_tensor).detach().cpu().numpy()

                
                base_dict = {"ag": ml_predictions, "lstm": lstm_predictions}
                try:
                    base_dict["st"] = st_predictions
                except Exception: pass
                try:
                    base_dict["tabnet"] = tabnet_preds
                except Exception: pass
                try:
                    base_dict["bnn"] = bnn_preds
                except Exception: pass
                final_predictions = stacking_predict_block(self, X, base_dict)
        
                    numbers = np.round(pred).astype(int)
                    numbers = np.clip(numbers, 1, 37)
                    numbers = np.sort(numbers)
                    append_prediction(numbers, base_confidence=1.0)

            if self.gan_model:
                for i in range(num_candidates):
                    set_global_seed(int(time.time() * 1000) % 100000 + i)  # 毎回異なるシード
                    gan_sample = self.gan_model.generate_samples(1)[0]
                
                    # ★ 数字にランダム性を追加（例：温度スケーリング）
                    logits = gan_sample / 0.7  # "温度" を下げるとシャープに、高くすると多様に
                    probs = logits / logits.sum()
                    numbers = np.random.choice(37, 7, replace=False, p=probs)
                    
                    append_prediction(np.sort(numbers + 1), base_confidence=0.8)

            if self.ppo_model:
                for i in range(num_candidates):
                    set_global_seed(random.randint(1000, 999999))  # 🔁 シードを毎回変更
                    obs = np.zeros(37, dtype=np.float32)
                
                    # 多様性確保のため deterministic=False に変更
                    action, _ = self.ppo_model.predict(obs, deterministic=False)
                
                    numbers = np.argsort(action)[-7:] + 1
                    append_prediction(np.sort(numbers), base_confidence=0.85)

            if self.diffusion_model:
                from diffusion_module import sample_diffusion_ddpm
                print("[INFO] Diffusion モデルによる生成を開始")
            
                for i in range(num_candidates):
                    set_global_seed(random.randint(1000, 999999))  # 🔁 乱数シードを毎回変える
            
                    try:
                        sample = sample_diffusion_ddpm(
                            self.diffusion_model,
                            self.diffusion_betas,
                            self.diffusion_alphas_cumprod,
                            dim=37,
                            num_samples=1  # ★ 1件ずつ生成して多様性を確保
                        )[0]
            
                        numbers = np.argsort(sample)[-7:] + 1
                        numbers = np.sort(numbers)
                        append_prediction(numbers, base_confidence=0.84)
            
                    except Exception as e:
                        print(f"[WARNING] Diffusion 生成中にエラー: {e}")

            if self.gnn_model:
                from gnn_core import build_cooccurrence_graph
                print("[INFO] GNN推論を開始")
                graph_data = build_cooccurrence_graph(latest_data)
                self.gnn_model.eval()
                with torch.no_grad():
                    gnn_scores = self.gnn_model(graph_data.x, graph_data.edge_index).squeeze().numpy()
                    for i in range(num_candidates):
                        set_global_seed(400 + i)
                        numbers = np.argsort(gnn_scores)[-7:] + 1
                        append_prediction(sorted([int(n) for sub in numbers for n in (sub if isinstance(sub, (list, np.ndarray)) else [sub])]), base_confidence=0.83)

            if self.bnn_model:
                from bnn_module import predict_bayesian_regression
                print("[INFO] BNNモデルによる予測を実行中")
            
                for i in range(num_candidates):
                    set_global_seed(random.randint(1000, 999999))  # 🔁 毎回異なるシードで予測
            
                    try:
                        bnn_preds = predict_bayesian_regression(
                            self.bnn_model,
                            self.bnn_guide,
                            X,
                            samples=1  # 🔁 1サンプルずつ個別生成
                        )
            
                        for pred in bnn_preds:
                            pred = np.array(pred).flatten()
                            numbers = np.round(pred).astype(int)
                            numbers = np.clip(numbers, 1, 37)
                            numbers = np.unique(numbers)
            
                            # 必要なら不足分をランダム補完（BNNは被りが出やすいため）
                            while len(numbers) < 7:
                                add = random.randint(1, 37)
                                if add not in numbers:
                                    numbers = np.append(numbers, add)
            
                            numbers = np.sort(numbers[:7])  # 念のため7個制限
                            append_prediction(numbers, base_confidence=0.83)
            
                    except Exception as e:
                        print(f"[WARNING] BNN予測中にエラー発生: {e}")

            print(f"[INFO] 総予測候補数（全モデル統合）: {len(all_predictions)}件")
            numbers_only = [pred[0] for pred in all_predictions]
            confidence_scores = [pred[1] for pred in all_predictions]
            return numbers_only, confidence_scores

        except Exception as e:
            print(f"[ERROR] 予測中にエラー発生: {e}")
            traceback.print_exc()
            return None, None
        
def evaluate_predictions(predictions, actual_numbers):
    matches = []
    for pred in predictions:
        match_count = len(set(pred[0]) & set(actual_numbers))
        matches.append(match_count)
    return {
        'max_matches': max(matches),
        'avg_matches': np.mean(matches),
        'predictions_with_matches': list(zip(predictions, matches))
    }
# 追加: 最新の抽せん日を取得する関数
official_url = "https://www.takarakuji-official.jp/ec/loto7/?kujiprdShbt=61&knyschm=0"

async def fetch_drawing_dates():
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(official_url, timeout=10) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    drawing_dates = []
                    date_elements = soup.select("dl.m_param.m_thumbSet_row")
                    for dl in date_elements:
                        dt_element = dl.find("dt", string="抽せん日")
                        if dt_element:
                            dd_element = dt_element.find_next_sibling("dd")
                            if dd_element:
                                formatted_date = dd_element.text.strip().replace("/", "-")
                                drawing_dates.append(formatted_date)
                    
                    return drawing_dates
                else:
                    print(f"HTTPエラー {response.status}: {official_url}")
        except Exception as e:
            print(f"抽せん日取得エラー: {e}")
    return []

async def get_latest_drawing_dates():
    dates = await fetch_drawing_dates()
    return dates

def parse_number_string(number_str):
    """
    予測番号や当選番号の文字列をリスト化する関数
    - スペース / カンマ / タブ 区切りに対応
    - "07 15 20 28 29 34 36" → [7, 15, 20, 28, 29, 34, 36]
    - "[7, 15, 20, 28, 29, 34, 36]" → [7, 15, 20, 28, 29, 34, 36]
    """
    if pd.isna(number_str):
        return []  # NaN の場合は空リストを返す
    
    # 不要な記号を削除（リスト形式の場合）
    number_str = number_str.strip("[]").replace("'", "").replace('"', '')

    # スペース・カンマ・タブで分割し、整数変換
    numbers = re.split(r'[\s,]+', number_str)

    # 数字のみにフィルタリングして整数変換
    return [int(n) for n in numbers if n.isdigit()]

def classify_rank(main_match, bonus_match):
    """本数字一致数とボーナス一致数からLoto7の等級を判定"""
    if main_match == 7:
        return "1等"
    elif main_match == 6 and bonus_match >= 1:
        return "2等"
    elif main_match == 6:
        return "3等"
    elif main_match == 5:
        return "4等"
    elif main_match == 4:
        return "5等"
    elif main_match == 3 and bonus_match >= 1:
        return "6等"
    else:
        return "該当なし"
    
def calculate_precision_recall_f1(evaluation_df):
    y_true = []
    y_pred = []

    for _, row in evaluation_df.iterrows():
        actual = set(row["当選本数字"])
        predicted = set(row["予測番号"])
        for n in range(1, 38):
            y_true.append(1 if n in actual else 0)
            y_pred.append(1 if n in predicted else 0)

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("\n=== 評価指標 ===")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 Score:  {f1:.3f}")

def evaluate_prediction_accuracy_with_bonus(predictions_file="loto7_predictions.csv", results_file="loto7.csv"):
    """
    予測結果と実際の当選結果を比較し、ボーナス数字を考慮して精度を評価し、等級も判定する
    """
    try:
        # === ✅ 空ファイルや構文エラーでも安全に読み込む ===
        try:
            predictions_df = pd.read_csv(predictions_file, encoding='utf-8-sig')
            if predictions_df.empty or predictions_df.shape[0] == 0 or "抽せん日" not in predictions_df.columns:
                print(f"[WARNING] 予測ファイルが空か無効です: {predictions_file}")
                return None
        except Exception as read_err:
            print(f"[WARNING] 予測ファイルの読み込み失敗: {read_err}")
            return None

        results_df = pd.read_csv(results_file, encoding='utf-8-sig')
        evaluation_results = []

        for index, row in predictions_df.iterrows():
            draw_date = row["抽せん日"]
            actual_row = results_df[results_df["抽せん日"] == draw_date]
            if actual_row.empty:
                continue

            actual_numbers = parse_number_string(actual_row.iloc[0]["本数字"])
            actual_bonus = parse_number_string(actual_row.iloc[0]["ボーナス数字"])

            for i in range(1, 6):  # 予測1〜5
                pred_col = f"予測{i}"
                if pred_col not in row or pd.isna(row[pred_col]):
                    continue

                try:
                    predicted_numbers = set(parse_number_string(row[pred_col]))
                    main_match = len(predicted_numbers & set(actual_numbers))
                    bonus_match = len(predicted_numbers & set(actual_bonus))
                    rank = classify_rank(main_match, bonus_match)

                    evaluation_results.append({
                        "抽せん日": draw_date,
                        "予測番号": list(predicted_numbers),
                        "当選本数字": actual_numbers,
                        "当選ボーナス": actual_bonus,
                        "本数字一致数": main_match,
                        "ボーナス一致数": bonus_match,
                        "信頼度": row.get(f"信頼度{i}", None),
                        "等級": rank
                    })

                except Exception as e:
                    print(f"予測データ処理エラー (行 {index}, 予測 {i}): {e}")
                    continue

        evaluation_df = pd.DataFrame(evaluation_results)
        evaluation_df.to_csv("loto7_prediction_evaluation_with_bonus.csv", index=False, encoding='utf-8-sig')
        print("予測精度の評価結果を保存しました: loto7_prediction_evaluation_with_bonus.csv")

        # 統計出力
        print("\n=== 予測精度の統計情報 ===")
        if not evaluation_df.empty:
            print(f"最大本数字一致数: {evaluation_df['本数字一致数'].max()}")
            print(f"平均本数字一致数: {evaluation_df['本数字一致数'].mean():.2f}")
            print(f"最大ボーナス一致数: {evaluation_df['ボーナス一致数'].max()}")
            print(f"平均ボーナス一致数: {evaluation_df['ボーナス一致数'].mean():.2f}")
            print("\n--- 等級の分布 ---")
            print(evaluation_df['等級'].value_counts())
            # ✅ 評価指標を追加で表示
            calculate_precision_recall_f1(evaluation_df)
        else:
            print("評価データがありません。")
                # === 評価結果をテキストファイルに出力 ===
        try:
            with open("loto7_evaluation_summary.txt", "w", encoding="utf-8") as f:
                f.write("=== 予測精度の統計情報 ===\n")
                f.write(f"最大本数字一致数: {evaluation_df['本数字一致数'].max()}\n")
                f.write(f"平均本数字一致数: {evaluation_df['本数字一致数'].mean():.2f}\n")
                f.write(f"最大ボーナス一致数: {evaluation_df['ボーナス一致数'].max()}\n")
                f.write(f"平均ボーナス一致数: {evaluation_df['ボーナス一致数'].mean():.2f}\n\n")

                f.write("--- 等級の分布 ---\n")
                f.write(f"{evaluation_df['等級'].value_counts().to_string()}\n\n")

                # Precision/Recall/F1
                y_true = []
                y_pred = []

                for _, row in evaluation_df.iterrows():
                    actual = set(row["当選本数字"])
                    predicted = set(row["予測番号"])
                    for n in range(1, 38):
                        y_true.append(1 if n in actual else 0)
                        y_pred.append(1 if n in predicted else 0)

                precision = precision_score(y_true, y_pred)
                recall = recall_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred)

                f.write("=== 評価指標 ===\n")
                f.write(f"Precision: {precision:.3f}\n")
                f.write(f"Recall:    {recall:.3f}\n")
                f.write(f"F1 Score:  {f1:.3f}\n")
            print("評価結果を loto7_evaluation_summary.txt に保存しました。")
        except Exception as e:
            print(f"[WARNING] テキストファイル出力失敗: {e}")

        return evaluation_df

    except Exception as e:
        print(f"予測精度の評価エラー: {e}")
        return None

# 予測結果をCSVファイルに保存する関数
def save_predictions_to_csv(predictions, drawing_date, filename="loto7_predictions.csv"):
    drawing_date = pd.to_datetime(drawing_date).strftime("%Y-%m-%d")
    row = {"抽せん日": drawing_date}

    for i, (numbers, confidence) in enumerate(predictions[:5], 1):
        row[f"予測{i}"] = ', '.join(map(str, numbers))
        row[f"信頼度{i}"] = round(confidence, 3)

    df = pd.DataFrame([row])

    if os.path.exists(filename):
        try:
            existing_df = pd.read_csv(filename, encoding='utf-8-sig')
            if "抽せん日" not in existing_df.columns:
                print(f"警告: CSVに'抽せん日'列が見つかりません。新規作成します。")
                existing_df = pd.DataFrame(columns=["抽せん日"] + [f"予測{i}" for i in range(1, 6)] + [f"信頼度{i}" for i in range(1, 6)])
            existing_df = existing_df[existing_df["抽せん日"] != drawing_date]
            df = pd.concat([existing_df, df], ignore_index=True)
        except Exception as e:
            print(f"CSVファイルの読み込みエラー: {e}。新規作成します。")
            df = pd.DataFrame([row])

    df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"予測結果を {filename} に保存しました。")

def is_running_with_streamlit():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except ImportError:
        return False

def main_with_improved_predictions():
    set_global_seed(42)  # ★追加
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    drawing_dates = asyncio.run(get_latest_drawing_dates())
    latest_drawing_date = drawing_dates[0] if drawing_dates else "不明"
    print("最新の抽せん日:", latest_drawing_date)

    try:
        data = pd.read_csv("loto7.csv")
        data["抽せん日"] = pd.to_datetime(data["抽せん日"], errors='coerce')
        print("データ読み込み完了")
    except Exception as e:
        print(f"データ読み込みエラー: {e}")
        return

    accuracy_results = evaluate_prediction_accuracy_with_bonus("loto7_predictions.csv", "loto7.csv")
    if accuracy_results is not None and not accuracy_results.empty:
        print("過去の予測精度を評価しました。")

    X, _, _ = preprocess_data(data)
    input_size = X.shape[1] if X is not None else 10
    hidden_size = 128
    output_size = 7

    predictor = LotoPredictor(input_size, hidden_size, output_size)

    try:
        print("モデルの学習を開始...")
        predictor.train_model(data, accuracy_results=accuracy_results)
        print("モデルの学習完了")
    except Exception as e:
        print(f"モデル学習エラー: {e}")
        return

    if is_running_with_streamlit():
        st.title("ロト7予測AI")
        if st.button("予測を実行"):
            try:
                latest_data = data.tail(10)
                target_date = latest_data["抽せん日"].max()
                history_data = data[data["抽せん日"] < target_date]  # 🔥 未来リーク防止

                predictions, confidence_scores = predictor.predict(latest_data)

                if predictions is None:
                    print("[ERROR] 予測に失敗したため処理を中断します。")
                    return

                verified_predictions = verify_predictions(list(zip(predictions, confidence_scores)), history_data)

                save_self_predictions(verified_predictions)

                for i, (numbers, confidence) in enumerate(verified_predictions[:5], 1):
                    st.write(f"予測 {i}: {numbers} (信頼度: {confidence:.3f})")

                save_predictions_to_csv(verified_predictions, latest_drawing_date)

            except Exception as e:
                st.error(f"予測エラー: {e}")
    else:
        print("[INFO] Streamlit以外の実行環境検出。通常のコンソール出力で予測を実行します。")
        try:
            latest_data = data.tail(10)
            target_date = latest_data["抽せん日"].max()
            history_data = data[data["抽せん日"] < target_date]  # 🔥 未来リーク防止

            predictions, confidence_scores = predictor.predict(latest_data)

            if predictions is None:
                print("[ERROR] 予測に失敗したため処理を中断します。")
                return

            verified_predictions = verify_predictions(list(zip(predictions, confidence_scores)), history_data)

            save_self_predictions(verified_predictions)

            print("\n=== 予測結果 ===")
            for i, (numbers, confidence) in enumerate(verified_predictions[:5], 1):
                print(f"予測 {i}: {numbers} (信頼度: {confidence:.3f})")

            save_predictions_to_csv(verified_predictions, latest_drawing_date)

        except Exception as e:
            print(f"予測エラー: {e}")
    
def calculate_pattern_score(numbers, historical_data=None):
    score = 0
    odd_count = sum(1 for n in numbers if n % 2 != 0)
    if 2 <= odd_count <= 5:
        score += 1
    gaps = [numbers[i+1] - numbers[i] for i in range(len(numbers)-1)]
    if min(gaps) >= 2:
        score += 1
    total = sum(numbers)
    if 100 <= total <= 150:
        score += 1
    if max(numbers) - min(numbers) >= 15:
        score += 1
    return score

def plot_prediction_analysis(predictions, historical_data):
    plt.figure(figsize=(15, 10))
    
    # 予測番号の分布
    plt.subplot(2, 2, 1)
    all_predicted_numbers = [num for pred in predictions for num in pred[0]]
    plt.hist(all_predicted_numbers, bins=37, range=(1, 38), alpha=0.7)
    plt.title('予測番号の分布')
    plt.xlabel('数字')
    plt.ylabel('頻度')
    
    # 信頼度スコアの分布
    plt.subplot(2, 2, 2)
    confidence_scores = [pred[1] for pred in predictions]
    plt.hist(confidence_scores, bins=20, alpha=0.7)
    plt.title('信頼度スコアの分布')
    plt.xlabel('信頼度')
    plt.ylabel('頻度')
    
    # 過去の当選番号との比較
    plt.subplot(2, 2, 3)
    historical_numbers = [num for numbers in historical_data['本数字'] for num in numbers]
    plt.hist(historical_numbers, bins=37, range=(1, 38), alpha=0.5, label='過去の当選')
    plt.hist(all_predicted_numbers, bins=37, range=(1, 38), alpha=0.5, label='予測')
    plt.title('予測 vs 過去の当選')
    plt.xlabel('数字')
    plt.ylabel('頻度')
    plt.legend()
    
    # パターン分析
    plt.subplot(2, 2, 4)
    pattern_scores = [calculate_pattern_score(pred[0]) for pred in predictions]
    plt.scatter(range(len(pattern_scores)), pattern_scores, alpha=0.5)
    plt.title('予測パターンスコア')
    plt.xlabel('予測インデックス')
    plt.ylabel('パターンスコア')
    
    plt.tight_layout()
    plt.savefig('prediction_analysis.png')
    plt.close()

def generate_evolution_graph(log_file="evolution_log.txt", output_file="evolution_graph.png"):
    """
    evolution_log.txtを読み込んで進化グラフを生成・保存する
    """
    if not os.path.exists(log_file):
        print(f"[WARNING] 進化ログ {log_file} が見つかりません")
        return

    dates = []
    counts = []

    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                parts = line.strip().split(":")
                date_part = parts[0].strip()
                count_part = parts[2].strip()

                date = pd.to_datetime(date_part)
                count = int(count_part.split()[0])

                dates.append(date)
                counts.append(count)
            except Exception as e:
                print(f"[WARNING] ログパース失敗: {e}")
                continue

    if not dates:
        print("[WARNING] 進化ログに有効なデータがありません")
        return

    # --- グラフ描画 ---
    plt.figure(figsize=(10, 6))
    plt.plot(dates, counts, marker='o', linestyle='-', color='blue')
    plt.title("自己進化履歴（自己予測データ件数推移）")
    plt.xlabel("日時")
    plt.ylabel("自己予測データ件数")
    plt.grid(True)
    plt.tight_layout()

    # --- 保存 ---
    plt.savefig(output_file)
    plt.close()
    print(f"[INFO] 進化履歴グラフを保存しました: {output_file}")

def verify_predictions(predictions, historical_data, top_k=5):
    def check_number_constraints(numbers):
        """予測数字配列の制約チェック"""
        return (
            isinstance(numbers, (list, np.ndarray)) and
            len(numbers) == 7 and
            len(np.unique(numbers)) == 7 and
            np.all((np.array(numbers) >= 1) & (np.array(numbers) <= 37))
        )

    def get_high_match_templates(historical_df, match_threshold=6):
        """過去の6本一致テンプレートを抽出"""
        unique_sets = set()
        rows = historical_df['本数字'].apply(lambda x: set(map(int, x)) if isinstance(x, list) else set())
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                intersect = rows[i] & rows[j]
                if len(intersect) >= match_threshold:
                    unique_sets.add(tuple(sorted(intersect)))
        return [set(t) for t in unique_sets]

    def penalize_overused_numbers(preds, threshold=0.05):
        """頻出数字を含む予測の信頼度を下げる"""
        all_nums = [n for pred in preds for n in pred[0]]
        freq = pd.Series(all_nums).value_counts(normalize=True)
        penalized = []
        for nums, conf in preds:
            penalty = sum(freq.get(n, 0) > threshold for n in nums) * 0.1
            penalized.append((nums, conf * (1 - penalty)))
        return penalized

    print("[INFO] 予測候補をフィルタリング中...")

    # --- 有効予測のみ抽出 ---
    valid_predictions = [
        (np.sort(pred), conf)
        for pred, conf in predictions
        if check_number_constraints(pred)
    ]
    if not valid_predictions:
        print("[WARNING] 有効な予測がありません")
        return []

    # --- 頻出数字ペナルティ ---
    valid_predictions = penalize_overused_numbers(valid_predictions)

    # --- 信頼度順に100件まで絞る ---
    valid_predictions.sort(key=lambda x: x[1], reverse=True)
    candidates = valid_predictions[:100]

    # --- カバレッジ最大化で top_k - 2 選抜 ---
    selected, used_numbers, used_flags = [], set(), [False] * len(candidates)
    while len(selected) < (top_k - 2):
        best_score, best_idx = -1, -1
        for idx, (nums, conf) in enumerate(candidates):
            if used_flags[idx]:
                continue
            combined = used_numbers | set(nums)
            coverage_score = len(combined)
            score = (coverage_score * 0.8) + (conf * 0.2)  # ⬅ カバレッジ重視に調整
            if score > best_score:
                best_score, best_idx = score, idx
        if best_idx == -1:
            break
        selected.append(candidates[best_idx])
        used_numbers.update(candidates[best_idx][0])
        used_flags[best_idx] = True

    # --- 強制6本構成テンプレート追加（最大2件） ---
    try:
        print("[INFO] 強制テンプレート構成の探索中...")
        historical_data['本数字'] = historical_data['本数字'].apply(
            lambda x: list(map(int, x)) if isinstance(x, list) else []
        )
        templates = get_high_match_templates(historical_data)

        added = 0
        tried = set()
        while added < 2 and len(tried) < len(templates):
            base = set(random.choice(templates))
            if tuple(sorted(base)) in tried:
                continue
            tried.add(tuple(sorted(base)))

            available = list(set(range(1, 38)) - base)
            if len(base) >= 6 and available:
                base = set(random.sample(base, 6))  # 6個だけ残す
                base.add(random.choice(available))  # 1個追加して7個に
                combo = np.sort(list(base))
                selected.append((combo, 1.0))
                added += 1

        if added > 0:
            print(f"[INFO] 強制6本構成を {added} 件追加しました")
        else:
            print("[INFO] 強制テンプレート構成は見つかりませんでした")
    except Exception as e:
        print(f"[WARNING] 強制構成作成中にエラー発生: {e}")

    print(f"[INFO] 最終選択された予測数: {len(selected)}")
    return selected


def extract_strong_features(evaluation_df, feature_df):
    """
    過去予測評価と特徴量を結合し、「本数字一致数」と相関の高い特徴量を抽出
    """
    # 🔒 入力データの検証
    if evaluation_df is None or evaluation_df.empty:
        print("[WARNING] 評価データが空のため、重要特徴量の抽出をスキップします。")
        return []

    if "抽せん日" not in evaluation_df.columns:
        print("[WARNING] 評価データに '抽せん日' 列が存在しません。重要特徴量の抽出をスキップします。")
        return []

    if feature_df is None or feature_df.empty or "抽せん日" not in feature_df.columns:
        print("[WARNING] 特徴量データが無効または '抽せん日' 列がありません。")
        return []

    # 🔧 日付型を明示的に揃える
    evaluation_df['抽せん日'] = pd.to_datetime(evaluation_df['抽せん日'], errors='coerce')
    feature_df['抽せん日'] = pd.to_datetime(feature_df['抽せん日'], errors='coerce')

    # ⛓ 結合
    merged = evaluation_df.merge(feature_df, on="抽せん日", how="inner")
    if merged.empty:
        print("[WARNING] 評価データと特徴量データの結合結果が空です。")
        return []

    # 📊 相関計算
    correlations = {}
    for col in feature_df.columns:
        if col in ["抽せん日", "本数字", "ボーナス数字"]:
            continue
        try:
            if not np.issubdtype(merged[col].dtype, np.number):
                continue
            corr = np.corrcoef(merged[col], merged["本数字一致数"])[0, 1]
            correlations[col] = abs(corr)
        except Exception:
            continue

    # 🔝 上位5特徴量を返す
    top_features = sorted(correlations.items(), key=lambda x: -x[1])[:5]
    return [f[0] for f in top_features]

def reinforce_features(X, feature_names, important_features, multiplier=1.5):
    """
    指定された重要特徴量を強調（値を倍率で増強）
    """
    reinforced_X = X.copy()
    for feat in important_features:
        if feat in feature_names:
            idx = feature_names.index(feat)
            reinforced_X[:, idx] *= multiplier
    return reinforced_X

# --- 🔥 新規追加関数 ---
def extract_high_match_patterns(dataframe, min_match=6):
    """過去データから高一致パターンだけ抽出"""
    high_match_combos = []
    for idx1, row1 in dataframe.iterrows():
        nums1 = set(row1['本数字'])
        for idx2, row2 in dataframe.iterrows():
            if idx1 >= idx2:
                continue
            nums2 = set(row2['本数字'])
            if len(nums1 & nums2) >= min_match:
                high_match_combos.append(sorted(nums1))
    return high_match_combos

def calculate_number_frequencies(dataframe):
    """過去データから番号出現頻度スコアを計算"""
    all_numbers = [num for nums in dataframe['本数字'] for num in nums]
    freq = pd.Series(all_numbers).value_counts().to_dict()
    return freq

def calculate_number_cycle_score(dataframe):
    """
    数字ごとの未出現期間（周期）に基づいたスコアを計算する。
    直近で出ていないほど高いスコアを付ける。
    """
    last_seen = {}
    today_index = len(dataframe)

    for idx, row in dataframe[::-1].iterrows():
        for number in row['本数字']:
            if number not in last_seen:
                last_seen[number] = today_index - idx  # 今何回分前に出たか

    # 最大未出周期 = 高スコアとするため反転（例: 未出日数が長いほど高得点）
    max_cycle = max(last_seen.values()) if last_seen else 1
    score = {n: last_seen.get(n, max_cycle) for n in range(1, 38)}
    return score

        
def bulk_predict_all_past_draws():
    set_global_seed(42)
    df = pd.read_csv("loto7.csv")
    df["抽せん日"] = pd.to_datetime(df["抽せん日"], errors='coerce')
    df = df.sort_values("抽せん日").reset_index(drop=True)
    print("[INFO] 抽せんデータ読み込み完了:", len(df), "件")

    pred_file = "loto7_predictions.csv"

    skip_dates = set()
    if os.path.exists(pred_file):
        try:
            pred_df = pd.read_csv(pred_file, encoding='utf-8-sig')
            if "抽せん日" in pred_df.columns:
                skip_dates = set(pd.to_datetime(pred_df["抽せん日"], errors='coerce').dropna().dt.strftime("%Y-%m-%d"))
        except Exception as e:
            print(f"[WARNING] 予測ファイル読み込みエラー: {e}")
    else:
        with open(pred_file, "w", encoding="utf-8-sig") as f:
            f.write("抽せん日,予測1,信頼度1,予測2,信頼度2,予測3,信頼度3,予測4,信頼度4,予測5,信頼度5\n")

    predictor_cache = {}

    for i in range(10, len(df)):
        set_global_seed(1000 + i)

        test_date = df.iloc[i]["抽せん日"]
        test_date_str = test_date.strftime("%Y-%m-%d")

        if test_date_str in skip_dates:
            print(f"[INFO] 既に予測済み: {test_date_str} → スキップ")
            continue

        print(f"\n=== {test_date_str} の予測を開始 ===")
        train_data = df.iloc[:i].copy()
        latest_data = df.iloc[i-10:i].copy()

        X, _, _ = preprocess_data(train_data)
        if X is None:
            print(f"[WARNING] {test_date_str} の学習データが無効です")
            continue

        input_size = X.shape[1]

        if i % 50 == 0 or input_size not in predictor_cache:
            print(f"[INFO] モデル再学習: {test_date_str} 時点")
            predictor = LotoPredictor(input_size, 128, 7)
            predictor.train_model(train_data)
            predictor_cache[input_size] = predictor
        else:
            predictor = predictor_cache[input_size]

        predictions, confidence_scores = predictor.predict(latest_data)
        if predictions is None:
            print(f"[ERROR] {test_date_str} の予測に失敗しました")
            continue

        verified_predictions = verify_predictions(list(zip(predictions, confidence_scores)), train_data)
        save_self_predictions(verified_predictions)
        save_predictions_to_csv(verified_predictions, test_date)
        git_commit_and_push("loto7_predictions.csv", "Auto update loto7_predictions.csv [skip ci]")

        model_dir = f"models/{test_date_str}"
        _save_all_models_no_self(predictor, model_dir)

        evaluate_prediction_accuracy_with_bonus("loto7_predictions.csv", "loto7.csv")

    # === 🆕 未来1回分の予測を追加 ===
    try:
        future_date = df["抽せん日"].max() + pd.Timedelta(days=7)
        future_date_str = future_date.strftime("%Y-%m-%d")

        if future_date_str not in skip_dates:
            print(f"\n=== {future_date_str} の未来予測を開始 ===")
            latest_data = df.tail(10).copy()
            train_data = df.copy()

            X, _, _ = preprocess_data(train_data)
            if X is None:
                print("[WARNING] 未来予測用の学習データが無効です")
            else:
                input_size = X.shape[1]
                if input_size not in predictor_cache:
                    predictor = LotoPredictor(input_size, 128, 7)
                    predictor.train_model(train_data)
                    predictor_cache[input_size] = predictor
                else:
                    predictor = predictor_cache[input_size]

                predictions, confidence_scores = predictor.predict(latest_data)
                if predictions is not None:
                    verified_predictions = verify_predictions(list(zip(predictions, confidence_scores)), train_data)
                    save_self_predictions(verified_predictions)
                    save_predictions_to_csv(verified_predictions, future_date)
                    git_commit_and_push("loto7_predictions.csv", "Auto predict future draw [skip ci]")
                    print(f"[INFO] 未来予測（{future_date_str}）完了")
        else:
            print(f"[INFO] 未来予測（{future_date_str}）は既に実行済みです")

    except Exception as e:
        print(f"[WARNING] 未来予測中にエラー発生: {e}")
        traceback.print_exc()

    print("\n=== 一括予測とモデル保存・評価が完了しました ===")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    bulk_predict_all_past_draws()


def log_prediction_summary(evaluation_df, log_path="prediction_accuracy_log.txt"):
    if evaluation_df is None or evaluation_df.empty:
        print("[WARNING] 評価データが空です")
        return

    with open(log_path, "a", encoding="utf-8") as f:
        for _, row in evaluation_df.iterrows():
            date = row["抽せん日"]
            match = row["本数字一致数"]
            bonus = row["ボーナス一致数"]
            rank = row["等級"]
            confidence = row.get("信頼度", "N/A")
            pred = row["予測番号"]

            f.write(f"{date}: 一致={match}本, ボーナス={bonus}, 等級={rank}, 信頼度={confidence}, 番号={pred}\n")

    print(f"[INFO] 予測精度履歴を {log_path} に追記しました")


# === 再学習サマリ・進化ログ（追記） ==========================================
import csv
from datetime import datetime
import subprocess

def _get_git_head():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "N/A"

def summarize_evaluation(evaluation_df):
    """
    evaluate_prediction_accuracy_with_bonus(...) が返す DataFrame を要約して dict を返す
    期待列: ["本数字一致数","ボーナス一致数","等級","当選本数字","予測番号"]
    """
    try:
        import pandas as pd  # 確実に pd があるように
    except Exception:
        pass
    if evaluation_df is None or len(evaluation_df) == 0:
        return None

    s = {}
    s["eval_rows"]        = int(len(evaluation_df))
    if "本数字一致数" in evaluation_df.columns:
        s["max_main_match"]   = int(evaluation_df["本数字一致数"].max())
        s["avg_main_match"]   = float(evaluation_df["本数字一致数"].mean())
    if "ボーナス一致数" in evaluation_df.columns:
        s["max_bonus_match"]  = int(evaluation_df["ボーナス一致数"].max())
        s["avg_bonus_match"]  = float(evaluation_df["ボーナス一致数"].mean())

    # 等級分布
    if "等級" in evaluation_df.columns:
        counts = evaluation_df["等級"].value_counts()
        for r in ["1等","2等","3等","4等","5等","6等","該当なし"]:
            s[f"rank_{r}"] = int(counts.get(r, 0))

    # Precision / Recall / F1 を番号レベルで算出（0除算は0扱い）
    y_true, y_pred = [], []
    if "当選本数字" in evaluation_df.columns and "予測番号" in evaluation_df.columns:
        for _, row in evaluation_df.iterrows():
            actual = set(row["当選本数字"]) if isinstance(row["当選本数字"], (list,set)) else set(row.get("当選本数字_list", []))  # 保険
            predicted = set(row["予測番号"]) if isinstance(row["予測番号"], (list,set)) else set(row.get("予測番号_list", []))
            # LOTO7 の番号範囲（1..37）に対して
            for n in range(1, 38):
                y_true.append(1 if n in actual else 0)
                y_pred.append(1 if n in predicted else 0)
        try:
            from sklearn.metrics import precision_score, recall_score, f1_score
            s["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
            s["recall"]    = float(recall_score(y_true, y_pred, zero_division=0))
            s["f1"]        = float(f1_score(y_true, y_pred, zero_division=0))
        except Exception:
            s["precision"] = s["recall"] = s["f1"] = 0.0
    return s

def write_evolution_log(evaluation_df, data_max_date=None, seed=None, model_dir=None,
                        top_features=None, csv_path="logs/evolution.csv", json_dir="logs/evolution_detail"):
    """
    再学習1回につきCSVに1行追記し、同内容をJSONスナップショットでも保存
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    summary = summarize_evaluation(evaluation_df) or {}
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    head = _get_git_head()

    row = {
        "timestamp": now,
        "data_max_date": str(data_max_date) if data_max_date is not None else "N/A",
        "seed": seed if seed is not None else "",
        "model_dir": model_dir if model_dir is not None else "",
        "git_head": head,
        **summary
    }
    if top_features:
        row["top_features"] = "|".join(map(str, top_features))

    # CSV追記（ヘッダ自動）
    write_header = not os.path.exists(csv_path)
    header = list(row.keys())
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header:
            w.writeheader()
        w.writerow(row)

    # JSONスナップショット
    json_name = datetime.now().strftime("%Y%m%d_%H%M%S") + ".json"
    with open(os.path.join(json_dir, json_name), "w", encoding="utf-8") as jf:
        import json as _json
        _json.dump(row, jf, ensure_ascii=False, indent=2)

    print(f"[LOG] 再学習サマリを保存: {csv_path} / {json_name}")
    return row

def generate_evolution_graph_from_csv(csv_path="logs/evolution.csv", metric="f1", output_file="logs/evolution_graph.png"):
    """
    evolution.csv（write_evolution_logで作成）から metric の推移グラフを作成
    """
    try:
        import pandas as _pd
        import matplotlib.pyplot as plt
    except Exception as e:
        print("[WARNING] 可視化用ライブラリの読み込みに失敗:", e)
        return

    if not os.path.exists(csv_path):
        print(f"[WARNING] 進化CSV {csv_path} が見つかりません")
        return

    df = _pd.read_csv(csv_path, encoding="utf-8")
    if "timestamp" not in df.columns or metric not in df.columns:
        print(f"[WARNING] CSVに必要な列がありません: timestamp / {metric}")
        return

    df["timestamp"] = _pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp")
    plt.figure(figsize=(10,6))
    plt.plot(df["timestamp"], df[metric], marker="o")
    plt.title(f"自己進化履歴（{metric} 推移）")
    plt.xlabel("日時")
    plt.ylabel(metric.upper())
    plt.grid(True)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"[LOG] 進化グラフを保存: {output_file}")
# === /再学習サマリ・進化ログ ここまで ==========================================
