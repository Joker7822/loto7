
# -*- coding: utf-8 -*-
"""
A lean, dependency-light rewrite of lottery_prediction.py for research/entertainment.

Design goals:
- Deterministic, reproducible training (set seeds).
- Multi-label formulation: predict P(n in 当選本数字) for n=1..37.
- OneVsRestClassifier(LogisticRegression) baseline + optional RandomForest.
- Time-aware split: train on past, validate on recent draws.
- Top-k (k=7) selection by probability to form candidate tickets; optional coverage diversification.
- Robust CSV parsing compatible with '抽せん日','本数字','ボーナス数字' formats used in your files.
- CLI: --input, --pred-out, --n-candidates, --latest-window.
- No heavy libs (GAN/PPO/AutoGluon/ONNX/Streamlit removed).

Disclaimer: Lottery draws are designed to be random; this code does NOT increase win odds beyond chance.
"""
import argparse
import os
import random
from datetime import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

def re_split(s: str) -> List[str]:
    import re
    return [t for t in re.split(r"[\s,、]+", s) if t]

def parse_number_string(number_str):
    """'7, 15, 20, 28, 29, 34, 36' や '[7 15 20 28 29 34 36]' などを整数配列に。
    NaN/空は [] を返す。
    """
    if pd.isna(number_str):
        return []
    s = str(number_str).strip().strip("[]").replace("'", "").replace('"', "")
    parts = [p for p in re_split(s)]
    nums = []
    for p in parts:
        try:
            n = int(p)
            if 1 <= n <= 37:
                nums.append(n)
        except Exception:
            continue
    return nums

def numbers_to_vec(nums: List[int], size: int = 37) -> np.ndarray:
    vec = np.zeros(size, dtype=int)
    for n in nums:
        if 1 <= n <= size:
            vec[n-1] = 1
    return vec

def create_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler, pd.DataFrame]:
    """数値特徴量を作成しスケーリング。y は 37 次元の 0/1 ベクトル。"""
    work = df.copy()
    work["抽せん日"] = pd.to_datetime(work["抽せん日"], errors="coerce")
    work = work.dropna(subset=["抽せん日"])
    work = work.sort_values("抽せん日").reset_index(drop=True)

    work["本数字"] = work["本数字"].apply(parse_number_string)
    work["ボーナス数字"] = work["ボーナス数字"].apply(parse_number_string)

    # フィルタ：本数字が7個の行のみ
    work = work[work["本数字"].apply(lambda x: isinstance(x, list) and len(x) == 7)]

    if work.empty:
        raise ValueError("入力データが空、または '本数字' の整形に失敗しました。"

)
    # --- 基本統計特徴
    arr = np.vstack(work["本数字"].values)
    diffs = np.diff(np.sort(arr, axis=1), axis=1)

    feats = pd.DataFrame(index=work.index)
    feats["奇数比"] = (arr % 2 != 0).sum(axis=1) / 7.0
    feats["合計"] = arr.sum(axis=1)
    feats["レンジ"] = arr.max(axis=1) - arr.min(axis=1)
    feats["標準偏差"] = arr.std(axis=1)
    feats["連番数"] = (diffs == 1).sum(axis=1)
    feats["最小間隔"] = diffs.min(axis=1)
    feats["最大間隔"] = diffs.max(axis=1)
    feats["曜日"] = work["抽せん日"].dt.dayofweek
    feats["月"] = work["抽せん日"].dt.month
    feats["年"] = work["抽せん日"].dt.year
    feats["平均"] = arr.mean(axis=1)

    # 直近5回出現率
    counts5 = []
    for i, row in work.iterrows():
        recent = work.loc[:i-1].tail(5)
        recent_nums = [n for r in recent["本数字"] for n in r] if not recent.empty else []
        counts5.append(sum(n in recent_nums for n in row["本数字"]) / 7.0)
    feats["直近5回出現率"] = counts5

    # y 作成
    y = np.vstack([numbers_to_vec(nums=nums, size=37) for nums in work["本数字"].values])

    # スケーリング
    scaler = MinMaxScaler()
    X = scaler.fit_transform(feats.values.astype(float))

    return X, y, scaler, work[["抽せん日", "本数字"]]

def time_aware_split(X, y, dates: pd.Series, test_ratio: float = 0.2):
    """時間順に訓練/テストへ分割。"""
    n = len(dates)
    split = int(round(n * (1 - test_ratio)))
    return X[:split], X[split:], y[:split], y[split:], dates.iloc[:split], dates.iloc[split:]

def fit_models(X_train, y_train, algo: str = "logreg"):
    if algo == "rf":
        base = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=RANDOM_SEED, n_jobs=-1)
    else:
        # ロジスティック回帰（L2）、クラス重みで不均衡対策
        base = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs", n_jobs=-1)
    clf = OneVsRestClassifier(base, n_jobs=-1)
    clf.fit(X_train, y_train)
    return clf

def evaluate_multilabel(clf, X_test, y_test) -> dict:
    proba = getattr(clf, "predict_proba", None)
    if proba is not None:
        P_list = clf.predict_proba(X_test)
        # For OneVsRest with RF, predict_proba returns list of arrays [n_samples,2] per class
        if isinstance(P_list, list):
            P = np.vstack([p[:,1] for p in P_list]).T
            y_pred = (P >= 0.5).astype(int)
        else:
            y_pred = (P_list >= 0.5).astype(int)
    else:
        y_pred = clf.predict(X_test)

    precision = precision_score(y_test.ravel(), y_pred.ravel(), zero_division=0)
    recall = recall_score(y_test.ravel(), y_pred.ravel(), zero_division=0)
    f1 = f1_score(y_test.ravel(), y_pred.ravel(), zero_division=0)
    return {"precision": precision, "recall": recall, "f1": f1}

def pick_ticket_from_probs(prob_vec: np.ndarray, k: int = 7) -> List[int]:
    """確率上位から被りを避けつつ7個選択。"""
    order = np.argsort(-prob_vec)  # high -> low
    chosen = []
    for idx in order:
        n = idx + 1
        if n not in chosen:
            chosen.append(n)
        if len(chosen) == k:
            break
    return sorted(chosen)

def diversify_tickets(prob_matrix: np.ndarray, n_tickets: int = 10, k: int = 7) -> List[List[int]]:
    """カバレッジを広げる簡易版グリーディ分散選択。"""
    tickets = []
    used = set()
    for _ in range(n_tickets):
        # 未使用の番号をやや優先
        bonus = np.array([0.1 if (i+1) not in used else 0.0 for i in range(prob_matrix.shape[1])])
        vec = prob_matrix.mean(axis=0) + bonus
        t = pick_ticket_from_probs(vec, k=k)
        tickets.append(t)
        used.update(t)
    return tickets

def save_predictions(drawing_date: str, tickets: List[List[int]], out_csv: str):
    row = {"抽せん日": drawing_date}
    for i, nums in enumerate(tickets[:5], 1):
        row[f"予測{i}"] = ", ".join(map(str, nums))
        row[f"信頼度{i}"] = round(1.0, 3)  # 本実装では確率の平均などに置換可
    df = pd.DataFrame([row])
    if os.path.exists(out_csv):
        try:
            exist = pd.read_csv(out_csv, encoding="utf-8-sig")
            if "抽せん日" in exist.columns:
                exist = exist[exist["抽せん日"] != drawing_date]
            df = pd.concat([exist, df], ignore_index=True)
        except Exception:
            pass
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

def main(args):
    # 入力読込
    src = pd.read_csv(args.input, encoding="utf-8-sig")
    required = {"抽せん日", "本数字", "ボーナス数字"}
    if not required.issubset(src.columns):
        raise ValueError("入力CSVに '抽せん日','本数字','ボーナス数字' 列が必要です。")

    X, y, scaler, meta = create_features(src)
    X_tr, X_te, y_tr, y_te, d_tr, d_te = time_aware_split(X, y, meta["抽せん日"], test_ratio=0.2)

    clf = fit_models(X_tr, y_tr, algo=args.algo)
    metrics = evaluate_multilabel(clf, X_te, y_te)

    # 直近ウィンドウから予測用特徴を作成（最後の latest_window 行）
    latest_window = max(1, args.latest_window)
    last_idx = len(meta) - latest_window
    last_idx = max(0, last_idx)
    X_latest = X[last_idx:]
    if X_latest.size == 0:
        X_latest = X[-1:]
    # 確率
    proba = getattr(clf, "predict_proba", None)
    if proba is not None:
        P_list = clf.predict_proba(X_latest)
        if isinstance(P_list, list):
            P = np.vstack([p[:,1] for p in P_list]).T
        else:
            P = P_list
    else:
        dec = clf.decision_function(X_latest)
        P = 1 / (1 + np.exp(-dec))

    # チケット生成
    tickets = diversify_tickets(P, n_tickets=args.n_candidates, k=7)

    # 抽せん日（最新行の日付 or 今日）
    try:
        last_date = str(meta["抽せん日"].iloc[-1].date())
    except Exception:
        last_date = datetime.now().strftime("%Y-%m-%d")

    save_predictions(last_date, tickets, args.pred_out)

    # 結果表示
    print("=== 評価（テスト期間）===")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print(f"F1:        {metrics['f1']:.3f}")
    print(f"予測を保存しました: {args.pred_out}")
    print("※ 抽選は本質的にランダムです。本コードは研究・娯楽目的のベースライン実装です。")


if __name__ == "__main__":
    import re
    parser = argparse.ArgumentParser(description="Loto7 Multi-label Baseline (research/entertainment)")
    parser.add_argument("--input", default="loto7.csv", help="入力CSV（抽せん日,本数字,ボーナス数字）") 
    parser.add_argument("--pred-out", default="loto7_predictions.csv", help="出力CSV（予測保存先）")
    parser.add_argument("--n-candidates", type=int, default=10, help="生成する候補チケット数")
    parser.add_argument("--latest-window", type=int, default=10, help="直近何件のデータを予測入力に使うか")
    parser.add_argument("--algo", choices=["logreg", "rf"], default="logreg", help="学習アルゴリズム")
    args = parser.parse_args()
    main(args)
