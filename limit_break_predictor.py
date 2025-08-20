#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LimitBreakPredictor: 既存の LotoPredictor を強化し、
- 進化的アルゴリズム（GA）
- 条件付き（制約付き）サンプリング（擬似的条件付きDiffusion/確率的生成）
- 多目的フィットネス（分布整合性・多様性・ルール適合度）
を統合して "限界突破" した候補生成を行うモジュール。

【使い方（単体実行）】
$ python limit_break_predictor.py

【使い方（既存コードと統合）】
from limit_break_predictor import LimitBreakPredictor, ConstraintConfig
lbp = LimitBreakPredictor()
final_preds = lbp.limit_break_predict(latest_data_df, n_out=50)
# CSV保存（次回の抽せん日を指定）
lbp.save_predictions(final_preds, drawing_date="2025-08-22", filename="loto7_predictions.csv")

# 予測をバックフィル（未保存日のみ埋める）
lbp.backfill_predictions(df_all, out_csv="loto7_predictions.csv", n_out=50)

# 最新分を一発で実行＆保存（自己予測CSV＋公開CSV）
lbp.run_and_save_latest(df_all, n_out=50, self_file="self_predictions.csv", out_csv="loto7_predictions.csv")
"""

from __future__ import annotations

import math
import random
import traceback
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Iterable

import numpy as np
import pandas as pd

# 既存実装から拝借（存在しない場合は安全にフォールバック）
try:
    from lottery_prediction import (
        LotoPredictor,
        preprocess_data,
        create_advanced_features,
        save_predictions_to_csv,
        set_global_seed,
        get_latest_drawing_dates,
    )
except Exception:
    LotoPredictor = object  # type: ignore
    def preprocess_data(df):
        # 最低限のフォールバック
        nums = df["本数字"].apply(lambda x: [int(v) for v in x] if isinstance(x, (list, tuple)) else [])
        X = pd.DataFrame({"mean": nums.apply(np.mean), "std": nums.apply(np.std)}).fillna(0.0).values
        return X, None, None
    def create_advanced_features(df):
        return df
    def save_predictions_to_csv(preds, drawing_date: str, filename: str = "loto7_predictions.csv"):
        row = {"抽せん日": drawing_date}
        for i, (nums, conf) in enumerate(preds[:5], 1):
            row[f"予測{i}"] = ", ".join(map(str, nums))
            row[f"信頼度{i}"] = round(float(conf), 3)
        pd.DataFrame([row]).to_csv(filename, index=False, encoding="utf-8-sig")
    def set_global_seed(seed: int = 42):
        random.seed(seed); np.random.seed(seed)

# 追加: 自己予測の保存/読込（Loto 側の実装を優先利用）
try:
    from lottery_prediction import save_self_predictions, load_self_predictions
except Exception:
    def save_self_predictions(predictions, file_path="self_predictions.csv", max_records=100):
        """フォールバック: 上位候補の番号だけをCSV末尾に追記（最大max_records行を維持）"""
        try:
            rows = [list(nums) for nums, _ in predictions]
            if os.path.exists(file_path):
                try:
                    existing = pd.read_csv(file_path, header=None)
                    rows = existing.values.tolist() + rows
                except Exception:
                    pass
            rows = rows[-max_records:]
            pd.DataFrame(rows).to_csv(file_path, index=False, header=False, encoding="utf-8-sig")
        except Exception as e:
            print(f"[WARN] save_self_predictions fallback failed: {e}")
    def load_self_predictions(file_path="self_predictions.csv", min_match_threshold=3, true_data=None):
        try:
            if not os.path.exists(file_path):
                return []
            df = pd.read_csv(file_path, header=None)
            return [row.dropna().astype(int).tolist() for _, row in df.iterrows()]
        except Exception as e:
            print(f"[WARN] load_self_predictions fallback failed: {e}")
            return []

import os

# ——————————————————————————————————————————————
# 制約（条件）設定
# ——————————————————————————————————————————————
@dataclass
class ConstraintConfig:
    # 奇数の個数レンジ（例：2〜5）
    odd_min: int = 2
    odd_max: int = 5
    # 合計値レンジ（例：100〜150）
    sum_min: int = 100
    sum_max: int = 150
    # 最小間隔（隣り合う差）
    min_gap: int = 0
    # 熱い数字（出現頻度が高い）を優先度重みとして使用
    hot_weight: float = 0.4
    # コールド数字（出現頻度が低い）を探索多様性として加味
    cold_weight: float = 0.2
    # 重複スコアの罰則
    duplicate_penalty: float = 0.3
    # 番号範囲
    low: int = 1
    high: int = 37

# ——————————————————————————————————————————————
# ヘルパー
# ——————————————————————————————————————————————
NumberSet = List[int]
PredWithScore = Tuple[NumberSet, float]


def _ensure_valid(numbers: Iterable[int], low: int = 1, high: int = 37) -> NumberSet:
    s = sorted(set(int(n) for n in numbers if low <= int(n) <= high))
    # 足りなければランダム補完
    while len(s) < 7:
        c = random.randint(low, high)
        if c not in s:
            s.append(c)
    # 多すぎれば先頭から切る
    return s[:7]


def _fitness_distribution(hist_df: pd.DataFrame, cand: NumberSet) -> float:
    """過去分布に近いか（だいたいの整合性）"""
    # 本数字の列を期待: "本数字" が配列 or "n1..n7"
    if "本数字" in hist_df.columns and isinstance(hist_df["本数字"].iloc[0], (list, tuple)):
        nums = hist_df["本数字"].explode().astype(int).to_list()
    else:
        cols = [c for c in hist_df.columns if c.startswith("n") and c[1:].isdigit()]
        nums = hist_df[cols].values.ravel().astype(int).tolist()
    if not nums:
        return 0.0
    hist = pd.Series(nums).value_counts(normalize=True)
    score = 0.0
    for n in cand:
        score += float(hist.get(n, 0.0))
    return score / len(cand)


def _fitness_diversity(pop: List[NumberSet]) -> float:
    """母集団全体の多様性（平均ハミング距離的なもの）"""
    if not pop:
        return 0.0
    dsum = 0.0
    cnt = 0
    for i in range(len(pop)):
        for j in range(i + 1, len(pop)):
            dsum += len(set(pop[i]).symmetric_difference(set(pop[j])))
            cnt += 1
    return dsum / max(cnt, 1)


def _fitness_rule(cand: NumberSet, cfg: ConstraintConfig) -> float:
    """ルール適合度（奇数個数・合計・最小間隔など）"""
    odds = sum(1 for x in cand if x % 2 == 1)
    if not (cfg.odd_min <= odds <= cfg.odd_max):
        return 0.0
    s = sum(cand)
    if not (cfg.sum_min <= s <= cfg.sum_max):
        return 0.0
    if cfg.min_gap > 0:
        arr = sorted(cand)
        gaps = [arr[i + 1] - arr[i] for i in range(len(arr) - 1)]
        if gaps and min(gaps) < cfg.min_gap:
            return 0.0
    return 1.0


def _score_candidate(hist_df: pd.DataFrame, cand: NumberSet, cfg: ConstraintConfig) -> float:
    """多目的スコアを合算"""
    a = _fitness_distribution(hist_df, cand)
    b = _fitness_rule(cand, cfg)
    # ここでは多様性は個体単体では評価しづらいので簡易に b に寄与
    return 0.6 * a + 0.4 * b


def _mutate(cand: NumberSet, low: int, high: int, rate: float = 0.3) -> NumberSet:
    s = set(cand)
    for _ in range(7):
        if random.random() < rate:
            # ランダムに入替
            old = random.choice(list(s))
            s.remove(old)
            new = random.randint(low, high)
            s.add(new)
    return _ensure_valid(s, low, high)


def _cross(a: NumberSet, b: NumberSet, low: int, high: int) -> NumberSet:
    k = random.randint(2, 5)
    s = set(random.sample(a, k)) | set(random.sample(b, 7 - k))
    return _ensure_valid(s, low, high)


def _biased_sample_by_freq(hist_df: pd.DataFrame, k: int, low: int, high: int) -> NumberSet:
    """出現頻度にバイアスした初期解生成"""
    # 過去出現頻度
    if "本数字" in hist_df.columns and isinstance(hist_df["本数字"].iloc[0], (list, tuple)):
        nums = hist_df["本数字"].explode().astype(int).to_list()
    else:
        cols = [c for c in hist_df.columns if c.startswith("n") and c[1:].isdigit()]
        nums = hist_df[cols].values.ravel().astype(int).tolist()
    vc = pd.Series(nums).value_counts()
    pool = list(range(low, high + 1))
    weights = np.array([vc.get(i, 1) for i in pool], dtype=float)
    weights = weights / weights.sum()
    chosen = np.random.choice(pool, size=k, replace=False, p=weights)
    return sorted(map(int, chosen.tolist()))

# ——————————————————————————————————————————————
# コア: LimitBreakPredictor
# ——————————————————————————————————————————————
class LimitBreakPredictor(LotoPredictor):
    def __init__(self, cfg: Optional[ConstraintConfig] = None, seed: int = 42):
        super().__init__() if hasattr(super(), "__init__") else None
        self.cfg = cfg or ConstraintConfig()
        set_global_seed(seed)

    # ——— 主要入口 ———
    def limit_break_predict(self, df: pd.DataFrame, n_out: int = 50) -> List[PredWithScore]:
        """
        df: 履歴データ（"抽せん日" 列、"本数字" or n1..n7 列を含む）
        n_out: 生成する候補数
        """
        # 学習特徴の作成（既存関数があれば使う）
        try:
            _ = create_advanced_features(df)
        except Exception:
            pass

        # 過去の傾向に合う初期母集団を生成
        pop: List[NumberSet] = []
        for _ in range(max(32, n_out)):
            pop.append(_biased_sample_by_freq(df, 7, self.cfg.low, self.cfg.high))

        # GA 的に進化
        scored = [(c, _score_candidate(df, c, self.cfg)) for c in pop]
        for _ in range(4):  # 世代数
            # 上位個体をエリート選抜
            scored.sort(key=lambda x: x[1], reverse=True)
            elites = [c for c, _s in scored[: max(8, len(scored)//4)]]
            # 交叉＋突然変異
            children = []
            while len(children) + len(elites) < max(32, n_out):
                if random.random() < 0.5 and len(elites) >= 2:
                    a, b = random.sample(elites, 2)
                    child = _cross(a, b, self.cfg.low, self.cfg.high)
                else:
                    parent = random.choice(elites)
                    child = _mutate(parent, self.cfg.low, self.cfg.high, rate=0.3)
                children.append(child)
            pop = elites + children
            scored = [(c, _score_candidate(df, c, self.cfg)) for c in pop]

        # スコア順に整列し、重複を排除しながら n_out 件取り出す
        scored.sort(key=lambda x: x[1], reverse=True)
        uniq: List[PredWithScore] = []
        used = set()
        for cand, sc in scored:
            key = tuple(sorted(cand))
            if key in used:
                continue
            used.add(key)
            uniq.append((list(key), float(round(sc, 6))))
            if len(uniq) >= n_out:
                break
        return uniq

    # 既存の CSV 保存（上位5件を1行化）
    def save_predictions(self, preds: List[PredWithScore], drawing_date: str, filename: str = "loto7_predictions.csv"):
        save_predictions_to_csv(preds, drawing_date, filename=filename)

# ——————————————————————————————————————————————
# CSV 追記系（上位5件を1行化）／バックフィル
# ——————————————————————————————————————————————
def _make_prediction_row(preds, drawing_date: str):
    row = {"抽せん日": drawing_date}
    for i, (nums, conf) in enumerate(preds[:5], 1):
        row[f"予測{i}"] = ", ".join(map(str, nums))
        row[f"信頼度{i}"] = round(float(conf), 3)
    return row

def _append_predictions_row(filename: str, row: dict):
    # 既存があれば読み取り→行を追加→抽せん日で重複排除→日付昇順で保存
    df_new = pd.DataFrame([row])
    if os.path.exists(filename):
        try:
            df_old = pd.read_csv(filename, encoding="utf-8-sig")
        except Exception:
            df_old = pd.read_csv(filename)
        # 列の取り揃え
        cols = list(dict.fromkeys(df_old.columns.tolist() + df_new.columns.tolist()))
        df_old = df_old.reindex(columns=cols)
        df_new = df_new.reindex(columns=cols)
        df = pd.concat([df_old, df_new], ignore_index=True)
        # 抽せん日を正規化・重複排除
        if "抽せん日" in df.columns:
            df["抽せん日"] = pd.to_datetime(df["抽せん日"], errors="coerce")
            df = df.drop_duplicates(subset=["抽せん日"], keep="first").sort_values("抽せん日")
            df["抽せん日"] = df["抽せん日"].dt.strftime("%Y-%m-%d")
        df.to_csv(filename, index=False, encoding="utf-8-sig")
    else:
        df_new.to_csv(filename, index=False, encoding="utf-8-sig")

def backfill_predictions(self, full_data: pd.DataFrame, out_csv: str = "loto7_predictions.csv", n_out: int = 50):
    """
    既存の out_csv を見て、未保存日のみを対象に
    - その日 d の時点までの履歴（<= d）で limit_break_predict を実行
    - 1行形式で追記保存
    """
    df = full_data.copy()
    if "抽せん日" not in df.columns:
        raise ValueError("full_data に '抽せん日' 列が必要です")
    df["抽せん日"] = pd.to_datetime(df["抽せん日"], errors="coerce")
    df = df.dropna(subset=["抽せん日"]).sort_values("抽せん日")

    all_dates = df["抽せん日"].dt.date.unique().tolist()

    existing_dates = set()
    if os.path.exists(out_csv):
        try:
            exists_df = pd.read_csv(out_csv, encoding="utf-8-sig")
        except Exception:
            exists_df = pd.read_csv(out_csv)
        if "抽せん日" in exists_df.columns:
            try:
                existing_dates = set(pd.to_datetime(exists_df["抽せん日"]).dt.date.tolist())
            except Exception:
                pass

    # 未保存のみループ
    for d in all_dates:
        if d in existing_dates:
            continue
        # d 当日の予測を、d 以前のみを使って生成
        subset = df[df["抽せん日"] <= pd.Timestamp(d)]
        preds = self.limit_break_predict(subset, n_out=n_out)
        row = _make_prediction_row(preds, drawing_date=str(d))
        _append_predictions_row(out_csv, row)
        print(f"[INFO] 予測を追記: {d}")

# クラスにメソッドをアタッチ
setattr(LimitBreakPredictor, "backfill_predictions", backfill_predictions)

# ——————————————————————————————————————————————
# “最新一回分を実行して保存” ヘルパー
# ————————————————————
def run_and_save_latest(self,
                        full_data: pd.DataFrame,
                        drawing_date: Optional[str] = None,
                        n_out: int = 50,
                        self_file: str = "self_predictions.csv",
                        out_csv: str = "loto7_predictions.csv"):
    """最新日の直前までの履歴で予測→自己予測CSVと公開CSVに保存する。"""
    df = full_data.copy()
    if "抽せん日" not in df.columns:
        raise ValueError("full_data に '抽せん日' 列が必要です")
    df["抽せん日"] = pd.to_datetime(df["抽せん日"], errors="coerce")
    latest = df["抽せん日"].max()
    if pd.isna(latest):
        raise ValueError("最新日付が取得できませんでした")
    if drawing_date is None:
        drawing_date = latest.strftime("%Y-%m-%d")
    history = df[df["抽せん日"] < latest]
    if history.empty and len(df) >= 2:
        history = df.iloc[:-1].copy()
    if history.empty:
        raise ValueError("履歴データが不足しています（2行以上必要）")
    preds = self.limit_break_predict(history, n_out=n_out)
    try:
        save_self_predictions(preds, file_path=self_file)
    except Exception as e:
        print(f"[WARN] save_self_predictions 呼び出しに失敗: {e}")
    row = _make_prediction_row(preds, drawing_date=drawing_date)
    _append_predictions_row(out_csv, row)
    print(f"[INFO] 最新分を保存しました: 抽せん日={drawing_date}, 件数={len(preds)}")

# クラスにメソッドをアタッチ
setattr(LimitBreakPredictor, "run_and_save_latest", run_and_save_latest)


# ——————————————————————————————————————————————
# スクリプト実行時のサンプル
# ——————————————————————————————————————————————
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="loto7.csv", help="履歴CSV（抽せん日, 本数字 など）")
    parser.add_argument("--out", default="loto7_predictions.csv", help="出力CSV（上位5件を1行化）")
    parser.add_argument("--n_out", type=int, default=50, help="生成候補数")
    parser.add_argument("--seed", type=int, default=42, help="乱数シード")
    parser.add_argument("--mode", choices=["latest", "backfill"], default="latest",
                        help="latest: 最新1回分だけ実行＆保存 / backfill: 未保存日を一括バックフィル")
    args = parser.parse_args()

    set_global_seed(args.seed)
    try:
        df = pd.read_csv(args.csv, encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(args.csv)
    # "抽せん日" 列の正規化（文字列→日付）
    if "抽せん日" in df.columns:
        df["抽せん日"] = pd.to_datetime(df["抽せん日"], errors="coerce")

    lbp = LimitBreakPredictor(seed=args.seed)

    if args.mode == "latest":
        lbp.run_and_save_latest(df, n_out=args.n_out, self_file="self_predictions.csv", out_csv=args.out)
    else:
        lbp.backfill_predictions(df, out_csv=args.out, n_out=args.n_out)
