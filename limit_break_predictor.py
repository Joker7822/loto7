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
lbp.save_predictions(final_preds, drawing_date_str)

依存：lottery_prediction.py 内の LotoPredictor / 各種ユーティリティ（存在すれば自動で活用）
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
    min_gap: int = 2
    # 数字のレンジ（max - min がこの値以上）
    min_range: int = 15
    # 1..37 の範囲強制
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
    return sorted(s[:7])


def _odd_count(nums: NumberSet) -> int:
    return sum(1 for n in nums if n % 2 != 0)


def _min_gap(nums: NumberSet) -> int:
    nums = sorted(nums)
    if len(nums) < 2:
        return 0
    return min(nums[i + 1] - nums[i] for i in range(len(nums) - 1))


def _range(nums: NumberSet) -> int:
    return max(nums) - min(nums)


def _within(v: float, lo: float, hi: float) -> float:
    """範囲内なら1.0、逸脱に応じて0へ線形減衰（クランプ）。"""
    if lo <= v <= hi:
        return 1.0
    d = min(abs(v - lo), abs(v - hi))
    width = max(1e-6, hi - lo)
    return max(0.0, 1.0 - d / width)


def constraint_score(nums: NumberSet, cfg: ConstraintConfig) -> float:
    """0〜1（高いほど良い）"""
    oc = _odd_count(nums)
    total = sum(nums)
    mg = _min_gap(nums)
    rg = _range(nums)
    s = 0.25 * _within(oc, cfg.odd_min, cfg.odd_max)
    s += 0.35 * _within(total, cfg.sum_min, cfg.sum_max)
    s += 0.20 * _within(mg, cfg.min_gap, 37)
    s += 0.20 * _within(rg, cfg.min_range, 37)
    return float(s)


def number_frequencies(historical_df: pd.DataFrame) -> Dict[int, float]:
    counts = {i: 0 for i in range(1, 38)}
    if "本数字" not in historical_df.columns:
        return counts
    for row in historical_df["本数字"]:
        if isinstance(row, (list, tuple)):
            for n in row:
                if 1 <= int(n) <= 37:
                    counts[int(n)] += 1
    total = sum(counts.values()) or 1
    return {k: v / total for k, v in counts.items()}


def pair_triple_frequencies(historical_df: pd.DataFrame) -> Tuple[Dict[Tuple[int, int], int], Dict[Tuple[int, int, int], int]]:
    pair_freq: Dict[Tuple[int, int], int] = {}
    triple_freq: Dict[Tuple[int, int, int], int] = {}
    if "本数字" not in historical_df.columns:
        return pair_freq, triple_freq
    for nums in historical_df["本数字"]:
        if not isinstance(nums, (list, tuple)):
            continue
        s = sorted(int(x) for x in nums)
        # ペア
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                p = (s[i], s[j])
                pair_freq[p] = pair_freq.get(p, 0) + 1
        # トリプル
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                for k in range(j + 1, len(s)):
                    t = (s[i], s[j], s[k])
                    triple_freq[t] = triple_freq.get(t, 0) + 1
    return pair_freq, triple_freq


def cooccurrence_score(nums: NumberSet, pair_freq, triple_freq) -> float:
    s = sorted(nums)
    # 正規化用に適当なスケール
    pf_sum = 0
    tf_sum = 0
    for i in range(7):
        for j in range(i + 1, 7):
            pf_sum += pair_freq.get((s[i], s[j]), 0)
    for i in range(7):
        for j in range(i + 1, 7):
            for k in range(j + 1, 7):
                tf_sum += triple_freq.get((s[i], s[j], s[k]), 0)
    # 対数圧縮して0〜1に押し込む（経験則）
    return float(1.0 - math.exp(-0.002 * (pf_sum + 0.5 * tf_sum)))


def diversity_penalty(nums: NumberSet, others: List[NumberSet]) -> float:
    if not others:
        return 0.0
    inters = []
    s = set(nums)
    for o in others:
        inters.append(len(s & set(o)))
    avg_inter = sum(inters) / len(inters)
    # 共通が多いほどペナルティ（0〜1に）
    return min(1.0, avg_inter / 7.0)


# ——————————————————————————————————————————————
# Evolutionary Search（GA）
# ——————————————————————————————————————————————
class EvolutionEngine:
    def __init__(self, cfg: ConstraintConfig, num_low: int = 1, num_high: int = 37):
        self.cfg = cfg
        self.low = num_low
        self.high = num_high

    def _fitness(
        self,
        cand: NumberSet,
        hist_df: pd.DataFrame,
        pair_freq,
        triple_freq,
        others: Optional[List[NumberSet]] = None,
    ) -> float:
        cscore = constraint_score(cand, self.cfg)
        co = cooccurrence_score(cand, pair_freq, triple_freq)
        div_pen = diversity_penalty(cand, others or [])
        # 多目的：制約・共起性を最大化、類似度ペナルティを最小化
        score = 0.55 * cscore + 0.45 * co - 0.25 * div_pen
        return float(score)

    def _crossover(self, a: NumberSet, b: NumberSet) -> NumberSet:
        # ランダムに一部を交叉（3〜4個をAから、残りをBから）
        k = random.randint(3, 4)
        part = set(random.sample(a, k))
        child = list(part)
        for x in b:
            if len(child) >= 7:
                break
            if x not in part:
                child.append(x)
        return _ensure_valid(child, self.low, self.high)

    def _mutate(self, x: NumberSet, num_freq: Dict[int, float]) -> NumberSet:
        y = x[:]
        m = random.randint(1, 2)
        for _ in range(m):
            idx = random.randrange(7)
            # 頻度に比例してサンプル（探索の偏りを与える）
            pool = list(range(self.low, self.high + 1))
            weights = np.array([num_freq.get(i, 1e-6) for i in pool], dtype=float)
            if weights.sum() <= 0:
                cand = random.randint(self.low, self.high)
            else:
                weights = weights / weights.sum()
                cand = int(np.random.choice(pool, p=weights))
            y[idx] = cand
        return _ensure_valid(y, self.low, self.high)

    def search(
        self,
        seed_population: List[NumberSet],
        hist_df: pd.DataFrame,
        generations: int = 40,
        pop_size: int = 120,
        elite: int = 12,
    ) -> List[NumberSet]:
        """GA 検索。各世代で母集団サイズを一定に保つよう修正。トーナメントは現在の個体数に基づく。"""
        set_global_seed(777)
        elite = max(1, min(elite, pop_size - 1))
        num_freq = number_frequencies(hist_df)
        pair_freq, triple_freq = pair_triple_frequencies(hist_df)

        # 初期集団（不足は頻度ガイドのランダムで補う）
        pop: List[NumberSet] = []
        pop.extend(_ensure_valid(s) for s in seed_population)
        while len(pop) < pop_size:
            pool = list(range(self.low, self.high + 1))
            weights = np.array([num_freq.get(i, 1e-6) for i in pool], dtype=float)
            weights = weights / (weights.sum() or 1)
            cand = list(np.random.choice(pool, size=7, replace=False, p=weights)) if weights is not None else list(np.random.choice(pool, size=7, replace=False))
            pop.append(_ensure_valid(cand, self.low, self.high))
        # 余剰があれば切り詰め
        if len(pop) > pop_size:
            pop = pop[:pop_size]

        for _gen in range(generations):
            N = len(pop)
            # 評価
            scores = [self._fitness(ind, hist_df, pair_freq, triple_freq, others=pop) for ind in pop]

            # エリート選択
            idxs = np.argsort(scores)[::-1]
            elites = [pop[i] for i in idxs[:elite]]

            # 親選択（現在の個体数 N ベース）
            parents: List[NumberSet] = []
            target_children = pop_size - elite
            if N < 4:
                # 個体が少ない場合はランダム選択にフォールバック
                while len(parents) < target_children:
                    parents.append(random.choice(pop))
                    parents.append(random.choice(pop))
            else:
                while len(parents) < target_children:
                    t = random.sample(range(N), k=4)
                    best = max(t, key=lambda i: scores[i])
                    parents.append(pop[best])
                    # 2体目も選んで交叉の多様性を確保
                    t2 = random.sample(range(N), k=4)
                    best2 = max(t2, key=lambda i: scores[i])
                    parents.append(pop[best2])

            # 交叉＋突然変異で target_children 件の子を作る
            children: List[NumberSet] = []
            i = 0
            while len(children) < target_children:
                a = parents[i % len(parents)]
                b = parents[(i + 1) % len(parents)]
                child = self._crossover(a, b)
                if random.random() < 0.9:
                    child = self._mutate(child, num_freq)
                children.append(child)
                i += 2

            pop = elites + children
            # 念のためサイズを固定
            if len(pop) > pop_size:
                pop = pop[:pop_size]
            elif len(pop) < pop_size:
                # ランダム補充
                while len(pop) < pop_size:
                    pool = list(range(self.low, self.high + 1))
                    weights = np.array([num_freq.get(i, 1e-6) for i in pool], dtype=float)
                    weights = weights / (weights.sum() or 1)
                    cand = list(np.random.choice(pool, size=7, replace=False, p=weights)) if weights is not None else list(np.random.choice(pool, size=7, replace=False))
                    pop.append(_ensure_valid(cand, self.low, self.high))

        # 最終スコアでソート
        final_scores = [self._fitness(ind, hist_df, pair_freq, triple_freq, others=[]) for ind in pop]
        order = np.argsort(final_scores)[::-1]
        return [pop[i] for i in order]


# ——————————————————————————————————————————————
# 擬似・条件付きサンプラー（Diffusion/GAN があれば活用、なければ確率サンプル）
# ——————————————————————————————————————————————
class ConditionalSampler:
    def __init__(self, cfg: ConstraintConfig):
        self.cfg = cfg

    def sample_with_constraints(
        self,
        base_predictor: Optional[LotoPredictor],
        hist_df: pd.DataFrame,
        n_samples: int = 100,
        accept_threshold: float = 0.75,
    ) -> List[NumberSet]:
        out: List[NumberSet] = []
        freq = number_frequencies(hist_df)
        pool = np.arange(self.cfg.low, self.cfg.high + 1)
        weights = np.array([freq.get(int(i), 1e-6) for i in pool], dtype=float)
        # 数値安定化：負値/NaN/Inf除去 → 正規化失敗時は一様分布
        weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
        s = weights.sum()
        if not np.isfinite(s) or s <= 0:
            weights = None  # → np.random.choice 側で一様分布
        else:
            weights = weights / s

        # 1) Diffusion 生成が使えるならそれを優先
        if base_predictor is not None and getattr(base_predictor, "diffusion_model", None) is not None:
            try:
                from diffusion_module import sample_diffusion_ddpm
                trials = 0
                while len(out) < n_samples and trials < n_samples * 10:
                    trials += 1
                    x = sample_diffusion_ddpm(
                        base_predictor.diffusion_model,
                        getattr(base_predictor, "diffusion_betas", None),
                        getattr(base_predictor, "diffusion_alphas_cumprod", None),
                        dim=37,
                        num_samples=1,
                    )[0]
                    # 上位7個を採用
                    nums = np.argsort(x)[-7:] + 1
                    nums = _ensure_valid(nums.tolist(), self.cfg.low, self.cfg.high)
                    if constraint_score(nums, self.cfg) >= accept_threshold:
                        out.append(nums)
            except Exception:
                pass

        # 2) Diffusion が使えない／足りない場合は確率サンプル
        while len(out) < n_samples:
            cand = list(np.random.choice(pool, size=7, replace=False, p=weights)) if weights is not None else list(np.random.choice(pool, size=7, replace=False))
            cand = _ensure_valid(cand, self.cfg.low, self.cfg.high)
            if constraint_score(cand, self.cfg) >= accept_threshold:
                out.append(cand)
        return out


# ——————————————————————————————————————————————
# 限界突破 Predictor（メイン）
# ——————————————————————————————————————————————
class LimitBreakPredictor:
    def __init__(self, cfg: Optional[ConstraintConfig] = None):
        self.cfg = cfg or ConstraintConfig()
        self.base: Optional[LotoPredictor] = None
        self._init_base()
        self.sampler = ConditionalSampler(self.cfg)
        self.engine = EvolutionEngine(self.cfg)

    def _init_base(self):
        try:
            # 入力次元は呼び出し時に決定するためダミーで初期化 → 予測前に置き換え
            self.base = LotoPredictor(input_size=10, hidden_size=128, output_size=7)  # type: ignore
        except Exception:
            self.base = None

    # ——— メインパイプライン ———
    def limit_break_predict(
        self,
        latest_data: pd.DataFrame,
        n_out: int = 50,
        ga_generations: int = 42,
        ga_pop_size: int = 160,
        sampler_n: int = 120,
    ) -> List[PredWithScore]:
        """最終的に n_out 件の（数字, 信頼度）を返す。"""
        set_global_seed(20250819)

        # 未来リークを避け、学習/特徴量用の履歴を作成
        latest_data = latest_data.copy()
        latest_data["抽せん日"] = pd.to_datetime(latest_data["抽せん日"], errors="coerce")
        target_date = latest_data["抽せん日"].max()
        history_df = latest_data[latest_data["抽せん日"] < target_date]
        if history_df.empty:
            history_df = latest_data.iloc[:-1].copy()

        # ベース予測候補（存在すれば）
        seed_candidates: List[NumberSet] = []
        if self.base is not None:
            try:
                # 入力次元を合わせるため一度前処理
                X, _, _ = preprocess_data(latest_data)
                input_size = X.shape[1] if X is not None and hasattr(X, "shape") else 10
                # base の入出力を上書き初期化（安全）
                self.base = LotoPredictor(input_size=input_size, hidden_size=128, output_size=7)  # type: ignore
                preds, confs = self.base.predict(latest_data, num_candidates=120)
                if preds is not None:
                    seed_candidates = [
                        _ensure_valid(p, self.cfg.low, self.cfg.high) for p in preds
                    ]
            except Exception:
                traceback.print_exc()

        # 条件付きサンプラーで強化（Diffusion/GAN 利用 or 確率サンプル）
        cond_samples = self.sampler.sample_with_constraints(
            base_predictor=self.base,
            hist_df=history_df,
            n_samples=sampler_n,
            accept_threshold=0.75,
        )

        seed_all = seed_candidates + cond_samples

        # 進化探索
        evolved = self.engine.search(
            seed_population=seed_all,
            hist_df=history_df,
            generations=ga_generations,
            pop_size=ga_pop_size,
            elite=max(ga_pop_size // 12, 8),
        )

        # 最終スコアリング
        pair_freq, triple_freq = pair_triple_frequencies(history_df)
        scored: List[PredWithScore] = []
        for c in evolved:
            cscore = constraint_score(c, self.cfg)
            co = cooccurrence_score(c, pair_freq, triple_freq)
            final = 0.6 * cscore + 0.4 * co
            # 信頼度 0.75〜1.15 に射影
            conf = 0.75 + 0.40 * final
            scored.append((c, float(conf)))

        # 同一候補の重複を除去
        uniq: Dict[Tuple[int, ...], float] = {}
        for nums, conf in scored:
            key = tuple(nums)
            if key not in uniq:
                uniq[key] = conf
            else:
                uniq[key] = max(uniq[key], conf)

        final = sorted(uniq.items(), key=lambda x: x[1], reverse=True)[:n_out]
        return [ (list(k), v) for k, v in final ]

    def save_predictions(self, predictions: List[PredWithScore], drawing_date: str, filename: str = "loto7_predictions.csv"):
        save_predictions_to_csv(predictions, drawing_date, filename=filename)


# ——————————————————————————————————————————————
# CLI エントリ
# ——————————————————————————————————————————————
if __name__ == "__main__":
    import asyncio

    def _get_latest_date_fallback(df: pd.DataFrame) -> str:
        d = pd.to_datetime(df["抽せん日"], errors="coerce").max()
        return (d + pd.Timedelta(days=7)).strftime("%Y-%m-%d") if pd.notna(d) else pd.Timestamp.today().strftime("%Y-%m-%d")

    try:
        data = pd.read_csv("loto7.csv", encoding="utf-8-sig")
        # リスト文字列を配列に
        def _to_list(x):
            if isinstance(x, list):
                return x
            if isinstance(x, str):
                x = x.strip("[]").replace("'", "").replace('"', "")
                arr = [int(t) for t in x.split() if t.isdigit()]
                if len(arr) == 7:
                    return arr
            return []
        data["本数字"] = data["本数字"].apply(_to_list)
        data["抽せん日"] = pd.to_datetime(data["抽せん日"], errors="coerce")
    except Exception as e:
        print(f"[ERROR] loto7.csv の読み込みに失敗しました: {e}")
        raise SystemExit(1)

    # 公式サイトから抽せん日を取れる環境なら使用（失敗時はフォールバック）
    draw_date = None
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        dates = loop.run_until_complete(get_latest_drawing_dates())
        if dates:
            draw_date = str(dates[0])
    except Exception:
        draw_date = None
    if not draw_date:
        draw_date = _get_latest_date_fallback(data)

    print(f"[INFO] 予測対象抽せん日: {draw_date}")

    lbp = LimitBreakPredictor()
    preds = lbp.limit_break_predict(data.tail(50), n_out=50)

    print("=== 限界突破 予測（上位5件） ===")
    for i, (nums, conf) in enumerate(preds[:5], 1):
        print(f"#{i}: {nums}  信頼度={conf:.3f}")

    lbp.save_predictions(preds, draw_date)
    print("[DONE] 予測を CSV に保存しました → loto7_predictions.csv")


# =============================================================
# 🔧 追加機能: 欠損回の一括予測（上書きではなく追記保存）
# =============================================================
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
    \"\"\"
    過去1回目から直近まで、まだ保存されていない回の予測を順次作成して追記保存する。
    - latest_data の各日付 d について、d 以前の履歴だけで予測し、その日の行を out_csv に追記。
    - 既に out_csv に存在する日付はスキップ。
    - save_predictions_to_csv の「毎回上書き」問題を回避するため、ここで追記保存を完結。
    \"\"\"
    # データの日付を正規化
    df = full_data.copy()
    df["抽せん日"] = pd.to_datetime(df["抽せん日"], errors="coerce")
    df = df.dropna(subset=["抽せん日"]).sort_values("抽せん日")

    # 既存保存のある日付を収集
    existing_dates = set()
    if os.path.exists(out_csv):
        try:
            ex = pd.read_csv(out_csv, encoding="utf-8-sig")
        except Exception:
            ex = pd.read_csv(out_csv)
        if "抽せん日" in ex.columns:
            existing_dates = set(pd.to_datetime(ex["抽せん日"], errors="coerce").dropna().dt.date)

    # すべての対象日
    all_dates = [d.date() for d in df["抽せん日"].unique()]

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
