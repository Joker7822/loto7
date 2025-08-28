#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LimitBreakPredictor（完全版サンプラー搭載）:
- GA（進化探索）
- 完全版 条件付きサンプリング（履歴ベースのエネルギー最小化：焼きなまし＋メトロポリス）
- 多目的スコア（分布整合性・多様性・ルール適合度）
"""

from __future__ import annotations

import math
import random
import traceback
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Iterable

import numpy as np
import pandas as pd

# 既存実装（無ければフォールバック）
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
    odd_min: int = 2
    odd_max: int = 5
    sum_min: int = 100
    sum_max: int = 150
    min_gap: int = 2
    min_range: int = 15
    low: int = 1
    high: int = 37

# ——————————————————————————————————————————————
# ヘルパー
# ——————————————————————————————————————————————
NumberSet = List[int]
PredWithScore = Tuple[NumberSet, float]

def _ensure_valid(numbers: Iterable[int], low: int = 1, high: int = 37) -> NumberSet:
    s = sorted(set(int(n) for n in numbers if low <= int(n) <= high))
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
    if lo <= v <= hi:
        return 1.0
    d = min(abs(v - lo), abs(v - hi))
    width = max(1e-6, hi - lo)
    return max(0.0, 1.0 - d / width)

def constraint_score(nums: NumberSet, cfg: ConstraintConfig) -> float:
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
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                p = (s[i], s[j])
                pair_freq[p] = pair_freq.get(p, 0) + 1
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                for k in range(j + 1, len(s)):
                    t = (s[i], s[j], s[k])
                    triple_freq[t] = triple_freq.get(t, 0) + 1
    return pair_freq, triple_freq

def cooccurrence_score(nums: NumberSet, pair_freq, triple_freq) -> float:
    s = sorted(nums)
    pf_sum = 0
    tf_sum = 0
    for i in range(7):
        for j in range(i + 1, 7):
            pf_sum += pair_freq.get((s[i], s[j]), 0)
    for i in range(7):
        for j in range(i + 1, 7):
            for k in range(j + 1, 7):
                tf_sum += triple_freq.get((s[i], s[j], s[k]), 0)
    return float(1.0 - math.exp(-0.002 * (pf_sum + 0.5 * tf_sum)))

def diversity_penalty(nums: NumberSet, others: List[NumberSet]) -> float:
    if not others:
        return 0.0
    inters = []
    s = set(nums)
    for o in others:
        inters.append(len(s & set(o)))
    avg_inter = sum(inters) / len(inters)
    return min(1.0, avg_inter / 7.0)

# ——————————————————————————————————————————————
# Evolutionary Search（GA）
# ——————————————————————————————————————————————
class EvolutionEngine:
    def __init__(self, cfg: ConstraintConfig, num_low: int = 1, num_high: int = 37):
        self.cfg = cfg
        self.low = num_low
        self.high = num_high

    def _fitness(self, cand: NumberSet, hist_df: pd.DataFrame, pair_freq, triple_freq, others: Optional[List[NumberSet]] = None) -> float:
        cscore = constraint_score(cand, self.cfg)
        co = cooccurrence_score(cand, pair_freq, triple_freq)
        div_pen = diversity_penalty(cand, others or [])
        score = 0.55 * cscore + 0.45 * co - 0.25 * div_pen
        return float(score)

    def _crossover(self, a: NumberSet, b: NumberSet) -> NumberSet:
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
            pool = list(range(self.low, self.high + 1))
            weights = np.array([num_freq.get(i, 1e-6) for i in pool], dtype=float)
            if weights.sum() <= 0:
                cand = random.randint(self.low, self.high)
            else:
                weights = weights / weights.sum()
                cand = int(np.random.choice(pool, p=weights))
            y[idx] = cand
        return _ensure_valid(y, self.low, self.high)

    def search(self, seed_population: List[NumberSet], hist_df: pd.DataFrame, generations: int = 40, pop_size: int = 120, elite: int = 12) -> List[NumberSet]:
        set_global_seed(777)
        elite = max(1, min(elite, pop_size - 1))
        num_freq = number_frequencies(hist_df)
        pair_freq, triple_freq = pair_triple_frequencies(hist_df)

        pop: List[NumberSet] = []
        pop.extend(_ensure_valid(s) for s in seed_population)
        while len(pop) < pop_size:
            pool = list(range(self.low, self.high + 1))
            weights = np.array([num_freq.get(i, 1e-6) for i in pool], dtype=float)
            weights = weights / (weights.sum() or 1)
            cand = list(np.random.choice(pool, size=7, replace=False, p=weights)) if weights is not None else list(np.random.choice(pool, size=7, replace=False))
            pop.append(_ensure_valid(cand, self.low, self.high))
        if len(pop) > pop_size:
            pop = pop[:pop_size]

        for _gen in range(generations):
            N = len(pop)
            scores = [self._fitness(ind, hist_df, pair_freq, triple_freq, others=pop) for ind in pop]

            idxs = np.argsort(scores)[::-1]
            elites = [pop[i] for i in idxs[:elite]]

            parents: List[NumberSet] = []
            target_children = pop_size - elite
            if N < 4:
                while len(parents) < target_children:
                    parents.append(random.choice(pop))
                    parents.append(random.choice(pop))
            else:
                while len(parents) < target_children:
                    t = random.sample(range(N), k=4)
                    best = max(t, key=lambda i: scores[i])
                    parents.append(pop[best])
                    t2 = random.sample(range(N), k=4)
                    best2 = max(t2, key=lambda i: scores[i])
                    parents.append(pop[best2])

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
            if len(pop) > pop_size:
                pop = pop[:pop_size]
            elif len(pop) < pop_size:
                while len(pop) < pop_size:
                    pool = list(range(self.low, self.high + 1))
                    weights = np.array([num_freq.get(i, 1e-6) for i in pool], dtype=float)
                    weights = weights / (weights.sum() or 1)
                    cand = list(np.random.choice(pool, size=7, replace=False, p=weights)) if weights is not None else list(np.random.choice(pool, size=7, replace=False))
                    pop.append(_ensure_valid(cand, self.low, self.high))

        final_scores = [self._fitness(ind, hist_df, pair_freq, triple_freq, others=[]) for ind in pop]
        order = np.argsort(final_scores)[::-1]
        return [pop[i] for i in order]

# ——————————————————————————————————————————————
# 完全版 条件付きサンプラー（焼きなまし＋メトロポリス）
# ——————————————————————————————————————————————
class ConditionalSampler:
    """
    履歴から単体/ペア/トリプルの共起構造を抽出し、制約と統合したエネルギーを最小化する
    サンプラー。学習や外部モデル不要。
    """
    def __init__(self, cfg: ConstraintConfig,
                 alpha: float = 0.60,   # 制約適合の重み
                 beta: float = 0.30,    # ペア/トリプル共起の重み
                 gamma: float = 0.10,   # 単体頻度の重み
                 ):
        self.cfg = cfg
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.freq_uni: Dict[int, float] = {}
        self.freq_pair: Dict[Tuple[int, int], int] = {}
        self.freq_tri: Dict[Tuple[int, int, int], int] = {}

    # ——— エネルギー（小さいほど良い） ———
    def _energy(self, nums: NumberSet) -> float:
        # 1) 制約→スコア[0..1]を「良いほど低エネ」に変換
        cons = constraint_score(nums, self.cfg)     # 0..1
        e_cons = 1.0 - cons                        # 0..1

        # 2) 共起：ペア/トリプル頻度を log 圧縮→平均正規化→良いほど低エネ
        s = sorted(nums)
        pf = 0.0; tf = 0.0; cntp = 0; cntt = 0
        for i in range(7):
            for j in range(i + 1, 7):
                pf += math.log1p(self.freq_pair.get((s[i], s[j]), 0))
                cntp += 1
        for i in range(7):
            for j in range(i + 1, 7):
                for k in range(j + 1, 7):
                    tf += math.log1p(self.freq_tri.get((s[i], s[j], s[k]), 0))
                    cntt += 1
        if cntp > 0:
            pf /= cntp
        if cntt > 0:
            tf /= cntt
        co_norm = 1.0 - math.exp(-(pf + 0.6 * tf))   # 0..1 に押し込み
        e_co = 1.0 - co_norm

        # 3) 単体頻度：出現確率の合計（平均）を使う
        fu = np.mean([self.freq_uni.get(int(x), 1e-9) for x in s])
        # スケール：0..1 に仮正規化（十分に小さい確率があるので安全に）
        fu_norm = min(1.0, max(0.0, fu * 50.0))
        e_uni = 1.0 - fu_norm

        # 合成エネルギー
        E = self.alpha * e_cons + self.beta * e_co + self.gamma * e_uni
        return float(E)

    # ——— 近傍提案（入替/置換/微移動） ———
    def _propose(self, nums: NumberSet) -> NumberSet:
        s = set(nums)
        mode = random.random()
        if mode < 0.34:
            # 置換：1つ抜いて未使用から1つ入れる
            victim = random.choice(nums)
            pool = [x for x in range(self.cfg.low, self.cfg.high + 1) if x not in s or x == victim]
            cand = random.choice(pool)
            t = nums[:]
            t[t.index(victim)] = cand
            return _ensure_valid(t, self.cfg.low, self.cfg.high)
        elif mode < 0.67:
            # 入替：位置2つをスワップ（順序に意味はないが、局所探索で揺らす）
            i, j = random.sample(range(7), 2)
            t = nums[:]
            t[i], t[j] = t[j], t[i]
            return _ensure_valid(t, self.cfg.low, self.cfg.high)
        else:
            # 微移動：1つ選んで ±1〜±3 の範囲で動かす
            i = random.randrange(7)
            step = random.choice([-3, -2, -1, 1, 2, 3])
            val = nums[i] + step
            val = min(self.cfg.high, max(self.cfg.low, val))
            t = nums[:]
            t[i] = val
            return _ensure_valid(t, self.cfg.low, self.cfg.high)

    # ——— 初期解の生成：頻度ガイド＋制約フィルタ ———
    def _init_candidate(self) -> NumberSet:
        pool = list(range(self.cfg.low, self.cfg.high + 1))
        w = np.array([self.freq_uni.get(i, 1e-9) for i in pool], dtype=float)
        if not np.isfinite(w.sum()) or w.sum() <= 0:
            cand = random.sample(pool, 7)
        else:
            w = w / w.sum()
            cand = list(np.random.choice(pool, size=7, replace=False, p=w))
        cand = _ensure_valid(cand, self.cfg.low, self.cfg.high)
        return cand

    def sample_with_constraints(
        self,
        base_predictor: Optional[LotoPredictor],
        hist_df: pd.DataFrame,
        n_samples: int = 100,
        accept_threshold: float = 0.75,
        max_steps: int = 1200,
        T_start: float = 1.2,
        T_end: float = 0.02,
    ) -> List[NumberSet]:
        # 周波数テーブル
        self.freq_uni = number_frequencies(hist_df)
        self.freq_pair, self.freq_tri = pair_triple_frequencies(hist_df)

        out: List[NumberSet] = []
        seen: set[Tuple[int, ...]] = set()

        # アニーリング係数
        def temp(t):
            # 指数スケジュール
            r = t / max(1, max_steps - 1)
            return T_start * (T_end / T_start) ** r

        trials = 0
        max_trials = n_samples * 30

        while len(out) < n_samples and trials < max_trials:
            trials += 1
            cur = self._init_candidate()
            E = self._energy(cur)

            for step in range(max_steps):
                T = temp(step)
                prop = self._propose(cur)
                Ep = self._energy(prop)
                dE = Ep - E
                if dE <= 0 or random.random() < math.exp(-dE / max(1e-8, T)):
                    cur, E = prop, Ep

            # 受理：制約スコアで下限確認（使い勝手維持）
            if constraint_score(cur, self.cfg) >= accept_threshold:
                key = tuple(sorted(cur))
                if key not in seen:
                    seen.add(key)
                    out.append(sorted(cur))

        # 足りなければランダムで充足（まれ）
        while len(out) < n_samples:
            c = self._init_candidate()
            out.append(sorted(c))

        return out

# ——————————————————————————————————————————————
# 限界突破 Predictor（メイン）
# ——————————————————————————————————————————————
class LimitBreakPredictor:
    def __init__(self, cfg: Optional[ConstraintConfig] = None):
        self.cfg = cfg or ConstraintConfig()
        self.base: Optional[LotoPredictor] = None
        self._init_base()
        self.sampler = ConditionalSampler(self.cfg)   # ← 完全版
        self.engine = EvolutionEngine(self.cfg)

    def _init_base(self):
        try:
            self.base = LotoPredictor(input_size=10, hidden_size=128, output_size=7)  # type: ignore
        except Exception:
            self.base = None

    def limit_break_predict(
        self,
        latest_data: pd.DataFrame,
        n_out: int = 50,
        ga_generations: int = 42,
        ga_pop_size: int = 160,
        sampler_n: int = 120,
    ) -> List[PredWithScore]:
        set_global_seed(20250819)

        latest_data = latest_data.copy()
        latest_data["抽せん日"] = pd.to_datetime(latest_data["抽せん日"], errors="coerce")
        target_date = latest_data["抽せん日"].max()
        history_df = latest_data[latest_data["抽せん日"] < target_date]
        if history_df.empty:
            history_df = latest_data.iloc[:-1].copy()

        # ベース予測候補（利用可能なら）—— X を使い安全ガード
        seed_candidates: List[NumberSet] = []
        try:
            X, _, _ = preprocess_data(history_df)
            input_size = int(X.shape[1]) if (X is not None and hasattr(X, "shape")) else 0
            if self.base is not None and input_size > 0:
                try:
                    self.base = LotoPredictor(input_size=input_size, hidden_size=128, output_size=7)  # type: ignore
                except Exception:
                    self.base = None
            if self.base is not None and hasattr(self.base, "predict"):
                if hasattr(self.base, "fit"):
                    try:
                        self.base.fit(X, None)  # type: ignore
                    except Exception:
                        pass
                try:
                    preds_confs = self.base.predict(X, num_candidates=120)  # type: ignore
                    preds = preds_confs[0] if isinstance(preds_confs, tuple) and len(preds_confs) >= 1 else preds_confs
                    if preds is not None:
                        seed_candidates = [_ensure_valid(p, self.cfg.low, self.cfg.high) for p in preds if p is not None]
                except Exception:
                    seed_candidates = []
        except Exception:
            seed_candidates = []

        # 完全版サンプラー
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

        # スコアリング
        pair_freq, triple_freq = pair_triple_frequencies(history_df)
        scored: List[PredWithScore] = []
        for c in evolved:
            cscore = constraint_score(c, self.cfg)
            co = cooccurrence_score(c, pair_freq, triple_freq)
            final = 0.6 * cscore + 0.4 * co
            conf = 0.75 + 0.40 * final
            scored.append((c, float(conf)))

        # 重複除去
        uniq: Dict[Tuple[int, ...], float] = {}
        for nums, conf in scored:
            key = tuple(nums)
            if key not in uniq:
                uniq[key] = conf
            else:
                uniq[key] = max(uniq[key], conf)

        final = sorted(uniq.items(), key=lambda x: x[1], reverse=True)[:n_out]
        return [(list(k), v) for k, v in final]

    def save_predictions(self, predictions: List[PredWithScore], drawing_date: str, filename: str = "loto7_predictions.csv"):
        save_predictions_to_csv(predictions, drawing_date, filename=filename)

# ——————————————————————————————————————————————
# CLI
# ——————————————————————————————————————————————
if __name__ == "__main__":
    import asyncio

    def _get_latest_date_fallback(df: pd.DataFrame) -> str:
        d = pd.to_datetime(df["抽せん日"], errors="coerce").max()
        return (d + pd.Timedelta(days=7)).strftime("%Y-%m-%d") if pd.notna(d) else pd.Timestamp.today().strftime("%Y-%m-%d")

    try:
        data = pd.read_csv("loto7.csv", encoding="utf-8-sig")
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

    print("\n=== 限界突破 予測（上位5件） ===")
    for i, (nums, conf) in enumerate(preds[:5], 1):
        print(f"#{i}: {nums}  信頼度={conf:.3f}")

    lbp.save_predictions(preds, draw_date)
    print("[DONE] 予測を CSV に保存しました → loto7_predictions.csv")


def bulk_limit_break_predict_all_past_draws():
    set_global_seed(42)
    df = pd.read_csv("loto7.csv")
    df["抽せん日"] = pd.to_datetime(df["抽せん日"], errors='coerce')
    df = df.sort_values("抽せん日").reset_index(drop=True)

    print("[INFO] 抽せんデータ読み込み完了:", len(df), "件")

    pred_file = "limit_break_predictions.csv"
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

    predictor = LimitBreakPredictor()

    for i in range(10, len(df)):
        set_global_seed(1000 + i)
        test_date = df.iloc[i]["抽せん日"]
        test_date_str = test_date.strftime("%Y-%m-%d")

        if test_date_str in skip_dates:
            print(f"[INFO] 既に予測済み: {test_date_str} → スキップ")
            continue

        print(f"\n=== {test_date_str} のLimitBreak予測を開始 ===")
        latest_data = df.iloc[i-10:i+1].copy()

        try:
            predictions = predictor.limit_break_predict(latest_data)
        except Exception as e:
            print(f"[ERROR] LimitBreak予測失敗: {test_date_str}: {e}")
            continue

        if not predictions:
            print(f"[WARNING] {test_date_str} の予測が空です")
            continue

        save_predictions_to_csv(predictions, test_date_str, filename=pred_file)
        git_commit_and_push(pred_file, "LimitBreak 過去予測更新 [skip ci]")

    print("\n=== LimitBreak 一括予測完了 ===")
