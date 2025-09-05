
# features.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
import numpy as np
import pandas as pd

def convert_to_number_list(x):
    if isinstance(x, list):
        return [int(n) for n in x if isinstance(n, (int, np.integer)) and 1 <= int(n) <= 37]
    if isinstance(x, str):
        cleaned = x.strip("[]").replace(",", " ").replace("'", "").replace('"', "")
        nums = [s for s in cleaned.split() if s.isdigit()]
        return [int(n) for n in nums if 1 <= int(n) <= 37]
    return []

def numbers_to_binary_vec(nums, size=37):
    vec = np.zeros(size, dtype=np.int16)
    for n in nums:
        if 1 <= n <= size:
            vec[n-1] = 1
    return vec

def lists_to_matrix(series_of_lists, size=37):
    return np.vstack([numbers_to_binary_vec(lst, size=size) for lst in series_of_lists])

def _pair_triple_stats(main_lists):
    pair_freq, triple_freq = {}, {}
    for nums in main_lists:
        s = sorted(nums)
        for p in combinations(s, 2):
            pair_freq[p] = pair_freq.get(p, 0) + 1
        for t in combinations(s, 3):
            triple_freq[t] = triple_freq.get(t, 0) + 1
    return pair_freq, triple_freq

def _gap_features(sorted_arr):
    diffs = np.diff(sorted_arr, axis=1)
    min_gap = diffs.min(axis=1)
    max_gap = diffs.max(axis=1)
    seq_count = (diffs == 1).sum(axis=1)
    rng = sorted_arr.max(axis=1) - sorted_arr.min(axis=1)
    return diffs, min_gap, max_gap, seq_count, rng

def _rolling_recent_hit_ratio(bin_mat, main_lists, window=5):
    cumsum = np.cumsum(bin_mat, axis=0)
    N = bin_mat.shape[0]
    recent_counts = np.zeros_like(bin_mat, dtype=np.int16)
    for i in range(N):
        prev = cumsum[i-1] if i > 0 else np.zeros(37, dtype=cumsum.dtype)
        base = cumsum[i-window-1] if i - window > 0 else np.zeros(37, dtype=cumsum.dtype)
        recent_counts[i] = (prev - base)
    ratios = np.zeros(N, dtype=np.float32)
    for i, nums in enumerate(main_lists):
        ratios[i] = recent_counts[i, np.array(nums)-1].sum() / 7.0
    return ratios

def _avg_last_seen_gap(main_lists):
    last_seen = {k: None for k in range(1, 38)}
    gaps = np.zeros(len(main_lists), dtype=np.float32)
    for i, nums in enumerate(main_lists):
        acc = 0; cnt = 0
        for n in nums:
            if last_seen[n] is None:
                acc += 0
            else:
                acc += (i - last_seen[n])
            cnt += 1
            last_seen[n] = i
        gaps[i] = acc / max(cnt, 1)
    return gaps

def create_advanced_features_fast(df):
    data = df.copy()
    data['本数字'] = data['本数字'].apply(convert_to_number_list)
    data['ボーナス数字'] = data['ボーナス数字'].apply(convert_to_number_list)
    data['抽せん日'] = pd.to_datetime(data['抽せん日'], errors='coerce')
    valid = (data['本数字'].apply(len) == 7) & (data['ボーナス数字'].apply(len) == 2)
    data = data.loc[valid].reset_index(drop=True)
    if data.empty:
        return data

    nums_arr = np.vstack([np.sort(lst) for lst in data['本数字']])
    bin_mat = lists_to_matrix(data['本数字'])

    freq_global = bin_mat.sum(axis=0)
    freq_score = (bin_mat @ freq_global.reshape(-1, 1)).ravel() / 7.0

    pair_freq, triple_freq = _pair_triple_stats(data['本数字'])
    pair_counts = np.zeros(len(data), dtype=np.int32)
    triple_counts = np.zeros(len(data), dtype=np.int32)
    for i, nums in enumerate(data['本数字']):
        s = sorted(nums)
        pair_counts[i] = sum(pair_freq.get(p, 0) for p in combinations(s, 2))
        triple_counts[i] = sum(triple_freq.get(t, 0) for t in combinations(s, 3))

    recent5_ratio = _rolling_recent_hit_ratio(bin_mat, data['本数字'], window=5)
    diffs, min_gap, max_gap, seq_count, rng = _gap_features(nums_arr)

    odd_ratio = (nums_arr % 2 != 0).sum(axis=1) / 7.0
    even_ratio = 1.0 - odd_ratio
    total = nums_arr.sum(axis=1)
    std = nums_arr.std(axis=1)
    mean = nums_arr.mean(axis=1)
    median = np.median(nums_arr, axis=1)

    dow = data['抽せん日'].dt.dayofweek.to_numpy()
    month = data['抽せん日'].dt.month.to_numpy()
    year = data['抽せん日'].dt.year.to_numpy()

    avg_gap = _avg_last_seen_gap(data['本数字'])

    feat = pd.DataFrame({
        '奇数比': odd_ratio,
        '偶数比': even_ratio,
        '本数字合計': total,
        'レンジ': rng,
        '標準偏差': std,
        '数字平均': mean,
        '中央値': median,
        '連番数': seq_count,
        '最小間隔': min_gap,
        '最大間隔': max_gap,
        '曜日': dow,
        '月': month,
        '年': year,
        '出現間隔平均': avg_gap,
        '出現頻度スコア': freq_score,
        'ペア出現頻度': pair_counts.astype(np.float32),
        'トリプル出現頻度': triple_counts.astype(np.float32),
        '直近5回出現率': recent5_ratio,
    })

    return pd.concat([data.reset_index(drop=True), feat], axis=1)

def preprocess_data_fast(data):
    processed = create_advanced_features_fast(data)
    if processed.empty:
        return None, None, None
    import numpy as np
    numeric_cols = processed.select_dtypes(include=[np.number]).columns
    X = processed[numeric_cols].fillna(0.0).to_numpy(dtype=np.float32)
    y = np.vstack([np.array(lst, dtype=np.int16) for lst in processed['本数字']])
    return X, y, None
