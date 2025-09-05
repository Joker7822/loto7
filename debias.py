
# debias.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np

def softmax(x):
    x = np.asarray(x, dtype=float)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / (ex.sum() + 1e-12)

def softmax_no_replace(scores, k=7, tau=0.9):
    """
    スコアから温度付き softmax 確率で、重複なしサンプリング。
    """
    scores = np.asarray(scores, dtype=float)
    p = softmax(scores / max(1e-6, tau))
    # 0だらけを避ける
    if np.all(p <= 0):
        p = np.ones_like(p) / p.size
    selected = []
    avail = np.arange(p.size)
    probs = p.copy()
    for _ in range(k):
        probs = probs / probs.sum()
        idx = np.random.choice(np.arange(probs.size), p=probs)
        selected.append(avail[idx])
        # remove chosen
        avail = np.delete(avail, idx)
        probs = np.delete(probs, idx)
    return np.sort(np.array(selected))

def adaptive_temperature_from_marg(marg):
    """
    マージナル分布の尖り具合で温度を自動調整。
    分布が尖っている（sum p^2 が大きい）ほど温度を上げて平坦化。
    """
    marg = np.asarray(marg, dtype=float)
    if marg.sum() <= 0:
        return 0.8
    p = marg / marg.sum()
    s2 = float((p ** 2).sum())  # 1/37..1
    # map s2->[1/37,1] to tau->[0.6, 1.6]
    tau = 0.6 + (s2 - 1/37) * (1.6 - 0.6) / (1 - 1/37)
    return float(np.clip(tau, 0.6, 1.6))

def stable_diverse_selection_balanced(numbers_only, confidence_scores, latest_data,
                                      k=30, lambda_div=0.6, beta_cov=0.35, cap_factor=1.5):
    """
    元の _stable_diverse_selection を拡張:
    - 既選択の数字使用回数（coverage）にペナルティ
    - 各数字の最大使用回数 cap を導入（約 cap_factor * 均等使用）
    """
    import numpy as _np
    import pandas as _pd
    import hashlib

    # シード（抽せん日）
    if isinstance(latest_data, _pd.DataFrame) and "抽せん日" in latest_data.columns:
        td = str(_pd.to_datetime(latest_data["抽せん日"].max()).date())
    else:
        td = "unknown"
    seed = int(hashlib.md5(td.encode()).hexdigest()[:8], 16)
    _np.random.seed(seed)

    conf = _np.array(confidence_scores, dtype=float)
    conf = (conf - conf.min()) / (conf.max() - conf.min() + 1e-9)

    # マージナルの推定（信頼度で重み付け）
    marg = _np.zeros(37)
    for cand, w in zip(numbers_only, conf):
        for n in cand:
            marg[n-1] += w
    marg = marg / (marg.sum() + 1e-12)

    # 温度を自動調整
    tau = adaptive_temperature_from_marg(marg)
    weights = _np.exp(conf / max(1e-6, tau))
    weights = weights / (weights.sum() + 1e-12)

    base_scores = [sum(_np.log(marg[n-1] + 1e-9) for n in cand) for cand in numbers_only]

    selected, used = [], set()
    usage = _np.zeros(37, dtype=float)
    total_slots = k * 7
    equal_quota = total_slots / 37.0
    cap = int(_np.ceil(equal_quota * cap_factor))

    for _ in range(min(k, len(numbers_only))):
        best_i, best_val = None, -1e18
        for i, cand in enumerate(numbers_only):
            if i in used:
                continue

            # Jaccard ペナルティ
            jpen = 0.0
            for s in selected:
                inter = len(set(cand) & set(s))
                union = len(set(cand) | set(s))
                jpen += inter / max(1, union)

            # coverage ペナルティ（過剰使用に重み）
            cov_pen = 0.0
            for n in cand:
                over = max(0.0, (usage[n-1] + 1) - equal_quota)
                cov_pen += over / max(1.0, equal_quota)
            cov_pen *= beta_cov

            val = base_scores[i] - lambda_div * jpen - cov_pen
            if val > best_val:
                best_val, best_i = val, i

        used.add(best_i)
        chosen = numbers_only[best_i]

        # cap を超えないよう微調整（必要なら過剰数字を代替）
        for idx, n in enumerate(list(chosen)):
            if usage[n-1] >= cap:
                # 代替候補: まだ少ない数字を優先
                under = _np.argsort(usage)  # 昇順
                for cand_n in under:
                    nn = int(cand_n + 1)
                    if nn not in chosen:
                        chosen[idx] = nn
                        break

        selected.append(sorted(chosen))
        for n in selected[-1]:
            usage[n-1] += 1.0

    return selected
