
# evaluation.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def parse_number_string(number_str):
    if pd.isna(number_str): return []
    s = str(number_str).strip('[]').replace("'", "").replace('"', "").replace(",", " ")
    parts = [p for p in s.split() if p.isdigit()]
    return [int(x) for x in parts if 1 <= int(x) <= 37]

def classify_rank(main_match: int, bonus_match: int) -> str:
    if main_match == 7: return "1等"
    elif main_match == 6 and bonus_match >= 1: return "2等"
    elif main_match == 6: return "3等"
    elif main_match == 5: return "4等"
    elif main_match == 4: return "5等"
    elif main_match == 3 and bonus_match >= 1: return "6等"
    else: return "該当なし"

def evaluate_predictions_with_bonus(predictions_file: str, results_file: str) -> pd.DataFrame:
    pred_df = pd.read_csv(predictions_file, encoding='utf-8-sig')
    res_df = pd.read_csv(results_file, encoding='utf-8-sig')
    rows = []
    for _, row in pred_df.iterrows():
        d = row.get("抽せん日")
        if pd.isna(d): continue
        tgt = res_df[res_df["抽せん日"] == d]
        if tgt.empty: continue
        actual_main = parse_number_string(tgt.iloc[0].get("本数字", ""))
        actual_bonus = parse_number_string(tgt.iloc[0].get("ボーナス数字", ""))
        for i in range(1, 6):
            col = f"予測{i}"
            if col not in row or pd.isna(row[col]): continue
            pred = set(parse_number_string(row[col]))
            m = len(pred & set(actual_main))
            b = len(pred & set(actual_bonus))
            rk = classify_rank(m, b)
            rows.append({
                "抽せん日": d, "予測番号": sorted(list(pred)),
                "当選本数字": actual_main, "当選ボーナス": actual_bonus,
                "本数字一致数": m, "ボーナス一致数": b,
                "信頼度": row.get(f"信頼度{i}", np.nan), "等級": rk
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out.to_csv("eval_model_detailed.csv", index=False, encoding='utf-8-sig')
    return out

def _rand_combo(size=37, k=7):
    return sorted(np.random.choice(np.arange(1, size+1), size=k, replace=False).tolist())

def simulate_random_baseline(results_file: str, n_per_draw: int, trials: int = 1000) -> pd.DataFrame:
    res_df = pd.read_csv(results_file, encoding='utf-8-sig')
    rows = []
    for _, r in res_df.iterrows():
        d = r.get("抽せん日")
        main = set(parse_number_string(r.get("本数字", "")))
        bonus = set(parse_number_string(r.get("ボーナス数字", "")))
        for t in range(trials):
            for j in range(n_per_draw):
                pred = set(_rand_combo())
                m = len(pred & main); b = len(pred & bonus)
                rows.append({"抽せん日": d, "trial": t, "idx": j,
                             "本数字一致数": m, "ボーナス一致数": b,
                             "等級": classify_rank(m, b)})
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv("eval_random_detailed.csv", index=False, encoding='utf-8-sig')
    return df

def summarize_random_baseline(df_random: pd.DataFrame) -> pd.DataFrame:
    best = (df_random.groupby(["抽せん日", "trial"])['本数字一致数']
                    .max().reset_index(name='best_main_matches'))
    return best

def summarize_comparison(model_detail: pd.DataFrame, random_best: pd.DataFrame,
                         output_prefix: str = "comparison") -> dict:
    model_best = (model_detail.groupby("抽せん日")["本数字一致数"]
                  .max().reset_index(name="best_main_matches_model"))
    model_avg_best = model_best['best_main_matches_model'].mean()
    random_avg_best = random_best['best_main_matches'].mean()
    model_ge4 = (model_best['best_main_matches_model'] >= 4).mean()
    random_ge4 = (random_best['best_main_matches'] >= 4).mean()

    plt.figure(figsize=(8,5))
    bins = np.arange(-0.5, 7.6, 1)
    plt.hist(random_best['best_main_matches'], bins=bins, alpha=0.6, label='Random best')
    plt.hist(model_best['best_main_matches_model'], bins=bins, alpha=0.6, label='Model best')
    plt.xlabel('Best main matches per draw'); plt.ylabel('Frequency')
    plt.title('Model vs Random (best-of-k per draw)')
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{output_prefix}_hist.png"); plt.close()

    comp = pd.DataFrame({
        'metric': ['avg_best_matches', 'p_ge4'],
        'model': [model_avg_best, model_ge4],
        'random': [random_avg_best, random_ge4],
        'lift_vs_random': [model_avg_best - random_avg_best, model_ge4 - random_ge4],
    })
    comp.to_csv(f"{output_prefix}_summary.csv", index=False, encoding='utf-8-sig')
    with open(f"{output_prefix}_summary.txt", 'w', encoding='utf-8') as f:
        f.write('=== Model vs Random Baseline Summary ===\n')
        f.write(f"Model avg best matches  : {model_avg_best:.3f}\n")
        f.write(f"Random avg best matches : {random_avg_best:.3f}\n")
        f.write(f"Lift (avg)              : {model_avg_best - random_avg_best:.3f}\n\n")
        f.write(f"Model P(best>=4)        : {model_ge4:.3%}\n")
        f.write(f"Random P(best>=4)       : {random_ge4:.3%}\n")
        f.write(f"Lift (>=4)              : {model_ge4 - random_ge4:.3%}\n")
    return {'model_avg_best': float(model_avg_best),
            'random_avg_best': float(random_avg_best),
            'model_p_ge4': float(model_ge4),
            'random_p_ge4': float(random_ge4)}
