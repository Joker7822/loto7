
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# predict_latest_topk_improved_v2.py
# Adds robust CSV schema normalization so it works even if column names differ

import os, re, json, argparse, random
import numpy as np
import pandas as pd
from datetime import datetime

from lottery_prediction import (
    LotoPredictor,
    preprocess_data,
    create_advanced_features,
    save_predictions_to_csv,
)

try:
    from lottery_prediction import _stable_diverse_selection as stable_diverse_selection
    _HAVE_DIVERSE = True
except Exception as e:
    print("[WARN] Could not import diversity selection:", e)
    _HAVE_DIVERSE = False

try:
    from limit_break_predictor import ConstraintConfig, constraint_score
    _HAVE_CONSTRAINT = True
except Exception:
    _HAVE_CONSTRAINT = False

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
    return len(set(candidate.tolist()) & set(pred.tolist())) / 7.0

def _normalize_weights(weights: dict, keys: list) -> dict:
    if not weights:
        return {k: 1.0/len(keys) for k in keys}
    w = {k: float(weights.get(k, 0.0)) for k in keys}
    s = sum(v for v in w.values() if np.isfinite(v))
    if s <= 0:
        return {k: 1.0/len(keys) for k in keys}
    return {k: v/s for k, v in w.items()}

def _load_stack_weights(path: str) -> dict:
    if not os.path.exists(path):
        print(f"[WARN] stacking.json not found: {path} -> using equal weights")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        params = json.load(f)
    if isinstance(params, dict):
        return params
    if isinstance(params, list):
        kv = {}
        for it in params:
            if isinstance(it, dict):
                m = it.get("model"); w = it.get("weight")
                if m is not None and w is not None:
                    kv[m] = float(w)
        if kv:
            return kv
    print("[WARN] Unknown stacking.json format -> using equal weights")
    return {}

def _zscore(x: np.ndarray) -> np.ndarray:
    mu = np.nanmean(x); sd = np.nanstd(x)
    if not np.isfinite(sd) or sd <= 1e-9:
        return np.zeros_like(x)
    return (x - mu) / sd

# ---------- CSV schema normalizer ----------
def _find_first(df, patterns):
    for pat in patterns:
        for c in df.columns:
            if re.search(pat, c, flags=re.I):
                return c
    return None

def _normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols = [str(c) for c in df.columns]

    date_col = _find_first(df, [r"抽.?せん.?日", r"抽選日", r"date", r"draw", r"開催日"])
    if date_col is None:
        df["抽せん日"] = pd.date_range("2000-01-01", periods=len(df), freq="7D")
    else:
        df["抽せん日"] = pd.to_datetime(df[date_col], errors="coerce")

    main_col = _find_first(df, [r"^本数字$", r"当選本数字", r"winning.*numbers", r"予測番号", r"numbers?$", r"本.?数?字?$"])
    bonus_col = _find_first(df, [r"^ボーナス数字$", r"当選ボーナス", r"bonus"])

    def _parse_cell_to_list(x):
        if isinstance(x, list):
            return [int(n) for n in x if str(n).isdigit() or isinstance(n, (int, np.integer))]
        if pd.isna(x):
            return []
        s = str(x).strip().strip("[]").replace("'", "").replace('"', "")
        return [int(n) for n in re.split(r"[\s,、]+", s) if n.isdigit()]

    if main_col is not None:
        df["本数字"] = df[main_col].apply(_parse_cell_to_list)
    else:
        cands = []
        for k in range(1, 15):
            for c in df.columns:
                if re.fullmatch(rf"(n|num|本|main)[ _-]?{k}", str(c), flags=re.I):
                    cands.append(c)
        if len(cands) >= 7:
            def _row_nums(row):
                arr = []
                for c in sorted(cands, key=lambda x: int(re.findall(r"\d+", str(x))[0]))[:7]:
                    v = row[c]
                    if pd.isna(v): 
                        continue
                    try:
                        arr.append(int(v))
                    except Exception:
                        pass
                return arr
            df["本数字"] = df.apply(_row_nums, axis=1)
        else:
            raise KeyError(f"CSVに本数字の列が見つかりません。候補列: {cols}")

    if bonus_col is not None:
        bl = df[bonus_col].apply(_parse_cell_to_list)
        df["ボーナス数字"] = bl.apply(lambda xs: xs[:2])
    else:
        b1 = _find_first(df, [r"(bonus|b)[ _-]?1$"])
        b2 = _find_first(df, [r"(bonus|b)[ _-]?2$"])
        if b1 and b2:
            def _row_bonus(row):
                out = []
                for c in [b1, b2]:
                    v = row[c]
                    if pd.isna(v): 
                        continue
                    try:
                        out.append(int(v))
                    except Exception:
                        pass
                return out[:2]
            df["ボーナス数字"] = df.apply(_row_bonus, axis=1)
        else:
            df["ボーナス数字"] = [[] for _ in range(len(df))]

    def _sanitize_seven(xs):
        xs = [int(n) for n in xs if 1 <= int(n) <= 37]
        xs = sorted(list(dict.fromkeys(xs)))
        return xs

    df["本数字"] = df["本数字"].apply(_sanitize_seven)
    df["ボーナス数字"] = df["ボーナス数字"].apply(lambda xs: [int(n) for n in xs][:2])

    return df[["抽せん日", "本数字", "ボーナス数字"]]

# ---- Build/Rank logic: import from v1 if available, else define light versions ----
def _import_from_v1(name):
    try:
        import importlib.util, sys, types, pathlib
        p = pathlib.Path(__file__).parent / "predict_latest_topk_improved.py"
        if p.exists():
            import importlib.util
            spec = importlib.util.spec_from_file_location("improv_v1", str(p))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return getattr(mod, name)
    except Exception as e:
        print("[WARN] failed to import from v1:", e)
    return None

_build_v1 = _import_from_v1("build_latest_predictions_per_model")
_rank_v1 = _import_from_v1("rank_candidates_with_stacking")

if _build_v1 is None or _rank_v1 is None:
    # fallback minimal (won't be used if v1 present)
    def _dummy(*a, **k):
        raise RuntimeError("Missing v1 implementation. Keep v1 file next to v2.")
    _build_v1 = _dummy
    _rank_v1 = _dummy

def build_latest_predictions_per_model(*args, **kwargs):
    return _build_v1(*args, **kwargs)

def rank_candidates_with_stacking(*args, **kwargs):
    return _rank_v1(*args, **kwargs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="loto7_prediction_evaluation_with_bonus.csv", help="path to history CSV")
    ap.add_argument("--models_dir", default="models/full", help="trained models directory")
    ap.add_argument("--stacking", default="optuna_results/stacking.json", help="stacking weights JSON")
    ap.add_argument("--samples", type=int, default=200, help="num samples per model")
    ap.add_argument("--topk", type=int, default=30, help="top-K before diversity selection")
    ap.add_argument("--noise", type=float, default=0.01, help="feature noise for deterministic models")
    ap.add_argument("--out", default="predictions_topK_improved.csv", help="analysis CSV output")
    ap.add_argument("--cweight", type=float, default=0.10, help="constraint score weight (default 0.10)")
    ap.add_argument("--norm", choices=["zscore", "minmax", "none"], default="zscore", help="normalization mode")
    ap.add_argument("--post_diverse_k", type=int, default=30, help="final K after diversity selection")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"{args.csv} not found.")

    raw = pd.read_csv(args.csv, encoding="utf-8-sig")
    try:
        df = _normalize_schema(raw)
    except Exception as e:
        print("[ERROR] CSV schema normalization failed:", e)
        print("[HINT] Available columns:", list(raw.columns))
        raise

    X_all, y_all, scaler = preprocess_data(df)
    if X_all is None:
        raise RuntimeError("preprocess failed.")
    X_latest = X_all[-1]

    predictor = LotoPredictor(input_size=X_all.shape[1], hidden_size=128, output_size=7)
    predictor.load_saved_models(args.models_dir)

    per_model_preds = build_latest_predictions_per_model(
        predictor=predictor,
        full_df=df,
        X_latest=X_latest,
        num_samples=args.samples,
        noise_level=args.noise
    )
    if not per_model_preds:
        raise RuntimeError("no predictions from any model.")

    pool = []
    for lst in per_model_preds.values():
        pool.extend([tuple(x.tolist()) for x in lst])
    uniq = sorted(set(pool))
    candidates = [np.array(x, dtype=int) for x in uniq]
    print(f"[INFO] unique candidates: {len(candidates)}")

    stack_weights = _load_stack_weights(args.stacking)
    ranked = rank_candidates_with_stacking(
        candidates,
        per_model_preds,
        stack_weights,
        use_constraints=True,
        constraint_weight=args.cweight,
        normalize_mode=args.norm,
    )

    topk = ranked[: args.topk]
    numbers_only = [cand.tolist() for cand, sc, _ in topk]
    confidence_scores = [float(sc) for _, sc, _ in topk]
    if _HAVE_DIVERSE and len(numbers_only) > 0:
        numbers_only = stable_diverse_selection(
            numbers_only, confidence_scores, df, k=min(args.post_diverse_k, len(numbers_only)),
            lambda_div=0.6, temperature=0.35
        )
        recon, seen = [], set()
        for nums in numbers_only:
            tpl = tuple(nums)
            if tpl in seen: 
                continue
            seen.add(tpl)
            for cand, sc, parts in topk:
                if tuple(cand.tolist()) == tpl:
                    recon.append((cand, sc, parts))
                    break
        topk = recon

    for i, (cand, sc, parts) in enumerate(topk, 1):
        print(f"[{i:02d}] {cand.tolist()}  score={sc:.4f}  parts={{{', '.join(f'{k}:{v:.3f}' for k,v in parts.items())}}}")

    rows = []
    for rank, (cand, sc, parts) in enumerate(topk, 1):
        row = {"rank": rank, "score": float(sc), "numbers": " ".join(map(str, cand.tolist()))}
        row.update({f"s_{k}": float(v) for k, v in parts.items()})
        rows.append(row)
    pd.DataFrame(rows).to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"[INFO] saved analysis CSV: {args.out}")

    drawing_date = datetime.now().strftime("%Y-%m-%d")
    save_list = [(cand.tolist(), float(sc)) for cand, sc, _ in topk[:5]]
    try:
        save_predictions_to_csv(save_list, drawing_date=drawing_date, filename="loto7_predictions_improved.csv")
        print("[INFO] saved standard CSV: loto7_predictions_improved.csv")
    except Exception as e:
        print("[WARN] failed to save standard CSV:", e)

if __name__ == "__main__":
    random.seed(42); np.random.seed(42)
    main()
