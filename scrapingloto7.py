
import re
import sys
import argparse
from datetime import datetime
from typing import List, Dict
import pandas as pd

import requests
from bs4 import BeautifulSoup

LASTRESULTS = "https://takarakuji.rakuten.co.jp/backnumber/loto7/lastresults/"
CSV_PATH_DEFAULT = "loto7.csv"

def _to_halfwidth(s: str) -> str:
    # Convert full-width digits to half-width; leave others unchanged
    fw = "０１２３４５６７８９／－．"
    hw = "0123456789/-."
    return s.translate(str.maketrans(fw, hw))

def parse_lastresults(html: str) -> List[Dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    text = _to_halfwidth(soup.get_text(" ", strip=True))

    # e.g. "第00638回 2025/08/08" 〜 の並びを拾う
    pat = re.compile(r"第\s*0*(\d{1,4})回\s*([0-9]{4}/[01][0-9]/[0-3][0-9])")
    rows = []

    for m in pat.finditer(text):
        draw_no = f"第{int(m.group(1))}回"
        date_iso = datetime.strptime(m.group(2), "%Y/%m/%d").strftime("%Y-%m-%d")

        # 以前のコードは日付成分(例: '08')を全て除外してしまい、
        # 抽選数字 08 や 09 まで落ちるバグがありました。
        # 修正版では「日付のすぐ後ろから」数字を9個(7+2)だけ拾います。
        window = text[m.end(): m.end() + 280]  # date の直後から限定して探索
        cand = re.findall(r"\b\d{1,2}\b", window)

        # 数字が多いページでも先頭から9個だけ採用（ロト7は7+2=9個）
        cand = [n.zfill(2) for n in cand][:9]

        if len(cand) == 9:
            main = cand[:7]
            bonus = cand[7:9]
            rows.append({
                "回別": draw_no,
                "抽せん日": date_iso,
                "本数字": " ".join(main),
                "ボーナス数字": " ".join(bonus),
            })

    # 回別で重複除去（後勝ち）
    dedup = {}
    for r in rows:
        dedup[r["回別"]] = r
    return list(dedup.values())

def merge_and_save(new_rows: List[Dict[str, str]], csv_path: str) -> pd.DataFrame:
    try:
        existing = pd.read_csv(csv_path, dtype=str)
    except FileNotFoundError:
        existing = pd.DataFrame(columns=["回別", "抽せん日", "本数字", "ボーナス数字"])

    df_new = pd.DataFrame(new_rows, columns=["回別", "抽せん日", "本数字", "ボーナス数字"])
    df_all = pd.concat([existing, df_new], ignore_index=True)

    def to_int_safe(x):
        try:
            return int("".join(ch for ch in str(x) if ch.isdigit()))
        except Exception:
            return -1

    if not df_all.empty:
        df_all["回別_num"] = df_all["回別"].map(to_int_safe)
        df_all["抽せん日_norm"] = pd.to_datetime(df_all["抽せん日"], errors="coerce")
        # 回別→日付で昇順、同一回は最後に現れたものを残す
        df_all = df_all.sort_values(["回別_num", "抽せん日_norm"], kind="stable")
        df_all = df_all.drop(columns=["回別_num", "抽せん日_norm"])
        df_all = df_all.drop_duplicates(subset=["回別"], keep="last")

    df_all.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return df_all

def fetch_html(url: str, timeout: int = 30, retries: int = 2) -> str:
    last_err = None
    headers = {"User-Agent": "Mozilla/5.0"}
    for _ in range(max(1, retries)):
        try:
            r = requests.get(url, timeout=timeout, headers=headers)
            r.raise_for_status()
            # normalize to str
            return r.text
        except Exception as e:
            last_err = e
    raise last_err

def main():
    parser = argparse.ArgumentParser(description="Scrape LOTO7 last results from Rakuten and update CSV.")
    parser.add_argument("--url", default=LASTRESULTS, help="Source URL (default: Rakuten lastresults)")
    parser.add_argument("--csv", default=CSV_PATH_DEFAULT, help="Output CSV path")
    args = parser.parse_args()

    try:
        html = fetch_html(args.url, timeout=30, retries=3)
    except Exception as e:
        print(f"HTTP error: {e}", file=sys.stderr)
        sys.exit(1)

    rows = parse_lastresults(html)
    if not rows:
        print("No data scraped from the page. The page structure might have changed.", file=sys.stderr)
        sys.exit(2)

    df = merge_and_save(rows, args.csv)
    print(f"Fetched {len(rows)} rows from: {args.url}")
    # Show the latest 12 draws
    with pd.option_context("display.max_colwidth", None):
        print(df.tail(12).to_string(index=False))

if __name__ == "__main__":
    main()
