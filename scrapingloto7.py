import re
import sys
from datetime import datetime
from typing import List, Dict
import pandas as pd

import requests
from bs4 import BeautifulSoup

LASTRESULTS = "https://takarakuji.rakuten.co.jp/backnumber/loto7/lastresults/"
CSV_PATH = "loto7.csv"

def parse_lastresults(html: str) -> List[Dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ", strip=True)
    pat = re.compile(r"第\s*0*(\d{1,4})回\s*([0-9]{4}/[01][0-9]/[0-3][0-9])")
    rows = []
    for m in pat.finditer(text):
        draw_no = f"第{int(m.group(1))}回"
        date = datetime.strptime(m.group(2), "%Y/%m/%d").strftime("%Y-%m-%d")
        start = max(0, m.start() - 250)
        end = min(len(text), m.end() + 250)
        window = text[start:end]
        nums = [n.zfill(2) for n in re.findall(r"\b\d{1,2}\b", window)]
        y, mo, d = m.group(2).split("/")
        nums = [n for n in nums if n not in {y, mo, d}]
        if len(nums) >= 9:
            main = nums[:7]
            bonus = nums[7:9]
            rows.append({
                "回別": draw_no,
                "抽せん日": date,
                "本数字": " ".join(main),
                "ボーナス数字": " ".join(bonus),
            })
    dedup = {}
    for r in rows:
        dedup[r["回別"]] = r
    return list(dedup.values())

def merge_and_save(new_rows: List[Dict[str, str]], csv_path: str = CSV_PATH) -> pd.DataFrame:
    try:
        existing = pd.read_csv(csv_path, dtype=str)
    except FileNotFoundError:
        existing = pd.DataFrame(columns=["回別", "抽せん日", "本数字", "ボーナス数字"])
    df_new = pd.DataFrame(new_rows, columns=["回別", "抽せん日", "本数字", "ボーナス数字"])
    df_all = pd.concat([existing, df_new], ignore_index=True)
    def to_int_safe(x):
        try: return int("".join(ch for ch in str(x) if ch.isdigit()))
        except: return -1
    if not df_all.empty:
        df_all["回別_num"] = df_all["回別"].map(to_int_safe)
        df_all["抽せん日_norm"] = pd.to_datetime(df_all["抽せん日"], errors="coerce")
        df_all = df_all.sort_values(["回別_num", "抽せん日_norm"], kind="stable")
        df_all = df_all.drop(columns=["回別_num", "抽せん日_norm"])
        df_all = df_all.drop_duplicates(subset=["回別"], keep="last")
    df_all.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return df_all

def main():
    try:
        r = requests.get(LASTRESULTS, timeout=30, headers={"User-Agent":"Mozilla/5.0"})
        r.raise_for_status()
    except Exception as e:
        print(f"HTTP error: {e}", file=sys.stderr)
        return
    rows = parse_lastresults(r.text)
    if not rows:
        print("No data scraped from Rakuten lastresults. The page structure might have changed.", file=sys.stderr)
        return
    df = merge_and_save(rows, CSV_PATH)
    print(f"Fetched {len(rows)} rows from Rakuten lastresults")
    print(df.tail(12).to_string(index=False))

if __name__ == "__main__":
    main()
