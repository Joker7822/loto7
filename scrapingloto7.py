import csv
from datetime import datetime
import time
import sys
from typing import List, Dict

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException
import pandas as pd

URL = "https://www.mizuhobank.co.jp/takarakuji/check/loto/loto7/index.html"
CSV_PATH = "loto7.csv"

def build_driver() -> webdriver.Chrome:
    opts = Options()
    # Headless & stability
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1280,1600")
    # Be polite & reduce resources
    opts.add_argument("--blink-settings=imagesEnabled=false")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument("--disable-infobars")
    opts.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36")
    driver = webdriver.Chrome(options=opts)
    driver.set_page_load_timeout(60)
    return driver

def wait_for_results(wait: WebDriverWait):
    """Wait until at least one result block is present."""
    # Any one of these classes should exist when data has rendered.
    locators = [
        (By.CSS_SELECTOR, ".js-lottery-issue-pc"),
        (By.CSS_SELECTOR, ".js-lottery-date-pc"),
        (By.CSS_SELECTOR, ".js-lottery-number-pc"),
    ]
    last_err = None
    for _ in range(2):  # two passes to give the site a moment
        for by, sel in locators:
            try:
                wait.until(EC.presence_of_element_located((by, sel)))
                return
            except TimeoutException as e:
                last_err = e
        time.sleep(1.2)
    # If we reach here, raise the last error
    if last_err:
        raise last_err

def parse_page(driver: webdriver.Chrome) -> List[Dict[str, str]]:
    wait = WebDriverWait(driver, 30)
    wait_for_results(wait)

    # Strategy:
    #   1) Use each issue element (js-lottery-issue-pc) as an anchor.
    #   2) For each anchor, find its nearest container (row/card), then read date, main numbers, bonus numbers within that container.
    results = []

    issue_elems = driver.find_elements(By.CSS_SELECTOR, ".js-lottery-issue-pc")
    for issue in issue_elems:
        try:
            # Find a stable container encompassing all bits for one draw.
            container = None
            try:
                container = issue.find_element(By.XPATH, "./ancestor::tr[1]")
            except NoSuchElementException:
                pass
            if container is None:
                try:
                    container = issue.find_element(By.XPATH, "./ancestor::div[1]")
                except NoSuchElementException:
                    container = issue

            # 1) 回別
            draw_no = issue.text.strip()
            if not draw_no:
                # Occasionally text is in a child span
                try:
                    draw_no = issue.find_element(By.XPATH, ".//*").text.strip()
                except Exception:
                    pass
            if not draw_no:
                continue

            # 2) 抽せん日
            date_text = ""
            for sel in [".js-lottery-date-pc", ".js-lottery-date", ".lottery-date"]:
                try:
                    date_text = container.find_element(By.CSS_SELECTOR, sel).text.strip()
                    if date_text:
                        break
                except NoSuchElementException:
                    continue

            # Normalize date like "2025年8月8日(金)" -> "2025-08-08"
            draw_date = ""
            if date_text:
                # Remove anything in parentheses (weekday)
                date_text_norm = date_text.split("（")[0].split("(")[0]
                date_text_norm = date_text_norm.replace("　", "").strip()
                try:
                    draw_date = datetime.strptime(date_text_norm, "%Y年%m月%d日").strftime("%Y-%m-%d")
                except ValueError:
                    # Fallback: try with zero-padded month/day issues removed
                    try:
                        # Replace possible spaces like "2025年 8月 8日"
                        date_text_norm = date_text_norm.replace(" ", "")
                        draw_date = datetime.strptime(date_text_norm, "%Y年%m月%d日").strftime("%Y-%m-%d")
                    except Exception:
                        draw_date = date_text  # keep original if parsing fails

            # 3) 本数字 (7個)
            main_nums_elems = container.find_elements(By.CSS_SELECTOR, ".js-lottery-number-pc")
            main_numbers = []
            for el in main_nums_elems:
                t = el.text.strip()
                if t:
                    main_numbers.append(t.zfill(2))
            if len(main_numbers) >= 7:
                main_numbers = main_numbers[:7]
            else:
                # As a fallback, try any digits within the container excluding parentheses
                if not main_numbers:
                    try:
                        raw = container.text
                        # Extract two-digit numbers standing alone (quick heuristic)
                        import re
                        main_numbers = re.findall(r"(?<!\d)(\d{1,2})(?!\d)", raw)
                        main_numbers = [n.zfill(2) for n in main_numbers if n.isdigit()][:7]
                    except Exception:
                        main_numbers = []

            # 4) ボーナス数字 (2個)
            bonus_elems = container.find_elements(By.CSS_SELECTOR, ".js-lottery-bonus-pc")
            bonus_numbers = []
            for el in bonus_elems:
                t = el.text.strip().strip("()（）")
                if t:
                    bonus_numbers.append(t.zfill(2))
            if len(bonus_numbers) >= 2:
                bonus_numbers = bonus_numbers[:2]

            if not (draw_no and draw_date and len(main_numbers) == 7 and len(bonus_numbers) >= 1):
                # If essential parts are missing, skip
                continue

            results.append({
                "回別": draw_no,
                "抽せん日": draw_date,
                "本数字": " ".join(main_numbers),
                "ボーナス数字": " ".join(bonus_numbers[:2]) if bonus_numbers else ""
            })
        except StaleElementReferenceException:
            # Try once more for a stale item
            try:
                issue_refreshed = driver.find_element(By.XPATH, f"//*[@class='js-lottery-issue-pc' and normalize-space(text())='{issue.text}']")
                if issue_refreshed:
                    issue = issue_refreshed
                    continue
            except Exception:
                pass
            continue
        except Exception:
            continue

    return results

def merge_and_save(new_rows: List[Dict[str, str]], csv_path: str = CSV_PATH):
    # Read existing CSV if present
    try:
        existing = pd.read_csv(csv_path, dtype=str)
    except FileNotFoundError:
        existing = pd.DataFrame(columns=["回別", "抽せん日", "本数字", "ボーナス数字"])

    df_new = pd.DataFrame(new_rows, columns=["回別", "抽せん日", "本数字", "ボーナス数字"])
    df_all = pd.concat([existing, df_new], ignore_index=True)

    # Normalize 回別 to numeric for sorting if possible
    def to_int_safe(x):
        try:
            return int("".join(ch for ch in str(x) if ch.isdigit()))
        except Exception:
            return None

    if not df_all.empty:
        df_all["回別_num"] = df_all["回別"].map(to_int_safe)
        df_all = df_all.sort_values(["回別_num", "抽せん日"], ascending=[True, True], kind="stable")
        df_all = df_all.drop(columns=["回別_num"])
        # Drop duplicates by 回別 (keep the latest/last occurrence)
        df_all = df_all.drop_duplicates(subset=["回別"], keep="last")

    # Save with BOM for easy Excel opening
    df_all.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return df_all

def main():
    driver = build_driver()
    try:
        driver.get(URL)
        # In case there is a lazy rendering, give it a moment
        time.sleep(1.0)
        data = parse_page(driver)
    finally:
        try:
            driver.quit()
        except Exception:
            pass

    if not data:
        print("No data scraped. The page structure might have changed.", file=sys.stderr)
        return

    df = merge_and_save(data, CSV_PATH)
    # Print only newly added rows for visibility
    print("Fetched rows:", len(data))
    # Display the last 5 rows as a quick check
    try:
        print(df.tail(5).to_string(index=False))
    except Exception:
        pass

if __name__ == "__main__":
    main()
