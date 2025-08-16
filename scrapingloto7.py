import csv
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd

# === Chrome設定 ===
options = Options()
options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

driver = webdriver.Chrome(options=options)
url = "https://www.mizuhobank.co.jp/takarakuji/check/loto/loto7/index.html"
driver.get(url)

wait = WebDriverWait(driver, 10)

data = []

try:
    draw_count = len(wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, "js-lottery-date-pc"))))

    for i in range(draw_count):
        draw_date_elements = driver.find_elements(By.CLASS_NAME, "js-lottery-date-pc")
        main_numbers_elements = driver.find_elements(By.CLASS_NAME, "js-lottery-number-pc")
        bonus_number_elements = driver.find_elements(By.CLASS_NAME, "js-lottery-bonus-pc")
        draw_number_elements = driver.find_elements(By.CLASS_NAME, "js-lottery-issue-pc")

        if not draw_date_elements[i].text.strip():
            continue

        draw_date = datetime.strptime(draw_date_elements[i].text, "%Y年%m月%d日").strftime("%Y-%m-%d")
        main_numbers = " ".join([main_numbers_elements[j].text for j in range(i * 7, (i + 1) * 7)])
        bonus_numbers = " ".join([bonus_number_elements[j].text.strip("()") for j in range(i * 2, (i + 1) * 2)])
        draw_number = draw_number_elements[i].text

        data.append({
            "回別": draw_number,
            "抽せん日": draw_date,
            "本数字": main_numbers,
            "ボーナス数字": bonus_numbers
        })

except Exception as e:
    print("Error occurred:", e)
finally:
    driver.quit()

# === 保存処理 ===
csv_path = "loto7.csv"
try:
    existing = pd.read_csv(csv_path)
    existing_dates = existing["抽せん日"].tolist()
    fieldnames = existing.columns.tolist()
except FileNotFoundError:
    existing = pd.DataFrame()
    existing_dates = []
    fieldnames = ["回別", "抽せん日", "本数字", "ボーナス数字"]

new_data = [row for row in data if row["抽せん日"] not in existing_dates]

if new_data:
    with open(csv_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if existing.empty:
            writer.writeheader()
        writer.writerows(new_data)

# 結果の表示
for row in new_data:
    print(row)
