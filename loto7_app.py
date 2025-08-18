import streamlit as st
st.set_page_config(page_title="Loto7予測AI", layout="wide")  # 必ず最上部！
import pandas as pd
import os
from datetime import datetime, time
from zoneinfo import ZoneInfo

# ========= ファイル定義 =========
LOG_FILE = "last_prediction_log.txt"
SCRAPING_LOG = "scraping_log.txt"

# ========= JST 時刻取得 =========
def now_jst():
    return datetime.now(ZoneInfo("Asia/Tokyo"))

# ========= 予測済みチェック =========
def already_predicted_today():
    today_str = now_jst().strftime("%Y-%m-%d")
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            last_run = f.read().strip()
            return last_run == today_str
    return False

def mark_prediction_done():
    today_str = now_jst().strftime("%Y-%m-%d")
    with open(LOG_FILE, "w") as f:
        f.write(today_str)

def display_scraping_log():
    if os.path.exists(SCRAPING_LOG):
        with open(SCRAPING_LOG, "r", encoding="utf-8") as f:
            log_content = f.read()
        st.markdown("### 🪵 スクレイピングログ")
        st.text_area("Log Output", log_content, height=300)

# ========= UI =========
st.markdown("<h1 style='color:#FF4B4B;'>🎯 loto7 予測AI</h1>", unsafe_allow_html=True)

menu = st.sidebar.radio("📌 メニュー", [
    "🧠 最新予測表示",
    "📊 予測評価",
    "📉 予測分析グラフ",
    "🧾 予測結果表示"
])

# 最新予測表示
if "最新予測" in menu:
    st.markdown("## 🧠 最新予測結果")
    if os.path.exists("loto7_predictions.csv"):
        try:
            pred_df = pd.read_csv("loto7_predictions.csv")
            latest_row = pred_df.sort_values("抽せん日", ascending=False).iloc[0]

            st.success("✅ 最新予測が取得されました")
            st.markdown(f"""
                <div style='padding: 1.5rem; background-color: #f0f8ff; border-radius: 10px; text-align: center;'>
                    <h2 style='color:#4B9CD3;'>📅 抽せん日: {latest_row['抽せん日']}</h2>
                    <p style='font-size: 2.8rem; color: #FF4B4B;'>🎯 <strong>予測:</strong> {latest_row['予測1']}</p>
                    <p style='font-size: 2.4rem; color: #00aa88;'>💡 <strong>予測:</strong> {latest_row['予測2']}</p>
                </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❌ 予測CSV読み込みエラー: {e}")
    else:
        st.warning("⚠️ loto7_predictions.csv が見つかりません。")

# 予測評価
elif "予測評価" in menu:
    st.markdown("## 📊 予測精度の評価")
    if st.button("🧪 評価を実行"):
        with st.spinner("評価中..."):
            evaluate_and_summarize_predictions()
        st.success("✅ 評価が完了しました")

    if os.path.exists("evaluation_summary.txt"):
        with open("loto7_evaluation_summary.txt", encoding="utf-8") as f:
            summary = f.read()
        st.text_area("📄 評価概要", summary, height=400)

    if os.path.exists("loto7_prediction_evaluation_with_bonus.csv"):
        eval_df = pd.read_csv("evaluation_result.csv")
        st.markdown("### 📋 評価結果")
        st.dataframe(eval_df, use_container_width=True)

# 予測分析グラフ
elif "分析グラフ" in menu:
    st.markdown("## 📉 予測の分析グラフ")

    if os.path.exists("evaluation_result.csv"):
        from numbers3_predictor import generate_progress_dashboard_text

        st.info("📊 月別収益・直近5日間の成績をテキストで表示します")

        try:
            # ダッシュボード.txtを生成
            generate_progress_dashboard_text()

            # 生成したファイルを読み込んで表示
            if os.path.exists("progress_dashboard.txt"):
                with open("progress_dashboard.txt", encoding="utf-8") as f:
                    dashboard_text = f.read()
                st.text_area("📈 成績ダッシュボード", dashboard_text, height=400)
            else:
                st.warning("⚠️ progress_dashboard.txt が生成されていません。")

        except Exception as e:
            st.error(f"❌ ダッシュボード生成中にエラー: {e}")

    else:
        st.warning("⚠️ evaluation_result.csv が見つかりません。先に予測・評価を実行してください。")

# 予測結果表示
elif "予測結果" in menu:
    st.markdown("## 🧾 最新の予測結果（過去10件）")
    if os.path.exists("loto7_predictions.csv"):
        pred_df = pd.read_csv("loto7_predictions.csv")
        st.dataframe(pred_df.sort_values("抽せん日", ascending=False).head(10), use_container_width=True)
    else:
        st.warning("⚠️ 予測結果がありません。まずは GitHub へ CSV をアップロードしてください。")
