# loto7_app.py

import streamlit as st
st.set_page_config(page_title="Loto7予測AI", layout="wide")  # ← 最初の st 呼び出し

import pandas as pd
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

# ========= パス定義（実行ファイル基準） =========
APP_DIR = Path(__file__).parent

LOG_FILE = APP_DIR / "last_prediction_log.txt"
SCRAPING_LOG = APP_DIR / "scraping_log.txt"

PRED_CSV = APP_DIR / "loto7_predictions.csv"     # 予測CSV（例）
EVAL_CSV = APP_DIR / "loto7_prediction_evaluation_with_bonus.csv"     # 評価結果CSV（統一）
EVAL_SUMMARY = APP_DIR / "loto7_evaluation_summary.txt"
PROGRESS_TXT = APP_DIR / "progress_dashboard.txt"

# ========= 時刻（JST） =========
def now_jst() -> datetime:
    return datetime.now(ZoneInfo("Asia/Tokyo"))

# ========= 実行済み（当日）管理 =========
def already_predicted_today() -> bool:
    today_str = now_jst().strftime("%Y-%m-%d")
    if LOG_FILE.exists():
        try:
            last_run = LOG_FILE.read_text(encoding="utf-8").strip()
            return last_run == today_str
        except Exception:
            return False
    return False

def mark_prediction_done() -> None:
    today_str = now_jst().strftime("%Y-%m-%d")
    try:
        LOG_FILE.write_text(today_str, encoding="utf-8")
    except Exception:
        pass

def display_scraping_log() -> None:
    if SCRAPING_LOG.exists():
        try:
            log_content = SCRAPING_LOG.read_text(encoding="utf-8")
            st.markdown("### 🪵 スクレイピングログ")
            st.text_area("Log Output", log_content, height=300)
        except Exception as e:
            st.warning(f"ログの読み込みに失敗しました: {e}")

# ========= ヘルパー（安全CSV読込） =========
def safe_read_csv(path: Path, uploader_key: str, label: str) -> pd.DataFrame | None:
    """
    CSVを安全に読み込み。無ければアップローダで受け付ける。
    失敗時は None を返す。
    """
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception as e:
            st.error(f"❌ {label} の読み込みでエラー: {e}")
    else:
        st.warning(f"⚠️ {label} が見つかりません（{path.name}）")
    up = st.file_uploader(f"{label} をアップロード（.csv）", type=["csv"], key=uploader_key)
    if up:
        try:
            return pd.read_csv(up)
        except Exception as e:
            st.error(f"❌ アップロードされた {label} の読み込みでエラー: {e}")
    return None

# ========= UI ヘッダー =========
st.markdown("<h1 style='color:#FF4B4B;'>🎯 Loto7 予測AI</h1>", unsafe_allow_html=True)

menu = st.sidebar.radio(
    "📌 メニュー",
    ["🧠 最新予測表示", "📊 予測評価", "📉 予測分析グラフ", "🧾 予測結果表示"]
)

# ========= 画面: 最新予測表示 =========
if "最新予測" in menu:
    st.markdown("## 🧠 最新予測結果")
    pred_df = safe_read_csv(PRED_CSV, "pred_uploader", "予測CSV")
    if isinstance(pred_df, pd.DataFrame) and not pred_df.empty:
        try:
            # 抽せん日が文字列の場合を考慮しつつ降順ソート
            if "抽せん日" in pred_df.columns:
                # 変換が失敗してもそのままソートにフォールバック
                try:
                    pred_df["_抽せん日_dt"] = pd.to_datetime(pred_df["抽せん日"], errors="coerce")
                    latest_row = pred_df.sort_values(["_抽せん日_dt", "抽せん日"], ascending=False).iloc[0]
                except Exception:
                    latest_row = pred_df.sort_values("抽せん日", ascending=False).iloc[0]
            else:
                latest_row = pred_df.iloc[-1]

            y1 = latest_row.get("予測1", "N/A")
            y2 = latest_row.get("予測2", "N/A")
            draw_date = latest_row.get("抽せん日", "N/A")

            st.success("✅ 最新予測が取得されました")
            st.markdown(
                f"""
                <div style='padding: 1.5rem; background-color: #f0f8ff; border-radius: 12px; text-align: center;'>
                    <h2 style='color:#4B9CD3; margin-bottom: 0.6rem;'>📅 抽せん日: {draw_date}</h2>
                    <p style='font-size: 2.4rem; color: #FF4B4B; margin: 0.3rem 0;'>🎯 <strong>予測1:</strong> {y1}</p>
                    <p style='font-size: 2.1rem; color: #00aa88; margin: 0.3rem 0;'>💡 <strong>予測2:</strong> {y2}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.error(f"❌ 最新予測の表示でエラー: {e}")
    else:
        st.info("予測CSVを用意/アップロードしてください。")

# ========= 画面: 予測評価 =========
elif "予測評価" in menu:
    st.markdown("## 📊 予測精度の評価")

    col1, col2 = st.columns(2)
    with col1:
        run_eval = st.button("🧪 評価を実行")
    with col2:
        show_log = st.checkbox("スクレイピングログを表示", value=False)

    if show_log:
        display_scraping_log()

    if run_eval:
        try:
            # 任意: あなたの評価ロジックに合わせて修正してください
            # 例: numbers3_predictor モジュールの関数を呼ぶ
            from loto7_predictor import evaluate_and_summarize_predictions  # 無ければ except へ
            with st.spinner("評価中..."):
                evaluate_and_summarize_predictions(
                    output_csv=str(EVAL_CSV),
                    summary_path=str(EVAL_SUMMARY),
                )
            st.success("✅ 評価が完了しました")
            mark_prediction_done()
        except ModuleNotFoundError:
            st.info("ℹ️ 評価ロジックのモジュールが見つかりませんでした。手動でCSVをアップロードして確認できます。")
        except NameError:
            st.info("ℹ️ evaluate_and_summarize_predictions が未定義です。")
        except Exception as e:
            st.error(f"❌ 評価中にエラー: {e}")

    # サマリ表示
    if EVAL_SUMMARY.exists():
        try:
            summary = EVAL_SUMMARY.read_text(encoding="utf-8")
            st.text_area("📄 評価概要", summary, height=400)
        except Exception as e:
            st.error(f"❌ 評価サマリ読み込みエラー: {e}")

    # 評価CSV表示
    eval_df = safe_read_csv(EVAL_CSV, "eval_uploader", "評価結果CSV")
    if isinstance(eval_df, pd.DataFrame) and not eval_df.empty:
        st.markdown("### 📋 評価結果")
        st.dataframe(eval_df, use_container_width=True)
    else:
        st.info("評価結果CSVが未読込です。実行するかアップロードしてください。")

# ========= 画面: 予測分析グラフ（簡易テキスト表示版） =========
elif "分析グラフ" in menu:
    st.markdown("## 📉 予測の分析グラフ（テキストダッシュボード）")

    eval_df = safe_read_csv(EVAL_CSV, "eval_uploader_graph", "評価結果CSV")
    if isinstance(eval_df, pd.DataFrame) and not eval_df.empty:
        st.info("📊 月別収益・直近成績をテキストで表示します。")
        try:
            # 例: ダッシュボード生成（存在しない場合も握りつぶす）
            from numbers3_predictor import generate_progress_dashboard_text
            generate_progress_dashboard_text(output_path=str(PROGRESS_TXT))
        except Exception:
            pass

        if PROGRESS_TXT.exists():
            try:
                dashboard_text = PROGRESS_TXT.read_text(encoding="utf-8")
                st.text_area("📈 成績ダッシュボード", dashboard_text, height=420)
            except Exception as e:
                st.error(f"❌ ダッシュボード読み込みエラー: {e}")
        else:
            st.warning("⚠️ progress_dashboard.txt が見つかりません。生成関数が無い/未実行の可能性があります。")
    else:
        st.warning("⚠️ 先に評価結果CSVを用意してください。")

# ========= 画面: 予測結果表示（一覧） =========
elif "予測結果" in menu:
    st.markdown("## 🧾 最新の予測結果（過去10件）")
    pred_df = safe_read_csv(PRED_CSV, "pred_uploader_list", "予測CSV")
    if isinstance(pred_df, pd.DataFrame) and not pred_df.empty:
        try:
            if "抽せん日" in pred_df.columns:
                # 表示用に並び替え
                try:
                    pred_df["_抽せん日_dt"] = pd.to_datetime(pred_df["抽せん日"], errors="coerce")
                    view_df = pred_df.sort_values(["_抽せん日_dt", "抽せん日"], ascending=False).drop(columns=["_抽せん日_dt"])
                except Exception:
                    view_df = pred_df.sort_values("抽せん日", ascending=False)
            else:
                view_df = pred_df.copy()

            st.dataframe(view_df.head(10), use_container_width=True)
        except Exception as e:
            st.error(f"❌ 予測一覧の表示でエラー: {e}")
    else:
        st.info("予測CSVを用意/アップロードしてください。")
