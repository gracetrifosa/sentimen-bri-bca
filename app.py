import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import io, requests, certifi

# ===== 0) CONFIG HALAMAN =====
st.set_page_config(page_title="Dashboard Analisis Sentimen", layout="wide")
st.title("📊 Dashboard Analisis Sentimen BRImo vs BCA")

# ===== 1) LOAD DATA OTOMATIS DARI GITHUB =====
URL = "https://raw.githubusercontent.com/gracetrifosa/sentimen-bri-bca/main/ulasan_dilabeli_indobert.csv"

@st.cache_data(show_spinner=False)
def load_data_url(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=30, verify=certifi.where())
    r.raise_for_status()
    try:
        return pd.read_csv(io.StringIO(r.text), sep=None, engine="python")
    except Exception:
        return pd.read_csv(io.StringIO(r.text), sep="\t")

df = load_data_url(URL)

# ===== 2) NORMALISASI KOLOM =====
rename_map = {
    "ulasan": "comment", "komentar": "comment", "review": "comment", "text": "comment",
    "sentimen": "sentiment", "label": "sentiment", "sentiment_label": "sentiment",
    "aplikasi": "app_name", "app": "app_name", "appname": "app_name"
}
for k, v in rename_map.items():
    if k in df.columns and v not in df.columns:
        df = df.rename(columns={k: v})

if "at" in df.columns and "date" not in df.columns:
    df["date"] = pd.to_datetime(df["at"], errors="coerce")

# ===== 3) VALIDASI KOLOM WAJIB =====
required = {"comment", "sentiment", "app_name"}
missing = required - set(df.columns)
if missing:
    st.error(f"Kolom wajib hilang: {', '.join(missing)}. "
             f"Harus ada {required}. Kolom terdeteksi: {list(df.columns)}")
    st.stop()

# ===== 4) NORMALISASI NILAI SENTIMEN =====
df["sentiment"] = (
    df["sentiment"].astype(str).str.lower().str.strip()
      .replace({"positive": "positif", "negative": "negatif", "neutral": "netral",
                "pos": "positif", "neg": "negatif"})
)

# ===== 5) FILTER =====
st.sidebar.header("🔎 Filter")
apps = sorted(df["app_name"].dropna().unique().tolist())
sentiments = ["positif", "negatif", "netral"]

colA, colB = st.sidebar.columns(2)
all_apps = colA.checkbox("Pilih semua app", value=True)
all_sents = colB.checkbox("Pilih semua sentimen", value=True)

f_apps = st.sidebar.multiselect("Pilih Aplikasi", apps, default=apps if all_apps else [])
f_sents = st.sidebar.multiselect("Pilih Sentimen", sentiments, default=sentiments if all_sents else [])

filtered = df[df["app_name"].isin(f_apps) & df["sentiment"].isin(f_sents)]

if filtered.empty:
    st.warning("Tidak ada data sesuai filter. Silakan pilih minimal satu aplikasi dan satu sentimen di sidebar.")
    st.stop()

# ===== 6) METRIK =====
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Ulasan", len(filtered))
c2.metric("Positif", (filtered["sentiment"] == "positif").sum())
c3.metric("Negatif", (filtered["sentiment"] == "negatif").sum())
c4.metric("Netral",  (filtered["sentiment"] == "netral").sum())

# ===== 7) DISTRIBUSI SENTIMEN (BAR) =====
st.subheader("Distribusi Sentimen")
counts = filtered["sentiment"].value_counts().reindex(sentiments, fill_value=0)

col1, col2 = st.columns([1,1])
with col1:
    fig1, ax1 = plt.subplots(figsize=(3,2))
    ax1.bar(counts.index, counts.values, color=["#4CAF50","#F44336","#FFC107"])
    ax1.set_ylabel("Jumlah Ulasan")
    st.pyplot(fig1, clear_figure=True)

# ===== 8) PERBANDINGAN PER APLIKASI (BAR) =====
st.subheader("Perbandingan Sentimen per Aplikasi")
if not filtered.empty and filtered["app_name"].nunique() > 0:
    pivot = (
        filtered.groupby(["app_name", "sentiment"]).size()
                .unstack(fill_value=0).reindex(columns=sentiments, fill_value=0)
    )
    if pivot.size > 0:
        with col2:
            fig2, ax2 = plt.subplots(figsize=(3.5,2.5))
            pivot.plot(kind="bar", ax=ax2)
            ax2.set_ylabel("Jumlah Ulasan")
            st.pyplot(fig2, clear_figure=True)
    else:
        st.info("Tidak ada kombinasi app_name×sentiment untuk ditampilkan.")

# ===== 9) WORD CLOUD =====
st.subheader("Word Cloud (berdasarkan hasil filter)")
stop_id = set(StopWordRemoverFactory().get_stop_words())
extra_stop = {"brimo", "bca", "bank", "aplikasi", "nya", "yg", "ga", "aja", "dll"}
stop_all = STOPWORDS.union(stop_id).union(extra_stop)

text_wc = " ".join(filtered["comment"].astype(str).tolist())
if text_wc.strip():
    wc = WordCloud(
        width=400, height=250,         
        background_color="white",
        stopwords=stop_all,
        max_words=50,
        max_font_size=60,               
        relative_scaling=0.5
    ).generate(text_wc)

    st.image(
        wc.to_array(),
        caption="Word Cloud Ringkas",
        width=600,                      
        use_container_width=False   
    )
else:
    st.info("Tidak ada teks untuk dibuat word cloud. Coba longgarkan filter.")
