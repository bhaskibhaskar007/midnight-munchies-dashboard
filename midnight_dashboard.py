import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import traceback
import json
import math

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# --------------------------
# Page config & CSS
# --------------------------
st.set_page_config(page_title="Midnight Munchies — Final Dashboard", layout="wide", page_icon="🌙")

def _load_css():
    css = """
    <style>
      body { background: linear-gradient(120deg,#000428 0%, #001219 100%); }
      .title { font-size:40px; font-weight:800; color:#fff; text-align:center;
               text-shadow:0 0 12px rgba(0,230,255,0.12); margin-bottom:6px; }
      .subtitle { color:#dbeafe; text-align:center; margin-bottom:14px; }
      .divider { height:3px; margin:18px 0; background: linear-gradient(90deg,transparent,#00eaff,transparent); }
      .section { color:#fff; font-weight:700; font-size:20px; margin-top:12px; }
      .glass { background: rgba(255,255,255,0.04); border-radius:12px; padding:12px; border:1px solid rgba(255,255,255,0.06); }
      .kpi { font-size:28px; font-weight:800; color:#00eaff; text-align:center; }
      .kpit { text-align:center; color:white; margin-top:6px; font-size:13px; }
      .small { font-size:13px; color:#dbeafe; }
      .btn { background:#00eaff; color:black; padding:6px 12px; border-radius:8px; }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

_load_css()

# --------------------------
# Helpers
# --------------------------
def detect_column(df, keywords):
    for kw in keywords:
        for c in df.columns:
            if kw.lower() in c.lower():
                return c
    return None

def safe_read_csv(path):
    try:
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception as e:
        st.error(f"Could not read CSV at {path}: {e}")
        st.stop()

def try_to_download_fig(fig, default_name):
    """Try to export fig to PNG using kaleido; fallback to HTML."""
    try:
        # requires kaleido
        img_bytes = fig.to_image(format="png", scale=2)
        return ("image/png", img_bytes, f"{default_name}.png")
    except Exception:
        # fallback to HTML
        html = fig.to_html(include_plotlyjs='cdn')
        return ("text/html", html.encode("utf-8"), f"{default_name}.html")

def write_pdf_insights(path, insights_lines):
    c = canvas.Canvas(path, pagesize=letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 750, "Midnight Munchies - Insights Report")
    c.setFont("Helvetica", 11)
    y = 720
    for line in insights_lines:
        c.drawString(50, y, line)
        y -= 18
        if y < 60:
            c.showPage()
            y = 750
    c.save()

# --------------------------
# Load data
# --------------------------
# Update this path if your CSV is somewhere else
# --------------------------
# Load data
# --------------------------
# Update this path if your CSV is somewhere else
# --------------------------
# Load data
# --------------------------
# Always load CSV from the same folder as this Python file
# --------------------------
# Load data (FINAL – CLEAN)
# --------------------------
import os
import pandas as pd

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "projectsurveyresponse.csv")

if not os.path.exists(DATA_PATH):
    st.error("❌ CSV file not found. Please place projectsurveyresponse.csv in the assets folder.")
    st.stop()

df_orig = pd.read_csv(DATA_PATH)



if not os.path.exists(DATA_PATH):
    st.error("❌ CSV file not found. Please place the CSV in the same folder as midnight_dashboard.py")
    st.stop()


df_orig = safe_read_csv(DATA_PATH)

# drop personal columns
df = df_orig.copy()
for c in list(df.columns):
    if any(x in c.lower() for x in ("name", "mail", "email", "phone")):
        df.drop(columns=[c], inplace=True, errors="ignore")

# standardize column names spaces trimmed already
# auto-detect important columns (robust)
snack_col = detect_column(df, ["go-to late-night snack", "go to late", "what's your go-to", "what's your go to", "snack"])
freq_col  = detect_column(df, ["how often", "frequency", "how frequently"])
time_col  = detect_column(df, ["when do your midnight cravings", "when do your", "craving time", "when do you"])
spend_col = detect_column(df, ["spend", "weekly spend", "how much do you typically spend"])
drive_col = detect_column(df, ["what drives", "drive", "reason"])
sleep_col = detect_column(df, ["sleep quality", "sleep"])
mood_col  = detect_column(df, ["next morning", "morning", "mood"])
with_col  = detect_column(df, ["who do you usually snack with", "who do", "with whom"])
habit_col = detect_column(df, ["funniest", "habit", "funny"])

# sanity: if any are None, set to a safe fallback so code doesn't crash; we will hide charts if missing
cols_detected = {
    "snack": snack_col, "freq": freq_col, "time": time_col, "spend": spend_col,
    "drive": drive_col, "sleep": sleep_col, "mood": mood_col, "with": with_col, "habit": habit_col
}

# create MoodScore for KPI (if mood present)
if mood_col in df:
    # create a deterministic mapping for moods by sorted unique values
    uniq_moods = list(df[mood_col].dropna().unique())
    mood_map = {m: i+1 for i, m in enumerate(uniq_moods)}
    df["MoodScore"] = df[mood_col].map(mood_map)
else:
    df["MoodScore"] = pd.NA

# --------------------------
# App structure: tabs
# --------------------------
tabs = st.tabs(["Home", "Analytics", "Story Mode", "ML", "Insights", "Chatbot", "Downloads", "Comments"])

# --------------------------
# Theme toggle (simple)
# --------------------------
with tabs[0]:
# --------------------------
# CSV Upload Section
# --------------------------
    st.markdown("### 📂 Upload your CSV file")

    uploaded_file = st.file_uploader(
    "Upload a CSV file to analyze",
    type=["csv"]
)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    
    # Home tab will also show theme toggle at top-right using radio
    st.markdown("<div style='display:flex; justify-content:space-between; align-items:center;'>", unsafe_allow_html=True)
    st.markdown("<div></div>", unsafe_allow_html=True)
    theme = st.radio("Theme", options=["Dark", "Light"], horizontal=True)
    st.markdown("</div>", unsafe_allow_html=True)

# set plotly template according to theme
plotly_template = "plotly_dark" if theme == "Dark" else "plotly_white"

# --------------------------
# HOME tab
# --------------------------
with tabs[0]:
    st.markdown("<div class='title'>🍕 Midnight Munchies</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Presentation-ready dashboard — press tabs to explore.</div>", unsafe_allow_html=True)
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.markdown("<b>Project:</b> Midnight Munchies - Student Late-night Snacking Survey<br>"
                "<b>Author:</b> Bhaskar <br>"
                "<b>Description:</b> An interactive data analytics dashboard analyzing student late-night snacking habits.", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # start buttons (navigation hint)
    st.markdown("**Quick links**")
    cols = st.columns(4)
    for i, name in enumerate(["Analytics","Story Mode","ML","Insights"]):
        cols[i].markdown(f"<div style='text-align:center'><a href='#{name}' style='color:#00eaff'>{name}</a></div>", unsafe_allow_html=True)

# --------------------------
# ANALYTICS tab
# --------------------------
with tabs[1]:
    st.markdown("<div class='section'>📊 Analytics</div>", unsafe_allow_html=True)

    # KPI row (6 cards)
    k1, k2, k3 = st.columns(3)
    k4, k5, k6 = st.columns(3)

    k1.markdown(f"<div class='glass'><div class='kpi'>{len(df)}</div><div class='kpit'>Total Responses</div></div>", unsafe_allow_html=True)
    if snack_col:
        k2.markdown(f"<div class='glass'><div class='kpi'>{df[snack_col].nunique()}</div><div class='kpit'>Unique Snacks</div></div>", unsafe_allow_html=True)
    else:
        k2.markdown(f"<div class='glass'><div class='kpi'>N/A</div><div class='kpit'>Unique Snacks</div></div>", unsafe_allow_html=True)
    if time_col:
        peak_time = df[time_col].value_counts().idxmax() if df[time_col].notna().any() else "N/A"
        k3.markdown(f"<div class='glass'><div class='kpi'>{peak_time}</div><div class='kpit'>Peak Craving Time</div></div>", unsafe_allow_html=True)
    else:
        k3.markdown(f"<div class='glass'><div class='kpi'>N/A</div><div class='kpit'>Peak Craving Time</div></div>", unsafe_allow_html=True)

    if spend_col:
        common_spend = df[spend_col].value_counts().idxmax() if df[spend_col].notna().any() else "N/A"
        k4.markdown(f"<div class='glass'><div class='kpi'>{common_spend}</div><div class='kpit'>Common Spend Range</div></div>", unsafe_allow_html=True)
    else:
        k4.markdown(f"<div class='glass'><div class='kpi'>N/A</div><div class='kpit'>Common Spend Range</div></div>", unsafe_allow_html=True)

    if drive_col:
        top_reason = df[drive_col].value_counts().idxmax() if df[drive_col].notna().any() else "N/A"
        k5.markdown(f"<div class='glass'><div class='kpi'>{top_reason}</div><div class='kpit'>Top Snacking Reason</div></div>", unsafe_allow_html=True)
    else:
        k5.markdown(f"<div class='glass'><div class='kpi'>N/A</div><div class='kpit'>Top Snacking Reason</div></div>", unsafe_allow_html=True)

    if "MoodScore" in df and not pd.isna(df["MoodScore"]).all():
        avg_mood = df["MoodScore"].mean()
        k6.markdown(f"<div class='glass'><div class='kpi'>{avg_mood:.1f}</div><div class='kpit'>Avg Mood Score</div></div>", unsafe_allow_html=True)
    else:
        k6.markdown(f"<div class='glass'><div class='kpi'>N/A</div><div class='kpit'>Avg Mood Score</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Charts: arrange in visually-appealing groups
    # Helper to safely run a chart
    def _safe_chart(fn):
        try:
            fn()
        except Exception:
            st.error("Chart rendering error:")
            st.code(traceback.format_exc())

    # 1. Snack donut + pie
    if snack_col:
        st.markdown("### 🍩 Snack Distribution")
        vc = df[snack_col].value_counts().reset_index()
        vc.columns = ["Snack", "Count"]
        fig = px.pie(vc, names="Snack", values="Count", hole=0.45, template=plotly_template, title="Snack Share (Donut)")
        st.plotly_chart(fig, width='stretch')
        # download button
        mime, data_bytes, fname = try_to_download_fig(fig, "snack_donut")
        st.download_button("Download snack chart", data=data_bytes, file_name=fname, mime=mime)
    else:
        st.info("Snack column not detected — snack charts hidden.")

    # 2. Treemap
    if snack_col:
        st.markdown("### 🌳 Treemap")
        vc2 = vc.copy()
        fig_tm = px.treemap(vc2, path=["Snack"], values="Count", color="Count", template=plotly_template)
        st.plotly_chart(fig_tm, width='stretch')
        mime, data_bytes, fname = try_to_download_fig(fig_tm, "snack_treemap")
        st.download_button("Download treemap", data=data_bytes, file_name=fname, mime=mime)

    # 3. Sunburst snack vs time
    if snack_col and time_col:
        st.markdown("### ☀️ Snack vs Time (Sunburst)")
        sun = df.groupby([snack_col, time_col]).size().reset_index(name="Count")
        fig_sb = px.sunburst(sun, path=[snack_col, time_col], values="Count", template=plotly_template)
        st.plotly_chart(fig_sb, width='stretch')
        mime, data_bytes, fname = try_to_download_fig(fig_sb, "sunburst")
        st.download_button("Download sunburst", data=data_bytes, file_name=fname, mime=mime)

    # 4. Funnel frequency
    if freq_col:
        st.markdown("### 🔻 Snack Frequency (Funnel)")
        freq_df = df[freq_col].value_counts().reset_index()
        freq_df.columns = ["Frequency", "Count"]
        fig_fun = px.funnel(freq_df, x="Count", y="Frequency", template=plotly_template)
        st.plotly_chart(fig_fun, width='stretch')
        mime, data_bytes, fname = try_to_download_fig(fig_fun, "funnel")
        st.download_button("Download funnel", data=data_bytes, file_name=fname, mime=mime)

    # 5. Polar moods
    if mood_col:
        st.markdown("### 🧭 Morning Mood (Polar)")
        mdf = df[mood_col].value_counts().reset_index()
        mdf.columns = ["Mood", "Count"]
        fig_pol = px.line_polar(mdf, r="Count", theta="Mood", line_close=True, template=plotly_template)
        st.plotly_chart(fig_pol, width='stretch')
        mime, data_bytes, fname = try_to_download_fig(fig_pol, "mood_polar")
        st.download_button("Download polar chart", data=data_bytes, file_name=fname, mime=mime)

    # 6. Heatmap snack vs mood
    if snack_col and mood_col:
        st.markdown("### 🔥 Heatmap — Snack vs Mood")
        pivot = pd.crosstab(df[snack_col], df[mood_col])
        fig_ht = px.imshow(pivot, text_auto=True, color_continuous_scale="Plasma", template=plotly_template)
        st.plotly_chart(fig_ht, width='stretch')
        mime, data_bytes, fname = try_to_download_fig(fig_ht, "heatmap")
        st.download_button("Download heatmap", data=data_bytes, file_name=fname, mime=mime)

    # 7. Trend over time (if Timestamp exists)
    if "Timestamp" in df.columns:
        try:
            st.markdown("### 📈 Responses Over Time")
            ts = df.copy()
            ts["Timestamp_parsed"] = pd.to_datetime(ts["Timestamp"], errors="coerce")
            ts["Day"] = ts["Timestamp_parsed"].dt.date
            trend = ts.groupby("Day").size().reset_index(name="Responses")
            fig_line = px.line(trend, x="Day", y="Responses", markers=True, template=plotly_template)
            st.plotly_chart(fig_line, width='stretch')
            mime, data_bytes, fname = try_to_download_fig(fig_line, "trend")
            st.download_button("Download trend chart", data=data_bytes, file_name=fname, mime=mime)
        except Exception:
            st.info("Timestamp column exists but could not parse dates.")

    # 8. Stacked bar: drive vs mood
    if drive_col and mood_col:
        st.markdown("### 🎯 Reason vs Morning Mood (Stacked)")
        stck = pd.crosstab(df[drive_col], df[mood_col])
        fig_stk = px.bar(stck, barmode="stack", template=plotly_template)
        st.plotly_chart(fig_stk, width='stretch')
        mime, data_bytes, fname = try_to_download_fig(fig_stk, "stacked_reason_mood")
        st.download_button("Download stacked chart", data=data_bytes, file_name=fname, mime=mime)

    # 9. Bubble chart: reason counts
    if drive_col:
        st.markdown("### 🔵 Bubble — Reasons Popularity")
        bub = df[drive_col].value_counts().reset_index()
        bub.columns = ["Reason", "Count"]
        fig_bub = px.scatter(bub, x="Reason", y="Count", size="Count", color="Count", template=plotly_template)
        st.plotly_chart(fig_bub, width='stretch')
        mime, data_bytes, fname = try_to_download_fig(fig_bub, "bubble")
        st.download_button("Download bubble chart", data=data_bytes, file_name=fname, mime=mime)

    st.markdown("<div style='margin-top:18px'/>", unsafe_allow_html=True)

# --------------------------
# STORY MODE tab (slide-by-slide)
# --------------------------
with tabs[2]:
    st.markdown("<div class='section'>📽 Story Mode — Walkthrough</div>", unsafe_allow_html=True)
    if "story_slide" not in st.session_state:
        st.session_state["story_slide"] = 0

    slides = [
        {"title": "Overview", "text": f"Total responses: {len(df)}. Most popular snack: {df[snack_col].value_counts().idxmax() if snack_col else 'N/A'}."},
        {"title": "Snack Patterns", "text": "Here we discuss which snacks are preferred and their share."},
        {"title": "Spending", "text": f"Most common spend range: {df[spend_col].value_counts().idxmax() if spend_col else 'N/A'}."},
        {"title": "Sleep Impact", "text": f"Common sleep outcome after snacking: {df[sleep_col].value_counts().idxmax() if sleep_col else 'N/A'}."},
        {"title": "Conclusions", "text": "Final recommendations & tips to improve sleep and reduce spend."}
    ]

    col_prev, col_slide, col_next = st.columns([1,6,1])
    with col_slide:
        st.markdown(f"### {slides[st.session_state['story_slide']]['title']}")
        st.write(slides[st.session_state['story_slide']]['text'])
        # show an example chart per slide (simple)
        if st.session_state['story_slide'] == 0 and snack_col:
            vc = df[snack_col].value_counts().reset_index()
            vc.columns = ["Snack","Count"]
            fig = px.bar(vc, x="Snack", y="Count", template=plotly_template)
            st.plotly_chart(fig, width='stretch')
        if st.session_state['story_slide'] == 1 and snack_col:
            vc = df[snack_col].value_counts().reset_index()
            vc.columns = ["Snack","Count"]
            fig = px.treemap(vc, path=["Snack"], values="Count", template=plotly_template)
            st.plotly_chart(fig, width='stretch')
        if st.session_state['story_slide'] == 2 and spend_col:
            vc = df[spend_col].value_counts().reset_index()
            vc.columns = ["Range","Count"]
            fig = px.bar(vc, x="Range", y="Count", template=plotly_template)
            st.plotly_chart(fig, width='stretch')
        if st.session_state['story_slide'] == 3 and sleep_col:
            vc = df[sleep_col].value_counts().reset_index()
            vc.columns = ["Sleep","Count"]
            fig = px.pie(vc, names="Sleep", values="Count", template=plotly_template)
            st.plotly_chart(fig, width='stretch')
    with col_prev:
        if st.button("◀ Prev") and st.session_state["story_slide"] > 0:
            st.session_state["story_slide"] -= 1
    with col_next:
        if st.button("Next ▶") and st.session_state["story_slide"] < len(slides)-1:
            st.session_state["story_slide"] += 1

# --------------------------
# ML tab
# --------------------------
with tabs[3]:
    st.markdown("<div class='section'>🤖 Machine Learning</div>", unsafe_allow_html=True)
    st.markdown("Simple classification model (Sleep Quality) — easy to explain in viva.", unsafe_allow_html=True)

    if not (snack_col and time_col and spend_col and drive_col and sleep_col):
        st.info("Not enough columns for ML training. ML section requires snack, time, spend-range, drive, sleep columns.")
    else:
        ml = df[[snack_col, time_col, spend_col, drive_col, sleep_col]].dropna().copy()
        enc_sn = LabelEncoder().fit(ml[snack_col].astype(str))
        enc_tm = LabelEncoder().fit(ml[time_col].astype(str))
        enc_sp = LabelEncoder().fit(ml[spend_col].astype(str))
        enc_dr = LabelEncoder().fit(ml[drive_col].astype(str))
        enc_sl = LabelEncoder().fit(ml[sleep_col].astype(str))

        ml["sn"] = enc_sn.transform(ml[snack_col].astype(str))
        ml["tm"] = enc_tm.transform(ml[time_col].astype(str))
        ml["sp"] = enc_sp.transform(ml[spend_col].astype(str))
        ml["dr"] = enc_dr.transform(ml[drive_col].astype(str))
        ml["sl"] = enc_sl.transform(ml[sleep_col].astype(str))

        X = ml[["sn","tm","sp","dr"]]
        y = ml["sl"]
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)
        clf = RandomForestClassifier(n_estimators=150, random_state=42)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        st.write(f"Model Accuracy: **{acc*100:.1f}%**")
        # feature importance
        fi = pd.DataFrame({
            "Feature": ["Snack","Craving Time","Spend Range","Reason"],
            "Importance": clf.feature_importances_
        }).sort_values("Importance", ascending=False)
        fig_fi = px.bar(fi, x="Feature", y="Importance", color="Importance", template=plotly_template)
        st.plotly_chart(fig_fi, width='stretch')

        st.markdown("#### Quick prediction (demo)")
        c1, c2, c3 = st.columns(3)
        choice_sn = c1.selectbox("Snack", ml[snack_col].unique())
        choice_tm = c2.selectbox("Time", ml[time_col].unique())
        choice_sp = c3.selectbox("Spend Range", ml[spend_col].unique())

        input_row = pd.DataFrame([{
            "sn": enc_sn.transform([choice_sn])[0],
            "tm": enc_tm.transform([choice_tm])[0],
            "sp": enc_sp.transform([choice_sp])[0],
            "dr": enc_dr.transform([ml[drive_col].dropna().unique()[0]])[0]  # use most common reason as default
        }])

        pred_code = clf.predict(input_row)[0]
        pred_label = enc_sl.inverse_transform([pred_code])[0]
        st.success(f"Predicted Sleep Quality: **{pred_label}**")

# --------------------------
# Insights tab (text insights & explanation)
# --------------------------
with tabs[4]:
    st.markdown("<div class='section'>🧠 Predictive Insights & Recommendations</div>", unsafe_allow_html=True)
    # Automatic insights generator: simple heuristics
    insights = []
    try:
        if snack_col and df[snack_col].notna().any():
            top_snack = df[snack_col].value_counts().idxmax()
            insights.append(f"Most popular snack: {top_snack}. Consider stocking less perishable options of this snack.")
        if time_col and df[time_col].notna().any():
            top_time = df[time_col].value_counts().idxmax()
            insights.append(f"Peak snacking window: {top_time}. Recommend targeted interventions (healthy options) at this time.")
        if spend_col and df[spend_col].notna().any():
            top_spend = df[spend_col].value_counts().idxmax()
            insights.append(f"Typical weekly spend range: {top_spend}. Consider budget-friendly meal plans or discounts.")
        if drive_col and df[drive_col].notna().any():
            top_drive = df[drive_col].value_counts().idxmax()
            insights.append(f"Top driver: {top_drive}. Address this with awareness (stress management / alternatives).")
        if sleep_col and df[sleep_col].notna().any():
            top_sleep = df[sleep_col].value_counts().idxmax()
            insights.append(f"Common sleep outcome: {top_sleep}. Suggest reducing heavy snacks at late hours.")
    except Exception:
        insights.append("Could not compute some insights due to missing columns.")

    st.markdown("### Key insights (auto-generated)")
    for i, line in enumerate(insights, start=1):
        st.write(f"{i}. {line}")

    st.markdown("### Recommendations (easy to explain)")
    st.write("""
    - Replace fried snacks with lighter options after 11 PM.
    - Provide awareness about sleep hygiene; avoid heavy snacking 1 hour before sleep.
    - Offer bundle deals for popular snacks to reduce average spend.
    - Organize campaigns around the peak snacking window to promote healthier choices.
    """)

# --------------------------
# Chatbot tab (simple local Q&A)
# --------------------------
with tabs[5]:
    st.markdown("<div class='section'>💬 Chatbot — Ask about the data</div>", unsafe_allow_html=True)
    st.markdown("Type simple questions like: 'most popular snack', 'when do people snack', 'spend range', 'top reason'")

    user_q = st.text_input("Ask a question about the dataset (simple):")
    if st.button("Ask") and user_q.strip() != "":
        q = user_q.lower()
        answer = "Sorry, I couldn't find an answer."
        # simple rule-based responses
        if "popular" in q or "most" in q or "top snack" in q or "most popular snack" in q:
            if snack_col: answer = f"The most popular snack is **{df[snack_col].value_counts().idxmax()}**."
        elif "when" in q or "time" in q or "craving" in q:
            if time_col: answer = f"Peak craving time: **{df[time_col].value_counts().idxmax()}**."
        elif "spend" in q or "money" in q or "how much" in q:
            if spend_col: answer = f"Most common spend range: **{df[spend_col].value_counts().idxmax()}**."
        elif "reason" in q or "drive" in q:
            if drive_col: answer = f"Top reason for snacking: **{df[drive_col].value_counts().idxmax()}**."
        elif "sleep" in q:
            if sleep_col: answer = f"Common sleep outcome after snacking: **{df[sleep_col].value_counts().idxmax()}**."
        elif "mood" in q:
            if mood_col: answer = f"Most common next-morning mood: **{df[mood_col].value_counts().idxmax()}**."
        st.markdown(answer)

# --------------------------
# Downloads tab (pdf + chart exports)
# --------------------------
with tabs[6]:
    st.markdown("<div class='section'>📦 Downloads</div>", unsafe_allow_html=True)
    st.markdown("Generate a quick PDF report of top insights or download charts shown in Analytics.")

    # PDF export
    if st.button("Generate Insights PDF"):
        lines = []
        if snack_col and df[snack_col].notna().any():
            lines.append(f"Most popular snack: {df[snack_col].value_counts().idxmax()}")
        if time_col and df[time_col].notna().any():
            lines.append(f"Peak craving time: {df[time_col].value_counts().idxmax()}")
        if spend_col and df[spend_col].notna().any():
            lines.append(f"Common spend range: {df[spend_col].value_counts().idxmax()}")
        if drive_col and df[drive_col].notna().any():
            lines.append(f"Top reason: {df[drive_col].value_counts().idxmax()}")
        if sleep_col and df[sleep_col].notna().any():
            lines.append(f"Typical sleep outcome: {df[sleep_col].value_counts().idxmax()}")
        path = "MidnightMunchies_Insights.pdf"
        write_pdf_insights(path, lines)
        with open(path, "rb") as f:
            st.download_button("Download PDF", data=f, file_name="MidnightMunchies_Insights.pdf", mime="application/pdf")

    st.markdown("**Download dataset (CSV)**")
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv_bytes, file_name="cleaned_midnight_munchies_export.csv", mime="text/csv")

# --------------------------
# Comments tab
# --------------------------
with tabs[7]:
    st.markdown("<div class='section'>💭 Comments & Feedback</div>", unsafe_allow_html=True)
    if "user_comments" not in st.session_state:
        st.session_state["user_comments"] = []
    name = st.text_input("Your name (optional)")
    comment = st.text_area("Your comment / feedback")
    if st.button("Add Comment"):
        if comment.strip() != "":
            st.session_state.user_comments.append({"name": name or "Anonymous", "comment": comment})
            st.success("Comment added.")
        else:
            st.error("Please enter a comment.")
    st.markdown("### Comments")
    for idx, c in enumerate(st.session_state.user_comments[::-1], start=1):
        st.markdown(f"**{c['name']}**: {c['comment']}")
    # download comments
    if st.session_state.user_comments:
        comments_df = pd.DataFrame(st.session_state.user_comments)
        st.download_button("Download comments CSV", data=comments_df.to_csv(index=False).encode("utf-8"),
                           file_name="comments.csv", mime="text/csv")

# --------------------------
# Audio summary (browser TTS) included in Home & Insights
# --------------------------
# Create a JS snippet to use browser speechSynthesis for summary text
summary_text = "This is the Midnight Munchies dashboard. Press play to hear the summary of insights."

# place audio player on Insights tab as well
with tabs[4]:
    st.markdown("<div class='section'>🔊 Audio Summary</div>", unsafe_allow_html=True)
    summary_for_audio = "Most students prefer " + (df[snack_col].value_counts().idxmax() if snack_col else "various snacks") + \
                        ". Peak snacking time: " + (df[time_col].value_counts().idxmax() if time_col else "various times") + \
                        ". Typical spend is " + (df[spend_col].value_counts().idxmax() if spend_col else "not available") + "."
    # html button to play using JS speechSynthesis
    play_js = f"""
    <div>
      <button onclick="
        const msg = new SpeechSynthesisUtterance({json.dumps(summary_for_audio)});
        msg.rate = 0.95;
        speechSynthesis.cancel();
        speechSynthesis.speak(msg);
      " style='background:#00eaff;border:none;padding:8px;border-radius:8px;color:black;font-weight:700'>Play Audio Summary</button>
    </div>
    """
    st.markdown(play_js, unsafe_allow_html=True)

# --------------------------
# Final note / Dataset preview (in Home)
# --------------------------
with tabs[0]:
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='small'>Dataset preview (first 10 rows)</div>", unsafe_allow_html=True)
    st.dataframe(df.head(10), width='stretch')

# End of file
