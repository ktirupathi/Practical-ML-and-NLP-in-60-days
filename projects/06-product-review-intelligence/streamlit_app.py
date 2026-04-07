"""Streamlit dashboard for Product Review Intelligence.

Features:
- Paste a review to get aspect cards with sentiment badges
- Overall sentiment display with star rating prediction
- Word cloud generated from the review text
- Sidebar with sample reviews to try
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

st.set_page_config(
    page_title="Product Review Intelligence",
    page_icon="magnifying glass",
    layout="wide",
)

SAMPLE_REVIEWS = {
    "Positive — Battery & Performance": (
        "Absolutely love this laptop! The battery life lasts over 12 hours with heavy use. "
        "Performance is blazing fast — no lag even with 20 browser tabs open. "
        "The build quality feels premium and the keyboard is a joy to type on."
    ),
    "Negative — Screen & Speaker": (
        "Very disappointed with this tablet. The screen resolution is terrible even at maximum "
        "brightness. The speaker sound is tinny and distorted at high volumes. "
        "For this price I expected much better display quality and audio performance."
    ),
    "Mixed — Camera good, Battery bad": (
        "The camera quality is outstanding, especially in low light conditions. Photos are crisp "
        "and detailed. However the battery life is absolutely dreadful — barely lasts 4 hours. "
        "Charging speed is also very slow compared to competitors."
    ),
    "Neutral — Average product": (
        "This headphone is okay for the price. Sound quality is decent but nothing special. "
        "The build feels a bit cheap but it works fine. Connectivity is stable. "
        "Overall a reasonable budget option if you are not an audiophile."
    ),
}

SENTIMENT_COLORS = {"positive": "#22c55e", "neutral": "#f59e0b", "negative": "#ef4444"}
SENTIMENT_BG = {"positive": "#f0fdf4", "neutral": "#fffbeb", "negative": "#fef2f2"}


@st.cache_resource(show_spinner="Loading AI models…")
def load_pipeline():
    """Load the prediction pipeline once and cache it."""
    try:
        from src.pipeline.prediction_pipeline import PredictionPipeline
        return PredictionPipeline()
    except Exception as e:
        return None, str(e)


# ── Page layout ────────────────────────────────────────────────────────────────

st.title("Product Review Intelligence")
st.markdown(
    "Paste a product review to extract **aspects**, analyse their **sentiment**, "
    "get a **star rating prediction**, and visualise a **word cloud**."
)

# Sidebar sample reviews
st.sidebar.header("Try a Sample Review")
sample_choice = st.sidebar.selectbox(
    "Load sample:", ["— Select —"] + list(SAMPLE_REVIEWS.keys())
)

pipeline = load_pipeline()
if isinstance(pipeline, tuple):
    st.error(f"Failed to load pipeline: {pipeline[1]}")
    st.stop()

# Input section
col_text, col_meta = st.columns([4, 1])
with col_text:
    default_text = SAMPLE_REVIEWS.get(sample_choice, "") if sample_choice != "— Select —" else ""
    review_text = st.text_area(
        "Review text", value=default_text, height=160,
        placeholder="e.g., Battery life is excellent but the screen is too dim…",
    )

with col_meta:
    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("Analyse Review", type="primary", use_container_width=True)

# ── Analysis ───────────────────────────────────────────────────────────────────

if analyze_btn and review_text.strip():
    with st.spinner("Analysing …"):
        result = pipeline.analyze(review_text)

    st.markdown("---")
    st.subheader("Analysis Results")

    # Overall sentiment + star rating
    overall = result["overall_sentiment"]
    rating = result["rating_prediction"]
    col_sent, col_stars, col_summary = st.columns([2, 2, 4])

    with col_sent:
        color = SENTIMENT_COLORS.get(overall, "#6b7280")
        bg = SENTIMENT_BG.get(overall, "#f9fafb")
        st.markdown(
            f"""<div style='background:{bg}; border-left:4px solid {color};
                padding:16px; border-radius:8px;'>
                <p style='margin:0; font-size:0.85em; color:#6b7280;'>Overall Sentiment</p>
                <p style='margin:0; font-size:1.6em; font-weight:700; color:{color};'>
                {overall.upper()}</p></div>""",
            unsafe_allow_html=True,
        )

    with col_stars:
        stars = "★" * rating + "☆" * (5 - rating)
        st.markdown(
            f"""<div style='background:#f0f9ff; border-left:4px solid #3b82f6;
                padding:16px; border-radius:8px;'>
                <p style='margin:0; font-size:0.85em; color:#6b7280;'>Predicted Rating</p>
                <p style='margin:0; font-size:1.6em; color:#f59e0b;'>{stars}</p>
                <p style='margin:0; font-size:0.9em; color:#374151;'>{rating} / 5 stars</p>
                </div>""",
            unsafe_allow_html=True,
        )

    with col_summary:
        st.markdown(
            f"""<div style='background:#f9fafb; border-left:4px solid #9ca3af;
                padding:16px; border-radius:8px; height:100%;'>
                <p style='margin:0; font-size:0.85em; color:#6b7280;'>Summary</p>
                <p style='margin:0; font-size:1em; color:#374151;'>{result["summary"]}</p>
                </div>""",
            unsafe_allow_html=True,
        )

    # Aspect cards
    st.markdown("### Aspect Breakdown")
    aspects = result.get("aspects", [])
    if aspects:
        cols = st.columns(min(len(aspects), 3))
        for i, asp in enumerate(aspects):
            col = cols[i % 3]
            sentiment = asp["sentiment"]
            color = SENTIMENT_COLORS.get(sentiment, "#6b7280")
            bg = SENTIMENT_BG.get(sentiment, "#f9fafb")
            score_pct = f"{asp['score'] * 100:.0f}%"
            with col:
                st.markdown(
                    f"""<div style='background:{bg}; border:1px solid {color};
                        border-radius:10px; padding:14px; margin-bottom:10px;'>
                        <p style='margin:0 0 4px; font-weight:600; color:#1f2937;'>
                        {asp["aspect"].title()}</p>
                        <span style='background:{color}; color:white; font-size:0.8em;
                        padding:2px 8px; border-radius:12px;'>{sentiment}</span>
                        <span style='color:#6b7280; font-size:0.8em; margin-left:6px;'>
                        {score_pct}</span></div>""",
                    unsafe_allow_html=True,
                )
    else:
        st.info("No specific product aspects detected in this review.")

    # Word cloud
    st.markdown("### Word Cloud")
    try:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt

        wc = WordCloud(width=800, height=300, background_color="white",
                       colormap="Blues", max_words=60).generate(review_text)
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
    except ImportError:
        st.markdown(
            "<div style='border:2px dashed #d1d5db; border-radius:10px; padding:40px; "
            "text-align:center; color:#9ca3af;'>"
            "Install <code>wordcloud</code> and <code>matplotlib</code> to enable word clouds."
            "</div>",
            unsafe_allow_html=True,
        )

elif analyze_btn:
    st.warning("Please enter some review text before clicking Analyse.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#9ca3af; font-size:0.85em;'>"
    "Product Review Intelligence | spaCy aspects · cardiffnlp RoBERTa sentiment · "
    "Amazon Reviews 2023 (Electronics)"
    "</div>",
    unsafe_allow_html=True,
)
