"""Streamlit dashboard for the Product Review Intelligence Engine."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

import streamlit as st

st.set_page_config(
    page_title="Product Review Intelligence",
    page_icon="🔍",
    layout="wide",
)

# --- Header ---
st.title("Product Review Intelligence Engine")
st.markdown(
    "Analyze product reviews for **sentiment** and **aspect-level insights**. "
    "Paste a review below or try one of the examples."
)

# --- Initialize pipeline ---
@st.cache_resource
def load_pipeline():
    """Load the prediction pipeline once and cache it."""
    try:
        from src.pipeline.prediction_pipeline import PredictionPipeline
        return PredictionPipeline()
    except FileNotFoundError:
        return None


pipeline = load_pipeline()

if pipeline is None:
    st.error(
        "Model not found. Please run `python train.py` first to train the model, "
        "then restart this app."
    )
    st.stop()

# --- Sidebar: Sample Reviews ---
st.sidebar.header("Sample Reviews")
sample_reviews = {
    "Positive (Battery)": (
        "Amazing battery life on this device. Lasts two full days with heavy use. "
        "The charging speed is also great, fully charged in under an hour."
    ),
    "Negative (Screen)": (
        "Very disappointed with the screen quality. The display is dim even at max "
        "brightness and the resolution is terrible for the price."
    ),
    "Mixed (Quality + Price)": (
        "The build quality feels solid and durable, but honestly it is way too "
        "expensive for what you get. Not worth the money."
    ),
    "Neutral (Shipping)": (
        "Product arrived on time. Packaging was okay. Nothing special about the "
        "delivery experience."
    ),
    "Positive (Sound + Performance)": (
        "The speaker sound is fantastic, crystal clear audio at all volume levels. "
        "Performance is fast and responsive with zero lag."
    ),
}

selected_sample = st.sidebar.selectbox(
    "Load a sample review:", ["-- Select --"] + list(sample_reviews.keys())
)

# --- Main Input ---
col_input, col_title = st.columns([3, 1])

with col_title:
    review_title = st.text_input("Review Title (optional)", value="")

with col_input:
    default_text = ""
    if selected_sample != "-- Select --":
        default_text = sample_reviews[selected_sample]

    review_text = st.text_area(
        "Paste your product review here:",
        value=default_text,
        height=150,
        placeholder="e.g., The battery life is amazing but the screen is too dim...",
    )

# --- Analyze Button ---
analyze_clicked = st.button("Analyze Review", type="primary", use_container_width=True)

if analyze_clicked and review_text.strip():
    with st.spinner("Analyzing review..."):
        result = pipeline.predict(
            review_text=review_text,
            review_title=review_title,
        )

    # --- Results ---
    st.markdown("---")
    st.subheader("Analysis Results")

    # Sentiment badge
    sentiment = result["sentiment"]
    confidence = result["confidence"]

    col_sent, col_conf = st.columns(2)

    with col_sent:
        color_map = {
            "positive": "green",
            "neutral": "orange",
            "negative": "red",
        }
        color = color_map.get(sentiment, "gray")
        st.markdown(
            f"### Sentiment: "
            f"<span style='color:{color}; font-weight:bold; font-size:1.3em;'>"
            f"{sentiment.upper()}</span>",
            unsafe_allow_html=True,
        )

    with col_conf:
        st.metric("Confidence", f"{confidence:.1%}")

    # --- Aspect Breakdown ---
    aspects = result["aspects"]
    if aspects:
        st.subheader("Aspect Breakdown")

        aspect_data = []
        for aspect, asp_sentiment in aspects.items():
            icon = {"positive": "+", "negative": "-", "neutral": "~"}.get(asp_sentiment, "~")
            aspect_data.append({
                "Aspect": aspect.capitalize(),
                "Sentiment": asp_sentiment.capitalize(),
                "Indicator": icon,
            })

        # Display as a styled table
        for item in aspect_data:
            asp_color = color_map.get(item["Sentiment"].lower(), "gray")
            st.markdown(
                f"- **{item['Aspect']}**: "
                f"<span style='color:{asp_color};'>{item['Sentiment']}</span>",
                unsafe_allow_html=True,
            )
    else:
        st.info("No specific product aspects detected in this review.")

    # --- Word Cloud Placeholder ---
    st.subheader("Word Cloud")
    st.markdown(
        """
        <div style="
            border: 2px dashed #ccc;
            border-radius: 10px;
            padding: 60px;
            text-align: center;
            color: #999;
            background-color: #fafafa;
        ">
            <p style="font-size: 1.2em;">Word Cloud Visualization</p>
            <p>Install the <code>wordcloud</code> package and enable this section
            to generate a word cloud from the review text.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

elif analyze_clicked:
    st.warning("Please enter some review text before analyzing.")

# --- Footer ---
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#888;'>"
    "Product Review Intelligence Engine | "
    "Powered by LightGBM, XGBoost, Logistic Regression | "
    "Dataset: Amazon Reviews 2023"
    "</div>",
    unsafe_allow_html=True,
)
