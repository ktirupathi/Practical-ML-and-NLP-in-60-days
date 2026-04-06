"""Streamlit web UI for the AI Resume Screening System."""

import streamlit as st

from src.pipeline.prediction_pipeline import PredictionPipeline

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Resume Screening",
    page_icon="📄",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Cached pipeline
# ---------------------------------------------------------------------------


@st.cache_resource
def load_pipeline() -> PredictionPipeline:
    """Load the prediction pipeline once and cache it."""
    return PredictionPipeline()


# ---------------------------------------------------------------------------
# Sample resumes for quick testing
# ---------------------------------------------------------------------------

SAMPLE_RESUMES = {
    "-- Select a sample --": "",
    "Data Science": (
        "Experienced data scientist with 5 years of expertise in Python, R, machine learning, "
        "deep learning, NLP, and statistical modeling. Built recommendation engines and fraud "
        "detection systems using TensorFlow and scikit-learn. Strong background in A/B testing, "
        "data visualization with Tableau, and big data processing with Spark. MS in Computer Science."
    ),
    "Web Designing": (
        "Creative web designer with 4 years of experience in HTML5, CSS3, JavaScript, React, "
        "and responsive design. Proficient in Adobe Creative Suite, Figma, and Sketch. "
        "Developed 50+ websites for clients across e-commerce, healthcare, and education. "
        "Strong eye for UX/UI design and cross-browser compatibility."
    ),
    "HR": (
        "HR professional with 6 years of experience in talent acquisition, employee relations, "
        "payroll management, and performance appraisals. Skilled in HRIS systems including "
        "Workday and SAP SuccessFactors. Led recruitment drives hiring 200+ employees annually. "
        "Certified SHRM-CP with strong knowledge of labor laws and compliance."
    ),
}


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def main():
    st.title("AI Resume Screening System")
    st.markdown(
        "Paste or upload a resume to automatically classify it into one of **25 professional categories**."
    )

    # Attempt to load the pipeline
    try:
        pipeline = load_pipeline()
    except FileNotFoundError:
        st.error(
            "Model artifacts not found. Please run `python train.py` first to train the model."
        )
        return

    st.divider()

    col_input, col_result = st.columns([1, 1])

    with col_input:
        st.subheader("Input Resume")

        # Tab layout: paste or upload
        tab_paste, tab_upload, tab_sample = st.tabs(
            ["Paste Text", "Upload File", "Sample Resumes"]
        )

        resume_text = ""

        with tab_paste:
            resume_text = st.text_area(
                "Paste your resume text below:",
                height=350,
                placeholder="Paste the full text of a resume here ...",
            )

        with tab_upload:
            uploaded_file = st.file_uploader(
                "Upload a .txt file",
                type=["txt"],
            )
            if uploaded_file is not None:
                resume_text = uploaded_file.read().decode("utf-8", errors="ignore")
                st.text_area("File contents:", resume_text, height=350, disabled=True)

        with tab_sample:
            choice = st.selectbox("Pick a sample resume:", list(SAMPLE_RESUMES.keys()))
            if choice != "-- Select a sample --":
                resume_text = SAMPLE_RESUMES[choice]
                st.text_area("Sample text:", resume_text, height=350, disabled=True)

        predict_btn = st.button("Classify Resume", type="primary", use_container_width=True)

    with col_result:
        st.subheader("Prediction Result")

        if predict_btn:
            if not resume_text or len(resume_text.strip()) < 20:
                st.warning("Please enter at least 20 characters of resume text.")
            else:
                with st.spinner("Analyzing resume ..."):
                    result = pipeline.predict(resume_text)

                # Main prediction
                st.success(f"**Predicted Category:** {result['predicted_category']}")

                if result["confidence"] is not None:
                    st.metric("Confidence", f"{result['confidence'] * 100:.1f}%")

                # Top 3 predictions chart
                if result["top_3_predictions"]:
                    st.markdown("#### Top 3 Predictions")

                    for item in result["top_3_predictions"]:
                        pct = item["confidence"] * 100
                        st.markdown(f"**{item['category']}**")
                        st.progress(item["confidence"], text=f"{pct:.1f}%")
        else:
            st.info("Enter a resume and click **Classify Resume** to see predictions.")

    # Sidebar: category list
    with st.sidebar:
        st.header("Supported Categories")
        try:
            categories = pipeline.categories
            for i, cat in enumerate(categories, 1):
                st.markdown(f"{i}. {cat}")
        except Exception:
            st.write("Categories will appear after training.")


if __name__ == "__main__":
    main()
