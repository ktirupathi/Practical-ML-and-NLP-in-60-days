"""
Streamlit Chatbot UI for the RAG Knowledge Base Chatbot.

Features:
- Chat-style conversational interface
- Source document display for each answer
- Confidence meter visualization
- Chat history with export option
- Adjustable retrieval parameters in sidebar
"""

import json
from datetime import datetime

import streamlit as st

from src.pipeline.prediction_pipeline import PredictionPipeline
from src.utils.common import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RAG Knowledge Base Chatbot",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
    st.session_state.pipeline_error = None


# ---------------------------------------------------------------------------
# Load pipeline (cached)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading RAG pipeline ...")
def load_pipeline() -> PredictionPipeline:
    """Load the prediction pipeline once and cache it."""
    return PredictionPipeline()


def ensure_pipeline():
    """Ensure the pipeline is loaded, showing errors if not."""
    if st.session_state.pipeline is None:
        try:
            st.session_state.pipeline = load_pipeline()
            st.session_state.pipeline_error = None
        except Exception as e:
            st.session_state.pipeline_error = str(e)
            logger.error(f"Failed to load pipeline: {e}")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("Settings")
    st.markdown("---")

    top_k = st.slider(
        "Number of source passages (top-k)",
        min_value=1,
        max_value=15,
        value=5,
        help="How many relevant passages to retrieve from the knowledge base.",
    )

    show_sources = st.checkbox("Show source passages", value=True)
    show_confidence = st.checkbox("Show confidence meter", value=True)

    st.markdown("---")
    st.subheader("About")
    st.markdown(
        """
        This chatbot uses **Retrieval-Augmented Generation (RAG)** to answer
        questions from a knowledge base built on **SQuAD 2.0 Wikipedia passages**.

        **How it works:**
        1. Your question is embedded into a vector
        2. Similar passages are retrieved from ChromaDB
        3. A language model generates an answer from the retrieved context

        **Models used:**
        - Embeddings: `all-MiniLM-L6-v2`
        - Generator: `flan-t5-base`
        """
    )

    st.markdown("---")

    # Export chat history
    if st.session_state.chat_history:
        export_data = json.dumps(
            st.session_state.chat_history, indent=2, default=str
        )
        st.download_button(
            label="Export chat history",
            data=export_data,
            file_name=f"chat_history_{datetime.now():%Y%m%d_%H%M%S}.json",
            mime="application/json",
        )

    if st.button("Clear chat history"):
        st.session_state.chat_history = []
        st.rerun()


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("RAG Knowledge Base Chatbot")
st.caption("Ask questions about Wikipedia topics - powered by SQuAD 2.0")

# Load pipeline
ensure_pipeline()

if st.session_state.pipeline_error:
    st.error(
        f"Could not load the RAG pipeline: {st.session_state.pipeline_error}\n\n"
        "Please run `python train.py` first to build the knowledge base."
    )
    st.stop()

# Display chat history
for entry in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(entry["question"])

    with st.chat_message("assistant"):
        st.write(entry["answer"])

        if show_confidence and "confidence" in entry:
            confidence = entry["confidence"]
            col1, col2 = st.columns([3, 1])
            with col1:
                st.progress(
                    confidence,
                    text=f"Confidence: {confidence:.1%}",
                )
            with col2:
                if confidence >= 0.7:
                    st.success("High")
                elif confidence >= 0.4:
                    st.warning("Medium")
                else:
                    st.error("Low")

        if show_sources and "sources" in entry and entry["sources"]:
            with st.expander(
                f"Source passages ({len(entry['sources'])} retrieved)"
            ):
                for i, src in enumerate(entry["sources"], 1):
                    st.markdown(f"**Source {i}** — *{src['title']}*")
                    st.text(src["text"][:500])
                    st.caption(
                        f"Distance: {src['distance']:.4f} | "
                        f"Context ID: {src['context_id'][:12]}..."
                    )
                    if i < len(entry["sources"]):
                        st.markdown("---")

# Chat input
question = st.chat_input("Ask a question about any topic ...")

if question:
    # Display user message immediately
    with st.chat_message("user"):
        st.write(question)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base and generating answer ..."):
            try:
                result = st.session_state.pipeline.predict(
                    question=question, top_k=top_k
                )
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

        st.write(result["answer"])

        if show_confidence:
            confidence = result["confidence"]
            col1, col2 = st.columns([3, 1])
            with col1:
                st.progress(
                    confidence,
                    text=f"Confidence: {confidence:.1%}",
                )
            with col2:
                if confidence >= 0.7:
                    st.success("High")
                elif confidence >= 0.4:
                    st.warning("Medium")
                else:
                    st.error("Low")

        if show_sources and result["sources"]:
            with st.expander(
                f"Source passages ({len(result['sources'])} retrieved)"
            ):
                for i, src in enumerate(result["sources"], 1):
                    st.markdown(f"**Source {i}** — *{src['title']}*")
                    st.text(src["text"][:500])
                    st.caption(
                        f"Distance: {src['distance']:.4f} | "
                        f"Context ID: {src['context_id'][:12]}..."
                    )
                    if i < len(result["sources"]):
                        st.markdown("---")

    # Save to chat history
    st.session_state.chat_history.append(
        {
            "question": question,
            "answer": result["answer"],
            "confidence": result["confidence"],
            "sources": result["sources"],
            "timestamp": datetime.now().isoformat(),
        }
    )
