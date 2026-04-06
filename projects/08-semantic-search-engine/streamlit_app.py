"""
Streamlit Search UI: Interactive semantic search interface.

Features:
- Query input box
- Adjustable top-k slider
- Results with passage text, similarity score, and passage ID
- Search latency display
- Backend selector (FAISS or ChromaDB)
"""

import streamlit as st
import time

st.set_page_config(
    page_title="Semantic Search Engine",
    page_icon="🔍",
    layout="wide",
)


@st.cache_resource
def load_pipeline():
    """Load the prediction pipeline (cached across reruns)."""
    from src.pipeline.prediction_pipeline import PredictionPipeline
    from src.config.configuration import Config

    config = Config()
    pipeline = PredictionPipeline(config)
    pipeline.load()
    return pipeline


def main():
    st.title("Semantic Search Engine")
    st.markdown("Search MS MARCO passages using sentence embeddings and vector similarity.")

    # Sidebar configuration
    st.sidebar.header("Settings")
    top_k = st.sidebar.slider("Number of results (top-k)", min_value=1, max_value=50, value=10)
    backend = st.sidebar.radio("Search backend", ["FAISS", "ChromaDB"])
    show_scores = st.sidebar.checkbox("Show similarity scores", value=True)
    show_ids = st.sidebar.checkbox("Show passage IDs", value=False)

    # Load pipeline
    try:
        pipeline = load_pipeline()
        st.sidebar.success(f"Index loaded: {pipeline.index.ntotal:,} passages")
    except FileNotFoundError:
        st.error(
            "Search index not found. Please run `python train.py` first to build the index."
        )
        st.stop()
    except Exception as e:
        st.error(f"Failed to load search pipeline: {e}")
        st.stop()

    # Search input
    query = st.text_input(
        "Enter your search query:",
        placeholder="e.g., What is machine learning?",
    )

    # Example queries
    st.markdown("**Try these example queries:**")
    example_cols = st.columns(4)
    examples = [
        "what is machine learning",
        "how does photosynthesis work",
        "explain python programming",
        "what is semantic search",
    ]
    for col, example in zip(example_cols, examples):
        if col.button(example, use_container_width=True):
            query = example

    if query:
        with st.spinner("Searching..."):
            try:
                if backend == "FAISS":
                    response = pipeline.search(query, top_k=top_k)
                else:
                    response = pipeline.search_chromadb(query, top_k=top_k)
            except Exception as e:
                st.error(f"Search failed: {e}")
                return

        # Display metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Results", len(response.results))
        col2.metric("Latency", f"{response.latency_ms:.1f} ms")
        col3.metric("Indexed Passages", f"{response.total_indexed:,}")

        st.markdown("---")

        # Display results
        if not response.results:
            st.warning("No results found.")
        else:
            for result in response.results:
                # Build result card
                with st.container():
                    header_parts = [f"**Rank {result.rank}**"]
                    if show_scores:
                        header_parts.append(f"Score: `{result.score:.4f}`")
                    if show_ids:
                        header_parts.append(f"ID: `{result.passage_id}`")

                    st.markdown(" | ".join(header_parts))
                    st.markdown(result.passage)

                    # Score bar
                    if show_scores:
                        # Normalize score for progress bar (scores are cosine sim in [0,1])
                        normalized = max(0.0, min(1.0, result.score))
                        st.progress(normalized)

                    st.markdown("---")


if __name__ == "__main__":
    main()
