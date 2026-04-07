"""
Streamlit UI — Semantic Search Engine (Project 8).

Features:
  - Search bar with example queries
  - Adjustable top-k slider
  - Cross-encoder re-ranking toggle
  - Results with highlighted snippets, similarity score bars
  - "Did you mean?" suggestion for misspelled queries
  - Latency + index-size metrics in the header
"""

import time

import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Semantic Search Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# Resource caching
# ─────────────────────────────────────────────────────────────────────────────


@st.cache_resource(show_spinner="Loading search index …")
def load_searcher():
    """Load the Searcher once and cache it for the app lifetime."""
    from src.components.searcher import Searcher
    from src.config.configuration import Config

    cfg = Config()
    searcher = Searcher(config=cfg, use_reranker=True)
    searcher.load()
    return searcher


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Settings")
    st.markdown("---")

    top_k = st.slider(
        "Results to return (top-k)",
        min_value=1, max_value=50, value=10,
        help="How many ranked passages to display.",
    )
    use_reranker = st.toggle(
        "Enable cross-encoder re-ranking",
        value=True,
        help=(
            "Re-rank FAISS candidates with a cross-encoder for higher "
            "precision (adds ~100–300 ms per query)."
        ),
    )
    show_scores = st.checkbox("Show similarity scores", value=True)
    show_passage_id = st.checkbox("Show passage IDs", value=False)
    show_full_passage = st.checkbox("Show full passage (not just snippet)", value=False)

    st.markdown("---")
    st.subheader("About")
    st.markdown(
        """
        **Semantic Search Engine** over MS MARCO (50K passages).

        **How it works:**
        1. Query is encoded with `all-MiniLM-L6-v2`
        2. FAISS IndexFlatIP retrieves top-50 candidates
        3. *(Optional)* Cross-encoder `ms-marco-MiniLM-L-6-v2` re-ranks
        4. Highlighted snippets are returned

        **Stack:** SentenceTransformers · FAISS · FastAPI · Streamlit
        """
    )

# ─────────────────────────────────────────────────────────────────────────────
# Load searcher
# ─────────────────────────────────────────────────────────────────────────────

try:
    searcher = load_searcher()
    n_indexed = searcher._index.ntotal if searcher._index else 0
except FileNotFoundError:
    st.error(
        "Search index not found.  "
        "Please run `python train.py` first to build the FAISS index."
    )
    st.stop()
except Exception as exc:
    st.error(f"Failed to load the search index: {exc}")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

st.title("Semantic Search Engine")
st.caption(
    f"Searching {n_indexed:,} MS MARCO passages · "
    f"Model: all-MiniLM-L6-v2 · "
    f"Re-ranker: {'enabled' if use_reranker else 'disabled'}"
)

# ─────────────────────────────────────────────────────────────────────────────
# Search input
# ─────────────────────────────────────────────────────────────────────────────

query = st.text_input(
    "Search query",
    placeholder="What is machine learning?",
    label_visibility="collapsed",
)

# Example queries as clickable buttons
st.markdown("**Try an example:**")
examples = [
    "what is machine learning",
    "how does photosynthesis work",
    "explain transformer neural networks",
    "what is semantic search",
    "history of the Roman Empire",
    "how do vaccines work",
    "what is gradient descent",
    "explain quantum computing",
]
cols = st.columns(4)
for i, example in enumerate(examples):
    if cols[i % 4].button(example, use_container_width=True, key=f"ex_{i}"):
        query = example

# ─────────────────────────────────────────────────────────────────────────────
# Execute search
# ─────────────────────────────────────────────────────────────────────────────

if query.strip():
    with st.spinner("Searching …"):
        try:
            t0 = time.time()
            response = searcher.search(
                query=query.strip(),
                top_k=top_k,
                use_reranker=use_reranker,
            )
            wall_ms = (time.time() - t0) * 1000
        except Exception as exc:
            st.error(f"Search failed: {exc}")
            st.stop()

    # ── "Did you mean?" ────────────────────────────────────────────────
    if response.did_you_mean:
        st.info(
            f'Did you mean: **{response.did_you_mean}**?  '
            f'[Search instead](/?query={response.did_you_mean})'
        )

    # ── Metrics row ────────────────────────────────────────────────────
    m1, m2, m3 = st.columns(3)
    m1.metric("Results returned", len(response.results))
    m2.metric("Search latency", f"{response.latency_ms:.0f} ms")
    m3.metric("Passages indexed", f"{response.total_indexed:,}")

    st.markdown("---")

    # ── Results ────────────────────────────────────────────────────────
    if not response.results:
        st.warning("No results found. Try a different query.")
    else:
        for result in response.results:
            with st.container():
                # Header line
                header_parts = [f"**#{result.rank}**"]
                if show_scores:
                    header_parts.append(f"Score: `{result.score:.4f}`")
                if show_passage_id:
                    header_parts.append(f"ID: `{result.passage_id}`")
                header_parts.append(f"Source: *{result.source}*")
                st.markdown("  ·  ".join(header_parts))

                # Passage text
                if show_full_passage:
                    st.markdown(result.passage)
                else:
                    # Show highlighted snippet (markdown bold)
                    snippet_text = result.snippet or result.passage[:300] + "…"
                    st.markdown(snippet_text)

                # Relevance bar
                if show_scores:
                    bar_val = max(0.0, min(1.0, float(result.score)))
                    st.progress(bar_val)

                st.markdown("---")
