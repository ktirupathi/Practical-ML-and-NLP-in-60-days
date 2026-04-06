# Day 26: Streamlit Dashboards for ML

## Learning Objectives

- Build interactive ML dashboards using Streamlit's component library
- Organize layouts with columns, tabs, sidebars, and expanders
- Use `st.cache_data` and `st.cache_resource` to avoid redundant computation
- Display model predictions, feature importance plots, and confusion matrices interactively
- Deploy a Streamlit app to Streamlit Community Cloud

## Key Concepts

### Rapid Prototyping with Streamlit

Streamlit turns Python scripts into web applications with no frontend code. You write
top-to-bottom Python, sprinkle in `st.slider()`, `st.selectbox()`, and `st.plotly_chart()`,
and Streamlit reruns the script on every interaction to keep the UI in sync. This reactive
model is perfect for ML demos --- data scientists can share a live dashboard with stakeholders
in minutes instead of days. The trade-off is that Streamlit is not designed for multi-page
production apps with authentication; it shines as a prototyping and communication tool.

### Caching for Performance

Because Streamlit reruns your entire script on every widget change, expensive operations like
loading a model or reading a large CSV would become bottlenecks without caching.
`st.cache_data` caches the return value of a function based on its arguments (great for
DataFrames and API calls), while `st.cache_resource` caches objects that should not be
serialized, such as database connections and loaded ML models. Proper caching makes the
difference between a sluggish demo and a snappy one.

### Layout and Visualization

Streamlit provides `st.columns` for side-by-side layouts, `st.tabs` for tabbed sections,
`st.sidebar` for controls, and `st.expander` for collapsible detail sections. It integrates
natively with Matplotlib, Plotly, Altair, and even custom HTML components, so you can
visualize anything from confusion matrices to SHAP waterfall plots.

## Practical Example

```python
# dashboard.py --- run with: streamlit run dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

st.set_page_config(page_title="Iris Explorer", layout="wide")
st.title("Iris Classification Dashboard")

# --- Cache the model so it is trained only once ---
@st.cache_resource
def load_model():
    X, y = load_iris(return_X_y=True, as_frame=True)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    return clf, X, y

clf, X, y = load_model()
feature_names = X.columns.tolist()
target_names = load_iris().target_names

# --- Sidebar controls ---
st.sidebar.header("Model Parameters")
n_trees = st.sidebar.slider("Number of trees", 10, 300, 100, step=10)
max_depth = st.sidebar.selectbox("Max depth", [None, 3, 5, 10, 20])

if st.sidebar.button("Retrain"):
    clf = RandomForestClassifier(
        n_estimators=n_trees, max_depth=max_depth, random_state=42
    )
    clf.fit(X, y)
    st.sidebar.success("Model retrained!")

# --- Main layout: two columns ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Feature Importance")
    importance = pd.Series(clf.feature_importances_, index=feature_names)
    st.bar_chart(importance.sort_values(ascending=True))

with col2:
    st.subheader("Confusion Matrix")
    preds = clf.predict(X)
    cm = confusion_matrix(y, preds)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(cm, display_labels=target_names).plot(ax=ax)
    st.pyplot(fig)

# --- Interactive prediction ---
st.subheader("Try a Prediction")
input_cols = st.columns(4)
values = []
for i, name in enumerate(feature_names):
    val = input_cols[i].number_input(name, value=float(X[name].mean()))
    values.append(val)

pred = clf.predict([values])[0]
proba = clf.predict_proba([values])[0]
st.metric("Predicted Class", target_names[pred], f"{proba[pred]*100:.1f}% confidence")
```

## Resources

- [Streamlit documentation](https://docs.streamlit.io/)
- [Streamlit caching guide](https://docs.streamlit.io/develop/concepts/architecture/caching)
- [Streamlit Community Cloud deployment](https://streamlit.io/cloud)

## Up Next

**Day 27 -- Docker for ML:** Containerize your ML application so it runs identically on any machine.
