
import streamlit as st
from views import dashboard, prediction, model_performance, influential_features

st.set_page_config(page_title="Solar Adoption Predictor", page_icon="☀️", layout="wide")

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.title("☀️ Solar Adoption")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigation", [
    "🏠 Dashboard",
    "🔍 Prediction",
    "📊 Model Performance",
    "🔑 Influential Features"
])

# ── Route to pages ────────────────────────────────────────────────────────────
if page == "🏠 Dashboard":
    dashboard.show()
elif page == "🔍 Prediction":
    prediction.show()
elif page == "📊 Model Performance":
    model_performance.show()
elif page == "🔑 Influential Features":
    influential_features.show()

