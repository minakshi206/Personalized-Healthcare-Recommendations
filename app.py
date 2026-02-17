import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# -----------------------------------
# Page Setup
# -----------------------------------
st.set_page_config(
    page_title="Healthcare Analytics Dashboard",
    page_icon="🩺",
    layout="wide"
)

# -----------------------------------
# Custom CSS
# -----------------------------------
st.markdown("""
<style>
    .main {
        background-color: #f6f8fb;
    }

    .metric-card {
        background: linear-gradient(135deg, #ff4b6e, #ff758c);
        padding: 20px;
        border-radius: 14px;
        color: white;
        box-shadow: 0px 6px 18px rgba(0,0,0,0.15);
    }

    .metric-title {
        font-size: 14px;
        opacity: 0.9;
    }

    .metric-value {
        font-size: 28px;
        font-weight: bold;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------
# Sidebar
# -----------------------------------
st.sidebar.title("🩸 Blood Bank Panel")
menu = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Donor Analysis", "Prediction System"]
)

# -----------------------------------
# Load Data
# -----------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("blood.csv")

df = load_data()

# -----------------------------------
# Train Model
# -----------------------------------
@st.cache_resource
def train_model(dataframe):
    X = dataframe.drop("Class", axis=1)
    y = dataframe["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=300,
            max_depth=7,
            random_state=42
        ))
    ])

    pipeline.fit(X_train, y_train)
    return pipeline

model = train_model(df)

# ===================================
# DASHBOARD PAGE
# ===================================
if menu == "Dashboard":

    st.title("🩺 Healthcare Analytics Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Total Donors</div>
                <div class="metric-value">{len(df)}</div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Avg Recency</div>
                <div class="metric-value">{round(df['Recency'].mean(), 2)}</div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Avg Frequency</div>
                <div class="metric-value">{round(df['Frequency'].mean(), 2)}</div>
            </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Total Blood Donated</div>
                <div class="metric-value">{int(df['Monetary'].sum())}</div>
            </div>
        """, unsafe_allow_html=True)

    st.write("### 📊 Dataset Snapshot")
    st.dataframe(df.head())

# ===================================
# DONOR ANALYSIS PAGE
# ===================================
elif menu == "Donor Analysis":

    st.title("📊 Donor Behaviour Analysis")

    class_counts = df["Class"].value_counts()
    st.bar_chart(class_counts)

    st.write("Feature Trends")
    st.line_chart(df[["Recency", "Frequency", "Monetary", "Time"]])

# ===================================
# PREDICTION PAGE
# ===================================
elif menu == "Prediction System":

    st.title("🧠 Donor Prediction System")

    col1, col2 = st.columns(2)

    with col1:
        recency = st.slider("Recency", 0, 50, 2)
        frequency = st.slider("Frequency", 0, 50, 10)

    with col2:
        monetary = st.slider("Monetary", 0, 10000, 2000)
        time = st.slider("Time", 0, 100, 20)

    if st.button("🔍 Analyze Donor"):

        input_data = pd.DataFrame({
            "Recency": [recency],
            "Frequency": [frequency],
            "Monetary": [monetary],
            "Time": [time]
        })

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        confidence = max(probability) * 100

        if prediction == 1:
            st.success(f"✅ Likely to Donate Blood (Confidence: {confidence:.2f}%)")
        else:
            st.error(f"❌ Unlikely to Donate Blood (Confidence: {confidence:.2f}%)")

st.sidebar.caption("ML Model: Random Forest Classifier")
