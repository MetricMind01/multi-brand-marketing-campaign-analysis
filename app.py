"""
Streamlit dashboard for multi-brand campaign revenue and profit/loss prediction.
Run:
    streamlit run app.py
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "outputs" / "models"
REPORT_DIR = ROOT / "outputs" / "reports"

st.set_page_config(
    page_title="Marketing Campaign Performance Predictor",
    page_icon="📊",
    layout="wide"
)

@st.cache_resource
def load_models():
    reg_path = MODEL_DIR / "best_regression_pipeline.joblib"
    cls_path = MODEL_DIR / "best_classification_pipeline.joblib"
    meta_path = MODEL_DIR / "model_metadata.joblib"
    if not (reg_path.exists() and cls_path.exists() and meta_path.exists()):
        return None
    return (
        joblib.load(reg_path),
        joblib.load(cls_path),
        joblib.load(meta_path)
    )

st.title("📊 Multi-Brand Marketing Campaign Performance Predictor")
st.write(
    "Analyse campaign inputs for Nykaa, Purplle and Tira. "
    "The application predicts expected revenue and campaign profitability."
)

loaded = load_models()
if loaded is None:
    st.error("Model files are missing. Run 01_data_cleaning_eda.py and 02_model_training.py first.")
    st.stop()

reg_model, cls_model, metadata = loaded

with st.sidebar:
    st.header("Model information")
    st.write(f"**Revenue model:** {metadata['best_regression_model']}")
    st.write(f"**Profit/Loss model:** {metadata['best_classification_model']}")
    st.caption("Metrics shown in the reports folder are measured on held-out test data.")

st.subheader("1. Campaign details")
c1, c2, c3 = st.columns(3)

with c1:
    brand = st.selectbox("Brand", ["Nykaa", "Purplle", "Tira"])
    campaign_type = st.selectbox(
        "Campaign type", ["Social Media", "Paid Ads", "Influencer", "Email", "SEO"]
    )
    target_audience = st.selectbox(
        "Target audience",
        ["College Students", "Working Women", "Youth", "Premium Shoppers", "Tier 2 City Customers"]
    )
    language = st.selectbox("Language", ["Hindi", "English", "Tamil", "Bengali"])

with c2:
    customer_segment = st.selectbox(
        "Customer segment",
        ["College Students", "Working Women", "Youth", "Premium Shoppers", "Tier 2 City Customers"]
    )
    duration = st.number_input("Duration (days)", min_value=1, max_value=365, value=15)
    impressions = st.number_input("Impressions", min_value=1, value=50000, step=1000)
    clicks = st.number_input("Clicks", min_value=0, value=3000, step=100)

with c3:
    leads = st.number_input("Leads", min_value=0, value=1000, step=50)
    conversions = st.number_input("Conversions", min_value=0, value=500, step=10)
    acquisition_cost = st.number_input(
        "Acquisition cost per conversion (₹)", min_value=0.0, value=200.0, step=10.0
    )
    engagement_score = st.slider("Engagement score", 0.0, 30.0, 12.0)

st.subheader("2. Marketing channels")
channel_options = ["WhatsApp", "YouTube", "Google", "Facebook", "Instagram", "Email"]
selected_channels = st.multiselect(
    "Select all channels used",
    channel_options,
    default=["Instagram", "YouTube"]
)

if clicks > impressions:
    st.warning("Clicks exceed impressions. Please check these campaign inputs.")
if conversions > clicks:
    st.warning("Conversions exceed clicks. Please check these campaign inputs.")

if st.button("Predict campaign performance", type="primary", use_container_width=True):
    if impressions <= 0:
        st.error("Impressions must be greater than zero.")
        st.stop()

    total_spend = conversions * acquisition_cost
    ctr = clicks / impressions
    conversion_rate = conversions / clicks if clicks > 0 else 0.0
    cost_per_conversion = total_spend / conversions if conversions > 0 else 0.0

    # Use exactly the columns and feature engineering used during training.
    meta_features = joblib.load(MODEL_DIR / "feature_metadata.joblib")
    row = {
        "Brand": brand,
        "Campaign_Type": campaign_type,
        "Target_Audience": target_audience,
        "Language": language,
        "Customer_Segment": customer_segment,
        "Duration": duration,
        "Impressions": impressions,
        "Clicks": clicks,
        "Leads": leads,
        "Conversions": conversions,
        "Acquisition_Cost": acquisition_cost,
        "Engagement_Score": engagement_score,
        "CTR": ctr,
        "Conversion_Rate": conversion_rate,
        "Cost_per_Conversion": cost_per_conversion,
        "Campaign_Month": 0,
        "Campaign_DayOfWeek": 0
    }
    for channel_col in meta_features["channel_features"]:
        channel_name = channel_col.replace("Channel_", "", 1)
        row[channel_col] = int(channel_name in selected_channels)

    input_df = pd.DataFrame([row])
    # Enforce the exact training feature order.
    input_df = input_df[meta_features["all_features"]]

    revenue_prediction = float(reg_model.predict(input_df)[0])
    class_prediction = int(cls_model.predict(input_df)[0])
    profit_probability = None
    if hasattr(cls_model, "predict_proba"):
        probabilities = cls_model.predict_proba(input_df)[0]
        classes = list(cls_model.classes_)
        if 1 in classes:
            profit_probability = float(probabilities[classes.index(1)])

    projected_roi = (
        (revenue_prediction - total_spend) / total_spend
        if total_spend > 0 else None
    )

    st.divider()
    st.subheader("3. Prediction results")
    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted revenue", f"₹{revenue_prediction:,.0f}")
    m2.metric("Estimated campaign spend", f"₹{total_spend:,.0f}")
    m3.metric("Predicted outcome", "PROFIT" if class_prediction == 1 else "LOSS")

    if projected_roi is not None:
        st.metric("Revenue-based projected ROI", f"{projected_roi:.2%}")
    if profit_probability is not None:
        st.write(f"Model-estimated probability of profit: **{profit_probability:.1%}**")

    st.caption(
        "Important: the classification model predicts the Profit_Flag separately "
        "from the revenue model. Its result can differ from the ROI calculated "
        "using predicted revenue and estimated spend."
    )

    chart = pd.DataFrame({
        "Metric": ["Estimated campaign spend", "Predicted revenue"],
        "Amount (₹)": [total_spend, revenue_prediction]
    }).set_index("Metric")
    st.bar_chart(chart)

    if class_prediction == 1:
        st.success("The classification model predicts a profitable campaign.")
    else:
        st.warning("The classification model predicts a loss-making campaign. Review spend, conversion rate and channel mix.")

st.divider()
st.subheader("4. Model evaluation")
st.write(
    f"Regression R² on held-out test data: "
    f"{metadata['regression_test_r2']:.3f}"
)
st.write(
    f"Classification accuracy on held-out test data: "
    f"{metadata['classification_test_accuracy']:.3f}"
)
st.caption(
    "GUVI benchmark targets: R² ≥ 0.95 and classification accuracy ≥ 0.95. "
    "These are evaluation targets, not guaranteed outcomes."
)

st.caption("Project: Multi-Brand Marketing Campaign Performance Analysis & Prediction")
