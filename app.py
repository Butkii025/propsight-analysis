import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os

st.set_page_config(page_title="PropSight Valuation Engine", layout="wide")

st.markdown("""
    <style>
    .hero-container {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 30px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 25px;
    }
    .metric-title { font-size: 1.2rem; opacity: 0.9; }
    .metric-value { font-size: 2.8rem; font-weight: bold; margin: 10px 0; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    required = ["ridge_model", "lasso_model", "gbr_model", "model_columns",
                "feature_medians", "feature_importances", "metrics"]
    models_exist = all(os.path.exists(f"models/{m}.pkl") for m in required)
    if not models_exist:
        st.error("⚠️ Administrative Warning: Serialized model files missing from the 'models/' directory.")
        st.stop()

    ridge = joblib.load("models/ridge_model.pkl")
    lasso = joblib.load("models/lasso_model.pkl")
    gbr = joblib.load("models/gbr_model.pkl")
    columns = joblib.load("models/model_columns.pkl")
    medians = joblib.load("models/feature_medians.pkl")
    importances = joblib.load("models/feature_importances.pkl")
    metrics = joblib.load("models/metrics.pkl")

    try:
        data = pd.read_csv("data/train.csv")
    except FileNotFoundError:
        st.error("⚠️ Ingested training file ('data/train.csv') not found.")
        st.stop()

    return ridge, lasso, gbr, columns, medians, importances, metrics, data

ridge_model, lasso_model, gbr_model, model_columns, feature_medians, feature_importances, metrics, train_df = load_assets()

st.markdown(f"""
    <div class="hero-container">
        <h1>PropSight Real Estate Valuation & Analytics Platform</h1>
        <p>End-to-End Automated Valuation Model Powered by Heterogeneous Blended Machine Learning ensemble
        (held-out test R²: {metrics['holdout_r2']*100:.2f}%, MAE: ${metrics['holdout_mae']:,.0f}).</p>
    </div>
    """, unsafe_allow_html=True)

st.sidebar.header("🏢 Property Attributes Configuration")

with st.sidebar.form("valuation_form"):
    st.subheader("📍 Location & Era")
    neighborhood = st.selectbox("Select Neighborhood Sector", sorted(train_df['Neighborhood'].unique()))
    year_built = st.slider("Year Property Was Constructed", int(train_df['YearBuilt'].min()), int(train_df['YearBuilt'].max()), 2000)

    st.subheader("📐 Spatial Dimensions")
    gr_liv_area = st.number_input("Above Ground Living Area (Sq Ft)", min_value=300, max_value=6000, value=1500)
    total_bsmt_sf = st.number_input("Total Basement Area (Sq Ft)", min_value=0, max_value=6000, value=1000)

    st.subheader("⭐ Material Quality & Finish")
    overall_qual = st.slider("Overall Structural Quality (1-10)", 1, 10, 6)
    overall_cond = st.slider("Overall Property Condition (1-10)", 1, 10, 5)

    st.subheader("🚗 Extra Capacities")
    tot_rooms = st.slider("Total Rooms Above Ground (Excluding Bathrooms)", 2, 14, 6)
    garage_cars = st.slider("Garage Car Parking Capacity", 0, 4, 2)

    submit_btn = st.form_submit_button("Calculate Estimated Value ($)")

if submit_btn:
    # --- FIX #1: default every feature the user didn't ask about to its
    # dataset median, not zero. Zero-filling raw-scale features (YrSold,
    # YearRemodAdd, LotArea, GarageYrBlt...) sends the linear models
    # (Ridge/Lasso) wildly out of distribution because their coefficients
    # were fit assuming those columns sit near typical historical values.
    input_data = pd.DataFrame([feature_medians], columns=model_columns)

    input_data['YearBuilt'] = year_built
    input_data['GrLivArea'] = gr_liv_area
    input_data['TotalBsmtSF'] = total_bsmt_sf
    input_data['OverallQual'] = overall_qual
    input_data['OverallCond'] = overall_cond
    input_data['TotRmsAbvGrd'] = tot_rooms
    input_data['GarageCars'] = garage_cars

    # Keep the engineered TotalSF feature consistent with what the user typed
    if 'TotalSF' in input_data.columns:
        input_data['TotalSF'] = total_bsmt_sf + gr_liv_area

    # --- FIX #2: model_columns now actually contains one-hot Neighborhood_*
    # columns (the original deployed model had none, so this selector was a
    # silent no-op). Reset all neighborhood dummies to 0, then set the
    # chosen one to 1.
    neigh_cols = [c for c in model_columns if c.startswith("Neighborhood_")]
    if neigh_cols:
        input_data[neigh_cols] = 0
    neigh_col = f"Neighborhood_{neighborhood}"
    if neigh_col in input_data.columns:
        input_data[neigh_col] = 1

    log_ridge = ridge_model.predict(input_data)[0]
    log_lasso = lasso_model.predict(input_data)[0]
    log_gbr = gbr_model.predict(input_data)[0]

    final_log_pred = (0.30 * log_ridge) + (0.30 * log_lasso) + (0.40 * log_gbr)
    predicted_usd = np.expm1(final_log_pred)

    st.markdown(f"""
        <div style="background-color: #f8f9fa; border-left: 5px solid #1e3c72; padding: 20px; border-radius: 8px; margin-bottom:25px;">
            <p style="margin:0; font-size:1.1rem; color:#555; text-transform: uppercase;">Estimated Market Valuation</p>
            <h2 style="margin:5px 0 0 0; color:#1e3c72; font-size:3rem;">$ {predicted_usd:,.2f}</h2>
            <p style="margin:5px 0 0 0; font-size:0.9rem; color:#888;">*Valuation calculated in USD based on historical US real estate transaction configurations.</p>
        </div>
        """, unsafe_allow_html=True)

st.header("📊 Market Insights & Visual Analytics")
col1, col2 = st.columns(2)

with col1:
    st.subheader("🏡 Budget Feasibility Curve")
    default_neighborhood = neighborhood if submit_btn else sorted(train_df['Neighborhood'].unique())[0]
    bar_data = train_df[train_df['Neighborhood'] == default_neighborhood].groupby(['TotRmsAbvGrd', 'GarageCars'])['SalePrice'].mean().reset_index()
    fig_bar = px.bar(bar_data, x="TotRmsAbvGrd", y="SalePrice", color="GarageCars",
                     title=f"Average Valuation in {default_neighborhood} by Configuration",
                     labels={"TotRmsAbvGrd": "Total Rooms", "SalePrice": "Avg Price ($)"},
                     barmode="group", color_continuous_scale="Viridis")
    st.plotly_chart(fig_bar, use_container_width=True)

with col2:
    st.subheader("🏗️ Market Quality Inventory Spread")
    train_df['QualityClass'] = pd.cut(train_df['OverallQual'], bins=[0,4,7,10], labels=['Budget / Investment', 'Standard Mid-Tier', 'High-End Luxury'])
    donut_data = train_df[train_df['Neighborhood'] == default_neighborhood]['QualityClass'].value_counts().reset_index()
    donut_data.columns = ['QualityClass', 'count']

    fig_pie = px.pie(donut_data, values="count", names="QualityClass", hole=0.4,
                     title=f"Property Structural Quality Distribution: {default_neighborhood}",
                     color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("---")
col3, col4 = st.columns(2)

with col3:
    st.subheader("📈 Historical Price Valuation Trend by Era")
    era_data = train_df.groupby('YearBuilt')['SalePrice'].median().reset_index()
    fig_line = px.line(era_data, x="YearBuilt", y="SalePrice",
                       title="Median Valuation Trend Based on Construction Era",
                       labels={"YearBuilt": "Year Built / Constructed", "SalePrice": "Median Price ($)"},
                       render_mode="svg")
    fig_line.update_traces(line_color="#1e3c72", line_width=2.5)
    st.plotly_chart(fig_line, use_container_width=True)

with col4:
    st.subheader("🎯 Key Drivers of Property Valuation")
    # FIX #3: this used to be a hardcoded fake array. It's now pulled
    # directly from gbr_model.feature_importances_.
    top_n = feature_importances.head(8)
    label_map = {
        "OverallQual": "Overall Material Quality", "TotalSF": "Total Square Footage",
        "GrLivArea": "Above Ground Living Space", "TotalBsmtSF": "Basement Footprint Capacity",
        "GarageCars": "Garage Car Spacing", "TotRmsAbvGrd": "Total Structural Rooms",
        "TotalBath": "Total Bathrooms", "KitchenQual": "Kitchen Quality",
        "YearRemodAdd": "Remodel Recency", "OverallCond": "Overall Condition",
        "LotArea": "Lot Size", "YearBuilt": "Construction Year",
        "CentralAir_Y": "Central Air Conditioning", "FireplaceQu": "Fireplace Quality",
        "BsmtFinSF1": "Finished Basement Area", "GarageArea": "Garage Area",
    }
    features_impact = pd.DataFrame({
        'Structural Feature': [label_map.get(idx, idx) for idx in top_n.index],
        'Relative Model Weight %': (top_n.values * 100).round(2)
    }).sort_values(by='Relative Model Weight %', ascending=True)

    fig_impact = px.bar(features_impact, x="Relative Model Weight %", y="Structural Feature", orientation='h',
                        title="Ensemble Gradient Boosting Feature Signatures (live, from trained model)",
                        labels={"Relative Model Weight %": "Impact Weight Percentage (%)"},
                        color="Relative Model Weight %", color_continuous_scale="Blues")
    st.plotly_chart(fig_impact, use_container_width=True)

st.markdown("---")
f_col1, f_col2 = st.columns([2, 1])

with f_col1:
    st.markdown("### 📝 About This Platform")
    st.markdown(
        "This system predicts property values in **US Dollars ($)** because it is natively trained on "
        "the **Ames Housing Dataset** obtained from the public library **Kaggle**. This dataset contains authentic, "
        "historical real estate pricing and structural specifications directly from the **USA**. Consequently, all spatial features "
        "(square footage), spatial metrics, structural conditions, and geographic sectors are derived explicitly from the United States housing market."
    )

with f_col2:
    st.markdown("### 🔗 Links & Navigation")
    st.markdown("""
        * Developer Profiles : ⚫ [Portfolio](https://p-vijay.vercel.app/)
        * 📁 [GitHub Repository](https://github.com/p-vijay/propsight)
    """)
    pdf_path = "Technical_Documentation.pdf"
    if os.path.exists(pdf_path):
        with open(pdf_path, "rb") as pdf_file:
            pdf_bytes = pdf_file.read()
        st.download_button(
            label="📄 Open System Documentation",
            data=pdf_bytes,
            file_name="Platform_Technical_Documentation.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    else:
        st.markdown("📄 [Project Documentation](#)")

st.markdown(
    "<div style='text-align: center; color: #6b7280; font-size: 0.85rem; margin-top: 2rem; border-top: 1px solid #e5e7eb; padding-top: 10px;'>"
    "© 2026 Real Estate Predictive Analytics Platform | Engineered by ⚫ Priyanshu Vijay. All Rights Reserved."
    "</div>",
    unsafe_allow_html=True
)