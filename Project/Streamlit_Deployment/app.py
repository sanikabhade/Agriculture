import os
import gzip
import streamlit as st
import pandas as pd
import plotly.express as px

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="AgriCast | Maharashtra",
    page_icon="🌾",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

HISTORICAL_PATH = os.path.join(
    BASE_DIR,
    "Maharashtra_Agri_Master_2024_Jul2025.csv.gz"
)

FORECAST_PATH = os.path.join(
    BASE_DIR,
    "Maharashtra_Crop_Market_12_Month_Forecast.csv"
)

# =====================================================
# DATA LOADING
# =====================================================

@st.cache_data
def load_historical_data():
    df = pd.read_csv(HISTORICAL_PATH, compression="gzip")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Modal Price"] = pd.to_numeric(
        df["Modal Price"], errors="coerce"
    )

    df = df[
        (df["Price Unit"] == "Rs./Quintal") &
        (df["Modal Price"] > 0)
    ].copy()

    return df


@st.cache_data
def load_forecast_data():
    df = pd.read_csv(FORECAST_PATH)
    df["Forecast_Date"] = pd.to_datetime(
        df["Forecast_Date"], errors="coerce"
    )
    df["Predicted_Modal_Price"] = pd.to_numeric(
        df["Predicted_Modal_Price"], errors="coerce"
    )
    return df


try:
    historical = load_historical_data()
    forecasts = load_forecast_data()
except Exception as e:
    st.error(f"Unable to load project data: {e}")
    st.stop()

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("🌾 AgriCast")
st.sidebar.caption("Maharashtra Agricultural Price Forecasting")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Price Forecasting",
        "Historical Analysis",
        "Market Comparison",
        "Model Information"
    ]
)

st.sidebar.divider()
st.sidebar.caption("Project: Crop and Market Modal Price Forecasting")
st.sidebar.caption("Forecast unit: Rs./Quintal")

# =====================================================
# DASHBOARD
# =====================================================

if page == "Dashboard":

    st.title("🌾 Maharashtra Crop Market Dashboard")
    st.write(
        "Explore historical agricultural market prices and "
        "LSTM-based future modal price projections."
    )

    total_records = len(historical)
    commodities = historical["Commodity"].nunique()
    markets = historical["Market"].nunique()
    districts = historical["District"].nunique()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Historical Records", f"{total_records:,}")
    c2.metric("Commodities", commodities)
    c3.metric("Markets", markets)
    c4.metric("Districts", districts)

    st.divider()

    st.subheader("Dataset Overview")

    left, right = st.columns(2)

    with left:
        monthly = (
            historical
            .assign(Month=historical["Date"].dt.to_period("M").astype(str))
            .groupby("Month", as_index=False)["Modal Price"]
            .mean()
        )

        fig = px.line(
            monthly,
            x="Month",
            y="Modal Price",
            markers=True,
            title="Average Historical Modal Price"
        )
        fig.update_layout(xaxis_title="Month", yaxis_title="Price (Rs./Quintal)")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        top_commodities = (
            historical.groupby("Commodity")["Modal Price"]
            .mean()
            .nlargest(10)
            .reset_index()
        )

        fig = px.bar(
            top_commodities,
            x="Modal Price",
            y="Commodity",
            orientation="h",
            title="Top 10 Commodities by Average Modal Price"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Forecast Dataset")

    f1, f2, f3 = st.columns(3)
    f1.metric("Forecast Records", f"{len(forecasts):,}")
    f2.metric("Commodity-Market Series", forecasts.groupby(
        ["Commodity", "Market"]
    ).ngroups)
    f3.metric(
        "Forecast Period",
        f"{forecasts['Forecast_Date'].min():%b %Y} – "
        f"{forecasts['Forecast_Date'].max():%b %Y}"
    )

# =====================================================
# PRICE FORECASTING
# =====================================================

elif page == "Price Forecasting":

    st.title("📈 Crop Price Forecasting")
    st.write(
        "Explore the precomputed monthly LSTM forecast "
        "for a selected commodity and market."
    )

    col1, col2, col3 = st.columns(3)

    commodity_list = sorted(forecasts["Commodity"].dropna().unique())

    with col1:
        selected_commodity = st.selectbox(
            "Select Commodity",
            commodity_list
        )

    available_markets = sorted(
        forecasts.loc[
            forecasts["Commodity"] == selected_commodity,
            "Market"
        ].dropna().unique()
    )

    with col2:
        selected_market = st.selectbox(
            "Select Market",
            available_markets
        )

    with col3:
        duration = st.selectbox(
            "Forecast Duration",
            list(range(1, 13)),
            index=11,
            format_func=lambda x: f"{x} month(s)"
        )

    result = forecasts[
        (forecasts["Commodity"] == selected_commodity) &
        (forecasts["Market"] == selected_market)
    ].sort_values("Forecast_Date").head(duration).copy()

    if result.empty:
        st.warning("No forecast available for this selection.")
    else:
        result["Forecast Month"] = result["Forecast_Date"].dt.strftime("%b %Y")

        minimum = result.loc[result["Predicted_Modal_Price"].idxmin()]
        maximum = result.loc[result["Predicted_Modal_Price"].idxmax()]

        a, b, c = st.columns(3)
        a.metric(
            "Average Forecast Price",
            f"₹{result['Predicted_Modal_Price'].mean():,.2f}"
        )
        b.metric(
            "Highest Forecast",
            f"₹{maximum['Predicted_Modal_Price']:,.2f}",
            maximum["Forecast Month"]
        )
        c.metric(
            "Lowest Forecast",
            f"₹{minimum['Predicted_Modal_Price']:,.2f}",
            minimum["Forecast Month"]
        )

        fig = px.line(
            result,
            x="Forecast_Date",
            y="Predicted_Modal_Price",
            markers=True,
            title=f"{selected_commodity} Price Forecast — {selected_market}"
        )
        fig.update_layout(
            xaxis_title="Forecast Month",
            yaxis_title="Predicted Modal Price (Rs./Quintal)"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Monthly Forecast Values")

        display = result[
            ["Forecast_Date", "Predicted_Modal_Price", "Price_Unit"]
        ].copy()

        display.columns = [
            "Forecast Date",
            "Predicted Modal Price",
            "Price Unit"
        ]

        display["Forecast Date"] = display["Forecast Date"].dt.strftime("%d %b %Y")
        display["Predicted Modal Price"] = display[
            "Predicted Modal Price"
        ].round(2)

        st.dataframe(display, use_container_width=True, hide_index=True)

        csv = result.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download Forecast CSV",
            data=csv,
            file_name="crop_price_forecast.csv",
            mime="text/csv"
        )

        st.info(
            "These are precomputed LSTM forecasts generated by the "
            "project's recursive monthly forecasting pipeline. "
            "They are projections, not guaranteed market prices."
        )

# =====================================================
# HISTORICAL ANALYSIS
# =====================================================

elif page == "Historical Analysis":

    st.title("📊 Historical Price Analysis")

    commodity_list = sorted(historical["Commodity"].dropna().unique())

    selected_commodity = st.selectbox(
        "Select Commodity",
        commodity_list
    )

    data = historical[
        historical["Commodity"] == selected_commodity
    ].copy()

    market_list = ["All Markets"] + sorted(data["Market"].dropna().unique())

    selected_market = st.selectbox(
        "Select Market",
        market_list
    )

    if selected_market != "All Markets":
        data = data[data["Market"] == selected_market]

    data = data.sort_values("Date")

    if data.empty:
        st.warning("No historical records found.")
    else:
        c1, c2, c3 = st.columns(3)

        c1.metric("Records", f"{len(data):,}")
        c2.metric("Average Price", f"₹{data['Modal Price'].mean():,.2f}")
        c3.metric("Maximum Price", f"₹{data['Modal Price'].max():,.2f}")

        fig = px.line(
            data,
            x="Date",
            y="Modal Price",
            title=f"Historical Modal Price — {selected_commodity}",
            markers=True
        )
        fig.update_layout(yaxis_title="Modal Price (Rs./Quintal)")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Historical Records")

        st.dataframe(
            data[
                ["Date", "District", "Market", "Commodity",
                 "Arrival Quantity", "Modal Price", "Price Unit"]
            ].sort_values("Date", ascending=False),
            use_container_width=True,
            hide_index=True
        )

# =====================================================
# MARKET COMPARISON
# =====================================================

elif page == "Market Comparison":

    st.title("🏪 Market Price Comparison")

    selected_commodity = st.selectbox(
        "Select Commodity",
        sorted(historical["Commodity"].dropna().unique())
    )

    data = historical[
        historical["Commodity"] == selected_commodity
    ].copy()

    market_summary = (
        data.groupby(["Market", "District"], as_index=False)
        .agg(
            Average_Price=("Modal Price", "mean"),
            Minimum_Price=("Modal Price", "min"),
            Maximum_Price=("Modal Price", "max"),
            Records=("Modal Price", "count")
        )
        .sort_values("Average_Price", ascending=False)
    )

    if market_summary.empty:
        st.warning("No market data available.")
    else:
        fig = px.bar(
            market_summary.head(15),
            x="Average_Price",
            y="Market",
            color="District",
            orientation="h",
            title=f"Top 15 Markets — {selected_commodity}"
        )
        fig.update_layout(
            xaxis_title="Average Modal Price (Rs./Quintal)"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Market-wise Price Summary")

        st.dataframe(
            market_summary.round(2),
            use_container_width=True,
            hide_index=True
        )

# =====================================================
# MODEL INFORMATION
# =====================================================

elif page == "Model Information":

    st.title("🧠 LSTM Model Information")

    st.subheader("Model Overview")

    st.markdown("""
    This project uses a Long Short-Term Memory (LSTM) neural
    network to model sequential patterns in agricultural
    commodity market prices.

    **Prediction target:** Modal Price

    **Price unit:** Rs./Quintal

    **Lookback window:** 14 previous observations

    **Series identification:** Commodity and Market

    **Architecture:**
    - Numerical input features
    - Categorical embedding layers
    - LSTM layer with 64 units
    - Dense layer with 32 neurons
    - Dropout regularization
    - Single price prediction output
    """)

    st.subheader("Input Features")

    st.write("""
    **Numerical:** Modal Price, Arrival Quantity, weather,
    soil characteristics, crop duration and engineered date features.

    **Categorical:** Commodity, Market, District and
    Typical Sowing Season.
    """)

    st.subheader("Historical Evaluation")

    st.write("Previously recorded evaluation metrics:")

    m1, m2, m3 = st.columns(3)
    m1.metric("R² Score", "0.8175")
    m2.metric("MAE", "₹531.13")
    m3.metric("RMSE", "₹869.27")

    st.caption(
        "These metrics refer to the filtered LSTM evaluation. "
        "They are historical test results and do not guarantee "
        "the accuracy of the 12-month projections."
    )

    st.subheader("Forecasting Methodology")

    st.markdown("""
    1. Historical observations are grouped by commodity and market.
    2. The model uses the previous 14 observations.
    3. Numerical features are scaled using saved preprocessing.
    4. Categorical features are represented using saved mappings.
    5. Monthly forecasts are generated recursively using the
       project's monthly forecasting assumptions.
    """)

    st.warning(
        "The 12-month projections use assumed future input features "
        "based on historical monthly values. Actual future weather, "
        "arrivals and market conditions may differ. The forecasts "
        "should be treated as experimental decision-support estimates."
    )

# =====================================================
# FOOTER
# =====================================================

st.sidebar.divider()
st.sidebar.caption("AgriCast | Academic Project")
st.sidebar.caption("Maharashtra, India")
