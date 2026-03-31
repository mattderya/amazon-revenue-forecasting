import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings
import re
import os

warnings.filterwarnings("ignore")

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

from prophet import Prophet
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ─── Page Config ───
st.set_page_config(
    page_title="📈 Revenue & Profitability Forecast",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    .stApp { font-family: 'Inter', sans-serif; }
    .block-container { padding-top: 0.8rem !important; max-width: 1200px; }

    .main-header {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4c1d95 100%);
        border-radius: 16px; padding: 2rem 2.5rem;
        margin-bottom: 1.5rem; text-align: center;
        box-shadow: 0 4px 24px rgba(139,92,246,0.25);
    }
    .main-header h1 { font-size: 2rem; font-weight: 700; color: #f5f3ff; margin: 0; letter-spacing: -0.5px; }
    .main-header p  { color: #c4b5fd; font-size: 0.95rem; margin: 0.4rem 0 0 0; }

    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding: 10px 20px; border-radius: 8px 8px 0 0; font-weight: 600; }

    section[data-testid="stSidebar"] { background: #0f0a1e; border-right: 1px solid #2e1065; }
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 { color: #a78bfa !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>📈 Time Series — Revenue &amp; Profitability Forecast</h1>
    <p>Weekly revenue and profit forecasting using SARIMA, Prophet, XGBoost and LSTM on Amazon e-commerce data</p>
</div>
""", unsafe_allow_html=True)


# ─── Helper Functions ───
def to_num(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.strip()
    s = s.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "NULL": pd.NA, "null": pd.NA})
    s = s.str.replace(r"[^\d,\.\-]", "", regex=True)
    has_dot   = s.str.contains(r"\.", na=False)
    has_comma = s.str.contains(",", na=False)
    eu = has_dot & has_comma
    s.loc[eu]         = s.loc[eu].str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    only_comma = has_comma & ~has_dot
    s.loc[only_comma] = s.loc[only_comma].str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def create_features(df, target="revenue"):
    df  = df.copy()
    idx = df.index
    df["week_of_year"] = idx.isocalendar().week.astype(int)
    df["month"]        = idx.month
    df["quarter"]      = idx.quarter
    df["year"]         = idx.year
    df["day_of_year"]  = idx.dayofyear
    for lag in [1, 2, 3, 4, 8, 12, 13, 26, 52]:
        if lag < len(df):
            df[f"lag_{lag}"] = df[target].shift(lag)
    for window in [4, 8, 13, 26]:
        if window < len(df):
            s = df[target].shift(1)
            df[f"rolling_mean_{window}"] = s.rolling(window).mean()
            df[f"rolling_std_{window}"]  = s.rolling(window).std()
            df[f"rolling_min_{window}"]  = s.rolling(window).min()
            df[f"rolling_max_{window}"]  = s.rolling(window).max()
    df["expanding_mean"] = df[target].shift(1).expanding().mean()
    df["trend"]          = np.arange(len(df))
    df["sin_week"]       = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["cos_week"]       = np.cos(2 * np.pi * df["week_of_year"] / 52)
    df["sin_month"]      = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"]      = np.cos(2 * np.pi * df["month"] / 12)
    return df


def eval_metrics(y_true, y_pred, name="model"):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return {"Model": name, "MAE": round(mae, 2), "RMSE": round(rmse, 2), "MAPE (%)": round(mape * 100, 2)}


def generate_synthetic_data(seed=42):
    """Generate realistic synthetic Amazon weekly time-series data (104 weeks)."""
    np.random.seed(seed)
    dates   = pd.date_range("2022-01-01", periods=104, freq="W")
    trend   = np.linspace(4000, 8500, 104)
    season  = 1800 * np.sin(np.arange(104) * 2 * np.pi / 52)
    holiday = np.array([900 if d.month in [11, 12] else 0 for d in dates])
    noise   = np.random.randn(104) * 350
    revenue = np.maximum(trend + season + holiday + noise, 500)
    profit  = revenue * np.random.uniform(0.18, 0.32, 104)
    orders  = (revenue / np.random.uniform(28, 55, 104)).astype(int)
    return pd.DataFrame({"date": dates, "revenue": revenue, "orders": orders, "profit": profit}).set_index("date")


# ─── Sidebar ───
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg", width=150)
    st.markdown("---")
    st.header("⚙️ Settings")

    st.subheader("📂 Data Source")
    data_source = st.radio(
        "Data loading method:",
        ["🎲 Synthetic Demo Data", "📁 Auto Load (Project CSVs)", "📤 Upload CSV Files"]
    )

    st.markdown("---")
    st.subheader("🎯 Forecast Target")
    target_choice = st.selectbox("Target variable:", ["Revenue", "Profit", "Both"])

    st.subheader("📅 Forecast Horizon")
    forecast_weeks = st.slider("Weeks to forecast:", 4, 26, 13)

    st.markdown("---")
    st.markdown(
        '<p style="color:#4a3f6b;font-size:0.72rem;text-align:center;">'
        'Time Series · SARIMA · Prophet · XGBoost · LSTM'
        '</p>', unsafe_allow_html=True,
    )

# Fixed model params (all models always run)
use_sarima = use_prophet = use_xgboost = use_lstm = True
sarima_p, sarima_d, sarima_q = 1, 1, 1
xgb_estimators, xgb_depth, xgb_lr = 500, 6, 0.05
lstm_epochs, lstm_lookback = 100, 12

# ─── Data Loading ───
df = None
ts = None
APP_DIR = os.path.dirname(os.path.abspath(__file__))

if data_source == "🎲 Synthetic Demo Data":
    ts = generate_synthetic_data()
    df = "demo"
    st.success("🎲 Synthetic demo data loaded — 104 weeks (2022–2024)")

elif data_source == "📁 Auto Load (Project CSVs)":
    csv1 = os.path.join(APP_DIR, "amazon_orders_2023_time_series.csv")
    csv2 = os.path.join(APP_DIR, "df_time_series.csv")
    if os.path.exists(csv1) and os.path.exists(csv2):
        df1, df2 = pd.read_csv(csv1, low_memory=False), pd.read_csv(csv2, low_memory=False)
        cols = ["amazon-order-id","purchase-date","order-status","fulfillment-channel",
                "sales-channel","ship-service-level","product-name","sku","asin",
                "quantity","item-price","shipping-price","ship-city","ship-state",
                "ship-country","is-business-order"]
        avail = [c for c in cols if c in df1.columns and c in df2.columns]
        df    = pd.concat([df1[avail], df2[avail]], ignore_index=True)
        st.success(f"✅ Project CSVs loaded! ({len(df1):,} + {len(df2):,} = {len(df):,} rows)")
    else:
        st.warning("⚠️ CSV files not found — falling back to synthetic demo data.")
        ts = generate_synthetic_data()
        df = "demo"

elif data_source == "📤 Upload CSV Files":
    c1, c2 = st.columns(2)
    with c1: file1 = st.file_uploader("amazon_orders_2023_time_series.csv", type="csv", key="f1")
    with c2: file2 = st.file_uploader("df_time_series.csv", type="csv", key="f2")
    if file1 and file2:
        df1, df2 = pd.read_csv(file1), pd.read_csv(file2)
        cols  = ["amazon-order-id","purchase-date","order-status","fulfillment-channel",
                 "sales-channel","ship-service-level","product-name","sku","asin",
                 "quantity","item-price","shipping-price","ship-city","ship-state",
                 "ship-country","is-business-order"]
        avail = [c for c in cols if c in df1.columns and c in df2.columns]
        df    = pd.concat([df1[avail], df2[avail]], ignore_index=True)
    else:
        st.info("👆 Upload both CSVs, or switch to Synthetic Demo Data in the sidebar.")
        st.stop()

# ─── Process Real CSV Data ───
if df is not None and not isinstance(df, str):
    with st.spinner("🔄 Cleaning and processing data..."):
        df["cost"]       = df["sku"].apply(lambda x: float(re.search(r"(\d+\.\d+)", str(x)).group(1)) if re.search(r"(\d+\.\d+)", str(x)) else None)
        df["total_cost"] = df["cost"] * df["quantity"]
        df["item_price_num"] = to_num(df["item-price"])
        df["total_cost_num"] = to_num(df["total_cost"])

        shipped    = df["order-status"].eq("Shipped")
        mask_ratio = shipped & df["item_price_num"].notna() & df["total_cost_num"].notna() & (df["total_cost_num"] != 0)
        if mask_ratio.sum() > 0:
            ratio = (df.loc[mask_ratio, "item_price_num"] / df.loc[mask_ratio, "total_cost_num"]).mean()
            if pd.notna(ratio) and np.isfinite(ratio) and ratio != 0:
                mask_fill = (df["total_cost_num"].isna() | (df["total_cost_num"] == 0)) & df["item_price_num"].notna()
                df.loc[mask_fill, "total_cost_num"] = df.loc[mask_fill, "item_price_num"] / ratio
        df["total_cost"]    = df["total_cost_num"]
        df["purchase-date"] = pd.to_datetime(df["purchase-date"], format="mixed", utc=True).dt.tz_localize(None)
        df = df[df["order-status"] == "Shipped"].copy()
        df = df[df["item-price"] > 0].copy()
        df = df[df["item-price"] <= df["item-price"].quantile(0.995)].copy()
        df["revenue"] = df["item-price"]
        df["date"]    = pd.to_datetime(df["purchase-date"].dt.date)

        daily = df.groupby("date").agg(
            total_revenue=("revenue", "sum"),
            total_cost_deneme=("total_cost", "sum"),
            total_orders=("amazon-order-id", "nunique"),
            total_units=("quantity", "sum"),
            unique_products=("asin", "nunique"),
            b2b_orders=("is-business-order", "sum"),
        ).reset_index()
        dr    = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
        daily = daily.set_index("date").reindex(dr).fillna(0).reset_index().rename(columns={"index": "date"})
        daily["est_profit"] = daily["total_revenue"] - daily["total_cost_deneme"]

        weekly = daily.groupby(pd.Grouper(key="date", freq="W")).agg(
            revenue=("total_revenue", "sum"),
            orders=("total_orders", "sum"),
            profit=("est_profit", "sum")
        ).reset_index()
        weekly = weekly[weekly["revenue"] > 0].reset_index(drop=True)
        ts = weekly[["date", "revenue", "orders", "profit"]].set_index("date")


# ─── Main Tabs ───
if df is not None and ts is not None:
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Explorer", "🤖 Model Training", "📈 Forecast Results",
        "🔮 Future Forecast", "📋 Summary Report"
    ])

    # ═══ TAB 1 — Data Explorer ═══
    with tab1:
        st.subheader("📊 Data Overview")
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("📅 Total Weeks",   len(ts))
        with c2: st.metric("💰 Total Revenue", f"${ts['revenue'].sum():,.0f}")
        with c3: st.metric("📦 Total Orders",  f"{ts['orders'].sum():,.0f}")
        with c4: st.metric("💎 Total Profit",  f"${ts['profit'].sum():,.0f}")
        st.markdown("---")

        fig_rev = go.Figure()
        fig_rev.add_trace(go.Scatter(x=ts.index, y=ts["revenue"], mode="lines+markers",
            name="Weekly Revenue", line=dict(color="#3b82f6", width=2), marker=dict(size=4)))
        fig_rev.add_trace(go.Scatter(x=ts.index, y=ts["revenue"].rolling(4).mean(),
            mode="lines", name="4-Week Moving Avg.", line=dict(color="#f59e0b", width=2, dash="dash")))
        fig_rev.update_layout(title="📈 Weekly Revenue Trend", xaxis_title="Date",
            yaxis_title="Revenue ($)", template="plotly_white", height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig_rev, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig_p = go.Figure()
            fig_p.add_trace(go.Bar(x=ts.index, y=ts["profit"],
                marker_color=np.where(ts["profit"] > 0, "#10b981", "#ef4444"), name="Weekly Profit"))
            fig_p.update_layout(title="💎 Weekly Profit", template="plotly_white", height=350)
            st.plotly_chart(fig_p, use_container_width=True)
        with col2:
            fig_o = go.Figure()
            fig_o.add_trace(go.Scatter(x=ts.index, y=ts["orders"], fill="tozeroy",
                fillcolor="rgba(99,102,241,0.2)", line=dict(color="#6366f1", width=2), name="Order Count"))
            fig_o.update_layout(title="📦 Weekly Order Count", template="plotly_white", height=350)
            st.plotly_chart(fig_o, use_container_width=True)

        with st.expander("📋 Raw Data Table"):
            st.dataframe(ts.reset_index(), use_container_width=True)

    # ═══ TAB 2 — Model Training ═══
    with tab2:
        st.subheader("🤖 Model Training & Evaluation")
        test_horizon = 13
        train_size   = len(ts) - test_horizon
        train        = ts.iloc[:train_size]
        test         = ts.iloc[train_size:]
        st.info(f"📌 Training: {len(train)} weeks | Test: {len(test)} weeks | Period: {test.index[0].strftime('%Y-%m-%d')} → {test.index[-1].strftime('%Y-%m-%d')}")

        targets = (["revenue"] if target_choice == "Revenue"
                   else ["profit"] if target_choice == "Profit"
                   else ["revenue", "profit"])

        if st.button("🚀 Train All Models", type="primary", use_container_width=True):
            for target in targets:
                lbl     = "Revenue" if target == "revenue" else "Profit"
                st.markdown(f"### 🎯 {lbl} Forecasting Models")
                results, predictions, trained_models = [], {}, {}
                progress    = st.progress(0)
                status      = st.empty()
                model_count = 4
                step        = 0

                # SARIMA
                status.text("⏳ Training SARIMA...")
                try:
                    sarima_fit  = SARIMAX(train[target], order=(sarima_p, sarima_d, sarima_q),
                        seasonal_order=(1,1,1,52), enforce_stationarity=False, enforce_invertibility=False
                    ).fit(disp=False, maxiter=200)
                    sarima_pred = sarima_fit.forecast(steps=len(test))
                    sarima_pred.index = test.index
                    predictions["SARIMA"]        = sarima_pred.values
                    trained_models["sarima_fit"] = sarima_fit
                    results.append(eval_metrics(test[target].values, sarima_pred.values, "SARIMA"))
                except Exception as e:
                    st.warning(f"SARIMA error: {e}")
                step += 1; progress.progress(step / model_count)

                # Prophet
                status.text("⏳ Training Prophet...")
                try:
                    m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                                changepoint_prior_scale=0.05, seasonality_mode="multiplicative")
                    m.fit(train.reset_index().rename(columns={"date": "ds", target: "y"}))
                    future         = m.make_future_dataframe(periods=len(test), freq="W")
                    prophet_pred   = m.predict(future).iloc[-len(test):]["yhat"].values
                    predictions["Prophet"]         = prophet_pred
                    trained_models["prophet_model"] = m
                    results.append(eval_metrics(test[target].values, prophet_pred, "Prophet"))
                except Exception as e:
                    st.warning(f"Prophet error: {e}")
                step += 1; progress.progress(step / model_count)

                # XGBoost
                status.text("⏳ Training XGBoost...")
                try:
                    ts_feat  = create_features(ts, target).dropna()
                    features = [c for c in ts_feat.columns if c not in ["revenue", "orders", "profit"]]
                    tr_idx   = ts_feat.index < test.index[0]
                    model_xgb = xgb.XGBRegressor(n_estimators=xgb_estimators, max_depth=xgb_depth,
                        learning_rate=xgb_lr, subsample=0.8, colsample_bytree=0.8,
                        reg_alpha=0.1, reg_lambda=1.0, random_state=42)
                    model_xgb.fit(ts_feat.loc[tr_idx, features], ts_feat.loc[tr_idx, target],
                        eval_set=[(ts_feat.loc[~tr_idx, features], ts_feat.loc[~tr_idx, target])], verbose=False)
                    xgb_pred = model_xgb.predict(ts_feat.loc[~tr_idx, features])
                    predictions["XGBoost"]         = xgb_pred
                    trained_models["xgb_model"]    = model_xgb
                    trained_models["xgb_features"] = features
                    results.append(eval_metrics(ts_feat.loc[~tr_idx, target].values, xgb_pred, "XGBoost"))
                except Exception as e:
                    st.warning(f"XGBoost error: {e}")
                step += 1; progress.progress(step / model_count)

                # LSTM
                status.text("⏳ Training LSTM...")
                try:
                    scaler = MinMaxScaler()
                    scaled = scaler.fit_transform(ts[[target]].values)
                    def make_seqs(data, lb):
                        X, y = [], []
                        for i in range(lb, len(data)):
                            X.append(data[i-lb:i, 0]); y.append(data[i, 0])
                        return np.array(X), np.array(y)
                    X_l, y_l = make_seqs(scaled, lstm_lookback)
                    X_l      = X_l.reshape(X_l.shape[0], X_l.shape[1], 1)
                    split    = len(X_l) - len(test)
                    mdl      = Sequential([LSTM(64, return_sequences=True, input_shape=(lstm_lookback,1)),
                                           Dropout(0.2), LSTM(32), Dropout(0.2), Dense(16, activation="relu"), Dense(1)])
                    mdl.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss="mse")
                    mdl.fit(X_l[:split], y_l[:split], epochs=lstm_epochs, batch_size=8,
                            validation_data=(X_l[split:], y_l[split:]),
                            callbacks=[EarlyStopping(patience=15, restore_best_weights=True)], verbose=0)
                    lstm_pred = scaler.inverse_transform(mdl.predict(X_l[split:], verbose=0)).flatten()
                    predictions["LSTM"]            = lstm_pred
                    trained_models["lstm_model"]   = mdl
                    trained_models["lstm_scaler"]  = scaler
                    results.append(eval_metrics(test[target].values, lstm_pred, "LSTM"))
                except Exception as e:
                    st.warning(f"LSTM error: {e}")
                step += 1; progress.progress(step / model_count)

                status.text("✅ All models trained!")
                progress.progress(1.0)
                st.session_state.update({
                    f"results_{target}": results, f"predictions_{target}": predictions,
                    f"trained_models_{target}": trained_models, f"train_{target}": train,
                    f"test_{target}": test, "ts": ts
                })
                if results:
                    df_r = pd.DataFrame(results).sort_values("MAPE (%)")
                    st.dataframe(df_r, use_container_width=True, hide_index=True)
                    best = df_r.iloc[0]
                    st.success(f"🏆 Best model: **{best['Model']}** (MAPE: {best['MAPE (%)']}%)")

    # ═══ TAB 3 — Forecast Results ═══
    with tab3:
        st.subheader("📈 Model Forecast Comparison")
        for target in ["revenue", "profit"]:
            lbl = "Revenue" if target == "revenue" else "Profit"
            if f"predictions_{target}" in st.session_state:
                preds      = st.session_state[f"predictions_{target}"]
                test_data  = st.session_state[f"test_{target}"]
                train_data = st.session_state[f"train_{target}"]
                ensemble   = np.mean(list(preds.values()), axis=0)
                st.markdown(f"### 🎯 {lbl} Forecast Charts")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=train_data.index, y=train_data[target], mode="lines",
                    name="Training Data", line=dict(color="#94a3b8", width=1.5)))
                fig.add_trace(go.Scatter(x=test_data.index, y=test_data[target], mode="lines+markers",
                    name="Actual (Test)", line=dict(color="#1f2937", width=3), marker=dict(size=6)))
                fig.add_trace(go.Scatter(x=test_data.index, y=ensemble, mode="lines+markers",
                    name="Ensemble Forecast", line=dict(color="#ef4444", width=2.5, dash="dot"), marker=dict(size=6)))
                fig.update_layout(title=f"📊 {lbl} — Ensemble vs Actual",
                    xaxis_title="Date", yaxis_title=f"{lbl} ($)", template="plotly_white", height=500,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02))
                st.plotly_chart(fig, use_container_width=True)
                fig_e = go.Figure()
                fig_e.add_trace(go.Histogram(x=test_data[target].values - ensemble, marker_color="#ef4444"))
                fig_e.update_layout(title="📉 Ensemble Error Distribution", template="plotly_white", height=350)
                st.plotly_chart(fig_e, use_container_width=True)
            else:
                st.warning("⚠️ Please train the models first in the 'Model Training' tab.")

    # ═══ TAB 4 — Future Forecast ═══
    with tab4:
        st.subheader(f"🔮 {forecast_weeks}-Week Future Forecast")
        for target in ["revenue", "profit"]:
            lbl = "Revenue" if target == "revenue" else "Profit"
            if f"trained_models_{target}" in st.session_state:
                trained = st.session_state[f"trained_models_{target}"]
                ts_data = st.session_state["ts"]
                st.markdown(f"### 🎯 {lbl} — Future Forecast")
                with st.spinner("🔮 Computing future forecast..."):
                    forecasts = {}
                    if "xgb_model" in trained:
                        xgb_fc  = []
                        temp_ts = ts_data[[target]].copy()
                        for _ in range(forecast_weeks):
                            nd = temp_ts.index[-1] + pd.Timedelta(weeks=1)
                            temp_ts.loc[nd] = np.nan
                            feats    = create_features(temp_ts, target).iloc[[-1]]
                            fc       = [c for c in feats.columns if c not in ["revenue","orders","profit"]]
                            feats[fc] = feats[fc].ffill().fillna(0)
                            pred = trained["xgb_model"].predict(feats[fc])[0]
                            temp_ts.loc[nd, target] = pred
                            xgb_fc.append(pred)
                        forecasts["XGBoost"] = xgb_fc
                    if "lstm_model" in trained:
                        sc       = trained["lstm_scaler"]
                        last_seq = sc.transform(ts_data[[target]].values)[-lstm_lookback:]
                        lstm_fc  = []
                        for _ in range(forecast_weeks):
                            ps    = trained["lstm_model"].predict(last_seq.reshape(1, lstm_lookback, 1), verbose=0)[0][0]
                            lstm_fc.append(sc.inverse_transform([[ps]])[0][0])
                            last_seq = np.append(last_seq[1:], [[ps]], axis=0)
                        forecasts["LSTM"] = lstm_fc
                    if len(forecasts) > 1:
                        forecasts["Ensemble"] = [np.mean(v) for v in zip(*forecasts.values())]
                    elif forecasts:
                        forecasts["Ensemble"] = list(forecasts.values())[0]

                    fc_dates = pd.date_range(ts_data.index[-1] + pd.Timedelta(weeks=1), periods=forecast_weeks, freq="W")
                    fig_fc = go.Figure()
                    fig_fc.add_trace(go.Scatter(x=ts_data.index, y=ts_data[target], mode="lines",
                        name="Historical Data", line=dict(color="#3b82f6", width=2)))
                    if "Ensemble" in forecasts:
                        fig_fc.add_trace(go.Scatter(
                            x=[ts_data.index[-1]] + list(fc_dates),
                            y=[ts_data[target].iloc[-1]] + list(forecasts["Ensemble"]),
                            mode="lines+markers", name="Ensemble Forecast",
                            line=dict(color="#ef4444", width=2.5, dash="dash"), marker=dict(size=6)))
                    fig_fc.update_layout(title=f"🔮 {lbl} — History & {forecast_weeks}-Week Forecast",
                        xaxis_title="Date", yaxis_title=f"{lbl} ($)", template="plotly_white", height=500)
                    st.plotly_chart(fig_fc, use_container_width=True)

                    if "Ensemble" in forecasts:
                        ens = forecasts["Ensemble"]
                        st.dataframe(pd.DataFrame({
                            "Week": fc_dates.strftime("%Y-%m-%d"),
                            "Ensemble Forecast": [f"${v:,.0f}" for v in ens]
                        }), use_container_width=True, hide_index=True)
                        c1, c2, c3 = st.columns(3)
                        with c1: st.metric(f"💰 Total {lbl} Forecast", f"${sum(ens):,.0f}")
                        with c2: st.metric("📊 Weekly Average", f"${np.mean(ens):,.0f}")
                        with c3:
                            lq     = ts_data[target].iloc[-13:].sum()
                            change = ((sum(ens) - lq) / lq * 100) if lq != 0 else 0
                            st.metric("📈 vs Previous Quarter", f"{change:.1f}%", delta=f"{change:.1f}%")
            else:
                st.warning("⚠️ Please train the models first in the 'Model Training' tab.")

    # ═══ TAB 5 — Summary Report ═══
    with tab5:
        st.subheader("📋 Project Summary Report")
        st.markdown("""
        ### 🎯 Future Revenue & Profitability Forecasting with Time Series

        **Objective:** Generate weekly revenue and profitability forecasts from Amazon e-commerce data
        to provide data-driven decision support for business strategy.

        ---

        #### 📌 Models Used
        | Model | Description | Advantage |
        |-------|-------------|-----------|
        | **SARIMA** | Seasonal ARIMA | Classic statistical approach |
        | **Prophet** | Facebook's time series model | Automatically captures seasonality & holidays |
        | **XGBoost** | Gradient boosting | Strong predictions with feature engineering |
        | **LSTM** | Deep learning (RNN) | Learns long-term dependencies |

        ---

        #### 📊 Data Pipeline
        1. **Data Loading** → 2 CSVs merged (or synthetic data generated)
        2. **Cleaning** → Cost extraction from SKU, outlier removal
        3. **Aggregation** → Daily → Weekly time series
        4. **Feature Engineering** → Lag, rolling, cyclical features
        5. **Modeling** → 4 model training runs
        6. **Ensemble** → XGBoost + LSTM weighted average
        7. **Forecast** → 13-week (quarterly) future projection
        """)

        for target in ["revenue", "profit"]:
            lbl = "Revenue" if target == "revenue" else "Profit"
            if f"results_{target}" in st.session_state and st.session_state[f"results_{target}"]:
                try:
                    st.markdown(f"#### 🏆 {lbl} Model Performance")
                    st.dataframe(pd.DataFrame(st.session_state[f"results_{target}"]).sort_values("MAPE (%)"),
                                 use_container_width=True, hide_index=True)
                except Exception:
                    st.info(f"⏳ {lbl} models not trained yet.")

        st.markdown("---")
        st.caption("🔧 Deployment: Streamlit  |  Models: SARIMA, Prophet, XGBoost, LSTM")
