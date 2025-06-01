import streamlit as st
import pandas as pd
from prophet import Prophet
from neuralprophet import NeuralProphet, set_log_level
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error # Kept for now, but MAE/RMSE display is removed
import plotly.express as px
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose

# ---- Page Config ----
st.set_page_config(page_title="Forecast Dashboard", layout="wide", initial_sidebar_state="expanded")

# Simplified CSS for a sleek, dark theme
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .reportview-container .main .block-container {
        padding-top: 2rem;
        padding-right: 3rem;
        padding-left: 3rem;
        padding-bottom: 2rem;
        max-width: 1400px;
        margin: auto;
    }
    .stApp {
        background-color: #1a1a2e;
        color: #e0e0e0;
    }
    .css-1d391kg { /* Sidebar */
        background-color: #0f0f1d;
        color: #e0e0e0;
        padding-top: 2rem;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #00e676;
        font-weight: 600;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .stMarkdown p {
        color: #c0c0c0;
    }
    .stButton>button {
        background: linear-gradient(45deg, #00e676, #00c853);
        color: #1a1a2e;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-size: 1rem;
        font-weight: 700;
        border: none;
        box-shadow: 0px 3px 10px rgba(0, 230, 118, 0.3);
        transition: all 0.3s ease;
        cursor: pointer;
    }
    .stButton>button:hover {
        background: linear-gradient(45deg, #00c853, #00e676);
        transform: translateY(-2px);
        box-shadow: 0px 5px 12px rgba(0, 230, 118, 0.5);
    }
    .stMetric {
        background-color: #2c2c4a;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0px 3px 8px rgba(0,0,0,0.2);
        text-align: center;
    }
    .stMetric > div > div:first-child { /* Metric label */
        color: #a0a0a0;
        font-size: 0.9rem;
    }
    .stMetric > div > div:last-child { /* Metric value */
        color: #00e676;
        font-size: 1.25rem;
        font-weight: 700;
    }
    .stSuccess {
        background-color: #1a3a2e;
        border-left: 5px solid #00e676;
        padding: 0.75rem;
        border-radius: 8px;
        color: #d0ffd0;
    }
    .stError {
        background-color: #3a1a1a;
        border-left: 5px solid #ff1744;
        padding: 0.75rem;
        border-radius: 8px;
        color: #ffd0d0;
    }
    .stInfo {
        background-color: #1a2e3a;
        border-left: 5px solid #2196F3;
        padding: 0.75rem;
        border-radius: 8px;
        color: #d0e6ff;
    }
    .stFileUploader label, .stSelectbox label, .stNumberInput label, .stDateInput label {
        color: #00e676;
    }
    .stTextInput>div>div>input, .stDateInput input {
        background-color: #0f0f1d;
        color: #e0e0e0;
        border: 1px solid #333333;
        border-radius: 8px;
        padding: 0.5rem;
    }
    .streamlit-expanderHeader {
        background-color: #2c2c4a;
        color: #00e676 !important;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        margin-bottom: 0.5rem;
    }
    .streamlit-expanderContent {
        background-color: #1a1a2e;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #2c2c4a;
    }
    .stPlotlyChart {
        background-color: #1a1a2e;
        border-radius: 10px;
        padding: 0.5rem;
    }
    .dataframe {
        background-color: #1a1a2e;
        color: #e0e0e0;
        border-radius: 10px;
    }
    .dataframe th {
        background-color: #00e676;
        color: #1a1a2e;
    }
    .dataframe td {
        border-top: 1px solid #333333;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---- Sidebar ----
with st.sidebar:
    st.markdown("## 📁 Upload CSV")
    uploaded_file = st.file_uploader("Upload a file with 'Date', 'Volume', and 'ImpactTag' columns", type="csv")

    st.markdown("---")
    st.markdown("### ⚙️ Forecast Settings")
    periods_to_forecast = st.number_input(
        "Number of future periods to forecast (e.g., days)",
        min_value=1,
        value=30,
        step=1,
        help="Enter the number of future periods (e.g., days) you want to forecast beyond your data's end date."
    )

    selected_model = st.selectbox(
        "Select Forecasting Model",
        ("Prophet", "NeuralProphet"),
        help="Select the forecasting model to use."
    )

    st.markdown("---")
    st.markdown("### 👤 Client Dashboard")
    st.markdown("Powered by Streamlit, Plotly, Prophet & NeuralProphet")

# ---- Main Content ----
st.title("📈 Forecasting Dashboard")
st.markdown("""
Welcome to the Forecasting Dashboard! This interactive tool allows you to upload your time series data
(CSV file with 'Date', 'Volume', and 'ImpactTag' columns) and generate future forecasts using
powerful models like Prophet and NeuralProphet.

**How to use this dashboard:**
1.  **Upload your CSV file** using the sidebar on the left. Ensure your file has the required columns.
2.  **Adjust Forecast Settings** in the sidebar, such as the number of periods to forecast and the model.
3.  **Run the Forecast** using the button in the main panel. The model will use all your historical data to predict future values.
4.  Explore the results, including interactive plots and downloadable forecast data.

The dashboard provides several sections to help you understand your data and the forecast:
* **Data Quality Checks:** Quickly assess the completeness and integrity of your uploaded data.
* **Exploratory Data Analysis (EDA):** Visualize trends, distributions, and seasonal patterns in your historical data.
* **Advanced Time Series Analysis:** Decompose your time series and examine rolling statistics.
* **Forecast Results:** View the interactive forecast plot and a table of predicted values.
""")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    dfh = df.copy()

    st.markdown("### 📊 Data Quality Checks")
    st.markdown("This section provides a quick overview of your data's quality, highlighting total rows, missing values, and duplicate entries. It's crucial to ensure data quality for reliable forecasting.")
    required_columns = ['Date', 'Volume', 'ImpactTag']
    missing_columns = [col for col in required_columns if col not in df.columns]

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Total Rows", len(df))
    with col2: st.metric("Missing Dates", df['Date'].isnull().sum())
    with col3: st.metric("Missing Volume", df['Volume'].isnull().sum())
    with col4: st.metric("Duplicate Dates", df.duplicated(subset='Date').sum())

    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        st.stop()

    try:
        df['Date'] = pd.to_datetime(df['Date'])
        dfh['Date'] = pd.to_datetime(dfh['Date'])
    except Exception as e:
        st.error(f"Date conversion failed: {e}")
        st.stop()

    df.sort_values('Date', inplace=True)
    dfh.sort_values('Date', inplace=True)
    st.success("Data quality checks passed.")

    st.subheader("🔍 Exploratory Data Analysis")
    st.markdown("""
    In this section, we explore various aspects of your historical data. These visualizations help in understanding underlying patterns, trends, seasonality, and the impact of special events before forecasting.
    * **Volume Trend Overview:** Shows the overall trend of your 'Volume' data over time.
    * **Holiday Detection:** Highlights data points tagged as holidays, showing their potential impact on volume.
    * **Volume Descriptive Statistics & Overall Box Plot:** Provides key statistical measures (table) and a box plot summarizing the overall distribution (median, quartiles, outliers) of the 'Volume' column.
    * **Volume Distribution (Histogram with Mean/Median):** A histogram showing the frequency distribution of volume values, with overlaid lines indicating the mean and median.
    * **Monthly Average Volume:** Displays the average volume for each month, helping to identify monthly seasonality.
    * **Average Volume by Impact Tag:** Shows how different tags in 'ImpactTag' correlate with average volume.
    """)

    col1_eda, col2_eda = st.columns([2, 1])
    with col1_eda:
        st.markdown("#### Volume Trend Overview")
        fig_volume = px.line(df, x='Date', y='Volume', title="Volume over Time", template="plotly_dark")
        fig_volume.update_layout(height=400, autosize=True)
        st.plotly_chart(fig_volume, use_container_width=True)

    with col2_eda:
        st.markdown("#### Holiday Detection")
        holidays_df_viz = df[df['ImpactTag'].str.contains('Holiday', na=False, case=False)]
        if not holidays_df_viz.empty:
            st.success(f"Holidays found: {len(holidays_df_viz)}")
            fig_holiday = px.scatter(holidays_df_viz, x='Date', y='Volume', color='ImpactTag', title="Holidays Detected", template="plotly_dark")
            fig_holiday.update_layout(height=400, autosize=True)
            st.plotly_chart(fig_holiday, use_container_width=True)
            st.dataframe(holidays_df_viz[['Date', 'Volume', 'ImpactTag']].head())
        else:
            st.info("No holidays marked in data.")

    col1_desc_stats, col2_dist = st.columns(2)
    with col1_desc_stats:
        st.markdown("#### Volume Descriptive Statistics")
        st.dataframe(df['Volume'].describe())

        st.markdown("#### Overall Volume Distribution (Box Plot)")
        if not df['Volume'].dropna().empty:
            fig_overall_boxplot = px.box(df, y='Volume', title="Overall Volume Distribution",
                                         template="plotly_dark", points="outliers")
            fig_overall_boxplot.update_layout(height=300, autosize=True)
            st.plotly_chart(fig_overall_boxplot, use_container_width=True)
        else:
            st.warning("Not enough volume data to display overall box plot.")

    with col2_dist:
        st.markdown("#### Volume Distribution (Histogram with Mean/Median)")
        volume_data = df['Volume'].dropna()
        if not volume_data.empty:
            fig_hist = px.histogram(volume_data, x='Volume', nbins=50,
                                    title='Volume Distribution with Mean & Median',
                                    template="plotly_dark", marginal="rug")

            mean_volume = volume_data.mean()
            median_volume = volume_data.median()

            fig_hist.add_vline(x=mean_volume, line_width=2, line_dash="dash", line_color="orange",
                              annotation_text=f"Mean: {mean_volume:.2f}",
                              annotation_position="top right",
                              annotation_font_size=10,
                              annotation_font_color="orange")
            fig_hist.add_vline(x=median_volume, line_width=2, line_dash="dash", line_color="yellow",
                              annotation_text=f"Median: {median_volume:.2f}",
                              annotation_position="bottom right",
                              annotation_font_size=10,
                              annotation_font_color="yellow")

            fig_hist.update_layout(height=400, autosize=True)
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.warning("Not enough volume data to display distribution histogram.")


    col1_monthly, col2_impact = st.columns(2)
    with col1_monthly:
        st.markdown("#### Monthly Average Volume")
        df['Month'] = df['Date'].dt.month_name()
        monthly_avg = df.groupby('Month')['Volume'].mean().reset_index()
        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
        monthly_avg['Month'] = pd.Categorical(monthly_avg['Month'], categories=month_order, ordered=True)
        monthly_avg = monthly_avg.sort_values('Month')
        fig_monthly = px.bar(monthly_avg, x='Month', y='Volume', title='Monthly Average Volume', template="plotly_dark")
        fig_monthly.update_layout(height=400, autosize=True)
        st.plotly_chart(fig_monthly, use_container_width=True)
        df.drop(columns=['Month'], inplace=True)
    with col2_impact:
        st.markdown("#### Average Volume by Impact Tag")
        tag_avg = df.groupby('ImpactTag')['Volume'].mean().reset_index()
        fig_tag = px.bar(tag_avg, x='ImpactTag', y='Volume', title='Average Volume by Impact Tag', template="plotly_dark")
        fig_tag.update_layout(height=400, autosize=True)
        st.plotly_chart(fig_tag, use_container_width=True)

    st.markdown("---")
    st.subheader("Advanced Time Series Analysis")
    st.markdown("""
    This section delves deeper into the characteristics of your time series data.
    * **Time Series Decomposition:** Breaks down the time series into its constituent components: Trend, Seasonality, and Residuals. This helps in understanding the underlying structure of the data. We assume an additive model and a yearly period (365 days) for decomposition.
    * **Box Plot of Volume by Year:** Visualizes the distribution of volume for each year, useful for spotting yearly variations or outliers.
    * **Rolling Mean and Standard Deviation:** Shows the moving average and standard deviation of the volume, which can help in identifying trends and volatility over time. You can adjust the window size for these calculations.
    """)

    col1_decomp, col2_decomp, col3_decomp = st.columns(3)
    with col1_decomp:
        st.markdown("##### Time Series Decomposition (Trend)")
        try:
            if len(df['Volume'].dropna()) >= 365 * 2:
                decomposition = seasonal_decompose(df['Volume'].dropna(), model='additive', period=365)
                trend_data = decomposition.trend.dropna()
                if not trend_data.empty:
                    fig_trend = px.line(x=df['Date'][trend_data.index], y=trend_data, title='Trend Component', template="plotly_dark")
                    fig_trend.update_layout(height=300, autosize=True)
                    st.plotly_chart(fig_trend, use_container_width=True)
                else: st.info("Trend component is empty after decomposition.")
            else:
                st.warning("Not enough data for yearly decomposition (period=365). Min 2 years required.")
        except Exception as e:
            st.warning(f"Trend Decomposition error: {e}")
    with col2_decomp:
        st.markdown("##### Time Series Decomposition (Seasonal)")
        try:
            if 'decomposition' in locals() and len(df['Volume'].dropna()) >= 365 * 2 :
                seasonal_data = decomposition.seasonal.dropna()
                if not seasonal_data.empty:
                    fig_seasonal = px.line(x=df['Date'][seasonal_data.index], y=seasonal_data, title='Seasonal Component', template="plotly_dark")
                    fig_seasonal.update_layout(height=300, autosize=True)
                    st.plotly_chart(fig_seasonal, use_container_width=True)
                else: st.info("Seasonal component is empty after decomposition.")
        except Exception as e:
            st.warning(f"Seasonal Decomposition error: {e}")
    with col3_decomp:
        st.markdown("##### Time Series Decomposition (Residual)")
        try:
            if 'decomposition' in locals() and len(df['Volume'].dropna()) >= 365 * 2:
                residual_data = decomposition.resid.dropna()
                if not residual_data.empty:
                    fig_residual = px.line(x=df['Date'][residual_data.index], y=residual_data, title='Residual Component', template="plotly_dark")
                    fig_residual.update_layout(height=300, autosize=True)
                    st.plotly_chart(fig_residual, use_container_width=True)
                else: st.info("Residual component is empty after decomposition.")
        except Exception as e:
            st.warning(f"Residual Decomposition error: {e}")

    col1_adv, col2_adv = st.columns(2)
    with col1_adv:
        st.markdown("#### Box Plot of Volume by Year")
        df['Year'] = df['Date'].dt.year
        if not df['Volume'].dropna().empty:
            fig_boxplot_year = px.box(df, x='Year', y='Volume', title='Volume Distribution by Year', template="plotly_dark")
            fig_boxplot_year.update_layout(height=400, autosize=True)
            st.plotly_chart(fig_boxplot_year, use_container_width=True)
        else:
            st.warning("Not enough volume data to display box plot by year.")
        df.drop(columns=['Year'], inplace=True)
    with col2_adv:
        st.markdown("#### Rolling Mean and Standard Deviation")
        min_slider_val = 1
        max_slider_val = max(min_slider_val, min(90, len(df)-1 if len(df)>1 else min_slider_val))
        default_slider_val = min(30, max_slider_val)
        if max_slider_val > min_slider_val :
            window = st.slider("Rolling Window Size", min_value=min_slider_val, max_value=max_slider_val, value=default_slider_val)
            if len(df) > window and not df['Volume'].dropna().empty :
                df_rolling = df.copy()
                df_rolling['Rolling_Mean'] = df_rolling['Volume'].rolling(window=window).mean()
                df_rolling['Rolling_Std'] = df_rolling['Volume'].rolling(window=window).std()
                fig_rolling = px.line(df_rolling.dropna(subset=['Rolling_Mean', 'Rolling_Std']), 
                                      x='Date', y=['Volume', 'Rolling_Mean', 'Rolling_Std'], 
                                      title=f'Rolling Mean and Std Dev (Window={window})', 
                                      template="plotly_dark")
                fig_rolling.update_layout(height=400, autosize=True)
                st.plotly_chart(fig_rolling, use_container_width=True)
            else:
                st.warning("Not enough data points for the selected rolling window or no volume data.")
        else:
            st.info("Not enough data to display rolling statistics slider.")

    st.markdown("---")
    st.subheader(f"🧠 Forecast with {selected_model}")
    st.markdown(f"The model will be trained on all your historical data to forecast **{periods_to_forecast} future periods** beyond the last date in your file. Click the button below to generate the forecast.")
    
    # Removed split_date configuration UI and related text.

    if st.button(f"🚀 Run Forecast with {selected_model}"):
        with st.spinner(f"Forecasting in progress using {selected_model}... This might take a moment."):
            df_model_input = df[['Date', 'Volume']].rename(columns={'Date': 'ds', 'Volume': 'y'})
            df_model_input['y'] = pd.to_numeric(df_model_input['y'], errors='coerce') 
            df_model_input.dropna(subset=['y'], inplace=True) 

            # df_train is now the full historical dataset
            df_train = df_model_input.copy()
            # df_test is effectively empty as we are forecasting beyond max date
            # df_test = pd.DataFrame(columns=['ds', 'y']) # Not strictly needed for this logic flow

            if df_train.empty:
                st.error("❌ No data available for training after processing. Please check your uploaded file.")
                st.stop()
            
            st.success(f"Using {len(df_train)} data points for training to forecast {periods_to_forecast} future periods.")

            forecast_df = pd.DataFrame()
            fig_forecast = go.Figure() 

            historical_actual_color = 'rgba(33, 150, 243, 0.7)' # Blueish for historical
            forecast_color = '#00e676'                          # Green
            interval_color = 'rgba(0, 230, 118, 0.2)'           # Light green for interval

            # --- Plot Historical Actuals (Common for both models) ---
            fig_forecast.add_trace(go.Scatter(
                x=df_train['ds'], y=df_train['y'], mode='markers',
                name='Historical Actuals', marker=dict(color=historical_actual_color, size=5)
            ))
            
            # --- Model-Specific Forecast and Interval ---
            if selected_model == "Prophet":
                holiday_df_prophet = dfh[['Date', 'ImpactTag']].dropna()
                holiday_df_prophet = holiday_df_prophet[holiday_df_prophet['ImpactTag'].str.contains('Holiday', na=False, case=False)]
                holiday_df_prophet = holiday_df_prophet.rename(columns={'Date': 'ds', 'ImpactTag': 'holiday'})
                if not holiday_df_prophet.empty:
                    holiday_df_prophet['ds'] = pd.to_datetime(holiday_df_prophet['ds'])
                else: holiday_df_prophet = None

                model = Prophet(holidays=holiday_df_prophet, daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
                model.fit(df_train) # Fit on all historical data
                future = model.make_future_dataframe(periods=periods_to_forecast) 
                forecast_df = model.predict(future)

                fig_forecast.add_trace(go.Scatter(
                    x=forecast_df['ds'], y=forecast_df['yhat'], mode='lines', 
                    name='Forecast', line=dict(color=forecast_color)
                ))
                fig_forecast.add_trace(go.Scatter(
                    x=forecast_df['ds'], y=forecast_df['yhat_upper'], mode='lines', 
                    line=dict(width=0), name='Confidence Interval Upper', showlegend=False,
                    fillcolor=interval_color 
                ))
                fig_forecast.add_trace(go.Scatter(
                    x=forecast_df['ds'], y=forecast_df['yhat_lower'], mode='lines', 
                    line=dict(width=0), fill='tonexty', name='Confidence Interval', 
                    showlegend=True, fillcolor=interval_color
                ))
            
            elif selected_model == "NeuralProphet":
                set_log_level("ERROR")
                model_np = NeuralProphet(
                    yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True,
                    quantiles=[0.05, 0.95] 
                )
                events_df_np = dfh[dfh['ImpactTag'].str.contains('Holiday', na=False, case=False)][['Date', 'ImpactTag']].copy().dropna()
                added_event_columns = []
                if not events_df_np.empty:
                    events_df_np.rename(columns={'Date': 'ds', 'ImpactTag': 'event'}, inplace=True)
                    events_df_np['ds'] = pd.to_datetime(events_df_np['ds'])
                    unique_event_tags = events_df_np['event'].unique()
                    for tag in unique_event_tags:
                        model_np.add_events(tag)
                        added_event_columns.append(tag)
                else: events_df_np = None

                df_train_np_fit = df_train.copy() # df_train is all historical data
                if added_event_columns: 
                    for tag in added_event_columns:
                        df_train_np_fit[tag] = 0
                        if events_df_np is not None:
                            event_specific_dates = events_df_np[events_df_np['event'] == tag]['ds']
                            df_train_np_fit.loc[df_train_np_fit['ds'].isin(event_specific_dates), tag] = 1
                
                try:
                    if df_train_np_fit.empty or df_train_np_fit['y'].isnull().all():
                        st.error("Training data for NeuralProphet is empty or 'y' contains all NaNs.")
                        st.stop()
                    model_np.fit(df_train_np_fit, freq="D") # Fit on all historical data
                except Exception as e:
                    st.error(f"Error fitting NeuralProphet: {e}")
                    st.info("Ensure daily frequency and valid numeric 'Volume' (y) for training.")
                    st.stop()
                
                # For make_future_dataframe, df should be the one model was fit on, 
                # including any event columns used during fit.
                future_np_df_spec = df_train_np_fit.copy() 
                # Ensure all event columns are present if they were added to model config, even if no future events are specified
                for tag in added_event_columns: 
                    if tag not in future_np_df_spec.columns: 
                        future_np_df_spec[tag] = 0
                
                future_np = model_np.make_future_dataframe(
                    df=future_np_df_spec, 
                    events_df=events_df_np if added_event_columns else None, 
                    periods=periods_to_forecast, 
                    n_historic_predictions=True # Predict over history as well
                )
                forecast_df = model_np.predict(future_np)

                fig_forecast.add_trace(go.Scatter(
                    x=forecast_df['ds'], y=forecast_df['yhat1'], mode='lines', 
                    name='Forecast', line=dict(color=forecast_color)
                ))
                if 'yhat1 5%' in forecast_df.columns and 'yhat1 95%' in forecast_df.columns:
                    fig_forecast.add_trace(go.Scatter(
                        x=forecast_df['ds'], y=forecast_df['yhat1 95%'], mode='lines', 
                        line=dict(width=0), name='Confidence Interval Upper', showlegend=False,
                        fillcolor=interval_color
                    ))
                    fig_forecast.add_trace(go.Scatter(
                        x=forecast_df['ds'], y=forecast_df['yhat1 5%'], mode='lines', 
                        line=dict(width=0), fill='tonexty', name='Confidence Interval', 
                        showlegend=True, fillcolor=interval_color
                    ))

            fig_forecast.update_layout(
                title_text=f"{selected_model} Forecast",
                height=600, autosize=True, template="plotly_dark",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.success(f"✅ Forecast complete with {selected_model}! Here's the result:")
            st.markdown("#### 📈 Interactive Forecast Plot")
            st.markdown("This plot displays the historical actuals (blue markers), the model's forecast (green line) extending beyond the historical data, and the confidence interval (shaded green area) for the predicted values.")
            if fig_forecast.data: 
                st.plotly_chart(fig_forecast, use_container_width=True)
            else:
                st.warning("Could not generate forecast plot.")

            st.markdown("#### 📊 Forecast Table (Last 10 entries of future forecast)")
            st.markdown("The table below shows the last 10 predicted values from the future forecast period, including the point forecast (`yhat`) and the lower/upper bounds of the confidence interval (`yhat_lower`, `yhat_upper`).")
            
            # Filter forecast_df to show only future dates for the table's "last 10 entries"
            last_historical_date = df_train['ds'].max()
            future_forecast_table_df = forecast_df[forecast_df['ds'] > last_historical_date]

            display_data = pd.DataFrame()
            if selected_model == "Prophet":
                display_data = future_forecast_table_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            elif selected_model == "NeuralProphet":
                cols_for_table = ['ds', 'yhat1']
                rename_for_table = {'yhat1': 'yhat'}
                if 'yhat1 5%' in future_forecast_table_df.columns and 'yhat1 95%' in future_forecast_table_df.columns:
                    cols_for_table.extend(['yhat1 5%', 'yhat1 95%'])
                    rename_for_table['yhat1 5%'] = 'yhat_lower'
                    rename_for_table['yhat1 95%'] = 'yhat_upper'
                display_data = future_forecast_table_df[cols_for_table].rename(columns=rename_for_table)
            
            # Show last 10 of future, or all future if less than 10
            st.dataframe(display_data.tail(min(10, len(display_data))).style.set_properties(**{'background-color': '#2c2c4a', 'color': '#e0e0e0', 'border': '1px solid #3a3a5a'}).set_table_styles([{'selector': 'th', 'props': [('background-color', '#00e676'), ('color', '#1a1a2e')]}]))

            csv_forecast = forecast_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"⬇️ Download Full Forecast as CSV",
                data=csv_forecast,
                file_name=f"{selected_model.lower()}_forecast_results.csv",
                mime="text/csv",
            )

            # Removed Performance Metrics section as there's no test set
            st.info("Performance metrics (MAE, RMSE) are not displayed as the model is trained on all historical data to forecast future periods, without a separate test set.")

else:
    st.info("👆 Please upload a CSV file to get started!")

st.markdown("---")
st.markdown("<div style='text-align: center; color: gray; font-size: 0.9em'>&copy; 2025 Forecast AI. All rights reserved.</div>", unsafe_allow_html=True)