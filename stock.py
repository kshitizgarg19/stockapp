import streamlit as st
from datetime import date, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from plotly import graph_objs as go

# Constants
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Function to get the next business day
def next_business_day(date):
    next_day = date + timedelta(days=1)
    while next_day.weekday() >= 5:  # Skip Saturday (5) and Sunday (6)
        next_day += timedelta(days=1)
    return next_day

# HTML/CSS for centered and bold title with dark theme
title_style = """
<style>
.title {
    text-align: center;
    font-weight: bold;
    font-size: 2.5em;
    margin-bottom: 1em;
}
body {
    background-color: #2d3436;
    color: white;
}
</style>
"""

# Render the title
st.markdown(title_style, unsafe_allow_html=True)
st.markdown("<div class='title'>Stock Price Prediction</div>", unsafe_allow_html=True)

# Summary section
st.subheader("About Stock Price Prediction")
st.write("""
Stock Price Prediction empowers investors by providing precise predictions of stock prices 
through advanced machine learning algorithms. Our platform leverages state-of-the-art 
techniques to analyze historical data, forecast trends, and assist users in making 
informed investment decisions.
""")

# Market selection
markets = ['US', 'India']
market = st.radio('Select market:', markets)

# Stock options for each market
if market == 'US':
    options = {
        'Apple (AAPL)': 'AAPL',
        'Microsoft (MSFT)': 'MSFT',
        'Amazon (AMZN)': 'AMZN',
        'Google (GOOGL)': 'GOOGL',
        'Tesla (TSLA)': 'TSLA',
        'S&P 500 ETF (SPY)': 'SPY'
    }
    currency = '$'
else:
    options = {
        'Reliance Industries (RELIANCE.NS)': 'RELIANCE.NS',
        'Tata Consultancy Services (TCS.NS)': 'TCS.NS',
        'HDFC Bank (HDFCBANK.NS)': 'HDFCBANK.NS',
        'ICICI Bank (ICICIBANK.NS)': 'ICICIBANK.NS',
        'Hindustan Unilever (HINDUNILVR.NS)': 'HINDUNILVR.NS',
        'ITC Ltd (ITC.NS)': 'ITC.NS',
        'State Bank of India (SBIN.NS)': 'SBIN.NS',
        'Axis Bank (AXISBANK.NS)': 'AXISBANK.NS',
        'Nifty 50 (NIFTY.NS)': '^NSEI',
        'Sensex (SENSEX.BSE)': '^BSESN'
    }
    currency = '₹'

# Stock selection
selected_stock = st.selectbox('Select dataset for prediction', list(options.keys()))
ticker = options[selected_stock]

# Optional custom ticker input
custom_ticker = st.text_input('Or enter a custom ticker (optional):')
if custom_ticker:
    ticker = custom_ticker

# Function to load data
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

# Load stock data
data_load_state = st.text('Loading data...')
data = load_data(ticker)
data_load_state.text('Loading data... done!')

# Display raw data
st.subheader('Raw data')
st.write(data.tail())

# Function to plot raw data
def plot_raw_data(data):
    if not data.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], mode='lines', name="Open Price", line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name="Close Price", line=dict(color='blue')))
        fig.update_layout(
            title="Time Series Data with Rangeslider",
            xaxis_title="Date",
            yaxis_title=f"Price ({currency})",
            xaxis_rangeslider_visible=True,
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("No data to display for the selected stock.")

# Check if data is valid and plot
if len(data) == 0:
    st.error('No data available for the selected stock. Please choose another stock or check your custom ticker.')
else:
    plot_raw_data(data)

    # Prepare data for training
    df_train = data[['Date', 'Close']]
    df_train['Date'] = pd.to_datetime(df_train['Date'])
    df_train['Date_ordinal'] = df_train['Date'].map(pd.Timestamp.toordinal)

    X = df_train[['Date_ordinal']]
    y = df_train['Close']

    # Scaling the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit the model
    model = GradientBoostingRegressor()
    model.fit(X_scaled, y)

    # Predict next business day's close price
    next_day = next_business_day(pd.to_datetime(TODAY))
    next_day_ordinal = np.array([[next_day.toordinal()]])
    next_day_scaled = scaler.transform(next_day_ordinal)
    next_day_prediction = model.predict(next_day_scaled)[0]

    # Forecast DataFrame
    forecast = pd.DataFrame({'Date': [next_day], 'Predicted Close': [next_day_prediction]})

    # Display forecast data
    st.subheader('Forecast data')
    st.write(forecast)

    # Plot forecast
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df_train['Date'], y=df_train['Close'], mode='lines', name="Actual Close", line=dict(color='green')))
    fig1.add_trace(go.Scatter(x=forecast['Date'], y=forecast['Predicted Close'], mode='lines+markers', name="Predicted Close", line=dict(color='royalblue')))
    fig1.update_layout(
        title="Forecast Plot",
        xaxis_title="Date",
        yaxis_title=f"Price ({currency})",
        template="plotly_dark"
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Components plot
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df_train['Date'], y=df_train['Close'], mode='markers', name='Actual Close', marker=dict(color='purple')))
    fig2.add_trace(go.Scatter(x=forecast['Date'], y=forecast['Predicted Close'], mode='lines+markers', name='Forecast', line=dict(color='orange')))
    fig2.update_layout(
        title="Forecast Components",
        xaxis_title="Date",
        yaxis_title=f"Price ({currency})",
        template="plotly_dark"
    )
    st.plotly_chart(fig2, use_container_width=True)
    # Display the predicted price
    st.subheader('Predicted Price')
    st.markdown(f"<div style='text-align: center; font-size: 24px;'>Predicted Close Price on {next_day.strftime('%A, %d-%m-%Y')}: {currency}{next_day_prediction:.2f}</div>", unsafe_allow_html=True)

# Footer with developer attribution
st.markdown(
    """
    <footer style="padding: 20px; background-color: #333; color: #fff; text-align: center;">
        <p style="font-size: 20px; font-weight: bold;">Developed and Maintained by Kshitiz Garg</p>
        <p>Connect with me:</p>
        <p>
            <a href="https://github.com/kshitizgarg19" target="_blank" style="color: #00bcd4;">GitHub</a> |
            <a href="https://www.linkedin.com/in/kshitiz-garg-898403207/" target="_blank" style="color: #00bcd4;">LinkedIn</a> |
            <a href="mailto:kshitizgarg19@gmail.com" style="color: #00bcd4;">Email</a>
        </p>
    </footer>
    """,
    unsafe_allow_html=True
)
