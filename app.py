import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# Initialize the Dash app
app = dash.Dash(__name__,suppress_callback_exceptions=True)

# App Layout
app.layout = html.Div([
    html.H1("📈 Stock Price Prediction App", style={'textAlign': 'center'}),
    
    dcc.Input(id="stock-symbol", type="text", placeholder="Enter Stock Symbol (e.g., TSLA)", value="TSLA", style={'width': '50%'}),
    
    dcc.DatePickerRange(
        id='date-range',
        start_date='2024-03-22',
        end_date='2025-03-22',
        display_format='YYYY-MM-DD'
    ),
    
    html.Button("Predict", id="predict-btn", n_clicks=0),
    
    html.Div(id="prediction-output", style={'margin-top': '20px', 'textAlign': 'center'}),
    
    dcc.Graph(id='stock-chart')
])

# Callback to update stock data and prediction
@app.callback(
    [Output('stock-chart', 'figure'), Output('prediction-output', 'children')],
    [Input('predict-btn', 'n_clicks')],
    [dash.State('stock-symbol', 'value'), dash.State('date-range', 'start_date'), dash.State('date-range', 'end_date')]
)
def update_stock_chart(n_clicks, symbol, start_date, end_date):
    if n_clicks == 0:
        return go.Figure(), ""

    # Fetch stock data
    df = yf.download(symbol, start=start_date, end=end_date)
    
    if df.empty:
        return go.Figure(), html.Div("⚠ Invalid Stock Symbol or No Data Available.", style={'color':'red','font-weight':'bold'})

    df['Days'] = np.arange(len(df))

    # Train a Linear Regression model
    model = LinearRegression()
    X = df[['Days']]
    y = df['Close']
    model.fit(X, y)
    
    # Predict future stock price
    future_price = model.predict(np.array[[len(df) + 30]])[0]

    # Plot stock prices
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index.astype(str), y=df['Close'], mode='lines', name='Actual Prices'))
    fig.add_trace(go.Scatter(x=df.index, y=model.predict(X), mode='lines', name='Regression Line', line=dict(dash='dot')))

    return fig, f"📌 Predicted Stock Price for {symbol}: ${round(future_price, 2)}"

# Run the app
if __name__ == "__main__":
    app.run(debug=True,use_reloader=False)