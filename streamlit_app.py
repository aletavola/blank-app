import streamlit as st
from pycoingecko import CoinGeckoAPI
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA

# Inicializar a API do CoinGecko
cg = CoinGeckoAPI()

# Configurar o título e a explicação inicial
st.title("Análise de Criptomoedas")
st.markdown("""
    ### Bem-vindo!
    Este aplicativo exibe a análise gráfica de criptomoedas populares, incluindo:
    - **Gráfico de Candlesticks:** Mostra as últimas 4 horas de variações de preço.
    - **Índice de Força Relativa (RSI):** Indica sobrecompra (RSI > 70) ou sobrevenda (RSI < 30).
    - **Previsão de Preços:** Calculada para as próximas 2 horas com base nos últimos 24 horas de dados.

    Use o menu abaixo para selecionar uma criptomoeda.
""")

# Lista de criptomoedas disponíveis
cryptocurrencies = {
    "Bitcoin": "bitcoin",
    "Ethereum": "ethereum",
    "XRP": "ripple",
    "Kaspa": "kaspa",
    "Binance Coin": "binancecoin",
    "Cardano": "cardano",
}

# Criar o combobox para selecionar a criptomoeda
selected_crypto = st.selectbox("Escolha a criptomoeda:", list(cryptocurrencies.keys()))

# Obter o ID da moeda selecionada
crypto_id = cryptocurrencies[selected_crypto]

# Obter o intervalo para os últimos 30 horas
end_date = datetime.now()
start_date = end_date - timedelta(hours=30)  # Últimas 30 horas

# Converter datas para timestamps
start_timestamp = int(start_date.timestamp())
end_timestamp = int(end_date.timestamp())

# Obter dados históricos da moeda selecionada
with st.spinner(f"Carregando dados de {selected_crypto}..."):
    historical_data = cg.get_coin_market_chart_range_by_id(
        id=crypto_id,
        vs_currency="usd",
        from_timestamp=start_timestamp,
        to_timestamp=end_timestamp,
    )

# Processar os dados para um DataFrame
prices = historical_data["prices"]  # Lista de [timestamp, preço]
df = pd.DataFrame(prices, columns=["timestamp", "price"])
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms") - timedelta(hours=3)  # Ajustar para GMT-3
df.set_index("timestamp", inplace=True)

# Adicionar colunas para "open", "high", "low", "close" no formato intradiário
df["open"] = df["price"]
df["high"] = df["price"]
df["low"] = df["price"]
df["close"] = df["price"]

# Consolidar os dados para intervalos de 15 minutos
ohlc = df.resample("15T").agg({
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last"
})
ohlc.dropna(inplace=True)

# Cálculo do RSI (Relative Strength Index)
window_length = 14
delta = ohlc["close"].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=window_length).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=window_length).mean()
rs = gain / loss
ohlc["rsi"] = 100 - (100 / (1 + rs))

# Valor mais recente do RSI
latest_rsi = ohlc["rsi"].iloc[-1]

# Treinar o modelo de previsão com 24 horas de dados
training_data = ohlc.iloc[-96:]  # Últimas 96 intervalos = 24 horas
model = ARIMA(training_data["close"], order=(3, 1, 1))
fitted_model = model.fit()

# Fazer previsões para as próximas 2 horas (8 intervalos)
forecast_steps = 8
forecast = fitted_model.forecast(steps=forecast_steps)
future_timestamps = [training_data.index[-1] + timedelta(minutes=15 * i) for i in range(1, forecast_steps + 1)]

# Criar DataFrame para previsões
future_df = pd.DataFrame({
    "timestamp": future_timestamps,
    "predicted_close": forecast
}).set_index("timestamp")

# Combinar dados históricos e previsões
combined_df = pd.concat([ohlc.iloc[-16:], future_df])

# Identificar máxima prevista nas próximas 2 horas
max_future_price = future_df["predicted_close"].max()
max_future_time = future_df["predicted_close"].idxmax()

# Gerar os gráficos
fig, (ax_candlestick, ax_rsi, ax_combined) = plt.subplots(3, 1, figsize=(14, 18), gridspec_kw={"height_ratios": [3, 1, 2]})

# Gráfico 1: Candlestick
mpf.plot(
    ohlc.iloc[-16:],  # Últimas 4 horas
    type="candle",
    style="yahoo",
    ax=ax_candlestick,
    ylabel="Preço (USD)"
)
ax_candlestick.set_title(f"Gráfico de Candlesticks ({selected_crypto}) - Últimas 4 Horas")

# Gráfico 2: RSI
ax_rsi.plot(ohlc.iloc[-16:].index, ohlc["rsi"].iloc[-16:], label="RSI", color="purple")
ax_rsi.axhline(70, color="red", linestyle="--", linewidth=0.8, label="Sobrecomprado (70)")
ax_rsi.axhline(30, color="green", linestyle="--", linewidth=0.8, label="Sobrevendido (30)")
ax_rsi.set_title("Índice de Força Relativa (RSI)")
ax_rsi.set_ylabel("RSI")
ax_rsi.legend()
ax_rsi.grid()

# Adicionar o valor mais recente do RSI no gráfico
ax_rsi.text(
    0.02, 0.95,  # Posição no canto superior esquerdo
    f"RSI Atual: {latest_rsi:.2f}",
    transform=ax_rsi.transAxes,
    fontsize=12,
    color="blue",
    bbox=dict(facecolor="white", alpha=0.7)
)

# Gráfico 3: Previsão
ax_combined.plot(ohlc.iloc[-16:].index, ohlc["close"].iloc[-16:], label="Últimas 4 Horas", color="blue")
ax_combined.plot(future_df.index, future_df["predicted_close"], label="Previsão (Próximas 2 Horas)", color="red", linestyle="--")
ax_combined.set_title(f"Previsão de Preços ({selected_crypto}) - Próximas 2 Horas")
ax_combined.set_ylabel("Preço (USD)")
ax_combined.legend()
ax_combined.grid()

# Exibir os gráficos no Streamlit
st.pyplot(fig)

# Exibir a máxima futura
st.markdown(f"### Máxima Prevista: ${max_future_price:.2f} USD em {max_future_time.strftime('%H:%M')}")

# Exibir recomendação
recent_moving_avg = ohlc["close"].iloc[-4:].mean()
if forecast.diff().mean() > 0 and forecast.iloc[-1] < recent_moving_avg:
    recommendation = "Recomendação: COMPRAR"
elif forecast.diff().mean() < 0 and forecast.iloc[-1] > recent_moving_avg:
    recommendation = "Recomendação: VENDER"
else:
    recommendation = "Recomendação: RETER"

st.markdown(f"### {recommendation}")

# Footer
st.markdown("""
---
**Por Alexandre Tavola - 2024**  
[http://www.alexandretavola.com](http://www.alexandretavola.com)
""")