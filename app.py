import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.set_page_config(
    page_title="Projet Série Temporelle",
    layout="wide"
)

st.title("Application Série Temporelle - ARIMA / SARIMA")

if "analysis_started" not in st.session_state:
    st.session_state.analysis_started = False

uploaded_file = st.file_uploader(
    "Charger un fichier CSV ou Excel",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:

    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Aperçu des données")
    st.dataframe(df.head())

    date_col = st.selectbox("Colonne de date", df.columns)
    value_col = st.selectbox("Colonne de la série", df.columns)

    if st.button("Lancer l'analyse"):
        st.session_state.analysis_started = True

    if st.session_state.analysis_started:

        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)
        df.set_index(date_col, inplace=True)

        serie = df[value_col]

        st.subheader("Série temporelle originale")
        fig, ax = plt.subplots()
        ax.plot(serie)
        ax.set_xlabel("Temps")
        ax.set_ylabel("Valeur")
        st.pyplot(fig)

        st.subheader("Décomposition STL")
        period = st.number_input("Période de saisonnalité", min_value=2, value=12)

        stl = STL(serie, period=period)
        result = stl.fit()

        fig, axes = plt.subplots(4, 1, figsize=(10, 8))
        axes[0].plot(serie)
        axes[0].set_title("Original")
        axes[1].plot(result.trend)
        axes[1].set_title("Tendance")
        axes[2].plot(result.seasonal)
        axes[2].set_title("Saisonnalité")
        axes[3].plot(result.resid)
        axes[3].set_title("Résidu")
        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("Test de stationnarité (ADF)")
        _, p_value, _, _, _, _ = adfuller(serie.dropna())

        if p_value < 0.05:
            st.success(f"Série stationnaire (p-value = {p_value:.4f})")
        else:
            st.warning(f"Série non stationnaire (p-value = {p_value:.4f})")

        st.subheader("ACF et PACF")

        max_lags = st.slider("Nombre de retards", 10, 60, 40)

        col1, col2 = st.columns(2)

        with col1:
            fig_acf, ax_acf = plt.subplots()
            plot_acf(serie.dropna(), lags=max_lags, ax=ax_acf)
            st.pyplot(fig_acf)

        with col2:
            fig_pacf, ax_pacf = plt.subplots()
            plot_pacf(serie.dropna(), lags=max_lags, ax=ax_pacf, method="ywm")
            st.pyplot(fig_pacf)

        st.subheader("Choix du modèle")
        model_type = st.selectbox("Modèle", ["ARIMA", "SARIMA"])
        horizon = st.number_input("Horizon de prédiction", 1, value=12)

        if model_type == "ARIMA":
            p = st.number_input("p", 0, 5, 1)
            d = st.number_input("d", 0, 2, 1)
            q = st.number_input("q", 0, 5, 1)

            model = ARIMA(serie, order=(p, d, q))
            model_fit = model.fit()
        else:
            p = st.number_input("p", 0, 5, 1)
            d = st.number_input("d", 0, 2, 1)
            q = st.number_input("q", 0, 5, 1)
            P = st.number_input("P", 0, 5, 1)
            D = st.number_input("D", 0, 2, 1)
            Q = st.number_input("Q", 0, 5, 1)
            s = st.number_input("Période saisonnière", 2, 24, 12)

            model = SARIMAX(
                serie,
                order=(p, d, q),
                seasonal_order=(P, D, Q, s)
            )
            model_fit = model.fit()

        forecast = model_fit.forecast(steps=horizon)

        st.subheader("Prédiction")
        fig, ax = plt.subplots()
        ax.plot(serie, label="Observé")
        ax.plot(forecast, label="Prédit")
        ax.legend()
        st.pyplot(fig)
