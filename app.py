import io
import numpy as np
import pandas as pd
import joblib
import altair as alt
import streamlit as st

st.set_page_config(page_title="Predicción de Churn", page_icon="📉", layout="wide")

NUM = ["Tenure", "NumberOfOrders", "AvgOrderValue", "TotalSpent",
       "LastPurchaseDays", "DiscountUsed", "SupportTickets"]
CAT = ["DeviceType", "Newsletter"]
REQUERIDAS = NUM + CAT
COLOR_RIESGO = alt.Scale(domain=["En riesgo", "Bajo riesgo"], range=["#d9534f", "#5b8def"])


@st.cache_resource
def cargar():
    return joblib.load("modelo_churn.joblib")


def limpiar(df):
    """Mismas reglas de limpieza usadas al entrenar: valores imposibles -> nulo
    (el pipeline del modelo los imputa con la mediana)."""
    d = df.copy()
    for c in NUM:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    for c in CAT:
        d[c] = d[c].where(d[c].isna(), d[c].astype(str).str.strip())
    d.loc[(d.Tenure < 0) | (d.Tenure > 120),
