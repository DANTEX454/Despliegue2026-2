import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Predicción de Aprobación de Curso", layout="centered")

st.title("🎓 Predicción de Aprobación de Curso")
st.write("Ingrese los datos del estudiante para calcular la predicción utilizando el modelo Bagging Optimizado.")

# 1. Cargar recursos necesarios
@st.cache_resource
def cargar_recursos():
    columnas_one_hot = joblib.load('/content/one_hot_columns.joblib')
    scaler = joblib.load('/content/min_max_scaler.joblib')
    modelo = joblib.load('/content/bagging_optimizado.joblib')
    return columnas_one_hot, scaler, modelo

try:
    columnas_one_hot, scaler, modelo = cargar_recursos()
    st.success("¡Modelos y transformadores cargados con éxito!")
except Exception as e:
    st.error(f"Error al cargar los archivos .joblib: {e}")
    st.stop()

# 2. Formulario de entrada de datos
st.subheader("Datos del Estudiante")

# Extraer las categorías posibles para Felder desde las columnas del one-hot
categorias_felder = [col.replace('Felder_', '') for col in columnas_one_hot if col.startswith('Felder_')]
if not categorias_felder:
    categorias_felder = ['equilibrio', 'intuitivo', 'reflexivo', 'secuencial', 'sensorial', 'verbal', 'visual']

felder_seleccionado = st.selectbox("Estilo de Aprendizaje (Felder)", opciones=categorias_felder)
examen_admision = st.number_input("Puntaje de Examen de Admisión", min_value=0.0, max_value=10.0, value=3.8, step=0.1)

# 3. Procesar datos al presionar el botón
if st.button("Realizar Predicción"):
    # Crear DataFrame con el formato inicial
    df_input = pd.DataFrame([{
        'Felder': felder_seleccionado,
        'Examen_admisión': examen_admision
    }])
    
    # Crear el DataFrame procesado con ceros según las columnas del modelo
    df_procesado = pd.DataFrame(0.0, index=[0], columns=columnas_one_hot)
    
    # Aplicar One-Hot para Felder
    columna_felder_activa = f"Felder_{felder_seleccionado}"
    if columna_felder_activa in df_procesado.columns:
        df_procesado[columna_felder_activa] = 1.0
        
    # Escalar examen de admisión
    try:
        valor_scaled = scaler.transform([[examen_admision]])[0][0]
        df_procesado['Examen_admision_scaled'] = valor_scaled
    except Exception as e:
        st.error(f"Error al escalar el examen de admisión: {e}")
        st.stop()
        
    # Asegurar el orden correcto de las columnas
    df_procesado = df_procesado[columnas_one_hot]
    
    # Mostrar datos que se enviarán al modelo
    st.subheader("Datos Procesados para el Modelo")
    st.dataframe(df_procesado)
    
    # Realizar predicción
    try:
        prediccion = modelo.predict(df_procesado)
        
        st.subheader("Resultado de la Predicción")
        st.metric(label="Nota Final Estimada", value=f"{prediccion[0]:.2f}")
        
        # Intentar mostrar probabilidades si aplica
        if hasattr(modelo, "predict_proba"):
            probabilidades = modelo.predict_proba(df_procesado)
            st.write(f"Probabilidades por clase: {probabilidades[0]}")
            
    except Exception as e:
        st.error(f"Error al realizar la predicción: {e}")
