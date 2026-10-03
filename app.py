import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Predicción de Aprobación de Curso", layout="wide")

st.title("🎓 Predicción de Aprobación de Curso")
st.write("Esta aplicación procesa las variables de entrada y realiza la predicción utilizando el modelo de Bagging Optimizado.")

# Determinar la ruta base dinámica
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Función para cargar los recursos
@st.cache_resource
def cargar_recursos():
    path_one_hot = os.path.join(BASE_DIR, 'one_hot_columns.joblib')
    if not os.path.exists(path_one_hot):
        path_one_hot = '/content/one_hot_columns.joblib'

    path_scaler = os.path.join(BASE_DIR, 'min_max_scaler.joblib')
    if not os.path.exists(path_scaler):
        path_scaler = '/content/min_max_scaler.joblib'

    path_modelo = os.path.join(BASE_DIR, 'bagging_optimizado.joblib')
    if not os.path.exists(path_modelo):
        path_modelo = '/content/bagging_optimizado.joblib'

    columnas_one_hot = joblib.load(path_one_hot)
    scaler = joblib.load(path_scaler)
    modelo = joblib.load(path_modelo)
    return columnas_one_hot, scaler, modelo

try:
    columnas_one_hot, scaler, modelo = cargar_recursos()
except Exception as e:
    st.error(f"Error al cargar recursos .joblib: {e}")
    st.stop()

# Crear pestañas para separar la predicción individual de la predicción por archivo
tab1, tab2 = st.tabs(["✍️ Predicción Individual", "📁 Subir Archivo Excel"])

with tab1:
    st.header("Datos de Entrada del Estudiante")
    categorias_felder = [col.replace('Felder_', '') for col in columnas_one_hot if col.startswith('Felder_')]
    if not categorias_felder:
        categorias_felder = ['sensorial', 'activo', 'visual', 'equilibrio', 'secuencial', 'reflexivo', 'verbal', 'intuitivo']

    felder_input = st.selectbox("Selecciona el estilo de aprendizaje (Felder):", categorias_felder)
    examen_input = st.number_input("Examen de Admisión:", min_value=0.0, max_value=10.0, value=3.83, step=0.01)

    if st.button("Realizar Predicción Individual"):
        try:
            df_input = pd.DataFrame([{
                'Felder': felder_input,
                'Examen_admisión': examen_input
            }])
            
            df_procesado = pd.DataFrame(0.0, index=[0], columns=columnas_one_hot)
            columna_felder_activa = f"Felder_{felder_input}"
            if columna_felder_activa in df_procesado.columns:
                df_procesado[columna_felder_activa] = 1.0
            
            df_procesado['Examen_admision_scaled'] = scaler.transform([[examen_input]])[0][0]
            df_procesado = df_procesado[columnas_one_hot]
            
            st.subheader("Datos Procesados")
            st.dataframe(df_procesado)
            
            prediccion = modelo.predict(df_procesado)
            st.success(f"La predicción de la Nota Final Estimada es: {prediccion[0]:.4f}")
        except Exception as e:
            st.error(f"Ocurrió un error: {e}")

with tab2:
    st.header("Predicción por Lotes mediante Excel")
    st.write("Sube un archivo de Excel que contenga al menos las columnas `Felder` y `Examen_admisión`.")
    
    archivo_cargado = st.file_uploader("Seleccione el archivo Excel (.xlsx):", type=["xlsx"])
    
    if archivo_cargado is not None:
        try:
            df_excel = pd.read_excel(archivo_cargado)
            st.write("Vista previa de los datos subidos:")
            st.dataframe(df_excel.head())
            
            # Validar columnas requeridas
            if 'Felder' in df_excel.columns and 'Examen_admisión' in df_excel.columns:
                if st.button("Procesar y Predicir Lote"):
                    # Crear matriz con ceros según estructura requerida
                    df_procesado_lote = pd.DataFrame(0.0, index=df_excel.index, columns=columnas_one_hot)
                    
                    # Aplicar One-Hot encoding para la columna 'Felder'
                    for val_felder in df_excel['Felder'].unique():
                        col_columna = f"Felder_{val_felder}"
                        if col_columna in df_procesado_lote.columns:
                            df_procesado_lote.loc[df_excel['Felder'] == val_felder, col_columna] = 1.0
                    
                    # Normalizar Examen_admisión
                    examenes_scaled = scaler.transform(df_excel[['Examen_admisión']])
                    df_procesado_lote['Examen_admision_scaled'] = examenes_scaled[:, 0]
                    
                    # Asegurar el orden de las columnas
                    df_procesado_lote = df_procesado_lote[columnas_one_hot]
                    
                    # Realizar la predicción para todo el dataset
                    predicciones_lote = modelo.predict(df_procesado_lote)
                    
                    # Añadir predicciones al dataframe original
                    df_resultado = df_excel.copy()
                    df_resultado['Nota_final_estimada'] = predicciones_lote
                    
                    st.subheader("Resultados de las Predicciones")
                    st.dataframe(df_resultado)
                    
                    # Permitir la descarga del resultado procesado
                    @st.cache_data
                    def convertir_excel(df_to_download):
                        import io
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            df_to_download.to_excel(writer, index=False, sheet_name='Predicciones')
                        return output.getvalue()
                    
                    excel_data = convertir_excel(df_resultado)
                    st.download_button(
                        label="Descargar resultados en Excel",
                        data=excel_data,
                        file_name="predicciones_aprobacion.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            else:
                st.error("El archivo Excel debe contener las columnas exactas: 'Felder' y 'Examen_admisión'.")
        except Exception as e:
            st.error(f"Ocurrió un error al procesar el archivo Excel: {e}")
