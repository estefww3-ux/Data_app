import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────────────
# Configuración
# ─────────────────────────────
st.set_page_config(page_title="Segmentador General", page_icon="📊", layout="wide")

st.title("📊 Segmentador de Datos (K-Means)")
st.write("Sube cualquier dataset CSV y segmenta usando clustering")

# ─────────────────────────────
# Subida de archivo
# ─────────────────────────────
archivo = st.file_uploader("Sube tu archivo CSV", type=["csv"])

if archivo is not None:

    df = pd.read_csv(archivo)
    st.subheader("Vista previa")
    st.dataframe(df.head())

    # ─────────────────────────────
    # Selección de columnas
    # ─────────────────────────────
    columnas_numericas = df.select_dtypes(include=np.number).columns.tolist()

    if len(columnas_numericas) < 2:
        st.error("Necesitas al menos 2 columnas numéricas")
    else:
        cols = st.multiselect(
            "Selecciona columnas para clustering",
            columnas_numericas,
            default=columnas_numericas[:2]
        )

        # Slider de K
        k = st.sidebar.slider("Número de clusters (K)", 2, 10, 3)

        if st.button("Ejecutar Clustering"):

            X = df[cols]

            # Escalar datos
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Modelo
            modelo = KMeans(n_clusters=k, random_state=42, n_init=10)
            df["Cluster"] = modelo.fit_predict(X_scaled)

            st.success("Clustering realizado correctamente")

            # ─────────────────────────────
            # Gráfico
            # ─────────────────────────────
            fig = px.scatter(
                df,
                x=cols[0],
                y=cols[1],
                color=df["Cluster"].astype(str),
                title="Clusters generados"
            )

            # Centroides
            centroides = scaler.inverse_transform(modelo.cluster_centers_)
            fig.add_scatter(
                x=centroides[:, 0],
                y=centroides[:, 1],
                mode='markers',
                marker=dict(size=12, color='black'),
                name='Centroides'
            )

            st.plotly_chart(fig, use_container_width=True)

            # ─────────────────────────────
            # Métricas
            # ─────────────────────────────
            st.subheader("📐 Métricas del Cluster")

            cluster_sel = st.selectbox("Selecciona un cluster", df["Cluster"].unique())
            datos = df[df["Cluster"] == cluster_sel][cols]

            st.write("Varianza:", datos.var().mean())
            st.write("Desviación estándar:", datos.std().mean())

            # ─────────────────────────────
            # Convergencia
            # ─────────────────────────────
            st.subheader("🔁 Convergencia")

            st.write(f"Iteraciones: {modelo.n_iter_}")
            st.write(f"Inercia: {modelo.inertia_:.2f}")

            st.markdown("""
            El algoritmo K-Means converge cuando los centroides dejan de cambiar significativamente
            entre iteraciones, minimizando la distancia entre los puntos y sus centros.
            """)
