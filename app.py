import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────────────────────────
# Configuración de la página
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Segmentador de Clientes RFM",
    page_icon="🛍️",
    layout="wide"
)

st.title("🛍️ Segmentador de Clientes — Análisis RFM")
st.markdown("Sube el archivo **Online-Retail.csv** original. La app lo limpia y construye los segmentos automáticamente.")

# ─────────────────────────────────────────
# Sidebar — Configuración
# ─────────────────────────────────────────
st.sidebar.header("⚙️ Configuración")
k = st.sidebar.slider("Número de clusters (K)", min_value=2, max_value=10, value=3)
st.sidebar.markdown("---")
st.sidebar.info("Sube el CSV, ajusta K y presiona **Ejecutar clustering**.")

# ─────────────────────────────────────────
# Paso 1: Carga del archivo
# ─────────────────────────────────────────
archivo = st.file_uploader("📂 Sube tu archivo CSV", type=["csv"])

if archivo is not None:

    # ─────────────────────────────────────
    # Paso 2: Limpieza automática
    # ─────────────────────────────────────
    with st.spinner("Limpiando datos..."):
        df_raw = pd.read_csv(archivo, encoding="ISO-8859-1")

        df = df_raw.copy()
        df.dropna(subset=["CustomerID"], inplace=True)
        df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]  # quitar cancelaciones
        df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
        df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    st.success(f"✅ Datos limpios: {len(df):,} filas | {df['CustomerID'].nunique():,} clientes únicos")

    with st.expander("👀 Ver muestra de datos limpios"):
        st.dataframe(df.head(10))

    # ─────────────────────────────────────
    # Paso 3: Construcción de tabla RFM
    # ─────────────────────────────────────
    fecha_referencia = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerID").agg(
        Recency   = ("InvoiceDate", lambda x: (fecha_referencia - x.max()).days),
        Frequency = ("InvoiceNo",   "nunique"),
        Monetary  = ("TotalPrice",  "sum")
    ).reset_index()

    with st.expander("📊 Ver tabla RFM calculada"):
        st.dataframe(rfm.head(10))
        st.caption(f"Total clientes: {len(rfm):,}")

    # ─────────────────────────────────────
    # Paso 4: Botón — Ejecutar clustering
    # ─────────────────────────────────────
    if st.button("🚀 Ejecutar clustering", type="primary"):
        with st.spinner("Entrenando KMeans..."):
            X = rfm[["Recency", "Frequency", "Monetary"]]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            modelo = KMeans(n_clusters=k, random_state=42, n_init=10)
            rfm["Cluster"] = modelo.fit_predict(X_scaled)

            st.session_state["rfm"]    = rfm
            st.session_state["modelo"] = modelo
            st.session_state["scaler"] = scaler
            st.session_state["k"]      = k

        st.success("✅ Clustering completado")

    # ─────────────────────────────────────
    # Paso 5: Resultados (persisten con session_state)
    # ─────────────────────────────────────
    if "rfm" in st.session_state:
        rfm    = st.session_state["rfm"]
        modelo = st.session_state["modelo"]
        k_used = st.session_state["k"]

        st.markdown("---")
        st.subheader("📈 Visualización de Clusters")

        # Gráfico 3D interactivo
        fig = px.scatter_3d(
            rfm,
            x="Recency", y="Frequency", z="Monetary",
            color=rfm["Cluster"].astype(str),
            title=f"Clusters RFM (K={k_used})",
            labels={"color": "Cluster"},
            opacity=0.7,
            color_discrete_sequence=px.colors.qualitative.Bold
        )

        # Centroides (desescalados)
        scaler = st.session_state["scaler"]
        centroides = scaler.inverse_transform(modelo.cluster_centers_)
        fig.add_trace(go.Scatter3d(
            x=centroides[:, 0],
            y=centroides[:, 1],
            z=centroides[:, 2],
            mode="markers+text",
            marker=dict(size=10, color="black", symbol="cross"),
            text=[f"C{i}" for i in range(k_used)],
            name="Centroides"
        ))

        st.plotly_chart(fig, use_container_width=True)

        # También gráfico 2D Recency vs Monetary
        st.subheader("🔍 Vista 2D — Recency vs Monetary")
        fig2 = px.scatter(
            rfm,
            x="Recency", y="Monetary",
            color=rfm["Cluster"].astype(str),
            hover_data=["CustomerID", "Frequency"],
            title="Recency vs Monetary por Cluster",
            labels={"color": "Cluster"},
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        st.plotly_chart(fig2, use_container_width=True)

        # ─────────────────────────────────
        # Paso 6: Métricas por cluster
        # ─────────────────────────────────
        st.markdown("---")
        st.subheader("📐 Métricas de Dispersión por Cluster")

        cluster_sel = st.selectbox(
            "Selecciona un cluster para inspeccionar:",
            sorted(rfm["Cluster"].unique()),
            format_func=lambda x: f"Cluster {x}"
        )

        datos_cluster = rfm[rfm["Cluster"] == cluster_sel][["Recency", "Frequency", "Monetary"]]

        col1, col2, col3 = st.columns(3)
        col1.metric("👥 Clientes", len(datos_cluster))
        col2.metric("📊 Varianza media", f"{datos_cluster.var().mean():.2f}")
        col3.metric("📉 Desv. Estándar media", f"{datos_cluster.std().mean():.2f}")

        with st.expander("Ver detalle de varianza y desviación por columna"):
            detalle = pd.DataFrame({
                "Varianza": datos_cluster.var(),
                "Desv. Estándar": datos_cluster.std(),
                "Media": datos_cluster.mean()
            })
            st.dataframe(detalle.style.format("{:.2f}"))

        # ─────────────────────────────────
        # Paso 7: Resumen de convergencia
        # ─────────────────────────────────
        st.markdown("---")
        st.subheader("🔁 Convergencia del Modelo")
        st.info(
            f"KMeans convergió en **{modelo.n_iter_}** iteraciones para K={k_used}. "
            f"La inercia final (suma de distancias al centroide) fue **{modelo.inertia_:,.2f}**."
        )

        # Descargar resultados
        st.markdown("---")
        csv_out = rfm.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Descargar tabla RFM con clusters",
            data=csv_out,
            file_name="rfm_clusters.csv",
            mime="text/csv"
        )    