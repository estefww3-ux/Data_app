
Copiar

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
 
# ─────────────────────────────────────────
# Configuración de la página
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Segmentador RFM Pro",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
# ─────────────────────────────────────────
# CSS personalizado — diseño elegante
# ─────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f1117; }
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e, #2a2a3e);
        border: 1px solid #3a3a5c;
        border-radius: 16px;
        padding: 20px 24px;
        text-align: center;
        margin: 6px 0;
    }
    .metric-card .label {
        font-size: 0.85rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 6px;
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        color: #fff;
    }
    .metric-card .sub { font-size: 0.8rem; color: #60a5fa; margin-top: 4px; }
    .badge-vip      { background:#7c3aed; color:#fff; padding:4px 14px; border-radius:20px; font-weight:700; font-size:0.85rem; }
    .badge-leal     { background:#059669; color:#fff; padding:4px 14px; border-radius:20px; font-weight:700; font-size:0.85rem; }
    .badge-riesgo   { background:#dc2626; color:#fff; padding:4px 14px; border-radius:20px; font-weight:700; font-size:0.85rem; }
    .badge-nuevo    { background:#d97706; color:#fff; padding:4px 14px; border-radius:20px; font-weight:700; font-size:0.85rem; }
    .badge-inactivo { background:#4b5563; color:#fff; padding:4px 14px; border-radius:20px; font-weight:700; font-size:0.85rem; }
    .section-divider { border: none; border-top: 1px solid #2a2a3e; margin: 32px 0; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #13131f, #1a1a2e);
        border-right: 1px solid #2a2a3e;
    }
</style>
""", unsafe_allow_html=True)
 
 
# ─────────────────────────────────────────
# Función: etiquetar clusters
# ─────────────────────────────────────────
def etiquetar_cluster(row, medians):
    r = row["Recency_med"]
    f = row["Frequency_med"]
    m = row["Monetary_med"]
    if m > medians["Monetary"] * 1.5 and f > medians["Frequency"]:
        return "🏆 Clientes VIP"
    elif r < medians["Recency"] and f >= medians["Frequency"]:
        return "💚 Clientes Leales"
    elif r > medians["Recency"] * 1.5:
        return "😴 Clientes Inactivos"
    elif r < medians["Recency"] * 0.5:
        return "🆕 Clientes Nuevos"
    else:
        return "⚠️ En Riesgo"
 
def color_badge(label):
    if "VIP"      in label: return "badge-vip"
    if "Leal"     in label: return "badge-leal"
    if "Riesgo"   in label: return "badge-riesgo"
    if "Nuevo"    in label: return "badge-nuevo"
    return "badge-inactivo"
 
 
# ─────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 Segmentador RFM Pro")
    st.markdown("---")
    st.markdown("### ⚙️ Configuración")
    k = st.slider("Número de clusters (K)", 2, 10, 3,
                  help="El Elbow Method te ayuda a elegir el K óptimo.")
    st.markdown("---")
    st.markdown("### 📖 ¿Qué es RFM?")
    st.markdown("""
- **R**ecency → Días desde última compra  
- **F**requency → Número de compras  
- **M**onetary → Dinero total gastado
    """)
    st.markdown("---")
    st.info("1️⃣ Sube el CSV\n\n2️⃣ Ajusta K\n\n3️⃣ Ejecuta clustering")
 
 
# ─────────────────────────────────────────
# Header
# ─────────────────────────────────────────
st.markdown('<p class="main-title">🎯 Segmentador de Clientes — RFM Pro</p>', unsafe_allow_html=True)
st.markdown("Carga el archivo **Online-Retail.csv** original · La app limpia, calcula RFM y segmenta automáticamente.")
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
 
archivo = st.file_uploader("📂 Sube tu archivo CSV", type=["csv"])
 
if archivo is not None:
 
    with st.spinner("🧹 Limpiando datos..."):
        df_raw = pd.read_csv(archivo, encoding="ISO-8859-1")
        df = df_raw.copy()
        df.dropna(subset=["CustomerID"], inplace=True)
        df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
        df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
        df["TotalPrice"]  = df["Quantity"] * df["UnitPrice"]
 
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    kpis = [
        ("🧾 Transacciones",  f"{len(df):,}",                    "registros limpios"),
        ("👥 Clientes únicos", f"{df['CustomerID'].nunique():,}", "CustomerIDs"),
        ("🌍 Países",          f"{df['Country'].nunique()}",      "mercados"),
        ("💰 Revenue total",   f"£{df['TotalPrice'].sum():,.0f}", "ingresos"),
    ]
    for col, (label, val, sub) in zip([col1, col2, col3, col4], kpis):
        col.markdown(f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{val}</div>
            <div class="sub">{sub}</div>
        </div>""", unsafe_allow_html=True)
 
    # RFM
    fecha_ref = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("CustomerID").agg(
        Recency   = ("InvoiceDate", lambda x: (fecha_ref - x.max()).days),
        Frequency = ("InvoiceNo",   "nunique"),
        Monetary  = ("TotalPrice",  "sum")
    ).reset_index()
 
    with st.expander("📊 Ver tabla RFM calculada"):
        st.dataframe(rfm.head(15), use_container_width=True)
        st.caption(f"Total clientes: {len(rfm):,}")
 
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
 
    # Elbow Method
    st.subheader("📐 Elbow Method — ¿Cuál es el K óptimo?")
    st.caption("Busca el 'codo' de la curva: donde la inercia deja de bajar bruscamente.")
 
    with st.spinner("Calculando Elbow..."):
        X_rfm = rfm[["Recency","Frequency","Monetary"]]
        scaler_e = StandardScaler()
        X_e = scaler_e.fit_transform(X_rfm)
        inercias = [KMeans(n_clusters=ki, random_state=42, n_init=10).fit(X_e).inertia_
                    for ki in range(2, 11)]
 
    fig_elbow = go.Figure()
    fig_elbow.add_trace(go.Scatter(
        x=list(range(2,11)), y=inercias, mode="lines+markers",
        line=dict(color="#a78bfa", width=3),
        marker=dict(size=10, color="#60a5fa", line=dict(color="#a78bfa", width=2))
    ))
    fig_elbow.add_vline(x=k, line_dash="dash", line_color="#34d399",
                        annotation_text=f"K={k} seleccionado",
                        annotation_font_color="#34d399")
    fig_elbow.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,20,35,0.8)",
        xaxis_title="K", yaxis_title="Inercia", height=350,
        margin=dict(l=40,r=40,t=30,b=40)
    )
    st.plotly_chart(fig_elbow, use_container_width=True)
 
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
 
    # Botón
    _, col_btn, _ = st.columns([1,2,1])
    with col_btn:
        run = st.button("🚀 Ejecutar Clustering", type="primary", use_container_width=True)
 
    if run:
        with st.spinner("Entrenando KMeans..."):
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_rfm)
            modelo = KMeans(n_clusters=k, random_state=42, n_init=10)
            rfm["Cluster"] = modelo.fit_predict(X_scaled)
            st.session_state.update({"rfm":rfm,"modelo":modelo,"scaler":scaler,"k":k})
        st.success("✅ Clustering completado")
 
    if "rfm" in st.session_state:
        rfm    = st.session_state["rfm"]
        modelo = st.session_state["modelo"]
        scaler = st.session_state["scaler"]
        k_used = st.session_state["k"]
 
        medians_global = rfm[["Recency","Frequency","Monetary"]].median()
        resumen = rfm.groupby("Cluster").agg(
            Clientes      = ("CustomerID","count"),
            Recency_med   = ("Recency",   "median"),
            Frequency_med = ("Frequency", "median"),
            Monetary_med  = ("Monetary",  "median")
        ).reset_index()
        resumen["Segmento"] = resumen.apply(
            lambda r: etiquetar_cluster(r, medians_global), axis=1)
 
        palette = ["#a78bfa","#60a5fa","#34d399","#f59e0b","#f87171",
                   "#e879f9","#38bdf8","#4ade80","#fb923c","#f43f5e"]
 
        # Tabla resumen
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.subheader("🗂️ Resumen de Segmentos")
        cols_res = st.columns(len(resumen))
        for i, (col, (_, row)) in enumerate(zip(cols_res, resumen.iterrows())):
            col.markdown(f"""
            <div class="metric-card">
                <div class="label">Cluster {int(row['Cluster'])}</div>
                <span class="{color_badge(row['Segmento'])}">{row['Segmento']}</span>
                <br><br>
                <div style="font-size:1.6rem;font-weight:700;color:{palette[i]}">{int(row['Clientes']):,}</div>
                <div class="sub">clientes</div>
                <hr style="border-color:#2a2a3e;margin:10px 0">
                <div style="font-size:0.8rem;color:#aaa;text-align:left">
                    📅 Recency: <b style="color:#fff">{row['Recency_med']:.0f}</b> días<br>
                    🔁 Frequency: <b style="color:#fff">{row['Frequency_med']:.0f}</b> compras<br>
                    💰 Monetary: <b style="color:#fff">£{row['Monetary_med']:,.0f}</b>
                </div>
            </div>""", unsafe_allow_html=True)
 
        # Gráfico 3D
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.subheader("🌐 Visualización 3D de Clusters")
        rfm_plot = rfm.merge(resumen[["Cluster","Segmento"]], on="Cluster")
        fig3d = px.scatter_3d(
            rfm_plot, x="Recency", y="Frequency", z="Monetary",
            color="Segmento", opacity=0.65,
            color_discrete_sequence=palette,
            title=f"Clusters RFM en 3D (K={k_used})",
            hover_data={"CustomerID":True,"Recency":True,"Frequency":True,"Monetary":":.0f"}
        )
        centroides = scaler.inverse_transform(modelo.cluster_centers_)
        fig3d.add_trace(go.Scatter3d(
            x=centroides[:,0], y=centroides[:,1], z=centroides[:,2],
            mode="markers+text",
            marker=dict(size=10, color="white", symbol="cross",
                        line=dict(color="black", width=2)),
            text=[f"C{i}" for i in range(k_used)],
            textfont=dict(color="white", size=12),
            name="Centroides"
        ))
        fig3d.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            height=520, margin=dict(l=0,r=0,t=40,b=0)
        )
        st.plotly_chart(fig3d, use_container_width=True)
 
        # Gráficos 2D
        st.subheader("🔍 Vistas 2D")
        fig2d = make_subplots(rows=1, cols=2,
                              subplot_titles=("Recency vs Monetary","Recency vs Frequency"))
        for i, seg in enumerate(rfm_plot["Segmento"].unique()):
            sub = rfm_plot[rfm_plot["Segmento"]==seg]
            fig2d.add_trace(go.Scatter(x=sub["Recency"], y=sub["Monetary"],
                mode="markers", name=seg,
                marker=dict(color=palette[i], opacity=0.6, size=5),
                legendgroup=seg), row=1, col=1)
            fig2d.add_trace(go.Scatter(x=sub["Recency"], y=sub["Frequency"],
                mode="markers", name=seg,
                marker=dict(color=palette[i], opacity=0.6, size=5),
                showlegend=False, legendgroup=seg), row=1, col=2)
        fig2d.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(20,20,35,0.8)", height=400,
            margin=dict(l=40,r=40,t=40,b=40)
        )
        st.plotly_chart(fig2d, use_container_width=True)
 
        # Métricas por cluster
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.subheader("📐 Métricas de Dispersión por Cluster")
        cluster_sel = st.selectbox(
            "Selecciona un cluster:",
            sorted(rfm["Cluster"].unique()),
            format_func=lambda x: f"Cluster {x} — {resumen[resumen['Cluster']==x]['Segmento'].values[0]}"
        )
        datos = rfm[rfm["Cluster"]==cluster_sel][["Recency","Frequency","Monetary"]]
        c1,c2,c3,c4 = st.columns(4)
        for col, (label, val) in zip([c1,c2,c3,c4],[
            ("👥 Clientes",         f"{len(datos):,}"),
            ("📊 Varianza media",   f"{datos.var().mean():,.2f}"),
            ("📉 Desv. Est. media", f"{datos.std().mean():,.2f}"),
            ("💰 Monetary media",   f"£{datos['Monetary'].mean():,.0f}"),
        ]):
            col.markdown(f"""
            <div class="metric-card">
                <div class="label">{label}</div>
                <div class="value" style="font-size:1.5rem">{val}</div>
            </div>""", unsafe_allow_html=True)
 
        with st.expander("Ver detalle por columna"):
            det = pd.DataFrame({
                "Varianza": datos.var(), "Desv. Estándar": datos.std(),
                "Media": datos.mean(), "Mínimo": datos.min(), "Máximo": datos.max()
            })
            st.dataframe(det.style.format("{:.2f}"), use_container_width=True)
 
        # Convergencia
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.subheader("🔁 Convergencia del Modelo")
        c1, c2 = st.columns(2)
        c1.markdown(f"""<div class="metric-card">
            <div class="label">Iteraciones hasta convergencia</div>
            <div class="value">{modelo.n_iter_}</div>
            <div class="sub">para K={k_used}</div>
        </div>""", unsafe_allow_html=True)
        c2.markdown(f"""<div class="metric-card">
            <div class="label">Inercia final</div>
            <div class="value">{modelo.inertia_:,.2f}</div>
            <div class="sub">suma de distancias al centroide</div>
        </div>""", unsafe_allow_html=True)
 
        # Descarga
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        rfm_out = rfm.merge(resumen[["Cluster","Segmento"]], on="Cluster")
        st.download_button(
            "⬇️ Descargar tabla RFM con clusters y segmentos",
            data=rfm_out.to_csv(index=False).encode("utf-8"),
            file_name="rfm_segmentos.csv",
            mime="text/csv",
            use_container_width=True
        )
 
