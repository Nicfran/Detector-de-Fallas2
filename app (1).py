import streamlit as st
import pandas as pd
import sqlite3
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Modelo Preventivo — Industria Neuquén",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  ESTILOS
# ─────────────────────────────────────────────
st.markdown("""
<style>
  /* Fondo principal */
  [data-testid="stAppViewContainer"] { background: #0f1117; }
  [data-testid="stSidebar"]          { background: #161b27; border-right: 1px solid #2a3045; }

  /* Métricas */
  [data-testid="metric-container"] {
    background: #161b27;
    border: 1px solid #2a3045;
    border-radius: 10px;
    padding: 16px 20px;
  }
  [data-testid="stMetricValue"]  { color: #38bdf8; font-size: 2rem !important; font-weight: 700; }
  [data-testid="stMetricLabel"]  { color: #94a3b8; font-size: .8rem; text-transform: uppercase; letter-spacing: .08em; }
  [data-testid="stMetricDelta"]  { color: #34d399; }

  /* Títulos */
  h1 { color: #f0f4ff !important; font-weight: 800 !important; letter-spacing: -.02em; }
  h2, h3 { color: #cbd5e1 !important; font-weight: 600 !important; }

  /* Botones */
  .stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #6366f1);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.55rem 1.4rem;
    transition: opacity .2s;
  }
  .stButton > button:hover { opacity: .88; }

  /* Separador */
  hr { border-color: #2a3045 !important; }

  /* Tablas */
  .stDataFrame { border: 1px solid #2a3045; border-radius: 8px; }

  /* Cajas de alerta */
  .alerta-falla {
    background: #3b0f0f;
    border: 1.5px solid #f87171;
    border-radius: 10px;
    padding: 18px 22px;
    margin-top: 12px;
  }
  .alerta-ok {
    background: #052e16;
    border: 1.5px solid #34d399;
    border-radius: 10px;
    padding: 18px 22px;
    margin-top: 12px;
  }

  /* Chips de info */
  .chip {
    display: inline-block;
    background: #1e2740;
    border: 1px solid #2a3045;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: .78rem;
    color: #94a3b8;
    margin: 2px;
  }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Modelo Preventivo")
    st.markdown("**Industria Neuquén**")
    st.markdown("---")

    st.markdown("### Fuente de datos")
    modo_datos = st.radio(
        "Seleccioná la fuente",
        ["Base de datos SQLite", "Subir CSV manualmente"],
        label_visibility="collapsed",
    )

    db_path = None
    df_uploaded = None

    if modo_datos == "Base de datos SQLite":
        db_path = st.text_input("Ruta de la base de datos", value="industria_Neuquen.db")
    else:
        archivo = st.file_uploader("Subir archivo CSV", type=["csv"])
        if archivo:
            df_uploaded = pd.read_csv(archivo)

    st.markdown("---")
    st.markdown("### Configuración del modelo")
    usar_grid = st.toggle("GridSearchCV (más lento, mejor resultado)", value=True)
    test_size = st.slider("Tamaño del set de prueba", 0.1, 0.4, 0.2, 0.05)

    if usar_grid:
        n_estimators = [100, 200]
        max_depth    = [3, 5, 7]
        lr_vals      = [0.01, 0.1]
    else:
        n_estimators = [100]
        max_depth    = [5]
        lr_vals      = [0.1]

    st.markdown("---")
    st.markdown(
        "<span class='chip'>XGBoost</span> <span class='chip'>SMOTE</span> "
        "<span class='chip'>StandardScaler</span>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
#  CABECERA
# ─────────────────────────────────────────────
st.markdown("# ⚙️ Modelo Preventivo de Fallas Industriales")
st.markdown(
    "Detección anticipada de fallas en equipos mediante sensores industriales. "
    "XGBoost + SMOTE para datos desbalanceados."
)
st.markdown("---")

# ─────────────────────────────────────────────
#  CARGA DE DATOS
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def cargar_sqlite(path):
    conn = sqlite3.connect(path)
    df = pd.read_sql_query("SELECT * FROM sensores", conn)
    conn.close()
    return df

def cargar_datos():
    if modo_datos == "Base de datos SQLite":
        try:
            return cargar_sqlite(db_path), None
        except Exception as e:
            return None, str(e)
    else:
        if df_uploaded is not None:
            return df_uploaded, None
        return None, "No se cargó ningún archivo."

df_raw, error = cargar_datos()

if error:
    st.error(f"❌ No se pudo cargar la base de datos: **{error}**")
    st.info(
        "Asegurate de que el archivo `industria_Neuquen.db` esté en la misma "
        "carpeta que `app.py`, o subí un CSV desde la barra lateral."
    )
    st.stop()

# ─────────────────────────────────────────────
#  TABS PRINCIPALES
# ─────────────────────────────────────────────
tab_datos, tab_modelo, tab_prediccion = st.tabs([
    "📊 Exploración de datos",
    "🤖 Entrenamiento del modelo",
    "🔍 Predicción individual",
])

# ══════════════════════════════════════════════
#  TAB 1 — EXPLORACIÓN
# ══════════════════════════════════════════════
with tab_datos:
    df = df_raw.copy()
    df["Diferencia_Temp"] = df["Temp_Proceso_C"] - df["Temp_Ambiente_C"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Registros totales", f"{len(df):,}")
    col2.metric("Variables", len(df.columns))
    fallas = int(df["Falla_Detectada"].sum())
    col3.metric("Fallas detectadas", f"{fallas:,}", delta=f"{fallas/len(df)*100:.1f}%")
    col4.metric("Sin falla", f"{len(df)-fallas:,}")

    st.markdown("### Vista previa del dataset")
    st.dataframe(df.head(10), use_container_width=True, height=240)

    c1, c2 = st.columns(2)

    # Distribución de clase
    with c1:
        st.markdown("#### Distribución de Falla_Detectada")
        conteo = df["Falla_Detectada"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 4), facecolor="#0f1117")
        ax.set_facecolor("#161b27")
        colors = ["#38bdf8", "#f87171"]
        ax.pie(
            conteo.values,
            labels=["Sin Falla", "Falla"],
            autopct="%1.1f%%",
            colors=colors,
            textprops={"color": "#cbd5e1", "fontsize": 11},
            wedgeprops={"linewidth": 2, "edgecolor": "#0f1117"},
        )
        ax.set_title("Proporción de clases", color="#f0f4ff", fontsize=12, pad=12)
        st.pyplot(fig, use_container_width=True)

    # Heatmap correlación
    with c2:
        st.markdown("#### Correlación de variables numéricas")
        numericas = df.select_dtypes(include=np.number).drop(
            columns=["ID_Registro", "Falla_Detectada"], errors="ignore"
        )
        fig2, ax2 = plt.subplots(figsize=(5, 4), facecolor="#0f1117")
        ax2.set_facecolor("#161b27")
        sns.heatmap(
            numericas.corr(),
            ax=ax2,
            cmap="Blues",
            annot=True,
            fmt=".2f",
            linewidths=0.4,
            linecolor="#0f1117",
            annot_kws={"size": 7, "color": "#f0f4ff"},
            cbar_kws={"shrink": 0.8},
        )
        ax2.tick_params(colors="#94a3b8", labelsize=7)
        ax2.set_title("Mapa de correlación", color="#f0f4ff", fontsize=12, pad=8)
        st.pyplot(fig2, use_container_width=True)

    # Estadísticas descriptivas
    st.markdown("### Estadísticas descriptivas")
    st.dataframe(df.describe().round(2), use_container_width=True)

# ══════════════════════════════════════════════
#  TAB 2 — ENTRENAMIENTO
# ══════════════════════════════════════════════
with tab_modelo:
    st.markdown("### Entrenamiento del modelo")
    st.markdown(
        "El pipeline aplica **StandardScaler → SMOTE → XGBoost**. "
        "Si activás GridSearchCV se optimizan hiperparámetros con validación cruzada (5 fold)."
    )

    if st.button("🚀 Entrenar modelo", type="primary"):
        df = df_raw.copy()
        df["Diferencia_Temp"] = df["Temp_Proceso_C"] - df["Temp_Ambiente_C"]

        drop_cols = ["ID_Registro", "Falla_Detectada", "Codigo_Producto", "Categoria"]
        drop_cols = [c for c in drop_cols if c in df.columns]
        X = df.drop(columns=drop_cols)
        y = df["Falla_Detectada"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        pipeline_xgb = ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote",  SMOTE(random_state=42)),
            ("xgb",    XGBClassifier(
                random_state=42,
                eval_metric="logloss",
                verbosity=0,
            )),
        ])

        param_grid = {
            "xgb__n_estimators": n_estimators,
            "xgb__max_depth":    max_depth,
            "xgb__learning_rate": lr_vals,
        }

        with st.spinner("Entrenando… esto puede tardar unos segundos."):
            if usar_grid:
                modelo_final = GridSearchCV(
                    estimator=pipeline_xgb,
                    param_grid=param_grid,
                    cv=5,
                    scoring="f1",
                    n_jobs=-1,
                )
            else:
                modelo_final = pipeline_xgb

            modelo_final.fit(X_train, y_train)
            y_pred = modelo_final.predict(X_test)

        # Guardar en session
        st.session_state["modelo"]   = modelo_final
        st.session_state["features"] = list(X.columns)
        st.session_state["X_test"]   = X_test
        st.session_state["y_test"]   = y_test
        st.session_state["y_pred"]   = y_pred

        if usar_grid:
            st.success(f"✅ Mejores parámetros: `{modelo_final.best_params_}`")
        else:
            st.success("✅ Modelo entrenado con parámetros por defecto.")

    # ── Resultados ──
    if "modelo" in st.session_state:
        y_test = st.session_state["y_test"]
        y_pred = st.session_state["y_pred"]
        modelo = st.session_state["modelo"]

        reporte = classification_report(y_test, y_pred, output_dict=True)

        st.markdown("---")
        st.markdown("### Métricas de evaluación")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy",  f"{reporte['accuracy']:.3f}")
        c2.metric("Precision (falla)", f"{reporte['1']['precision']:.3f}")
        c3.metric("Recall (falla)",    f"{reporte['1']['recall']:.3f}")
        c4.metric("F1-Score (falla)",  f"{reporte['1']['f1-score']:.3f}")

        left, right = st.columns(2)

        # Matriz de confusión
        with left:
            st.markdown("#### Matriz de confusión")
            est = modelo.best_estimator_ if hasattr(modelo, "best_estimator_") else modelo
            clases = est.classes_ if hasattr(est, "classes_") else [0, 1]
            matriz = confusion_matrix(y_test, y_pred, labels=clases)

            fig3, ax3 = plt.subplots(figsize=(5, 4.5), facecolor="#0f1117")
            ax3.set_facecolor("#161b27")
            disp = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=clases)
            disp.plot(cmap="Blues", values_format="d", ax=ax3)
            ax3.set_title("Matriz de confusión — XGBoost", color="#f0f4ff", fontsize=11, pad=10)
            ax3.tick_params(colors="#94a3b8")
            ax3.xaxis.label.set_color("#94a3b8")
            ax3.yaxis.label.set_color("#94a3b8")
            fig3.patch.set_facecolor("#0f1117")
            st.pyplot(fig3, use_container_width=True)

        # Importancia de features (XGBoost)
        with right:
            st.markdown("#### Importancia de variables")
            try:
                xgb_model = est.named_steps["xgb"]
                importancias = xgb_model.feature_importances_
                feats = st.session_state["features"]
                fi_df = pd.DataFrame({"Variable": feats, "Importancia": importancias})
                fi_df = fi_df.sort_values("Importancia", ascending=True)

                fig4, ax4 = plt.subplots(figsize=(5, 4.5), facecolor="#0f1117")
                ax4.set_facecolor("#161b27")
                colors_bar = ["#38bdf8" if v == fi_df["Importancia"].max() else "#4f6a8f"
                              for v in fi_df["Importancia"]]
                ax4.barh(fi_df["Variable"], fi_df["Importancia"], color=colors_bar, edgecolor="none", height=0.6)
                ax4.set_xlabel("Importancia", color="#94a3b8", fontsize=9)
                ax4.tick_params(colors="#94a3b8", labelsize=8)
                ax4.set_title("Feature Importance — XGBoost", color="#f0f4ff", fontsize=11, pad=10)
                ax4.spines["top"].set_visible(False)
                ax4.spines["right"].set_visible(False)
                ax4.spines["left"].set_color("#2a3045")
                ax4.spines["bottom"].set_color("#2a3045")
                fig4.patch.set_facecolor("#0f1117")
                st.pyplot(fig4, use_container_width=True)
            except Exception:
                st.info("Importancia de variables no disponible para este estimador.")

        # Reporte completo
        with st.expander("📋 Reporte de clasificación completo"):
            st.code(classification_report(y_test, y_pred, target_names=["Sin Falla", "Falla"]))

# ══════════════════════════════════════════════
#  TAB 3 — PREDICCIÓN INDIVIDUAL
# ══════════════════════════════════════════════
with tab_prediccion:
    st.markdown("### Predicción individual de falla")

    if "modelo" not in st.session_state:
        st.warning("⚠️ Primero entrenás el modelo en la pestaña **Entrenamiento del modelo**.")
    else:
        features = st.session_state["features"]

        # Rangos de referencia del dataset
        df_ref = df_raw.copy()
        df_ref["Diferencia_Temp"] = df_ref["Temp_Proceso_C"] - df_ref["Temp_Ambiente_C"]

        st.markdown("Ingresá los valores del sensor para predecir si hay falla:")

        cols_input = st.columns(3)
        valores = {}
        for i, feat in enumerate(features):
            col = cols_input[i % 3]
            mn  = float(df_ref[feat].min()) if feat in df_ref else 0.0
            mx  = float(df_ref[feat].max()) if feat in df_ref else 100.0
            med = float(df_ref[feat].median()) if feat in df_ref else (mn + mx) / 2
            valores[feat] = col.number_input(
                feat,
                min_value=round(mn - abs(mn) * 0.5, 2),
                max_value=round(mx + abs(mx) * 0.5, 2),
                value=round(med, 2),
                step=round((mx - mn) / 100, 4) if mx != mn else 0.1,
            )

        if st.button("🔍 Predecir", type="primary"):
            entrada = pd.DataFrame([valores])
            modelo  = st.session_state["modelo"]
            pred    = modelo.predict(entrada)[0]
            proba   = modelo.predict_proba(entrada)[0]

            p_falla = proba[1] if len(proba) > 1 else proba[0]
            p_ok    = 1 - p_falla

            if pred == 1:
                st.markdown(
                    f"<div class='alerta-falla'>"
                    f"<h3 style='color:#f87171;margin:0'>⚠️ FALLA DETECTADA</h3>"
                    f"<p style='color:#fca5a5;margin:8px 0 0'>Probabilidad de falla: <strong>{p_falla*100:.1f}%</strong></p>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div class='alerta-ok'>"
                    f"<h3 style='color:#34d399;margin:0'>✅ SIN FALLA</h3>"
                    f"<p style='color:#6ee7b7;margin:8px 0 0'>Probabilidad de falla: <strong>{p_falla*100:.1f}%</strong></p>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            # Gauge de probabilidad
            st.markdown("#### Probabilidad de falla")
            fig5, ax5 = plt.subplots(figsize=(7, 1.4), facecolor="#0f1117")
            ax5.set_facecolor("#0f1117")
            bar_color = "#f87171" if pred == 1 else "#34d399"
            ax5.barh([0], [p_falla], color=bar_color, height=0.5)
            ax5.barh([0], [1 - p_falla], left=[p_falla], color="#1e2740", height=0.5)
            ax5.set_xlim(0, 1)
            ax5.set_yticks([])
            ax5.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax5.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], color="#94a3b8", fontsize=9)
            ax5.spines[:].set_visible(False)
            ax5.axvline(0.5, color="#f59e0b", linewidth=1.5, linestyle="--", alpha=0.6)
            ax5.text(0.5, 0.65, "Umbral 50%", color="#f59e0b", fontsize=8,
                     ha="center", transform=ax5.transAxes)
            fig5.patch.set_facecolor("#0f1117")
            st.pyplot(fig5, use_container_width=True)

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#475569;font-size:.78rem'>"
    "Modelo Preventivo · Industria Neuquén · XGBoost + SMOTE + GridSearchCV"
    "</p>",
    unsafe_allow_html=True,
)
