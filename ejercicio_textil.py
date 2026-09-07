# app_pl_dinamica.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pulp

# =============================================
# CONFIGURACIÓN DE LA PÁGINA
# =============================================
st.set_page_config(
    page_title="Investigación de Operaciones - Grupo Nova",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================
# ESTILOS CSS (LETRAS VISIBLES)
# =============================================
st.markdown("""
<style>
    .stApp { background-color: #f0f2f6; }
    
    /* === CARÁTULA === */
    .main-header {
        background: linear-gradient(135deg, #0f1a3a, #1a2a6c, #2d4373);
        padding: 35px 30px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 6px 12px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,215,0,0.2);
    }
    .main-header h1 {
        color: #ffffff !important;
        font-size: 2.8em;
        margin: 0;
        font-weight: bold;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    .main-header .subtitulo {
        color: #ffd700 !important;
        font-size: 1.2em;
        margin: 8px 0 0 0;
    }
    .main-header .tema {
        color: #ffffff !important;
        font-size: 1.1em;
        margin: 12px 0 0 0;
        background: rgba(255,215,0,0.15);
        padding: 8px 20px;
        border-radius: 20px;
        display: inline-block;
        border: 1px solid rgba(255,215,0,0.3);
    }
    
    /* === INTEGRANTES === */
    .integrantes-card {
        background: linear-gradient(135deg, #1a2a6c, #2d4373) !important;
        padding: 18px 25px;
        border-radius: 10px;
        margin-bottom: 20px;
        border-left: 5px solid #ffd700;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    }
    .integrantes-card .titulo {
        color: #ffd700 !important;
        font-size: 1.05em;
        font-weight: bold;
        margin-bottom: 8px;
        text-align: center;
    }
    .integrantes-card .nombre {
        color: #ffffff !important;
        font-size: 0.95em;
        margin: 2px 0;
        text-align: center;
    }
    .integrantes-card .docente {
        color: #87CEEB !important;
        font-size: 0.95em;
        margin: 5px 0 2px 0;
        text-align: center;
    }
    .integrantes-card .año {
        color: #cccccc !important;
        font-size: 0.9em;
        margin: 2px 0 0 0;
        text-align: center;
    }
    
    /* === TÍTULOS === */
    .section-title {
        color: #1a2a6c !important;
        font-size: 1.8em;
        font-weight: bold;
        border-bottom: 3px solid #ffd700;
        padding-bottom: 10px;
        margin: 30px 0 20px 0;
    }
    .subsection-title {
        color: #2c3e50 !important;
        font-size: 1.3em;
        font-weight: bold;
        margin: 20px 0 15px 0;
    }
    
    /* === TARJETAS === */
    .data-card {
        background: white !important;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
        text-align: center;
        height: 100%;
    }
    .data-card .label {
        color: #555 !important;
        font-size: 0.85em;
        font-weight: 600;
        text-transform: uppercase;
    }
    .data-card .value {
        color: #1a2a6c !important;
        font-size: 1.1em;
        font-weight: bold;
        margin: 8px 0;
    }
    
    .result-card {
        background: white !important;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 5px solid #27ae60;
        text-align: center;
        height: 100%;
    }
    .result-card .value {
        color: #1a2a6c !important;
        font-size: 2em;
        font-weight: bold;
    }
    .result-card .label {
        color: #555 !important;
        font-size: 0.9em;
        font-weight: 500;
    }
    
    /* === TABLA === */
    .sensibilidad-table {
        background: white !important;
        border-collapse: collapse;
        width: 100%;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .sensibilidad-table th {
        background: #1a2a6c !important;
        color: white !important;
        padding: 12px;
        text-align: center;
        font-weight: 600;
    }
    .sensibilidad-table td {
        background: white !important;
        color: #1a1a1a !important;
        padding: 10px;
        text-align: center;
        border-bottom: 1px solid #eee;
    }
    .sensibilidad-table tr:hover td {
        background: #f8f9fa !important;
    }
    
    /* === CAJA DE ANÁLISIS === */
    .analysis-box {
        background: white !important;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #e67e22;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin: 15px 0;
    }
    .analysis-box p {
        color: #1a1a1a !important;
        font-size: 1.05em;
        line-height: 1.6;
    }
    .analysis-box strong {
        color: #1a2a6c !important;
    }
    
    /* === BOTÓN === */
    .stButton > button {
        background: #1a2a6c !important;
        color: white !important;
        font-weight: bold;
        font-size: 1.2em;
        border-radius: 8px;
        padding: 12px 40px;
        border: none;
        width: 100%;
    }
    .stButton > button:hover {
        background: #2d4373 !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(26,42,108,0.3);
    }
    
    /* === PIE DE PÁGINA === */
    .footer {
        background: linear-gradient(135deg, #0f1a3a, #1a2a6c) !important;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-top: 40px;
    }
    .footer p {
        color: white !important;
        margin: 3px 0;
        font-size: 0.95em;
    }
    .footer .año {
        color: #ffd700 !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# =============================================
# CARÁTULA
# =============================================
st.markdown("""
<div class="main-header">
    <h1>🧵 INVESTIGACIÓN DE OPERACIONES</h1>
    <div class="subtitulo">PROGRAMACIÓN LINEAL - GRUPO NOVA</div>
    <div class="tema">📌 Resolución de ejercicio grupal - Industrias del Norte S.A.C.</div>
</div>
""", unsafe_allow_html=True)

# =============================================
# INTEGRANTES
# =============================================
st.markdown("""
<div class="integrantes-card">
    <div class="titulo">👥 INTEGRANTES</div>
    <div class="nombre">• Katty Trujillo Santiago</div>
    <div class="nombre">• Michel Fonseca Guevara</div>
    <div class="nombre">• Jeiser Pinchi Lomas</div>
    <div class="nombre">• Yube Young Ruiz</div>
    <div class="nombre">• Leonardo Ortiz Magallanes</div>
    <div class="nombre">• Bettina Rodríguez Sanchez</div>
    <div class="docente">👨‍🏫 Docente: Mg. Enrique Jannier Boy Vasquez</div>
    <div class="año">📅 Año: 2026</div>
</div>
""", unsafe_allow_html=True)

# =============================================
# CONFIGURACIÓN DINÁMICA
# =============================================
st.markdown('<div class="section-title">⚙️ CONFIGURACIÓN DEL PROBLEMA</div>', unsafe_allow_html=True)

# Tipo de problema
col_tipo1, col_tipo2 = st.columns(2)
with col_tipo1:
    tipo_problema = st.radio("🎯 Tipo:", ["Maximización", "Minimización"], index=0, horizontal=True)
with col_tipo2:
    variables_enteras = st.checkbox("🔢 Variables enteras", value=True)

# Número de variables y restricciones
col_n1, col_n2 = st.columns(2)
with col_n1:
    n_variables = st.number_input("📊 Variables:", min_value=2, max_value=10, value=2, step=1)
with col_n2:
    n_restricciones = st.number_input("📋 Restricciones:", min_value=1, max_value=10, value=3, step=1)

# Función objetivo
st.markdown("### 📈 FUNCIÓN OBJETIVO")
cols_obj = st.columns(n_variables)
coef_objetivo = []
for i in range(n_variables):
    with cols_obj[i]:
        default = 35.0 if i == 0 else 50.0 if i == 1 else 0.0
        val = st.number_input(f"Coef X{i+1}", value=default, step=1.0, format="%.2f", key=f"obj_{i}")
        coef_objetivo.append(val)

# Restricciones
st.markdown("### ⛓️ RESTRICCIONES")
restricciones = []
for r in range(n_restricciones):
    st.markdown(f"**Restricción {r+1}:**")
    cols_r = st.columns(n_variables + 1)
    coefs = []
    for i in range(n_variables):
        with cols_r[i]:
            if r == 0:
                default = 2.0 if i == 0 else 3.0
            elif r == 1:
                default = 1.0 if i == 0 else 2.0
            else:
                default = 3.0 if i == 0 else 2.0
            val = st.number_input(f"X{i+1}", value=default, step=0.5, format="%.2f", key=f"r{r}_c{i}")
            coefs.append(val)
    with cols_r[-1]:
        disp = st.number_input("Disponibilidad", value=240.0 - r*30, step=10.0, format="%.2f", key=f"r{r}_d")
    restricciones.append({"coefs": coefs, "disponibilidad": disp})

# Nombres de variables
st.markdown("### 🏷️ NOMBRES")
cols_nombres = st.columns(n_variables)
nombres_variables = []
for i in range(n_variables):
    with cols_nombres[i]:
        nombre = st.text_input(f"Var {i+1}", value=f"X{i+1}", key=f"nom_{i}")
        nombres_variables.append(nombre)

# =============================================
# MODELO MATEMÁTICO (PRIMAL)
# =============================================
st.markdown('<div class="section-title">📐 MODELO MATEMÁTICO (PRIMAL)</div>', unsafe_allow_html=True)

modelo_text = f"**{'MAX' if tipo_problema == 'Maximización' else 'MIN'}** Z = "
modelo_text += " + ".join([f"{coef_objetivo[i]}·{nombres_variables[i]}" for i in range(n_variables)])
modelo_text += "\n\n**Sujeto a:**\n"
for r in range(n_restricciones):
    coefs = restricciones[r]["coefs"]
    disp = restricciones[r]["disponibilidad"]
    terms = []
    for i in range(n_variables):
        if coefs[i] != 0:
            sign = "+" if coefs[i] > 0 else ""
            terms.append(f"{sign}{coefs[i]}·{nombres_variables[i]}")
    modelo_text += "  " + " ".join(terms) + f" ≤ {disp}\n"
modelo_text += f"\n{', '.join(nombres_variables)} ≥ 0"
if variables_enteras:
    modelo_text += " y ENTEROS"

st.markdown(f"```\n{modelo_text}\n```")

# =============================================
# MODELO DUAL
# =============================================
st.markdown('<div class="section-title">🔄 MODELO DUAL</div>', unsafe_allow_html=True)

# Generar modelo dual automáticamente
dual_text = f"**{'MIN' if tipo_problema == 'Maximización' else 'MAX'}** W = "
dual_text += " + ".join([f"{restricciones[r]['disponibilidad']}·Y{r+1}" for r in range(n_restricciones)])
dual_text += "\n\n**Sujeto a:**\n"
for i in range(n_variables):
    terms = []
    for r in range(n_restricciones):
        coef = restricciones[r]["coefs"][i]
        if coef != 0:
            sign = "+" if coef > 0 else ""
            terms.append(f"{sign}{coef}·Y{r+1}")
    dual_text += "  " + " ".join(terms) + f" ≥ {coef_objetivo[i]}\n"
dual_text += f"\nY₁, Y₂, ..., Y{n_restricciones} ≥ 0"

st.markdown(f"```\n{dual_text}\n```")

# =============================================
# FUNCIÓN RESOLVER
# =============================================
def resolver_problema():
    if tipo_problema == "Maximización":
        problema = pulp.LpProblem("PL_Dinamico", pulp.LpMaximize)
    else:
        problema = pulp.LpProblem("PL_Dinamico", pulp.LpMinimize)
    
    variables = []
    for i in range(n_variables):
        if variables_enteras:
            var = pulp.LpVariable(nombres_variables[i], lowBound=0, cat='Integer')
        else:
            var = pulp.LpVariable(nombres_variables[i], lowBound=0)
        variables.append(var)
    
    problema += sum(coef_objetivo[i] * variables[i] for i in range(n_variables))
    
    for r in range(n_restricciones):
        coefs = restricciones[r]["coefs"]
        disp = restricciones[r]["disponibilidad"]
        problema += sum(coefs[i] * variables[i] for i in range(n_variables)) <= disp
    
    problema.solve()
    
    resultados = {
        "variables": [var.varValue for var in variables],
        "z": pulp.value(problema.objective),
        "status": problema.status,
        "recursos_usados": []
    }
    
    for r in range(n_restricciones):
        coefs = restricciones[r]["coefs"]
        usado = sum(coefs[i] * resultados["variables"][i] for i in range(n_variables))
        disp = restricciones[r]["disponibilidad"]
        resultados["recursos_usados"].append({
            "usado": usado,
            "sobrante": disp - usado,
            "disponible": disp
        })
    
    return resultados

# =============================================
# BOTÓN RESOLVER
# =============================================
if st.button("🚀 RESOLVER PROBLEMA", use_container_width=True):
    with st.spinner('🔄 Resolviendo...'):
        resultado = resolver_problema()
    
    # =========================================
    # RESULTADOS
    # =========================================
    st.markdown('<div class="section-title">📊 RESULTADOS</div>', unsafe_allow_html=True)
    
    cols_res = st.columns(min(n_variables, 3))
    for i in range(n_variables):
        idx = i % 3
        with cols_res[idx]:
            st.markdown(f"""
            <div class="result-card">
                <div class="label">{nombres_variables[i]}</div>
                <div class="value">{resultado['variables'][i]:.0f if variables_enteras else resultado['variables'][i]:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
    
    col_z = st.columns(1)[0]
    with col_z:
        st.markdown(f"""
        <div class="result-card" style='border-left-color: #e67e22;'>
            <div class="label">💰 {'Utilidad' if tipo_problema == 'Maximización' else 'Costo'} Máxima</div>
            <div class="value">{resultado['z']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # =========================================
    # ANÁLISIS DE SENSIBILIDAD
    # =========================================
    st.markdown('<div class="section-title">📋 ANÁLISIS DE SENSIBILIDAD</div>', unsafe_allow_html=True)
    
    # Tabla de recursos
    tabla_html = """
    <table class="sensibilidad-table">
        <thead>
            <tr>
                <th>Recurso</th>
                <th>Disponible</th>
                <th>Utilizado</th>
                <th>Sobrante</th>
                <th>Estado</th>
            </tr>
        </thead>
        <tbody>
    """
    for r in range(n_restricciones):
        disp = restricciones[r]["disponibilidad"]
        usado = resultado["recursos_usados"][r]["usado"]
        sobrante = resultado["recursos_usados"][r]["sobrante"]
        estado = "✅ Sobrante" if sobrante > 0 else "⚠️ Usado completo"
        color = "green" if sobrante > 0 else "red"
        tabla_html += f"""
        <tr>
            <td><b>Recurso {r+1}</b></td>
            <td>{disp:.2f}</td>
            <td>{usado:.2f}</td>
            <td>{sobrante:.2f}</td>
            <td><span style='color:{color}; font-weight:bold;'>{estado}</span></td>
        </tr>
        """
    tabla_html += "</tbody></table>"
    st.markdown(tabla_html, unsafe_allow_html=True)
    
    # =========================================
    # PRECIOS SOMBRA
    # =========================================
    st.markdown('<div class="subsection-title">💲 PRECIOS SOMBRA</div>', unsafe_allow_html=True)
    
    for r in range(n_restricciones):
        sobrante = resultado["recursos_usados"][r]["sobrante"]
        if sobrante > 0:
            precio = "S/ 0.00"
            interpretacion = "Recurso sobrante → Aumentarlo NO mejora la utilidad"
        else:
            precio = "S/ 0.00 (estimado)"
            interpretacion = "Recurso usado completo → Aumentarlo podría mejorar la utilidad"
        
        st.markdown(f"""
        <div style='background: white; padding: 12px; border-radius: 8px; margin: 5px 0; border-left: 4px solid #3498db;'>
            <b>Recurso {r+1}:</b> 
            <span style='color: #1a2a6c; font-weight: bold;'>{precio}</span>
            <span style='color: #555; font-size: 0.9em;'> → {interpretacion}</span>
        </div>
        """, unsafe_allow_html=True)
    
    # =========================================
    # MÉTODO GRÁFICO (solo 2 variables)
    # =========================================
    if n_variables == 2:
        st.markdown('<div class="section-title">📈 MÉTODO GRÁFICO</div>', unsafe_allow_html=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        x_vals = np.linspace(0, max(100, max([r["disponibilidad"] for r in restricciones])), 200)
        
        for r in range(n_restricciones):
            coefs = restricciones[r]["coefs"]
            disp = restricciones[r]["disponibilidad"]
            if coefs[1] != 0:
                y = (disp - coefs[0] * x_vals) / coefs[1]
                ax.plot(x_vals, y, label=f'Recurso {r+1}', linewidth=2)
        
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.axvline(x=0, color='black', linewidth=0.5)
        
        y_min = np.full_like(x_vals, np.inf)
        for r in range(n_restricciones):
            coefs = restricciones[r]["coefs"]
            disp = restricciones[r]["disponibilidad"]
            if coefs[1] != 0:
                y = (disp - coefs[0] * x_vals) / coefs[1]
                y_min = np.minimum(y_min, y)
        ax.fill_between(x_vals, 0, y_min, where=(y_min > 0), alpha=0.2, color='gray')
        
        x1 = resultado['variables'][0]
        x2 = resultado['variables'][1] if n_variables > 1 else 0
        ax.plot(x1, x2, 'go', markersize=14, label=f'Óptimo', zorder=5)
        
        ax.set_xlabel(nombres_variables[0], fontsize=12)
        ax.set_ylabel(nombres_variables[1] if n_variables > 1 else "X₂", fontsize=12)
        ax.set_title('REGIÓN FACTIBLE Y PUNTO ÓPTIMO', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim(0, max(90, max([r["disponibilidad"] for r in restricciones]) * 0.8))
        ax.set_ylim(0, max(90, max([r["disponibilidad"] for r in restricciones]) * 0.8))
        
        st.pyplot(fig)
    
    # =========================================
    # MENSAJE DE ÉXITO
    # =========================================
    st.success(f"✅ Problema resuelto correctamente! {'Utilidad' if tipo_problema == 'Maximización' else 'Costo'} = {resultado['z']:.2f}")

# =============================================
# PIE DE PÁGINA
# =============================================
st.markdown("""
<div class="footer">
    <p>🧵 INVESTIGACIÓN DE OPERACIONES - GRUPO NOVA</p>
    <p style='font-size:0.85em; opacity:0.8;'>Resolución de ejercicio grupal - Industrias del Norte S.A.C.</p>
    <p style='font-size:0.85em; opacity:0.8;'>Mg. Enrique Jannier Boy Vasquez - Docente</p>
    <p class="año">📅 2026</p>
</div>
""", unsafe_allow_html=True)