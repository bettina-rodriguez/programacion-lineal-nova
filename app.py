import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import linprog

# ============================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================

st.set_page_config(
    page_title="Programación Lineal - Dualidad",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# CSS PERSONALIZADO
# ============================================================

st.markdown("""
<style>
    .main-title {
        font-size: 42px;
        font-weight: bold;
        text-align: center;
        color: #1a73e8;
        margin-bottom: 10px;
    }
    .subtitle {
        font-size: 18px;
        text-align: center;
        color: #666;
        margin-bottom: 30px;
    }
    .section-title {
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 15px;
        color: #1a73e8;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .result-box {
        background-color: #e8f5e9;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #4caf50;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #2196f3;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# TÍTULO PRINCIPAL
# ============================================================

st.markdown('<div class="main-title">📊 Programación Lineal</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Primal · Dual · Método Gráfico · Sensibilidad · Simplex</div>', unsafe_allow_html=True)

# ============================================================
# BARRA LATERAL - CONFIGURACIÓN
# ============================================================

with st.sidebar:
    st.header("⚙️ Configuración")
    
    tipo_problema = st.selectbox(
        "Tipo de problema",
        ["Maximización", "Minimización"]
    )
    
    num_variables = st.selectbox(
        "Número de variables",
        [2, 3, 4, 5],
        index=0
    )
    
    num_restricciones = st.selectbox(
        "Número de restricciones",
        [2, 3, 4, 5],
        index=0
    )
    
    st.markdown("---")
    
    st.subheader("📌 Métodos")
    mostrar_primal = st.checkbox("Primal", value=True)
    mostrar_dual = st.checkbox("Dual", value=True)
    mostrar_grafico = st.checkbox("Método gráfico", value=True)
    mostrar_sensibilidad = st.checkbox("Sensibilidad", value=True)
    mostrar_simplex = st.checkbox("Simplex paso a paso", value=False)
    mostrar_dos_fases = st.checkbox("Simplex Dos Fases", value=False)
    
    st.markdown("---")
    
    solucion_automatica = st.checkbox("Solución automática", value=True)

# ============================================================
# FUNCIÓN OBJETIVO
# ============================================================

st.markdown('<div class="section-title">🎯 Función objetivo</div>', unsafe_allow_html=True)
st.write("Ingrese los coeficientes:")

# Crear DataFrame para coeficientes
cols_obj = st.columns(num_variables)
coeficientes = []
for i, col in enumerate(cols_obj):
    with col:
        val = st.number_input(
            f"X{i+1}",
            value=40.0 if i == 0 else (30.0 if i == 1 else 0.0),
            step=1.0,
            key=f"c_{i}"
        )
        coeficientes.append(val)

# ============================================================
# RESTRICCIONES
# ============================================================

st.markdown('<div class="section-title">🔒 Restricciones</div>', unsafe_allow_html=True)

# Crear tablas para restricciones
restricciones = []
for r in range(num_restricciones):
    st.write(f"### Restricción {r+1}")
    cols_rest = st.columns(num_variables + 2)
    
    fila = []
    for v in range(num_variables):
        with cols_rest[v]:
            val = st.number_input(
                f"X{v+1}",
                value=2.0 if (r == 0 and v == 0) else (1.0 if (r == 0 and v == 1) else (1.0 if (r == 1 and v == 0) else 2.0)),
                step=1.0,
                key=f"a_{r}_{v}"
            )
            fila.append(val)
    
    with cols_rest[num_variables]:
        signo = st.selectbox(
            "Signo",
            ["≤", "=", "≥"],
            index=0,
            key=f"signo_{r}"
        )
    
    with cols_rest[num_variables + 1]:
        resultado = st.number_input(
            "Valor",
            value=100.0 if r == 0 else 80.0,
            step=1.0,
            key=f"b_{r}"
        )
    
    restricciones.append({
        'coeficientes': fila,
        'signo': signo,
        'resultado': resultado
    })

# ============================================================
# MODELO MATEMÁTICO
# ============================================================

st.markdown('<div class="section-title">📐 Modelo matemático</div>', unsafe_allow_html=True)

# Construir función objetivo
if tipo_problema == "Maximización":
    obj_text = f"MAX Z = "
else:
    obj_text = f"MIN Z = "

for i, c in enumerate(coeficientes):
    if i > 0 and c >= 0:
        obj_text += f" + "
    if c < 0:
        obj_text += f" - "
    obj_text += f"{abs(c):g}X{i+1}"

st.latex(obj_text)

# Construir restricciones
for r in restricciones:
    rest_text = ""
    for i, coef in enumerate(r['coeficientes']):
        if i > 0 and coef >= 0:
            rest_text += " + "
        if coef < 0:
            rest_text += " - "
        rest_text += f"{abs(coef):g}X{i+1}"
    rest_text += f" {r['signo']} {r['resultado']:g}"
    st.latex(rest_text)

# No negatividad
st.latex(r"X_1, X_2, \ldots, X_n \geq 0")

# ============================================================
# BOTÓN RESOLVER
# ============================================================

if st.button("🔴 RESOLVER PROBLEMA", use_container_width=True, type="primary"):
    
    # Preparar datos
    c = np.array(coeficientes)
    A = np.array([r['coeficientes'] for r in restricciones])
    b = np.array([r['resultado'] for r in restricciones])
    
    # Manejar signos
    A_ub = []
    b_ub = []
    A_eq = []
    b_eq = []
    
    for i, r in enumerate(restricciones):
        if r['signo'] == '≤':
            A_ub.append(r['coeficientes'])
            b_ub.append(r['resultado'])
        elif r['signo'] == '≥':
            A_ub.append([-x for x in r['coeficientes']])
            b_ub.append(-r['resultado'])
        else:  # '='
            A_eq.append(r['coeficientes'])
            b_eq.append(r['resultado'])
    
    # Resolver
    if tipo_problema == "Maximización":
        resultado = linprog(
            -c,
            A_ub=np.array(A_ub) if A_ub else None,
            b_ub=np.array(b_ub) if b_ub else None,
            A_eq=np.array(A_eq) if A_eq else None,
            b_eq=np.array(b_eq) if b_eq else None,
            bounds=[(0, None)] * len(c),
            method="highs"
        )
        
        if resultado.success:
            z = -resultado.fun
        else:
            z = None
    else:
        resultado = linprog(
            c,
            A_ub=np.array(A_ub) if A_ub else None,
            b_ub=np.array(b_ub) if b_ub else None,
            A_eq=np.array(A_eq) if A_eq else None,
            b_eq=np.array(b_eq) if b_eq else None,
            bounds=[(0, None)] * len(c),
            method="highs"
        )
        if resultado.success:
            z = resultado.fun
        else:
            z = None
    
    # ============================================================
    # MOSTRAR RESULTADOS
    # ============================================================
    
    if resultado.success:
        st.success("✅ Problema resuelto exitosamente!")
        
        # Crear pestañas
        tabs = []
        if mostrar_primal:
            tabs.append("📊 Primal")
        if mostrar_dual:
            tabs.append("🔄 Dual")
        if mostrar_grafico and num_variables <= 2:
            tabs.append("📐 Método Gráfico")
        if mostrar_sensibilidad:
            tabs.append("📈 Sensibilidad")
        if mostrar_simplex:
            tabs.append("📝 Simplex")
        if mostrar_dos_fases:
            tabs.append("⚡ Dos Fases")
        
        tab_objects = st.tabs(tabs)
        tab_index = 0
        
        # ============================================================
        # TAB 1 - PRIMAL
        # ============================================================
        
        if mostrar_primal:
            with tab_objects[tab_index]:
                st.header("📊 Solución del Problema Primal")
                
                # Mostrar variables
                cols = st.columns(min(len(resultado.x), 4))
                for i, (col, val) in enumerate(zip(cols, resultado.x)):
                    with col:
                        st.metric(f"X{i+1}", f"{val:.4f}")
                
                # Mostrar valor óptimo
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                if tipo_problema == "Maximización":
                    st.metric("Valor Óptimo Z*", f"{z:.4f}", delta=f"Máximo")
                else:
                    st.metric("Valor Óptimo Z*", f"{z:.4f}", delta=f"Mínimo")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Mostrar uso de recursos
                st.subheader("📊 Uso de Recursos")
                uso = A @ resultado.x
                data = {
                    "Restricción": [f"R{i+1}" for i in range(len(uso))],
                    "Disponible": b,
                    "Utilizado": uso,
                    "Holgura": b - uso
                }
                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True)
        
        # ============================================================
        # TAB 2 - DUAL
        # ============================================================
        
        if mostrar_dual:
            if mostrar_primal:
                tab_index += 1
            with tab_objects[tab_index]:
                st.header("🔄 Construcción del Problema Dual")
                
                # Construir dual
                if tipo_problema == "Maximización":
                    # Primal: MAX cX, Ax <= b, X >= 0
                    # Dual: MIN bY, A^T Y >= c, Y >= 0
                    dual_c = b
                    dual_A = -A.T
                    dual_b = -c
                else:
                    # Primal: MIN cX, Ax >= b, X >= 0
                    # Dual: MAX bY, A^T Y <= c, Y >= 0
                    dual_c = -b
                    dual_A = A.T
                    dual_b = c
                
                # Resolver dual
                if tipo_problema == "Maximización":
                    dual_result = linprog(
                        dual_c,
                        A_ub=dual_A,
                        b_ub=dual_b,
                        bounds=[(0, None)] * len(dual_c),
                        method="highs"
                    )
                    if dual_result.success:
                        w = dual_result.fun
                        y = dual_result.x
                    else:
                        w = None
                        y = None
                else:
                    dual_result = linprog(
                        -dual_c,
                        A_ub=dual_A,
                        b_ub=dual_b,
                        bounds=[(0, None)] * len(dual_c),
                        method="highs"
                    )
                    if dual_result.success:
                        w = -dual_result.fun
                        y = dual_result.x
                    else:
                        w = None
                        y = None
                
                if y is not None:
                    # Mostrar variables duales
                    cols = st.columns(min(len(y), 4))
                    for i, (col, val) in enumerate(zip(cols, y)):
                        with col:
                            st.metric(f"Y{i+1}", f"{val:.4f}")
                    
                    # Mostrar valor óptimo dual
                    if w is not None:
                        st.metric("Valor Óptimo W*", f"{w:.4f}")
                    
                    # Verificar dualidad fuerte
                    if z is not None and w is not None and np.isclose(z, w):
                        st.success(f"✅ Se cumple la dualidad fuerte: Z* = W* = {z:.4f}")
                    else:
                        st.warning("⚠️ No se cumple la dualidad fuerte")
        
        # ============================================================
        # TAB 3 - GRÁFICO
        # ============================================================
        
        if mostrar_grafico and num_variables <= 2:
            if (mostrar_primal or mostrar_dual):
                tab_index += 1
            with tab_objects[tab_index]:
                st.header("📐 Método Gráfico")
                
                # Solo para 2 variables
                if len(resultado.x) == 2:
                    x1, x2 = resultado.x
                    
                    # Encontrar límites
                    x_max = max(10, x1 * 1.5)
                    y_max = max(10, x2 * 1.5)
                    
                    # Crear gráfico
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    # Dibujar restricciones
                    x_vals = np.linspace(0, x_max, 100)
                    
                    for i, r in enumerate(restricciones):
                        a1, a2 = r['coeficientes'][0], r['coeficientes'][1]
                        b_val = r['resultado']
                        
                        if a2 != 0:
                            y_vals = (b_val - a1 * x_vals) / a2
                            y_vals = np.where(y_vals >= 0, y_vals, np.nan)
                            ax.plot(x_vals, y_vals, label=f"R{i+1}: {a1:g}X1 + {a2:g}X2 {r['signo']} {b_val:g}")
                    
                    # Marcar punto óptimo
                    ax.scatter(x1, x2, color='red', s=200, zorder=5, label=f"Óptimo ({x1:.2f}, {x2:.2f})")
                    
                    # Configurar gráfico
                    ax.set_xlim(0, x_max)
                    ax.set_ylim(0, y_max)
                    ax.set_xlabel("X1")
                    ax.set_ylabel("X2")
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    ax.set_title("Región Factible y Punto Óptimo")
                    
                    st.pyplot(fig)
        
        # ============================================================
        # TAB 4 - SENSIBILIDAD
        # ============================================================
        
        if mostrar_sensibilidad:
            if (mostrar_primal or mostrar_dual or (mostrar_grafico and num_variables <= 2)):
                tab_index += 1
            with tab_objects[tab_index]:
                st.header("📈 Análisis de Sensibilidad")
                
                if mostrar_dual and y is not None:
                    st.subheader("Precios Sombra")
                    
                    cols = st.columns(min(len(y), 4))
                    for i, (col, val) in enumerate(zip(cols, y)):
                        with col:
                            st.metric(f"Y{i+1}", f"{val:.4f}")
                    
                    st.info("""
                    **Interpretación de los precios sombra:**
                    
                    Los precios sombra (Y) indican cuánto aumentaría el valor óptimo Z si se incrementara en una unidad el recurso correspondiente.
                    """)
        
        # ============================================================
        # TAB 5 - SIMPLEX
        # ============================================================
        
        if mostrar_simplex:
            tab_index += 1
            with tab_objects[tab_index]:
                st.header("📝 Simplex Paso a Paso")
                st.info("⏳ Esta funcionalidad está en desarrollo. Próximamente disponible.")
        
        # ============================================================
        # TAB 6 - DOS FASES
        # ============================================================
        
        if mostrar_dos_fases:
            tab_index += 1
            with tab_objects[tab_index]:
                st.header("⚡ Simplex Dos Fases")
                st.info("⏳ Esta funcionalidad está en desarrollo. Próximamente disponible.")
    
    else:
        st.error("❌ No se pudo resolver el problema. Verifica que los datos sean correctos.")
        st.write("Mensaje de error:", resultado.message)

else:
    # Mensaje inicial
    st.info("""
    👈 **Configura tu problema** en la barra lateral y completa los datos del problema.
    
    Luego presiona el botón **'RESOLVER PROBLEMA'** para ver los resultados.
    
    **Valores por defecto:**
    - MAX Z = 40X₁ + 30X₂
    - Sujeto a: 2X₁ + X₂ ≤ 100, X₁ + 2X₂ ≤ 80, X₁, X₂ ≥ 0
    """)