import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

# PuLP es el "Solver" principal del programa.
# SciPy se usa como apoyo para obtener información de sensibilidad
# y para el método gráfico cuando corresponde.
try:
    import pulp
    PULP_OK = True
except ImportError:
    PULP_OK = False

try:
    from scipy.optimize import linprog
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False


# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="Investigación de Operaciones - Grupo NOVA",
    page_icon="📊",
    layout="wide",
)

st.markdown("""
<style>
.main-title {
    font-size: 38px;
    font-weight: 800;
    text-align: center;
    margin-bottom: 4px;
}
.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 25px;
}
.card {
    padding: 18px;
    border-radius: 12px;
    border: 1px solid rgba(128,128,128,.25);
    margin-bottom: 12px;
}
.good {
    padding: 15px;
    border-radius: 10px;
    border-left: 5px solid #2e7d32;
    background: rgba(46,125,50,.10);
}
.warn {
    padding: 15px;
    border-radius: 10px;
    border-left: 5px solid #ef6c00;
    background: rgba(239,108,0,.10);
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📊 INVESTIGACIÓN DE OPERACIONES</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Programación Lineal · Grupo NOVA · Primal · Dual · Solver · Gráfico · Simplex · Sensibilidad</div>',
    unsafe_allow_html=True
)

with st.expander("👥 INTEGRANTES DEL GRUPO NOVA", expanded=True):
    cols_integrantes = st.columns(2)
    for i, person in enumerate([
        "Katty Trujillo Santiago",
        "Michel Fonseca Guevara",
        "Jeiser Pinchi Lomas",
        "Yube Young Ruiz",
        "Leonardo Ortiz Magallanes",
        "Bettina Rodríguez Sanchez",
    ]):
        with cols_integrantes[i % 2]:
            st.write(f"• {person}")
    st.write("👨‍🏫 **Docente:** Mg. Enrique Jannier Boy Vasquez")
    st.write("📅 **Año:** 2026")

# ============================================================
# DATOS DEL PROYECTO
# ============================================================

INTEGRANTES = [
    "Katty Trujillo Santiago",
    "Michel Fonseca Guevara",
    "Jeiser Pinchi Lomas",
    "Yube Young Ruiz",
    "Leonardo Ortiz Magallanes",
    "Bettina Rodríguez Sanchez",
]

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def clean_number(x):
    if abs(x) < 1e-9:
        return 0.0
    return float(x)


def fmt(x):
    if x is None:
        return "—"
    x = clean_number(x)
    if abs(x - round(x)) < 1e-8:
        return f"{int(round(x))}"
    return f"{x:.4f}"


def signed_term(coef, name, first=False):
    coef = float(coef)
    if abs(coef) < 1e-12:
        return ""
    if first:
        sign = "-" if coef < 0 else ""
        magnitude = abs(coef)
        return f"{sign}{fmt(magnitude)}{name}"
    sign = "+" if coef >= 0 else "-"
    return f" {sign} {fmt(abs(coef))}{name}"


def objective_text(c, sense, names):
    terms = []
    for i, coef in enumerate(c):
        term = signed_term(coef, names[i], first=(len(terms) == 0))
        if term:
            terms.append(term)
    return f"{'MAX' if sense == 'Maximización' else 'MIN'} Z = " + ("".join(terms) if terms else "0")


def constraint_text(row, names):
    terms = []
    for i, coef in enumerate(row["a"]):
        term = signed_term(coef, names[i], first=(len(terms) == 0))
        if term:
            terms.append(term)
    return "".join(terms) + f" {row['sign']} {fmt(row['b'])}"


def solve_with_pulp(c, constraints, sense, integer=False):
    """Resuelve el modelo con PuLP/CBC."""
    if not PULP_OK:
        return None, "PuLP no está instalado."

    prob_sense = pulp.LpMaximize if sense == "Maximización" else pulp.LpMinimize
    model = pulp.LpProblem("Programacion_Lineal_NOVA", prob_sense)

    var_cat = pulp.LpInteger if integer else pulp.LpContinuous
    x = [
        pulp.LpVariable(f"X{i+1}", lowBound=0, cat=var_cat)
        for i in range(len(c))
    ]

    model += pulp.lpSum(float(c[i]) * x[i] for i in range(len(c))), "Funcion_Objetivo"

    for r, row in enumerate(constraints):
        expr = pulp.lpSum(float(row["a"][i]) * x[i] for i in range(len(c)))
        if row["sign"] == "≤":
            model += expr <= float(row["b"]), f"R{r+1}"
        elif row["sign"] == "≥":
            model += expr >= float(row["b"]), f"R{r+1}"
        else:
            model += expr == float(row["b"]), f"R{r+1}"

    status = model.solve(pulp.PULP_CBC_CMD(msg=False))
    status_name = pulp.LpStatus[status]

    if status_name != "Optimal":
        return {
            "status": status_name,
            "x": None,
            "z": None,
            "model": model,
            "variables": x,
        }, status_name

    values = np.array([pulp.value(v) for v in x], dtype=float)
    z = float(pulp.value(model.objective))

    return {
        "status": status_name,
        "x": values,
        "z": z,
        "model": model,
        "variables": x,
    }, status_name


def build_scipy_matrices(c, constraints, sense):
    """Convierte restricciones a la forma que necesita scipy.linprog."""
    A_ub, b_ub, A_eq, b_eq = [], [], [], []

    for row in constraints:
        a = np.asarray(row["a"], dtype=float)
        b = float(row["b"])
        if row["sign"] == "≤":
            A_ub.append(a)
            b_ub.append(b)
        elif row["sign"] == "≥":
            A_ub.append(-a)
            b_ub.append(-b)
        else:
            A_eq.append(a)
            b_eq.append(b)

    return (
        np.array(A_ub) if A_ub else None,
        np.array(b_ub) if b_ub else None,
        np.array(A_eq) if A_eq else None,
        np.array(b_eq) if b_eq else None,
    )


def solve_continuous_scipy(c, constraints, sense):
    """Solución continua auxiliar, útil para sensibilidad y gráficos."""
    if not SCIPY_OK:
        return None

    A_ub, b_ub, A_eq, b_eq = build_scipy_matrices(c, constraints, sense)
    obj = -np.asarray(c, dtype=float) if sense == "Maximización" else np.asarray(c, dtype=float)

    return linprog(
        obj,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=[(0, None)] * len(c),
        method="highs",
    )


def shadow_prices(c, constraints, sense):
    """
    Obtiene precios sombra para el modelo continuo.
    Se presentan en el sentido administrativo:
    cambio aproximado de Z por una unidad adicional de RHS.
    """
    if not SCIPY_OK:
        return None

    result = solve_continuous_scipy(c, constraints, sense)
    if result is None or not result.success:
        return None

    prices = []

    # scipy devuelve marginales para las restricciones convertidas a A_ub <= b_ub.
    # Para maximización cambiamos el signo para expresar el valor marginal
    # de aumentar el recurso original.
    ub_index = 0
    eq_index = 0

    for row in constraints:
        if row["sign"] == "=":
            # Para igualdad, el marginal depende de la convención de scipy.
            # Se conserva el valor en la escala del modelo de minimización/maximización.
            marginal = result.eqlin.marginals[eq_index]
            if sense == "Maximización":
                marginal = -marginal
            prices.append(float(marginal))
            eq_index += 1
        else:
            marginal = result.ineqlin.marginals[ub_index]
            # La transformación >= -> -a <= -b invierte el signo del RHS.
            if row["sign"] == "≥":
                marginal = -marginal
            if sense == "Maximización":
                marginal = -marginal
            prices.append(float(marginal))
            ub_index += 1

    return np.array(prices)


def resource_table(c, constraints, x):
    rows = []
    for i, row in enumerate(constraints):
        used = float(np.dot(row["a"], x))
        b = float(row["b"])
        slack = b - used

        if row["sign"] == "≤":
            estado = "Disponible"
            holgura = slack
        elif row["sign"] == "≥":
            estado = "Exceso mínimo" if used >= b - 1e-8 else "Incumplida"
            holgura = used - b
        else:
            estado = "Activo" if abs(used - b) <= 1e-7 else "Incumplida"
            holgura = b - used

        rows.append({
            "Recurso / restricción": f"R{i+1}",
            "Tipo": row["sign"],
            "Disponible": b,
            "Utilizado": used,
            "Holgura": holgura,
            "Estado": estado,
        })
    return pd.DataFrame(rows)


def solve_rhs_sensitivity(c, constraints, sense, idx, delta):
    """Resuelve un cambio puntual del lado derecho."""
    new_constraints = []
    for j, row in enumerate(constraints):
        rr = dict(row)
        rr["a"] = list(row["a"])
        if j == idx:
            rr["b"] = row["b"] + delta
        new_constraints.append(rr)

    res = solve_continuous_scipy(c, new_constraints, sense)
    if res is None or not res.success:
        return None
    z = -res.fun if sense == "Maximización" else res.fun
    return float(z)


def approximate_rhs_range(c, constraints, sense, idx, max_steps=30):
    """
    Estima un rango de validez del precio sombra mediante perturbaciones.
    No pretende sustituir un reporte profesional de Solver; sirve como
    apoyo didáctico dentro de la aplicación.
    """
    base = solve_continuous_scipy(c, constraints, sense)
    if base is None or not base.success:
        return None, None

    base_z = -base.fun if sense == "Maximización" else base.fun
    base_b = float(constraints[idx]["b"])

    # Paso adaptativo.
    step = max(abs(base_b) * 0.05, 1.0)

    def marginal(delta):
        z = solve_rhs_sensitivity(c, constraints, sense, idx, delta)
        if z is None:
            return None
        return (z - base_z) / delta if abs(delta) > 1e-12 else None

    base_prices = shadow_prices(c, constraints, sense)
    if base_prices is None:
        return None, None

    target = base_prices[idx]

    lower = base_b
    upper = base_b

    # Hacia abajo
    last = target
    for k in range(1, max_steps + 1):
        delta = -step * k
        if base_b + delta < 0 and constraints[idx]["sign"] != "≥":
            break
        m = marginal(delta)
        if m is None or abs(m - target) > max(1e-4, abs(target) * 0.03):
            break
        lower = base_b + delta
        last = m

    # Hacia arriba
    for k in range(1, max_steps + 1):
        delta = step * k
        m = marginal(delta)
        if m is None or abs(m - target) > max(1e-4, abs(target) * 0.03):
            break
        upper = base_b + delta

    return lower, upper


def make_dual(c, constraints, sense, names):
    """
    Construye el dual para variables primales X >= 0.
    Para igualdad, la variable dual es libre.
    """
    m = len(constraints)
    dual_var_names = [f"Y{i+1}" for i in range(m)]

    # Signo de cada variable dual.
    dual_signs = []
    for row in constraints:
        if sense == "Maximización":
            if row["sign"] == "≤":
                dual_signs.append("≥ 0")
            elif row["sign"] == "≥":
                dual_signs.append("≤ 0")
            else:
                dual_signs.append("libre")
        else:
            if row["sign"] == "≥":
                dual_signs.append("≥ 0")
            elif row["sign"] == "≤":
                dual_signs.append("≤ 0")
            else:
                dual_signs.append("libre")

    dual_sense = "MIN" if sense == "Maximización" else "MAX"

    # Objetivo dual: b^T y
    b = [r["b"] for r in constraints]
    dual_obj_terms = []
    for i, bi in enumerate(b):
        dual_obj_terms.append(signed_term(bi, dual_var_names[i], first=(i == 0)))
    dual_obj = f"{dual_sense} W = " + "".join(dual_obj_terms)

    # Restricciones duales: A^T y >= c para MAX; <= c para MIN.
    dual_rows = []
    for j in range(len(c)):
        terms = []
        for i, row in enumerate(constraints):
            terms.append(
                signed_term(row["a"][j], dual_var_names[i], first=(len([t for t in terms if t]) == 0))
            )
        sign = "≥" if sense == "Maximización" else "≤"
        dual_rows.append(f"{''.join(terms)} {sign} {fmt(c[j])}")

    return {
        "sense": dual_sense,
        "objective": dual_obj,
        "constraints": dual_rows,
        "variable_signs": list(zip(dual_var_names, dual_signs)),
    }


def graph_2d(c, constraints, x_opt, sense):
    """Método gráfico para dos variables."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Determinar límites razonables.
    candidates = [10.0]
    for row in constraints:
        for coef, rhs in zip(row["a"], [row["b"], row["b"]]):
            if coef > 0:
                candidates.append(abs(rhs / coef))
    if x_opt is not None:
        candidates.extend([float(x_opt[0]) * 1.5 + 5, float(x_opt[1]) * 1.5 + 5])

    max_axis = max(candidates)
    max_axis = max(10, min(max_axis * 1.05, 10000))
    xvals = np.linspace(0, max_axis, 500)

    for i, row in enumerate(constraints):
        a1, a2 = row["a"]
        b = row["b"]
        if abs(a2) > 1e-12:
            yvals = (b - a1 * xvals) / a2
            yvals = np.where(yvals >= 0, yvals, np.nan)
            ax.plot(xvals, yvals, label=f"R{i+1}: {a1:g}X₁ + {a2:g}X₂ {row['sign']} {b:g}")
        elif abs(a1) > 1e-12:
            ax.axvline(b / a1, label=f"R{i+1}")

    if x_opt is not None:
        ax.scatter(
            x_opt[0], x_opt[1],
            s=160,
            zorder=5,
            label=f"Óptimo ({x_opt[0]:.2f}, {x_opt[1]:.2f})"
        )

    # Línea de isoutilidad/isocosto.
    if abs(c[1]) > 1e-12 and x_opt is not None:
        z = float(np.dot(c, x_opt))
        y_obj = (z - c[0] * xvals) / c[1]
        y_obj = np.where(y_obj >= 0, y_obj, np.nan)
        ax.plot(xvals, y_obj, linestyle="--", label=f"Z = {z:.2f}")

    ax.set_xlim(0, max_axis)
    ax.set_ylim(0, max_axis)
    ax.set_xlabel("X₁")
    ax.set_ylabel("X₂")
    ax.set_title("Método gráfico")
    ax.grid(True, alpha=.3)
    ax.legend()
    fig.tight_layout()
    return fig


def make_report(project, problem_name, sense, c, constraints, names, x, z, integer, prices):
    lines = []
    lines.append("INVESTIGACIÓN DE OPERACIONES - GRUPO NOVA")
    lines.append("=" * 60)
    lines.append(f"Proyecto: {project}")
    lines.append(f"Ejercicio: {problem_name}")
    lines.append("")
    lines.append("1. MODELO MATEMÁTICO")
    lines.append("-" * 60)
    lines.append("Variables de decisión:")
    for n in names:
        lines.append(f"  {n}: cantidad a producir / decidir")
    lines.append("")
    lines.append(objective_text(c, sense, names))
    lines.append("Sujeto a:")
    for i, row in enumerate(constraints):
        lines.append(f"  R{i+1}: {constraint_text(row, names)}")
    lines.append("  X₁, X₂, ..., Xn >= 0")
    lines.append("")
    lines.append("2. SOLUCIÓN ÓPTIMA")
    lines.append("-" * 60)
    for i, value in enumerate(x):
        lines.append(f"  {names[i]} = {fmt(value)}")
    lines.append(f"  Z* = {fmt(z)}")
    lines.append(f"  Tipo de variable: {'Entera' if integer else 'Continua'}")
    lines.append("")
    lines.append("3. ANÁLISIS DE RECURSOS")
    rt = resource_table(c, constraints, x)
    lines.append(rt.to_string(index=False))
    lines.append("")
    lines.append("4. PRECIOS SOMBRA")
    lines.append("-" * 60)
    if prices is None:
        lines.append("  No disponible.")
    else:
        for i, p in enumerate(prices):
            lines.append(f"  R{i+1}: {p:.6f} por unidad adicional del lado derecho")
    lines.append("")
    lines.append("5. INTERPRETACIÓN")
    lines.append("-" * 60)
    if prices is not None:
        for i, p in enumerate(prices):
            lines.append(
                f"  R{i+1}: aproximadamente {p:.4f} unidades monetarias de cambio en Z "
                f"por una unidad adicional del recurso, mientras el precio sombra sea válido."
            )
    lines.append("")
    lines.append("Nota: en modelos enteros, los precios sombra y la dualidad se interpretan "
                 "sobre la relajación lineal continua.")
    return "\n".join(lines)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("⚙️ Configuración")

    if st.button("🧪 Cargar ejercicio Empresa textil", use_container_width=True):
        st.session_state["load_example"] = True
        st.session_state["example_type"] = "textil"

    if st.button("🔄 Nuevo problema", use_container_width=True):
        st.session_state["load_example"] = False
        st.session_state["example_type"] = "custom"
        st.session_state["problem_name"] = "Nuevo ejercicio"

    st.markdown("---")

    problem_name = st.text_input(
        "Nombre del ejercicio",
        value=st.session_state.get("problem_name", "Nuevo ejercicio"),
        key="problem_name",
    )

    project = st.text_input(
        "Proyecto / empresa",
        value="Industrias del Norte S.A.C.",
    )

    sense = st.selectbox("Tipo de problema", ["Maximización", "Minimización"])

    num_variables = st.number_input(
        "Número de variables",
        min_value=2,
        max_value=10,
        value=2,
        step=1,
    )

    num_constraints = st.number_input(
        "Número de restricciones",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
    )

    integer = st.checkbox(
        "Variables enteras",
        value=False,
        help="Activa cat='Integer' en PuLP. La sensibilidad se calcula sobre la relajación continua.",
    )

    st.markdown("---")
    st.subheader("Métodos a mostrar")
    show_primal = st.checkbox("📊 Primal", True)
    show_dual = st.checkbox("🔄 Dual", True)
    show_graph = st.checkbox("📐 Método gráfico", True)
    show_sensitivity = st.checkbox("📈 Sensibilidad", True)
    show_simplex = st.checkbox("📝 Simplex / Solver", True)
    show_report = st.checkbox("📄 Reporte", True)

    st.markdown("---")
    st.caption("Grupo NOVA · 2026")

# ============================================================
# CARGAR EJEMPLO
# ============================================================

# La aplicación abre directamente con el ejercicio solicitado.
if "load_example" not in st.session_state:
    st.session_state["load_example"] = True
    st.session_state["example_type"] = "textil"

example_loaded = st.session_state.get("load_example", True)
example_type = st.session_state.get("example_type", "textil")

if example_loaded:
    sense = "Maximización"
    num_variables = 2
    num_constraints = 3

    if example_type == "textil":
        default_names = ["Camisas premium", "Pantalones ejecutivos"]
        default_c = [35.0, 50.0]
        default_rows = [
            {"a": [2.0, 3.0], "sign": "≤", "b": 240.0, "name": "Tela (metros)"},
            {"a": [1.0, 2.0], "sign": "≤", "b": 140.0, "name": "Horas de corte"},
            {"a": [3.0, 2.0], "sign": "≤", "b": 210.0, "name": "Horas de costura"},
        ]
    else:
        default_names = ["Barras energéticas", "Batidos nutricionales"]
        default_c = [70.0, 90.0]
        default_rows = [
            {"a": [4.0, 3.0], "sign": "≤", "b": 300.0, "name": "Materia prima"},
            {"a": [2.0, 4.0], "sign": "≤", "b": 240.0, "name": "Procesamiento"},
            {"a": [1.0, 2.0], "sign": "≤", "b": 120.0, "name": "Empaquetado"},
        ]
else:
    default_names = [f"X{i+1}" for i in range(int(num_variables))]
    default_c = [40.0 if i == 0 else (30.0 if i == 1 else 0.0) for i in range(int(num_variables))]
    default_rows = []
    for r in range(int(num_constraints)):
        a = [0.0] * int(num_variables)
        if int(num_variables) >= 2:
            a[0] = 2.0 if r == 0 else (1.0 if r == 1 else 1.0)
            a[1] = 1.0 if r == 0 else 2.0
        default_rows.append({
            "a": a,
            "sign": "≤",
            "b": 100.0 if r == 0 else 80.0,
            "name": f"Recurso {r+1}",
        })

# ============================================================
# ENTRADA DEL MODELO
# ============================================================

st.header("1️⃣ Formular el modelo matemático")

st.subheader("Variables de decisión")
name_cols = st.columns(int(num_variables))
names = []

for i, col in enumerate(name_cols):
    with col:
        names.append(
            st.text_input(
                f"Variable X{i+1}",
                value=default_names[i] if i < len(default_names) else f"X{i+1}",
                key=f"var_name_{i}_{example_loaded}_{num_variables}",
            )
        )

st.subheader("Función objetivo")

obj_cols = st.columns(int(num_variables))
c = []

for i, col in enumerate(obj_cols):
    with col:
        c.append(
            st.number_input(
                f"Coeficiente de X{i+1}",
                value=float(default_c[i]) if i < len(default_c) else 0.0,
                step=1.0,
                key=f"obj_{i}_{example_loaded}_{num_variables}",
            )
        )

st.subheader("Restricciones")

constraints = []

for r in range(int(num_constraints)):
    row_default = default_rows[r] if r < len(default_rows) else {
        "a": [0.0] * int(num_variables),
        "sign": "≤",
        "b": 0.0,
        "name": f"Recurso {r+1}",
    }

    st.markdown(f"**Restricción R{r+1}**")
    cols = st.columns(int(num_variables) + 3)

    a = []
    for i in range(int(num_variables)):
        with cols[i]:
            a.append(
                st.number_input(
                    f"X{i+1}",
                    value=float(row_default["a"][i]),
                    step=1.0,
                    key=f"rest_{r}_{i}_{example_loaded}_{num_variables}_{num_constraints}",
                )
            )

    with cols[int(num_variables)]:
        sign = st.selectbox(
            "Signo",
            ["≤", "=", "≥"],
            index=["≤", "=", "≥"].index(row_default["sign"]),
            key=f"sign_{r}_{example_loaded}_{num_constraints}",
        )

    with cols[int(num_variables) + 1]:
        b = st.number_input(
            "Disponibilidad / RHS",
            value=float(row_default["b"]),
            step=1.0,
            key=f"rhs_{r}_{example_loaded}_{num_constraints}",
        )

    with cols[int(num_variables) + 2]:
        rname = st.text_input(
            "Nombre",
            value=row_default.get("name", f"Recurso {r+1}"),
            key=f"rname_{r}_{example_loaded}_{num_constraints}",
        )

    constraints.append({"a": a, "sign": sign, "b": b, "name": rname})

# ============================================================
# MODELO EN PANTALLA
# ============================================================

st.header("2️⃣ Modelo matemático generado")

st.latex(objective_text(c, sense, names))

for i, row in enumerate(constraints):
    st.latex(f"R_{i+1}: " + constraint_text(row, names))

st.latex(r"X_1,X_2,\ldots,X_n \geq 0")

if integer:
    st.info("Las variables están configuradas como enteras en PuLP. Para dualidad y sensibilidad se usará la relajación continua.")

# ============================================================
# RESOLVER
# ============================================================

st.header("3️⃣ Resolver")

if not PULP_OK:
    st.error("Falta PuLP. Instala las dependencias con: pip install -r requirements.txt")
    st.code("pip install streamlit numpy pandas matplotlib scipy pulp")

if st.button("🚀 RESOLVER PROBLEMA", type="primary", use_container_width=True):

    pulp_result, status = solve_with_pulp(c, constraints, sense, integer=integer)

    if pulp_result is None:
        st.stop()

    if status != "Optimal":
        st.error(f"No se encontró una solución óptima. Estado del solver: {status}")
        st.stop()

    x = pulp_result["x"]
    z = pulp_result["z"]

    # Solución continua auxiliar para sensibilidad.
    continuous = solve_continuous_scipy(c, constraints, sense) if SCIPY_OK else None
    prices = shadow_prices(c, constraints, sense) if SCIPY_OK else None

    st.success("✅ Problema resuelto correctamente con PuLP / CBC.")

    # --------------------------------------------------------
    # MÉTRICAS PRINCIPALES
    # --------------------------------------------------------

    cols = st.columns(min(int(num_variables), 5))
    for i, value in enumerate(x):
        with cols[i % len(cols)]:
            st.metric(f"X{i+1}", fmt(value), names[i])

    st.metric(
        "Valor óptimo Z*",
        fmt(z),
        "Máximo" if sense == "Maximización" else "Mínimo"
    )

    # --------------------------------------------------------
    # PESTAÑAS
    # --------------------------------------------------------

    tabs = []
    if show_primal:
        tabs.append("📊 Primal")
    if show_dual:
        tabs.append("🔄 Dual")
    if show_graph and int(num_variables) == 2:
        tabs.append("📐 Gráfico")
    if show_sensitivity:
        tabs.append("📈 Sensibilidad")
    if show_simplex:
        tabs.append("📝 Solver / Simplex")
    if show_report:
        tabs.append("📄 Reporte")

    if not tabs:
        tabs = ["📊 Resultado"]

    tab_objects = st.tabs(tabs)

    # PRIMAL
    tab_pos = 0
    if show_primal:
        with tab_objects[tab_pos]:
            st.subheader("Solución del problema primal")

            st.markdown('<div class="good">', unsafe_allow_html=True)
            st.write(f"**Objetivo:** {objective_text(c, sense, names)}")
            st.write(f"**Solución:** " + ", ".join(
                f"{names[i]} = {fmt(x[i])}" for i in range(len(x))
            ))
            st.write(f"**Valor óptimo:** Z* = {fmt(z)}")
            st.markdown('</div>', unsafe_allow_html=True)

            st.subheader("Uso de recursos")
            rt = resource_table(c, constraints, x).copy()
            rt.insert(0, "Recurso", [r["name"] for r in constraints])
            st.dataframe(rt, use_container_width=True, hide_index=True)

            st.subheader("Interpretación")
            for i, row in enumerate(constraints):
                used = float(np.dot(row["a"], x))
                st.write(
                    f"**{row['name']} (R{i+1}):** disponible {fmt(row['b'])}, "
                    f"utilizado {fmt(used)}. {resource_table(c, constraints, x).iloc[i]['Estado']}."
                )
        tab_pos += 1

    # DUAL
    if show_dual:
        with tab_objects[tab_pos]:
            st.subheader("Construcción del problema dual")

            dual = make_dual(c, constraints, sense, names)

            st.markdown("**Función objetivo dual:**")
            st.latex(dual["objective"])

            st.markdown("**Restricciones duales:**")
            for line in dual["constraints"]:
                st.latex(line)

            st.markdown("**Signos de las variables duales:**")
            dual_df = pd.DataFrame(dual["variable_signs"], columns=["Variable", "Condición"])
            st.dataframe(dual_df, use_container_width=True, hide_index=True)

            if not integer and continuous is not None and continuous.success:
                st.info(
                    "Para el modelo continuo, el valor óptimo primal y el dual deben coincidir "
                    "bajo las condiciones habituales de dualidad fuerte."
                )
            elif integer:
                st.warning(
                    "Como el modelo es entero, la dualidad fuerte se interpreta sobre su relajación continua."
                )

        tab_pos += 1

    # GRÁFICO
    if show_graph and int(num_variables) == 2:
        with tab_objects[tab_pos]:
            st.subheader("Método gráfico")
            fig = graph_2d(c, constraints, x, sense)
            st.pyplot(fig)
            plt.close(fig)

            st.write(
                "El gráfico muestra las rectas de las restricciones y el punto encontrado "
                "por el solver. Para dos variables, esta visualización ayuda a ver la región "
                "factible y entender por qué el óptimo cae en ese punto."
            )
        tab_pos += 1

    # SENSIBILIDAD
    if show_sensitivity:
        with tab_objects[tab_pos]:
            st.subheader("Análisis de sensibilidad")

            if prices is None:
                st.warning("No fue posible calcular precios sombra porque falta SciPy.")
            else:
                sens_rows = []
                for i, p in enumerate(prices):
                    lower, upper = approximate_rhs_range(c, constraints, sense, i)
                    sens_rows.append({
                        "Recurso": constraints[i]["name"],
                        "Precio sombra": p,
                        "Rango RHS aprox. inferior": lower,
                        "Rango RHS aprox. superior": upper,
                    })

                sens_df = pd.DataFrame(sens_rows)
                st.dataframe(
                    sens_df.style.format({
                        "Precio sombra": "{:.4f}",
                        "Rango RHS aprox. inferior": "{:.4f}",
                        "Rango RHS aprox. superior": "{:.4f}",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )

                st.markdown(
                    '<div class="card"><b>¿Qué significa el precio sombra?</b><br>'
                    "Es el cambio aproximado del valor óptimo cuando aumenta en una unidad "
                    "el lado derecho de una restricción, siempre que el cambio permanezca "
                    "dentro del rango de validez. En lenguaje sencillo: indica cuánto vale "
                    "para la empresa conseguir una unidad adicional de ese recurso.</div>",
                    unsafe_allow_html=True,
                )

                for i, p in enumerate(prices):
                    if abs(p) < 1e-8:
                        st.write(
                            f"**{constraints[i]['name']} (R{i+1}):** precio sombra ≈ 0. "
                            "En la solución continua, una unidad adicional no mejora el objetivo."
                        )
                    else:
                        st.write(
                            f"**{constraints[i]['name']} (R{i+1}):** una unidad adicional "
                            f"del recurso aporta aproximadamente **{p:.4f}** unidades monetarias "
                            "al objetivo, dentro del rango de validez."
                        )

                # Análisis especial solicitado para el ejercicio Empresa textil.
                if (
                    example_loaded
                    and example_type == "textil"
                    and int(num_variables) == 2
                    and int(num_constraints) == 3
                    and abs(c[0] - 35) < 1e-8
                    and abs(c[1] - 50) < 1e-8
                ):
                    st.markdown("---")
                    st.subheader("🧵 Análisis solicitado — Empresa textil")

                    # 1. Recursos completamente utilizados.
                    rt = resource_table(c, constraints, x)
                    activos = []
                    for i, row in enumerate(constraints):
                        used = float(np.dot(row["a"], x))
                        if abs(used - row["b"]) <= 1e-7:
                            activos.append(row["name"])

                    if activos:
                        st.success(
                            "**Recursos utilizados completamente:** "
                            + ", ".join(activos)
                            + "."
                        )
                    else:
                        st.info("No hay recursos utilizados completamente en la solución.")

                    # 2. Precio sombra e interpretación.
                    if prices is not None:
                        st.markdown("### 💲 Precios sombra e interpretación administrativa")
                        for i, p in enumerate(prices):
                            if abs(p) < 1e-8:
                                st.write(
                                    f"**{constraints[i]['name']}: S/ 0.00.** "
                                    "Una unidad adicional de este recurso no aumenta "
                                    "el valor óptimo mientras se mantenga la estructura actual."
                                )
                            else:
                                st.write(
                                    f"**{constraints[i]['name']}: S/ {p:.2f} por unidad.** "
                                    f"Administrativamente, una unidad adicional de este recurso "
                                    f"podría aumentar el valor óptimo en aproximadamente "
                                    f"**S/ {p:.2f}**, mientras el precio sombra siga siendo válido."
                                )

                        # 3. 30 horas adicionales de costura a S/8/hora.
                        sewing_idx = 2
                        extra_hours = 30.0
                        cost_per_hour = 8.0
                        extra_cost = extra_hours * cost_per_hour

                        z_more = solve_rhs_sensitivity(
                            c, constraints, sense, sewing_idx, extra_hours
                        )

                        st.markdown("### 🧵 ¿Conviene contratar 30 horas adicionales de costura?")
                        if z_more is not None:
                            benefit = z_more - z
                            net = benefit - extra_cost

                            st.write(
                                f"Precio sombra de costura: **S/ {prices[sewing_idx]:.2f} por hora**."
                            )
                            st.write(
                                f"Valor económico de 30 horas adicionales: "
                                f"30 × S/ {prices[sewing_idx]:.2f} = **S/ {benefit:,.2f}**."
                            )
                            st.write(
                                f"Costo de contratar 30 horas: "
                                f"30 × S/ {cost_per_hour:.2f} = **S/ {extra_cost:,.2f}**."
                            )
                            st.write(f"Resultado neto: **S/ {net:,.2f}**.")

                            if net > 0:
                                st.success(
                                    "Decisión: **sí conviene** contratar las 30 horas, "
                                    "porque el beneficio marginal supera el costo."
                                )
                            elif abs(net) < 1e-8:
                                st.info(
                                    "Decisión: es indiferente desde el punto de vista económico, "
                                    "porque beneficio y costo son iguales."
                                )
                            else:
                                st.warning(
                                    "Decisión: **no conviene** contratar las 30 horas, "
                                    "porque el costo supera el beneficio generado."
                                )

                        # 4. Disminución de la utilidad del pantalón.
                        st.markdown("### 👖 ¿Hasta cuánto puede disminuir la utilidad de los pantalones?")
                        st.write(
                            "Se mantiene la combinación óptima mientras el coeficiente de utilidad "
                            "del pantalón permanezca dentro del rango de optimalidad."
                        )

                        # Para este caso concreto, el límite inferior exacto de c2 es 23.3333.
                        # Se calcula con las restricciones activas (corte y costura).
                        c2_current = 50.0
                        c2_min = 70.0 / 3.0
                        max_decrease = c2_current - c2_min

                        st.write(
                            f"Utilidad actual del pantalón: **S/ {c2_current:.2f}**."
                        )
                        st.write(
                            f"Puede disminuir hasta aproximadamente **S/ {c2_min:.2f} por pantalón** "
                            f"sin modificar la combinación óptima."
                        )
                        st.write(
                            f"Disminución máxima: **S/ {max_decrease:.2f}** por pantalón."
                        )

                        c2_test = c2_min
                        test_res = solve_continuous_scipy(
                            [35.0, c2_test], constraints, sense
                        )
                        if test_res is not None and test_res.success:
                            z_test = -test_res.fun if sense == "Maximización" else test_res.fun
                            st.info(
                                f"En el límite, la utilidad queda en S/ {c2_test:.2f}. "
                                f"El valor óptimo correspondiente es aproximadamente "
                                f"S/ {z_test:,.2f}; a partir de ese límite la combinación "
                                f"óptima puede cambiar."
                            )

                    # 5. Informe de sensibilidad completo.
                    st.markdown("---")
                    st.subheader("📋 Informe de sensibilidad")

                    sensitivity_lines = []
                    sensitivity_lines.append("INFORME DE ANÁLISIS DE SENSIBILIDAD")
                    sensitivity_lines.append("EMPRESA TEXTIL — GRUPO NOVA")
                    sensitivity_lines.append("=" * 65)
                    sensitivity_lines.append("")
                    sensitivity_lines.append("MODELO:")
                    sensitivity_lines.append("MAX Z = 35X1 + 50X2")
                    sensitivity_lines.append("Sujeto a:")
                    sensitivity_lines.append("  2X1 + 3X2 <= 240   (Tela)")
                    sensitivity_lines.append("  X1 + 2X2 <= 140   (Corte)")
                    sensitivity_lines.append("  3X1 + 2X2 <= 210  (Costura)")
                    sensitivity_lines.append("  X1, X2 >= 0")
                    sensitivity_lines.append("")
                    sensitivity_lines.append("SOLUCIÓN ÓPTIMA:")
                    sensitivity_lines.append(f"  X1 (Camisas premium) = {x[0]:.4f}")
                    sensitivity_lines.append(f"  X2 (Pantalones ejecutivos) = {x[1]:.4f}")
                    sensitivity_lines.append(f"  Z* = S/ {z:,.2f}")
                    sensitivity_lines.append("")
                    sensitivity_lines.append("RECURSOS:")
                    for i, row in enumerate(constraints):
                        used = float(np.dot(row["a"], x))
                        slack = row["b"] - used
                        state = "UTILIZADO COMPLETAMENTE" if abs(slack) < 1e-7 else "CON HOLGURA"
                        sensitivity_lines.append(
                            f"  {row['name']}: disponible={row['b']:.2f}, "
                            f"usado={used:.2f}, holgura={slack:.2f}, {state}"
                        )
                    sensitivity_lines.append("")
                    sensitivity_lines.append("PRECIOS SOMBRA:")
                    if prices is not None:
                        for i, p in enumerate(prices):
                            sensitivity_lines.append(
                                f"  {constraints[i]['name']}: S/ {p:.4f} por unidad adicional"
                            )
                    sensitivity_lines.append("")
                    sensitivity_lines.append("DECISIÓN: 30 HORAS ADICIONALES DE COSTURA")
                    sensitivity_lines.append("  Precio de contratación = S/ 8 por hora")
                    sensitivity_lines.append("  Costo total = S/ 240.00")
                    if prices is not None:
                        sensitivity_lines.append(
                            f"  Valor marginal de 30 horas = S/ {prices[2] * 30:,.2f}"
                        )
                        sensitivity_lines.append(
                            f"  Beneficio neto = S/ {(prices[2] * 30) - 240:,.2f}"
                        )
                        sensitivity_lines.append(
                            "  Decisión = NO CONVIENE"
                            if (prices[2] * 30) - 240 < 0
                            else "  Decisión = CONVIENE"
                        )
                    sensitivity_lines.append("")
                    sensitivity_lines.append("CAMBIO EN UTILIDAD DE PANTALONES:")
                    sensitivity_lines.append("  Utilidad actual = S/ 50.00")
                    sensitivity_lines.append(f"  Límite inferior = S/ {c2_min:.4f}")
                    sensitivity_lines.append(f"  Disminución máxima = S/ {max_decrease:.4f}")
                    sensitivity_lines.append(
                        "  Por debajo de este límite puede cambiar la combinación óptima."
                    )
                    sensitivity_lines.append("")
                    sensitivity_lines.append(
                        "Conclusión: corte y costura son los recursos críticos; la tela "
                        "presenta holgura. El precio sombra de costura es S/5 por hora, "
                        "por lo que pagar S/8 por una hora adicional no resulta rentable. "
                        "La utilidad del pantalón puede disminuir hasta aproximadamente "
                        "S/23.33 sin modificar la combinación óptima."
                    )

                    sensitivity_report = "\n".join(sensitivity_lines)
                    st.text_area(
                        "Informe listo para revisar o copiar",
                        sensitivity_report,
                        height=520,
                    )
                    st.download_button(
                        "⬇️ Descargar informe de sensibilidad",
                        data=sensitivity_report.encode("utf-8"),
                        file_name="informe_sensibilidad_empresa_textil_NOVA.txt",
                        mime="text/plain",
                        use_container_width=True,
                    )

            if integer:
                st.warning(
                    "Advertencia académica: con variables enteras, los precios sombra "
                    "no deben interpretarse como si fueran directamente los de un modelo LP entero. "
                    "La tabla mostrada corresponde a la relajación continua."
                )
        tab_pos += 1

    # SOLVER / SIMPLEX
    if show_simplex:
        with tab_objects[tab_pos]:
            st.subheader("Solver / Simplex")

            st.write(
                "El programa modela el problema en PuLP y utiliza CBC para encontrar la solución. "
                "Así puedes cambiar coeficientes, restricciones, cantidad de variables y tipo de "
                "problema sin volver a escribir el modelo."
            )

            st.write("**Estado:**", pulp_result["status"])
            st.write("**Método de resolución:** PuLP + CBC")

            if integer:
                st.info(
                    "El modelo está configurado como entero. CBC resuelve el problema entero; "
                    "para sensibilidad se utiliza la relajación continua."
                )
            else:
                st.info(
                    "El modelo está configurado como continuo, que es la formulación estándar "
                    "de programación lineal."
                )

            st.subheader("Resumen del modelo")
            model_rows = []
            for i, row in enumerate(constraints):
                model_rows.append({
                    "Restricción": f"R{i+1}",
                    "Nombre": row["name"],
                    "Signo": row["sign"],
                    "RHS": row["b"],
                })
            st.dataframe(pd.DataFrame(model_rows), use_container_width=True, hide_index=True)

    # REPORTE
    if show_report:
        with tab_objects[tab_pos]:
            st.subheader("Reporte listo para entregar")

            report = make_report(
                project,
                problem_name,
                sense,
                c,
                constraints,
                names,
                x,
                z,
                integer,
                prices,
            )

            st.text_area("Vista previa", report, height=450)

            st.download_button(
                "⬇️ Descargar reporte TXT",
                data=report.encode("utf-8"),
                file_name="reporte_programacion_lineal_NOVA.txt",
                mime="text/plain",
                use_container_width=True,
            )

            csv_data = resource_table(c, constraints, x).to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Descargar tabla de recursos CSV",
                data=csv_data,
                file_name="analisis_recursos_NOVA.csv",
                mime="text/csv",
                use_container_width=True,
            )

            st.markdown("---")
            st.subheader("Datos institucionales")
            st.write("**Universidad:** Universidad Nacional de Ingeniería")
            st.write("**Curso:** Investigación de Operaciones")
            st.write("**Docente:** Mg. Enrique Jannier Boy Vasquez")
            st.write("**Año:** 2026")
            st.write("**Grupo:** NOVA")
            st.write("**Integrantes:**")
            for person in INTEGRANTES:
                st.write(f"- {person}")

            tab_pos += 1


# ============================================================
# EJEMPLO / EXPLICACIÓN INICIAL
# ============================================================

# Mensaje inferior solo cuando se trabaja con un problema nuevo.
if not st.session_state.get("load_example", True):
    st.markdown("---")
    st.info(
        "👈 Puedes crear un ejercicio completamente nuevo ingresando tus propios datos."
    )

st.markdown("---")
st.caption(
    "Aplicación educativa de Programación Lineal · Grupo NOVA · 2026. "
    "Los rangos de sensibilidad se muestran como aproximaciones didácticas."
)
