"""
Optimizador de Hélices Navales — Versión 8
Interfaz web con Streamlit

Serie B Wageningen · Kaplan Ka 4-70 · Ka 5-75
Basado en métodos de series sistemáticas (Birk 2019)
"""

import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from core import biblioteca_coeficientes as coefs
from core.cavitacion import VerificacionCavitacion
from core.gestor_motores import (
    aproximar_curva_motor,
    cargar_biblioteca,
    listar_motores,
    obtener_datos_motor,
)
from core.optimizador import PropellerOptimizer

# ─────────────────────────────────────────────────────────────────────────────
# Configuración de página
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Optimizador de Hélices Navales",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; }
    [data-testid="metric-container"] {
        background: #f0f4f8;
        border-radius: 8px;
        padding: 6px 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.title("⚓ Optimizador de Hélices Navales")
st.caption(
    "Serie B de Wageningen · Kaplan Ka 4-70 · Ka 5-75  —  Versión 8  |  "
    "Método de series sistemáticas (Birk 2019)"
)
st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# Biblioteca de motores (cacheada)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def _cargar_biblioteca():
    return cargar_biblioteca()


biblioteca = _cargar_biblioteca()
nombres_motores = listar_motores(biblioteca)

# ─────────────────────────────────────────────────────────────────────────────
# Mapa de configuraciones de hélice
# ─────────────────────────────────────────────────────────────────────────────
PROP_MAP = {
    "Serie B Wageningen (hélice libre)": {
        "tipo": "SERIE_B",
        "nombre": "Serie B Wageningen",
    },
    "Ka 4-70 con Tobera 19A (Kaplan)": {
        "tipo": "KAPLAN",
        "nombre": "Ka 4-70 (Con Tobera 19A)",
        "matriz": coefs.COEF_KAPLAN_KA470_19A,
        "con_tobera": True,
        "z": 4,
        "ear": 0.70,
    },
    "Ka 4-70 sin Tobera (Kaplan libre)": {
        "tipo": "KAPLAN",
        "nombre": "Ka 4-70 (Sin Tobera)",
        "matriz": coefs.COEF_KAPLAN_KA470_19A,
        "con_tobera": False,
        "z": 4,
        "ear": 0.70,
    },
    "Ka 5-75 con Tobera 19A (Kaplan)": {
        "tipo": "KAPLAN",
        "nombre": "Ka 5-75 (Con Tobera 19A)",
        "matriz": coefs.COEF_KAPLAN_KA575_19A,
        "con_tobera": True,
        "z": 5,
        "ear": 0.75,
    },
    "Ka 5-75 sin Tobera (Kaplan libre)": {
        "tipo": "KAPLAN",
        "nombre": "Ka 5-75 (Sin Tobera)",
        "matriz": coefs.COEF_KAPLAN_KA575_19A,
        "con_tobera": False,
        "z": 5,
        "ear": 0.75,
    },
}

TAREA_LABEL = {
    1: "1 — Optimizar diámetro  (Potencia + Ratio + Vs)",
    2: "2 — Optimizar diámetro  (Empuje requerido + Ratio + Vs)",
    3: "3 — Optimizar RPM/Ratio (Potencia + D fijo + Vs)",
    4: "4 — Optimizar RPM/Ratio (Empuje requerido + D fijo + Vs)",
    5: "5 — Bollard Pull / Tiro de remolque",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers de gráficas
# ─────────────────────────────────────────────────────────────────────────────
def _fig_motor(datos_m: dict, bhp_op: float, rpm_op: float, nombre: str):
    """Gráfica de curva de potencia del motor."""
    curva = datos_m["performance_curve"]
    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    ax.plot(curva["rpm"], curva["power_hp"], "b-o", lw=2, ms=5, label="Curva motor")
    ax.axvline(rpm_op, color="red", ls="--", alpha=0.5)
    ax.axhline(bhp_op, color="red", ls="--", alpha=0.5)
    ax.plot(rpm_op, bhp_op, "ro", ms=9,
            label=f"Punto operación\n{bhp_op:.0f} HP @ {rpm_op:.0f} RPM")
    ax.set_xlabel("RPM motor")
    ax.set_ylabel("Potencia [HP]")
    ax.set_title(nombre, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _fig_helice(opt: PropellerOptimizer, resultado: dict):
    """Diagrama de aguas libres KT · 10KQ · η vs J."""
    d = opt.generar_datos_grafica_helice(resultado)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    ax.plot(d["j_arr"], d["kt_arr"],   "b-", lw=2, label="KT")
    ax.plot(d["j_arr"], d["kq10_arr"], "r-", lw=2, label="10·KQ")
    ax.plot(d["j_arr"], d["efic_arr"], "g-", lw=2, label="η")
    j_op = d["j_op"]
    ax.axvline(j_op, color="k", ls="--", alpha=0.45, label=f"J = {j_op:.3f}")
    ax.plot(j_op, d["kt_op"],        "b^", ms=8, zorder=5)
    ax.plot(j_op, d["kq_op"] * 10,  "r^", ms=8, zorder=5)
    ax.plot(j_op, d["efic_op"],      "g^", ms=8, zorder=5)
    ax.set_xlabel("J — Coeficiente de avance")
    ax.set_ylabel("KT · 10KQ · η")
    ax.set_title(f"Hélice en aguas libres  —  P/D = {resultado['PD']:.3f}", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    return fig


def _fig_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────────────────────────────────────
# Generación de PDF en memoria
# ─────────────────────────────────────────────────────────────────────────────
def _generar_pdf(res: dict, img1: bytes | None, img2: bytes | None) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        Image as RLImage,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        topMargin=2 * cm, bottomMargin=2 * cm,
        leftMargin=2.5 * cm, rightMargin=2.5 * cm,
    )
    styles = getSampleStyleSheet()
    story = []

    def h1(t): return Paragraph(f"<b>{t}</b>", styles["Heading1"])
    def h2(t): return Paragraph(f"<b>{t}</b>", styles["Heading2"])
    def p(t):  return Paragraph(t, styles["Normal"])

    # Título
    story += [
        h1("Reporte — Optimización de Hélice Naval"),
        Spacer(1, 0.3 * cm),
        p(f"<b>Motor:</b> {res.get('_nombre_motor', '—')}  —  "
          f"{res.get('_bhp_op', 0):.0f} HP @ {res.get('_rpm_motor_op', 0):.0f} RPM"),
        p(f"<b>Serie:</b> {res['tipo_helice']}  |  <b>Tarea de diseño:</b> {res['tarea']}"),
        Spacer(1, 0.5 * cm),
    ]

    # Tabla de parámetros principales
    task = res["tarea"]
    filas = [["Parámetro", "Valor"]]
    filas += [
        ["P/D óptimo",         f"{res['PD']:.4f}"],
        ["Diámetro D",         f"{res['D_m']:.4f} m"],
        ["RPM hélice",         f"{res['rpm_helice']:.2f}"],
        ["Coef. avance J",     f"{res['J']:.4f}"],
        ["Eficiencia η",       f"{res['eficiencia'] * 100:.2f} %"],
        ["KT",                 f"{res['KT']:.5f}"],
        ["KQ",                 f"{res['KQ']:.5f}"],
        ["EAR (Ae/Ao)",        f"{res['EAR']:.3f}"],
        ["Palas Z",            f"{res['Z']}"],
        ["Va (avance hélice)", f"{res['va_ms']:.3f} m/s"],
        ["Wake factor w",      f"{res['wake_factor']:.3f}"],
        ["Empuje bruto T",     f"{res['thrust_bruto_kN']:.3f} kN"],
    ]
    if task != 5:
        filas.append(["Empuje neto T",
                       f"{res.get('thrust_neto_kN', '—'):.3f} kN"
                       if res.get('thrust_neto_kN') else "—"])
    else:
        filas.append(["Tiro neto",
                       f"{res.get('thrust_neto_tonf', 0):.3f} ton-fuerza"])
    if res.get("pd_kw"):
        filas.append(["Potencia al eje Pd", f"{res['pd_kw']:.2f} kW"])
    if res.get("ratio_i"):
        filas.append(["Ratio reductora i", f"{res['ratio_i']:.4f} : 1"])

    tbl = Table(filas, colWidths=[8 * cm, 7 * cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1,  0), colors.HexColor("#1f4e79")),
        ("TEXTCOLOR",    (0, 0), (-1,  0), colors.white),
        ("FONTNAME",     (0, 0), (-1,  0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#eef2f7")]),
        ("GRID",         (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE",     (0, 0), (-1, -1), 10),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
    ]))
    story += [h2("Parámetros de la Hélice Óptima"), tbl, Spacer(1, 0.5 * cm)]

    # Advertencias
    if res.get("advertencias"):
        story.append(h2("Notas y Advertencias"))
        for adv in res["advertencias"]:
            story.append(p(f"⚠ {adv}"))
        story.append(Spacer(1, 0.3 * cm))

    # Verificación de cavitación (si existe en resultado)
    if res.get("_cav"):
        cav = res["_cav"]
        story += [
            h2("Verificación de Cavitación"),
            p(f"<b>Keller:</b>  EAR mín = {cav['ear_min_keller']:.4f}  →  "
              f"{'OK' if cav['ok_keller'] else 'INSUFICIENTE'}"),
            p(f"<b>Burrill:</b>  σb = {cav['sigma_b']:.4f}  |  "
              f"τc_lim = {cav['tau_c_lim']:.4f}  |  τc_real = {cav['tau_c_real']:.4f}  →  "
              f"{'OK' if cav['ok_burrill_carga'] else 'EXCEDE'}"),
            p(f"<b>Velocidad punta:</b>  Vtip = {cav['vtip_ms']:.2f} m/s  →  "
              f"Riesgo {cav['nivel_vtip']}"),
            Spacer(1, 0.3 * cm),
        ]

    # Gráficas
    ancho = 14 * cm
    if img1:
        story += [
            h2("Gráfica 1 — Curva de Potencia del Motor"),
            RLImage(io.BytesIO(img1), width=ancho, height=ancho * 0.5),
            Spacer(1, 0.4 * cm),
        ]
    if img2:
        story += [
            h2("Gráfica 2 — Diagrama de Aguas Libres (KT · 10KQ · η)"),
            RLImage(io.BytesIO(img2), width=ancho, height=ancho * 0.6),
            Spacer(1, 0.4 * cm),
        ]

    doc.build(story)
    buf.seek(0)
    return buf.read()


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Inputs
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ Parámetros de Diseño")

    # ── Motor ─────────────────────────────────────────────────────────────────
    with st.expander("🔧 Motor", expanded=True):
        motor_mode = st.radio(
            "Fuente",
            ["Biblioteca de motores", "Ingreso manual"],
            horizontal=True,
            label_visibility="collapsed",
        )
        if motor_mode == "Biblioteca de motores":
            nombre_motor = st.selectbox(
                "Motor", nombres_motores, label_visibility="collapsed"
            )
            datos_motor = obtener_datos_motor(biblioteca, nombre_motor)
            bhp_nom = float(datos_motor["nominal_hp"])
            rpm_nom = float(datos_motor["nominal_rpm"])
        else:
            nombre_motor = "Motor manual"
            col_a, col_b = st.columns(2)
            bhp_nom = float(col_a.number_input("HP nominal", 50, 10000, 500, 10))
            rpm_nom = float(col_b.number_input("RPM nominal", 500, 3500, 1800, 50))
            datos_motor = aproximar_curva_motor(bhp_nom, rpm_nom)

        st.info(f"**{nombre_motor}**  \n{bhp_nom:.0f} HP @ {rpm_nom:.0f} RPM")

        usar_nominal = st.checkbox(
            "Usar valores nominales como punto de operación", value=True
        )
        if usar_nominal:
            bhp_op, rpm_motor_op = bhp_nom, rpm_nom
        else:
            col_c, col_d = st.columns(2)
            bhp_op = float(
                col_c.number_input("HP operación", 1.0, bhp_nom * 1.2, bhp_nom, 1.0)
            )
            rpm_motor_op = float(
                col_d.number_input("RPM operación", 100.0, rpm_nom * 1.2, rpm_nom, 10.0)
            )

    # ── Tipo de hélice ────────────────────────────────────────────────────────
    with st.expander("🌀 Tipo de Hélice", expanded=True):
        tipo_helice_str = st.selectbox(
            "Serie",
            list(PROP_MAP.keys()),
            label_visibility="collapsed",
        )
        prop_config = PROP_MAP[tipo_helice_str]

    # ── Fluido ────────────────────────────────────────────────────────────────
    with st.expander("🌊 Fluido", expanded=False):
        fluido_opc = st.selectbox(
            "Fluido",
            ["Agua de mar 15°C  (1026 kg/m³)", "Agua dulce 15°C  (1000 kg/m³)", "Personalizado"],
            label_visibility="collapsed",
        )
        if "mar" in fluido_opc:
            rho = 1026.021
        elif "dulce" in fluido_opc:
            rho = 1000.0
        else:
            rho = st.number_input("Densidad ρ [kg/m³]", 900.0, 1100.0, 1026.021, 0.1)
        re_m6 = st.number_input(
            "Reynolds Re [×10⁶]", 0.1, 1000.0, 1.0, 0.1,
            help="Número de Reynolds en radio 0.75R (afecta corrección de Serie B)"
        )
        re = re_m6 * 1e6

    # ── Geometría ─────────────────────────────────────────────────────────────
    with st.expander("📐 Geometría de Hélice", expanded=True):
        if prop_config["tipo"] == "SERIE_B":
            z = st.slider("Número de palas Z", 2, 7, 4)
            ae_ao = st.slider("Relación de área EAR (Ae/Ao)", 0.30, 1.05, 0.693, 0.001, format="%.3f")
        else:
            z = prop_config["z"]
            ae_ao = prop_config["ear"]
            st.info(f"Kaplan: Z = **{z}** palas fijas · EAR = **{ae_ao:.2f}** fija")

    # ── Tarea de diseño ───────────────────────────────────────────────────────
    with st.expander("📋 Tarea de Diseño", expanded=True):
        tarea = st.selectbox(
            "Tarea",
            [1, 2, 3, 4, 5],
            format_func=lambda x: TAREA_LABEL[x],
            label_visibility="collapsed",
        )

    # ── Factores de corrección ────────────────────────────────────────────────
    with st.expander("🔄 Factores de Corrección", expanded=False):
        wake_factor = st.slider(
            "Factor de estela w", 0.0, 0.30, 0.2865, 0.0001, format="%.4f",
            help="Va = (1 - w)·Vs.  Típico: 0.10–0.25 para barcos mercantes"
        )
        thrust_deduction = st.slider(
            "Deducción de empuje t", 0.0, 0.20, 0.190, 0.0001, format="%.4f",
            help="T_neto = T_bruto · (1 - t).  Típico: 0.05–0.15"
        )
        margen_servicio = st.slider(
            "Margen de servicio Δs", 1.0, 2.0, 1.0, 0.05,
            help="Factor de seguridad multiplicado al empuje requerido"
        )
        delta_d = st.slider(
            "Ajuste estela no uniforme Δd", 1.0, 1.05, 1.0, 0.01,
            help="J_adj = Δd · J_opt.  Típico: 1.00–1.03"
        )

    # ── Datos específicos de la tarea ─────────────────────────────────────────
    with st.expander("📊 Datos de Diseño", expanded=True):
        eta_s = st.number_input(
            "Eficiencia del eje ηs", 0.85, 1.0, 0.97, 0.01,
            help="Pérdidas mecánicas en línea de ejes y chumaceras"
        )
        eta_r = st.number_input(
            "Eficiencia rotativa ηR", 0.90, 1.30, 1.009, 1.001, format="%.3f",
            help="Corrección por flujo no uniforme. Típico: 0.95–1.05"
        )

        # Variables que se definen según la tarea seleccionada
        vs_nudos = ratio_i = t_kn = d_fijo = None
        d_bp = vs_nudos_bp = t_deduct5 = r_tug_kn = None

        if tarea in (1, 2, 3, 4):
            vs_nudos = st.number_input(
                "Velocidad del buque Vs [nudos]", 1.0, 40.0, 10.0, 0.5
            )

        if tarea in (1, 2):
            ratio_i = st.number_input(
                "Relación reductora i  (RPM_motor / RPM_hélice)",
                1.0, 20.0, 3.5, 0.1,
                help="RPM_hélice = RPM_motor / i"
            )
            if tarea == 2:
                t_kn = st.number_input(
                    "Empuje requerido T [kN]", 1.0, 10000.0, 100.0, 5.0
                )

        elif tarea in (3, 4):
            d_fijo = st.number_input(
                "Diámetro máximo D [m]", 0.3, 12.0, 1.2, 0.05
            )
            if tarea == 4:
                t_kn = st.number_input(
                    "Empuje requerido T [kN]", 1.0, 10000.0, 100.0, 5.0
                )

        elif tarea == 5:
            d_bp = st.number_input(
                "Diámetro de hélice D [m]", 0.3, 12.0, 1.2, 0.05
            )
            vs_nudos_bp = st.number_input(
                "Velocidad de remolque Vs [nudos]  (0 = punto fijo)",
                0.0, 5.0, 0.0, 0.5
            )
            t_deduct5 = st.slider(
                "Deducción de empuje t (Tarea 5)", 0.0, 0.20, 0.0, 0.01
            )
            r_tug_kn = 0.0
            if vs_nudos_bp and vs_nudos_bp > 0:
                r_tug_kn = st.number_input(
                    "Resistencia propia del remolcador [kN]", 0.0, 5000.0, 0.0, 5.0
                )

    st.divider()
    run_btn = st.button(
        "🚀 Ejecutar Optimización", type="primary", use_container_width=True
    )


# ═════════════════════════════════════════════════════════════════════════════
# LÓGICA DE OPTIMIZACIÓN (se ejecuta al presionar el botón)
# ═════════════════════════════════════════════════════════════════════════════
if run_btn:
    with st.spinner("Calculando optimización..."):
        try:
            pd_w = bhp_op * 746.0 * eta_s  # HP → W, aplicando ηs

            if tarea == 1:
                rpm_helice = rpm_motor_op / ratio_i
                inputs = [pd_w, rpm_helice, vs_nudos, eta_r]

            elif tarea == 2:
                rpm_helice = rpm_motor_op / ratio_i
                inputs = [t_kn * 1000.0, rpm_helice, vs_nudos]

            elif tarea == 3:
                inputs = [pd_w, d_fijo, vs_nudos, eta_r]

            elif tarea == 4:
                inputs = [t_kn * 1000.0, d_fijo, vs_nudos]

            else:  # tarea == 5
                vs_5 = vs_nudos_bp if vs_nudos_bp is not None else 0.0
                r_5  = r_tug_kn   if r_tug_kn   is not None else 0.0
                td_5 = t_deduct5  if t_deduct5   is not None else 0.0
                inputs = [bhp_op, rpm_motor_op, eta_s, d_bp, vs_5, td_5, r_5 * 1000.0, eta_r]

            opt = PropellerOptimizer(prop_config=prop_config, rho=rho, re=re)
            resultado = opt.run_optimization(
                task=tarea,
                inputs=inputs,
                z=z,
                afac=ae_ao,
                margen_servicio=margen_servicio,
                delta_d=delta_d,
                thrust_deduction=thrust_deduction,
                wake_factor=wake_factor,
            )

            # Enriquecer resultado con datos del contexto de la app
            resultado["rpm_motor"]      = rpm_motor_op
            resultado["_nombre_motor"]  = nombre_motor
            resultado["_bhp_op"]        = bhp_op
            resultado["_rpm_motor_op"]  = rpm_motor_op

            # Calcular ratio para tareas 3 y 4 (el core no lo incluye)
            if tarea in (3, 4) and resultado.get("rpm_helice", 0) > 0:
                resultado["ratio_i"] = rpm_motor_op / resultado["rpm_helice"]

            st.session_state["resultado"]   = resultado
            st.session_state["optimizer"]   = opt
            st.session_state["datos_motor"] = datos_motor

        except Exception as exc:
            st.error(f"❌ Error en la optimización: {exc}")
            st.stop()


# ═════════════════════════════════════════════════════════════════════════════
# PANTALLA DE BIENVENIDA (si no hay resultado aún)
# ═════════════════════════════════════════════════════════════════════════════
if "resultado" not in st.session_state:
    col1, col2, col3 = st.columns(3)
    col1.info(
        "**Paso 1 — Motor**\n\n"
        "Seleccione un motor CAT de la biblioteca o ingrese HP y RPM manualmente."
    )
    col2.info(
        "**Paso 2 — Hélice y tarea**\n\n"
        "Elija la serie (Wageningen B o Kaplan), la geometría y la tarea de diseño."
    )
    col3.info(
        "**Paso 3 — Optimizar**\n\n"
        "Haga clic en **Ejecutar Optimización** para obtener D, P/D y η óptimos."
    )

    st.markdown("""
---
### ¿Qué calcula este optimizador?

Dado un motor marino con su curva de potencia y los requerimientos operativos,
el sistema determina automáticamente la geometría óptima de la hélice y
verifica los criterios de cavitación de **Keller** y **Burrill-Emerson**.

| Tarea | Datos fijos | Salida optimizada |
|-------|-------------|-------------------|
| 1 | Potencia + ratio reductora + Vs | Diámetro D |
| 2 | Empuje T + ratio reductora + Vs | Diámetro D |
| 3 | Potencia + diámetro D + Vs | RPM / ratio reductora |
| 4 | Empuje T + diámetro D + Vs | RPM / ratio reductora |
| 5 | Potencia + diámetro D | Máximo tiro (Bollard Pull) |

**Series disponibles:**
- **Serie B Wageningen** — hélice libre, 2–7 palas, EAR 0.40–1.05
- **Kaplan Ka 4-70** — 4 palas, con o sin tobera 19A
- **Kaplan Ka 5-75** — 5 palas, con o sin tobera 19A
""")
    st.stop()


# ═════════════════════════════════════════════════════════════════════════════
# PANEL DE RESULTADOS
# ═════════════════════════════════════════════════════════════════════════════
resultado          = st.session_state["resultado"]
opt                = st.session_state["optimizer"]
datos_motor_cached = st.session_state["datos_motor"]
task               = resultado["tarea"]

st.subheader(f"📊 Resultados — Tarea {task}: {TAREA_LABEL[task].split('—')[1].strip()}")

# ── Métricas principales ──────────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Diámetro D",     f"{resultado['D_m']:.3f} m")
c2.metric("P/D óptimo",     f"{resultado['PD']:.4f}")
c3.metric("Eficiencia η",   f"{resultado['eficiencia'] * 100:.1f} %")
c4.metric("RPM hélice",     f"{resultado['rpm_helice']:.1f}")
c5.metric("Empuje bruto",   f"{resultado['thrust_bruto_kN']:.1f} kN")
c6.metric("Coef. avance J", f"{resultado['J']:.4f}")

c1b, c2b, c3b, c4b, c5b, c6b = st.columns(6)
c1b.metric("KT",  f"{resultado['KT']:.5f}")
c2b.metric("KQ",  f"{resultado['KQ']:.5f}")
c3b.metric("Va",  f"{resultado['va_ms']:.3f} m/s")
c4b.metric("EAR", f"{resultado['EAR']:.3f}")
c5b.metric("Palas Z", str(resultado["Z"]))

# Sexta métrica de segunda fila: variable según tarea
if task == 5:
    c6b.metric("Tiro neto", f"{resultado.get('thrust_neto_tonf', 0):.3f} tf")
elif resultado.get("ratio_i"):
    c6b.metric("Ratio i reductora", f"{resultado['ratio_i']:.3f} : 1")
elif resultado.get("pd_kw"):
    c6b.metric("Pd eje", f"{resultado['pd_kw']:.1f} kW")

# Empuje neto (tareas 1–4)
if task != 5 and resultado.get("thrust_neto_kN") is not None:
    st.caption(
        f"Empuje neto: **{resultado['thrust_neto_kN']:.3f} kN**  "
        f"(deducción t = {resultado.get('thrust_deduction', 0):.2f})"
    )

# ── Advertencias ──────────────────────────────────────────────────────────────
if resultado.get("advertencias"):
    for adv in resultado["advertencias"]:
        st.warning(f"⚠ {adv}")

st.divider()

# ── Pestañas ──────────────────────────────────────────────────────────────────
tab_charts, tab_cav, tab_pdf = st.tabs(
    ["📈 Gráficas", "🔍 Verificación de Cavitación", "📄 Reporte PDF"]
)

# ── Pestaña: Gráficas ────────────────────────────────────────────────────────
with tab_charts:
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.markdown("**Curva de Potencia del Motor**")
        try:
            fig1 = _fig_motor(
                datos_motor_cached,
                resultado["_bhp_op"],
                resultado["_rpm_motor_op"],
                resultado["_nombre_motor"],
            )
            st.pyplot(fig1, use_container_width=True)
            st.session_state["img1"] = _fig_bytes(fig1)
            plt.close(fig1)
        except Exception as e:
            st.warning(f"Gráfica de motor no disponible: {e}")
            st.session_state["img1"] = None

    with col_g2:
        st.markdown("**Diagrama de Aguas Libres  KT · 10KQ · η**")
        try:
            fig2 = _fig_helice(opt, resultado)
            st.pyplot(fig2, use_container_width=True)
            st.session_state["img2"] = _fig_bytes(fig2)
            plt.close(fig2)
        except Exception as e:
            st.warning(f"Diagrama de hélice no disponible: {e}")
            st.session_state["img2"] = None

# ── Pestaña: Cavitación ──────────────────────────────────────────────────────
with tab_cav:
    st.markdown("### Verificación de Cavitación y Velocidad de Punta")
    st.info(
        "Complete los parámetros adicionales y haga clic en **Verificar** "
        "para evaluar los criterios de Keller, Burrill-Emerson y velocidad de punta."
    )

    col_cv1, col_cv2, col_cv3 = st.columns(3)
    h0 = col_cv1.number_input(
        "Inmersión del eje h₀ [m]", 0.5, 20.0, 3.0, 0.1,
        help="Profundidad del eje bajo la línea de flotación"
    )
    tipo_buque_cav = col_cv2.selectbox(
        "Tipo de embarcación",
        ["single", "twin_slow", "twin_fast"],
        format_func={
            "single":     "Monohélice  (K = 0.20)",
            "twin_slow":  "Bihélice lento/carga  (K = 0.10)",
            "twin_fast":  "Bihélice rápido/naval  (K = 0.00)",
        }.get,
        help="Constante K del criterio de Keller"
    )
    curvas_burrill  = list(VerificacionCavitacion.BURRILL_CURVES.keys())
    idx_def         = min(3, len(curvas_burrill) - 1)  # 5% recomendado para mercante
    nombre_curva_b  = col_cv3.selectbox(
        "Curva de Burrill-Emerson", curvas_burrill, index=idx_def,
        help="Límite de carga de cavitación.  Recomendado 5% para buques mercantes"
    )

    if st.button("🔬 Verificar Cavitación", type="secondary"):
        try:
            verif  = VerificacionCavitacion(rho=rho)
            cav_r  = verif.calcular(
                T_N          = resultado["thrust_bruto_N"],
                D            = resultado["D_m"],
                PD_ratio     = resultado["PD"],
                n_rps        = resultado["n_rps"],
                va           = resultado["va_ms"],
                Z            = resultado["Z"],
                h0           = h0,
                ear_actual   = resultado["EAR"],
                tipo_buque   = tipo_buque_cav,
                nombre_curva = nombre_curva_b,
            )
            st.session_state["cav_result"] = cav_r
            # Guardar en resultado para incluirlo en el PDF
            resultado["_cav"] = cav_r
        except Exception as e:
            st.error(f"Error en verificación: {e}")

    if "cav_result" in st.session_state:
        cav_r = st.session_state["cav_result"]
        st.divider()

        # A) Keller
        st.markdown("#### A) Criterio de Keller (1966)")
        ck1, ck2, ck3 = st.columns(3)
        ck1.metric("EAR mínima requerida",  f"{cav_r['ear_min_keller']:.4f}")
        ck2.metric("EAR actual del diseño", f"{cav_r['ear_actual']:.4f}")
        if cav_r["ok_keller"]:
            ck3.success("✅ CUMPLE — Keller")
        else:
            ck3.error("❌ INSUFICIENTE — Keller\nAumente el EAR en Geometría de Hélice")

        # B) Burrill
        st.markdown("#### B) Criterio de Burrill-Emerson")
        cb1, cb2, cb3, cb4 = st.columns(4)
        cb1.metric("Núm. cavitación σb", f"{cav_r['sigma_b']:.4f}")
        cb2.metric("τc límite (curva)",  f"{cav_r['tau_c_lim']:.4f}")
        cb3.metric("τc real",            f"{cav_r['tau_c_real']:.4f}")
        if cav_r["ok_burrill_carga"]:
            cb4.success("✅ Carga dentro del límite")
        else:
            cb4.error("❌ Carga excede el límite")

        cb5, cb6, _ = st.columns(3)
        cb5.metric("EAR mínima (Burrill)", f"{cav_r['ear_min_burrill']:.4f}")
        if cav_r["ok_burrill_ear"]:
            cb6.success("✅ EAR suficiente (Burrill)")
        else:
            cb6.warning("⚠ EAR insuficiente (Burrill)")

        # C) Velocidad de punta
        st.markdown("#### C) Velocidad en la Punta")
        ct1, ct2, ct3 = st.columns(3)
        ct1.metric("V_tip", f"{cav_r['vtip_ms']:.2f} m/s")
        ct2.metric("V_tip", f"{cav_r['vtip_kt']:.2f} nudos")
        nivel = cav_r["nivel_vtip"]
        mensajes = {
            "BAJA":  "✅ Riesgo BAJO — cavitación de punta reducida",
            "MEDIA": "⚠ Riesgo MEDIO — aceptable con buen diseño de pala",
            "ALTA":  "❌ Riesgo ALTO — probable cavitación, vibración y erosión",
        }
        if nivel == "BAJA":
            ct3.success(mensajes[nivel])
        elif nivel == "MEDIA":
            ct3.warning(mensajes[nivel])
        else:
            ct3.error(mensajes[nivel])

# ── Pestaña: PDF ─────────────────────────────────────────────────────────────
with tab_pdf:
    st.markdown("### Generar Reporte PDF")
    st.markdown(
        "El reporte incluye todos los parámetros de optimización, "
        "las gráficas y —si fue ejecutada— la verificación de cavitación.  \n"
        "**Vaya primero a la pestaña 📈 Gráficas** para generarlas antes de descargar."
    )

    if st.button("📄 Generar PDF", type="secondary"):
        try:
            pdf_bytes = _generar_pdf(
                resultado,
                st.session_state.get("img1"),
                st.session_state.get("img2"),
            )
            st.download_button(
                label="⬇️ Descargar  reporte_helice.pdf",
                data=pdf_bytes,
                file_name="reporte_helice.pdf",
                mime="application/pdf",
            )
            st.success("PDF generado correctamente. Haga clic en el botón de descarga.")
        except ImportError:
            st.error(
                "ReportLab no está instalado.  \n"
                "Ejecute: `pip install reportlab`"
            )
        except Exception as e:
            st.error(f"Error generando PDF: {e}")
