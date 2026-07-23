# -*- coding: utf-8 -*-
"""
================================================================================
  OPTIMIZACIÓN DE INVENTARIOS EN CADENA DE SUMINISTRO DE CONSTRUCCIÓN
  Modelo (Q, r) / EOQ-ROP  —  Hopp, W. & Spearman, M., "Factory Physics" (3ra ed.)
--------------------------------------------------------------------------------
  VERSIÓN WEB (Streamlit) — migración 1:1 desde optimizacion_inventarios_gui.py
  ---------------------------------------------------------------------------
    · Barra lateral : parámetros del material y opciones de la política
    · Área principal: tarjetas de datos generales, tabla de políticas óptimas
                      y curva de trade-off Capital Inmovilizado vs Fill Rate
    · El núcleo de cálculo es IDÉNTICO al de la versión de escritorio: no se
      ha modificado ninguna fórmula del modelo.

  NOTAS TÉCNICAS IMPORTANTES (leer):
  -------------------------------------------------------------------------
  1) FILL RATE vs. NIVEL DE SERVICIO DE CICLO (CSL):
     Se usa z = norm.ppf(fill_rate), aproximación del nivel de servicio de
     ciclo (probabilidad de NO tener quiebre en un ciclo). Es la aproximación
     estándar usada en la práctica. La opción «fill rate exacto» resuelve en
     cambio F = 1 - σ·L(z)/Q por bisección.

  2) CONSISTENCIA DE UNIDADES EN EL EOQ:
     h se ingresa en [moneda/unidad/día]; para usar demanda ANUAL (D=d·365)
     se anualiza también h (h_anual = h·365), de modo que
     EOQ = sqrt(2·A·D_anual / h_anual) = sqrt(2·A·d/h),  dimensionalmente
     correcto.

  Ejecución local :  streamlit run app_inventarios.py
  Requisitos      :  streamlit, numpy, pandas, scipy, matplotlib
================================================================================
"""

import io

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm

import matplotlib
matplotlib.use("Agg")                        # backend no interactivo (servidor)
from matplotlib.figure import Figure

# Días efectivos por año para anualizar
DIAS_ANIO = 365

# Dimensiones y resolución con las que se exporta la gráfica al descargarla
# como PNG (independiente del tamaño con el que se ve en pantalla).
EXPORT_FIGSIZE_PULGADAS = (8, 6)   # (ancho, alto) en pulgadas
EXPORT_DPI = 300


# =============================================================================
# 1. FUNCIONES DE CÁLCULO (núcleo del modelo Factory Physics) — sin cambios
# =============================================================================

def factor_seguridad(fill_rate):
    """Factor de seguridad z = Φ⁻¹(fill_rate) (nivel de servicio de ciclo)."""
    return norm.ppf(fill_rate)


def funcion_perdida_normal(z):
    """Función de pérdida estándar normal L(z) = φ(z) - z·[1 - Φ(z)]."""
    return norm.pdf(z) - z * (1.0 - norm.cdf(z))


def z_para_fillrate_exacto(fill_rate, sigma, Q, z_min=-3.0, z_max=8.0):
    """Resuelve z para el fill rate EXACTO: F = 1 - σ·L(z)/Q (bisección)."""
    L_obj = (1.0 - fill_rate) * Q / sigma
    a, b = z_min, z_max
    for _ in range(200):
        m = 0.5 * (a + b)
        if funcion_perdida_normal(m) > L_obj:
            a = m
        else:
            b = m
    return 0.5 * (a + b)


def desviacion_demanda_leadtime(d, sigma_d, l, sigma_l):
    """σ² = ℓ·σ_D² + d²·σ_L²  ->  devuelve σ."""
    varianza = l * sigma_d**2 + d**2 * sigma_l**2
    return np.sqrt(varianza)


def stock_seguridad(z, sigma):
    """SS = z · σ  (se acota a >= 0)."""
    return max(z * sigma, 0.0)


def punto_reorden(d, l, ss):
    """ROP = d·ℓ + SS."""
    return d * l + ss


def costo_mantenimiento_efectivo(h, costo_unitario=None, tasa_oportunidad_anual=0.0):
    """h_efectivo = h_físico + h_capital ; h_capital = costo_u·(tasa/365)."""
    h_capital = 0.0
    if costo_unitario is not None and tasa_oportunidad_anual:
        h_capital = costo_unitario * (tasa_oportunidad_anual / DIAS_ANIO)
    return h + h_capital


def cantidad_eoq(A, d, h_efectivo):
    """EOQ = sqrt(2·A·D_anual / h_anual) = sqrt(2·A·d / h_efectivo)."""
    D_anual = d * DIAS_ANIO
    h_anual = h_efectivo * DIAS_ANIO
    return np.sqrt(2.0 * A * D_anual / h_anual)


def cantidad_por_frecuencia(d, frecuencia_semanal):
    """ROQ = d · (7 / frecuencia_semanal)."""
    T = 7.0 / frecuencia_semanal
    return d * T


def inventario_promedio(roq, ss):
    """I_prom = ROQ/2 + SS."""
    return roq / 2.0 + ss


def capital_inmovilizado(i_prom, costo_unitario=None):
    """C = I_prom · costo_unitario  (o I_prom si no hay costo unitario)."""
    if costo_unitario is not None:
        return i_prom * costo_unitario
    return i_prom


# =============================================================================
# 2. CONSTRUCCIÓN DE POLÍTICAS Y CURVAS DE TRADE-OFF — sin cambios
# =============================================================================

def calcular_politica(fill_rate, params, roq=None, usar_fillrate_exacto=False):
    d       = params["d"]
    sigma_d = params["sigma_d"]
    l       = params["l"]
    sigma_l = params["sigma_l"]
    A        = params["A"]
    h        = params["h"]
    costo_u  = params.get("costo_unitario")
    tasa_opp = params.get("costo_oportunidad_anual", 0.0)

    sigma = desviacion_demanda_leadtime(d, sigma_d, l, sigma_l)
    h_eff = costo_mantenimiento_efectivo(h, costo_u, tasa_opp)
    Q = cantidad_eoq(A, d, h_eff) if roq is None else roq

    if usar_fillrate_exacto:
        z = z_para_fillrate_exacto(fill_rate, sigma, Q)
    else:
        z = factor_seguridad(fill_rate)

    ss     = stock_seguridad(z, sigma)
    rop    = punto_reorden(d, l, ss)
    i_prom = inventario_promedio(Q, ss)
    capital = capital_inmovilizado(i_prom, costo_u)

    D_anual = d * DIAS_ANIO
    costo_pedido_anual  = (D_anual / Q) * A
    costo_almacen_anual = i_prom * h * DIAS_ANIO
    tasa_oport = params.get("costo_oportunidad_anual", 0.0)
    costo_oport_anual   = (capital * tasa_oport) if costo_u is not None else 0.0
    costo_total_anual   = costo_pedido_anual + costo_almacen_anual + costo_oport_anual

    return {
        "fill_rate": fill_rate,
        "z": z,
        "sigma_LT": sigma,
        "ROQ": Q,
        "SS": ss,
        "ROP": rop,
        "I_prom": i_prom,
        "capital": capital,
        "costo_pedido_anual": costo_pedido_anual,
        "costo_almacen_anual": costo_almacen_anual,
        "costo_oport_anual": costo_oport_anual,
        "costo_total_anual": costo_total_anual,
    }


def curva_tradeoff(fill_rates, params, roq=None, usar_fillrate_exacto=False):
    filas = [calcular_politica(fr, params, roq, usar_fillrate_exacto)
             for fr in fill_rates]
    return pd.DataFrame(filas)


def tabla_resumen_objetivo(params, fill_objetivo, frecuencias_semana=None,
                           usar_fillrate_exacto=False):
    """
    Construye la tabla resumen SOLO con las políticas de frecuencia fija
    ingresadas por el usuario (ej. 0.5, 1, 2 ped/sem). La política EOQ ya
    no se incluye como fila de la tabla (aunque sigue disponible como
    curva de referencia en la gráfica cuando no se ingresan frecuencias).
    Si el usuario no ingresó ninguna frecuencia, se usa el EOQ como
    respaldo para no dejar la tabla vacía.
    """
    filas = []

    if frecuencias_semana:
        for f in frecuencias_semana:
            roq = cantidad_por_frecuencia(params["d"], f)
            pol = calcular_politica(fill_objetivo, params, roq=roq,
                                    usar_fillrate_exacto=usar_fillrate_exacto)
            filas.append(_fila_resumen(f"{f} ped/sem", f, pol, params))
    else:
        pol = calcular_politica(fill_objetivo, params, roq=None,
                                usar_fillrate_exacto=usar_fillrate_exacto)
        filas.append(_fila_resumen("EOQ", None, pol, params))

    return pd.DataFrame(filas)


def _fila_resumen(etiqueta, frecuencia, pol, params):
    costo_u = params.get("costo_unitario")
    moneda = params.get("moneda", "USD")
    unidad_cap = moneda if costo_u is not None else "und"
    return {
        "Política": etiqueta,
        "Frec [ped/sem]": frecuencia if frecuencia is not None else "-",
        "ROQ [und]": round(pol["ROQ"], 1),
        "ROP [und]": round(pol["ROP"], 1),
        "SS [und]": round(pol["SS"], 1),
        "I_prom [und]": round(pol["I_prom"], 1),
        f"Capital [{unidad_cap}]": round(pol["capital"], 1),
        f"Costo total/año [{unidad_cap}]": round(pol["costo_total_anual"], 1),
        "Fill Rate [%]": round(pol["fill_rate"] * 100, 1),
    }


# =============================================================================
# 3. CONFIGURACIÓN VISUAL Y VALORES POR DEFECTO
# =============================================================================
st.set_page_config(
    page_title="Optimización de Inventarios — Cadena de Suministro",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paleta sobria: azul marino / gris ejecutivo, con rojo reservado a la
# referencia crítica (fill rate objetivo). Las curvas de trade-off conservan
# el ciclo de color por defecto de Matplotlib, igual que en el ejecutable.
COL = {
    "navy":   "#1C3144",   # azul marino — encabezados (mismo tono del Treeview)
    "slate":  "#5a6b7d",   # gris ejecutivo — texto secundario
    "line":   "#d8dee5",   # bordes tenues
    "card":   "#f7f9fb",   # fondo de tarjeta
    "accent": "#c0392b",   # rojo — fill rate objetivo / valores destacados
}

# Valores por defecto (caso Pernos de Anclaje)
DEFAULTS = {
    "material":    "Pernos de Anclaje",
    "d":           16.20,
    "sigma_d":     2.00,
    "l":           8.50,
    "sigma_l":     0.50,
    "A":           1200.0,
    "h":           0.10,
    "costo_u":     200.79,
    "fr_min":      70.0,
    "fr_max":      99.0,
    "n_puntos":    50,
    "frecuencias": "0.5, 1, 2",
    "costo_opp":   14.0,
    "fr_objetivo": 95.0,
    "moneda":      "S/",
}

# -----------------------------------------------------------------------
# Constantes de conversión de unidades para varillas/fierros de construcción
# -----------------------------------------------------------------------
LONGITUD_VARILLA_M = 9.0          # Longitud estándar de cada varilla [m]
PESO_VARILLA_9M_KG = 56.82        # Peso de 1 varilla de 32 mm x 9 m [kg]
                                  # (~6.313 kg/m de acero corrugado de 32 mm,
                                  # densidad estándar del acero 7,850 kg/m3)

st.markdown(
    f"""
    <style>
      html, body, [class*="css"] {{
          font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
      }}
      .block-container {{ padding-top: 2.0rem; padding-bottom: 2.5rem; }}
      h1, h2, h3, h4 {{ color: {COL['navy']}; letter-spacing: .2px; }}
      .inv-title {{
          font-size: 1.55rem; font-weight: 700; color: {COL['navy']};
          margin-bottom: .15rem;
      }}
      .inv-sub {{ font-size: .92rem; color: {COL['slate']}; margin-bottom: 1.0rem; }}
      .inv-block {{
          font-size: 1.05rem; font-weight: 700; color: {COL['navy']};
          border-left: 3px solid {COL['navy']}; padding-left: .55rem;
          margin: 1.1rem 0 .55rem 0;
      }}
      .inv-note {{ font-size: .82rem; color: {COL['slate']}; font-style: italic; }}
      .inv-material {{
          font-size: 1.02rem; font-weight: 700; color: {COL['navy']};
          margin: .2rem 0 .6rem 0;
      }}
      div[data-testid="stMetric"] {{
          background: {COL['card']};
          border: 1px solid {COL['line']};
          border-radius: 6px;
          padding: 12px 14px 10px 14px;
      }}
      div[data-testid="stMetricLabel"] p {{
          font-size: .74rem !important; font-weight: 700 !important;
          color: {COL['navy']} !important; text-transform: uppercase;
          letter-spacing: .3px; line-height: 1.15;
      }}
      div[data-testid="stMetricValue"] {{
          font-size: 1.18rem !important; font-weight: 700 !important;
          color: {COL['accent']} !important;
      }}
      section[data-testid="stSidebar"] {{ border-right: 1px solid {COL['line']}; }}
      div[data-testid="stDataFrame"] {{ border: 1px solid {COL['line']}; border-radius: 6px; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# 4. PANEL DE CONTROL (BARRA LATERAL)
# =============================================================================
def leer_parametros_sidebar():
    """Recrea el panel izquierdo del ejecutable Tkinter. Devuelve
    (params, usar_exacto, calcular). Replica las mismas validaciones y el
    mismo recorte a 0.9999 del fill rate."""
    sb = st.sidebar
    sb.markdown("<div class='inv-title' style='font-size:1.05rem'>PARÁMETROS DE ENTRADA</div>",
                unsafe_allow_html=True)
    sb.caption("Modelo (Q, r) / EOQ-ROP — Factory Physics")

    with sb.expander("Datos del material", expanded=True):
        material = st.text_input("Material / producto", value=DEFAULTS["material"])
        d = st.number_input("d — demanda media (und/día)", min_value=0.0,
                            value=DEFAULTS["d"], step=0.1, format="%.3f")
        sigma_d = st.number_input("σ_D — desv. de la demanda (und/día)", min_value=0.0,
                                  value=DEFAULTS["sigma_d"], step=0.1, format="%.3f")
        l = st.number_input("ℓ — lead time medio (días)", min_value=0.0,
                            value=DEFAULTS["l"], step=0.25, format="%.3f")
        sigma_l = st.number_input("σ_L — desv. del lead time (días)", min_value=0.0,
                                  value=DEFAULTS["sigma_l"], step=0.05, format="%.3f")

    with sb.expander("Costos", expanded=True):
        moneda = st.text_input("Símbolo de moneda", value=DEFAULTS["moneda"])
        A = st.number_input("A — costo por pedido (moneda/pedido)", min_value=0.0,
                            value=DEFAULTS["A"], step=50.0, format="%.2f")
        h = st.number_input("h — costo de mantener (moneda/und/día)", min_value=0.0,
                            value=DEFAULTS["h"], step=0.01, format="%.4f")
        usar_costo_u = st.checkbox("Considerar costo unitario del material", value=True)
        costo_u_val = st.number_input("Costo unitario (moneda/und)", min_value=0.0,
                                      value=DEFAULTS["costo_u"], step=1.0, format="%.2f",
                                      disabled=not usar_costo_u)
        costo_opp = st.number_input("Costo de oportunidad (% anual)", min_value=0.0,
                                    value=DEFAULTS["costo_opp"], step=0.5, format="%.2f")

    with sb.expander("Política y curva de trade-off", expanded=True):
        frec_txt = st.text_input("Frecuencias (ped/sem), separadas por comas",
                                 value=DEFAULTS["frecuencias"],
                                 help="Vacío = se usa la política EOQ como referencia.")
        fr_min = st.number_input("Fill rate mínimo (%)", min_value=0.0, max_value=100.0,
                                 value=DEFAULTS["fr_min"], step=1.0, format="%.2f")
        fr_max = st.number_input("Fill rate máximo (%)", min_value=0.0, max_value=100.0,
                                 value=DEFAULTS["fr_max"], step=1.0, format="%.2f")
        n_puntos = st.number_input("Número de puntos para la curva", min_value=2,
                                   value=DEFAULTS["n_puntos"], step=5)
        fr_objetivo = st.number_input("Fill rate objetivo para la tabla resumen (%)",
                                      min_value=0.0, max_value=100.0,
                                      value=DEFAULTS["fr_objetivo"], step=1.0, format="%.2f")
        usar_exacto = st.checkbox(
            "Usar fill rate EXACTO (función de pérdida normal) en vez de la "
            "aproximación z = Φ⁻¹(FR)", value=False)

    sb.markdown("")
    calcular = sb.button("Calcular y Graficar", type="primary", width="stretch")

    # ---- Validaciones equivalentes a las del ejecutable -------------------
    errores = []
    if fr_max <= fr_min:
        errores.append("El «Fill rate máximo» debe ser mayor que el «Fill rate mínimo».")
    if d <= 0:
        errores.append("La demanda media d debe ser > 0.")
    if h <= 0:
        errores.append("El costo de mantener h debe ser > 0.")
    if frec_txt.strip():
        try:
            frecuencias = [float(x.strip().replace(",", "."))
                           for x in frec_txt.replace(";", ",").split(",") if x.strip()]
        except ValueError:
            frecuencias = None
            errores.append("El campo «Frecuencias (ped/sem)» debe ser números separados por comas.")
        else:
            if any(f <= 0 for f in frecuencias):
                errores.append("Cada frecuencia (ped/sem) debe ser > 0.")
    else:
        frecuencias = None

    costo_u = float(costo_u_val) if usar_costo_u else None

    params = {
        "material": material.strip() or "Material",
        "d": float(d),
        "sigma_d": float(sigma_d),
        "l": float(l),
        "sigma_l": float(sigma_l),
        "A": float(A),
        "h": float(h),
        "costo_unitario": costo_u,
        "costo_oportunidad_anual": float(costo_opp) / 100.0,
        "moneda": moneda.strip() or "USD",
        "fill_rates": np.linspace(fr_min / 100.0, min(fr_max / 100.0, 0.9999), int(n_puntos)),
        "frecuencias_semana": frecuencias,
        "fill_objetivo": min(fr_objetivo / 100.0, 0.9999),
    }
    return params, bool(usar_exacto), bool(calcular), errores


# =============================================================================
# 5. ORQUESTACIÓN DEL CÁLCULO (equivalente a calcular_y_graficar)
# =============================================================================
def ejecutar_analisis(params, usar_exacto):
    """Construye las curvas de trade-off, los ROQ que las definen y la tabla
    resumen al fill rate objetivo. Misma secuencia que el ejecutable."""
    fill_rates = params["fill_rates"]
    frecuencias = params["frecuencias_semana"]

    curvas = {}
    roqs = {}
    if frecuencias:
        for f in frecuencias:
            roq = cantidad_por_frecuencia(params["d"], f)
            etiqueta = f"{f} ped/sem (ROQ={roq:.0f})"
            curvas[etiqueta] = curva_tradeoff(
                fill_rates, params, roq=roq, usar_fillrate_exacto=usar_exacto)
            roqs[etiqueta] = roq
    else:
        h_eff = costo_mantenimiento_efectivo(
            params["h"], params.get("costo_unitario"),
            params.get("costo_oportunidad_anual", 0.0))
        eoq = cantidad_eoq(params["A"], params["d"], h_eff)
        etiqueta = f"EOQ (ROQ={eoq:.0f})"
        curvas[etiqueta] = curva_tradeoff(
            fill_rates, params, roq=None, usar_fillrate_exacto=usar_exacto)
        roqs[etiqueta] = None      # None -> el núcleo recalcula el EOQ

    tabla = tabla_resumen_objetivo(
        params, fill_objetivo=params["fill_objetivo"],
        frecuencias_semana=frecuencias, usar_fillrate_exacto=usar_exacto)

    return {"curvas": curvas, "roqs": roqs, "tabla": tabla,
            "params": params, "usar_exacto": usar_exacto}


# =============================================================================
# 6. GRÁFICA DE TRADE-OFF
# =============================================================================
def fig_tradeoff(curvas, params, roqs, usar_exacto):
    """Curva de compensación Capital Inmovilizado vs Fill Rate, con la línea
    vertical del fill rate objetivo y los puntos de intersección evaluados
    ANALÍTICAMENTE (no interpolados), de modo que cada marcador coincide
    exactamente con el «Capital» de la tabla resumen."""
    moneda = params.get("moneda", "USD")
    unidad_cap = moneda if params.get("costo_unitario") is not None else "unidades"

    fig = Figure(figsize=(11.0, 5.2), dpi=110, facecolor="white")
    ax = fig.add_subplot(111)

    lineas = []
    for etiqueta, df in curvas.items():
        x = np.asarray(df["fill_rate"] * 100.0, dtype=float)
        y = np.asarray(df["capital"], dtype=float)
        linea, = ax.plot(x, y, marker="o", linewidth=2, markersize=3.5, label=etiqueta)
        lineas.append((linea, etiqueta, x, y))

    # --- Línea vertical del fill rate objetivo + intersecciones -------------
    fr_obj = params.get("fill_objetivo")
    if fr_obj is not None and lineas:
        fr_obj_pct = fr_obj * 100.0
        x_min = min(xs.min() for _l, _e, xs, _ys in lineas)
        x_max = max(xs.max() for _l, _e, xs, _ys in lineas)

        # Solo se dibuja si el objetivo cae dentro del rango graficado;
        # de lo contrario se distorsionarían los ejes.
        if x_min <= fr_obj_pct <= x_max:
            ax.axvline(fr_obj_pct, color="red", linestyle="--", linewidth=1.6,
                       alpha=0.9, zorder=4,
                       label=f"Fill rate objetivo = {fr_obj_pct:.1f} %")
            roqs = roqs or {}
            for linea, etiqueta, _xs, _ys in lineas:
                color = linea.get_color()
                pol = calcular_politica(fr_obj, params, roq=roqs.get(etiqueta),
                                        usar_fillrate_exacto=usar_exacto)
                y_int = pol["capital"]
                ax.plot([fr_obj_pct], [y_int], marker="o", markersize=9,
                        markerfacecolor=color, markeredgecolor="black",
                        markeredgewidth=1.2, linestyle="None", zorder=6)
                # Etiqueta fija junto al punto (visible también en el PNG)
                ax.annotate(
                    f"{y_int:,.0f}",
                    xy=(fr_obj_pct, y_int), xytext=(-8, 8),
                    textcoords="offset points", ha="right", va="bottom",
                    fontsize=7.5, color="#1b1b1b", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                              edgecolor=color, alpha=0.85, linewidth=0.8))

    ax.set_xlabel("Fill Rate [%]")
    ax.set_ylabel(f"Capital inmovilizado [{unidad_cap}]")
    ax.set_title(f"Trade-off: Capital Inmovilizado vs. Fill Rate — {params['material']}",
                 color=COL["navy"], fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(title="Política de reabastecimiento", fontsize=8)
    fig.tight_layout()
    return fig


def descargar_png(fig, nombre, etiqueta="Descargar gráfica (PNG 300 dpi)"):
    """Exporta con tamaño y resolución FIJOS (8x6 in, 300 dpi), igual que el
    ejecutable, con independencia de cómo se vea en pantalla."""
    tamano_pantalla = fig.get_size_inches().copy()
    buf = io.BytesIO()
    try:
        fig.set_size_inches(*EXPORT_FIGSIZE_PULGADAS)
        fig.savefig(buf, format="png", dpi=EXPORT_DPI, bbox_inches="tight",
                    facecolor="white")
    finally:
        fig.set_size_inches(*tamano_pantalla)
    st.download_button(etiqueta, data=buf.getvalue(), file_name=nombre,
                       mime="image/png")


# =============================================================================
# 7. TABLA DE POLÍTICAS (conversión varillas -> toneladas de acero)
# =============================================================================
def tabla_politicas_formateada(tabla, moneda):
    """Reproduce el Treeview del ejecutable.
    ROQ, ROP y SS se muestran en TONELADAS de acero (masa), convirtiendo cada
    varilla (32 mm x 9 m) mediante PESO_VARILLA_9M_KG = 56.82 kg/varilla
    (= 0.05682 ton/varilla). Junto a ROP y SS en toneladas se agrega también
    su equivalente en número de varillas/pernos."""
    filas = []
    for fila in tabla.itertuples(index=False):
        # "frecuencia" ya no se muestra, pero el DataFrame interno la conserva.
        politica, _frecuencia, roq, rop, ss, i_prom, capital, costo_total, fill_rate = fila

        roq_ton = (roq * PESO_VARILLA_9M_KG) / 1000.0
        rop_ton = (rop * PESO_VARILLA_9M_KG) / 1000.0
        ss_ton  = (ss * PESO_VARILLA_9M_KG) / 1000.0

        # roq/rop/ss ya están expresados en "und" = varillas -> conteo directo
        rop_varillas = rop
        ss_varillas = ss

        filas.append({
            "Política":                 politica,
            "ROQ [ton]":                f"{roq_ton:,.2f} ton",
            "ROP [ton]":                f"{rop_ton:,.2f} ton",
            "ROP [varillas]":           f"{rop_varillas:,.1f}",
            "SS [ton]":                 f"{ss_ton:,.2f} ton",
            "SS [varillas]":            f"{ss_varillas:,.1f}",
            "I_prom [und]":             f"{i_prom:,.1f}",
            f"Capital [{moneda}]":      f"{capital:,.1f}",
            f"Costo total/año [{moneda}]": f"{costo_total:,.1f}",
            "Fill Rate [%]":            f"{fill_rate:.1f}",
        })
    return pd.DataFrame(filas)


# =============================================================================
# 8. ENCABEZADO Y EJECUCIÓN
# =============================================================================
st.markdown(
    "<div class='inv-title'>Optimización de Inventarios — Cadena de Suministro "
    "de Construcción</div>"
    "<div class='inv-sub'>Modelo (Q, r) / EOQ-ROP bajo Factory Physics "
    "(Hopp &amp; Spearman, 3.ª ed.) · curva de compensación entre nivel de "
    "servicio y capital inmovilizado</div>",
    unsafe_allow_html=True,
)

params, usar_exacto, calcular, errores = leer_parametros_sidebar()

if "res" not in st.session_state:
    st.session_state.res = None

if calcular:
    if errores:
        st.session_state.res = None
        for msg in errores:
            st.error(msg)
    else:
        try:
            with st.spinner("Calculando políticas y curvas de trade-off…"):
                st.session_state.res = ejecutar_analisis(params, usar_exacto)
        except ZeroDivisionError:
            st.session_state.res = None
            st.error("Ocurrió una división entre cero. Revise los valores ingresados.")
        except Exception as e:
            st.session_state.res = None
            st.error(f"No se pudo completar el cálculo: {e}")

res = st.session_state.res

if res is None:
    st.info("Configure los parámetros en el panel lateral y presione "
            "«Calcular y Graficar».")
else:
    p = res["params"]
    moneda = p.get("moneda", "USD")
    costo_u = p.get("costo_unitario")

    # ------------------------------------------------------------------
    # BLOQUE 1 — Datos generales del material (tarjetas)
    # ------------------------------------------------------------------
    st.markdown("<div class='inv-block'>Datos generales del material</div>",
                unsafe_allow_html=True)
    st.markdown(
        f"<div class='inv-material'>{p['material']} &nbsp;·&nbsp; Fill rate objetivo "
        f"de la tabla: {p['fill_objetivo'] * 100:.1f} %</div>",
        unsafe_allow_html=True,
    )

    tarjetas = [
        ("d [und/día]",     f"{p['d']:g} und/día"),
        ("σ_D [und/día]",   f"{p['sigma_d']:g} und/día"),
        ("ℓ [días]",        f"{p['l']:g} días"),
        ("σ_L [días]",      f"{p['sigma_l']:g} días"),
        ("A [/pedido]",     f"{moneda} {p['A']:,.2f}"),
        ("h [/und/día]",    f"{moneda} {p['h']:,.4f}"),
        ("Costo unitario",  f"{moneda} {costo_u:,.2f}" if costo_u is not None
                            else "— (no aplica)"),
        ("Costo oportunidad", f"{p['costo_oportunidad_anual'] * 100:.2f} % anual"),
    ]
    for inicio in (0, 4):
        cols = st.columns(4)
        for col, (titulo, valor) in zip(cols, tarjetas[inicio:inicio + 4]):
            col.metric(titulo, valor)

    # Métricas derivadas útiles para el análisis (no estaban como tarjetas en
    # el ejecutable, pero se calculan con el mismo núcleo).
    sigma_lt = desviacion_demanda_leadtime(p["d"], p["sigma_d"], p["l"], p["sigma_l"])
    h_eff = costo_mantenimiento_efectivo(p["h"], costo_u,
                                         p.get("costo_oportunidad_anual", 0.0))
    eoq_ref = cantidad_eoq(p["A"], p["d"], h_eff)
    st.markdown(
        f"<div class='inv-note'>σ de la demanda durante el lead time = "
        f"{sigma_lt:,.2f} und &nbsp;·&nbsp; h efectivo = {moneda} {h_eff:,.4f} "
        f"/und/día &nbsp;·&nbsp; EOQ de referencia = {eoq_ref:,.1f} und "
        f"&nbsp;·&nbsp; Método de z: "
        f"{'fill rate exacto (función de pérdida)' if res['usar_exacto'] else 'aproximación z = Φ⁻¹(FR)'}"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------
    # BLOQUE 2 — Políticas óptimas de reabastecimiento
    # ------------------------------------------------------------------
    st.markdown("<div class='inv-block'>Políticas óptimas de reabastecimiento</div>",
                unsafe_allow_html=True)
    df_vista = tabla_politicas_formateada(res["tabla"], moneda)
    st.dataframe(df_vista, width="stretch", hide_index=True)
    st.markdown(
        f"<div class='inv-note'>ROQ, ROP y SS se expresan en toneladas de acero "
        f"convirtiendo cada varilla de 32 mm × {LONGITUD_VARILLA_M:g} m mediante "
        f"{PESO_VARILLA_9M_KG:g} kg/varilla (= 0.05682 ton/varilla). Todas las "
        f"políticas se evalúan al fill rate objetivo de "
        f"{p['fill_objetivo'] * 100:.1f} %.</div>",
        unsafe_allow_html=True,
    )

    cdesc1, cdesc2, cdesc3 = st.columns([1, 1, 3])
    with cdesc1:
        st.download_button("Descargar tabla (CSV)",
                           data=df_vista.to_csv(index=False).encode("utf-8-sig"),
                           file_name="politicas_reabastecimiento.csv", mime="text/csv")
    with cdesc2:
        st.download_button("Descargar tabla base (CSV)",
                           data=res["tabla"].to_csv(index=False).encode("utf-8-sig"),
                           file_name="politicas_base_und.csv", mime="text/csv",
                           help="Valores sin convertir, en unidades (varillas).")

    # ------------------------------------------------------------------
    # BLOQUE 3 — Curva de compensación
    # ------------------------------------------------------------------
    st.markdown("<div class='inv-block'>Curva de compensación: Fill Rate vs. "
                "Capital Inmovilizado</div>", unsafe_allow_html=True)
    fig = fig_tradeoff(res["curvas"], p, res["roqs"], res["usar_exacto"])
    st.pyplot(fig, width="stretch")

    cg1, cg2 = st.columns([1, 3])
    with cg1:
        descargar_png(fig, "trade_off_inventario.png")

    with st.expander("Valores tabulados de las curvas de trade-off"):
        marcos = []
        for etiqueta, df in res["curvas"].items():
            m = pd.DataFrame({
                "Política": etiqueta,
                "Fill Rate [%]": np.round(df["fill_rate"] * 100.0, 3),
                "z": np.round(df["z"], 4),
                "ROQ [und]": np.round(df["ROQ"], 2),
                "SS [und]": np.round(df["SS"], 2),
                "ROP [und]": np.round(df["ROP"], 2),
                "I_prom [und]": np.round(df["I_prom"], 2),
                f"Capital [{moneda}]": np.round(df["capital"], 2),
                f"Costo total/año [{moneda}]": np.round(df["costo_total_anual"], 2),
            })
            marcos.append(m)
        df_curvas = pd.concat(marcos, ignore_index=True)
        st.dataframe(df_curvas, width="stretch", hide_index=True, height=320)
        st.download_button("Descargar curvas (CSV)",
                           data=df_curvas.to_csv(index=False).encode("utf-8-sig"),
                           file_name="curvas_tradeoff.csv", mime="text/csv")

st.markdown("---")
st.markdown(
    "<div class='inv-note'>Núcleo de cálculo idéntico al ejecutable de escritorio "
    "validado (optimizacion_inventarios_gui). Marco teórico: Hopp &amp; Spearman, "
    "<i>Factory Physics</i> (3.ª ed.), modelo (Q, r) / EOQ-ROP.</div>",
    unsafe_allow_html=True,
)


# =============================================================================
# 9. ARCHIVOS DE DESPLIEGUE (Streamlit Community Cloud)
# =============================================================================
# --- .streamlit/config.toml  (fuerza el tema claro de forma permanente) -----
#
# [theme]
# primaryColor="#2980b9"
# backgroundColor="#ffffff"
# secondaryBackgroundColor="#f4f6f7"
# textColor="#2c3e50"
# font="sans serif"
#
# --- requirements.txt --------------------------------------------------------
#
# streamlit>=1.50
# numpy>=1.26
# pandas>=2.0
# scipy>=1.11
# matplotlib>=3.8
#
# --- Estructura del repositorio ---------------------------------------------
#
# mi-repo/
# ├── app_inventarios.py
# ├── requirements.txt
# └── .streamlit/
#     └── config.toml
# ============================================================================
