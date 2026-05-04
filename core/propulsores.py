"""
================================================================================
  core/propulsores.py
  CAPA FÍSICA: Modelos matemáticos de hélices Serie B y Serie Kaplan
================================================================================
  Contiene las clases que evalúan los coeficientes hidrodinámicos
  (KT, KQ, η) de cada familia de hélices a partir de sus polinomios
  de regresión.

  REGLA DE ORO: Este módulo no contiene print(), input() ni plt.show().
  Todas las funciones retornan diccionarios o tuplas con los resultados.

  Referencias:
    - Birk (2019). Fundamentals of Ship Hydrodynamics. Wiley.
    - Oosterveld & van Oossanen (1975). Wageningen B-Series polynomials.
    - Kuiper (1992). The Wageningen Propeller Series.
================================================================================
"""

from math import pi, log10
import numpy as np

# Importamos los coeficientes polinómicos del mismo paquete (core/)
from . import biblioteca_coeficientes as coefs


# ─────────────────────────────────────────────────────────────────────────────
# CLASE: SerieKaplan
# ─────────────────────────────────────────────────────────────────────────────

class SerieKaplan:
    """
    Modelo hidrodinámico de hélices Kaplan Ka 4-70 y Ka 5-75
    con o sin tobera aceleradora 19A.

    La serie Kaplan fue desarrollada especialmente para aplicaciones de
    bajo avance (remolcadores, pesqueros), donde la tobera aceleradora
    aumenta significativamente el empuje a bajas velocidades.

    Parámetros del constructor
    --------------------------
    pfac : float
        Relación paso-diámetro P/D (típicamente 0.5 a 1.4).
    matriz_coeficientes : np.ndarray
        Arreglo de NumPy con los coeficientes polinómicos de la serie
        (importado de biblioteca_coeficientes.py). Columnas esperadas:
          [0] exp_PD, [1] exp_J, [2] Ctp (hélice+tobera), [3] Ctn (tobera), [4] Cq
    con_tobera : bool
        True  → empuje total sistema (hélice + tobera): KT = KTP
        False → solo empuje de las palas sin tobera:    KT = KTP – KTN
    """

    def __init__(self, pfac: float, matriz_coeficientes: np.ndarray, con_tobera: bool = True):
        self.pfac       = pfac
        self.coefs      = matriz_coeficientes   # referencia al arreglo de coeficientes
        self.con_tobera = con_tobera

    # ── Método principal de evaluación ────────────────────────────────────────
    def evaluate(self, j_input):
        """
        Calcula KT, KQ y η para uno o varios valores del coeficiente de avance J.

        Parámetros
        ----------
        j_input : float o array-like
            Coeficiente de avance J = Va / (n·D).

        Retorna
        -------
        (kt_final, kq, efic) : tuple[float|ndarray, float|ndarray, float|ndarray]
            kt_final : Coeficiente de empuje (KT)
            kq       : Coeficiente de torque (KQ)
            efic     : Eficiencia en aguas abiertas η = J·KT / (2π·KQ)
        """
        # Convertimos J a arreglo (permite vectorización con un solo valor o un rango)
        j_values   = np.atleast_1d(np.asarray(j_input, dtype=float))
        # Expandimos j para operar en broadcasting con los coeficientes (forma: N×1 vs M)
        j_expanded = j_values[:, np.newaxis]

        # Extraemos exponentes de P/D y de J desde la matriz de coeficientes
        exp_pd = self.coefs[:, 0]   # potencias de P/D en cada término polinómico
        exp_j  = self.coefs[:, 1]   # potencias de J   en cada término polinómico

        # ── Evaluación polinómica ──────────────────────────────────────────────
        # Cada término: C_i × (P/D)^exp_pd_i × J^exp_j_i
        # Sumamos todos los términos para obtener el coeficiente total
        ktp = np.sum(self.coefs[:, 2] * (self.pfac ** exp_pd) * (j_expanded ** exp_j), axis=1)
        # KTP: empuje combinado hélice + tobera (componente propulsora principal)

        ktn = np.sum(self.coefs[:, 3] * (self.pfac ** exp_pd) * (j_expanded ** exp_j), axis=1)
        # KTN: empuje aportado solo por la tobera (no disponible si no hay tobera)

        kq  = np.sum(self.coefs[:, 4] * (self.pfac ** exp_pd) * (j_expanded ** exp_j), axis=1)
        # KQ: coeficiente de torque (par motor necesario)

        # Seleccionamos el empuje según si hay tobera o no
        kt_final = ktp if self.con_tobera else (ktp - ktn)

        # ── Eficiencia en aguas abiertas ──────────────────────────────────────
        # η = J · KT / (2π · KQ)  →  protegemos contra división por cero
        efic = np.divide(
            j_values * kt_final,
            2.0 * pi * kq,
            where=(kq != 0),
            out=np.zeros_like(kq)
        )

        # Si la entrada fue un escalar o un único valor, devolvemos escalares
        if np.isscalar(j_input) or len(j_values) == 1:
            return float(kt_final[0]), float(kq[0]), float(efic[0])
        return kt_final, kq, efic


# ─────────────────────────────────────────────────────────────────────────────
# CLASE: SerieB
# ─────────────────────────────────────────────────────────────────────────────

class SerieB:
    """
    Modelo hidrodinámico de hélices Wageningen Serie B (hélice abierta).

    La Serie B es la base de referencia mundial para hélice marina comercial.
    Los polinomios de regresión (Oosterveld & van Oossanen, 1975) permiten
    calcular KT y KQ en función de J, P/D, Ae/Ao y Z.

    Incluye correcciones de Reynolds para números Re > 2×10^6 (Birk, 2019).

    Parámetros del constructor
    --------------------------
    pfac : float
        Relación paso-diámetro P/D.
    afac : float
        Relación de área expandida Ae/Ao (EAR). Típico: 0.40 – 1.05.
    z : int
        Número de palas. Típico: 2 – 7.
    re : float
        Número de Reynolds en la sección 0.75R. Típico: 1×10^6.
    """

    def __init__(self, pfac: float, afac: float, z: int, re: float):
        self.pfac = pfac
        self.afac = afac
        self.z    = z
        self.re   = re

        # Coeficientes polinómicos leídos de la biblioteca de datos
        self.dkt = coefs.COEF_SERIE_B_KT   # tabla de coeficientes para KT
        self.dkq = coefs.COEF_SERIE_B_KQ   # tabla de coeficientes para KQ

    # ── Método principal de evaluación ────────────────────────────────────────
    def evaluate(self, j_input):
        """
        Calcula KT, KQ y η para uno o varios valores de J.

        Parámetros
        ----------
        j_input : float o array-like
            Coeficiente de avance J = Va / (n·D).

        Retorna
        -------
        (kt, kq, efic) : tuple[float|ndarray, float|ndarray, float|ndarray]
        """
        j_values   = np.atleast_1d(np.asarray(j_input, dtype=float))
        j_expanded = j_values[:, np.newaxis]   # forma (N,1) para broadcasting

        # ── Evaluación del polinomio de regresión Serie B ─────────────────────
        # Forma general de cada término:  C · J^a · (P/D)^b · (Ae/Ao)^c · Z^d
        kt = np.sum(
            self.dkt[:, 0]                  # coeficiente C del término
            * (j_expanded ** self.dkt[:, 1]) # J elevado a su exponente
            * (self.pfac   ** self.dkt[:, 2]) # P/D elevado a su exponente
            * (self.afac   ** self.dkt[:, 3]) # Ae/Ao elevado a su exponente
            * (self.z      ** self.dkt[:, 4]),# Z elevado a su exponente
            axis=1
        )
        kq = np.sum(
            self.dkq[:, 0]
            * (j_expanded ** self.dkq[:, 1])
            * (self.pfac   ** self.dkq[:, 2])
            * (self.afac   ** self.dkq[:, 3])
            * (self.z      ** self.dkq[:, 4]),
            axis=1
        )

        # ── Corrección de Reynolds (solo si Re > 2×10^6) ─────────────────────
        # Las tablas estándar asumen Re = 2×10^6. Para Re mayor se aplica
        # una corrección empírica (Birk 2019, Ecuaciones 47.14 y 47.15).
        if self.re > 2e6:
            log_re = log10(self.re) - 0.301   # log10(Re / 2e6)

            delkt = (
                0.000353485
                - 0.00333758  * self.afac * j_values**2
                - 0.00478125  * self.afac * self.pfac * j_values
                + 0.000257792 * log_re**2 * self.afac * j_values**2
                + 0.0000643192* log_re * self.pfac**6 * j_values**2
                - 0.0000110636* log_re**2 * self.pfac**6 * j_values**2
                - 0.0000276305* log_re**2 * self.z * self.afac * j_values**2
                + 0.0000954   * log_re * self.z * self.afac * self.pfac * j_values
                + 0.0000032049* log_re * self.z**2 * self.afac * self.pfac**3 * j_values
            )
            delkq = (
                -0.000591412
                + 0.00696898  * self.pfac
                - 0.0000666654* self.z * self.pfac**6
                + 0.0160818   * self.afac**2
                - 0.000938091 * log_re * self.pfac
                - 0.00059593  * log_re * self.pfac**2
                + 0.0000782099* log_re**2 * self.pfac**2
                + 0.0000052199* log_re * self.z * self.afac * j_values**2
                - 0.00000088528*(log_re**2) * self.z * self.afac * self.pfac * j_values
                + 0.0000230171 * log_re * self.z * self.pfac**6
                - 0.00000184341*(log_re**2) * self.z * self.pfac**6
                - 0.00400252  * log_re * self.afac**2
                + 0.000220915 * log_re**2 * self.afac**2
            )
            kt = kt + delkt
            kq = kq + delkq

        # ── Eficiencia en aguas abiertas ──────────────────────────────────────
        efic = np.divide(
            j_values * kt,
            2.0 * pi * kq,
            where=(kq != 0),
            out=np.zeros_like(kq)
        )

        if np.isscalar(j_input) or len(j_values) == 1:
            return float(kt[0]), float(kq[0]), float(efic[0])
        return kt, kq, efic


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN DE VALIDACIÓN (pura, sin UI)
# ─────────────────────────────────────────────────────────────────────────────

def validar_configuracion_helice(prop_config: dict) -> tuple[bool, str]:
    """
    Valida que el diccionario de configuración de hélice tenga los campos
    mínimos necesarios antes de instanciar una hélice.

    Parámetros
    ----------
    prop_config : dict
        Diccionario con al menos la clave 'tipo' ('SERIE_B' o 'KAPLAN').
        Para Kaplan se esperan además: 'matriz', 'con_tobera', 'z', 'ear'.

    Retorna
    -------
    (valido, mensaje) : tuple[bool, str]
        valido  : True si la configuración es correcta.
        mensaje : Descripción del error si no es válida, cadena vacía si es OK.
    """
    tipo = prop_config.get('tipo')
    if tipo not in ('SERIE_B', 'KAPLAN'):
        return False, f"Tipo de hélice desconocido: '{tipo}'. Use 'SERIE_B' o 'KAPLAN'."
    if tipo == 'KAPLAN':
        for campo in ('matriz', 'con_tobera', 'z', 'ear'):
            if campo not in prop_config:
                return False, f"Falta el campo '{campo}' en prop_config para Serie Kaplan."
    return True, ""
