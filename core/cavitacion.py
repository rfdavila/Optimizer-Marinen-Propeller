"""
================================================================================
  core/cavitacion.py
  MÓDULO DE CAVITACIÓN: Criterios de Keller, Burrill y velocidad de punta
================================================================================
  La cavitación es la vaporización local del agua cuando la presión cae
  por debajo de la presión de vapor. En hélices puede erosionar las palas,
  generar ruido y reducir el empuje.

  Este módulo implementa tres criterios de verificación:
    A) Criterio de Keller        → calcula EAR mínima para evitar cavitación
    B) Criterio de Burrill       → basado en el número de cavitación σb y la
                                   curva límite de Burrill-Emerson
    C) Velocidad en la punta     → limita el tip speed (erosión por cavitación)

  REGLA DE ORO: Sin print(), input() ni plt.show(). Todos los resultados se
                retornan en diccionarios para que app/ los presente al usuario.

  Referencias:
    - Keller (1966). Scheepsschroeven. Schip en Werf.
    - Burrill & Emerson (1963). Propeller cavitation: further tests.
    - Birk (2019). Fundamentals of Ship Hydrodynamics. Wiley, Capítulos 45-46.
================================================================================
"""

from math import pi, sqrt


class VerificacionCavitacion:
    """
    Realiza los tres criterios estándar de verificación de cavitación para
    una hélice en condiciones de operación dadas.

    Constantes de clase (valores por defecto)
    -----------------------------------------
    P_VAPOR_PA : float  Presión de vapor del agua de mar a 15 °C [Pa]
    P_ATM_PA   : float  Presión atmosférica estándar [Pa]
    G          : float  Aceleración gravitacional [m/s²]
    VTIP_BAJA  : float  Umbral inferior de velocidad en punta [m/s]
    VTIP_ALTA  : float  Umbral superior de velocidad en punta [m/s]
    """

    # ── Constantes físicas por defecto ────────────────────────────────────────
    P_VAPOR_PA = 1704.0      # presión de vapor agua de mar 15 °C [Pa]
    P_ATM_PA   = 101_325.0   # presión atmosférica estándar [Pa]
    G          = 9.81        # aceleración gravitacional [m/s²]
    VTIP_BAJA  = 30.0        # m/s → riesgo bajo de cavitación de punta
    VTIP_ALTA  = 46.0        # m/s → riesgo alto de cavitación de punta

    # ── Curvas límite de Burrill-Emerson (coeficientes a, b, c) ─────────────
    # Forma de la curva: τc_lim = a · σb^b + c
    # Cada entrada es una tupla (a, b, c) para la curva correspondiente.
    BURRILL_CURVES = {
        '30% cavitación (Burrill-Emerson, model tests)':  ( 0.910,  0.335, -0.346),
        '20% cavitación (Burrill-Emerson, model tests)':  ( 1.147,  0.195, -0.672),
        '10% cavitación – buques navales / rápidos':      ( 1.267,  0.126, -0.912),
        '5% cavitación – buques mercantes (recomendado)': ( 0.715,  0.184, -0.437),
        '2.5% cavitación (Burrill-Emerson, model tests)': ( 0.611,  0.189, -0.372),
        'Límite inferior (remolcadores, trawlers)':       ( 0.527,  0.155, -0.324),
        'Límite superior – buques mercantes':             ( 0.304,  0.497, -0.034),
        'Límite superior – buques navales':               ( 0.464,  0.497, -0.089),
    }

    # ── Constructor ───────────────────────────────────────────────────────────
    def __init__(self, rho: float = 1026.021, p_atm: float = None, p_vapor: float = None):
        """
        Parámetros
        ----------
        rho     : densidad del fluido [kg/m³]. Por defecto agua de mar 15 °C.
        p_atm   : presión atmosférica [Pa].   None → usa P_ATM_PA = 101 325 Pa.
        p_vapor : presión de vapor [Pa].       None → usa P_VAPOR_PA = 1 704 Pa.
        """
        self.rho     = rho
        self.p_atm   = p_atm   if p_atm   is not None else self.P_ATM_PA
        self.p_vapor = p_vapor if p_vapor is not None else self.P_VAPOR_PA

    # ── Método auxiliar: presión estática ─────────────────────────────────────
    def _presion_estatica(self, h0: float) -> float:
        """
        Calcula la presión total en el eje de la hélice.

        p0 = p_atm + ρ·g·h0

        Parámetros
        ----------
        h0 : float  Inmersión del eje de la hélice [m].

        Retorna
        -------
        float : presión estática total [Pa].
        """
        return self.p_atm + self.rho * self.G * h0

    # ── A) Criterio de Keller ─────────────────────────────────────────────────
    def keller(self, T_N: float, D: float, Z: int, h0: float,
               tipo_buque: str = 'twin_fast') -> float:
        """
        Calcula la EAR mínima requerida según el criterio de Keller (1966).

        Fórmula:
            EAR_min = (1.3 + 0.3·Z)·T / [(p0 - pv)·D²] + K

        Donde K depende del tipo de embarcación:
            'single'    → K = 0.20  (monohélice, máximo estela)
            'twin_slow' → K = 0.10  (bihélice lento)
            'twin_fast' → K = 0.00  (bihélice rápido, mínimo estela)

        Parámetros
        ----------
        T_N       : Empuje [N].
        D         : Diámetro de la hélice [m].
        Z         : Número de palas.
        h0        : Inmersión del eje [m].
        tipo_buque: Tipo de embarcación (clave del diccionario K_valores).

        Retorna
        -------
        float : EAR mínima adimensional [-].
        """
        K_valores = {'single': 0.2, 'twin_slow': 0.1, 'twin_fast': 0.0}
        K  = K_valores.get(tipo_buque, 0.0)
        p0 = self._presion_estatica(h0)
        dp = p0 - self.p_vapor          # diferencia de presión efectiva [Pa]
        return ((1.3 + 0.3 * Z) * T_N) / (dp * D**2) + K

    # ── B) Criterio de Burrill ────────────────────────────────────────────────
    def burrill(self, T_N: float, D: float, PD_ratio: float, n_rps: float,
                va: float, h0: float, nombre_curva: str) -> dict:
        """
        Calcula parámetros de cavitación según Burrill-Emerson.

        Metodología (Birk 2019, Cap. 46):
          1. Velocidad representativa en radio 0.7R:
             v1 = sqrt(Va² + (π·n·0.7·D)²)
          2. Presión dinámica de referencia:
             q07 = 0.5·ρ·v1²
          3. Número de cavitación del propulsor:
             σb = (p0 - pv) / q07
          4. Carga de empuje límite desde la curva de Burrill:
             τc_lim = a·σb^b + c
          5. Área proyectada mínima:
             Ap_min = T / (q07·τc_lim)
          6. Conversión a EAR mínima con factor de forma tf:
             tf = 1.067 - 0.229·(P/D)    (Troost, 1951)
             Ad_min = Ap_min / tf
             EAR_min = Ad_min / (π·D²/4)

        Parámetros
        ----------
        T_N         : Empuje [N].
        D           : Diámetro [m].
        PD_ratio    : Relación paso-diámetro P/D [-].
        n_rps       : Velocidad de giro [rev/s].
        va          : Velocidad de avance [m/s].
        h0          : Inmersión del eje [m].
        nombre_curva: Clave de BURRILL_CURVES (ej. '5% cavitación...').

        Retorna
        -------
        dict con claves:
            ear_min_burrill : float  EAR mínima requerida [-]
            sigma_b         : float  Número de cavitación del propulsor [-]
            tau_c_lim       : float  Carga límite de la curva de Burrill [-]
            tau_c_real      : float  Carga real del propulsor (en disco) [-]
        """
        # Velocidad representativa en radio 0.7R
        v1  = sqrt(va**2 + (pi * n_rps * 0.7 * D)**2)
        # Presión dinámica de referencia en el radio 0.7R
        q07 = 0.5 * self.rho * v1**2

        p0      = self._presion_estatica(h0)
        sigma_b = (p0 - self.p_vapor) / q07   # número de cavitación adimensional

        # Coeficientes de la curva de Burrill seleccionada
        a, b, c   = self.BURRILL_CURVES[nombre_curva]
        tau_c_lim = a * sigma_b**b + c         # carga límite para esta curva

        # Carga real del propulsor (usando área del disco completo)
        Ap_disco   = pi * D**2 / 4.0
        tau_c_real = T_N / (q07 * Ap_disco)

        # Área proyectada mínima para cumplir el límite de Burrill
        Ap_min = T_N / (q07 * tau_c_lim)

        # Factor de forma tf: convierte área proyectada → área desarrollada
        tf = 1.067 - 0.229 * PD_ratio
        if tf <= 0:
            tf = 0.1   # protección contra valores inválidos de P/D muy alto

        # EAR mínima = área desarrollada mínima / área del disco
        Ad_min  = Ap_min / tf
        Ao      = pi * D**2 / 4.0
        ear_min = Ad_min / Ao

        return {
            'ear_min_burrill': ear_min,
            'sigma_b':         sigma_b,
            'tau_c_lim':       tau_c_lim,
            'tau_c_real':      tau_c_real,
        }

    # ── C) Velocidad en la punta ──────────────────────────────────────────────
    def velocidad_punta(self, n_rps: float, D: float) -> dict:
        """
        Calcula la velocidad en la punta de la pala y la clasifica.

        Fórmula: Vtip = π · n · D

        Clasificación:
            BAJA  : Vtip < 30 m/s  → riesgo muy reducido
            MEDIA : 30 ≤ Vtip ≤ 46 → riesgo moderado, aceptable con buen diseño
            ALTA  : Vtip > 46 m/s  → probable cavitación de punta

        Parámetros
        ----------
        n_rps : float  Velocidad de giro [rev/s].
        D     : float  Diámetro de la hélice [m].

        Retorna
        -------
        dict con claves:
            vtip_ms       : float  Velocidad en la punta [m/s]
            vtip_kt       : float  Velocidad en la punta [nudos]
            clasificacion : str    'BAJA', 'MEDIA' o 'ALTA'
        """
        vtip = pi * n_rps * D

        if vtip < self.VTIP_BAJA:
            clasificacion = 'BAJA'
        elif vtip <= self.VTIP_ALTA:
            clasificacion = 'MEDIA'
        else:
            clasificacion = 'ALTA'

        return {
            'vtip_ms':       vtip,
            'vtip_kt':       vtip * 1.944,   # conversión m/s → nudos
            'clasificacion': clasificacion,
        }

    # ── Método unificado: calcula TODOS los criterios a la vez ───────────────
    def calcular(self, T_N: float, D: float, PD_ratio: float, n_rps: float,
                 va: float, Z: int, h0: float, ear_actual: float,
                 tipo_buque: str, nombre_curva_burrill: str) -> dict:
        """
        Ejecuta los tres criterios de cavitación y retorna un diccionario
        completo con todos los resultados y estados de verificación.

        Parámetros
        ----------
        T_N                 : Empuje [N].
        D                   : Diámetro de la hélice [m].
        PD_ratio            : Relación paso-diámetro P/D [-].
        n_rps               : Velocidad de giro [rev/s].
        va                  : Velocidad de avance Va [m/s].
        Z                   : Número de palas.
        h0                  : Inmersión del eje [m].
        ear_actual          : EAR de la hélice seleccionada [-].
        tipo_buque          : 'single', 'twin_slow' o 'twin_fast'.
        nombre_curva_burrill: Clave de BURRILL_CURVES.

        Retorna
        -------
        dict con todas las variables calculadas y campos de estado booleano.
        """
        p0          = self._presion_estatica(h0)
        ear_keller  = self.keller(T_N, D, Z, h0, tipo_buque)
        bur         = self.burrill(T_N, D, PD_ratio, n_rps, va, h0, nombre_curva_burrill)
        vtip_res    = self.velocidad_punta(n_rps, D)

        return {
            # ── Datos de entrada ──────────────────────────────────────────────
            'T_kN':               T_N / 1000.0,
            'D_m':                D,
            'PD_ratio':           PD_ratio,
            'rpm_helice':         n_rps * 60.0,
            'va_ms':              va,
            'Z_palas':            Z,
            'h0_m':               h0,
            'ear_actual':         ear_actual,
            'tipo_buque':         tipo_buque,
            'nombre_curva':       nombre_curva_burrill,
            'p_atm_Pa':           self.p_atm,
            'p_vapor_Pa':         self.p_vapor,
            'p0_Pa':              p0,

            # ── A) Keller ─────────────────────────────────────────────────────
            'ear_min_keller':     ear_keller,
            'ok_keller':          ear_actual >= ear_keller,

            # ── B) Burrill ────────────────────────────────────────────────────
            'ear_min_burrill':    bur['ear_min_burrill'],
            'sigma_b':            bur['sigma_b'],
            'tau_c_lim':          bur['tau_c_lim'],
            'tau_c_real':         bur['tau_c_real'],
            'ok_burrill_carga':   bur['tau_c_real'] <= bur['tau_c_lim'],
            'ok_burrill_ear':     ear_actual >= bur['ear_min_burrill'],

            # ── C) Velocidad en punta ─────────────────────────────────────────
            'vtip_ms':            vtip_res['vtip_ms'],
            'vtip_kt':            vtip_res['vtip_kt'],
            'nivel_vtip':         vtip_res['clasificacion'],
        }
