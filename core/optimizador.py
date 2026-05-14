"""
================================================================================
  core/optimizador.py
  MOTOR DE OPTIMIZACIÓN DE HÉLICES NAVALES — PropellerOptimizer
================================================================================

PROPÓSITO
---------
Implementa las cinco tareas clásicas de diseño de hélices marinas mediante
optimización numérica sobre los polinomios de las series sistemáticas
(Birk 2019, Caps. 48.2–48.4).

TAREAS DE DISEÑO
----------------
  Tarea 1  Diámetro óptimo D   — dados: Potencia P_D, RPM fijas, Vs
  Tarea 2  Diámetro óptimo D   — dado:  Empuje T, RPM fijas, Vs
  Tarea 3  RPM óptimas (n)     — dados: Potencia P_D, D fijo, Vs
  Tarea 4  RPM óptimas (n)     — dado:  Empuje T, D fijo, Vs
  Tarea 5  Bollard Pull / Tiro — dados: Potencia, D, Vs (puede ser 0)

FUNDAMENTO MATEMÁTICO
---------------------
La variable de diseño es P/D (relación paso-diámetro). Para cada P/D candidato
se evalúa la hélice con los polinomios de la serie y se busca el punto de
autopropulsión donde la curva K_{T,Q}(J) intersecta la curva de demanda de la
embarcación.

La demanda se parametriza mediante la constante de diseño C, que resulta de
combinar las definiciones adimensionales de K_T, K_Q y J:

  Definiciones:
    J   = Va / (n·D)                    coeficiente de avance [-]
    K_T = T / (ρ·n²·D⁴)                coeficiente de empuje [-]
    K_Q = Q / (ρ·n²·D⁵)                coeficiente de torque [-]
    η   = J·K_T / (2π·K_Q)             eficiencia en aguas abiertas [-]

  Constantes de diseño (derivadas algebraicamente para cada tarea):

    Tareas de RPM fijas → D = Va/(n·J)  →  sustituyendo en K_T y K_Q:
      C₁ = K_Q / J⁵ = P_D·η_R·n² / (2π·ρ·Va⁵)    [Tarea 1]
      C₂ = K_T / J⁴ = T·n²         / (ρ·Va⁴)      [Tarea 2]

    Tareas de D fijo    → n = Va/(D·J)  →  sustituyendo en K_T y K_Q:
      C₃ = K_Q / J³ = P_D·η_R / (2π·ρ·D²·Va³)     [Tarea 3]
      C₄ = K_T / J² = T        / (ρ·D²·Va²)        [Tarea 4]

La eficiencia η se maximiza sobre P/D ∈ [0.5, 1.4] con
scipy.optimize.minimize_scalar (método 'bounded', sin gradiente).

CORRECCIONES Y FACTORES
-----------------------
  wake_factor w     : Va = (1 - w)·Vs   — estela media del casco
  margen_servicio Δs: amplía la demanda de diseño (P_D o T) × Δs
  delta_d Δd        : J_adj = Δd·J_opt  — compensa estela no uniforme
  thrust_deduction t: T_neto = T_bruto·(1 - t)

UNIDADES INTERNAS (SI estricto)
---------------------------------
  Longitudes   [m]        Fuerzas    [N]        Velocidades  [m/s]
  RPM → rev/s  n = RPM/60            Potencia   [W]
  Densidad ρ   [kg/m³]

INTERFAZ CON app.py
-------------------
Sin print(), input() ni plt.show().
  run_optimization()              → dict con todos los resultados
  generar_datos_grafica_helice()  → arrays para gráfica KT/KQ/η vs J
  generar_datos_grafica_motor()   → arrays para gráfica curva motor

NORMA DE CODIFICACIÓN
---------------------
  Docstrings : NumPy style  (https://numpydoc.readthedocs.io)
  Estilo     : PEP 8 + anotaciones de tipo PEP 484

REFERENCIAS
-----------
  Birk, L. (2019). Fundamentals of Ship Hydrodynamics. Wiley.
  Oosterveld & Van Oossanen (1975). Wageningen B-screw series. ISP.
  Kuiper, G. (1992). The Wageningen Propeller Series. MARIN Report.
================================================================================
"""

from __future__ import annotations

from math import pi
from typing import Optional

import numpy as np
from scipy.optimize import minimize_scalar, root_scalar

from core.propulsores import SerieB, SerieKaplan

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES DE CONVERSIÓN Y LÍMITES OPERATIVOS
# ─────────────────────────────────────────────────────────────────────────────

#: 1 nudo = 1852 m / 3600 s = 0.514444 m/s
KNOTS_TO_MS: float = 0.514444

#: 1 HP (mecánico) = 745.699872 W  →  se usa 746 W (convención marina)
HP_TO_W: float = 746.0

#: Intervalo de búsqueda de raíces para J  (coeficiente de avance)
J_BRACKET: tuple[float, float] = (0.01, 1.40)

#: Límites de optimización para P/D  (relación paso-diámetro)
PD_BOUNDS: tuple[float, float] = (0.50, 1.40)


# =============================================================================
# CLASE PRINCIPAL
# =============================================================================

class PropellerOptimizer:
    """
    Optimizador de hélices para las Series B (Wageningen) y Kaplan Ka.

    Aplica el método de series sistemáticas para hallar la geometría óptima
    (P/D, D o n) que maximiza la eficiencia en aguas abiertas η, satisfaciendo
    simultáneamente las ecuaciones de equilibrio hidropropulsivo.

    Parameters
    ----------
    prop_config : dict
        Configuración de la serie. Claves requeridas:

        - ``'tipo'``   : ``'SERIE_B'`` o ``'KAPLAN'``
        - ``'nombre'`` : str, nombre descriptivo para informes

        Adicionales para Kaplan:

        - ``'matriz'``     : np.ndarray, coeficientes polinómicos
        - ``'con_tobera'`` : bool, True = empuje hélice + tobera 19A
        - ``'z'``          : int, número de palas (fijo en Kaplan)
        - ``'ear'``        : float, EAR fija de la serie Kaplan

    rho : float, optional
        Densidad del fluido [kg/m³]. Por defecto agua de mar 15 °C.
    re : float, optional
        Número de Reynolds en sección 0.75R [-].
        Activa la corrección de Reynolds de Serie B si Re > 2×10⁶.
    """

    def __init__(
        self,
        prop_config: dict,
        rho: float = 1026.021,
        re: float = 1e6,
    ) -> None:
        self.prop_config = prop_config
        self.rho = rho
        self.re = re

    # =========================================================================
    # MÉTODOS PRIVADOS — Construcción de instancia de hélice
    # =========================================================================

    def _crear_helice(self, pfac: float, z: int, afac: float) -> SerieB | SerieKaplan:
        """
        Instancia la clase de hélice correcta según ``prop_config['tipo']``.

        Principio de inyección de dependencias: la lógica de optimización
        no distingue entre series; solo llama a ``.evaluate(J)``.

        Parameters
        ----------
        pfac : float
            Relación paso-diámetro P/D del candidato evaluado.
        z : int
            Número de palas (ignorado en Kaplan).
        afac : float
            EAR Ae/Ao (ignorada en Kaplan).

        Returns
        -------
        SerieB or SerieKaplan

        Raises
        ------
        ValueError
            Si ``prop_config['tipo']`` no es reconocido.
        """
        tipo = self.prop_config["tipo"]
        if tipo == "SERIE_B":
            return SerieB(pfac, afac, z, self.re)
        if tipo == "KAPLAN":
            return SerieKaplan(
                pfac=pfac,
                matriz_coeficientes=self.prop_config["matriz"],
                con_tobera=self.prop_config["con_tobera"],
            )
        raise ValueError(
            f"Tipo de hélice no reconocido: '{tipo}'. Use 'SERIE_B' o 'KAPLAN'."
        )

    # =========================================================================
    # MÉTODOS PRIVADOS — Resolución del punto de autopropulsión
    # =========================================================================

    def _find_equilibrium_J(
        self,
        prop: SerieB | SerieKaplan,
        task: int,
        C: float,
    ) -> Optional[float]:
        """
        Resuelve la ecuación implícita del punto de autopropulsión.

        La hélice opera en equilibrio cuando su curva K_{T,Q}(J) intersecta
        la curva de demanda de la embarcación (función lineal en J^n de
        pendiente C). El residuo a anular es:

          Tarea 1:  K_Q(J) − C·J⁵ = 0
          Tarea 2:  K_T(J) − C·J⁴ = 0
          Tarea 3:  K_Q(J) − C·J³ = 0
          Tarea 4:  K_T(J) − C·J² = 0

        El método de Brent (brentq) garantiza convergencia si existe una raíz
        en J ∈ [0.01, 1.40].

        Parameters
        ----------
        prop : SerieB or SerieKaplan
        task : int
            Número de tarea (1 a 4).
        C : float
            Constante de diseño de la tarea.

        Returns
        -------
        float or None
            J en el punto de autopropulsión, o ``None`` si no converge.
        """
        _task = task  # captura por closure (evita bug si task se reasigna)

        def residuo(j: float) -> float:
            kt, kq, _ = prop.evaluate(j)
            if _task == 1:
                return kq - C * j**5
            if _task == 2:
                return kt - C * j**4
            if _task == 3:
                return kq - C * j**3
            return kt - C * j**2  # task 4

        try:
            sol = root_scalar(residuo, bracket=list(J_BRACKET), method="brentq")
            return sol.root
        except ValueError:
            return None  # raíz no existe en J_BRACKET (P/D fuera de rango)

    # =========================================================================
    # MÉTODOS PRIVADOS — Funciones objetivo para scipy
    # =========================================================================

    def _objective_efficiency(
        self,
        pfac: float,
        task: int,
        C: float,
        z: int,
        afac: float,
    ) -> float:
        """
        Función objetivo para maximize_scalar en tareas 1–4.

        Retorna −η (scipy minimiza, nosotros maximizamos η).
        Si no existe punto de equilibrio para este P/D, devuelve 1.0
        (equivale a η = 0, máxima penalización).

        Parameters
        ----------
        pfac : float
        task : int
        C : float
        z : int
        afac : float

        Returns
        -------
        float
            −η en el punto de autopropulsión, o 1.0 si no converge.
        """
        prop = self._crear_helice(pfac, z, afac)
        j_eq = self._find_equilibrium_J(prop, task, C)
        if j_eq is None:
            return 1.0
        _, _, efic = prop.evaluate(j_eq)
        return -efic

    def _objective_task5(
        self,
        pfac: float,
        pd_w: float,
        d: float,
        va: float,
        z: int,
        afac: float,
        eta_r: float,
    ) -> float:
        """
        Función objetivo de la Tarea 5: maximiza el empuje bruto.

        Retorna −T [N] (scipy minimiza; nosotros maximizamos T).

        **Bollard puro** (Va = 0): J = 0, n se despeja de la ecuación
        de torque a velocidad nula:

          P_O = 2π·ρ·n³·D⁵·K_Q(0)  →  n = (P_O / (2π·ρ·D⁵·K_Q))^(1/3)

        **Remolque** (Va > 0): n se determina por balance entre potencia
        disponible y absorbida:

          2π·ρ·n³·D⁵·K_Q(J) = P_O,   J = Va/(n·D)

        En ambos casos: T = K_T·ρ·n²·D⁴.

        Parameters
        ----------
        pfac : float
        pd_w : float   Potencia al eje P_D [W].
        d : float      Diámetro D [m].
        va : float     Velocidad de avance Va [m/s].
        z : int
        afac : float
        eta_r : float  Eficiencia rotativa η_R [-].

        Returns
        -------
        float
            −T_bruto [N], o 1×10⁶ en caso de fallo numérico.
        """
        prop = self._crear_helice(pfac, z, afac)
        po_w = pd_w * eta_r  # potencia entregada al propulsor [W]

        if va == 0.0:
            kt, kq, _ = prop.evaluate(0.0)
            if kq <= 0:
                return 1e6
            n_rps = (po_w / (2.0 * pi * self.rho * d**5 * kq)) ** (1.0 / 3.0)
            return -(kt * self.rho * n_rps**2 * d**4)

        def balance_potencia(n_guess: float) -> float:
            j_g = va / (n_guess * d)
            _, kq, _ = prop.evaluate(j_g)
            return 2.0 * pi * self.rho * n_guess**3 * d**5 * kq - po_w

        try:
            n_rps = root_scalar(
                balance_potencia, bracket=[0.1, 100.0], method="brentq"
            ).root
            j_val = va / (n_rps * d)
            kt, _, _ = prop.evaluate(j_val)
            return -(kt * self.rho * n_rps**2 * d**4)
        except ValueError:
            return 1e6

    def _find_optimal_pitch_at_j_adj(
        self,
        j_adj: float,
        task: int,
        C: float,
        z: int,
        afac: float,
    ) -> Optional[tuple[float, float, float, float]]:
        """
        Halla el P/D de equilibrio para un J ajustado fijo (paso tras Δd).

        Con J_adj fijado, la única incógnita es P/D. Como K_T y K_Q son
        monótonamente crecientes con P/D a J constante ,existe a lo sumo
        una raíz en P/D ∈ [0.5, 1.4].

        Parameters
        ----------
        j_adj : float   Coeficiente de avance ajustado.
        task : int
        C : float
        z : int
        afac : float

        Returns
        -------
        tuple (pfac_adj, kt_adj, kq_adj, efic_adj) or None
        """
        def residual(pfac: float) -> float:
            prop = self._crear_helice(pfac, z, afac)
            kt, kq, _ = prop.evaluate(j_adj)
            if task == 1:
                return kq - C * j_adj**5
            if task == 2:
                return kt - C * j_adj**4
            if task == 3:
                return kq - C * j_adj**3
            return kt - C * j_adj**2  # task 4

        try:
            res = root_scalar(residual, bracket=list(PD_BOUNDS), method="brentq")
            pfac_adj = res.root
            prop_adj = self._crear_helice(pfac_adj, z, afac)
            kt_adj, kq_adj, efic_adj = prop_adj.evaluate(j_adj)
            return pfac_adj, kt_adj, kq_adj, efic_adj
        except ValueError:
            return None

    # =========================================================================
    # MÉTODOS PRIVADOS — Preparación de parámetros por tarea
    # =========================================================================

    def _preparar_tarea(
        self,
        task: int,
        inputs: list,
        margen_servicio: float,
        wake_factor: float,
        advertencias: list[str],
    ) -> dict:
        """
        Extrae los parámetros de ``inputs`` y calcula la constante de diseño C.

        La constante C es específica de cada tarea y resulta de eliminar las
        incógnitas dimensionales (D o n) combinando las definiciones de K_T,
        K_Q y J. Ver derivación en el docstring del módulo.

        Parameters
        ----------
        task : int             Número de tarea (1 a 4).
        inputs : list          Lista de parámetros de la tarea.
        margen_servicio : float  Factor Δs (amplía la demanda de diseño).
        wake_factor : float    Fracción de estela w.
        advertencias : list    Lista compartida; se agrega in-place.

        Returns
        -------
        dict con claves:
            ``va``        velocidad de avance Va [m/s]
            ``C``         constante de diseño adimensional
            ``n``         velocidad del eje [rev/s] (0 si no aplica)
            ``d``         diámetro D [m] (0 si no aplica)
            ``vs_nudos``  velocidad del buque [nudos]
            ``pd_w``      potencia al eje [W] (None si no aplica)
            ``eta_r``     eficiencia rotativa [-] (1.0 si no aplica)
        """
        n = d = 0.0
        pd_w = None
        eta_r = 1.0

        if task == 1:
            pd_w, rpm_helice, vs_nudos, eta_r = inputs
            n = rpm_helice / 60.0
            va = self.convertir_velocidades(vs_nudos, wake_factor)["va_ms"]
            pd_diseno = pd_w * margen_servicio
            # C₁ = K_Q/J⁵ = P_D·η_R·n² / (2π·ρ·Va⁵)
            C = (pd_diseno * eta_r * n**2) / (2.0 * pi * self.rho * va**5)

        elif task == 2:
            t_n_input, rpm_helice, vs_nudos = inputs
            n = rpm_helice / 60.0
            va = self.convertir_velocidades(vs_nudos, wake_factor)["va_ms"]
            t_n_diseno = t_n_input * margen_servicio
            # C₂ = K_T/J⁴ = T·n² / (ρ·Va⁴)
            C = (t_n_diseno * n**2) / (self.rho * va**4)
            if margen_servicio > 1.0:
                advertencias.append(
                    f"Empuje de diseño incrementado por margen de servicio "
                    f"Δs={margen_servicio:.2f}: "
                    f"T_diseno = {t_n_diseno / 1000:.2f} kN "
                    f"(original {t_n_input / 1000:.2f} kN)"
                )

        elif task == 3:
            pd_w, d, vs_nudos, eta_r = inputs
            va = self.convertir_velocidades(vs_nudos, wake_factor)["va_ms"]
            pd_diseno = pd_w * margen_servicio
            # C₃ = K_Q/J³ = P_D·η_R / (2π·ρ·D²·Va³)
            C = (pd_diseno * eta_r) / (2.0 * pi * self.rho * d**2 * va**3)

        else:  # task == 4
            t_n_input, d, vs_nudos = inputs
            va = self.convertir_velocidades(vs_nudos, wake_factor)["va_ms"]
            t_n_diseno = t_n_input * margen_servicio
            # C₄ = K_T/J² = T / (ρ·D²·Va²)
            C = t_n_diseno / (self.rho * d**2 * va**2)
            if margen_servicio > 1.0:
                advertencias.append(
                    f"Empuje de diseño con margen Δs={margen_servicio:.2f}: "
                    f"T_diseno = {t_n_diseno / 1000:.2f} kN"
                )

        return {
            "va": va, "C": C, "n": n, "d": d,
            "vs_nudos": vs_nudos, "pd_w": pd_w, "eta_r": eta_r,
        }

    # =========================================================================
    # MÉTODOS ESTÁTICOS PÚBLICOS — Conversiones y correcciones
    # =========================================================================

    @staticmethod
    def convertir_velocidades(vs_nudos: float, wake_factor: float = 0.0) -> dict:
        """
        Convierte la velocidad del buque a SI y calcula la velocidad efectiva
        de avance de la hélice Va.

        La fracción de estela w (wake fraction) refleja que la hélice trabaja
        en la estela del casco: el agua llega más lenta que el buque.

          Va = (1 − w)·Vs

        Para w = 0 (aguas abiertas): Va = Vs.

        Parameters
        ----------
        vs_nudos : float    Velocidad del buque Vs [nudos].
        wake_factor : float  Fracción de estela w [-]. Típico: 0.10–0.25.

        Returns
        -------
        dict
            ``'vs_ms'``      Vs en [m/s]
            ``'va_ms'``      Va = (1 − w)·Vs [m/s]
            ``'wake_factor'``  w usado
        """
        vs_ms = vs_nudos * KNOTS_TO_MS
        va_ms = (1.0 - wake_factor) * vs_ms
        return {"vs_ms": vs_ms, "va_ms": va_ms, "wake_factor": wake_factor}

    @staticmethod
    def aplicar_ajuste_estela(
        j_optimo: float,
        delta_d: float,
        va: float,
        n_rps: float,
        d: float,
        tarea: int,
    ) -> dict:
        """
        Aplica el factor de ajuste de estela no uniforme Δd sobre J óptimo.

        La estela real no es uniforme: la hélice opera en un campo de
        velocidades heterogéneo que degrada η respecto al valor calculado en
        aguas abiertas. Para compensarlo se incrementa el J de diseño:

          J_adj = Δd · J_opt   (Δd típicamente 1.00–1.05)

        Manteniendo Va constante, aumentar J implica:

          Tareas 1, 2  (RPM fijas)  →  reducir D:  D_adj = Va / (n · J_adj)
          Tareas 3, 4  (D fijo)     →  reducir n:  n_adj = Va / (D · J_adj)

        Parameters
        ----------
        j_optimo : float   J antes del ajuste.
        delta_d : float    Factor Δd [-]. 1.0 = sin ajuste.
        va : float         Va [m/s].
        n_rps : float      n [rev/s] en el punto óptimo.
        d : float          D [m] en el punto óptimo.
        tarea : int        1 o 2 → ajustar D;  3 o 4 → ajustar n.

        Returns
        -------
        dict
            ``'j_adj'``      J ajustado
            ``'d_adj'``      D ajustado [m]
            ``'n_adj_rps'``  n ajustado [rev/s]
        """
        j_adj = delta_d * j_optimo

        if tarea in (1, 2):
            # n conocido → D se reduce para alcanzar J_adj
            d_adj = va / (n_rps * j_adj) if (n_rps > 0 and j_adj > 0) else d
            n_adj = n_rps
        else:
            # D conocido → n se reduce para alcanzar J_adj
            n_adj = va / (d * j_adj) if (d > 0 and j_adj > 0) else n_rps
            d_adj = d

        return {"j_adj": j_adj, "d_adj": d_adj, "n_adj_rps": n_adj}

    # =========================================================================
    # MÉTODO PRINCIPAL — run_optimization
    # =========================================================================

    def run_optimization(
        self,
        task: int,
        inputs: list,
        z: int = 4,
        afac: float = 0.65,
        margen_servicio: float = 1.0,
        delta_d: float = 1.0,
        thrust_deduction: float = 0.0,
        wake_factor: float = 0.0,
    ) -> dict:
        """
        Ejecuta la optimización de hélice para la tarea de diseño indicada.

        Parameters
        ----------
        task : int
            Número de tarea (1 a 5).
        inputs : list
            Parámetros específicos. Estructura según tarea:

            +-------+-----------------------------------------------------+
            | Tarea | inputs                                              |
            +=======+=====================================================+
            |   1   | [pd_w [W], rpm_helice [RPM], vs_nudos, eta_r]      |
            +-------+-----------------------------------------------------+
            |   2   | [t_n [N], rpm_helice [RPM], vs_nudos]              |
            +-------+-----------------------------------------------------+
            |   3   | [pd_w [W], d [m], vs_nudos, eta_r]                 |
            +-------+-----------------------------------------------------+
            |   4   | [t_n [N], d [m], vs_nudos]                         |
            +-------+-----------------------------------------------------+
            |   5   | [bhp [HP], rpm_motor [RPM], eta_s, d [m],          |
            |       |  vs_nudos, t_deduction, r_tug_n [N], eta_r]        |
            +-------+-----------------------------------------------------+

        z : int, optional
            Número de palas (ignorado en Kaplan).
        afac : float, optional
            EAR Ae/Ao (ignorada en Kaplan).
        margen_servicio : float, optional
            Factor Δs ∈ [1.0, 2.0]. Amplía la demanda de diseño.
        delta_d : float, optional
            Factor de estela no uniforme Δd ∈ [1.0, 1.05].
        thrust_deduction : float, optional
            Deducción de empuje t ∈ [0.0, 0.20].
        wake_factor : float, optional
            Fracción de estela w ∈ [0.0, 0.30].

        Returns
        -------
        dict
            Parámetros de hélice, condición de operación, potencia/empuje,
            tren propulsivo y advertencias. Ver claves en el código.

        Raises
        ------
        ValueError
            Si ``task`` no está en [1, 5].
        """
        advertencias: list[str] = []

        # ── Tarea 5 tiene flujo propio (bollard/remolque) ─────────────────────
        if task == 5:
            return self._run_task5(inputs, z, afac, wake_factor, advertencias)

        if task not in (1, 2, 3, 4):
            raise ValueError(f"Tarea {task} no válida. Use 1, 2, 3, 4 o 5.")

        # ── Extraer parámetros y constante de diseño C ────────────────────────
        params = self._preparar_tarea(task, inputs, margen_servicio, wake_factor, advertencias)
        va       = params["va"]
        C        = params["C"]
        n        = params["n"]
        d        = params["d"]
        vs_nudos = params["vs_nudos"]
        pd_w     = params["pd_w"]

        # ── Optimización de P/D: maximizar η en aguas abiertas ───────────────
        # minimize_scalar 'bounded' evalúa _objective_efficiency(pfac, ...)
        # que devuelve −η. El mínimo de −η equivale al máximo de η.
        res_opt = minimize_scalar(
            self._objective_efficiency,
            bounds=PD_BOUNDS,
            args=(task, C, z, afac),
            method="bounded",
        )
        opt_pfac = res_opt.x
        opt_prop = self._crear_helice(opt_pfac, z, afac)

        # ── Punto de autopropulsión en el P/D óptimo ─────────────────────────
        j_op = self._find_equilibrium_J(opt_prop, task, C)
        if j_op is None:
            advertencias.append(
                "No se encontró J de autopropulsión en J ∈ [0.01, 1.40]. "
                "Verifique los parámetros de entrada."
            )
            j_op = 0.5  # valor de respaldo para continuar el cálculo

        kt_op, kq_op, efic_op = opt_prop.evaluate(j_op)

        # ── Dimensiones según tarea ───────────────────────────────────────────
        if task in (1, 2):
            # n es conocida (RPM fija) → calcular D = Va / (n·J)
            d_op = va / (n * j_op) if (n > 0 and j_op > 0) else 0.0
            n_op = n
        else:
            # D es conocido → calcular n = Va / (D·J)
            n_op = va / (d * j_op) if (d > 0 and j_op > 0) else 0.0
            d_op = d

        # ── Corrección por estela no uniforme (Δd) ───────────────────────────
        # Ajusta J hacia un valor ligeramente mayor para operar fuera de la
        # zona de estela más perturbada, modificando D o n según la tarea.
        if delta_d != 1.0:
            ajuste  = self.aplicar_ajuste_estela(j_op, delta_d, va, n_op, d_op, task)
            j_adj   = ajuste["j_adj"]
            d_adj   = ajuste["d_adj"]
            n_adj   = ajuste["n_adj_rps"]

            # Con J_adj fijo, reoptimizar P/D para mantener el equilibrio
            pitch_result = self._find_optimal_pitch_at_j_adj(j_adj, task, C, z, afac)
            if pitch_result is not None:
                opt_pfac, kt_adj, kq_adj, efic_adj = pitch_result
            else:
                # Fallback: evaluar P/D original en J_adj
                kt_adj, kq_adj, efic_adj = opt_prop.evaluate(j_adj)
                advertencias.append(
                    "No se encontró P/D de equilibrio en J_adj; "
                    "se mantiene P/D óptimo original."
                )

            advertencias.append(
                f"Ajuste de estela no uniforme Δd={delta_d:.3f} aplicado: "
                f"J_opt={j_op:.4f} → J_adj={j_adj:.4f}  |  "
                f"PD_adj={opt_pfac:.4f}  |  "
                f"D_adj={d_adj:.4f} m  |  η_adj={efic_adj:.4f}"
            )
            j_op, d_op, n_op      = j_adj, d_adj, n_adj
            kt_op, kq_op, efic_op = kt_adj, kq_adj, efic_adj

        # ── Empuje bruto y neto ───────────────────────────────────────────────
        # T = K_T · ρ · n² · D⁴   (definición adimensional resuelta en T)
        thrust_bruto_n = kt_op * self.rho * n_op**2 * d_op**4
        thrust_neto_n  = thrust_bruto_n * (1.0 - thrust_deduction)
        if thrust_deduction > 0.0:
            advertencias.append(
                f"Deducción de empuje t={thrust_deduction:.3f} aplicada: "
                f"T_bruto={thrust_bruto_n / 1000:.2f} kN → "
                f"T_neto={thrust_neto_n / 1000:.2f} kN"
            )

        rpm_helice_sal = n_op * 60.0
        # Pd solo disponible en tareas con potencia como entrada (1 y 3)
        pd_kw = pd_w / 1000.0 if task in (1, 3) else None

        return {
            "tarea":            task,
            "tipo_helice":      self.prop_config.get("nombre", "N/D"),
            # ── Parámetros de hélice ──────────────────────────────────────────
            "PD":               opt_pfac,           # P/D óptimo [-]
            "D_m":              d_op,               # diámetro D [m]
            "rpm_helice":       rpm_helice_sal,      # RPM de la hélice
            "n_rps":            n_op,               # n [rev/s]
            "J":                j_op,               # coeficiente de avance [-]
            "eficiencia":       efic_op,            # η en aguas abiertas [-]
            "KT":               kt_op,              # coef. de empuje [-]
            "KQ":               kq_op,              # coef. de torque [-]
            "EAR":              afac,               # Ae/Ao del diseño [-]
            "Z":                z,                  # número de palas
            "C_diseno":         C,                  # constante de diseño
            # ── Condición de operación ────────────────────────────────────────
            "vs_nudos":         vs_nudos,           # Vs [nudos]
            "va_ms":            va,                 # Va [m/s]
            "wake_factor":      wake_factor,        # w [-]
            "thrust_deduction": thrust_deduction,   # t [-]
            "margen_servicio":  margen_servicio,    # Δs [-]
            "delta_d":          delta_d,            # Δd [-]
            # ── Potencia y empuje ─────────────────────────────────────────────
            "pd_kw":            pd_kw,              # Pd [kW] o None
            "thrust_bruto_kN":  thrust_bruto_n / 1000.0,
            "thrust_neto_kN":   thrust_neto_n  / 1000.0,
            "thrust_bruto_N":   thrust_bruto_n,     # en N → verificación cavitación
            # ── Tren propulsivo ───────────────────────────────────────────────
            "rpm_motor":        None,   # app.py lo agrega desde el contexto
            "ratio_i":          None,   # app.py calcula ratio para tareas 3-4
            # ── Metadatos ─────────────────────────────────────────────────────
            "advertencias":     advertencias,
        }

    # =========================================================================
    # MÉTODO PRIVADO — Lógica completa de la Tarea 5
    # =========================================================================

    def _run_task5(
        self,
        inputs: list,
        z: int,
        afac: float,
        wake_factor: float,
        advertencias: list[str],
    ) -> dict:
        """
        Tarea 5: Bollard Pull / Tiro de Remolque.

        Objetivo: maximizar el empuje bruto T sobre P/D con D y P_D fijos.

        El tiro neto descuenta la deducción de empuje y la resistencia propia
        del remolcador (si se indica):

          T_neto = T_bruto·(1 − t) − R_tug
          T_neto [ton-fuerza] = T_neto [N] / 9810

        Parameters
        ----------
        inputs : list
            [bhp [HP], rpm_motor [RPM], eta_s [-], d [m], vs_nudos [kn],
             t_deduction [-], r_tug_n [N], eta_r [-]]
        z : int
        afac : float
        wake_factor : float
        advertencias : list[str]   Lista compartida; se agrega in-place.

        Returns
        -------
        dict   Mismo esquema de claves que ``run_optimization``.
        """
        bhp, rpm_motor, eta_s, d, vs_nudos, t_deduction_t5, r_tug_n, eta_r = inputs

        # Convertir BHP a potencia al eje en Watts: P_D = BHP × HP_TO_W × η_s
        pd_w = bhp * HP_TO_W * eta_s

        va = self.convertir_velocidades(vs_nudos, wake_factor)["va_ms"]

        # ── Maximizar T: minimize_scalar devuelve P/D óptimo ─────────────────
        res_opt = minimize_scalar(
            self._objective_task5,
            bounds=PD_BOUNDS,
            args=(pd_w, d, va, z, afac, eta_r),
            method="bounded",
        )
        opt_pfac = res_opt.x
        opt_prop = self._crear_helice(opt_pfac, z, afac)
        po_w     = pd_w * eta_r  # potencia entregada al propulsor [W]

        # ── Punto de operación ────────────────────────────────────────────────
        if va == 0.0:
            # Bollard puro: J = 0, n se despeja de la ecuación de torque
            j_op = 0.0
            kt_op, kq_op, efic_op = opt_prop.evaluate(0.0)
            n_op = (po_w / (2.0 * pi * self.rho * d**5 * kq_op)) ** (1.0 / 3.0)
        else:
            # Remolque: balance potencia absorbida = potencia disponible
            def balance_pot(n_guess: float) -> float:
                j_g = va / (n_guess * d)
                _, kq, _ = opt_prop.evaluate(j_g)
                return 2.0 * pi * self.rho * n_guess**3 * d**5 * kq - po_w

            n_op      = root_scalar(balance_pot, bracket=[0.1, 100.0], method="brentq").root
            j_op      = va / (n_op * d)
            kt_op, kq_op, efic_op = opt_prop.evaluate(j_op)

        # ── Empuje bruto y tiro neto ──────────────────────────────────────────
        thrust_bruto_n   = kt_op * self.rho * n_op**2 * d**4
        thrust_neto_n    = thrust_bruto_n * (1.0 - t_deduction_t5) - r_tug_n
        # 1 ton-fuerza ≈ 9810 N  (g = 9.81 m/s², masa = 1000 kg)
        thrust_neto_tonf = thrust_neto_n / 9810.0

        rpm_helice_sal  = n_op * 60.0
        ratio_reductora = rpm_motor / rpm_helice_sal if rpm_helice_sal > 0 else 0.0

        return {
            "tarea":            5,
            "tipo_helice":      self.prop_config.get("nombre", "N/D"),
            # ── Parámetros de hélice ──────────────────────────────────────────
            "PD":               opt_pfac,
            "D_m":              d,
            "rpm_helice":       rpm_helice_sal,
            "n_rps":            n_op,
            "J":                j_op,
            "eficiencia":       efic_op,
            "KT":               kt_op,
            "KQ":               kq_op,
            "EAR":              afac,
            "Z":                z,
            # ── Condición de operación ────────────────────────────────────────
            "vs_nudos":         vs_nudos,
            "va_ms":            va,
            "wake_factor":      wake_factor,
            "thrust_deduction": t_deduction_t5,
            # ── Potencia y empuje ─────────────────────────────────────────────
            "pd_kw":            pd_w / 1000.0,
            "bhp_hp":           bhp,
            "thrust_bruto_kN":  thrust_bruto_n / 1000.0,
            "thrust_neto_tonf": thrust_neto_tonf,
            "thrust_bruto_N":   thrust_bruto_n,   # en N para verificación de cavitación
            # ── Tren propulsivo ───────────────────────────────────────────────
            "rpm_motor":        rpm_motor,
            "ratio_i":          ratio_reductora,
            # ── Extras ───────────────────────────────────────────────────────
            "resistencia_kN":   r_tug_n / 1000.0,
            "advertencias":     advertencias,
        }

    # =========================================================================
    # MÉTODOS PÚBLICOS — Datos para gráficas (sin plt.show)
    # =========================================================================

    def generar_datos_grafica_helice(
        self,
        resultado: dict,
        j_min: float = 0.01,
        j_max: float = 1.20,
        n_puntos: int = 60,
    ) -> dict:
        """
        Calcula los arrays necesarios para el diagrama de aguas libres
        (KT, 10·KQ y η en función de J).

        No genera ninguna figura. La capa ``app/`` construye el gráfico
        con matplotlib o plotly a partir de estos arrays.

        Parameters
        ----------
        resultado : dict    Salida de ``run_optimization()``.
        j_min : float       Límite inferior del rango de J.
        j_max : float       Límite superior del rango de J.
        n_puntos : int      Puntos de discretización.

        Returns
        -------
        dict
            ``'j_arr'``         np.ndarray, rango de J
            ``'kt_arr'``        np.ndarray, K_T(J)
            ``'kq10_arr'``      np.ndarray, 10·K_Q(J)
            ``'efic_arr'``      np.ndarray, η(J)
            ``'j_op'``          float, J en el punto de operación
            ``'kt_op'``         float, K_T en el punto de operación
            ``'kq_op'``         float, K_Q en el punto de operación
            ``'efic_op'``       float, η en el punto de operación
            ``'req_curve_arr'`` np.ndarray o None, curva de demanda
            ``'task'``          int
            ``'PD'``            float, P/D óptimo
        """
        opt_pfac = resultado["PD"]
        afac     = resultado["EAR"]
        z        = resultado["Z"]
        task     = resultado["tarea"]
        j_op     = resultado["J"]
        C        = resultado.get("C_diseno")

        prop     = self._crear_helice(opt_pfac, z, afac)
        j_arr    = np.linspace(j_min, j_max, n_puntos)
        kt_arr, kq_arr, efic_arr = prop.evaluate(j_arr)
        kq10_arr = kq_arr * 10.0  # factor ×10 convencional en diagramas de hélice

        kt_op, kq_op, efic_op = prop.evaluate(j_op)

        # Curva de demanda: intersecta las curvas K_T / K_Q en el punto de operación
        req_curve_arr = None
        if C is not None and task in (1, 2, 3, 4):
            if task == 1:
                req_curve_arr = 10.0 * C * j_arr**5   # escala ×10 para superimponerla a 10·KQ
            elif task == 2:
                req_curve_arr = C * j_arr**4
            elif task == 3:
                req_curve_arr = 10.0 * C * j_arr**3
            elif task == 4:
                req_curve_arr = C * j_arr**2

        return {
            "j_arr":         j_arr,
            "kt_arr":        kt_arr,
            "kq10_arr":      kq10_arr,
            "efic_arr":      efic_arr,
            "j_op":          j_op,
            "kt_op":         kt_op,
            "kq_op":         kq_op,
            "efic_op":       efic_op,
            "req_curve_arr": req_curve_arr,
            "task":          task,
            "PD":            opt_pfac,
        }

    def generar_datos_grafica_motor(
        self,
        datos_motor: dict,
        bhp_op: float,
        rpm_op: float,
    ) -> dict:
        """
        Genera los arrays para la gráfica de curva de potencia del motor.

        Parameters
        ----------
        datos_motor : dict
            Datos del motor (de ``gestor_motores``).
            Debe tener ``'performance_curve'`` con ``'rpm'`` y ``'power_hp'``.
        bhp_op : float   Potencia en el punto de operación [HP].
        rpm_op : float   RPM en el punto de operación.

        Returns
        -------
        dict
            ``'rpm_arr'``      list, RPM de la curva
            ``'power_arr'``    list, potencia [HP]
            ``'rpm_op'``       float
            ``'power_op_hp'``  float
            ``'nombre_motor'`` str
        """
        curva = datos_motor.get("performance_curve", {})
        return {
            "rpm_arr":      curva.get("rpm",      []),
            "power_arr":    curva.get("power_hp", []),
            "rpm_op":       rpm_op,
            "power_op_hp":  bhp_op,
            "nombre_motor": datos_motor.get("nombre", "Motor"),
        }
