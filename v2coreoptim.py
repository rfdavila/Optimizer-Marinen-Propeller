"""
core/optimizador.py
===================
Motor de optimización de hélices — PropellerOptimizer.

Implementa las cinco tareas de diseño de hélices (Birk 2019, caps. 48.2–48.4).
Este módulo es puramente matemático: no emite output al usuario, no llama
a ``plt.show()`` y no usa ``input()``. Toda interacción con el usuario ocurre
en ``app/cli_main.py`` o ``app/streamlit_app.py``.

Tareas disponibles
------------------
1. Optimiza diámetro D      → dada Potencia, RPM, Vs
2. Optimiza diámetro D      → dado Empuje T, RPM, Vs
3. Optimiza RPM             → dada Potencia, D, Vs
4. Optimiza RPM             → dado Empuje T, D, Vs
5. Bollard Pull / Remolque  → dada Potencia, D, Vs (puede ser 0)

Novedades v7
------------
- Velocidad de entrada Vs [nudos] convertida internamente a Va = (1−w)·Vs [m/s]
- Margen de servicio Δs sobre empuje o potencia requerida
- Ajuste de estela no uniforme Δd sobre J óptimo
- Wake factor w y thrust deduction t integrados en todas las tareas

Referencias
-----------
- Birk (2019). Fundamentals of Ship Hydrodynamics. Wiley.
- HydroComp PropExpert 2005 User's Guide.
"""

from __future__ import annotations

import logging
from math import pi
from typing import Optional

import numpy as np
from scipy.optimize import minimize_scalar, root_scalar

from core.propulsores import SerieB, SerieKaplan

# ---------------------------------------------------------------------------
# Constantes del módulo
# ---------------------------------------------------------------------------
KNOTS_TO_MS: float = 0.514444   # Factor de conversión nudos → m/s (1 nudo = 1852/3600 m/s)
J_BRACKET: tuple[float, float] = (0.01, 1.40)   # Intervalo de búsqueda de J (Brent)
PD_BOUNDS: tuple[float, float] = (0.50, 1.40)   # Límites de P/D para minimize_scalar

log = logging.getLogger(__name__)


# ===========================================================================
# Clase principal
# ===========================================================================

class PropellerOptimizer:
    """Optimizador de hélices para las series B Wageningen y Kaplan Ka.

    Parameters
    ----------
    prop_config : dict
        Configuración de la hélice. Claves obligatorias:
          ``tipo``   : ``'SERIE_B'`` o ``'KAPLAN'``
          ``nombre`` : nombre descriptivo (aparece en los reportes)
        Claves adicionales requeridas solo para Kaplan:
          ``matriz``     : np.ndarray con los coeficientes polinómicos
          ``con_tobera`` : bool (True si usa tobera Kort)
          ``z``          : int  (palas fijas de la serie)
          ``ear``        : float (EAR fijo de la serie)
    rho : float
        Densidad del fluido en kg/m³. Por defecto agua de mar a 15 °C.
    re : float
        Número de Reynolds en el radio 0.75R. Afecta la corrección de Serie B.

    Raises
    ------
    ValueError
        Si ``prop_config['tipo']`` no es reconocido.
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

    # -----------------------------------------------------------------------
    # API pública principal
    # -----------------------------------------------------------------------

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
        """Ejecuta la optimización de hélice para la tarea indicada.

        Parameters
        ----------
        task : int
            Número de tarea de diseño, 1 a 5.
        inputs : list
            Parámetros específicos de la tarea. Ver sección Notes.
        z : int
            Número de palas (ignorado en Kaplan, donde es fijo por serie).
        afac : float
            Relación de área expandida Ae/Ao — EAR.
            Ignorado en Kaplan, donde es fijo por serie.
        margen_servicio : float
            Factor Δs ∈ [1.0, 2.0]. Multiplica al empuje o potencia requeridos
            para incorporar márgenes de ensuciamiento y condiciones de mar.
        delta_d : float
            Factor de ajuste de estela no uniforme Δd ∈ [1.0, 1.05].
            Eleva J óptimo para obtener una hélice más conservadora.
        thrust_deduction : float
            Fracción de deducción de empuje t ∈ [0.0, 0.20].
            Empuje neto = T_bruto × (1 − t).
        wake_factor : float
            Fracción de estela w ∈ [0.0, 0.30].
            Va = (1 − w) × Vs.

        Returns
        -------
        dict
            Diccionario con los resultados. Claves garantizadas:

            ``PD``               Relación paso-diámetro óptima.
            ``D_m``              Diámetro de la hélice [m].
            ``rpm_helice``       Velocidad de giro [RPM].
            ``J``                Coeficiente de avance de operación.
            ``eficiencia``       η₀ en el punto de operación.
            ``KT``, ``KQ``       Coeficientes adimensionales.
            ``thrust_bruto_kN``  Empuje bruto [kN].
            ``thrust_neto_kN``   Empuje neto tras deducción [kN].
            ``advertencias``     Lista de mensajes de precaución activos.

        Notes
        -----
        Estructura de ``inputs`` según tarea:

        ============  ================================================
        Tarea         inputs
        ============  ================================================
        1             ``[pd_w [W], rpm [RPM], vs_nudos, eta_r]``
        2             ``[t_n [N], rpm [RPM], vs_nudos]``
        3             ``[pd_w [W], d [m], vs_nudos, eta_r]``
        4             ``[t_n [N], d [m], vs_nudos]``
        5             ``[bhp [HP], rpm_motor [RPM], eta_s, d [m],``
                      ``vs_nudos, t_deduction, r_tug_n [N], eta_r]``
        ============  ================================================

        Raises
        ------
        ValueError
            Si ``task`` está fuera del rango 1–5.
        """
        self._validar_task(task)

        if task == 5:
            return self._ejecutar_tarea_5(inputs, z, afac, wake_factor)
        else:
            return self._ejecutar_tareas_1_4(
                task, inputs, z, afac,
                margen_servicio, delta_d, thrust_deduction, wake_factor,
            )

    # -----------------------------------------------------------------------
    # API pública auxiliar
    # -----------------------------------------------------------------------

    def generar_datos_grafica_helice(
        self,
        resultado: dict,
        j_min: float = 0.01,
        j_max: float = 1.20,
        n_puntos: int = 60,
    ) -> dict:
        """Calcula los arrays necesarios para la gráfica KT / 10KQ / η vs J.

        No genera ninguna figura; retorna datos crudos para que ``app/``
        construya el gráfico con matplotlib o Plotly.

        Parameters
        ----------
        resultado : dict
            Resultado previo de :meth:`run_optimization`.
        j_min, j_max : float
            Rango del coeficiente de avance J para la curva.
        n_puntos : int
            Resolución de la curva (número de puntos).

        Returns
        -------
        dict
            Arrays ``j_arr``, ``kt_arr``, ``kq10_arr``, ``efic_arr``,
            punto de operación ``j_op / kt_op / kq_op / efic_op``,
            curva de demanda ``req_curve_arr`` y metadatos ``task``, ``PD``.
        """
        prop = self._crear_helice(resultado["PD"], resultado["Z"], resultado["EAR"])
        j_arr = np.linspace(j_min, j_max, n_puntos)
        kt_arr, kq_arr, efic_arr = prop.evaluate(j_arr)

        # Curva de demanda para tareas 1–4 (requiere la constante C de diseño)
        C = resultado.get("C_diseno")
        task = resultado["tarea"]
        req_curve: Optional[np.ndarray] = None
        if C is not None and task in (1, 2, 3, 4):
            req_curve = self._curva_demanda(task, C, j_arr)

        kt_op, kq_op, efic_op = prop.evaluate(resultado["J"])

        return {
            "j_arr":         j_arr,
            "kt_arr":        kt_arr,
            "kq10_arr":      kq_arr * 10.0,
            "efic_arr":      efic_arr,
            "j_op":          resultado["J"],
            "kt_op":         kt_op,
            "kq_op":         kq_op,
            "efic_op":       efic_op,
            "req_curve_arr": req_curve,
            "task":          task,
            "PD":            resultado["PD"],
        }

    @staticmethod
    def convertir_velocidades(vs_nudos: float, wake_factor: float = 0.0) -> dict:
        """Convierte la velocidad del buque Vs [nudos] a velocidad de avance Va [m/s].

        El wake factor w representa la diferencia entre la velocidad del buque
        y la velocidad real del agua que llega a la hélice.
        Para aguas abiertas: w = 0 (sin estela).

        Parameters
        ----------
        vs_nudos : float
            Velocidad del buque en nudos.
        wake_factor : float
            Fracción de estela w, rango típico 0.0 a 0.30.

        Returns
        -------
        dict
            Claves: ``vs_ms`` [m/s], ``va_ms`` [m/s], ``wake_factor``.
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
        """Aplica el factor de ajuste de estela no uniforme Δd sobre J óptimo.

        La estela no uniforme (non-uniform wake) degrada el rendimiento respecto
        al valor calculado en aguas abiertas. Para compensarlo se trabaja con
        un J mayor: ``J_adj = Δd · J_opt`` (Δd típicamente 1.00–1.05).

        Aumentar J con Va constante implica:

        - Tareas 1 y 2 (RPM fijas): reducir D → ``D_adj = Va / (n · J_adj)``
        - Tareas 3 y 4 (D fijo):    reducir n → ``n_adj = Va / (D · J_adj)``

        Parameters
        ----------
        j_optimo : float   J óptimo antes del ajuste.
        delta_d  : float   Factor de ajuste (1.00 = sin ajuste).
        va       : float   Velocidad de avance Va [m/s].
        n_rps    : float   Velocidad de giro en rev/s.
        d        : float   Diámetro de la hélice [m].
        tarea    : int     1 o 2 → RPM fijas; 3 o 4 → D fijo.

        Returns
        -------
        dict
            Claves: ``j_adj``, ``d_adj`` [m], ``n_adj_rps`` [rev/s].
        """
        j_adj = delta_d * j_optimo

        if tarea in (1, 2):
            d_adj = va / (n_rps * j_adj) if (n_rps > 0 and j_adj > 0) else d
            n_adj = n_rps
        else:
            n_adj = va / (d * j_adj) if (d > 0 and j_adj > 0) else n_rps
            d_adj = d

        return {"j_adj": j_adj, "d_adj": d_adj, "n_adj_rps": n_adj}

    # -----------------------------------------------------------------------
    # Métodos privados — fábrica y resolución numérica
    # -----------------------------------------------------------------------

    def _crear_helice(self, pfac: float, z: int, afac: float):
        """Instancia SerieB o SerieKaplan según ``prop_config['tipo']``.

        Aplica el patrón fábrica: el optimizador no necesita saber qué
        serie usa internamente; delega esa decisión en la configuración.

        Parameters
        ----------
        pfac : float   Relación paso-diámetro P/D del candidato.
        z    : int     Número de palas (ignorado en Kaplan).
        afac : float   EAR (ignorado en Kaplan).

        Returns
        -------
        SerieB o SerieKaplan

        Raises
        ------
        ValueError
            Si ``prop_config['tipo']`` no es ``'SERIE_B'`` ni ``'KAPLAN'``.
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
        raise ValueError(f"Tipo de hélice no reconocido: '{tipo}'. Use 'SERIE_B' o 'KAPLAN'.")

    def _find_equilibrium_J(
        self, prop, task: int, design_constant: float
    ) -> Optional[float]:
        """Resuelve la ecuación implícita de autopropulsión por el método de Brent.

        Encuentra J tal que KT(J) o KQ(J) iguala la curva de demanda:

        ========  ======================
        Tarea     Ecuación a resolver
        ========  ======================
        1         KQ(J) = C · J⁵
        2         KT(J) = C · J⁴
        3         KQ(J) = C · J³
        4         KT(J) = C · J²
        ========  ======================

        Parameters
        ----------
        prop           : instancia de SerieB o SerieKaplan.
        task           : int  número de tarea (1–4).
        design_constant: float  constante C de diseño.

        Returns
        -------
        float o None
            J de autopropulsión, o ``None`` si no existe raíz en el intervalo.
        """
        def residuo(j: float) -> float:
            kt, kq, _ = prop.evaluate(j)
            if task == 1: return kq - design_constant * j ** 5
            if task == 2: return kt - design_constant * j ** 4
            if task == 3: return kq - design_constant * j ** 3
            if task == 4: return kt - design_constant * j ** 2

        try:
            return root_scalar(residuo, bracket=list(J_BRACKET), method="brentq").root
        except ValueError:
            log.warning("_find_equilibrium_J: sin raíz en J=%s para task=%d, C=%.5f",
                        J_BRACKET, task, design_constant)
            return None

    # -----------------------------------------------------------------------
    # Métodos privados — funciones objetivo para scipy
    # -----------------------------------------------------------------------

        def _objective_efficiency(
            self, pfac: float, task: int, design_constant: float, z: int, afac: float
        ) -> float:
            """Función objetivo para tareas 1–4: minimiza (−η₀).

            scipy minimiza; nosotros queremos maximizar η₀, de ahí el signo negativo.
            Si el P/D candidato no produce un J válido, penaliza con 1.0
            (equivalente a eficiencia cero).
            """
            prop = self._crear_helice(pfac, z, afac)
            j_eq = self._find_equilibrium_J(prop, task, design_constant)
            if j_eq is None:
                return 1.0   # penalización: P/D inviable
            _, _, eta = prop.evaluate(j_eq) #descarta kt y kq no lo necesitamos solo la eficiencia
            return -eta      # negativo → scipy minimiza → encontramos máximo de η

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
        """Función objetivo para tarea 5: minimiza (−T_bruto).

        Cubre dos condiciones de operación:

        - ``va == 0`` (bollard puro): J = 0, despeja n de la ecuación de torque.
        - ``va > 0`` (remolque a velocidad): balance iterativo potencia–torque.
        """
        prop = self._crear_helice(pfac, z, afac)
        po_w = pd_w * eta_r   # potencia neta entregada al propulsor [W]

        if va == 0.0:
            kt, kq, _ = prop.evaluate(0.0)
            if kq <= 0:
                return 1e6   # geometría degenera (KQ ≤ 0)
            # Despejando n de: Q = 2π·ρ·n²·D⁵·KQ
            n_rps = (po_w / (2.0 * pi * self.rho * d ** 5 * kq)) ** (1.0 / 3.0)
            return -(kt * self.rho * n_rps ** 2 * d ** 4)

        def balance(n_guess: float) -> float:
            """Diferencia entre potencia absorbida por la hélice y la disponible."""
            j_g = va / (n_guess * d)
            _, kq, _ = prop.evaluate(j_g)
            return 2.0 * pi * self.rho * n_guess ** 3 * d ** 5 * kq - po_w

        try:
            n_rps = root_scalar(balance, bracket=[0.1, 100.0], method="brentq").root
            j_val = va / (n_rps * d)
            kt, _, _ = prop.evaluate(j_val)
            return -(kt * self.rho * n_rps ** 2 * d ** 4)
        except ValueError:
            return 1e6   # no converge: P/D inviable para este Va

    # -----------------------------------------------------------------------
    # Métodos privados — ejecutores por tarea
    # -----------------------------------------------------------------------

    def _preparar_constante(
        self,
        task: int,
        inputs: list,
        wake_factor: float,
        margen_servicio: float,
        advertencias: list,
    ) -> tuple[float, float, float, float]:
        """Extrae parámetros de ``inputs``, convierte velocidades y calcula C.

        Returns
        -------
        tuple
            ``(C, va, n_rps_o_d, parametro_secundario)``
            - Tareas 1, 2: parametro_secundario = n [rev/s],  n_rps_o_d = n
            - Tareas 3, 4: parametro_secundario = d [m],      n_rps_o_d = d
        """
        vel = self.convertir_velocidades(
            inputs[2] if task in (1, 2) else inputs[2], wake_factor
        )
        va = vel["va_ms"]

        if task == 1:
            pd_w, rpm, _, eta_r = inputs
            n = rpm / 60.0
            C = (pd_w * margen_servicio * n ** 2 * eta_r) / (2.0 * pi * self.rho * va ** 5)
            return C, va, n, eta_r

        if task == 2:
            t_n, rpm, _ = inputs
            n = rpm / 60.0
            t_diseno = t_n * margen_servicio
            C = (t_diseno * n ** 2) / (self.rho * va ** 4)
            if margen_servicio > 1.0:
                advertencias.append(
                    f"Margen Δs={margen_servicio:.2f}: T_diseño={t_diseno/1000:.2f} kN"
                    f" (original {t_n/1000:.2f} kN)"
                )
            return C, va, n, None

        if task == 3:
            pd_w, d, _, eta_r = inputs
            C = (pd_w * margen_servicio * eta_r) / (2.0 * pi * self.rho * d ** 2 * va ** 3)
            return C, va, d, eta_r

        # task == 4
        t_n, d, _ = inputs
        t_diseno = t_n * margen_servicio
        C = t_diseno / (self.rho * d ** 2 * va ** 2)
        if margen_servicio > 1.0:
            advertencias.append(f"Margen Δs={margen_servicio:.2f}: T_diseño={t_diseno/1000:.2f} kN")
        return C, va, d, None

    def _ejecutar_tareas_1_4(
        self,
        task: int,
        inputs: list,
        z: int,
        afac: float,
        margen_servicio: float,
        delta_d: float,
        thrust_deduction: float,
        wake_factor: float,
    ) -> dict:
        """Ejecuta las tareas 1 a 4: maximización de eficiencia en aguas abiertas."""
        advertencias: list[str] = []
        vs_nudos = inputs[2]

        C, va, param, _ = self._preparar_constante(
            task, inputs, wake_factor, margen_servicio, advertencias
        )

        # Optimización de P/D para máxima eficiencia
        res = minimize_scalar(
            self._objective_efficiency, #funcion objetivo que analiza que va evaluar la eficiencia para diferentes P/D
            bounds=PD_BOUNDS, #Rango de trabajo para P/D
            args=(task, C, z, afac), #parametros que se mantienen contantes de la funcion objetivo, el unico que varia es el P/D que es el que se optimiza
            method="bounded", #metodo
        )
        opt_pfac = res.x
        opt_prop = self._crear_helice(opt_pfac, z, afac)

        # Punto de autopropulsión
        j_op = self._find_equilibrium_J(opt_prop, task, C)
        if j_op is None:
            advertencias.append(
                "No se encontró J de autopropulsión en el intervalo [0.01, 1.40]. "
                "Verifique los parámetros de entrada."
            )
            j_op = 0.5   # valor de respaldo para evitar colapso

        kt_op, kq_op, efic_op = opt_prop.evaluate(j_op)

        # Dimensiones del punto de operación
        if task in (1, 2):
            n_op, d_op = param, va / (param * j_op) if (param > 0 and j_op > 0) else 0.0
        else:
            d_op, n_op = param, va / (param * j_op) if (param > 0 and j_op > 0) else 0.0

        # Corrección de estela no uniforme (solo si Δd ≠ 1.0)
        if delta_d != 1.0:
            ajuste = self.aplicar_ajuste_estela(j_op, delta_d, va, n_op, d_op, task)
            j_op, d_op, n_op = ajuste["j_adj"], ajuste["d_adj"], ajuste["n_adj_rps"]
            kt_op, kq_op, efic_op = opt_prop.evaluate(j_op)
            advertencias.append(
                f"Ajuste Δd={delta_d:.3f}: J_adj={j_op:.4f} | D_adj={d_op:.4f} m | η={efic_op:.4f}"
            )

        # Empuje bruto y neto
        thrust_bruto = kt_op * self.rho * n_op ** 2 * d_op ** 4
        thrust_neto  = thrust_bruto * (1.0 - thrust_deduction)
        if thrust_deduction > 0.0:
            advertencias.append(
                f"Deducción t={thrust_deduction:.3f}: "
                f"T_bruto={thrust_bruto/1000:.2f} kN → T_neto={thrust_neto/1000:.2f} kN"
            )

        return {
            "tarea": task,
            "tipo_helice":     self.prop_config.get("nombre", "N/D"),
            "PD":              opt_pfac,
            "D_m":             d_op,
            "rpm_helice":      n_op * 60.0,
            "n_rps":           n_op,
            "J":               j_op,
            "eficiencia":      efic_op,
            "KT":              kt_op,
            "KQ":              kq_op,
            "EAR":             afac,
            "Z":               z,
            "C_diseno":        C,
            "vs_nudos":        vs_nudos,
            "va_ms":           va,
            "wake_factor":     wake_factor,
            "thrust_deduction":thrust_deduction,
            "margen_servicio": margen_servicio,
            "delta_d":         delta_d,
            "pd_kw":           (inputs[0] / 1000.0) if task in (1, 3) else None,
            "thrust_bruto_kN": thrust_bruto / 1000.0,
            "thrust_neto_kN":  thrust_neto  / 1000.0,
            "thrust_bruto_N":  thrust_bruto,
            "rpm_motor":       None,
            "ratio_i":         None,
            "advertencias":    advertencias,
        }

    def _ejecutar_tarea_5(
        self,
        inputs: list,
        z: int,
        afac: float,
        wake_factor: float,
    ) -> dict:
        """Ejecuta la tarea 5: Bollard Pull o remolque a velocidad.

        Maximiza el empuje bruto de la hélice dentro de la potencia disponible.
        """
        bhp, rpm_motor, eta_s, d, vs_nudos, t_deduction, r_tug_n, eta_r = inputs

        pd_w = bhp * 746.0 * eta_s          # Potencia al eje [W]
        va   = self.convertir_velocidades(vs_nudos, wake_factor)["va_ms"]

        # Optimización P/D → máximo empuje bruto
        res = minimize_scalar(
            self._objective_task5,
            bounds=PD_BOUNDS,
            args=(pd_w, d, va, z, afac, eta_r),
            method="bounded",
        )
        opt_pfac = res.x
        opt_prop = self._crear_helice(opt_pfac, z, afac)
        po_w     = pd_w * eta_r   # potencia en el propulsor [W]

        # Condición de operación (bollard puro o remolque a Va > 0)
        if va == 0.0:
            j_op = 0.0
            kt_op, kq_op, efic_op = opt_prop.evaluate(0.0)
            n_op = (po_w / (2.0 * pi * self.rho * d ** 5 * kq_op)) ** (1.0 / 3.0)
        else:
            def balance(n_guess: float) -> float:
                j_g = va / (n_guess * d)
                _, kq, _ = opt_prop.evaluate(j_g)
                return 2.0 * pi * self.rho * n_guess ** 3 * d ** 5 * kq - po_w

            n_op = root_scalar(balance, bracket=[0.1, 100.0], method="brentq").root
            j_op = va / (n_op * d)
            kt_op, kq_op, efic_op = opt_prop.evaluate(j_op)

        thrust_bruto   = kt_op * self.rho * n_op ** 2 * d ** 4
        thrust_neto_n  = thrust_bruto * (1.0 - t_deduction) - r_tug_n
        thrust_neto_tf = thrust_neto_n / 9810.0   # conversión N → ton-fuerza

        rpm_helice  = n_op * 60.0
        ratio_i     = rpm_motor / rpm_helice if rpm_helice > 0 else 0.0

        return {
            "tarea":            5,
            "tipo_helice":      self.prop_config.get("nombre", "N/D"),
            "PD":               opt_pfac,
            "D_m":              d,
            "rpm_helice":       rpm_helice,
            "n_rps":            n_op,
            "J":                j_op,
            "eficiencia":       efic_op,
            "KT":               kt_op,
            "KQ":               kq_op,
            "EAR":             afac,
            "Z":                z,
            "vs_nudos":         vs_nudos,
            "va_ms":            va,
            "wake_factor":      wake_factor,
            "thrust_deduction": t_deduction,
            "pd_kw":            pd_w / 1000.0,
            "bhp_hp":           bhp,
            "thrust_bruto_kN":  thrust_bruto / 1000.0,
            "thrust_neto_tonf": thrust_neto_tf,
            "thrust_bruto_N":   thrust_bruto,
            "rpm_motor":        rpm_motor,
            "ratio_i":          ratio_i,
            "resistencia_kN":   r_tug_n / 1000.0,
            "advertencias":     [],
        }

    # -----------------------------------------------------------------------
    # Métodos privados — utilidades menores
    # -----------------------------------------------------------------------

    @staticmethod
    def _validar_task(task: int) -> None:
        """Lanza ValueError si ``task`` no está en el rango 1–5."""
        if task not in (1, 2, 3, 4, 5):
            raise ValueError(f"Tarea {task} no válida. Use un valor entre 1 y 5.")

    @staticmethod
    def _curva_demanda(task: int, C: float, j_arr: np.ndarray) -> np.ndarray:
        """Calcula la curva de demanda KT o KQ para graficar junto a las curvas de la hélice."""
        if task == 1: return 10.0 * C * j_arr ** 5
        if task == 2: return C * j_arr ** 4
        if task == 3: return 10.0 * C * j_arr ** 3
        if task == 4: return C * j_arr ** 2