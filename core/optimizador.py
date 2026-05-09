"""
================================================================================
  core/optimizador.py
  MOTOR DE OPTIMIZACIÓN DE HÉLICES – PropellerOptimizer
================================================================================
  Implementa las 5 tareas de diseño de hélices (Birk 2019, Caps. 48.2-48.4):

    Tarea 1: Optimiza diámetro D  →  dada: Potencia, RATIO, Vs
    Tarea 2: Optimiza diámetro D  →  dado: Empuje T, RATIO, Vs
    Tarea 3: Optimiza RPM (RATIO) →  dado: Potencia, D, Vs
    Tarea 4: Optimiza RPM (RATIO) →  dado: Empuje T, D, Vs
    Tarea 5: Bollard Pull / Tiro  →  dado: Potencia, D, Vs (puede ser 0)

  MEJORAS v7 incorporadas:
    · Velocidad de entrada Vs [nudos] → convierte internamente a Va = (1-w)·Vs [m/s]
    · Margen de servicio Δs (factor 1–2 sobre el empuje requerido T)
    · Ajuste de estela no uniforme Δd (factor J_adj = Δd · J_opt)
      - Fija RPM: reduce el diámetro  D_adj = Va / (n · J_adj)
      - Fija D  : reduce las RPM      n_adj = Va / (D · J_adj)
    · Wake factor (w) y thrust deduction (t) integrados en todas las tareas
    · Sin print(), input() ni plt.show() → solo retorna diccionarios

  REGLA DE ORO: Este módulo es puramente matemático. Todo output al usuario
                ocurre en app/cli_main.py o app/streamlit_app.py.

  Referencias:
    - Birk (2019). Fundamentals of Ship Hydrodynamics. Wiley.
    - HydroComp PropExpert 2005 User's Guide.
================================================================================
"""

from math import pi
import numpy as np
from scipy.optimize import root_scalar, minimize_scalar

from core.propulsores import SerieB, SerieKaplan

# Constante de conversión de nudos a metros por segundo
KNOTS_TO_MS = 0.514444


class PropellerOptimizer:
    """
    Optimizador de hélices para las Series B Wageningen y Kaplan Ka.

    Parámetros del constructor
    --------------------------
    prop_config : dict
        Configuración de la hélice. Claves requeridas:
          'tipo'       : 'SERIE_B' o 'KAPLAN'
          'nombre'     : nombre descriptivo (para reportes)
        Para Kaplan además:
          'matriz'     : np.ndarray con los coeficientes polinómicos
          'con_tobera' : bool
          'z'          : int  (número de palas fijo de la serie)
          'ear'        : float (EAR fijo de la serie)
    rho : float
        Densidad del fluido [kg/m³]. Por defecto agua de mar 15 °C.
    re : float
        Número de Reynolds en radio 0.75R. Afecta corrección de Serie B.
    """

    def __init__(self, prop_config: dict, rho: float = 1026.021, re: float = 1e6):
        self.prop_config = prop_config
        self.rho         = rho
        self.re          = re

    # ─────────────────────────────────────────────────────────────────────────
    # MÉTODO AUXILIAR: Crear instancia de hélice según configuración
    # ─────────────────────────────────────────────────────────────────────────
    def _crear_helice(self, pfac: float, z: int, afac: float):
        """
        Instancia SerieB o SerieKaplan según prop_config['tipo'].
        Principio de Inyección de Dependencias: el optimizador no
        "pregunta" qué serie usar, recibe la configuración ya resuelta.
        """
        if self.prop_config['tipo'] == 'SERIE_B':
            return SerieB(pfac, afac, z, self.re)
        elif self.prop_config['tipo'] == 'KAPLAN':
            return SerieKaplan(
                pfac=pfac,
                matriz_coeficientes=self.prop_config['matriz'],
                con_tobera=self.prop_config['con_tobera']
            )
        else:
            raise ValueError(f"Tipo de hélice no reconocido: {self.prop_config['tipo']}")

    # ─────────────────────────────────────────────────────────────────────────
    # MÉTODO AUXILIAR: Encontrar J de equilibrio
    # ─────────────────────────────────────────────────────────────────────────
    def _find_equilibrium_J(self, prop, task: int, design_constant: float) -> float:
        """
        Resuelve la ecuación implícita de punto de autopropulsión:

          Tarea 1: KQ(J) = C · J^5   (constante de diseño basada en torque)
          Tarea 2: KT(J) = C · J^4   (constante de diseño basada en empuje)
          Tarea 3: KQ(J) = C · J^3
          Tarea 4: KT(J) = C · J^2

        Usa el método de Brent (brentq) que garantiza convergencia si existe
        una raíz en el intervalo [0.01, 1.4].

        Retorna
        -------
        float : J óptimo de autopropulsión, o None si no converge de la ecuacion de equilibrio
        """
        def objetivo(j_guess):
            kt, kq, _ = prop.evaluate(j_guess)
            if task == 1: return kq - design_constant * j_guess**5
            if task == 2: return kt - design_constant * j_guess**4
            if task == 3: return kq - design_constant * j_guess**3
            if task == 4: return kt - design_constant * j_guess**2
        try:
            res = root_scalar(objetivo, bracket=[0.01, 1.4], method='brentq')
            return res.root
        except ValueError:
            return None   # la raíz no existe en el intervalo (P/D fuera de rango)

    # ─────────────────────────────────────────────────────────────────────────
    # FUNCIÓN OBJETIVO: Minimiza 1-η para tareas 1-4
    # ─────────────────────────────────────────────────────────────────────────
    def _objective_efficiency(self, pfac: float, task: int,
                              design_constant: float, z: int, afac: float) -> float:
        """
        Función objetivo para minimize_scalar.
        Retorna -η (negativo porque scipy minimiza, nosotros maximizamos η).
        """
        prop = self._crear_helice(pfac, z, afac)  # Serie B/Kaplan para la búsqueda
        j_eq = self._find_equilibrium_J(prop, task, design_constant)
        if j_eq is None:
            return 1.0   # penalización: eficiencia cero si no hay solución
        _, _, efic = prop.evaluate(j_eq)
        return -efic     # negativo: scipy minimiza, nosotros maximizamos

    # ─────────────────────────────────────────────────────────────────────────
    # FUNCIÓN OBJETIVO: Maximiza empuje (Tarea 5 – Bollard/Towing)
    # ─────────────────────────────────────────────────────────────────────────
    def _objective_task5(self, pfac: float, pd_w: float, d: float,
                         va: float, z: int, afac: float, eta_r: float) -> float:
        """
        Función objetivo de la Tarea 5: maximiza empuje bruto.
        Retorna -T (negativo para que scipy lo minimice).
        """
        prop = self._crear_helice(pfac, z, afac)
        po_w = pd_w * eta_r   # potencia entregada al propulsor [W]

        if va == 0.0:
            # Condición Bollard puro (Va = 0): no hay avance, solo giro
            kt, kq, _ = prop.evaluate(0.0)
            if kq <= 0:
                return 1e6
            # Despejamos n de la ecuación de torque: Q = 2π·ρ·n²·D⁵·KQ
            n_rps = (po_w / (2.0 * pi * self.rho * d**5 * kq)) ** (1.0 / 3.0)
            return -(kt * self.rho * n_rps**2 * d**4)
        else:
            # Condición de remolque (Va > 0): balance potencia-torque
            def balance_potencia(n_guess):
                j_g = va / (n_guess * d)
                _, kq, _ = prop.evaluate(j_g)
                return 2.0 * pi * self.rho * n_guess**3 * d**5 * kq - po_w
            try:
                n_rps = root_scalar(balance_potencia, bracket=[0.1, 100],
                                    method='brentq').root
                j_val = va / (n_rps * d)
                kt, _, _ = prop.evaluate(j_val)
                return -(kt * self.rho * n_rps**2 * d**4)
            except ValueError:
                return 1e6

    # ─────────────────────────────────────────────────────────────────────────
    # CONVERSIÓN DE VELOCIDADES
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def convertir_velocidades(vs_nudos: float, wake_factor: float = 0.0) -> dict:
        """
        Convierte la velocidad del buque Vs [nudos] en:
          · Vs_ms : velocidad del buque [m/s]
          · Va_ms : velocidad de avance de la hélice [m/s] = (1 - w) · Vs_ms

        El wake factor (fracción de estela) w representa la diferencia entre
        la velocidad del buque y la velocidad real del agua que llega a la hélice.
        Para aguas abiertas w = 0 (sin estela).

        Parámetros
        ----------
        vs_nudos    : float  Velocidad del buque [nudos].
        wake_factor : float  Fracción de estela w (0 a 0.30).

        Retorna
        -------
        dict con 'vs_ms', 'va_ms', 'wake_factor'.
        """
        vs_ms = vs_nudos * KNOTS_TO_MS
        va_ms = (1.0 - wake_factor) * vs_ms
        return {'vs_ms': vs_ms, 'va_ms': va_ms, 'wake_factor': wake_factor}

    # ─────────────────────────────────────────────────────────────────────────
    # AJUSTE DE ESTELA NO UNIFORME (corrección Δd)
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def aplicar_ajuste_estela(j_optimo: float, delta_d: float,
                              va: float, n_rps: float, d: float,
                              tarea: int) -> dict:
        """
        Aplica el factor de ajuste de estela no uniforme Δd sobre el J óptimo.

        La estela no uniforme (non-uniform wake) degrada el rendimiento respecto
        al valor calculado en aguas abiertas. Para compensarlo, se aumenta J:
            J_adj = Δd · J_opt   (Δd típicamente 1.00 a 1.05)

        El aumento de J, manteniendo Va constante, implica:
          · Si RPM fijas (tareas 1, 2): reducir D →  D_adj = Va / (n · J_adj)
          · Si D fijo   (tareas 3, 4): reducir n →  n_adj = Va / (D · J_adj)

        Parámetros
        ----------
        j_optimo : float  J óptimo antes del ajuste.
        delta_d  : float  Factor de ajuste (1.00 = sin ajuste).
        va       : float  Velocidad de avance Va [m/s].
        n_rps    : float  RPM de la hélice en rev/s.
        d        : float  Diámetro de la hélice [m].
        tarea    : int    1 o 2 → RPM fijas; 3 o 4 → D fijo.

        Retorna
        -------
        dict con 'j_adj', 'd_adj', 'n_adj_rps'.
        """
        j_adj = delta_d * j_optimo   # J ajustado
        #calcular el paso diametro ajustaso
        #va usar como funcion objetivo la eficiencia y J ajutado  es valor fijo
        #se maximizar esa funcion objetivo con deferentes pasos

        if tarea in (1, 2):
            # RPM fijas → ajustamos el diámetro
            d_adj   = va / (n_rps * j_adj) if (n_rps > 0 and j_adj > 0) else d
            n_adj   = n_rps
        else:
            # Diámetro fijo → ajustamos las RPM
            n_adj   = va / (d * j_adj) if (d > 0 and j_adj > 0) else n_rps
            d_adj   = d

        return {'j_adj': j_adj, 'd_adj': d_adj, 'n_adj_rps': n_adj}

    # ─────────────────────────────────────────────────────────────────────────
    # MÉTODO PRINCIPAL: run_optimization
    # ─────────────────────────────────────────────────────────────────────────
    def run_optimization(self, task: int, inputs: list, z: int = 4,
                         afac: float = 0.65,
                         margen_servicio: float = 1.0,
                         delta_d: float = 1.0,
                         thrust_deduction: float = 0.0,
                         wake_factor: float = 0.0) -> dict:
        """
        Ejecuta la optimización de hélice para la tarea indicada.

        PARÁMETROS GENERALES
        --------------------
        task : int  Número de tarea de diseño (1 a 5).
        inputs : list
            Lista de parámetros específicos de la tarea (ver detalle abajo).
        z : int
            Número de palas (ignorado en Kaplan, que lo tiene fijo).
        afac : float
            Relación Ae/Ao (EAR). Ignorado en Kaplan, que lo tiene fijo.
        margen_servicio : float
            Factor Δs ∈ [1.0, 2.0]. Multiplica al empuje requerido T para
            incorporar márgenes de ensuciamiento, mar, etc.
        delta_d : float
            Factor de ajuste de estela no uniforme Δd ∈ [1.0, 1.05].
            Multiplica al J óptimo para buscar una hélice más conservadora.
        thrust_deduction : float
            Fracción de deducción de empuje t ∈ [0.0, 0.20].
            El empuje neto = T_bruto · (1 - t).
        wake_factor : float
            Fracción de estela w ∈ [0.0, 0.30].
            Va = (1 - w) · Vs.  (Para Tareas 1-4 ya viene incorporado en Va.)

        ESTRUCTURA DE inputs POR TAREA
        --------------------------------
        Tarea 1: [pd_w [W], rpm_helice [RPM], vs_nudos, eta_r]
        Tarea 2: [t_n [N], rpm_helice [RPM], vs_nudos]
        Tarea 3: [pd_w [W], d [m], vs_nudos, eta_r]
        Tarea 4: [t_n [N], d [m], vs_nudos]
        Tarea 5: [bhp [HP], rpm_motor [RPM], eta_s, d [m], vs_nudos,
                  t_deduction, r_tug_n [N], eta_r]

        RETORNA
        -------
        dict con todos los resultados del proceso de optimización:
          - parametros_helice: PD, D, RPM, J, η, KT, KQ, EAR, Z
          - condicion_operacion: Va, Vs, w, t
          - potencia_empuje: Pd [W], T_bruto [N], T_neto [N]
          - tren_propulsivo: RPM_motor, RPM_helice, ratio_i
          - tarea: número de tarea ejecutada
          - tipo_helice: nombre de la serie
          - advertencias: lista de mensajes de precaución
        """
        advertencias = []

        # ══════════════════════════════════════════════════════════════════════
        # TAREA 5: Bollard Pull / Tiro de Remolque
        # ══════════════════════════════════════════════════════════════════════
        if task == 5:
            bhp, rpm_motor, eta_s, d, vs_nudos, t_deduction_t5, r_tug_n, eta_r = inputs
            pd_w = bhp * 746.0 * eta_s   # potencia al eje [W]

            # Convertir Vs → Va (incorporando wake_factor)
            vel = self.convertir_velocidades(vs_nudos, wake_factor)
            va  = vel['va_ms']

            # Búsqueda del P/D óptimo que maximiza el empuje bruto
            res = minimize_scalar(
                self._objective_task5,
                bounds=(0.5, 1.4),
                args=(pd_w, d, va, z, afac, eta_r),
                method='bounded'
            )
            opt_pfac = res.x
            opt_prop = self._crear_helice(opt_pfac, z, afac)
            po_w     = pd_w * eta_r   # potencia en el propulsor [W]

            # ── Calcular condición de operación ───────────────────────────────
            if va == 0.0:
                j_op = 0.0
                kt_op, kq_op, efic_op = opt_prop.evaluate(0.0)
                n_op = (po_w / (2.0 * pi * self.rho * d**5 * kq_op)) ** (1.0 / 3.0)
            else:
                def balance_pot(n_guess):
                    j_g = va / (n_guess * d)
                    _, kq, _ = opt_prop.evaluate(j_g)
                    return 2.0 * pi * self.rho * n_guess**3 * d**5 * kq - po_w
                n_op      = root_scalar(balance_pot, bracket=[0.1, 100], method='brentq').root
                j_op      = va / (n_op * d)
                kt_op, kq_op, efic_op = opt_prop.evaluate(j_op)

            thrust_bruto_n   = kt_op * self.rho * n_op**2 * d**4
            thrust_neto_n    = thrust_bruto_n * (1.0 - t_deduction_t5) - r_tug_n
            thrust_neto_tonf = thrust_neto_n / 9810.0   # [ton-fuerza]

            rpm_helice_sal   = n_op * 60.0
            ratio_reductora  = rpm_motor / rpm_helice_sal if rpm_helice_sal > 0 else 0.0

            return {
                'tarea':           5,
                'tipo_helice':     self.prop_config.get('nombre', 'N/D'),
                # ── Parámetros de hélice ──────────────────────────────────────
                'PD':              opt_pfac,
                'D_m':             d,
                'rpm_helice':      rpm_helice_sal,
                'n_rps':           n_op,
                'J':               j_op,
                'eficiencia':      efic_op,
                'KT':              kt_op,
                'KQ':              kq_op,
                'EAR':             afac,
                'Z':               z,
                # ── Condición de operación ────────────────────────────────────
                'vs_nudos':        vs_nudos,
                'va_ms':           va,
                'wake_factor':     wake_factor,
                'thrust_deduction':t_deduction_t5,
                # ── Potencia y empuje ─────────────────────────────────────────
                'pd_kw':           pd_w / 1000.0,
                'bhp_hp':          bhp,
                'thrust_bruto_kN': thrust_bruto_n / 1000.0,
                'thrust_neto_tonf':thrust_neto_tonf,
                'thrust_bruto_N':  thrust_bruto_n,      # para cavitación
                # ── Tren propulsivo ───────────────────────────────────────────
                'rpm_motor':       rpm_motor,
                'ratio_i':         ratio_reductora,
                # ── Extras ───────────────────────────────────────────────────
                'resistencia_kN':  r_tug_n / 1000.0,
                'advertencias':    advertencias,
            }

        # ══════════════════════════════════════════════════════════════════════
        # TAREAS 1 – 4
        # ══════════════════════════════════════════════════════════════════════
        else:
            # ── Extraer parámetros de entrada según tarea ─────────────────────
            if task == 1:
                pd_w, rpm_helice_input, vs_nudos, eta_r = inputs
                n    = rpm_helice_input / 60.0
                # Convertir Vs → Va
                vel  = self.convertir_velocidades(vs_nudos, wake_factor)
                va   = vel['va_ms']
                # Aplicar margen de servicio a la potencia requerida
                pd_w_diseno = pd_w * margen_servicio  # aplicar antes de calcular C
                # Constante de diseño para Tarea 1 (basada en torque):
                # C = KQ / J^5 = Pd·η_R·n² / (2π·ρ·Va^5)
                C = (pd_w_diseno* n**2 * eta_r) / (2.0 * pi * self.rho * va**5)

            elif task == 2:
                t_n_input, rpm_helice_input, vs_nudos = inputs
                n     = rpm_helice_input / 60.0
                vel   = self.convertir_velocidades(vs_nudos, wake_factor)
                va    = vel['va_ms']
                # Aplicar margen de servicio al empuje requerido
                t_n_diseño = t_n_input * margen_servicio
                # Constante de diseño para Tarea 2 (basada en empuje):
                # C = KT / J^4 = T·n² / (ρ·Va^4)
                C = (t_n_diseño * n**2) / (self.rho * va**4)
                if margen_servicio > 1.0:
                    advertencias.append(
                        f"Empuje de diseño incrementado por margen de servicio "
                        f"Δs={margen_servicio:.2f}: "
                        f"T_diseño = {t_n_diseño/1000:.2f} kN "
                        f"(original {t_n_input/1000:.2f} kN)"
                    )

            elif task == 3:
                pd_w, d, vs_nudos, eta_r = inputs
                vel = self.convertir_velocidades(vs_nudos, wake_factor)
                va  = vel['va_ms']
                #Aplicar margen de servicio a la potencia requerida
                pd_w_diseno = pd_w * margen_servicio  # aplicar antes de calcular C
                # Constante de diseño para Tarea 3:
                # C = KQ / J^3 = Pd·η_R / (2π·ρ·D²·Va^3)
                C = (pd_w_diseno * eta_r) / (2.0 * pi * self.rho * d**2 * va**3)

            elif task == 4:
                t_n_input, d, vs_nudos = inputs
                vel = self.convertir_velocidades(vs_nudos, wake_factor)
                va  = vel['va_ms']
                # Aplicar margen de servicio
                t_n_diseño = t_n_input * margen_servicio
                C = t_n_diseño / (self.rho * d**2 * va**2)
                if margen_servicio > 1.0:
                    advertencias.append(
                        f"Empuje de diseño con margen Δs={margen_servicio:.2f}: "
                        f"T_diseño = {t_n_diseño/1000:.2f} kN"
                    )

            else:
                raise ValueError(f"Tarea {task} no válida. Use 1-5.")

            # ── Optimización de P/D (maximizar eficiencia) ────────────────────
            res = minimize_scalar(
                self._objective_efficiency,
                bounds=(0.5, 1.4),
                args=(task, C, z, afac),
                method='bounded'
            )
            opt_pfac = res.x
            opt_prop = self._crear_helice(opt_pfac, z, afac)

            # ── J de autopropulsión ───────────────────────────────────────────
            j_op = self._find_equilibrium_J(opt_prop, task, C)
            if j_op is None:
                advertencias.append("⚠ No se encontró J de autopropulsión en [0.01, 1.4]. "
                                    "Verifique los parámetros de entrada.")
                j_op = 0.5   # valor de respaldo para no colapsar

            kt_op, kq_op, efic_op = opt_prop.evaluate(j_op)

            # ── Dimensiones según tarea ───────────────────────────────────────
            if task in (1, 2):
                # Tareas 1 y 2: n conocido, se calcula D
                d_op  = va / (n * j_op) if (n > 0 and j_op > 0) else 0.0
                n_op  = n
            else:
                # Tareas 3 y 4: D conocido, se calcula n
                n_op  = va / (d * j_op) if (d > 0 and j_op > 0) else 0.0
                d_op  = d

            # ── Aplicar ajuste de estela no uniforme (Δd) ────────────────────
            if delta_d != 1.0:
                ajuste = self.aplicar_ajuste_estela(j_op, delta_d, va, n_op, d_op, task)
                j_adj  = ajuste['j_adj']
                d_adj  = ajuste['d_adj']
                n_adj  = ajuste['n_adj_rps']
                # Re-evaluar la hélice en las condiciones ajustadas
                kt_adj, kq_adj, efic_adj = opt_prop.evaluate(j_adj)

                advertencias.append(
                    f"Ajuste de estela no uniforme Δd={delta_d:.3f} aplicado: "
                    f"J_opt={j_op:.4f} → J_adj={j_adj:.4f}  |  "
                    f"D_adj={d_adj:.4f} m  |  η_adj={efic_adj:.4f}"
                )
                # Actualizar valores con el ajuste
                j_op, d_op, n_op    = j_adj, d_adj, n_adj
                kt_op, kq_op, efic_op = kt_adj, kq_adj, efic_adj

            # ── Empuje bruto y neto ───────────────────────────────────────────
            thrust_bruto_n = kt_op * self.rho * n_op**2 * d_op**4
            thrust_neto_n  = thrust_bruto_n * (1.0 - thrust_deduction)
            if thrust_deduction > 0.0:
                advertencias.append(
                    f"Deducción de empuje t={thrust_deduction:.3f} aplicada: "
                    f"T_bruto={thrust_bruto_n/1000:.2f} kN → "
                    f"T_neto={thrust_neto_n/1000:.2f} kN"
                )

            rpm_helice_sal = n_op * 60.0

            # ── Ratio reductora (solo tareas 3 y 4) ──────────────────────────
            # Para tareas 1 y 2, rpm_motor es un input externo → ratio = N/A
            ratio_i = None

            # ── Potencia consumida ────────────────────────────────────────────
            if task == 1:
                pd_kw = pd_w / 1000.0
            elif task == 3:
                pd_kw = pd_w / 1000.0
            else:
                pd_kw = None   # para tarea 2 y 4 no hay Pd directo de entrada

            return {
                'tarea':           task,
                'tipo_helice':     self.prop_config.get('nombre', 'N/D'),
                # ── Parámetros de hélice ──────────────────────────────────────
                'PD':              opt_pfac,
                'D_m':             d_op,
                'rpm_helice':      rpm_helice_sal,
                'n_rps':           n_op,
                'J':               j_op,
                'eficiencia':      efic_op,
                'KT':              kt_op,
                'KQ':              kq_op,
                'EAR':             afac,
                'Z':               z,
                'C_diseno':        C,
                # ── Condición de operación ────────────────────────────────────
                'vs_nudos':        vs_nudos if task in (1, 2, 3, 4) else None,
                'va_ms':           va,
                'wake_factor':     wake_factor,
                'thrust_deduction':thrust_deduction,
                'margen_servicio': margen_servicio,
                'delta_d':         delta_d,
                # ── Potencia y empuje ─────────────────────────────────────────
                'pd_kw':           pd_kw,
                'thrust_bruto_kN': thrust_bruto_n / 1000.0,
                'thrust_neto_kN':  thrust_neto_n  / 1000.0,
                'thrust_bruto_N':  thrust_bruto_n,   # para cavitación (en N)
                # ── Tren propulsivo ───────────────────────────────────────────
                'rpm_motor':       None,   # app/ lo agrega si aplica
                'ratio_i':         ratio_i,
                # ── Advertencias / notas ──────────────────────────────────────
                'advertencias':    advertencias,
            }

    # ─────────────────────────────────────────────────────────────────────────
    # MÉTODO: Generar datos de gráficas (sin plt.show)
    # ─────────────────────────────────────────────────────────────────────────
    def generar_datos_grafica_helice(self, resultado: dict,
                                     j_min: float = 0.01,
                                     j_max: float = 1.20,
                                     n_puntos: int = 60) -> dict:
        """
        Calcula los datos necesarios para la Gráfica 2 (KT, 10KQ, η vs J).

        NO genera ninguna figura; retorna arrays que app/ usará para
        construir el gráfico con matplotlib o plotly.

        Parámetros
        ----------
        resultado : dict  Resultado de run_optimization().
        j_min, j_max : float  Rango de J para la curva.
        n_puntos : int  Número de puntos de la curva.

        Retorna
        -------
        dict con:
            j_arr, kt_arr, kq10_arr, efic_arr : arrays de la curva
            j_op, kt_op, kq_op, efic_op       : punto de operación
            req_curve_arr                      : curva de demanda (KT/KQ requerida)
            task, PD                           : para el título del gráfico
        """
        opt_pfac = resultado['PD']
        afac     = resultado['EAR']
        z        = resultado['Z']
        task     = resultado['tarea']
        j_op     = resultado['J']
        C        = resultado.get('C_diseno', None)

        prop     = self._crear_helice(opt_pfac, z, afac)
        j_arr    = np.linspace(j_min, j_max, n_puntos)
        kt_arr, kq_arr, efic_arr = prop.evaluate(j_arr)
        kq10_arr = kq_arr * 10.0

        kt_op, kq_op, efic_op = prop.evaluate(j_op)

        # Curva de demanda según tarea
        req_curve_arr = None
        if C is not None and task in (1, 2, 3, 4):
            if task == 1:   req_curve_arr = 10.0 * C * j_arr**5
            elif task == 2: req_curve_arr = C * j_arr**4
            elif task == 3: req_curve_arr = 10.0 * C * j_arr**3
            elif task == 4: req_curve_arr = C * j_arr**2

        return {
            'j_arr':          j_arr,
            'kt_arr':         kt_arr,
            'kq10_arr':       kq10_arr,
            'efic_arr':       efic_arr,
            'j_op':           j_op,
            'kt_op':          kt_op,
            'kq_op':          kq_op,
            'efic_op':        efic_op,
            'req_curve_arr':  req_curve_arr,
            'task':           task,
            'PD':             opt_pfac,
        }

    def generar_datos_grafica_motor(self, datos_motor: dict,
                                    bhp_op: float, rpm_op: float) -> dict:
        """
        Genera datos para la Gráfica 1 (Curva de potencia del motor vs RPM).

        Parámetros
        ----------
        datos_motor : dict  Datos del motor (de gestor_motores).
        bhp_op      : float BHP en el punto de operación seleccionado.
        rpm_op      : float RPM en el punto de operación.

        Retorna
        -------
        dict con arrays rpm_arr, power_arr, y punto_op para la gráfica.
        """
        curva = datos_motor.get('performance_curve', {})
        rpms  = curva.get('rpm',      [])
        potencias = curva.get('power_hp', [])

        return {
            'rpm_arr':     rpms,
            'power_arr':   potencias,
            'rpm_op':      rpm_op,
            'power_op_hp': bhp_op,
            'nombre_motor':datos_motor.get('nombre', 'Motor'),
        }
