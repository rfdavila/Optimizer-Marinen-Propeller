"""
================================================================================
  core/gestor_motores.py
  GESTIÓN DE LA BIBLIOTECA DE MOTORES
================================================================================
  Funciones puras para leer y consultar el archivo biblioteca_motores.json.

  REGLA DE ORO: Sin print(), input() ni plt.show().
  Las funciones reciben rutas y claves, y retornan datos estructurados.

  Diseño (Inyección de Dependencias):
    El nombre del archivo (ruta) se pasa como argumento. Esto permite
    que los tests usen archivos de prueba sin modificar el código.
================================================================================
"""

import json
import os
from typing import Optional
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES DE CARGA
# ─────────────────────────────────────────────────────────────────────────────

def cargar_biblioteca(ruta: Optional[str] = None) -> dict:
    """
    Lee el archivo JSON de la biblioteca de motores y retorna su contenido.

    Parámetros
    ----------
    ruta : str, opcional
        Ruta completa al archivo JSON. Si es None, se busca automáticamente
        en la carpeta data/ relativa a este archivo.

    Retorna
    -------
    dict : Diccionario con todos los motores del archivo.
           Clave  = nombre del motor (str).
           Valor  = dict con 'nominal_hp', 'nominal_rpm', 'performance_curve'.

    Excepciones
    -----------
    FileNotFoundError : Si el archivo no existe en la ruta indicada.
    json.JSONDecodeError : Si el archivo JSON está malformado.
    """
    if ruta is None:
        # Construir la ruta por defecto: data/biblioteca_motores.json
        _aqui = os.path.dirname(os.path.abspath(__file__))
        ruta  = os.path.join(_aqui, '..', 'data', 'biblioteca_motores.json')

    ruta = os.path.abspath(ruta)   # resolver '..' para evitar rutas relativas ambiguas

    if not os.path.isfile(ruta):
        raise FileNotFoundError(
            f"No se encontró la biblioteca de motores en:\n  {ruta}\n"
            "Verifique que el archivo exista en la carpeta data/."
        )

    with open(ruta, 'r', encoding='utf-8') as f:
        biblioteca = json.load(f)

    return biblioteca


def obtener_datos_motor(biblioteca: dict, nombre: str) -> dict:
    """
    Recupera los datos de un motor específico de la biblioteca.

    Parámetros
    ----------
    biblioteca : dict  Diccionario cargado por cargar_biblioteca().
    nombre     : str   Nombre exacto del motor (clave en el JSON).

    Retorna
    -------
    dict : Datos del motor seleccionado con claves:
           'nominal_hp', 'nominal_rpm', 'performance_curve'.

    Excepciones
    -----------
    KeyError : Si el nombre del motor no existe en la biblioteca.
    """
    if nombre not in biblioteca:
        disponibles = list(biblioteca.keys())
        raise KeyError(
            f"Motor '{nombre}' no encontrado. "
            f"Motores disponibles: {disponibles}"
        )
    return biblioteca[nombre]


def listar_motores(biblioteca: dict) -> list[str]:
    """
    Retorna la lista de nombres de todos los motores en la biblioteca.

    Parámetros
    ----------
    biblioteca : dict  Diccionario cargado por cargar_biblioteca().

    Retorna
    -------
    list[str] : Lista ordenada de nombres de motores.
    """
    return list(biblioteca.keys())


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN: CURVA DE POTENCIA APROXIMADA (para motor manual)
# ─────────────────────────────────────────────────────────────────────────────

def aproximar_curva_motor(nominal_hp: float, nominal_rpm: float,
                          num_puntos: int = 10) -> dict:
    """
    Aproxima la curva de potencia de un motor marino a partir de sus
    valores nominales, usando la función polinómica estándar de la
    literatura de motores marinos CAT / propulsión marina:

        P(n) = P_nom · (n / n_nom)^3

    Esta relación cúbica (ley del cubo) es la aproximación clásica para
    motores diesel acoplados a hélice en condiciones de libre avance.
    Se añade un punto de carga ligera y el punto máximo para mayor realismo.

    Parámetros
    ----------
    nominal_hp  : float  Potencia nominal del motor [HP].
    nominal_rpm : float  RPM nominales del motor.
    num_puntos  : int    Cantidad de puntos en la curva (por defecto 10).

    Retorna
    -------
    dict con claves:
        'nominal_hp'       : float
        'nominal_rpm'      : float
        'performance_curve': dict con 'rpm' y 'power_hp' (listas)
    """
    # Rango de RPM desde ralentí (~50 % de nominal) hasta nominal
    rpm_min = nominal_rpm * 0.50
    rpms    = np.linspace(rpm_min, nominal_rpm, num_puntos)

    # Ley del cubo: potencia proporcional al cubo de las RPM
    potencias = nominal_hp * (rpms / nominal_rpm) ** 3

    return {
        'nominal_hp':        float(nominal_hp),
        'nominal_rpm':       float(nominal_rpm),
        'performance_curve': {
            'rpm':      [round(float(r), 1) for r in rpms],
            'power_hp': [round(float(p), 2) for p in potencias],
        }
    }
