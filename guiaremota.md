# Optimizador de Hélices Navales — v8

Aplicación web para el diseño y optimización de hélices marinas basada en los métodos de **series sistemáticas** (Birk 2019):

- **Serie B de Wageningen** — hélice libre, 2–7 palas, EAR 0.40–1.05
- **Kaplan Ka 4-70** — 4 palas, con o sin tobera 19A
- **Kaplan Ka 5-75** — 5 palas, con o sin tobera 19A

Permite resolver las **5 tareas de diseño** clásicas, verifica los criterios de cavitación de **Keller** y **Burrill-Emerson**, genera gráficas interactivas y exporta un **reporte PDF** completo.

---

## Estructura del proyecto

```
v8/
├── app.py                    ← Interfaz Streamlit (punto de entrada)
├── requirements.txt          ← Dependencias Python
├── README.md                 ← Este archivo
├── core/
│   ├── __init__.py
│   ├── optimizador.py        ← Motor de optimización (5 tareas)
│   ├── propulsores.py        ← Modelos Serie B y Kaplan
│   ├── cavitacion.py         ← Verificación de cavitación
│   ├── gestor_motores.py     ← Gestión de biblioteca de motores
│   └── biblioteca_coeficientes.py  ← Coeficientes polinómicos
└── data/
    └── biblioteca_motores.json     ← 38 motores Caterpillar diesel
```

---

## Instalación y ejecución local

### Requisitos previos

- **Python 3.10 o superior** — descarga en [python.org](https://www.python.org/downloads/)
- **pip** (viene incluido con Python)

### Pasos de instalación

**1. Clonar o descargar el proyecto**

```bash
# Con git:
git clone <url-del-repositorio>
cd v8

# O simplemente copie la carpeta v8/ a su equipo y ábrala en la terminal.
```

**2. (Recomendado) Crear un entorno virtual**

```bash
# Windows (PowerShell):
python -m venv venv
.\venv\Scripts\Activate.ps1

# macOS / Linux:
python3 -m venv venv
source venv/bin/activate
```

**3. Instalar dependencias**

```bash
pip install -r requirements.txt
```

**4. Ejecutar la aplicación**

```bash
streamlit run app.py
```

El navegador se abrirá automáticamente en `http://localhost:8501`.  
Si no se abre solo, abra ese enlace manualmente.

### Detener la aplicación

Presione `Ctrl + C` en la terminal donde corre Streamlit.

---

## Uso rápido

1. **Motor** — Seleccione un motor de la biblioteca CAT o ingrese HP y RPM manualmente.
2. **Tipo de hélice** — Elija la serie (Wageningen B o Kaplan) en el panel izquierdo.
3. **Geometría** — Número de palas Z y relación de área EAR (o fijos para Kaplan).
4. **Tarea de diseño** — Seleccione una de las 5 tareas disponibles.
5. **Datos de diseño** — Complete velocidad, ratio reductora, diámetro fijo, etc.
6. Haga clic en **🚀 Ejecutar Optimización**.
7. Explore los resultados en las pestañas **Gráficas**, **Cavitación** y **Reporte PDF**.

---

## Publicar en la web para que cualquiera lo pruebe

### Opción A — Streamlit Community Cloud (gratuito, recomendado)

Streamlit ofrece alojamiento gratuito para aplicaciones públicas.

**Paso 1: Subir el código a GitHub**

```bash
# Desde la carpeta v8/
git init
git add .
git commit -m "Optimizador Hélices Navales v8"
git branch -M main
git remote add origin https://github.com/<tu-usuario>/<nombre-repo>.git
git push -u origin main
```

**Paso 2: Crear cuenta en Streamlit Cloud**

- Vaya a [share.streamlit.io](https://share.streamlit.io)
- Inicie sesión con su cuenta de GitHub

**Paso 3: Desplegar la aplicación**

1. Haga clic en **"New app"**
2. Seleccione el repositorio y la rama (`main`)
3. En **"Main file path"** escriba: `app.py`
4. Haga clic en **"Deploy!"**

En 2–3 minutos tendrá una URL pública del tipo:
```
https://<tu-usuario>-helices-navales.streamlit.app
```

Comparta esa URL y cualquier persona podrá usar la aplicación sin instalar nada.

---

### Opción B — Railway (gratuito con cuenta)

Railway permite despliegues rápidos desde GitHub.

**1. Suba el código a GitHub** (igual que en Opción A, Paso 1).

**2. Agregue el archivo `Procfile`** en la raíz del proyecto:

```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

**3. Despliegue en Railway**

- Vaya a [railway.app](https://railway.app) y cree una cuenta.
- Haga clic en **"New Project"** → **"Deploy from GitHub repo"**.
- Seleccione su repositorio.
- Railway detecta el `Procfile` y despliega automáticamente.
- Obtendrá una URL pública para compartir.

---

### Opción C — Render (gratuito)

**1. Suba el código a GitHub.**

**2. Vaya a [render.com](https://render.com)** y cree una cuenta.

**3. Cree un nuevo "Web Service":**
- Conecte su repositorio de GitHub.
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
- **Environment:** Python 3

**4. Haga clic en "Create Web Service"** y obtendrá su URL pública.

---

### Opción D — Ejecutar en la nube sin código (Google Colab)

Si solo desea probarlo sin configurar servidores:

```python
# En una celda de Google Colab:
!pip install streamlit numpy scipy matplotlib reportlab pyngrok -q

# Suba los archivos del proyecto a Colab y luego:
from pyngrok import ngrok
import subprocess, threading

def run():
    subprocess.run(["streamlit", "run", "app.py",
                    "--server.port=8501", "--server.headless=true"])

threading.Thread(target=run, daemon=True).start()
public_url = ngrok.connect(8501)
print(f"URL pública: {public_url}")
```

---

## Dependencias

| Paquete | Versión mínima | Uso |
|---------|---------------|-----|
| `streamlit` | 1.32.0 | Interfaz web |
| `numpy` | 1.24.0 | Cálculo numérico |
| `scipy` | 1.11.0 | Optimización (Brent, minimize_scalar) |
| `matplotlib` | 3.7.0 | Gráficas |
| `reportlab` | 4.0.0 | Generación de PDF |

---

## Tareas de diseño disponibles

| Tarea | Datos de entrada | Resultado |
|-------|-----------------|-----------|
| 1 | Potencia + ratio reductora + Vs | **Diámetro D** óptimo |
| 2 | Empuje T + ratio reductora + Vs | **Diámetro D** óptimo |
| 3 | Potencia + D fijo + Vs | **RPM / ratio** óptimos |
| 4 | Empuje T + D fijo + Vs | **RPM / ratio** óptimos |
| 5 | Potencia + D + Vs (0 = punto fijo) | **Máximo tiro** (Bollard Pull) |

---

## Verificación de cavitación

Una vez obtenido el resultado de optimización, en la pestaña **Cavitación** se evalúan:

- **Criterio de Keller (1966)** — EAR mínima según empuje, número de palas e inmersión
- **Criterio de Burrill-Emerson** — Número de cavitación σb y límite τc por curvas empíricas
- **Velocidad de punta** — Clasificación BAJA / MEDIA / ALTA según Vtip = π·n·D

---

## Referencia

Birk, L. (2019). *Fundamentals of Ship Hydrodynamics: Fluid Mechanics, Ship Resistance and Propulsion*. Wiley.


"""
# 1. Crear entorno virtual
python -m venv venv

# 2. Activar (verás (venv) al inicio del prompt)
.\venv\Scripts\Activate.ps1

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ahora sí, correr la app
streamlit run app.py

Entrar al entorno virtual (activar):

Verás (venv) al inicio del prompt.

Salir del entorno virtual (desactivar):

El (venv) desaparece del prompt y vuelves al Python global.

Notas:

Siempre activa antes de trabajar en tu proyecto
Desactiva cuando termines (o simplemente cierra la terminal)
Si cierras PowerShell, se desactiva automáticamente
Para ejecutar la app, debe estar activado el entorno
"""