import os
import re
import traceback
import xml.etree.ElementTree as ET
import secrets
import json
import html
import base64
import math
import urllib.parse
from pathlib import Path
from functools import wraps
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from time import time
from time import sleep as time_module_sleep
import anthropic
import requests
from flask import Flask, jsonify, render_template, request, Response
app = Flask(__name__)
app.json.sort_keys = False  # mantiene el orden Euro, Yen, Libra, Yuan

HEADERS_WEB = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

# ---------- tendencias de monedas (Frankfurter) ----------
API = "https://api.frankfurter.dev/v1"
MONEDAS = {"EUR": "Euro", "JPY": "Yen", "GBP": "Libra", "CNY": "Yuan"}
PERIODOS = {"1m": 30, "6m": 182, "1a": 365}

# Guarda las respuestas 1 hora para no consultar la API en cada visita
cache = {}
DURACION_CACHE = 3600

# ---------- noticias (Google News RSS con feeds de respaldo) ----------
NOTICIAS_API = "https://news.google.com/rss/search"
CONSULTAS_NOTICIAS = [
    "dólar geopolítica when:7d",
    "divisas guerra aranceles when:7d",
    "tipo de cambio banco central when:7d",
    "economía mundial divisas",
]
FEEDS_RESPALDO = [
    "https://feeds.bbci.co.uk/news/business/rss.xml",
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://www.aljazeera.com/xml/rss/all.xml",
]
PALABRAS_CLAVE = [
    "dollar", "euro", "yen", "yuan", "currency", "central bank", "tariff",
    "sanction", "trade", "inflation", "oil", "economy", "war", "rate",
]
cache_noticias = {"hora": 0, "datos": None}
DURACION_NOTICIAS = 1800  # 30 minutos

# ---------- Inflacion de paises ----------

# nodo → código de país del Banco Mundial (moneda o líder político, según corresponda)

PAISES_MACRO = {
    "EUR": "EMU", "JPY": "JPN", "GBP": "GBR", "CNY": "CHN",
    "USA": "USA", "CHINA": "CHN", "UE": "EMU", "JAPON": "JPN", "RUSIA": "RUS",
    "BRL": "BRA", "MXN": "MEX", "ZAR": "ZAF",
}

BANDERAS = {
    "EUR": "🇪🇺", "JPY": "🇯🇵", "GBP": "🇬🇧", "CNY": "🇨🇳",
    "USA": "🇺🇸", "CHINA": "🇨🇳", "UE": "🇪🇺", "JAPON": "🇯🇵", "RUSIA": "🇷🇺",
    "BRL": "🇧🇷", "MXN": "🇲🇽", "ZAR": "🇿🇦",
    "CL=F": "🇺🇸", "BZ=F": "🇬🇧", "GC=F": "🇬🇧", "NG=F": "🇺🇸", "^TNX": "🇺🇸",
    "HG=F": "🇨🇱", "KC=F": "🇧🇷", "ZS=F": "🇦🇷", "PL=F": "🇿🇦",
}

COLOR_MARCA = {
    "SNDK": "#CC0000", "META": "#0866FF", "TSLA": "#CC0000", "MSFT": "#00A4EF",
    "AMZN": "#FF9900", "GOOGL": "#4285F4", "NVDA": "#76B900", "GS": "#7399C6",
    "BZ=F": "#1B1B1B", "CL=F": "#1B1B1B", "GC=F": "#D4AF37", "NG=F": "#2E86AB",
    "^TNX": "#4B5563", "HG=F": "#B87333", "KC=F": "#6F4E37", "ZS=F": "#8B9B4A", "PL=F": "#94A3B8",
}

INDICADORES_BANCO_MUNDIAL = {
    "inflacion": "FP.CPI.TOTL.ZG",
    "pib": "NY.GDP.MKTP.KD.ZG",
    "desempleo": "SL.UEM.TOTL.ZS",
    "tasa_interes": "FR.INR.RINR",
}
BANCO_MUNDIAL_API = "https://api.worldbank.org/v2/country/{codigo}/indicator/{indicador}"
cache_macro = {"hora": 0, "datos": None}
DURACION_MACRO = 86400 * 7

# ---------- asistente de IA ----------
cliente_ia = anthropic.Anthropic() if os.environ.get("ANTHROPIC_API_KEY") else None
MODELO_IA = "claude-sonnet-5"
INSTRUCCIONES_IA = (
    "Eres el asistente de Investing.AI, una página sobre divisas internacionales "
    "y su relación con la geopolítica. Responde en español, de forma clara y breve. "
    "Explica conceptos de tipo de cambio, bancos centrales y economía mundial. "
    "No des recomendaciones de inversión personalizadas y aclara que tus respuestas "
    "son informativas y no asesoría financiera."
)
MAX_MENSAJES = 20
MAX_CARACTERES = 2000

CARPETA_SENTIMIENTO = Path("sentimiento")
CARPETA_SENTIMIENTO.mkdir(exist_ok=True)

LIDERES = {
    "USA": ["Trump", "White House", "Estados Unidos"],
    "CHINA": ["Xi Jinping", "Beijing", "China"],
    "UE": ["Von der Leyen", "European Union", "Unión Europea"],
    "JAPON": ["Japan", "Tokyo", "Japón"],
    "RUSIA": ["Putin", "Kremlin", "Rusia"],
}

PALABRAS_GDELT = {
    "USA": ["Trump"],
    "CHINA": ["Xi Jinping"],
    "UE": ["Von der Leyen"],
    "JAPON": ["Japan"],
    "RUSIA": ["Putin"],
}

PROMPT_SENTIMIENTO = (
    "Vas a recibir titulares de noticias reales. Clasifica el sentimiento geopolítico "
    "general hacia la economía global en una escala de -1 (muy negativo: guerra, crisis, "
    "sanciones) a 1 (muy positivo: cooperación, distensión, acuerdos). "
    "Responde ÚNICAMENTE con un número decimal (ej: -0.4), sin texto adicional."
)

# ---------- monedas emergentes (Latinoamérica y África) ----------
MONEDAS_EMERGENTES = {"BRL": "Real brasileño", "MXN": "Peso mexicano", "ZAR": "Rand sudafricano"}
cache_emergentes = {"hora": 0, "datos": None}
DURACION_EMERGENTES = 3600

def obtener_tendencias_emergentes():
    ahora = time()
    if cache_emergentes["datos"] and ahora - cache_emergentes["hora"] < DURACION_EMERGENTES:
        return cache_emergentes["datos"]

    hasta = date.today()
    desde = hasta - timedelta(days=30)
    respuesta = requests.get(
        f"{API}/{desde}..{hasta}",
        params={"base": "USD", "symbols": ",".join(MONEDAS_EMERGENTES)},
        timeout=10,
    )
    respuesta.raise_for_status()
    tasas = respuesta.json()["rates"]

    series = {}
    for codigo in MONEDAS_EMERGENTES:
        serie = {f: tasas[f][codigo] for f in tasas if codigo in tasas[f]}
        if serie:
            series[codigo] = serie

    cache_emergentes["hora"] = ahora
    cache_emergentes["datos"] = series
    return series

# ---------- Commodities ----------

COMMODITIES = {
    "BZ=F": "Petróleo Brent",
    "CL=F": "Petróleo WTI",
    "GC=F": "Oro",
    "NG=F": "Gas natural",
    "^TNX": "Bono EE. UU. 10 años",
    "HG=F": "Cobre",
    "KC=F": "Café",
    "ZS=F": "Soja",
    "PL=F": "Platino",
}

# ---------- acciones (Yahoo Finance) ----------
ACCIONES = {
    "SNDK": "Sandisk",
    "META": "Meta",
    "TSLA": "Tesla",
    "MSFT": "Microsoft",
    "AMZN": "Amazon",
    "GOOGL": "Alphabet",
    "NVDA": "Nvidia",
    "GS": "Goldman Sachs",
}
YAHOO_GRAFICO = "https://query1.finance.yahoo.com/v8/finance/chart/"
cache_acciones = {"hora": 0, "datos": None}
DURACION_ACCIONES = 300  # 5 minutos

# ---------- mapa de contagio financiero (correlaciones) ----------
cache_correlaciones = {"hora": 0, "datos": None}
DURACION_CORRELACIONES = 3600  # 1 hora

# coordenadas (lat, lon) para ubicar cada nodo en el mapa mundial
COORDENADAS = {
    # monedas: centro financiero/político de la zona
    "EUR": (50.85, 4.35),     # Bruselas
    "JPY": (35.68, 139.69),   # Tokio
    "GBP": (51.51, -0.13),    # Londres
    "CNY": (39.90, 116.40),   # Pekín

    # commodities: hub de referencia del benchmark
    "CL=F": (35.98, -96.77),  # Cushing, Oklahoma (WTI)
    "BZ=F": (59.50, 1.50),    # Mar del Norte (Brent)
    "GC=F": (51.51, -0.13),   # Londres (London Bullion Market)
    "NG=F": (29.85, -93.35),  # Henry Hub, Luisiana
    "^TNX": (38.90, -77.04),  # Washington D.C. (Tesoro de EE. UU.)
    "HG=F": (-33.45, -70.65), # Santiago de Chile (principal productor de cobre)
    "KC=F": (-15.79, -47.88), # Brasil (principal productor de café)
    "ZS=F": (-34.60, -58.38), # Buenos Aires (soja, cono sur)
    "PL=F": (-25.75, 28.19),  # Sudáfrica (principal productor de platino)
    "BRL": (-15.79, -47.88),  # Brasília
    "MXN": (19.43, -99.13),   # Ciudad de México
    "ZAR": (-25.75, 28.19),   # Pretoria

    # declaraciones por líder: capital política
    # declaraciones por líder: capital política
    "USA": (38.90, -77.04),   # Washington D.C.
    "CHINA": (39.90, 116.40), # Pekín
    "UE": (50.85, 4.35),      # Bruselas
    "JAPON": (35.68, 139.69), # Tokio
    "RUSIA": (55.75, 37.62),  # Moscú
}

# ancla geográfica del clúster de empresas (Wall Street, Nueva York);
# todas las acciones de ACCIONES son de EE. UU., así que comparten este punto
# y el frontend las despliega en abanico alrededor de él
COORDENADAS_CLUSTER_ACCIONES = (40.71, -74.01)
GRUPO_ACCIONES = "acciones_usa"

GDELT_API = "https://api.gdeltproject.org/api/v2/doc/doc"
cache_gdelt = {}
DURACION_GDELT = 21600  # 6 horas

def obtener_tendencia_gdelt(clave_cache, palabras_clave, timespan="1d"):
    ahora = time()
    guardado = cache_gdelt.get(clave_cache)
    if guardado and ahora - guardado["hora"] < DURACION_GDELT:
        return guardado["datos"]

    consulta = " OR ".join(f'"{p}"' if " " in p else p for p in palabras_clave)
    resultado = guardado["datos"] if guardado else None

    for intento, espera in enumerate((0, 5, 15)):
        if espera:
            time_module_sleep(espera)
        try:
            respuesta_vol = requests.get(
                GDELT_API,
                params={"query": consulta, "mode": "timelinevolinfo", "timespan": timespan, "format": "json"},
                headers=HEADERS_WEB, timeout=25,
            )
            if respuesta_vol.status_code == 429:
                print(f"Error GDELT: {clave_cache} 429, reintento {intento + 1}/3")
                continue
            respuesta_vol.raise_for_status()
            serie_vol = respuesta_vol.json()["timeline"][0]["data"]
            volumen = serie_vol[-1]["value"] if serie_vol else 0.0
            resultado = {"volumen": round(volumen, 3)}
            break
        except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as error:
            print("Error GDELT:", clave_cache, type(error).__name__, error)
            break

    cache_gdelt[clave_cache] = {"hora": ahora, "datos": resultado}
    return resultado

def calcular_sentimiento_ia(titulares):
    if cliente_ia is None or not titulares:
        return None
    texto = "\n".join(f"- {t}" for t in titulares[:15])
    try:
        respuesta = cliente_ia.messages.create(
            model=MODELO_IA,
            max_tokens=10,
            system=PROMPT_SENTIMIENTO,
            messages=[{"role": "user", "content": texto}],
        )
        crudo = "".join(b.text for b in respuesta.content if b.type == "text").strip()
        return max(-1.0, min(1.0, float(crudo)))
    except (anthropic.APIError, ValueError) as error:
        print("Error sentimiento IA:", type(error).__name__, error)
        return None


def guardar_sentimiento_diario(clave, valor):
    archivo = CARPETA_SENTIMIENTO / f"{clave}.json"
    historial = json.loads(archivo.read_text(encoding="utf-8")) if archivo.exists() else {}
    if valor is not None:
        historial[date.today().isoformat()] = valor
        limite = (date.today() - timedelta(days=365)).isoformat()
        historial = {f: v for f, v in historial.items() if f >= limite}
        archivo.write_text(json.dumps(historial), encoding="utf-8")
    return historial

def leer_historial_sentimiento(clave):
    archivo = CARPETA_SENTIMIENTO / f"{clave}.json"
    if not archivo.exists():
        return {}
    return json.loads(archivo.read_text(encoding="utf-8"))


def obtener_series_sentimiento():
    series = {"Sentimiento global": leer_historial_sentimiento("global")}
    for lider in LIDERES:
        historial = leer_historial_sentimiento(f"lider_{lider}")
        if historial:
            series[f"Declaraciones {lider}"] = historial
    return series

def obtener_indicadores_macro():
    ahora = time()
    if cache_macro["datos"] and ahora - cache_macro["hora"] < DURACION_MACRO:
        return cache_macro["datos"]

    resultado = {}
    for pais in set(PAISES_MACRO.values()):
        indicadores_pais = {}
        for nombre, codigo_indicador in INDICADORES_BANCO_MUNDIAL.items():
            try:
                respuesta = requests.get(
                    BANCO_MUNDIAL_API.format(codigo=pais, indicador=codigo_indicador),
                    params={"format": "json", "mrnev": 1},
                    timeout=10,
                )
                respuesta.raise_for_status()
                filas = respuesta.json()[1]
                valor = filas[0]["value"]
                if valor is not None:
                    indicadores_pais[nombre] = {"valor": round(valor, 2), "anio": filas[0]["date"]}
            except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as error:
                print("Error indicador macro:", pais, nombre, type(error).__name__, error)
        if indicadores_pais:
            resultado[pais] = indicadores_pais

    cache_macro["hora"] = ahora
    cache_macro["datos"] = resultado
    return resultado

def leer_historial_accion(simbolo):
    respuesta = requests.get(
        YAHOO_GRAFICO + urllib.parse.quote(simbolo, safe="="),
        params={"range": "1mo", "interval": "1d"},
        headers=HEADERS_WEB,
        timeout=10,
    )
    respuesta.raise_for_status()
    resultado = respuesta.json()["chart"]["result"][0]
    marcas = resultado["timestamp"]
    cierres = resultado["indicators"]["quote"][0]["close"]

    serie = {}
    for marca, precio in zip(marcas, cierres):
        if precio is None:
            continue
        fecha = datetime.fromtimestamp(marca, tz=timezone.utc).strftime("%Y-%m-%d")
        serie[fecha] = precio
    return serie


def obtener_historiales_acciones():
    def segura(simbolo):
        try:
            return simbolo, leer_historial_accion(simbolo)
        except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as error:
            print("Error historial acción:", simbolo, type(error).__name__, error)
            return simbolo, {}

    with ThreadPoolExecutor(max_workers=8) as pool:
        return dict(pool.map(segura, ACCIONES))


def pearson(x, y):
    n = len(x)
    if n < 2:
        return 0.0
    mx, my = sum(x) / n, sum(y) / n
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    vx = sum((xi - mx) ** 2 for xi in x)
    vy = sum((yi - my) ** 2 for yi in y)
    if vx == 0 or vy == 0:
        return 0.0
    return cov / math.sqrt(vx * vy)

def desvio_estandar(valores):
    n = len(valores)
    if n < 2:
        return 0.0
    media = sum(valores) / n
    varianza = sum((v - media) ** 2 for v in valores) / (n - 1)
    return varianza ** 0.5

def retornos(serie):
    fechas = sorted(serie)
    return {
        fechas[i]: (serie[fechas[i]] / serie[fechas[i - 1]] - 1)
        for i in range(1, len(fechas))
        if serie[fechas[i - 1]]
    }

def obtener_correlaciones():
    macro = obtener_indicadores_macro()
    ahora = time()
    if cache_correlaciones["datos"] and ahora - cache_correlaciones["hora"] < DURACION_CORRELACIONES:
        return cache_correlaciones["datos"]

    series = {}

    try:
        tendencias_1m = obtener_tendencias("1m")
        for codigo, moneda in tendencias_1m["monedas"].items():
            serie = dict(zip(tendencias_1m["fechas"], moneda["valores"]))
            series[codigo] = retornos(serie)
    except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as error:
        print("Error tendencias para correlaciones:", type(error).__name__, error)

    def segura(simbolo):
        try:
            return simbolo, retornos(leer_historial_accion(simbolo))
        except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as error:
            print("Error historial acción:", simbolo, type(error).__name__, error)
            return simbolo, None

    with ThreadPoolExecutor(max_workers=8) as pool:
        resultados = list(pool.map(segura, ACCIONES))
    for simbolo, serie in resultados:
        if serie:
            series[simbolo] = serie

    def segura_commodity(simbolo):
        try:
            return simbolo, retornos(leer_historial_accion(simbolo))
        except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as error:
            print("Error historial commodity:", simbolo, type(error).__name__, error)
            return simbolo, None

    with ThreadPoolExecutor(max_workers=4) as pool:
        resultados_commodities = list(pool.map(segura_commodity, COMMODITIES))
    for simbolo, serie in resultados_commodities:
        if serie:
            series[simbolo] = serie

    try:
        for codigo, serie in obtener_tendencias_emergentes().items():
            series[codigo] = retornos(serie)
    except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as error:
        print("Error monedas emergentes:", type(error).__name__, error)

    # Punto 3: sentimiento geopolítico general
    try:
        titulares_generales = [n["titulo"] for n in obtener_noticias()]
    except Exception:
        titulares_generales = []
    sentimiento_global = calcular_sentimiento_ia(titulares_generales)
    historial_global = guardar_sentimiento_diario("global", sentimiento_global)
    if len(historial_global) >= 5:
        series["SENTIMIENTO"] = historial_global

    # Punto 4: declaraciones/menciones por líder
    tendencia_por_lider = {}
    for lider, palabras_clave in LIDERES.items():
        titulares_lider = [titulo for titulo in titulares_generales if any(p.lower() in titulo.lower() for p in palabras_clave)]
        sentimiento_lider = calcular_sentimiento_ia(titulares_lider) if titulares_lider else None
        historial_lider = guardar_sentimiento_diario(f"lider_{lider}", sentimiento_lider)
        tendencia_por_lider[lider] = obtener_tendencia_gdelt(lider, PALABRAS_GDELT[lider])
        time_module_sleep(2)
        if len(historial_lider) >= 5:
            series[lider] = historial_lider

    aristas = []
    claves = list(series.keys())
    for i in range(len(claves)):
        for j in range(i + 1, len(claves)):
            a, b = claves[i], claves[j]
            fechas_comunes = sorted(set(series[a]) & set(series[b]))
            if len(fechas_comunes) < 5:
                continue
            valores_a = [series[a][f] for f in fechas_comunes]
            valores_b = [series[b][f] for f in fechas_comunes]
            correlacion = pearson(valores_a, valores_b)
            if correlacion is not None:
                aristas.append({"a": a, "b": b, "valor": round(correlacion, 3)})

    centralidad = {clave: 0.0 for clave in claves}
    for arista in aristas:
        centralidad[arista["a"]] += abs(arista["valor"])
        centralidad[arista["b"]] += abs(arista["valor"])

    NOMBRES_SENTIMIENTO = {"SENTIMIENTO": "Sentimiento global", **{l: f"Declaraciones {l}" for l in LIDERES}}

    nodos = []
    for codigo in claves:
        serie = series[codigo]
        if not serie:
            continue
        ultima_fecha = max(serie)
        es_sentimiento = codigo in NOMBRES_SENTIMIENTO
        es_accion = codigo in ACCIONES

        # categoría para el color del borde en el mapa: económico / político / social
        if codigo in MONEDAS or codigo in MONEDAS_EMERGENTES or codigo in COMMODITIES or es_accion:
            categoria = "economico"
        elif codigo == "SENTIMIENTO":
            categoria = "social"
        elif codigo in LIDERES:
            categoria = "politico"
        else:
            categoria = "economico"

        # coordenadas: las acciones comparten el ancla del clúster, el resto
        # usa su propia entrada en COORDENADAS; el sentimiento global no
        # tiene ubicación (queda None y el frontend lo trata como flotante)
        if es_accion:
            lat, lon = COORDENADAS_CLUSTER_ACCIONES
        else:
            lat, lon = COORDENADAS.get(codigo, (None, None))

        nodos.append({
            "id": codigo,
            "nombre": MONEDAS.get(codigo, MONEDAS_EMERGENTES.get(codigo, COMMODITIES.get(codigo, ACCIONES.get(codigo, NOMBRES_SENTIMIENTO.get(codigo, codigo))))),
            "tipo": "moneda" if (codigo in MONEDAS or codigo in MONEDAS_EMERGENTES) else ("commodity" if codigo in COMMODITIES else ("sentimiento" if es_sentimiento else "accion")),
            "lat": lat,
            "lon": lon,
            "grupo": GRUPO_ACCIONES if es_accion else None,
            "variacion": round(serie[ultima_fecha] * (1 if es_sentimiento else 100), 2),
            "volatilidad": round(desvio_estandar(list(serie.values())) * (1 if es_sentimiento else 100), 2),
            "centralidad": round(centralidad[codigo], 2),
            "macro": macro.get(PAISES_MACRO.get(codigo)),
            "bandera": BANDERAS.get(codigo, "🇺🇸" if es_accion else None),
            "color_marca": COLOR_MARCA.get(codigo),
            "volumen_tendencia": tendencia_por_lider.get(codigo, {}).get("volumen") if tendencia_por_lider.get(codigo) else None,
        })

    datos = {"nodos": nodos, "pares": aristas}
    if len(nodos) < 2:
        raise RuntimeError("No hay suficientes series con datos para calcular correlaciones")

    guardar_snapshot_si_nuevo(datos)

    cache_correlaciones["hora"] = ahora
    cache_correlaciones["datos"] = datos
    return datos

UMBRAL_DESVIOS = 2.0       # cuántos desvíos estándar de diferencia se considera "inusual"
MINIMO_HISTORIAL = 4       # días de historial previos necesarios para evaluar un par
DESVIO_MINIMO = 0.05       # piso para no marcar como alerta un par casi constante


def cargar_todos_los_snapshots():
    snapshots = []
    for archivo in sorted(CARPETA_SNAPSHOTS.glob("*.json")):
        try:
            snapshots.append(json.loads(archivo.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
    return snapshots


def obtener_alertas():
    snapshots = cargar_todos_los_snapshots()
    if len(snapshots) < MINIMO_HISTORIAL + 1:
        return {"alertas": [], "dias_historial": len(snapshots)}

    *anteriores, hoy = snapshots
    nombres = {n["id"]: n["nombre"] for n in hoy["nodos"]}

    historico = {}
    for snapshot in anteriores:
        for par in snapshot["pares"]:
            clave = tuple(sorted((par["a"], par["b"])))
            historico.setdefault(clave, []).append(par["valor"])

    ultimo_anterior = {}
    for par in anteriores[-1]["pares"]:
        ultimo_anterior[tuple(sorted((par["a"], par["b"])))] = par["valor"]

    alertas = []
    for par in hoy["pares"]:
        clave = tuple(sorted((par["a"], par["b"])))
        serie = historico.get(clave, [])
        if len(serie) < MINIMO_HISTORIAL:
            continue

        promedio = sum(serie) / len(serie)
        desvio = max(desvio_estandar(serie), DESVIO_MINIMO)
        z = (par["valor"] - promedio) / desvio

        valor_previo = ultimo_anterior.get(clave)
        cambio_de_signo = (
            valor_previo is not None
            and abs(valor_previo) >= 0.15 and abs(par["valor"]) >= 0.15
            and (valor_previo > 0) != (par["valor"] > 0)
        )

        if abs(z) >= UMBRAL_DESVIOS or cambio_de_signo:
            alertas.append({
                "a": par["a"],
                "b": par["b"],
                "nombre_a": nombres.get(par["a"], par["a"]),
                "nombre_b": nombres.get(par["b"], par["b"]),
                "valor_hoy": par["valor"],
                "promedio_historico": round(promedio, 3),
                "z": round(z, 2),
                "cambio_de_signo": cambio_de_signo,
                "dias_historial": len(serie),
            })

    alertas.sort(key=lambda a: abs(a["z"]), reverse=True)
    return {"alertas": alertas[:8], "dias_historial": len(anteriores)}

# ---------- resumen del día (IA) ----------
ARCHIVO_RESUMEN = Path("resumen_dia.json")
cache_resumen = {"fecha": None, "texto": None}

PROMPT_RESUMEN = (
    "Eres el analista de G20.VIX. A partir de los titulares y datos que recibas, "
    "escribe UN SOLO PÁRRAFO (máximo 80 palabras) en español, en tono periodístico "
    "y directo, resumiendo qué está pasando hoy a nivel político, económico y social "
    "en el mundo y cómo se conecta con el movimiento de divisas, acciones y "
    "commodities. No uses encabezados, listas ni markdown, solo el párrafo. "
    "No des recomendaciones de inversión."
)


def obtener_resumen_dia():
    hoy = date.today().isoformat()

    if cache_resumen["fecha"] == hoy and cache_resumen["texto"]:
        return cache_resumen["texto"]

    if ARCHIVO_RESUMEN.exists():
        guardado = json.loads(ARCHIVO_RESUMEN.read_text(encoding="utf-8"))
        if guardado.get("fecha") == hoy and guardado.get("texto"):
            cache_resumen.update(guardado)
            return guardado["texto"]

    if cliente_ia is None:
        raise RuntimeError("Asistente no disponible para generar el resumen")

    try:
        titulares = [n["titulo"] for n in obtener_noticias()]
    except Exception:
        titulares = []

    lineas_corr = []
    try:
        correlaciones = obtener_correlaciones()
        nombres = {n["id"]: n["nombre"] for n in correlaciones["nodos"]}
        principales = sorted(correlaciones["pares"], key=lambda p: abs(p["valor"]), reverse=True)[:5]
        lineas_corr = [
            f"{nombres.get(p['a'], p['a'])} y {nombres.get(p['b'], p['b'])}: {p['valor']:+.2f}"
            for p in principales
        ]
    except Exception:
        pass

    entrada = "Titulares de hoy:\n" + "\n".join(f"- {t}" for t in titulares[:15])
    if lineas_corr:
        entrada += "\n\nCorrelaciones más fuertes del último mes:\n" + "\n".join(lineas_corr)

    respuesta = cliente_ia.messages.create(
        model=MODELO_IA,
        max_tokens=220,
        system=PROMPT_RESUMEN,
        messages=[{"role": "user", "content": entrada}],
    )
    texto = "".join(b.text for b in respuesta.content if b.type == "text").strip()

    cache_resumen.update({"fecha": hoy, "texto": texto})
    ARCHIVO_RESUMEN.write_text(json.dumps({"fecha": hoy, "texto": texto}, ensure_ascii=False), encoding="utf-8")
    return texto

# ---------- cadenas de televisión (artículos y videos) ----------
FEEDS_ARTICULOS = {
    "BBC News": "https://feeds.bbci.co.uk/news/business/rss.xml",
    "Al Jazeera": "https://www.aljazeera.com/xml/rss/all.xml",
    "CNBC": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "Sky News": "https://feeds.skynews.com/feeds/rss/business.xml",
    "France 24": "https://www.france24.com/en/rss",
    "Bloomberg": "https://feeds.bloomberg.com/markets/news.rss",
}
# ID de canal (empieza con UC) o @usuario del canal de YouTube
CANALES_VIDEO = {
    "Bloomberg TV": "UCIALMKvObZNtJ6AmdCLP7Lg",
    "Al Jazeera": "UCNye-wNBqNL5ZzHSJj3l8Bg",
    "CNBC": "@CNBCtelevision",
    "DW News": "@DWNews",
    "Sky News": "@SkyNews",
    "France 24": "@FRANCE24English",
}
NS = {
    "a": "http://www.w3.org/2005/Atom",
    "yt": "http://www.youtube.com/xml/schemas/2015",
}
cache_medios = {"hora": 0, "datos": None}
cache_canales = {}
DURACION_MEDIOS = 900  # 15 minutos


# =====================================================================
# tendencias
# =====================================================================
def obtener_tendencias(periodo):
    ahora = time()
    if periodo in cache and ahora - cache[periodo][0] < DURACION_CACHE:
        return cache[periodo][1]

    hasta = date.today()
    desde = hasta - timedelta(days=PERIODOS[periodo])

    respuesta = requests.get(
        f"{API}/{desde}..{hasta}",
        params={"base": "USD", "symbols": ",".join(MONEDAS)},
        timeout=10,
    )
    respuesta.raise_for_status()
    tasas = respuesta.json()["rates"]

    fechas = sorted(tasas)
    monedas = {}
    for codigo, nombre in MONEDAS.items():
        valores = [tasas[f][codigo] for f in fechas if codigo in tasas[f]]
        variacion = (valores[-1] / valores[0] - 1) * 100
        monedas[codigo] = {
            "nombre": nombre,
            "valores": valores,
            "ultimo": valores[-1],
            "variacion": round(variacion, 2),
        }

    datos = {"fechas": fechas, "monedas": monedas}
    cache[periodo] = (ahora, datos)
    return datos


# =====================================================================
# noticias
# =====================================================================
def leer_feed(url, params=None):
    respuesta = requests.get(url, params=params, headers=HEADERS_WEB, timeout=15)
    print("Noticias:", url, "| estado", respuesta.status_code, "|", len(respuesta.content), "bytes")
    respuesta.raise_for_status()
    return ET.fromstring(respuesta.content).findall("./channel/item")


def obtener_noticias():
    ahora = time()
    if cache_noticias["datos"] and ahora - cache_noticias["hora"] < DURACION_NOTICIAS:
        return cache_noticias["datos"]

    items, filtrar = [], False

    # 1) Google News en español
    for consulta in CONSULTAS_NOTICIAS:
        try:
            items = leer_feed(
                NOTICIAS_API,
                {"q": consulta, "hl": "es-419", "gl": "PE", "ceid": "PE:es-419"},
            )
        except (requests.RequestException, ET.ParseError) as error:
            print("Error Google News:", type(error).__name__, error)
            break
        if items:
            break

    # 2) feeds de respaldo, filtrados por palabras clave
    if not items:
        filtrar = True
        for url in FEEDS_RESPALDO:
            try:
                items += leer_feed(url)
            except (requests.RequestException, ET.ParseError) as error:
                print("Error respaldo:", url, type(error).__name__, error)

    lista, vistos = [], set()
    for item in items:
        titulo = (item.findtext("title") or "").strip()
        resumen = (item.findtext("description") or "").lower()
        fuente = (item.findtext("source") or "").strip()

        if fuente and titulo.endswith(f" - {fuente}"):
            titulo = titulo[: -len(fuente) - 3]
        if not titulo or titulo in vistos:
            continue
        if filtrar and not any(p in (titulo.lower() + " " + resumen) for p in PALABRAS_CLAVE):
            continue
        vistos.add(titulo)

        try:
            fecha = parsedate_to_datetime(item.findtext("pubDate")).strftime("%d/%m/%Y %H:%M")
        except (TypeError, ValueError):
            fecha = ""

        lista.append({
            "titulo": titulo,
            "url": item.findtext("link") or "#",
            "fuente": fuente or "Noticias",
            "fecha": fecha,
        })
        if len(lista) == 8:
            break

    if lista:
        cache_noticias["hora"] = ahora
        cache_noticias["datos"] = lista
        return lista
    if cache_noticias["datos"]:
        return cache_noticias["datos"]
    raise RuntimeError("Ninguna fuente de noticias respondió")


# =====================================================================
# acciones
# =====================================================================
def leer_accion(simbolo):
    respuesta = requests.get(
        YAHOO_GRAFICO + simbolo,
        params={"range": "1d", "interval": "1d"},
        headers=HEADERS_WEB,
        timeout=10,
    )
    respuesta.raise_for_status()
    meta = respuesta.json()["chart"]["result"][0]["meta"]

    precio = meta["regularMarketPrice"]
    anterior = meta.get("previousClose") or meta.get("chartPreviousClose")
    variacion = round((precio / anterior - 1) * 100, 2) if anterior else None

    return {
        "simbolo": simbolo,
        "nombre": ACCIONES[simbolo],
        "precio": round(precio, 2),
        "variacion": variacion,
    }


def obtener_acciones():
    ahora = time()
    if cache_acciones["datos"] and ahora - cache_acciones["hora"] < DURACION_ACCIONES:
        return cache_acciones["datos"]

    def segura(simbolo):
        try:
            return leer_accion(simbolo)
        except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as error:
            print("Error acción:", simbolo, type(error).__name__, error)
            return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        resultados = list(pool.map(segura, ACCIONES))
    datos = [r for r in resultados if r]

    if datos:
        cache_acciones["hora"] = ahora
        cache_acciones["datos"] = datos
        return datos
    if cache_acciones["datos"]:
        return cache_acciones["datos"]
    raise RuntimeError("No se pudo consultar ninguna acción")


# =====================================================================
# cadenas de televisión
# =====================================================================
def a_fecha(texto, iso=False):
    try:
        fecha = datetime.fromisoformat(texto) if iso else parsedate_to_datetime(texto)
    except (TypeError, ValueError):
        return None
    if fecha.tzinfo is None:
        fecha = fecha.replace(tzinfo=timezone.utc)
    return fecha


def imagen_de(item):
    for nodo in item.iter():
        etiqueta = nodo.tag.rsplit("}", 1)[-1]
        url = nodo.get("url")
        if not url:
            continue
        tipo = nodo.get("type") or nodo.get("medium") or ""
        if etiqueta == "thumbnail" or (etiqueta in ("content", "enclosure") and tipo.startswith("image")):
            return url
    return None


def leer_articulos(fuente, url):
    try:
        respuesta = requests.get(url, headers=HEADERS_WEB, timeout=12)
        respuesta.raise_for_status()
        items = ET.fromstring(respuesta.content).findall("./channel/item")
    except (requests.RequestException, ET.ParseError) as error:
        print("Error medios:", fuente, type(error).__name__, error)
        return []

    salida = []
    for item in items:
        titulo = (item.findtext("title") or "").strip()
        enlace = (item.findtext("link") or "").strip()
        fecha = a_fecha(item.findtext("pubDate"))
        if not titulo or not enlace.startswith(("http://", "https://")) or not fecha:
            continue

        imagen = imagen_de(item)
        if imagen and not imagen.startswith("https://"):
            imagen = None
        if imagen and "ichef.bbci.co.uk" in imagen:
            imagen = imagen.replace("/240/", "/800/")   # miniatura de BBC en mayor tamaño

        salida.append({
            "tipo": "articulo",
            "fuente": fuente,
            "titulo": titulo,
            "url": enlace,
            "imagen": imagen,
            "ts": int(fecha.timestamp()),
        })
        if len(salida) == 3:
            break
    return salida


def resolver_canal(valor):
    if valor.startswith("UC"):
        return valor
    if valor in cache_canales:
        return cache_canales[valor]

    respuesta = requests.get(
        f"https://www.youtube.com/{valor}",
        headers=HEADERS_WEB,
        cookies={"CONSENT": "YES+1"},
        timeout=12,
    )
    respuesta.raise_for_status()
    for patron in (
        r'rel="canonical" href="https://www\.youtube\.com/channel/(UC[\w-]{22})"',
        r'"externalId":"(UC[\w-]{22})"',
    ):
        encontrado = re.search(patron, respuesta.text)
        if encontrado:
            cache_canales[valor] = encontrado.group(1)
            return encontrado.group(1)
    raise ValueError(f"No se encontró el ID del canal {valor}")


def leer_videos(fuente, canal):
    try:
        id_canal = resolver_canal(canal)
        respuesta = requests.get(
            "https://www.youtube.com/feeds/videos.xml",
            params={"channel_id": id_canal},
            headers=HEADERS_WEB,
            timeout=12,
        )
        respuesta.raise_for_status()
        entradas = ET.fromstring(respuesta.content).findall("a:entry", NS)
    except (requests.RequestException, ET.ParseError, ValueError) as error:
        print("Error videos:", fuente, type(error).__name__, error)
        return []

    salida = []
    for entrada in entradas:
        id_video = entrada.findtext("yt:videoId", namespaces=NS) or ""
        titulo = (entrada.findtext("a:title", namespaces=NS) or "").strip()
        fecha = a_fecha(entrada.findtext("a:published", namespaces=NS), iso=True)
        if not re.fullmatch(r"[\w-]{11}", id_video) or not titulo or not fecha:
            continue

        salida.append({
            "tipo": "video",
            "fuente": fuente,
            "titulo": titulo,
            "url": f"https://www.youtube.com/watch?v={id_video}",
            "video_id": id_video,
            "imagen": f"https://i.ytimg.com/vi/{id_video}/maxresdefault.jpg",
            "imagen_respaldo": f"https://i.ytimg.com/vi/{id_video}/hqdefault.jpg",
            "ts": int(fecha.timestamp()),
        })
        if len(salida) == 2:
            break
    return salida


def obtener_medios():
    ahora = time()
    if cache_medios["datos"] and ahora - cache_medios["hora"] < DURACION_MEDIOS:
        return cache_medios["datos"]

    with ThreadPoolExecutor(max_workers=12) as pool:
        tareas_a = [pool.submit(leer_articulos, f, u) for f, u in FEEDS_ARTICULOS.items()]
        tareas_v = [pool.submit(leer_videos, f, c) for f, c in CANALES_VIDEO.items()]
        articulos = [n for t in tareas_a for n in t.result()]
        videos = [n for t in tareas_v for n in t.result()]

    por_fecha = lambda n: n["ts"]
    videos.sort(key=por_fecha, reverse=True)
    articulos.sort(key=por_fecha, reverse=True)

    # el video más reciente va primero (tarjeta grande) y el resto se ordena por fecha
    principal = videos[:1]
    resto = sorted(videos[1:3] + articulos[:6], key=por_fecha, reverse=True)
    datos = principal + resto

    if datos:
        cache_medios["hora"] = ahora
        cache_medios["datos"] = datos
        return datos
    if cache_medios["datos"]:
        return cache_medios["datos"]
    raise RuntimeError("Ninguna fuente de medios respondió")

CONTRASENA_ADMIN = os.environ.get("ADMIN_PASSWORD")

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
GITHUB_REPO = os.environ.get("GITHUB_REPO")   # formato: "tu-usuario/tu-repo"
GITHUB_RAMA = os.environ.get("GITHUB_RAMA", "main")

def requiere_admin(vista):
    @wraps(vista)
    def envoltura(*args, **kwargs):
        auth = request.authorization
        if not CONTRASENA_ADMIN or not auth or not secrets.compare_digest(auth.password, CONTRASENA_ADMIN):
            return Response(
                "Acceso restringido", 401,
                {"WWW-Authenticate": 'Basic realm="Panel de administración"'},
            )
        return vista(*args, **kwargs)
    return envoltura
CARPETA_ARTICULOS = Path("articulos")
CARPETA_ARTICULOS.mkdir(exist_ok=True)
CARPETA_SNAPSHOTS = Path("snapshots")
CARPETA_SNAPSHOTS.mkdir(exist_ok=True)


def descargar_snapshots_de_github():
    if not (GITHUB_TOKEN and GITHUB_REPO):
        return
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/web/snapshots"
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}
    try:
        respuesta = requests.get(url, headers=headers, params={"ref": GITHUB_RAMA}, timeout=15)
        if respuesta.status_code == 404:
            return
        respuesta.raise_for_status()
        for archivo in respuesta.json():
            if not archivo["name"].endswith(".json"):
                continue
            destino = CARPETA_SNAPSHOTS / archivo["name"]
            if destino.exists():
                continue
            contenido = requests.get(archivo["download_url"], timeout=15)
            contenido.raise_for_status()
            destino.write_bytes(contenido.content)
    except requests.RequestException as error:
        print("Error descargando snapshots de GitHub:", type(error).__name__, error)


descargar_snapshots_de_github()


def subir_snapshot_a_github(fecha, contenido_texto):
    if not (GITHUB_TOKEN and GITHUB_REPO):
        return
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/web/snapshots/{fecha}.json"
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}
    payload = {
        "message": f"Snapshot de correlaciones: {fecha}",
        "content": base64.b64encode(contenido_texto.encode("utf-8")).decode("ascii"),
        "branch": GITHUB_RAMA,
    }
    try:
        respuesta = requests.put(url, headers=headers, json=payload, timeout=15)
        respuesta.raise_for_status()
    except requests.RequestException as error:
        print("Error subiendo snapshot a GitHub:", type(error).__name__, error)


def guardar_snapshot_si_nuevo(datos):
    hoy = date.today().isoformat()
    archivo = CARPETA_SNAPSHOTS / f"{hoy}.json"
    if archivo.exists():
        return

    snapshot = {
        "fecha": hoy,
        "nodos": [{"id": n["id"], "nombre": n["nombre"], "categoria": n["categoria"]} for n in datos["nodos"]],
        "pares": datos["pares"],
    }
    try:
        contenido = json.dumps(snapshot, ensure_ascii=False)
        archivo.write_text(contenido, encoding="utf-8")
        subir_snapshot_a_github(hoy, contenido)
    except OSError as error:
        print("Error guardando snapshot:", type(error).__name__, error)

def descargar_de_github():
    if not (GITHUB_TOKEN and GITHUB_REPO):
        return
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/web/articulos"
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}
    try:
        respuesta = requests.get(url, headers=headers, params={"ref": GITHUB_RAMA}, timeout=15)
        if respuesta.status_code == 404:
            return  # la carpeta todavía no existe en el repositorio
        respuesta.raise_for_status()
        for archivo in respuesta.json():
            if not archivo["name"].endswith(".json"):
                continue
            contenido = requests.get(archivo["download_url"], timeout=15)
            contenido.raise_for_status()
            (CARPETA_ARTICULOS / archivo["name"]).write_bytes(contenido.content)
    except requests.RequestException as error:
        print("Error descargando de GitHub:", type(error).__name__, error)


descargar_de_github()

def generar_slug(titulo):
    base = re.sub(r"[^a-z0-9]+", "-", titulo.lower()).strip("-")
    slug, contador = base, 1
    while (CARPETA_ARTICULOS / f"{slug}.json").exists():
        contador += 1
        slug = f"{base}-{contador}"
    return slug


def texto_a_html(texto):
    parrafos = [p.strip() for p in re.split(r"\n\s*\n", texto.strip()) if p.strip()]
    return "".join(f"<p>{html.escape(p).replace(chr(10), '<br>')}</p>" for p in parrafos)

def subir_a_github(slug, contenido_texto):
    if not (GITHUB_TOKEN and GITHUB_REPO):
        print("GitHub no configurado, se omite la sincronización")
        return
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/web/articulos/{slug}.json"
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}
    payload = {
        "message": f"Nuevo artículo: {slug}",
        "content": base64.b64encode(contenido_texto.encode("utf-8")).decode("ascii"),
        "branch": GITHUB_RAMA,
    }
    try:
        respuesta = requests.put(url, headers=headers, json=payload, timeout=15)
        respuesta.raise_for_status()
    except requests.RequestException as error:
        print("Error subiendo a GitHub:", type(error).__name__, error)


def eliminar_de_github(slug):
    if not (GITHUB_TOKEN and GITHUB_REPO):
        return
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/web/articulos/{slug}.json"
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}
    try:
        info = requests.get(url, headers=headers, params={"ref": GITHUB_RAMA}, timeout=15)
        info.raise_for_status()
        payload = {"message": f"Eliminar artículo: {slug}", "sha": info.json()["sha"], "branch": GITHUB_RAMA}
        respuesta = requests.delete(url, headers=headers, json=payload, timeout=15)
        respuesta.raise_for_status()
    except requests.RequestException as error:
        print("Error eliminando de GitHub:", type(error).__name__, error)


@app.route("/admin/crear", methods=["POST"])
@requiere_admin
def admin_crear():
    datos = request.form
    titulo = (datos.get("titulo") or "").strip()
    cuerpo = (datos.get("cuerpo") or "").strip()
    if not titulo or not cuerpo:
        return jsonify(error="Falta el título o el texto"), 400

    slug = generar_slug(titulo)
    articulo = {
        "titulo": titulo,
        "bajada": (datos.get("bajada") or "").strip(),
        "autor": (datos.get("autor") or "").strip(),
        "materia": (datos.get("materia") or "").strip(),
        "genero": datos.get("genero") if datos.get("genero") in ("hombre", "mujer") else "",
        "cuerpo_html": texto_a_html(cuerpo),
        "slug": slug,
        "fecha": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "ts": int(time()),
    }
    contenido = json.dumps(articulo, ensure_ascii=False)
    (CARPETA_ARTICULOS / f"{slug}.json").write_text(contenido, encoding="utf-8")
    subir_a_github(slug, contenido)
    return jsonify(articulo)

@app.route("/admin")
@requiere_admin
def admin():
    articulos = sorted(CARPETA_ARTICULOS.glob("*.json"), reverse=True)
    lista = [json.loads(a.read_text(encoding="utf-8")) for a in articulos]
    return render_template("admin.html", articulos=lista)


@app.route("/admin/eliminar/<slug>", methods=["POST"])
@requiere_admin
def admin_eliminar(slug):
    slug_limpio = re.sub(r'[^a-z0-9-]', '', slug)
    (CARPETA_ARTICULOS / f"{slug_limpio}.json").unlink(missing_ok=True)
    eliminar_de_github(slug_limpio)
    return jsonify(ok=True)

@app.route("/articulos/<slug>")
def ver_articulo(slug):
    ruta = CARPETA_ARTICULOS / f"{re.sub(r'[^a-z0-9-]', '', slug)}.json"
    if not ruta.exists():
        return "No encontrado", 404
    return render_template("articulo.html", articulo=json.loads(ruta.read_text(encoding="utf-8")))

@app.route("/api/articulos")
def listar_articulos():
    archivos = sorted(CARPETA_ARTICULOS.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    articulos = []
    for archivo in archivos:
        try:
            datos = json.loads(archivo.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        articulos.append({
            "slug": datos.get("slug", archivo.stem),
            "titulo": datos.get("titulo", ""),
            "bajada": datos.get("bajada", ""),
            "fecha": datos.get("fecha", ""),
            "autor": datos.get("autor", ""),
            "materia": datos.get("materia", ""),
            "genero": datos.get("genero", ""),
            "ts": datos.get("ts", int(archivo.stat().st_mtime)),
        })
    return jsonify(articulos)
# =====================================================================
# rutas
# =====================================================================
@app.route("/")
def inicio():
    return render_template("index.html")


@app.route("/api/tendencias")
def tendencias():
    periodo = request.args.get("periodo", "6m")
    if periodo not in PERIODOS:
        return jsonify(error="Periodo no válido"), 400
    try:
        return jsonify(obtener_tendencias(periodo))
    except (requests.RequestException, KeyError, IndexError):
        return jsonify(error="No se pudo consultar el tipo de cambio"), 502


@app.route("/api/noticias")
def noticias():
    try:
        return jsonify(obtener_noticias())
    except Exception as error:
        traceback.print_exc()
        return jsonify(error=f"{type(error).__name__}: {error}"), 502


@app.route("/api/acciones")
def acciones():
    try:
        return jsonify(obtener_acciones())
    except Exception as error:
        traceback.print_exc()
        return jsonify(error=f"{type(error).__name__}: {error}"), 502


@app.route("/api/medios")
def medios():
    try:
        return jsonify(obtener_medios())
    except Exception as error:
        traceback.print_exc()
        return jsonify(error=f"{type(error).__name__}: {error}"), 502


@app.route("/api/chat", methods=["POST"])
def chat():
    if cliente_ia is None:
        return jsonify(error="Asistente no disponible"), 503

    datos = request.get_json(silent=True) or {}
    historial = datos.get("mensajes", [])

    if not isinstance(historial, list) or not historial:
        return jsonify(error="Mensaje vacío"), 400

    mensajes = []
    for m in historial[-MAX_MENSAJES:]:
        if (
            not isinstance(m, dict)
            or m.get("role") not in ("user", "assistant")
            or not isinstance(m.get("content"), str)
            or not m["content"].strip()
        ):
            return jsonify(error="Formato no válido"), 400
        mensajes.append({"role": m["role"], "content": m["content"][:MAX_CARACTERES]})

    if mensajes[0]["role"] != "user" or mensajes[-1]["role"] != "user":
        return jsonify(error="Formato no válido"), 400

    try:
        respuesta = cliente_ia.messages.create(
            model=MODELO_IA,
            max_tokens=800,
            system=INSTRUCCIONES_IA,
            messages=mensajes,
        )
        texto = "".join(b.text for b in respuesta.content if b.type == "text")
        return jsonify(respuesta=texto)
    except anthropic.APIError as error:
        print("Error IA:", type(error).__name__, error)
        return jsonify(error="No se pudo obtener respuesta"), 502

@app.route("/api/correlaciones")
def correlaciones():
    try:
        return jsonify(obtener_correlaciones())
    except Exception as error:
        traceback.print_exc()
        return jsonify(error=f"{type(error).__name__}: {error}"), 502

@app.route("/api/resumen-dia")
def resumen_dia():
    try:
        return jsonify(fecha=date.today().isoformat(), texto=obtener_resumen_dia())
    except Exception as error:
        traceback.print_exc()
        return jsonify(error=f"{type(error).__name__}: {error}"), 502

@app.route("/api/sentimiento")
def sentimiento():
    try:
        return jsonify(obtener_series_sentimiento())
    except Exception as error:
        traceback.print_exc()
        return jsonify(error=f"{type(error).__name__}: {error}"), 502

@app.route("/api/snapshots")
def snapshots_disponibles():
    fechas = sorted(p.stem for p in CARPETA_SNAPSHOTS.glob("*.json"))
    return jsonify(fechas)

@app.route("/api/snapshot/<fecha>")
def snapshot_por_fecha(fecha):
    fecha_limpia = re.sub(r"[^0-9-]", "", fecha)
    ruta = CARPETA_SNAPSHOTS / f"{fecha_limpia}.json"
    if not ruta.exists():
        return jsonify(error="Snapshot no encontrado"), 404
    return jsonify(json.loads(ruta.read_text(encoding="utf-8")))

@app.route("/api/alertas")
def alertas():
    try:
        return jsonify(obtener_alertas())
    except Exception as error:
        traceback.print_exc()
        return jsonify(error=f"{type(error).__name__}: {error}"), 502

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)