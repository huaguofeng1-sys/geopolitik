import os
import re
import csv
import io
import traceback
import xml.etree.ElementTree as ET
import secrets
import json
import html
import base64
import math
import urllib.parse
import threading
from pathlib import Path
from functools import wraps
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from time import time
from time import sleep as time_module_sleep
import anthropic
import requests
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.ensemble import RandomForestRegressor
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
    "DX-Y.NYB": "🇺🇸",
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

PROMPT_SENTIMIENTO = (
    "Vas a recibir titulares de noticias reales. Clasifica el sentimiento geopolítico "
    "general hacia la economía global en una escala de -1 (muy negativo: guerra, crisis, "
    "sanciones) a 1 (muy positivo: cooperación, distensión, acuerdos). "
    "Responde ÚNICAMENTE con un número decimal (ej: -0.4), sin texto adicional."
)

# ---------- análisis de sentimiento local (FinBERT) ----------
DISPOSITIVO_TORCH = "cuda" if torch.cuda.is_available() else "cpu"
NOMBRE_MODELO_SENTIMIENTO = "ProsusAI/finbert"
tokenizador_sentimiento = AutoTokenizer.from_pretrained(NOMBRE_MODELO_SENTIMIENTO)
modelo_sentimiento = AutoModelForSequenceClassification.from_pretrained(NOMBRE_MODELO_SENTIMIENTO)
modelo_sentimiento.to(DISPOSITIVO_TORCH)
modelo_sentimiento.eval()
print(f"FinBERT cargado en: {DISPOSITIVO_TORCH}")


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
    "DX-Y.NYB": "Índice dólar (DXY)",
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
    "DX-Y.NYB": (38.90, -77.04),  # Washington D.C.

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

def calcular_sentimiento_ia(titulares):
    if not titulares:
        return None
    textos = titulares[:15]
    try:
        entradas = tokenizador_sentimiento(
            textos, padding=True, truncation=True, max_length=128, return_tensors="pt"
        ).to(DISPOSITIVO_TORCH)

        with torch.no_grad():
            salida = modelo_sentimiento(**entradas)
            probabilidades = torch.nn.functional.softmax(salida.logits, dim=-1)

        id2label = modelo_sentimiento.config.id2label
        idx_pos = next(i for i, n in id2label.items() if n.lower() == "positive")
        idx_neg = next(i for i, n in id2label.items() if n.lower() == "negative")

        # score por titular: prob. positiva menos prob. negativa (neutral no aporta), en [-1, 1]
        puntajes = (probabilidades[:, idx_pos] - probabilidades[:, idx_neg]).tolist()
        promedio = sum(puntajes) / len(puntajes)
        return max(-1.0, min(1.0, round(promedio, 4)))
    except Exception as error:
        print("Error sentimiento local:", type(error).__name__, error)
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
    for lider, palabras_clave in LIDERES.items():
        titulares_lider = [titulo for titulo in titulares_generales if any(p.lower() in titulo.lower() for p in palabras_clave)]
        sentimiento_lider = calcular_sentimiento_ia(titulares_lider) if titulares_lider else None
        historial_lider = guardar_sentimiento_diario(f"lider_{lider}", sentimiento_lider)
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
            "categoria": categoria,
            "fecha_dato": ultima_fecha,
            "lat": lat,
            "lon": lon,
            "grupo": GRUPO_ACCIONES if es_accion else None,
            "variacion": round(serie[ultima_fecha] * (1 if es_sentimiento else 100), 2),
            "volatilidad": round(desvio_estandar(list(serie.values())) * (1 if es_sentimiento else 100), 2),
            "centralidad": round(centralidad[codigo], 2),
            "macro": macro.get(PAISES_MACRO.get(codigo)),
            "bandera": BANDERAS.get(codigo, "🇺🇸" if es_accion else None),
            "color_marca": COLOR_MARCA.get(codigo),
        })

    nodos_por_id = {n["id"]: n for n in nodos}
    for n in nodos:
        n["pronostico"] = calcular_pronostico(n["id"], nodos_por_id, aristas)

    datos = {"nodos": nodos, "pares": aristas}
    if len(nodos) < 2:
        raise RuntimeError("No hay suficientes series con datos para calcular correlaciones")

    guardar_snapshot_si_nuevo(datos)

    cache_correlaciones["hora"] = ahora
    cache_correlaciones["datos"] = datos
    return datos

def calcular_pronostico(id_nodo, nodos_por_id, pares):
    """Estimación de corto plazo a partir de los vecinos más correlacionados.
    Solo usa activos con retorno de precio: el puntaje de sentimiento (-1 a 1)
    no está en % y distorsionaría el promedio."""
    nodo = nodos_por_id.get(id_nodo)
    if not nodo or nodo.get("tipo") == "sentimiento":
        return None

    vecinos = []
    for p in pares:
        if p["a"] == id_nodo:
            otro_id = p["b"]
        elif p["b"] == id_nodo:
            otro_id = p["a"]
        else:
            continue
        otro = nodos_por_id.get(otro_id)
        if otro and otro.get("tipo") != "sentimiento" and otro.get("variacion") is not None:
            vecinos.append((otro, p["valor"]))

    vecinos.sort(key=lambda t: abs(t[1]), reverse=True)
    vecinos = vecinos[:6]
    if not vecinos:
        return {"movimiento": 0.0, "confianza": 0, "semaforo": "neutral", "factores": []}

    suma_ponderada = sum(c * o["variacion"] for o, c in vecinos)
    suma_pesos = sum(abs(c) for _, c in vecinos)
    movimiento = suma_ponderada / suma_pesos if suma_pesos else 0.0
    confianza = min(100, round(suma_pesos / len(vecinos) * 100))

    volatilidad = nodo.get("volatilidad") or 1
    if movimiento > volatilidad * 0.3:
        semaforo = "alcista"
    elif movimiento < -volatilidad * 0.3:
        semaforo = "bajista"
    else:
        semaforo = "neutral"

    return {
        "movimiento": round(movimiento, 3),
        "confianza": confianza,
        "semaforo": semaforo,
        "factores": [{"id": o["id"], "nombre": o["nombre"], "correlacion": c} for o, c in vecinos],
    }


COLUMNAS_FEATURE_DATASET = {
    "sentimiento": "SENTIMIENTO",
    "brent": "BZ=F",
    "usd": "DX-Y.NYB",
}

def construir_dataset_historico():
    snapshots = cargar_todos_los_snapshots()
    filas = []

    for previo, siguiente in zip(snapshots, snapshots[1:]):
        nodos_prev = {n["id"]: n for n in previo["nodos"]
                      if all(k in n for k in ("variacion", "tipo", "volatilidad", "fecha_dato"))}
        if not nodos_prev:
            continue
        nodos_sig = {n["id"]: n for n in siguiente["nodos"] if "variacion" in n and "fecha_dato" in n}

        # valores de contexto conocidos ESE día (antes de que ocurra el retorno futuro)
        contexto = {
            etiqueta: nodos_prev[codigo]["variacion"]
            for etiqueta, codigo in COLUMNAS_FEATURE_DATASET.items()
            if codigo in nodos_prev
        }

        for id_nodo, n_prev in nodos_prev.items():
            if n_prev["tipo"] == "sentimiento":
                continue   # el sentimiento es una feature de contexto, no un activo a predecir
            n_sig = nodos_sig.get(id_nodo)
            if not n_sig or n_sig["fecha_dato"] <= n_prev["fecha_dato"]:
                continue   # sin dato nuevo (fin de semana/feriado): no hay nada que comparar

            filas.append({
                "fecha": n_prev["fecha_dato"],
                "ticker": id_nodo,
                **contexto,
                "vol": n_prev["volatilidad"],
                "retorno_futuro": round(n_sig["variacion"] / 100, 5),
            })

    return filas

def evaluar_precision(dias=30):
    """Compara el pronóstico de cada snapshot con lo que pasó en el siguiente."""
    snapshots = cargar_todos_los_snapshots()
    limite = (date.today() - timedelta(days=dias)).isoformat()
    filas, evaluables = [], 0

    for previo, siguiente in zip(snapshots, snapshots[1:]):
        if siguiente.get("fecha", "") < limite:
            continue
        # los snapshots viejos no traen variacion/tipo/volatilidad: se saltan
        nodos_prev = {n["id"]: n for n in previo["nodos"]
                      if all(k in n for k in ("variacion", "tipo", "volatilidad"))}
        if not nodos_prev:
            continue
        evaluables += 1
        nodos_sig = {n["id"]: n for n in siguiente["nodos"] if "variacion" in n}

        for id_nodo, n_prev in nodos_prev.items():
            n_sig = nodos_sig.get(id_nodo)
            # sin dato nuevo (fin de semana, feriado): no hay nada que comparar
            if not n_sig or n_sig.get("fecha_dato", "") <= n_prev.get("fecha_dato", ""):
                continue
            pron = calcular_pronostico(id_nodo, nodos_prev, previo["pares"])
            if not pron or pron["semaforo"] == "neutral":
                continue
            real = n_sig["variacion"]
            acierto = real > 0 if pron["semaforo"] == "alcista" else real < 0
            filas.append({"id": id_nodo, "nombre": n_prev["nombre"], "acierto": acierto})

    total = len(filas)
    aciertos = sum(1 for f in filas if f["acierto"])
    por = {}
    for f in filas:
        d = por.setdefault(f["id"], {"id": f["id"], "nombre": f["nombre"], "n": 0, "aciertos": 0})
        d["n"] += 1
        d["aciertos"] += f["acierto"]
    por_activo = sorted(
        ({**d, "tasa": round(100 * d["aciertos"] / d["n"])} for d in por.values()),
        key=lambda d: d["n"], reverse=True,
    )
    return {
        "dias_evaluables": evaluables,
        "total": total,
        "aciertos": aciertos,
        "tasa": round(100 * aciertos / total) if total else None,
        "por_activo": por_activo,
    }

def crear_features(ticker, fecha):
    """
    Arma el vector de variables para un ticker en una fecha dada, usando
    únicamente los snapshots ya guardados en disco (no vuelve a consultar
    las APIs externas). Esto es importante: así el resultado es reproducible
    -pedís la misma fecha dos veces y te da lo mismo-, algo que necesitás
    si más adelante entrenás algo con estos datos.

    Devuelve None si ese día no hay snapshot, o si al ticker le falta el
    dato esencial (variación de precio) para esa fecha.
    """
    snapshots = sorted(cargar_todos_los_snapshots(), key=lambda s: s.get("fecha", ""))
    idx = next((i for i, s in enumerate(snapshots) if s.get("fecha") == fecha), None)
    if idx is None:
        return None

    def nodos_del_dia(i):
        return {n["id"]: n for n in snapshots[i].get("nodos", [])}

    nodos_hoy = nodos_del_dia(idx)
    nodo_ticker = nodos_hoy.get(ticker)
    if not nodo_ticker or nodo_ticker.get("variacion") is None:
        return None

    def variacion_de(id_nodo):
        n = nodos_hoy.get(id_nodo)
        return n.get("variacion") if n else None

    # retorno_5d: compuesto (no la suma simple) de los últimos 5 snapshots
    # disponibles para ese ticker, incluyendo el de "fecha"
    retornos = []
    for i in range(max(0, idx - 4), idx + 1):
        n = nodos_del_dia(i).get(ticker)
        if n and n.get("variacion") is not None:
            retornos.append(n["variacion"])
    retorno_5d = None
    if retornos:
        factor = 1.0
        for r in retornos:
            factor *= (1 + r / 100)
        retorno_5d = round((factor - 1) * 100, 3)

    # proxy de fortaleza del dólar: promedio de cómo se movieron las 4
    # monedas principales ese día (todas suben juntas cuando el dólar se
    # fortalece, porque "variacion" ya está expresada así, ver crearTarjetaMoneda en el frontend)
    monedas_mayores = ["EUR", "JPY", "GBP", "CNY"]
    valores_dolar = [v for m in monedas_mayores if (v := variacion_de(m)) is not None]
    usd = round(sum(valores_dolar) / len(valores_dolar), 3) if valores_dolar else None

    # sentimiento geopolítico: promedio del sentimiento de los 5 líderes ese día
    sentimiento_lideres = [v for l in LIDERES if (v := variacion_de(l)) is not None]
    sentimiento_geopolitico = (
        round(sum(sentimiento_lideres) / len(sentimiento_lideres), 4) if sentimiento_lideres else None
    )

    # indicador macro: el dato más reciente cacheado (es información que cambia
    # poco, así que usar el valor actual como aproximación para fechas pasadas
    # recientes es razonable; si necesitás precisión histórica real habría que
    # guardarlo también en cada snapshot)
    inflacion_usa = obtener_indicadores_macro().get("USA", {}).get("inflacion", {}).get("valor")

    return {
        "ticker": ticker,
        "fecha": fecha,
        "sentimiento_global": variacion_de("SENTIMIENTO"),
        "sentimiento_geopolitico": sentimiento_geopolitico,
        "retorno_1d": nodo_ticker.get("variacion"),
        "retorno_5d": retorno_5d,
        "volatilidad": nodo_ticker.get("volatilidad"),
        "brent": variacion_de("BZ=F"),
        "oro": variacion_de("GC=F"),
        "usd": usd,
        "tasa_10y": variacion_de("^TNX"),
        "inflacion_usa": inflacion_usa,
    }

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

def construir_dataset_features():
    """
    Arma una fila por (ticker, día) usando crear_features(), con el retorno
    del día siguiente como etiqueta a predecir (retorno_futuro). A diferencia
    de construir_dataset_pronostico(), reutiliza exactamente la misma función
    que usás para consultar features sueltas, así que no hay dos lugares
    calculando lo mismo de formas distintas.
    """
    snapshots = sorted(cargar_todos_los_snapshots(), key=lambda s: s.get("fecha", ""))
    filas = []

    for i in range(len(snapshots) - 1):   # el último día no tiene "mañana" para la etiqueta
        fecha = snapshots[i].get("fecha")
        nodos_siguiente = {n["id"]: n for n in snapshots[i + 1].get("nodos", [])}

        for ticker in ACCIONES:
            features = crear_features(ticker, fecha)
            if not features:
                continue   # a ese ticker le faltaba algo esencial ese día

            nodo_siguiente = nodos_siguiente.get(ticker)
            if not nodo_siguiente or nodo_siguiente.get("variacion") is None:
                continue   # sin dato del día siguiente, no hay etiqueta que ponerle

            features["retorno_futuro"] = nodo_siguiente["variacion"]
            filas.append(features)

    return filas

FEATURES_MODELO = [
    "sentimiento_global", "sentimiento_geopolitico", "retorno_1d",
    "retorno_5d", "volatilidad", "brent", "oro", "usd", "tasa_10y", "inflacion_usa",
]
cache_modelo = {"hora": 0, "modelo": None, "n_filas": 0}
DURACION_MODELO = 3600 * 6   # reentrena cada 6 horas (o antes, si hay más datos nuevos)


def entrenar_modelo():
    """Entrena (o reutiliza) un RandomForest sobre el histórico de snapshots.
    Devuelve (modelo, filas_usadas). modelo es None si todavía no hay datos
    suficientes para que el entrenamiento tenga sentido."""
    ahora = time()
    filas = construir_dataset_features()
    filas_validas = [
        f for f in filas
        if all(f.get(c) is not None for c in FEATURES_MODELO) and f.get("retorno_futuro") is not None
    ]

    si_cacheado = (
        cache_modelo["modelo"]
        and ahora - cache_modelo["hora"] < DURACION_MODELO
        and cache_modelo["n_filas"] == len(filas_validas)
    )
    if si_cacheado:
        return cache_modelo["modelo"], filas_validas

    if len(filas_validas) < 20:
        return None, filas_validas

    X = [[f[c] for c in FEATURES_MODELO] for f in filas_validas]
    y = [f["retorno_futuro"] for f in filas_validas]

    modelo = RandomForestRegressor(n_estimators=200, max_depth=4, random_state=0)
    modelo.fit(X, y)

    cache_modelo.update({"hora": ahora, "modelo": modelo, "n_filas": len(filas_validas)})
    return modelo, filas_validas


def predecir_ml(ticker):
    modelo, filas_validas = entrenar_modelo()
    if modelo is None:
        return None

    snapshots = cargar_todos_los_snapshots()
    if not snapshots:
        return None
    fecha_mas_reciente = max(s.get("fecha", "") for s in snapshots)

    features = crear_features(ticker, fecha_mas_reciente)
    if not features or any(features.get(c) is None for c in FEATURES_MODELO):
        return None

    X = [[features[c] for c in FEATURES_MODELO]]
    prediccion = modelo.predict(X)[0]

    importancias = sorted(
        zip(FEATURES_MODELO, modelo.feature_importances_),
        key=lambda t: t[1], reverse=True,
    )[:5]

    return {
        "ticker": ticker,
        "fecha_base": fecha_mas_reciente,
        "retorno_estimado": round(float(prediccion), 3),
        "filas_entrenamiento": len(filas_validas),
        "factores_principales": [
            {"variable": v, "importancia": round(float(i), 3)} for v, i in importancias
        ],
    }
def construir_dataset_pronostico():
    snapshots = cargar_todos_los_snapshots()

    if len(snapshots) < 2:
        return []

    dataset = []

    for i in range(len(snapshots) - 1):

        actual = snapshots[i]
        siguiente = snapshots[i + 1]

        nodos_actual = {
            n["id"]: n
            for n in actual.get("nodos", [])
        }

        nodos_siguiente = {
            n["id"]: n
            for n in siguiente.get("nodos", [])
        }

        nodo_sentimiento = nodos_actual.get("SENTIMIENTO")

        sentimiento = None

        if nodo_sentimiento:
            sentimiento = nodo_sentimiento.get("variacion")

        for ticker in ACCIONES:

            nodo_actual = nodos_actual.get(ticker)
            nodo_siguiente = nodos_siguiente.get(ticker)

            if not nodo_actual or not nodo_siguiente:
                continue

            features = crear_features(
                ticker,
                actual.get("fecha")
            )

            if not features:
                continue

            retorno_actual = nodo_actual.get("variacion")
            volatilidad = nodo_actual.get("volatilidad")
            retorno_futuro = nodo_siguiente.get("variacion")

            # No utilizar datos incompletos
            if features["retorno_1d"] is None:
                continue

            if features["volatilidad"] is None:
                continue

            if retorno_futuro is None:
                continue

            dataset.append({
                "fecha": actual.get("fecha"),
                "ticker": ticker,
                "sentimiento_global": features["sentimiento_global"],
                "retorno_1d": features["retorno_1d"],
                "volatilidad": features["volatilidad"],
                "brent": features["brent"],
                "oro": features["oro"],
                "tasa_10y": features["tasa_10y"],
                "retorno_futuro": retorno_futuro
            })

    return dataset

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
        "nodos": [
            {
                "id": n["id"], "nombre": n["nombre"], "categoria": n["categoria"],
                "tipo": n["tipo"], "variacion": n["variacion"],
                "volatilidad": n["volatilidad"], "fecha_dato": n["fecha_dato"],
            }
            for n in datos["nodos"]
        ],
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

@app.route("/api/dataset-features")
def dataset_features():
    try:
        filas = construir_dataset_features()
        return jsonify(total_filas=len(filas), filas=filas)
    except Exception as error:
        traceback.print_exc()
        return jsonify(error=f"{type(error).__name__}: {error}"), 502
@app.route("/api/dataset-pronostico")
def dataset_pronostico():
    try:
        diagnostico = construir_dataset_pronostico()

        return jsonify({
            "total_registros": len(diagnostico),
            "diagnostico": diagnostico
        })

    except Exception as error:
        traceback.print_exc()

        return jsonify({
            "error": f"{type(error).__name__}: {error}"
        }), 500
    
@app.route("/api/debug-snapshot")
def debug_snapshot():
    snapshots = cargar_todos_los_snapshots()

    if not snapshots:
        return jsonify({
            "error": "No hay snapshots"
        })

    ultimo = snapshots[-1]

    return jsonify({
        "fecha": ultimo.get("fecha"),
        "cantidad_nodos": len(ultimo.get("nodos", [])),
        "nodos": ultimo.get("nodos", [])
    })

@app.route("/api/debug-datos-modelo")
def debug_datos_modelo():
    snapshots = cargar_todos_los_snapshots()

    if not snapshots:
        return jsonify({
            "error": "No hay snapshots"
        })

    ultimo = snapshots[-1]

    nodos = {
        n["id"]: n
        for n in ultimo.get("nodos", [])
    }

    resultado = {}

    for ticker in ACCIONES:
        nodo = nodos.get(ticker)

        if not nodo:
            continue

        resultado[ticker] = {
            "variacion": nodo.get("variacion"),
            "volatilidad": nodo.get("volatilidad"),
            "fecha_dato": nodo.get("fecha_dato")
        }

    sentimiento = nodos.get("SENTIMIENTO")

    resultado["SENTIMIENTO"] = {
        "variacion": (
            sentimiento.get("variacion")
            if sentimiento
            else None
        ),
        "fecha_dato": (
            sentimiento.get("fecha_dato")
            if sentimiento
            else None
        )
    }

    return jsonify({
        "fecha_snapshot": ultimo.get("fecha"),
        "datos": resultado
    })

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

@app.route("/api/features/<ticker>/<fecha>")
def features_de_ticker(ticker, fecha):
    resultado = crear_features(ticker, fecha)
    if resultado is None:
        return jsonify(error=f"Sin datos para {ticker} en {fecha}"), 404
    return jsonify(resultado)

@app.route("/api/pronostico-ml/<ticker>")
def pronostico_ml(ticker):
    try:
        resultado = predecir_ml(ticker.upper())
        if resultado is None:
            return jsonify(error="Todavía no hay suficiente historial para entrenar el modelo (hacen falta más días)"), 503
        return jsonify(resultado)
    except Exception as error:
        traceback.print_exc()
        return jsonify(error=f"{type(error).__name__}: {error}"), 502

@app.route("/api/precision")
def precision():
    try:
        return jsonify(evaluar_precision())
    except Exception as error:
        traceback.print_exc()
        return jsonify(error=f"{type(error).__name__}: {error}"), 502

@app.route("/admin/dataset.csv")
@requiere_admin
def dataset_csv():
    filas = construir_dataset_historico()
    if not filas:
        return jsonify(error="Todavía no hay suficiente historial de snapshots"), 404

    columnas = ["fecha", "ticker", "sentimiento", "brent", "usd", "vol", "retorno_futuro"]
    buffer = io.StringIO()
    escritor = csv.DictWriter(buffer, fieldnames=columnas, extrasaction="ignore")
    escritor.writeheader()
    escritor.writerows(filas)

    return Response(
        buffer.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=dataset_historico.csv"},
    )


# ---------- tarea autónoma: recalcula y guarda el snapshot sin que nadie visite ----------
INTERVALO_TAREA_HORAS = 6
DEBUG = os.environ.get("FLASK_DEBUG") == "1"


def tarea_autonoma():
    try:
        cache_correlaciones["hora"] = 0      # fuerza a recalcular con datos frescos
        obtener_correlaciones()              # también guarda el snapshot del día
        print("Tarea autónoma OK:", datetime.now().isoformat(timespec="seconds"))
    except Exception as error:
        print("Tarea autónoma falló:", type(error).__name__, error)


def iniciar_planificador():
    def bucle():
        time_module_sleep(20)                # deja que el servidor termine de arrancar
        while True:
            tarea_autonoma()
            time_module_sleep(INTERVALO_TAREA_HORAS * 3600)
    threading.Thread(target=bucle, daemon=True).start()

@app.route("/api/analizar-noticia-local", methods=["POST"])
def analizar_noticia_local():
    datos = request.get_json(silent=True) or {}
    texto = (datos.get("texto") or "").strip()
    if not texto:
        return jsonify(error="Falta el campo 'texto'"), 400

    def calcular_embedding():
        entradas = tokenizador_sentimiento(
            texto, truncation=True, max_length=128, return_tensors="pt"
        ).to(DISPOSITIVO_TORCH)
        with torch.no_grad():
            salida_base = modelo_sentimiento.bert(**entradas)
        return salida_base.pooler_output.squeeze().tolist()

    def calcular_sentimiento_uno():
        return calcular_sentimiento_ia([texto])

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futuro_embedding = pool.submit(calcular_embedding)
            futuro_sentimiento = pool.submit(calcular_sentimiento_uno)
            embedding = futuro_embedding.result()
            sentimiento = futuro_sentimiento.result()
    except Exception as error:
        traceback.print_exc()
        return jsonify(error=f"{type(error).__name__}: {error}"), 502
        
    return jsonify(
        texto=texto,
        sentimiento=sentimiento,
        embedding=embedding,
        dimension=len(embedding)
    )
   
if __name__ == "__main__":

        # con debug=True Flask arranca dos procesos; solo el hijo debe correr la tarea

        if not DEBUG or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
             iniciar_planificador()

        app.run(host="0.0.0.0", port=5000, debug=DEBUG)
