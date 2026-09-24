import os
import re
import traceback
import xml.etree.ElementTree as ET
import secrets
import json
import html
import base64
import math
from pathlib import Path
from functools import wraps
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from time import time
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


def leer_historial_accion(simbolo):
    respuesta = requests.get(
        YAHOO_GRAFICO + simbolo,
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


def obtener_correlaciones():
    ahora = time()
    if cache_correlaciones["datos"] and ahora - cache_correlaciones["hora"] < DURACION_CORRELACIONES:
        return cache_correlaciones["datos"]

    tendencias = obtener_tendencias("1m")
    series = {
        codigo: dict(zip(tendencias["fechas"], moneda["valores"]))
        for codigo, moneda in tendencias["monedas"].items()
    }
    for simbolo, historial in obtener_historiales_acciones().items():
        if historial:
            series[simbolo] = historial

    fechas_comunes = None
    for historial in series.values():
        claves = set(historial.keys())
        fechas_comunes = claves if fechas_comunes is None else (fechas_comunes & claves)
    fechas_comunes = sorted(fechas_comunes) if fechas_comunes else []

    if len(fechas_comunes) < 5:
        raise RuntimeError("No hay suficientes datos históricos en común")

    retornos = {
        nodo: [
            (historial[fechas_comunes[i]] - historial[fechas_comunes[i - 1]]) / historial[fechas_comunes[i - 1]]
            for i in range(1, len(fechas_comunes))
        ]
        for nodo, historial in series.items()
    }

    nodos = list(retornos.keys())
    pares = []
    for i, a in enumerate(nodos):
        for b in nodos[i + 1:]:
            pares.append({"a": a, "b": b, "valor": round(pearson(retornos[a], retornos[b]), 3)})

    datos = {
        "nodos": [
            {"id": n, "tipo": "moneda" if n in MONEDAS else "accion", "nombre": MONEDAS.get(n) or ACCIONES.get(n, n)}
            for n in nodos
        ],
        "pares": pares,
        "dias": len(fechas_comunes),
    }

    cache_correlaciones["hora"] = ahora
    cache_correlaciones["datos"] = datos
    return datos

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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)