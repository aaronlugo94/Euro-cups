"""
daily_renpho.py — V6.0 Production Grade
Cambios vs V5.1 (Gemini):
  1. INSERT OR IGNORE + rowcount  → atomicidad real sin race condition
  2. Context managers (with)       → conexiones SQLite garantizadas cerradas
  3. analizar_con_ia con fallback  → nunca retorna None
  4. Prompt monolingüe (español)   → inferencia más predecible
  5. logging con exc_info          → trazas completas en cada error
  6. Validación de None pre-delta  → no más TypeError en f-strings
"""

import os
import sqlite3
import requests
import pytz
import time
import logging
from datetime import datetime
from google import genai
from renpho import RenphoClient

# ─── CONFIG ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

TZ      = pytz.timezone(os.getenv("TZ", "America/Phoenix"))
DRY_RUN = os.getenv("DRY_RUN", "0") == "1"
DB_PATH = "/app/data/mis_datos_renpho.db"

REQUIRED_VARS = [
    "RENPHO_EMAIL", "RENPHO_PASSWORD",
    "GOOGLE_API_KEY",
    "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID",
]
env_vars = {var: os.getenv(var) for var in REQUIRED_VARS}
faltantes = [v for v, k in env_vars.items() if not k]
if faltantes:
    raise RuntimeError(f"Faltan variables de entorno: {', '.join(faltantes)}")


# ─── BASE DE DATOS ─────────────────────────────────────────────────────────────

def inicializar_db():
    """Crea la tabla y aplica migraciones en caliente si hacen falta."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS pesajes (
                Fecha          TEXT PRIMARY KEY,
                Peso_kg        REAL,
                Grasa_Porcentaje REAL,
                Agua           REAL,
                Musculo        REAL,
                BMR            INTEGER,
                VisFat         REAL,
                BMI            REAL,
                EdadMetabolica INTEGER,
                FatFreeWeight  REAL,
                Timestamp      INTEGER UNIQUE   -- UNIQUE es la clave anti race-condition
            )
        """)
        # Migración en caliente: añade Timestamp si la tabla ya existía sin ella
        columnas = {row[1] for row in conn.execute("PRAGMA table_info(pesajes)")}
        if "Timestamp" not in columnas:
            conn.execute("ALTER TABLE pesajes ADD COLUMN Timestamp INTEGER UNIQUE")
            logging.info("Migración aplicada: columna Timestamp añadida.")
        # Índices para queries eficientes con 500+ registros
        conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON pesajes (Timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_fecha ON pesajes (Fecha)")
        conn.commit()


def guardar_si_es_nuevo(m: dict) -> bool:
    """
    Intenta insertar el pesaje usando INSERT OR IGNORE.
    Retorna True si se insertó (es nuevo), False si ya existía.

    ATOMICIDAD REAL: SQLite gestiona el UNIQUE constraint en Timestamp.
    No hay SELECT previo, no hay ventana de race condition.
    """
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute("""
            INSERT OR IGNORE INTO pesajes
            (Fecha, Peso_kg, Grasa_Porcentaje, Agua, Musculo,
             BMR, VisFat, BMI, EdadMetabolica, FatFreeWeight, Timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            m["fecha_str"], m["peso"], m["grasa"], m["agua"], m["musculo_pct"],
            m["bmr"], m["grasa_visceral"], m["bmi"], m["edad_metabolica"],
            m["fat_free_weight"], m["time_stamp"],
        ))
        conn.commit()
        insertado = cur.rowcount == 1  # 0 = ya existía (IGNORE), 1 = nuevo

    if insertado:
        logging.info("💾 Pesaje persistido en SQLite.")
    else:
        logging.info("💤 Timestamp ya existente en BD. Pesaje duplicado ignorado.")

    return insertado


def obtener_comparativa_semana(fecha_actual_str: str) -> dict | None:
    """Busca el pesaje más cercano a hace 7 días (mínimo 6 días atrás)."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute("""
                SELECT Peso_kg, Grasa_Porcentaje, Musculo, Agua
                FROM pesajes
                WHERE Fecha <= date(?, '-6 day')
                ORDER BY Fecha DESC
                LIMIT 1
            """, (fecha_actual_str,)).fetchone()

        if row:
            return {"peso": row[0], "grasa": row[1], "musculo_pct": row[2], "agua": row[3]}

    except Exception:
        logging.warning("No se pudo obtener comparativa semanal.", exc_info=True)

    return None


# ─── EXTRACCIÓN RENPHO ─────────────────────────────────────────────────────────

def obtener_datos_renpho() -> dict:
    logging.info("🔄 Extrayendo telemetría de Renpho...")
    try:
        cliente  = RenphoClient(env_vars["RENPHO_EMAIL"], env_vars["RENPHO_PASSWORD"])
        mediciones = None

        try:
            mediciones = cliente.get_all_measurements()
        except Exception as e:
            logging.warning(f"get_all_measurements falló: {e}. Intentando fallback por MAC...")

        if not mediciones:
            devices = cliente.get_device_info()
            if not devices:
                raise ValueError("No hay dispositivos vinculados a la cuenta Renpho.")
            mac = devices[0].get("mac")
            if not mac:
                raise ValueError("El dispositivo no tiene dirección MAC.")
            mediciones = cliente.get_measurements(
                table_name=mac, user_id=cliente.user_id, total_count=10
            )

        if not mediciones:
            raise ValueError("La API de Renpho devolvió lista vacía de mediciones.")

        u = max(mediciones, key=lambda x: x.get("time_stamp", 0))

        ts = u.get("time_stamp")
        if not ts:
            raise ValueError("La medición no tiene timestamp válido.")

        datos = {
            "time_stamp":      ts,
            "fecha_str":       datetime.fromtimestamp(ts, TZ).strftime("%Y-%m-%d"),
            "peso":            u.get("weight"),
            "grasa":           u.get("bodyfat"),
            "agua":            u.get("water"),
            "bmi":             u.get("bmi"),
            "bmr":             u.get("bmr"),
            "edad_metabolica": u.get("bodyage"),
            "grasa_visceral":  u.get("visfat"),
            "masa_muscular_kg":u.get("sinew"),
            "musculo_pct":     u.get("muscle"),
            "fat_free_weight": u.get("fatFreeWeight"),
            "proteina":        u.get("protein"),
            "masa_osea":       u.get("bone"),
        }

        # Validación estricta de campos críticos para los cálculos de delta
        campos_criticos = ["peso", "grasa", "musculo_pct", "agua"]
        nulos = [c for c in campos_criticos if datos.get(c) is None]
        if nulos:
            raise ValueError(f"Datos corruptos/faltantes de la báscula: {nulos}")

        return datos

    except Exception:
        logging.error("Error crítico extrayendo datos de Renpho.", exc_info=True)
        raise


# ─── ANÁLISIS IA ───────────────────────────────────────────────────────────────

def analizar_con_ia(m: dict, semana_pasada: dict | None) -> str:
    """
    Genera análisis clínico con Gemini.
    GARANTÍA: siempre retorna un string, nunca None.
    """
    logging.info("🧠 Generando análisis clínico con IA...")
    client = genai.Client(api_key=env_vars["GOOGLE_API_KEY"])

    comparativa = ""
    if semana_pasada:
        comparativa = (
            "\n--- COMPARATIVA VS HACE 7 DÍAS ---\n"
            f"Peso:    {semana_pasada['peso']} kg  →  {m['peso']} kg  "
            f"(variación: {m['peso'] - semana_pasada['peso']:+.2f} kg)\n"
            f"Grasa:   {semana_pasada['grasa']}%  →  {m['grasa']}%\n"
            f"Músculo: {semana_pasada['musculo_pct']}%  →  {m['musculo_pct']}%\n"
            f"Agua:    {semana_pasada['agua']}%  →  {m['agua']}%\n"
        )

    # Prompt monolingüe en español, estructurado y sin ambigüedad
    prompt = f"""Eres un experto en recomposición corporal. Analiza las siguientes métricas y responde ÚNICAMENTE en el formato HTML indicado.

MÉTRICAS DE HOY:
- Peso: {m['peso']} kg  |  BMI: {m['bmi']}
- Músculo esquelético: {m['musculo_pct']}%
- Grasa corporal: {m['grasa']}%  |  Visceral: {m['grasa_visceral']}
- Agua: {m['agua']}%  |  Proteína: {m['proteina']}%
- Edad metabólica: {m['edad_metabolica']} años
{comparativa}

INSTRUCCIÓN: ¿Vamos en la dirección correcta? Evalúa la tendencia de 7 días y responde SOLO con este bloque, sin texto adicional:

<b>📊 Análisis de la Semana:</b> [evaluación de la tendencia]

<b>🎯 Acción del Día:</b> [recomendación concreta de nutrición o entrenamiento]

<i>🔥 Foco: [una frase motivadora]</i>

REGLA: Usa ÚNICAMENTE las etiquetas <b> e <i>. Está PROHIBIDO usar <br>, <hr>, <ul>, <li>, <h1>, <h2>, <h3> o cualquier otra etiqueta HTML."""

    for intento in range(3):
        try:
            respuesta = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
            texto = respuesta.text.strip() if respuesta and respuesta.text else ""
            if texto:
                return texto
            logging.warning(f"Intento {intento + 1}: respuesta vacía de Gemini.")
        except Exception as e:
            logging.warning(f"Intento {intento + 1} fallido: {e}")
            if intento < 2:
                time.sleep(2)

    # Fallback garantizado — nunca retorna None
    logging.error("Gemini falló tras 3 intentos. Usando fallback de análisis.")
    return "<i>⚠️ Análisis de IA no disponible temporalmente. Revisa los logs.</i>"


# ─── TELEGRAM ─────────────────────────────────────────────────────────────────

_HTML_SANITIZE = [
    ("<br>", "\n"), ("<br/>", "\n"), ("<br />", "\n"),
    ("<ul>", ""), ("</ul>", ""), ("<li>", "• "), ("</li>", "\n"),
    ("<hr>", "---"), ("<hr/>", "---"),
    ("<p>", ""), ("</p>", "\n"),
    ("<strong>", "<b>"), ("</strong>", "</b>"),
    ("<h1>", ""), ("</h1>", "\n"),
    ("<h2>", ""), ("</h2>", "\n"),
    ("<h3>", ""), ("</h3>", "\n"),
]

def enviar_telegram(mensaje: str) -> None:
    if DRY_RUN:
        logging.info(f"[DRY RUN] Mensaje Telegram:\n{mensaje}")
        return

    for old, new in _HTML_SANITIZE:
        mensaje = mensaje.replace(old, new)

    url     = f"https://api.telegram.org/bot{env_vars['TELEGRAM_BOT_TOKEN']}/sendMessage"
    payload = {"chat_id": env_vars["TELEGRAM_CHAT_ID"], "text": mensaje, "parse_mode": "HTML"}

    res = requests.post(url, json=payload, timeout=10)
    if res.status_code == 200:
        return

    logging.warning(f"Telegram rechazó HTML ({res.status_code}): {res.text}. Reintentando en texto plano...")
    payload.pop("parse_mode")
    res2 = requests.post(url, json=payload, timeout=10)
    if res2.status_code != 200:
        logging.error(f"Error crítico enviando a Telegram: {res2.text}")


# ─── UTILIDADES ───────────────────────────────────────────────────────────────

def calcular_delta(hoy: float, ayer: float | None, invert_colors: bool = False) -> str:
    """Retorna string vacío si ayer es None — evita TypeError en f-strings."""
    if ayer is None:
        return ""
    diff = hoy - ayer
    if abs(diff) < 0.05:
        return " ⚪"
    emoji = ("🟢" if diff < 0 else "🔴") if invert_colors else ("🟢" if diff > 0 else "🔴")
    return f" (Δ {diff:+.1f} {emoji})"


# ─── FLUJO PRINCIPAL ──────────────────────────────────────────────────────────

def ejecutar_diario() -> bool:
    try:
        inicializar_db()
        m = obtener_datos_renpho()

        # Atomicidad real: INSERT OR IGNORE decide si es nuevo sin race condition
        if not guardar_si_es_nuevo(m):
            return True  # Silencioso: no bloquea job_dieta.py los domingos

        logging.info("🚀 Nuevo pesaje detectado. Generando reporte...")
        semana_pasada = obtener_comparativa_semana(m["fecha_str"])

        # Los None ya están validados en obtener_datos_renpho, pero calcular_delta
        # también los maneja por si acaso (defensa en profundidad)
        d_peso  = calcular_delta(m["peso"],        semana_pasada["peso"] if semana_pasada else None, invert_colors=True)
        d_grasa = calcular_delta(m["grasa"],       semana_pasada["grasa"] if semana_pasada else None, invert_colors=True)
        d_musc  = calcular_delta(m["musculo_pct"], semana_pasada["musculo_pct"] if semana_pasada else None)
        d_agua  = calcular_delta(m["agua"],        semana_pasada["agua"] if semana_pasada else None)

        analisis = analizar_con_ia(m, semana_pasada)

        reporte = (
            f"📊 <b>REPORTE DE SALUD (VS HACE 7 DÍAS)</b>\n\n"
            f"⚖️ <b>Peso:</b> {m['peso']} kg{d_peso}\n"
            f"💪 <b>Músculo Esquelético:</b> {m['musculo_pct']}%{d_musc}\n"
            f"🥓 <b>Grasa:</b> {m['grasa']}%{d_grasa} (Visceral: {m['grasa_visceral']})\n"
            f"💧 <b>Agua:</b> {m['agua']}%{d_agua}\n"
            f"📅 <b>Edad Metabólica:</b> {m['edad_metabolica']} años\n\n"
            f"🤖 <b>Análisis IA:</b>\n{analisis}"
        )

        enviar_telegram(reporte)
        logging.info("✅ Flujo de pesaje completado y notificado.")
        return True

    except Exception:
        logging.error("🔴 Error crítico en el flujo diario.", exc_info=True)
        enviar_telegram("🔴 <b>Error Crítico — Sistema Renpho:</b> Revisa los logs en Railway.")
        return False


if __name__ == "__main__":
    ejecutar_diario()
