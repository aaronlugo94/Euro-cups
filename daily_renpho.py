"""
daily_renpho.py — V6.0 Production Grade (Análisis Diario)
Diferencias clave vs script original:
  - SQLite como única fuente de verdad (adiós metrics.json)
  - Idempotencia por Timestamp real de báscula (no fecha sistema)
  - Comparativa vs pesaje anterior REAL en BD (no "ayer" por calendario)
  - Todos los datos de Renpho aprovechados: score de composición corporal,
    alertas clínicas, tendencia de 7 días, clasificaciones por rangos
  - analizar_con_ia nunca retorna None
  - Context managers en todas las conexiones SQLite
  - Prompt monolingüe con contexto rico y estructurado
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


# ─── RANGOS CLÍNICOS DE REFERENCIA ────────────────────────────────────────────
# Usados para clasificar cada métrica y generar alertas automáticas.
# Ajusta los rangos según tu perfil (edad, sexo, objetivo).

RANGOS = {
    "bmi":            {"optimo": (18.5, 24.9), "alerta": (25.0, 29.9), "critico": (30.0, 99)},
    "grasa_hombre":   {"optimo": (10.0, 20.0), "alerta": (20.1, 25.0), "critico": (25.1, 100)},
    "visceral":       {"optimo": (1,    9),    "alerta": (10,   14),   "critico": (15,   30)},
    "agua":           {"optimo": (50.0, 65.0), "alerta": (45.0, 49.9), "critico": (0,    44.9)},
    "proteina":       {"optimo": (16.0, 20.0), "alerta": (14.0, 15.9), "critico": (0,    13.9)},
}

def clasificar(valor, metrica):
    """Retorna emoji de semáforo según rangos clínicos."""
    if valor is None or metrica not in RANGOS:
        return ""
    r = RANGOS[metrica]
    if r["optimo"][0] <= valor <= r["optimo"][1]:
        return " 🟢"
    elif r["alerta"][0] <= valor <= r["alerta"][1]:
        return " 🟡"
    elif r["critico"][0] <= valor <= r["critico"][1]:
        return " 🔴"
    return ""


# ─── BASE DE DATOS ─────────────────────────────────────────────────────────────

def inicializar_db():
    """Crea tabla completa con TODOS los campos de Renpho + índices."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS pesajes (
                Fecha             TEXT PRIMARY KEY,
                Timestamp         INTEGER UNIQUE,
                Peso_kg           REAL,
                Grasa_Porcentaje  REAL,
                Agua              REAL,
                Musculo_Pct       REAL,
                Musculo_kg        REAL,
                BMR               INTEGER,
                VisFat            REAL,
                BMI               REAL,
                EdadMetabolica    INTEGER,
                FatFreeWeight     REAL,
                Proteina          REAL,
                MasaOsea          REAL
            )
        """)
        # Migración en caliente para BDs existentes con esquema viejo
        # CRÍTICO: ALTER TABLE no soporta UNIQUE — la unicidad la maneja
        # el CREATE UNIQUE INDEX de abajo, que sí es compatible con migraciones
        columnas = {row[1] for row in conn.execute("PRAGMA table_info(pesajes)")}
        migraciones = {
            "Timestamp":     "INTEGER",  # Sin UNIQUE aquí
            "Musculo_kg":    "REAL",
            "FatFreeWeight": "REAL",
            "Proteina":      "REAL",
            "MasaOsea":      "REAL",
        }
        for col, tipo in migraciones.items():
            if col not in columnas:
                conn.execute(f"ALTER TABLE pesajes ADD COLUMN {col} {tipo}")
                logging.info(f"Migración aplicada: columna {col} añadida.")

        # UNIQUE INDEX — equivalente al constraint pero compatible con ALTER TABLE
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_timestamp ON pesajes (Timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_fecha ON pesajes (Fecha)")
        conn.commit()


def guardar_si_es_nuevo(m: dict) -> bool:
    """
    INSERT OR IGNORE por Timestamp único.
    Retorna True si insertó (nuevo), False si ya existía.
    Atomicidad real: SQLite gestiona el UNIQUE constraint.
    """
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute("""
            INSERT OR IGNORE INTO pesajes
            (Fecha, Timestamp, Peso_kg, Grasa_Porcentaje, Agua, Musculo_Pct,
             Musculo_kg, BMR, VisFat, BMI, EdadMetabolica, FatFreeWeight, Proteina, MasaOsea)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            m["fecha_str"], m["time_stamp"],
            m["peso"], m["grasa"], m["agua"], m["musculo_pct"],
            m["masa_muscular_kg"], m["bmr"], m["grasa_visceral"], m["bmi"],
            m["edad_metabolica"], m["fat_free_weight"], m["proteina"], m["masa_osea"],
        ))
        conn.commit()
        insertado = cur.rowcount == 1

    if insertado:
        logging.info("💾 Pesaje persistido en SQLite.")
    else:
        logging.info("💤 Timestamp ya existe. Pesaje duplicado ignorado.")
    return insertado


def obtener_pesaje_anterior(fecha_actual_str: str) -> dict | None:
    """
    Retorna el pesaje inmediatamente anterior al de hoy.
    No asume 'ayer' — funciona aunque hayas saltado días.
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute("""
                SELECT Peso_kg, Grasa_Porcentaje, Musculo_Pct, Musculo_kg, Agua,
                       BMI, VisFat, Proteina, EdadMetabolica, Fecha
                FROM pesajes
                WHERE Fecha < ?
                ORDER BY Fecha DESC
                LIMIT 1
            """, (fecha_actual_str,)).fetchone()
        if row:
            return {
                "peso": row[0], "grasa": row[1], "musculo_pct": row[2],
                "masa_muscular_kg": row[3], "agua": row[4], "bmi": row[5],
                "grasa_visceral": row[6], "proteina": row[7],
                "edad_metabolica": row[8], "fecha": row[9],
            }
    except Exception:
        logging.warning("No se pudo obtener pesaje anterior.", exc_info=True)
    return None


def obtener_tendencia_7_dias(fecha_actual_str: str) -> dict | None:
    """
    Extrae métricas clave de los últimos 7 días para calcular
    tendencias reales (no solo delta puntual).
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute("""
                SELECT Fecha, Peso_kg, Grasa_Porcentaje, Musculo_Pct, Agua
                FROM pesajes
                WHERE Fecha <= ? AND Fecha >= date(?, '-7 day')
                ORDER BY Fecha ASC
            """, (fecha_actual_str, fecha_actual_str)).fetchall()

        if len(rows) < 2:
            return None

        # Calculamos tendencia lineal simple (pendiente)
        n = len(rows)
        pesos = [r[1] for r in rows if r[1]]
        grasas = [r[2] for r in rows if r[2]]

        def pendiente(serie):
            if len(serie) < 2:
                return 0
            x_mean = (len(serie) - 1) / 2
            y_mean = sum(serie) / len(serie)
            num = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(serie))
            den = sum((i - x_mean) ** 2 for i in range(len(serie)))
            return round(num / den, 4) if den else 0

        return {
            "dias_medidos": n,
            "peso_tendencia_dia": pendiente(pesos),    # kg/día (negativo = bajando)
            "grasa_tendencia_dia": pendiente(grasas),  # %/día
            "peso_min": min(pesos) if pesos else None,
            "peso_max": max(pesos) if pesos else None,
        }
    except Exception:
        logging.warning("No se pudo calcular tendencia 7 días.", exc_info=True)
    return None


# ─── EXTRACCIÓN RENPHO ─────────────────────────────────────────────────────────

def obtener_datos_renpho() -> dict:
    logging.info("🔄 Extrayendo telemetría de Renpho...")
    try:
        cliente    = RenphoClient(env_vars["RENPHO_EMAIL"], env_vars["RENPHO_PASSWORD"])
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
            raise ValueError("La API de Renpho devolvió lista vacía.")

        u = max(mediciones, key=lambda x: x.get("time_stamp", 0))

        ts = u.get("time_stamp")
        if not ts:
            raise ValueError("La medición no tiene timestamp válido.")

        datos = {
            "time_stamp":       ts,
            "fecha_str":        datetime.fromtimestamp(ts, TZ).strftime("%Y-%m-%d"),
            "peso":             u.get("weight"),
            "grasa":            u.get("bodyfat"),
            "agua":             u.get("water"),
            "bmi":              u.get("bmi"),
            "bmr":              u.get("bmr"),
            "edad_metabolica":  u.get("bodyage"),
            "grasa_visceral":   u.get("visfat"),
            "masa_muscular_kg": u.get("sinew"),
            "musculo_pct":      u.get("muscle"),
            "fat_free_weight":  u.get("fatFreeWeight"),
            "proteina":         u.get("protein"),
            "masa_osea":        u.get("bone"),
        }

        campos_criticos = ["peso", "grasa", "musculo_pct", "agua"]
        nulos = [c for c in campos_criticos if datos.get(c) is None]
        if nulos:
            raise ValueError(f"Datos críticos faltantes de la báscula: {nulos}")

        return datos

    except Exception:
        logging.error("Error crítico extrayendo datos de Renpho.", exc_info=True)
        raise


# ─── SCORE DE COMPOSICIÓN CORPORAL ────────────────────────────────────────────

def calcular_score_composicion(m: dict) -> tuple[int, str]:
    """
    Score de 0-100 basado en 4 métricas clave ponderadas.
    Retorna (score, descripción).
    Ajusta los pesos y rangos óptimos a tu objetivo personal.
    """
    score = 0

    # Grasa corporal (35 pts) — objetivo: <18%
    grasa = m.get("grasa", 99)
    if grasa <= 15:    score += 35
    elif grasa <= 18:  score += 28
    elif grasa <= 22:  score += 18
    elif grasa <= 27:  score += 8
    else:              score += 0

    # Músculo % (30 pts) — objetivo: >40%
    musc = m.get("musculo_pct", 0)
    if musc >= 45:     score += 30
    elif musc >= 40:   score += 24
    elif musc >= 35:   score += 15
    elif musc >= 30:   score += 7
    else:              score += 0

    # Agua % (20 pts) — objetivo: 55-65%
    agua = m.get("agua", 0)
    if 55 <= agua <= 65:  score += 20
    elif 50 <= agua < 55: score += 14
    elif agua >= 45:      score += 7
    else:                 score += 0

    # Grasa visceral (15 pts) — objetivo: ≤9
    visc = m.get("grasa_visceral", 99)
    if visc <= 7:    score += 15
    elif visc <= 9:  score += 11
    elif visc <= 12: score += 5
    else:            score += 0

    if score >= 80:   desc = "Élite 🏆"
    elif score >= 65: desc = "Muy bueno 💪"
    elif score >= 50: desc = "En progreso 📈"
    elif score >= 35: desc = "Necesita atención ⚠️"
    else:             desc = "Zona de riesgo 🚨"

    return score, desc


# ─── ANÁLISIS IA ───────────────────────────────────────────────────────────────

def analizar_con_ia(m: dict, anterior: dict | None, tendencia: dict | None) -> str:
    """
    Análisis clínico con contexto rico: pesaje anterior real + tendencia 7 días.
    GARANTÍA: siempre retorna string, nunca None.
    """
    logging.info("🧠 Generando análisis clínico con IA...")
    client = genai.Client(api_key=env_vars["GOOGLE_API_KEY"])

    # Bloque de comparativa vs pesaje anterior
    ctx_anterior = ""
    if anterior:
        dias_desde = (
            datetime.strptime(m["fecha_str"], "%Y-%m-%d") -
            datetime.strptime(anterior["fecha"], "%Y-%m-%d")
        ).days
        ctx_anterior = (
            f"\n--- COMPARATIVA VS PESAJE ANTERIOR (hace {dias_desde} día(s)) ---\n"
            f"Peso:    {anterior['peso']} kg  →  {m['peso']} kg  "
            f"({m['peso'] - anterior['peso']:+.2f} kg)\n"
            f"Grasa:   {anterior['grasa']}%  →  {m['grasa']}%  "
            f"({m['grasa'] - anterior['grasa']:+.1f}%)\n"
            f"Músculo: {anterior['musculo_pct']}%  →  {m['musculo_pct']}%  "
            f"({m['musculo_pct'] - anterior['musculo_pct']:+.1f}%)\n"
            f"Agua:    {anterior['agua']}%  →  {m['agua']}%  "
            f"({m['agua'] - anterior['agua']:+.1f}%)\n"
        )

    # Bloque de tendencia semanal
    ctx_tendencia = ""
    if tendencia:
        dir_peso  = "bajando" if tendencia["peso_tendencia_dia"] < -0.01 else \
                    "subiendo" if tendencia["peso_tendencia_dia"] > 0.01 else "estable"
        dir_grasa = "bajando" if tendencia["grasa_tendencia_dia"] < -0.005 else \
                    "subiendo" if tendencia["grasa_tendencia_dia"] > 0.005 else "estable"
        ctx_tendencia = (
            f"\n--- TENDENCIA ÚLTIMOS 7 DÍAS ({tendencia['dias_medidos']} mediciones) ---\n"
            f"Peso:  {dir_peso} ({tendencia['peso_tendencia_dia']:+.3f} kg/día)  "
            f"[rango: {tendencia['peso_min']}–{tendencia['peso_max']} kg]\n"
            f"Grasa: {dir_grasa} ({tendencia['grasa_tendencia_dia']:+.4f} %/día)\n"
        )

    score, desc_score = calcular_score_composicion(m)

    prompt = f"""Eres un experto en recomposición corporal y fisiología del ejercicio. Analiza las siguientes métricas con criterio clínico y responde ÚNICAMENTE en el formato indicado.

MÉTRICAS DE HOY — {m['fecha_str']}:
- Peso: {m['peso']} kg  |  BMI: {m['bmi']}  |  Peso libre de grasa: {m['fat_free_weight']} kg
- Músculo esquelético: {m['musculo_pct']}% ({m['masa_muscular_kg']} kg absolutos)
- Grasa corporal: {m['grasa']}%  |  Grasa visceral: {m['grasa_visceral']}
- Agua corporal: {m['agua']}%  |  Proteína corporal: {m['proteina']}%
- Masa ósea: {m['masa_osea']} kg  |  BMR: {m['bmr']} kcal/día
- Edad metabólica: {m['edad_metabolica']} años
- Score de composición corporal: {score}/100 ({desc_score})
{ctx_anterior}{ctx_tendencia}

INSTRUCCIÓN: Distingue entre ruido hídrico normal (fluctuaciones de agua) y cambios reales en masa grasa/muscular. Considera que el peso diario puede variar 1-2 kg por hidratación sin significar cambio real.

Responde SOLO con este bloque exacto, sin texto adicional antes ni después:

<b>📊 Análisis de Hoy:</b> [evaluación clínica del pesaje, distinguiendo ruido hídrico de cambio real. Si hay tendencia de 7 días, úsala para contextualizar]

<b>⚡ BMR y Energía:</b> [una línea: qué significan sus {m['bmr']} kcal de BMR para su ingesta objetivo hoy]

<b>🎯 Acción Concreta:</b> [recomendación específica de nutrición O entrenamiento para las próximas 24 horas, basada en los datos]

<i>🔥 Foco: [una frase motivadora personalizada basada en su situación real]</i>

REGLA ABSOLUTA: Usa ÚNICAMENTE las etiquetas <b> e <i>. Prohibido <br>, <hr>, <ul>, <li>, <h1>, <h2>, <h3> o cualquier otra etiqueta HTML."""

    for intento in range(3):
        try:
            respuesta = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
            texto = respuesta.text.strip() if respuesta and respuesta.text else ""
            if texto:
                return texto
            logging.warning(f"Intento {intento + 1}: Gemini devolvió respuesta vacía.")
        except Exception as e:
            logging.warning(f"Intento {intento + 1} fallido: {e}")
            if intento < 2:
                time.sleep(2)

    logging.error("Gemini falló tras 3 intentos.")
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
        logging.info(f"[DRY RUN] Telegram:\n{mensaje}")
        return

    for old, new in _HTML_SANITIZE:
        mensaje = mensaje.replace(old, new)

    url     = f"https://api.telegram.org/bot{env_vars['TELEGRAM_BOT_TOKEN']}/sendMessage"
    payload = {"chat_id": env_vars["TELEGRAM_CHAT_ID"], "text": mensaje, "parse_mode": "HTML"}

    res = requests.post(url, json=payload, timeout=10)
    if res.status_code == 200:
        return

    logging.warning(f"Telegram rechazó HTML ({res.status_code}). Reintentando en texto plano...")
    payload.pop("parse_mode")
    res2 = requests.post(url, json=payload, timeout=10)
    if res2.status_code != 200:
        logging.error(f"Error crítico enviando a Telegram: {res2.text}")


# ─── UTILIDADES ───────────────────────────────────────────────────────────────

def calcular_delta(hoy: float, ayer: float | None, invert_colors: bool = False) -> str:
    if ayer is None:
        return ""
    diff = hoy - ayer
    if abs(diff) < 0.05:
        return " ⚪"
    emoji = ("🟢" if diff < 0 else "🔴") if invert_colors else ("🟢" if diff > 0 else "🔴")
    return f" (Δ {diff:+.2f} {emoji})"


def generar_alertas(m: dict) -> str:
    """
    Genera alertas clínicas automáticas basadas en rangos.
    Retorna string vacío si todo está en orden.
    """
    alertas = []

    if m.get("grasa_visceral") and m["grasa_visceral"] >= 10:
        alertas.append(f"⚠️ Grasa visceral elevada ({m['grasa_visceral']}) — riesgo metabólico")
    if m.get("agua") and m["agua"] < 50:
        alertas.append(f"💧 Hidratación baja ({m['agua']}%) — bebe más agua hoy")
    if m.get("proteina") and m["proteina"] < 16:
        alertas.append(f"🥩 Proteína corporal baja ({m['proteina']}%) — revisa ingesta proteica")
    if m.get("edad_metabolica") and m.get("bmr"):
        # Si la edad metabólica es muy superior a la real (no tenemos edad real, usamos BMR como proxy)
        if m["edad_metabolica"] > 45:
            alertas.append(f"📅 Edad metabólica alta ({m['edad_metabolica']} años) — prioriza músculo")

    if not alertas:
        return ""
    return "\n🚨 <b>Alertas Clínicas:</b>\n" + "\n".join(f"  {a}" for a in alertas) + "\n"


# ─── FLUJO PRINCIPAL ──────────────────────────────────────────────────────────

def ejecutar_diario() -> bool:
    try:
        inicializar_db()
        m = obtener_datos_renpho()

        if not guardar_si_es_nuevo(m):
            return True  # Sin bloquear job_dieta.py los domingos

        logging.info("🚀 Nuevo pesaje detectado. Generando reporte...")

        anterior  = obtener_pesaje_anterior(m["fecha_str"])
        tendencia = obtener_tendencia_7_dias(m["fecha_str"])
        score, desc_score = calcular_score_composicion(m)
        alertas   = generar_alertas(m)
        analisis  = analizar_con_ia(m, anterior, tendencia)

        # Deltas vs pesaje anterior real (no "ayer" por calendario)
        ant = anterior  # alias corto
        d_peso  = calcular_delta(m["peso"],          ant["peso"] if ant else None,          invert_colors=True)
        d_grasa = calcular_delta(m["grasa"],         ant["grasa"] if ant else None,         invert_colors=True)
        d_musc  = calcular_delta(m["musculo_pct"],   ant["musculo_pct"] if ant else None)
        d_agua  = calcular_delta(m["agua"],          ant["agua"] if ant else None)
        d_visc  = calcular_delta(m["grasa_visceral"],ant["grasa_visceral"] if ant else None, invert_colors=True)

        # Días desde último pesaje (para contextualizar los deltas)
        dias_desde_str = ""
        if ant:
            dias = (
                datetime.strptime(m["fecha_str"], "%Y-%m-%d") -
                datetime.strptime(ant["fecha"], "%Y-%m-%d")
            ).days
            dias_desde_str = f" <i>(vs hace {dias} día{'s' if dias != 1 else ''})</i>"

        # Tendencia en el título
        if tendencia:
            dir_peso = "📉" if tendencia["peso_tendencia_dia"] < -0.01 else \
                       "📈" if tendencia["peso_tendencia_dia"] > 0.01 else "➡️"
            tendencia_str = f"  {dir_peso} {tendencia['peso_tendencia_dia']:+.2f} kg/día (7d)"
        else:
            tendencia_str = ""

        reporte = (
            f"📊 <b>REPORTE DIARIO — {m['fecha_str']}</b>{dias_desde_str}\n"
            f"{'─' * 30}\n"
            f"🏆 <b>Score Composición:</b> {score}/100 — {desc_score}{tendencia_str}\n\n"
            f"⚖️  <b>Peso:</b>            {m['peso']} kg{d_peso}\n"
            f"💪  <b>Músculo:</b>         {m['musculo_pct']}% ({m['masa_muscular_kg']} kg){d_musc}\n"
            f"🥓  <b>Grasa:</b>           {m['grasa']}%{d_grasa}{clasificar(m['grasa'], 'grasa_hombre')}\n"
            f"🫀  <b>Grasa Visceral:</b>  {m['grasa_visceral']}{d_visc}{clasificar(m['grasa_visceral'], 'visceral')}\n"
            f"💧  <b>Agua:</b>            {m['agua']}%{d_agua}{clasificar(m['agua'], 'agua')}\n"
            f"🧬  <b>Proteína:</b>        {m['proteina']}%{clasificar(m['proteina'], 'proteina')}\n"
            f"🦴  <b>Masa Ósea:</b>       {m['masa_osea']} kg\n"
            f"⚡  <b>BMR:</b>             {m['bmr']} kcal/día\n"
            f"📅  <b>Edad Metabólica:</b> {m['edad_metabolica']} años\n"
            f"📐  <b>BMI:</b>             {m['bmi']}{clasificar(m['bmi'], 'bmi')}\n"
            f"{alertas}"
            f"\n🤖 <b>Análisis IA:</b>\n{analisis}"
        )

        enviar_telegram(reporte)
        logging.info("✅ Reporte diario completado y enviado.")
        return True

    except Exception:
        logging.error("🔴 Error crítico en el flujo diario.", exc_info=True)
        enviar_telegram("🔴 <b>Error Crítico — Reporte Diario:</b> Revisa los logs en Railway.")
        return False


if __name__ == "__main__":
    ejecutar_diario()
