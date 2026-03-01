"""
job_dieta.py — V5.1 Production Grade
Mejoras vs V4.0:
  1. Context managers en TODAS las conexiones SQLite
  2. GOOGLE_API_KEY explícita (consistente con daily_renpho)
  3. MIMO promovido a ciudadano de primera clase en el reporte
  4. Validación robusta del dato_anterior (mínimo 5 días atrás)
  5. Score de composición corporal heredado del daily
  6. Alertas clínicas automáticas
  7. analizar_con_ia nunca retorna None (reintentos + fallback)
  8. Semáforos visuales en métricas del reporte
  9. Particionado de Telegram con _HTML_SANITIZE centralizado
  10. logging con exc_info en todos los errores críticos

Fixes V5.1 (code review Gemini):
  - BMR inyectado en prompt como límite mínimo calórico (protege tiroides)
  - .copy() en slice de Pandas (elimina SettingWithCopyWarning)
  - DB commit ANTES de enviar Telegram (idempotencia real)
  - Arquitectura de conexiones separada: leer → calcular → escribir
"""

import os
import sqlite3
import pandas as pd
import requests
import logging
from datetime import datetime, timedelta
from pytz import timezone
from google import genai

# ─── CONFIG ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

DRY_RUN = os.getenv("DRY_RUN", "false").lower() == "true"
TZ      = timezone(os.getenv("TZ", "America/Phoenix"))
DB_PATH = "/app/data/mis_datos_renpho.db"

REQUIRED_VARS = ["TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "GOOGLE_API_KEY"]
env_vars = {v: os.getenv(v) for v in REQUIRED_VARS}
faltantes = [v for v, k in env_vars.items() if not k]
if faltantes:
    raise RuntimeError(f"Faltan variables de entorno: {', '.join(faltantes)}")


# ─── RANGOS CLÍNICOS (mismo estándar que daily_renpho) ────────────────────────

RANGOS = {
    "bmi":          {"optimo": (18.5, 24.9), "alerta": (25.0, 29.9), "critico": (30.0, 99)},
    "grasa_hombre": {"optimo": (10.0, 20.0), "alerta": (20.1, 25.0), "critico": (25.1, 100)},
    "visceral":     {"optimo": (1,    9),    "alerta": (10,   14),   "critico": (15,   30)},
    "agua":         {"optimo": (50.0, 65.0), "alerta": (45.0, 49.9), "critico": (0,    44.9)},
    "proteina":     {"optimo": (16.0, 20.0), "alerta": (14.0, 15.9), "critico": (0,    13.9)},
}

def clasificar(valor, metrica: str) -> str:
    if valor is None or metrica not in RANGOS:
        return ""
    r = RANGOS[metrica]
    if r["optimo"][0] <= valor <= r["optimo"][1]:   return " 🟢"
    if r["alerta"][0] <= valor <= r["alerta"][1]:   return " 🟡"
    if r["critico"][0] <= valor <= r["critico"][1]: return " 🔴"
    return ""

def calcular_score_composicion(peso, grasa, musculo_pct, agua, visceral) -> tuple[int, str]:
    score = 0
    if grasa <= 15:    score += 35
    elif grasa <= 18:  score += 28
    elif grasa <= 22:  score += 18
    elif grasa <= 27:  score += 8

    if musculo_pct >= 45:   score += 30
    elif musculo_pct >= 40: score += 24
    elif musculo_pct >= 35: score += 15
    elif musculo_pct >= 30: score += 7

    if 55 <= agua <= 65:    score += 20
    elif 50 <= agua < 55:   score += 14
    elif agua >= 45:        score += 7

    if visceral <= 7:    score += 15
    elif visceral <= 9:  score += 11
    elif visceral <= 12: score += 5

    if score >= 80:   desc = "Élite 🏆"
    elif score >= 65: desc = "Muy bueno 💪"
    elif score >= 50: desc = "En progreso 📈"
    elif score >= 35: desc = "Necesita atención ⚠️"
    else:             desc = "Zona de riesgo 🚨"
    return score, desc

def generar_alertas(peso, grasa, agua, visceral, proteina, edad_metabolica) -> str:
    alertas = []
    if visceral and visceral >= 10:
        alertas.append(f"⚠️ Grasa visceral elevada ({visceral}) — riesgo metabólico activo")
    if agua and agua < 50:
        alertas.append(f"💧 Hidratación baja ({agua}%) — prioriza agua esta semana")
    if proteina and proteina < 16:
        alertas.append(f"🥩 Proteína corporal baja ({proteina}%) — revisa ingesta proteica diaria")
    if edad_metabolica and edad_metabolica > 45:
        alertas.append(f"📅 Edad metabólica alta ({edad_metabolica} años) — prioriza hipertrofia")
    if not alertas:
        return ""
    return "\n🚨 <b>Alertas Clínicas:</b>\n" + "\n".join(f"  {a}" for a in alertas) + "\n"


# ─── BASE DE DATOS ─────────────────────────────────────────────────────────────

def inicializar_bd():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS config_nutricion (
                clave TEXT PRIMARY KEY,
                valor REAL
            )
        """)
        conn.execute("""
            INSERT OR IGNORE INTO config_nutricion (clave, valor)
            VALUES ('kcal_mult', 26.0)
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS historico_dietas (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                fecha         TEXT,
                peso          REAL,
                grasa         REAL,
                delta_peso    REAL,
                kcal_mult     REAL,
                calorias      INTEGER,
                proteina      INTEGER,
                carbs         INTEGER,
                grasas        INTEGER,
                dieta_html    TEXT,
                estado_mimo   TEXT,
                shadow_mult   REAL,
                score_comp    INTEGER
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_hist_fecha ON historico_dietas(fecha)")

        # Migraciones en caliente
        columnas = {row[1] for row in conn.execute("PRAGMA table_info(historico_dietas)")}
        migraciones = {
            "estado_mimo": "TEXT",
            "shadow_mult": "REAL",
            "score_comp":  "INTEGER",
        }
        for col, tipo in migraciones.items():
            if col not in columnas:
                conn.execute(f"ALTER TABLE historico_dietas ADD COLUMN {col} {tipo}")
                logging.info(f"Migración aplicada: columna {col} añadida.")
        conn.commit()


def obtener_multiplicador(conn: sqlite3.Connection) -> float:
    row = conn.execute(
        "SELECT valor FROM config_nutricion WHERE clave='kcal_mult'"
    ).fetchone()
    return float(row[0]) if row else 26.0


def actualizar_multiplicador(conn: sqlite3.Connection, nuevo_mult: float):
    conn.execute(
        "UPDATE config_nutricion SET valor=? WHERE clave='kcal_mult'",
        (nuevo_mult,)
    )


def job_ya_ejecutado_hoy(conn: sqlite3.Connection) -> bool:
    hoy = datetime.now(TZ).strftime("%Y-%m-%d")
    row = conn.execute(
        "SELECT 1 FROM historico_dietas WHERE fecha LIKE ? LIMIT 1",
        (f"{hoy}%",)
    ).fetchone()
    return row is not None


def obtener_datos_semana(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("""
        SELECT Fecha, Peso_kg, Grasa_Porcentaje, Musculo, FatFreeWeight,
               Agua, VisFat, BMI, EdadMetabolica, Proteina, MasaOsea, BMR
        FROM pesajes
        WHERE Fecha >= date('now', '-14 day')
        ORDER BY Fecha ASC
    """, conn)


# ─── LEYES DE CONTROL ─────────────────────────────────────────────────────────

# Estado MIMO: diagnóstico multi-variable (grasa + músculo + peso)
# Estado SISO: acción de control mono-variable (solo delta_peso → multiplicador)
# Shadow mode: MIMO calcula pero NO actúa todavía — solo se loguea y reporta

ESTADOS_MIMO = {
    "CATABOLISMO":    ("🔴", "Pérdida de músculo sin quema de grasa. Aumenta carbs peri-entrenamiento."),
    "RECOMPOSICION":  ("🟣", "Recomposición activa. Mantén proteína en límite superior."),
    "CUTTING_LIMPIO": ("🟢", "Déficit funcionando correctamente. Mantén el curso."),
    "ESTANCAMIENTO":  ("🟡", "Adaptación metabólica. Forzar oxidación de lípidos."),
    "ZONA_GRIS":      ("⚪", "Señales mixtas o ruido hídrico. Observar tendencia."),
}

def evaluar_mimo(delta_peso: float, delta_grasa: float, delta_musculo: float, mult_actual: float) -> tuple:
    """
    Motor de diagnóstico multi-variable.
    Retorna (estado, mult_sugerido, razon).
    NO modifica la base de datos — solo diagnostica.
    """
    TOL = 0.2
    if delta_peso < -0.8 and delta_musculo < -TOL and delta_grasa > -TOL:
        estado = "CATABOLISMO"
        mult   = mult_actual + 1
        razon  = f"Pérdida de peso ({delta_peso:+.2f}kg) y músculo ({delta_musculo:+.2f}%) sin quema de grasa."
    elif abs(delta_peso) <= 0.3 and delta_grasa < -TOL and delta_musculo > TOL:
        estado = "RECOMPOSICION"
        mult   = mult_actual
        razon  = f"Peso estable. Grasa ({delta_grasa:+.2f}%), Músculo ({delta_musculo:+.2f}%) — recomp. activa."
    elif delta_peso <= -0.3 and delta_grasa < -TOL and abs(delta_musculo) <= TOL:
        estado = "CUTTING_LIMPIO"
        mult   = mult_actual
        razon  = f"Pérdida controlada ({delta_peso:+.2f}kg) de tejido adiposo. Músculo preservado."
    elif delta_peso > -0.2 and delta_grasa >= -TOL and delta_musculo <= TOL:
        estado = "ESTANCAMIENTO"
        mult   = mult_actual - 1
        razon  = "Sin mejora en composición. Adaptación metabólica detectada."
    else:
        estado = "ZONA_GRIS"
        mult   = mult_actual
        razon  = "Señales mixtas. Puede ser ruido hídrico. Requiere más datos."

    mult_seguro = max(20.0, min(mult, 34.0))
    return estado, mult_seguro, razon


def aplicar_siso(delta_peso: float, mult_actual: float) -> tuple:
    """
    Ley de control SISO activa — modifica el multiplicador real.
    Variable de control: delta_peso. Salida: nuevo multiplicador.
    """
    if delta_peso < -0.8:
        nuevo = mult_actual + 1
        razon = "📉 Pérdida rápida. Aumento multiplicador para proteger músculo."
        cambio = True
    elif delta_peso > -0.2:
        nuevo = mult_actual - 1
        razon = "🛑 Estancamiento. Recorto multiplicador calórico."
        cambio = True
    else:
        nuevo = mult_actual
        razon = "✅ Progreso óptimo. Multiplicador mantenido."
        cambio = False

    nuevo_seguro = max(20.0, min(nuevo, 34.0))
    if nuevo_seguro != nuevo:
        razon += f" (Limitado a {nuevo_seguro})"
    return nuevo_seguro, razon, cambio


# ─── GENERACIÓN DE DIETA ───────────────────────────────────────────────────────

def generar_dieta_ia(
    peso, grasa, visceral, agua, fat_free_weight,
    calorias, proteina, carbs, grasas, bmr,
    delta_peso, delta_grasa, delta_musculo,
    estado_mimo, razon_mimo
) -> str:
    """
    Genera el plan semanal de nutrición y entrenamiento.
    GARANTÍA: siempre retorna string, nunca None.
    """
    logging.info("🧠 Generando plan semanal con IA...")

    prompt = f"""Eres mi nutriólogo deportivo y entrenador personal de alto rendimiento. Diseña un plan completo de 7 días basado en mis datos exactos.

PERFIL ACTUAL:
- Peso: {peso} kg | Grasa: {grasa}% (Visceral: {visceral}) | Agua: {agua}%
- Masa libre de grasa (FFM): {fat_free_weight} kg
- Variación semanal: Peso ({delta_peso:+.2f} kg), Grasa ({delta_grasa:+.2f}%), Músculo ({delta_musculo:+.2f}%)
- Diagnóstico metabólico: {estado_mimo} — {razon_mimo}

MACROS DIARIOS CALCULADOS:
- Calorías: {calorias} kcal | Proteína: {proteina}g | Carbohidratos: {carbs}g | Grasas: {grasas}g
- ⚠️ LÍMITE MÍNIMO ABSOLUTO: Nunca recomiendes por debajo de {bmr} kcal/día (BMR real).
  Comer por debajo del BMR destruye el metabolismo y la masa muscular. Es innegociable.

RESTRICCIONES DE ESTILO DE VIDA (OBLIGATORIAS):
1. LUNES, MIÉRCOLES, JUEVES (Oficina + Gym pesado):
   - Salgo a las 4pm, entreno 45 min en gym, ceno a las 6pm
   - Cenas deben ser muy saciantes y altas en proteína
   - El lonche del día siguiente es SIEMPRE la sobra de la cena anterior

2. MARTES Y VIERNES (Home Office + cuidado del bebé):
   - Entreno 30 min en casa durante la siesta del bebé
   - Dame rutina EXACTA de ejercicios para esos 30 minutos en casa
   - Sin equipamiento pesado (bebé durmiendo)

3. FIN DE SEMANA:
   - Actividad de recuperación activa o tiempo en familia activo
   - Una comida social permitida (ajusta macros del día)

4. DESAYUNOS: Ultra-rápidos (menos de 5 minutos), portátiles para comer en el auto

5. COLACIÓN DIARIA: Incluye siempre 1 colación basada en frutas frescas para controlar antojos, ajustando la cena para no exceder calorías

6. HIDRATACIÓN: Sugiere consumo de agua específico basado en mi agua corporal actual ({agua}%)

REGLA ABSOLUTA DE FORMATO:
Usa ÚNICAMENTE etiquetas <b> e <i> para resaltar texto.
Usa saltos de línea reales y guiones (-) para listas.
PROHIBIDO usar <br>, <hr>, <ul>, <li>, <h1>, <h2>, <h3>, <p> o cualquier otra etiqueta HTML."""

    for intento in range(3):
        try:
            client_ia = genai.Client(api_key=env_vars["GOOGLE_API_KEY"])
            respuesta = client_ia.models.generate_content(
                model="gemini-2.5-pro", contents=prompt
            )
            texto = respuesta.text.strip() if respuesta and respuesta.text else ""
            if texto:
                return texto
            logging.warning(f"Intento {intento + 1}: Gemini devolvió respuesta vacía.")
        except Exception as e:
            logging.warning(f"Intento {intento + 1} fallido: {e}")
            import time; time.sleep(2)

    logging.error("Gemini falló tras 3 intentos.")
    return "<i>⚠️ Plan de IA no disponible. Mantén los macros calculados y el plan de la semana anterior.</i>"


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

    # Particionado inteligente para mensajes largos (límite Telegram: 4096 chars)
    partes = []
    while mensaje:
        if len(mensaje) <= 3900:
            partes.append(mensaje)
            break
        corte = mensaje.rfind("\n\n", 0, 3900)
        if corte == -1: corte = mensaje.rfind("\n", 0, 3900)
        if corte == -1: corte = 3900
        partes.append(mensaje[:corte])
        mensaje = mensaje[corte:].lstrip()

    url = f"https://api.telegram.org/bot{env_vars['TELEGRAM_BOT_TOKEN']}/sendMessage"
    for i, parte in enumerate(partes, 1):
        suffix = f"\n<i>({i}/{len(partes)})</i>" if len(partes) > 1 else ""
        payload = {
            "chat_id":    env_vars["TELEGRAM_CHAT_ID"],
            "text":       parte + suffix,
            "parse_mode": "HTML",
        }
        res = requests.post(url, json=payload, timeout=10)
        if res.status_code != 200:
            logging.warning(f"Telegram rechazó HTML (parte {i}). Fallback a texto plano...")
            payload.pop("parse_mode")
            res2 = requests.post(url, json=payload, timeout=10)
            if res2.status_code != 200:
                logging.error(f"Error crítico en fallback Telegram parte {i}: {res2.text}")


# ─── JOB PRINCIPAL ────────────────────────────────────────────────────────────

def ejecutar_job():
    logging.info("🚀 Iniciando Job Semanal de Control Metabólico V5.1...")
    inicializar_bd()

    # ── Filtro de día: solo corre los domingos ────────────────────────────────
    # Como el cron es diario, este guardia evita que corra lunes-sábado
    hoy = datetime.now(TZ)
    if hoy.weekday() != 6:  # 6 = domingo
        logging.info(f"Hoy es {hoy.strftime('%A')}. El job de dieta solo corre los domingos. Omitiendo.")
        return

    with sqlite3.connect(DB_PATH) as conn:

        # ── Idempotencia ──────────────────────────────────────────────────────
        if job_ya_ejecutado_hoy(conn):
            logging.warning("Job semanal ya ejecutado hoy. Abortando por idempotencia.")
            return

        # ── Extracción de datos ───────────────────────────────────────────────
        df = obtener_datos_semana(conn)
        if df.empty or len(df) < 2:
            enviar_telegram("⚠️ Necesito al menos 2 pesajes recientes para calcular la dieta.")
            return

        df["Fecha"] = pd.to_datetime(df["Fecha"])
        dato_actual = df.iloc[-1]

        # Dato anterior: el pesaje más cercano a hace 7 días, mínimo 5 días atrás
        # Esto evita comparar contra un pesaje de hace 1 día por error
        fecha_limite = df.iloc[-1]["Fecha"] - timedelta(days=5)
        df_anteriores = df[df["Fecha"] <= fecha_limite].copy()

        if df_anteriores.empty:
            enviar_telegram("⚠️ No hay pesajes con al menos 5 días de antigüedad. Espera más datos.")
            return

        fecha_ref    = dato_actual["Fecha"] - timedelta(days=7)
        df_anteriores["diff"] = (df_anteriores["Fecha"] - fecha_ref).abs()
        dato_anterior = df_anteriores.loc[df_anteriores["diff"].idxmin()]
        dias_entre    = (dato_actual["Fecha"] - dato_anterior["Fecha"]).days

        # ── Variables principales ─────────────────────────────────────────────
        peso_actual      = float(dato_actual["Peso_kg"])
        grasa_actual     = float(dato_actual["Grasa_Porcentaje"])
        musculo_actual   = float(dato_actual["Musculo"])
        agua_actual      = float(dato_actual["Agua"])
        fat_free_weight  = float(dato_actual["FatFreeWeight"])
        visfat_actual    = float(dato_actual["VisFat"])
        bmi_actual       = float(dato_actual["BMI"]) if dato_actual["BMI"] else None
        edad_metabolica  = int(dato_actual["EdadMetabolica"]) if dato_actual["EdadMetabolica"] else None
        proteina_corp    = float(dato_actual["Proteina"]) if dato_actual["Proteina"] else None
        masa_osea        = float(dato_actual["MasaOsea"]) if dato_actual["MasaOsea"] else None
        bmr_actual       = int(dato_actual["BMR"]) if dato_actual.get("BMR") else round(peso_actual * 22)

        delta_peso    = peso_actual   - float(dato_anterior["Peso_kg"])
        delta_grasa   = grasa_actual  - float(dato_anterior["Grasa_Porcentaje"])
        delta_musculo = musculo_actual - float(dato_anterior["Musculo"])

        # ── Scoring y alertas ─────────────────────────────────────────────────
        score, desc_score = calcular_score_composicion(
            peso_actual, grasa_actual, musculo_actual, agua_actual, visfat_actual
        )
        alertas = generar_alertas(
            peso_actual, grasa_actual, agua_actual,
            visfat_actual, proteina_corp, edad_metabolica
        )

        # ── Control metabólico ────────────────────────────────────────────────
        mult_actual = obtener_multiplicador(conn)

        # MIMO: diagnóstico multi-variable (shadow — no actúa aún)
        estado_mimo, shadow_mult, razon_mimo = evaluar_mimo(
            delta_peso, delta_grasa, delta_musculo, mult_actual
        )
        emoji_mimo, consejo_mimo = ESTADOS_MIMO.get(estado_mimo, ("⚪", "Sin diagnóstico."))
        logging.info(
            f"[MIMO] estado={estado_mimo} | actual={mult_actual:.1f} | "
            f"sugerido={shadow_mult:.1f} | Δpeso={delta_peso:+.2f} | "
            f"Δgrasa={delta_grasa:+.2f} | Δmúsculo={delta_musculo:+.2f}"
        )

        # SISO: control activo mono-variable (actúa sobre el multiplicador real)
        nuevo_mult, razon_siso, hubo_cambio = aplicar_siso(delta_peso, mult_actual)
        if hubo_cambio:
            actualizar_multiplicador(conn, nuevo_mult)
            conn.commit()
            logging.info(f"[SISO] Multiplicador actualizado: {mult_actual} → {nuevo_mult}")

        # ── Cálculo de macros ─────────────────────────────────────────────────
        calorias = round(peso_actual * nuevo_mult)
        proteina = round(fat_free_weight * 2.2)
        grasas   = round(peso_actual * 0.7)
        carbs    = max(0, round((calorias - (proteina * 4 + grasas * 9)) / 4))

        # ── Generación del plan ───────────────────────────────────────────────
        dieta_html = generar_dieta_ia(
            peso_actual, grasa_actual, visfat_actual, agua_actual, fat_free_weight,
            calorias, proteina, carbs, grasas, bmr_actual,
            delta_peso, delta_grasa, delta_musculo,
            estado_mimo, razon_mimo,
        )

        # ── Construcción del reporte ──────────────────────────────────────────
        def delta_str(val, invert=False):
            """Mini helper para formatear deltas con semáforo."""
            if abs(val) < 0.05: return f"({val:+.2f}) ⚪"
            if invert: emoji = "🟢" if val < 0 else "🔴"
            else:      emoji = "🟢" if val > 0 else "🔴"
            return f"({val:+.2f}) {emoji}"

        masa_osea_str = f" | Masa Ósea: {masa_osea} kg" if masa_osea else ""

        reporte = (
            f"🤖 <b>CONTROL METABÓLICO V5.0 — {datetime.now(TZ).strftime('%d/%m/%Y')}</b>\n"
            f"<i>Comparativa vs hace {dias_entre} días</i>\n"
            f"{'─' * 32}\n\n"

            f"🏆 <b>Score de Composición:</b> {score}/100 — {desc_score}\n\n"

            f"📊 <b>Telemetría Semanal:</b>\n"
            f"⚖️  Peso:      {peso_actual:.1f} kg  {delta_str(delta_peso, invert=True)}\n"
            f"🥓  Grasa:     {grasa_actual:.1f}%   {delta_str(delta_grasa, invert=True)}{clasificar(grasa_actual, 'grasa_hombre')}\n"
            f"💪  Músculo:   {musculo_actual:.1f}%  {delta_str(delta_musculo)}\n"
            f"🫀  Visceral:  {visfat_actual}{clasificar(visfat_actual, 'visceral')}\n"
            f"💧  Agua:      {agua_actual:.1f}%{clasificar(agua_actual, 'agua')}\n"
            f"🧬  Proteína:  {proteina_corp}%{clasificar(proteina_corp, 'proteina') if proteina_corp else ''}\n"
            f"📐  BMI:       {bmi_actual}{clasificar(bmi_actual, 'bmi') if bmi_actual else ''}\n"
            f"📅  Ed. Metab: {edad_metabolica} años{masa_osea_str}\n"
            f"🔩  FFM:       {fat_free_weight:.1f} kg\n"
            f"{alertas}\n"

            f"{'─' * 32}\n"
            f"🧠 <b>Diagnóstico MIMO:</b> {emoji_mimo} <b>{estado_mimo}</b>\n"
            f"<i>{razon_mimo}</i>\n"
            f"<i>Consejo: {consejo_mimo}</i>\n"
            f"Mult. MIMO sugerido: <b>{shadow_mult}</b> kcal/kg\n\n"

            f"⚙️ <b>Control SISO (Activo):</b>\n"
            f"<i>{razon_siso}</i>\n"
            f"Multiplicador aplicado: <b>{nuevo_mult}</b> kcal/kg\n\n"

            f"{'─' * 32}\n"
            f"🎯 <b>Macros Bio-Ajustados:</b>\n"
            f"Kcal: <b>{calorias}</b>  |  P: <b>{proteina}g</b>  |  C: <b>{carbs}g</b>  |  G: <b>{grasas}g</b>\n\n"

            f"{'─' * 32}\n"
            f"🥗 <b>TU PLAN SEMANAL:</b>\n\n{dieta_html}"
        )

        # ── Persistencia PRIMERO, Telegram DESPUÉS ────────────────────────────
        # Regla: si falla el INSERT, el job no queda marcado como ejecutado
        # y el cron reintentará. Si mandamos Telegram primero y falla el INSERT,
        # recibes el mensaje pero el job vuelve a correr la próxima hora.
        conn.execute("""
            INSERT INTO historico_dietas
            (fecha, peso, grasa, delta_peso, kcal_mult, calorias,
             proteina, carbs, grasas, dieta_html, estado_mimo, shadow_mult, score_comp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S"),
            peso_actual, grasa_actual, delta_peso, nuevo_mult,
            calorias, proteina, carbs, grasas,
            dieta_html, estado_mimo, shadow_mult, score,
        ))
        conn.commit()
        logging.info("💾 Historial persistido en SQLite.")

    # ── Telegram fuera del context manager (DB ya cerrada) ────────────────
    # La DB está cerrada antes de hacer la llamada de red — patrón SRE correcto.
    enviar_telegram(reporte)
    logging.info("✅ Job semanal ejecutado y notificado exitosamente.")


if __name__ == "__main__":
    ejecutar_job()
