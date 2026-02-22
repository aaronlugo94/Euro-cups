import os
import sqlite3
import pandas as pd
from google import genai  # <-- IMPORTACIÓN NUEVA
import requests
import logging
from datetime import datetime, timedelta
from pytz import timezone

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
DRY_RUN = os.getenv("DRY_RUN", "false").lower() == "true"
TZ = timezone(os.getenv("TZ", "America/Phoenix"))

# Ya no necesitamos genai.configure() aquí, el cliente nuevo lo toma automático de os.environ
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
ARCHIVO_DB = "/app/data/mis_datos_renpho.db"

# ==========================================
# ESTADO & TELEGRAM
# ==========================================
def inicializar_bd(ruta_db):
    conexion = sqlite3.connect(ruta_db)
    cursor = conexion.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS config_nutricion (clave TEXT PRIMARY KEY, valor REAL)")
    cursor.execute("INSERT OR IGNORE INTO config_nutricion (clave, valor) VALUES ('kcal_mult', 26.0)")
    cursor.execute('''CREATE TABLE IF NOT EXISTS historico_dietas (
        id INTEGER PRIMARY KEY AUTOINCREMENT, fecha TEXT, peso REAL, grasa REAL, delta_peso REAL,
        kcal_mult REAL, calorias INTEGER, proteina INTEGER, carbs INTEGER, grasas INTEGER, dieta_html TEXT)''')
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_hist_fecha ON historico_dietas(fecha)")
    conexion.commit()
    conexion.close()

def obtener_estado_actual(ruta_db):
    conexion = sqlite3.connect(ruta_db)
    row = conexion.cursor().execute("SELECT valor FROM config_nutricion WHERE clave='kcal_mult'").fetchone()
    conexion.close()
    return float(row[0]) if row else 26.0

def actualizar_estado(ruta_db, nuevo_mult):
    conexion = sqlite3.connect(ruta_db)
    conexion.cursor().execute("UPDATE config_nutricion SET valor=? WHERE clave='kcal_mult'", (nuevo_mult,))
    conexion.commit()
    conexion.close()

def enviar_mensaje_telegram(mensaje):
    if DRY_RUN: return logging.info(f"DRY RUN: {mensaje}")
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    
    # 🧹 FILTRO SANITARIO: Telegram odia estas etiquetas. Las convertimos a texto normal.
    mensaje = mensaje.replace("<br>", "\n").replace("<br/>", "\n").replace("<ul>", "").replace("</ul>", "").replace("<li>", "• ").replace("</li>", "\n")
    
    texto_seguro = mensaje if len(mensaje) < 4000 else mensaje[:3900] + "\n\n[... Menú truncado por Telegram ...]"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": texto_seguro, "parse_mode": "HTML"}
    
    res = requests.post(url, json=payload)
    if res.status_code != 200:
        logging.error(f"⚠️ Error HTML en Telegram: {res.text}. Enviando texto plano...")
        del payload["parse_mode"]  # <-- Esta es la forma correcta y blindada de quitarlo
        res2 = requests.post(url, json=payload)
        if res2.status_code != 200:
            logging.error(f"⚠️ Error CRÍTICO en fallback: {res2.text}"))

# ==========================================
# LEYES DE CONTROL (MIMO SHADOW & SISO ACTIVO)
# ==========================================
def evaluar_estado_metabolico(delta_peso, delta_grasa, delta_musculo, kcal_mult_actual):
    TOL = 0.2
    if delta_peso < -0.8 and delta_musculo < -TOL and delta_grasa > -TOL:
        return "CATABOLISMO", kcal_mult_actual + 1, "Aumentar carbs peri-entrenamiento.", f"Pérdida de peso ({delta_peso:+.2f}kg) y músculo ({delta_musculo:+.2f}kg) sin quema clara de grasa. Estrés."
    elif abs(delta_peso) <= 0.3 and delta_grasa < -TOL and delta_musculo > TOL:
        return "RECOMPOSICION", kcal_mult_actual, "Mantener proteína en límite superior.", f"Peso estable con recomposición: Grasa ({delta_grasa:+.2f}%), Músculo ({delta_musculo:+.2f}kg)."
    elif delta_peso <= -0.3 and delta_grasa < -TOL and abs(delta_musculo) <= TOL:
        return "CUTTING_LIMPIO", kcal_mult_actual, "Déficit funcionando.", f"Pérdida de peso controlada ({delta_peso:+.2f}kg) de tejido adiposo."
    elif delta_peso > -0.2 and delta_grasa >= -TOL and delta_musculo <= TOL:
        return "ESTANCAMIENTO", kcal_mult_actual - 1, "Forzar oxidación de lípidos.", "Adaptación metabólica sin mejora en composición."
    else:
        return "ZONA_GRIS", kcal_mult_actual, "Observar tendencia.", "Señales mixtas o ruido hídrico. Requiere más datos."

def aplicar_ley_de_control(delta_peso, kcal_mult_actual):
    nuevo_mult, cambio = kcal_mult_actual, False
    if delta_peso < -0.8:
        nuevo_mult += 1; razon = "📉 Pérdida rápida. Aumento multiplicador para proteger músculo."; cambio = True
    elif delta_peso > -0.2:
        nuevo_mult -= 1; razon = "🛑 Estancamiento. Recorto multiplicador calórico."; cambio = True
    else:
        razon = "✅ Progreso óptimo. Mantengo multiplicador."
    
    nuevo_mult_seguro = max(20.0, min(nuevo_mult, 34.0))
    if nuevo_mult_seguro != nuevo_mult: razon += f" (Limitado a {nuevo_mult_seguro})"
    return nuevo_mult_seguro, razon, cambio

# ==========================================
# JOB PRINCIPAL
# ==========================================
def ejecutar_job():
    logging.info("Iniciando Job Semanal de Control Metabólico...")
    inicializar_bd(ARCHIVO_DB)
    
    conexion = sqlite3.connect(ARCHIVO_DB)
    df = pd.read_sql_query("SELECT Fecha, Peso_kg, Grasa_Porcentaje, Musculo, FatFreeWeight, Agua, VisFat, BMI, EdadMetabolica FROM pesajes WHERE Fecha >= date('now', '-14 day') ORDER BY Fecha ASC", conexion)
    conexion.close()

    if df.empty or len(df) < 2:
        return enviar_mensaje_telegram("⚠️ Error: Necesito al menos 2 pesajes recientes para calcular la dieta.")

    df['Fecha'] = pd.to_datetime(df['Fecha']).dt.tz_localize('UTC', ambiguous='NaT', nonexistent='NaT').dt.tz_convert(TZ)
    dato_actual = df.iloc[-1]
    fecha_hace_una_semana = datetime.now(TZ) - timedelta(days=7)
    df['diff_dias'] = (df['Fecha'] - fecha_hace_una_semana).abs()
    dato_anterior = df.loc[df['diff_dias'].idxmin()]
    
    peso_actual, grasa_actual = float(dato_actual['Peso_kg']), float(dato_actual['Grasa_Porcentaje'])
    fat_free_weight = float(dato_actual['FatFreeWeight'])
    
    delta_peso = peso_actual - float(dato_anterior['Peso_kg'])
    delta_grasa = grasa_actual - float(dato_anterior['Grasa_Porcentaje'])
    delta_musculo = float(dato_actual['Musculo']) - float(dato_anterior['Musculo'])
    
    kcal_mult_actual = obtener_estado_actual(ARCHIVO_DB)

    # === SHADOW MODE MIMO (Solo Lectura) ===
    try:
        estado_mimo, shadow_mult, shadow_macros, shadow_razon = evaluar_estado_metabolico(delta_peso, delta_grasa, delta_musculo, kcal_mult_actual)
        logging.info(f"[SHADOW_MIMO] estado={estado_mimo} | kcal_actual={kcal_mult_actual:.1f} | kcal_sugerido={shadow_mult:.1f} | Δpeso={delta_peso:.2f}kg | Δgrasa={delta_grasa:.2f}% | Δmusculo={delta_musculo:.2f}kg")
        logging.info(f"[SHADOW_MIMO] razon={shadow_razon}")
    except Exception as e:
        logging.exception(f"[SHADOW_MIMO] Error: {e}")
        estado_mimo, shadow_mult, shadow_macros, shadow_razon = "ERROR", kcal_mult_actual, "Shadow Mode falló.", "Error evaluación MIMO."

    # === LEY SISO (Aplica cambios reales) ===
    nuevo_mult, razon_control, hubo_cambio = aplicar_ley_de_control(delta_peso, kcal_mult_actual)
    if hubo_cambio: actualizar_estado(ARCHIVO_DB, nuevo_mult)

    # === CÁLCULO DE MACROS ===
    calorias = round(peso_actual * nuevo_mult)
    proteina = round(fat_free_weight * 2.2) 
    grasas = round(peso_actual * 0.7) 
    carbs = max(0, round((calorias - (proteina * 4 + grasas * 9)) / 4))

    # === GENERACIÓN DE MENÚ (NUEVO SDK) ===
    prompt = f"""Eres mi nutriólogo deportivo. Diseña un plan de comidas de 7 días.
    Perfil: Peso: {peso_actual}kg | Grasa: {grasa_actual}% (Visceral: {dato_actual['VisFat']}) | Agua: {dato_actual['Agua']}% | FFM: {fat_free_weight}kg.
    Macros estrictos diarios: Kcal: {calorias} | P: {proteina}g | C: {carbs}g | G: {grasas}g.
    Nota: Grasa visceral en {dato_actual['VisFat']}. Prioriza omega 3 y antiinflamatorios.
    REGLA: Usa formato HTML básico (<b>, <i>, <ul>, <li>). NO uses Markdown. NO respondas con bloques de código."""
    
    try:
        # Usamos el cliente nuevo
        client = genai.Client() # Toma la API_KEY del entorno automáticamente
        respuesta = client.models.generate_content(model='gemini-2.5-pro', contents=prompt)
        
        if not respuesta or not hasattr(respuesta, "text") or not respuesta.text.strip(): raise ValueError("Respuesta IA vacía.")
        dieta_html = respuesta.text.strip()
        if len(dieta_html) > 3000: dieta_html = dieta_html[:3000] + "\n\n<i>... [Menú truncado por longitud. Revisa los primeros días] ...</i>"
    except Exception as e:
        return enviar_mensaje_telegram("⚠️ Error al contactar IA para generar menú.")

    # === NOTIFICAR ===
    mensaje_telegram = (
        f"🤖 <b>CONTROL METABÓLICO V4.0</b> 🤖\n\n"
        f"📊 <b>Telemetría Semanal:</b>\n"
        f"• Peso: {peso_actual:.1f} kg (Δ {delta_peso:+.2f} kg)\n"
        f"• FFM: {fat_free_weight:.1f} kg\n\n"
        f"🧠 <b>Acción del Sistema (SISO):</b>\n"
        f"<i>{razon_control}</i>\n"
        f"Multiplicador actual: {nuevo_mult} kcal/kg\n\n"
        f"🎯 <b>Macros Bio-Ajustados:</b>\n"
        f"Kcal: {calorias} | P: {proteina}g | C: {carbs}g | G: {grasas}g\n\n"
        f"🥗 <b>TU MENÚ:</b>\n\n{dieta_html}\n\n"
        f"👻 <b>Shadow Mode (MIMO):</b>\n"
        f"• Estado: <b>{estado_mimo}</b>\n"
        f"• Mult. Sugerido: {shadow_mult}\n"
        f"• Diagnóstico: <i>{shadow_razon}</i>"
    )
    enviar_mensaje_telegram(mensaje_telegram)
    
    conexion = sqlite3.connect(ARCHIVO_DB)
    conexion.cursor().execute('''INSERT INTO historico_dietas (fecha, peso, grasa, delta_peso, kcal_mult, calorias, proteina, carbs, grasas, dieta_html)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', (datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S"), peso_actual, grasa_actual, delta_peso, nuevo_mult, calorias, proteina, carbs, grasas, dieta_html))
    conexion.commit()
    conexion.close()
    logging.info("Job ejecutado exitosamente.")

if __name__ == "__main__":
    ejecutar_job()
