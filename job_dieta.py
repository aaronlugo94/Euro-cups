import os
import sqlite3
import pandas as pd
from google import genai
import requests
import logging
from datetime import datetime, timedelta
from pytz import timezone

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
DRY_RUN = os.getenv("DRY_RUN", "false").lower() == "true"
TZ = timezone(os.getenv("TZ", "America/Phoenix"))

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
    
    # Migración en caliente: Añadir columnas MIMO si no existen
    try: cursor.execute("ALTER TABLE historico_dietas ADD COLUMN estado_mimo TEXT")
    except sqlite3.OperationalError: pass
    try: cursor.execute("ALTER TABLE historico_dietas ADD COLUMN shadow_mult REAL")
    except sqlite3.OperationalError: pass
    
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
    
    # 🧹 FILTRO SANITARIO AGRESIVO (Destruye listas, líneas, párrafos y títulos)
    mensaje = mensaje.replace("<br>", "\n").replace("<br/>", "\n").replace("<ul>", "").replace("</ul>", "").replace("<li>", "• ").replace("</li>", "\n")
    mensaje = mensaje.replace("<hr>", "---").replace("<hr/>", "---").replace("<p>", "").replace("</p>", "\n").replace("<strong>", "<b>").replace("</strong>", "</b>")
    mensaje = mensaje.replace("<h1>", "").replace("</h1>", "\n").replace("<h2>", "").replace("</h2>", "\n").replace("<h3>", "").replace("</h3>", "\n")
    
    partes = []
    while len(mensaje) > 0:
        if len(mensaje) <= 3900:
            partes.append(mensaje)
            break
        corte = mensaje.rfind('\n\n', 0, 3900)
        if corte == -1: corte = mensaje.rfind('\n', 0, 3900)
        if corte == -1: corte = 3900
        partes.append(mensaje[:corte])
        mensaje = mensaje[corte:].lstrip()
        
    for parte in partes:
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": parte, "parse_mode": "HTML"}
        res = requests.post(url, json=payload)
        if res.status_code != 200:
            logging.error(f"⚠️ Error HTML en Telegram: {res.text}. Enviando texto plano...")
            del payload["parse_mode"]
            res2 = requests.post(url, json=payload)
            if res2.status_code != 200:
                logging.error(f"⚠️ Error CRÍTICO en fallback: {res2.text}")

# ==========================================
# LEYES DE CONTROL (MIMO SHADOW & SISO ACTIVO)
# ==========================================
def evaluar_estado_metabolico(delta_peso, delta_grasa, delta_musculo, kcal_mult_actual):
    TOL = 0.2
    if delta_peso < -0.8 and delta_musculo < -TOL and delta_grasa > -TOL:
        estado, mult, macros, razon = "CATABOLISMO", kcal_mult_actual + 1, "Aumentar carbs peri-entrenamiento.", f"Pérdida de peso ({delta_peso:+.2f}kg) y músculo ({delta_musculo:+.2f}%) sin quema clara de grasa. Estrés."
    elif abs(delta_peso) <= 0.3 and delta_grasa < -TOL and delta_musculo > TOL:
        estado, mult, macros, razon = "RECOMPOSICION", kcal_mult_actual, "Mantener proteína en límite superior.", f"Peso estable con recomposición: Grasa ({delta_grasa:+.2f}%), Músculo ({delta_musculo:+.2f}%)."
    elif delta_peso <= -0.3 and delta_grasa < -TOL and abs(delta_musculo) <= TOL:
        estado, mult, macros, razon = "CUTTING_LIMPIO", kcal_mult_actual, "Déficit funcionando.", f"Pérdida de peso controlada ({delta_peso:+.2f}kg) de tejido adiposo."
    elif delta_peso > -0.2 and delta_grasa >= -TOL and delta_musculo <= TOL:
        estado, mult, macros, razon = "ESTANCAMIENTO", kcal_mult_actual - 1, "Forzar oxidación de lípidos.", "Adaptación metabólica sin mejora en composición."
    else:
        estado, mult, macros, razon = "ZONA_GRIS", kcal_mult_actual, "Observar tendencia.", "Señales mixtas o ruido hídrico. Requiere más datos."
    
    return estado, max(20.0, min(mult, 34.0)), macros, razon

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
    
    # 🛡️ PROTECCIÓN DE IDEMPOTENCIA (ACTIVA)
    hoy = datetime.now(TZ).strftime("%Y-%m-%d")
    inicializar_bd(ARCHIVO_DB)
    conn = sqlite3.connect(ARCHIVO_DB)
    existe = conn.cursor().execute("SELECT 1 FROM historico_dietas WHERE fecha LIKE ? LIMIT 1", (f"{hoy}%",)).fetchone()
    if existe:
        logging.warning("⚠️ Job semanal ya ejecutado hoy. Abortando por idempotencia.")
        #conn.close()
        #return
    
    df = pd.read_sql_query("SELECT Fecha, Peso_kg, Grasa_Porcentaje, Musculo, FatFreeWeight, Agua, VisFat, BMI, EdadMetabolica FROM pesajes WHERE Fecha >= date('now', '-14 day') ORDER BY Fecha ASC", conn)
    conn.close()

    if df.empty or len(df) < 2:
        return enviar_mensaje_telegram("⚠️ Error: Necesito al menos 2 pesajes recientes para calcular la dieta.")

    df['Fecha'] = pd.to_datetime(df['Fecha']).dt.tz_localize('UTC', ambiguous='NaT', nonexistent='NaT').dt.tz_convert(TZ)
    dato_actual = df.iloc[-1]
    fecha_hace_una_semana = datetime.now(TZ) - timedelta(days=7)
    df['diff_dias'] = (df['Fecha'] - fecha_hace_una_semana).abs()
    dato_anterior = df.loc[df['diff_dias'].idxmin()]
    
    peso_actual, grasa_actual = float(dato_actual['Peso_kg']), float(dato_actual['Grasa_Porcentaje'])
    fat_free_weight = float(dato_actual['FatFreeWeight'])
    musculo_actual_pct = float(dato_actual['Musculo'])
    agua_actual = float(dato_actual['Agua'])
    visfat_actual = float(dato_actual['VisFat'])
    edad_metabolica = int(dato_actual['EdadMetabolica'])
    
    delta_peso = peso_actual - float(dato_anterior['Peso_kg'])
    delta_grasa = grasa_actual - float(dato_anterior['Grasa_Porcentaje'])
    delta_musculo_pct = musculo_actual_pct - float(dato_anterior['Musculo'])
    
    kcal_mult_actual = obtener_estado_actual(ARCHIVO_DB)

    try:
        estado_mimo, shadow_mult, shadow_macros, shadow_razon = evaluar_estado_metabolico(delta_peso, delta_grasa, delta_musculo_pct, kcal_mult_actual)
        logging.info(f"[SHADOW_MIMO] estado={estado_mimo} | kcal_actual={kcal_mult_actual:.1f} | kcal_sugerido={shadow_mult:.1f} | Δpeso={delta_peso:.2f}kg | Δgrasa={delta_grasa:.2f}% | Δmusculo={delta_musculo_pct:.2f}%")
        logging.info(f"[SHADOW_MIMO] razon={shadow_razon}")
    except Exception as e:
        logging.exception(f"[SHADOW_MIMO] Error: {e}")
        estado_mimo, shadow_mult, shadow_macros, shadow_razon = "ERROR", kcal_mult_actual, "Shadow Mode falló.", "Error evaluación MIMO."

    nuevo_mult, razon_control, hubo_cambio = aplicar_ley_de_control(delta_peso, kcal_mult_actual)
    if hubo_cambio: actualizar_estado(ARCHIVO_DB, nuevo_mult)

    calorias = round(peso_actual * nuevo_mult)
    proteina = round(fat_free_weight * 2.2) 
    grasas = round(peso_actual * 0.7) 
    carbs = max(0, round((calorias - (proteina * 4 + grasas * 9)) / 4))

    prompt = f"""Eres mi nutriólogo deportivo y entrenador personal. Diseña un plan de 7 días.
    Perfil: Peso: {peso_actual}kg | Grasa: {grasa_actual}% (Visceral: {visfat_actual}) | Agua: {agua_actual}% | FFM: {fat_free_weight}kg.
    Macros diarios: Kcal: {calorias} | P: {proteina}g | C: {carbs}g | G: {grasas}g.

    REGLAS DE ESTILO DE VIDA (ESTRICTAS):
    1. LUNES, MIERCOLES Y JUEVES (Oficina y Gym Pesado): Salgo 4pm, entreno 45 min en gym, ceno 6pm. Cenas deben ser saciantes. El lonche es SIEMPRE la sobra de la cena anterior.
    2. MARTES Y VIERNES (Home Office y Bebé): Entreno en casa 30 min aprovechando la siesta del bebé. DAME UNA SUGERENCIA DE RUTINA/EJERCICIO EXACTO PARA ESTOS 30 MINUTOS EN CASA.
    3. FIN DE SEMANA: Sugiéreme un tiempo activo o actividad de recuperación.
    4. Desayunos: Ultra-rápidos (<5 mins) y portátiles para comer en el auto camino a la oficina.
    5. Snacks/Frutas: INCLUYE SIEMPRE 1 colación al día basada en FRUTAS FRESCAS para controlar antojos y dar vitaminas, ajustando las porciones de la cena para no pasarnos de calorías.
    
    REGLA ESTRICTA DE FORMATO: Usa SOLO etiquetas <b> e <i> para resaltar. Usa saltos de línea reales (\\n) y guiones (-) para listas. PROHIBIDO usar <br>, <hr>, <ul>, <li>, <h1>, <h2>, <h3>, <p> o cualquier otra etiqueta HTML."""
    
    try:
        client = genai.Client()
        respuesta = client.models.generate_content(model='gemini-2.5-pro', contents=prompt)
        if not respuesta or not hasattr(respuesta, "text") or not respuesta.text.strip(): raise ValueError("Respuesta IA vacía.")
        dieta_html = respuesta.text.strip()
    except Exception as e:
        return enviar_mensaje_telegram("⚠️ Error al contactar IA para generar menú.")

    mensaje_telegram = (
        f"🤖 <b>CONTROL METABÓLICO V4.0</b> 🤖\n\n"
        f"📊 <b>Telemetría Semanal Completa:</b>\n"
        f"• Peso: {peso_actual:.1f} kg (Δ {delta_peso:+.2f} kg)\n"
        f"• Grasa: {grasa_actual:.1f}% (Δ {delta_grasa:+.2f} %)\n"
        f"• Músculo Esquelético: {musculo_actual_pct:.1f}% (Δ {delta_musculo_pct:+.2f} %)\n"
        f"• Masa Libre de Grasa (FFM): {fat_free_weight:.1f} kg\n"
        f"• Agua Corporal: {agua_actual:.1f}%\n"
        f"• Grasa Visceral: {visfat_actual}\n"
        f"• Edad Metabólica: {edad_metabolica} años\n\n"
        f"🧠 <b>Acción del Sistema (SISO):</b>\n"
        f"<i>{razon_control}</i>\n"
        f"Multiplicador actual: {nuevo_mult} kcal/kg\n\n"
        f"🎯 <b>Macros Bio-Ajustados:</b>\n"
        f"Kcal: {calorias} | P: {proteina}g | C: {carbs}g | G: {grasas}g\n\n"
        f"🥗 <b>TU MENÚ Y ENTRENAMIENTO:</b>\n\n{dieta_html}\n\n"
        f"👻 <b>Shadow Mode (MIMO):</b>\n"
        f"• Estado: <b>{estado_mimo}</b>\n"
        f"• Mult. Sugerido: {shadow_mult}\n"
        f"• Diagnóstico: <i>{shadow_razon}</i>"
    )
    enviar_mensaje_telegram(mensaje_telegram)
    
    conexion = sqlite3.connect(ARCHIVO_DB)
    conexion.cursor().execute('''INSERT INTO historico_dietas (fecha, peso, grasa, delta_peso, kcal_mult, calorias, proteina, carbs, grasas, dieta_html, estado_mimo, shadow_mult)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', (datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S"), peso_actual, grasa_actual, delta_peso, nuevo_mult, calorias, proteina, carbs, grasas, dieta_html, estado_mimo, shadow_mult))
    conexion.commit()
    conexion.close()
    logging.info("Job ejecutado exitosamente.")

if __name__ == "__main__":
    ejecutar_job()
