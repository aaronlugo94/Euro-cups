import os
import json
import sqlite3
import requests
import pytz
import time
from datetime import datetime, timedelta
from google import genai
from renpho import RenphoClient 

TZ = pytz.timezone(os.getenv("TZ", "America/Phoenix")) 
DRY_RUN = os.getenv("DRY_RUN", "0") == "1"

def log(msg):
    timestamp = datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")

REQUIRED_VARS = ["RENPHO_EMAIL", "RENPHO_PASSWORD", "GOOGLE_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"]
env_vars = {var: os.getenv(var) for var in REQUIRED_VARS}
if not all(env_vars.values()):
    raise RuntimeError(f"❌ Faltan variables de entorno: {', '.join([v for v, k in env_vars.items() if not k])}")

def obtener_datos_renpho():
    log("🔄 Extrayendo telemetría de Renpho...")
    try:
        cliente = RenphoClient(env_vars["RENPHO_EMAIL"], env_vars["RENPHO_PASSWORD"])
        mediciones = None
        try: mediciones = cliente.get_all_measurements()
        except: pass
            
        if not mediciones:
            user_id = cliente.user_id
            devices = cliente.get_device_info()
            mac = devices[0].get('mac', '') if devices else ''
            mediciones = cliente.get_measurements(table_name=mac, user_id=user_id, total_count=10)

        if not mediciones: raise ValueError("No se encontraron mediciones.")

        # Tomar la medición más reciente
        mediciones = sorted(mediciones, key=lambda x: x.get("time_stamp", 0), reverse=True)
        u = mediciones[0]
        
        # Extraemos el timestamp exacto para saber si es un pesaje nuevo
        timestamp_exacto = u.get("time_stamp")
        fecha_logica = datetime.fromtimestamp(timestamp_exacto, TZ).strftime('%Y-%m-%d')
        
        return {
            "time_stamp": timestamp_exacto, "fecha_str": fecha_logica,
            "peso": u.get("weight"), "grasa": u.get("bodyfat"), "agua": u.get("water"),
            "bmi": u.get("bmi"), "bmr": u.get("bmr"), "edad_metabolica": u.get("bodyage"),
            "grasa_visceral": u.get("visfat"), "masa_muscular_kg": u.get("sinew"),
            "musculo_pct": u.get("muscle"), "fat_free_weight": u.get("fatFreeWeight"),
            "proteina": u.get("protein"), "masa_osea": u.get("bone")
        }
    except Exception as e:
        raise RuntimeError(f"Error en extracción: {e}")

def es_pesaje_nuevo(timestamp_actual):
    ruta_estado = "/app/data/ultimo_pesaje.txt"
    os.makedirs(os.path.dirname(ruta_estado), exist_ok=True)
    if os.path.exists(ruta_estado):
        with open(ruta_estado, "r") as f:
            ultimo_ts = f.read().strip()
            if str(timestamp_actual) == ultimo_ts:
                return False # Ya procesamos este pesaje
    return True # Es un pesaje completamente nuevo

def marcar_como_procesado(timestamp_actual):
    ruta_estado = "/app/data/ultimo_pesaje.txt"
    with open(ruta_estado, "w") as f:
        f.write(str(timestamp_actual))

def guardar_en_sqlite(m):
    log("💾 Persistiendo en SQLite (Single Source of Truth)...")
    db_path = "/app/data/mis_datos_renpho.db"
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute('''
            CREATE TABLE IF NOT EXISTS pesajes (
                Fecha TEXT PRIMARY KEY, Peso_kg REAL, Grasa_Porcentaje REAL, Agua REAL, 
                Musculo REAL, BMR INTEGER, VisFat REAL, BMI REAL, EdadMetabolica INTEGER, FatFreeWeight REAL
            )
        ''')
        cur.execute('''
            INSERT OR REPLACE INTO pesajes 
            (Fecha, Peso_kg, Grasa_Porcentaje, Agua, Musculo, BMR, VisFat, BMI, EdadMetabolica, FatFreeWeight)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (m['fecha_str'], m['peso'], m['grasa'], m['agua'], m['musculo_pct'], 
              m['bmr'], m['grasa_visceral'], m['bmi'], m['edad_metabolica'], m['fat_free_weight']))
        conn.commit()
        conn.close()
    except Exception as e:
        log(f"⚠️ Error SQLite: {e}")

def obtener_comparativa_semana(fecha_actual_str):
    db_path = "/app/data/mis_datos_renpho.db"
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # Buscar el pesaje más cercano a hace 7 días (mínimo 6 días atrás para ser precisos)
        cursor.execute('''
            SELECT Peso_kg, Grasa_Porcentaje, Musculo, Agua 
            FROM pesajes 
            WHERE Fecha <= date(?, '-6 day') 
            ORDER BY Fecha DESC LIMIT 1
        ''', (fecha_actual_str,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return {"peso": row[0], "grasa": row[1], "musculo_pct": row[2], "agua": row[3]}
    except: pass
    return None

def analizar_con_ia(m, semana_pasada):
    log("🧠 Generando análisis clínico...")
    client = genai.Client(api_key=env_vars["GOOGLE_API_KEY"])
    
    contexto_comparativo = ""
    if semana_pasada:
        contexto_comparativo = (
            f"\n--- COMPARATIVA VS HACE UNA SEMANA ---\n"
            f"Peso: {semana_pasada['peso']}kg -> {m['peso']}kg (Variación: {m['peso'] - semana_pasada['peso']:+.2f}kg)\n"
            f"Grasa: {semana_pasada['grasa']}% -> {m['grasa']}%\n"
            f"Músculo: {semana_pasada['musculo_pct']}% -> {m['musculo_pct']}%\n"
            f"Agua: {semana_pasada['agua']}% -> {m['agua']}%\n"
        )

    prompt = f"""Analiza estas métricas de salud:
    - Peso: {m['peso']}kg | BMI: {m['bmi']}
    - Músculo Esquelético: {m['musculo_pct']}%
    - Grasa Corporal: {m['grasa']}% | Visceral: {m['grasa_visceral']}
    - Agua: {m['agua']}% | Proteína: {m['proteina']}%
    - Edad Metabólica: {m['edad_metabolica']} años
    {contexto_comparativo}
    Actúa como experto en recomposición corporal. Analiza la comparativa de 7 días: ¿Vamos en la dirección correcta? Responde SOLO en este formato estricto HTML:
    <b>📊 Análisis de la Semana:</b> (Impacto y evaluación de la tendencia de 7 días)\n\n
    <b>🎯 Acción del Día:</b> (Nutrición/Entrenamiento)\n\n
    <i>🔥 Foco: (1 frase motivadora)</i>
    REGLA ESTRICTA: Usa SOLO etiquetas <b> e <i> para resaltar. PROHIBIDO usar <br>, <hr>, <ul>, <li>, <h1>, <h2>, <h3> o cualquier otra etiqueta."""
    
    for intento in range(3):
        try:
            respuesta = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
            if respuesta and respuesta.text: return respuesta.text.strip()
        except Exception as e:
            if intento == 2: return f"<i>⚠️ Error conectando con motor analítico: {e}</i>"
            time.sleep(2)

def enviar_telegram(mensaje):
    if DRY_RUN: return log(f"DRY RUN: {mensaje}")
    url = f"https://api.telegram.org/bot{env_vars['TELEGRAM_BOT_TOKEN']}/sendMessage"
    
    mensaje = mensaje.replace("<br>", "\n").replace("<br/>", "\n").replace("<ul>", "").replace("</ul>", "").replace("<li>", "• ").replace("</li>", "\n")
    mensaje = mensaje.replace("<hr>", "---").replace("<hr/>", "---").replace("<p>", "").replace("</p>", "\n").replace("<strong>", "<b>").replace("</strong>", "</b>")
    mensaje = mensaje.replace("<h1>", "").replace("</h1>", "\n").replace("<h2>", "").replace("</h2>", "\n").replace("<h3>", "").replace("</h3>", "\n")
    
    payload = {"chat_id": env_vars["TELEGRAM_CHAT_ID"], "text": mensaje, "parse_mode": "HTML"}
    res = requests.post(url, json=payload)
    if res.status_code != 200:
        log(f"⚠️ Telegram rechazó el HTML. Fallback a texto plano...")
        del payload["parse_mode"]
        requests.post(url, json=payload)

def calcular_delta(hoy, ayer, invert_colors=False):
    if ayer is None: return ""
    diff = hoy - ayer
    if abs(diff) < 0.05: return " ⚪" 
    
    if invert_colors: # Para Peso y Grasa (Bajar es Bueno 🟢)
        emoji = "🟢" if diff < 0 else "🔴"
    else:             # Para Músculo y Agua (Subir es Bueno 🟢)
        emoji = "🟢" if diff > 0 else "🔴"
        
    return f" (Δ {diff:+.1f} {emoji})"

def ejecutar_diario():
    try:
        m = obtener_datos_renpho()
        
        # EL GUARDIA DE SEGURIDAD: Revisa si ya te habías pesado
        if not es_pesaje_nuevo(m['time_stamp']):
            log("💤 No hay pesajes nuevos en la báscula. Ignorando silenciosamente.")
            return True # Retorna True para no bloquear el Job de Dieta en Domingo
        
        # Si llegamos aquí, ¡TE ACABAS DE PESAR!
        log("🚀 ¡Nuevo pesaje detectado! Procesando comparativa...")
        guardar_en_sqlite(m)
        semana_pasada = obtener_comparativa_semana(m['fecha_str'])
        
        # Calculamos visuales
        d_peso = calcular_delta(m['peso'], semana_pasada['peso'], invert_colors=True) if semana_pasada else ""
        d_grasa = calcular_delta(m['grasa'], semana_pasada['grasa'], invert_colors=True) if semana_pasada else ""
        d_musc = calcular_delta(m['musculo_pct'], semana_pasada['musculo_pct'], invert_colors=False) if semana_pasada else ""
        d_agua = calcular_delta(m['agua'], semana_pasada['agua'], invert_colors=False) if semana_pasada else ""

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
        
        # Guardamos la estampa de tiempo para no volver a mandarlo hoy
        marcar_como_procesado(m['time_stamp'])
        log("✅ Flujo de pesaje completado y notificado.")
        return True
    except Exception as e:
        log(f"🔴 Error Crítico en Ingesta: {e}")
        return False

if __name__ == "__main__":
    ejecutar_diario()
