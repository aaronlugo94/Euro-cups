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

        mediciones = sorted(mediciones, key=lambda x: x.get("time_stamp", 0), reverse=True)
        u = mediciones[0]
        
        return {
            "peso": u.get("weight"), "grasa": u.get("bodyfat"), "agua": u.get("water"),
            "bmi": u.get("bmi"), "bmr": u.get("bmr"), "edad_metabolica": u.get("bodyage"),
            "grasa_visceral": u.get("visfat"), "masa_muscular_kg": u.get("sinew"),
            "musculo_pct": u.get("muscle"), "fat_free_weight": u.get("fatFreeWeight"),
            "proteina": u.get("protein"), "masa_osea": u.get("bone")
        }
    except Exception as e:
        raise RuntimeError(f"Error en extracción: {e}")

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
        fecha_logica = str(datetime.now(TZ).date())
        cur.execute('''
            INSERT OR REPLACE INTO pesajes 
            (Fecha, Peso_kg, Grasa_Porcentaje, Agua, Musculo, BMR, VisFat, BMI, EdadMetabolica, FatFreeWeight)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (fecha_logica, m['peso'], m['grasa'], m['agua'], m['musculo_pct'], 
              m['bmr'], m['grasa_visceral'], m['bmi'], m['edad_metabolica'], m['fat_free_weight']))
        conn.commit()
        conn.close()
    except Exception as e:
        log(f"⚠️ Error SQLite: {e}")

def manejar_historial(metricas):
    directorio_volumen = "/app/data"
    ruta_archivo = os.path.join(directorio_volumen, "metrics.json")
    hoy = str(datetime.now(TZ).date())
    ayer = str(datetime.now(TZ).date() - timedelta(days=1))
    data = {}
    os.makedirs(directorio_volumen, exist_ok=True)

    if os.path.exists(ruta_archivo):
        try:
            with open(ruta_archivo, "r") as f: data = json.load(f)
        except: pass

    datos_ayer = data.get(ayer)

    if hoy in data:
        log("ℹ️ Idempotencia JSON activa. Actualizando solo DB.")
        guardar_en_sqlite(metricas)
        return datos_ayer, True

    data[hoy] = metricas
    with open(ruta_archivo, "w") as f: json.dump(data, f, indent=2)
    guardar_en_sqlite(metricas)
    return datos_ayer, False

def analizar_con_ia(m, datos_ayer):
    log("🧠 Generando análisis clínico...")
    client = genai.Client(api_key=env_vars["GOOGLE_API_KEY"])
    
    # 🧠 INYECCIÓN DE CONTEXTO TEMPORAL PARA LA IA
    contexto_ayer = ""
    if datos_ayer:
        contexto_ayer = (
            f"\n--- COMPARATIVA VS AYER ---\n"
            f"Peso: {datos_ayer['peso']}kg -> {m['peso']}kg (Variación: {m['peso'] - datos_ayer['peso']:+.2f}kg)\n"
            f"Grasa: {datos_ayer['grasa']}% -> {m['grasa']}%\n"
            f"Músculo: {datos_ayer['masa_muscular_kg']}kg -> {m['masa_muscular_kg']}kg\n"
            f"Agua: {datos_ayer['agua']}% -> {m['agua']}%\n"
        )

    prompt = f"""Analiza estas métricas de salud:
    - Peso: {m['peso']}kg | BMI: {m['bmi']}
    - Masa Muscular: {m['masa_muscular_kg']}kg
    - Grasa Corporal: {m['grasa']}% | Visceral: {m['grasa_visceral']}
    - Agua: {m['agua']}% | Proteína: {m['proteina']}%
    - Edad Metabólica: {m['edad_metabolica']} años
    {contexto_ayer}
    Actúa como experto en recomposición corporal. Analiza la comparativa con ayer: ¿es mejora real, empeoramiento o simple ruido hídrico? Responde SOLO en este formato estricto HTML:
    <b>📊 Análisis Diario:</b> (Impacto y evaluación del cambio)\n\n
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
    
    # 🧹 FILTRO SANITARIO AGRESIVO ACTUALIZADO
    mensaje = mensaje.replace("<br>", "\n").replace("<br/>", "\n").replace("<ul>", "").replace("</ul>", "").replace("<li>", "• ").replace("</li>", "\n")
    mensaje = mensaje.replace("<hr>", "---").replace("<hr/>", "---").replace("<p>", "").replace("</p>", "\n").replace("<strong>", "<b>").replace("</strong>", "</b>")
    mensaje = mensaje.replace("<h1>", "").replace("</h1>", "\n").replace("<h2>", "").replace("</h2>", "\n").replace("<h3>", "").replace("</h3>", "\n")
    
    payload = {"chat_id": env_vars["TELEGRAM_CHAT_ID"], "text": mensaje, "parse_mode": "HTML"}
    
    res = requests.post(url, json=payload)
    if res.status_code != 200:
        log(f"⚠️ Telegram rechazó el HTML. Fallback a texto plano... Error: {res.text}")
        del payload["parse_mode"]
        res2 = requests.post(url, json=payload)
        if res2.status_code != 200:
            log(f"⚠️ Error CRÍTICO en fallback: {res2.text}")

# 🚦 FUNCIÓN HELPER PARA LOS SEMÁFOROS
def calcular_delta(hoy, ayer, invert_colors=False):
    if ayer is None: return ""
    diff = hoy - ayer
    if abs(diff) < 0.05: return " ⚪" # Neutral
    
    if invert_colors: # Para Peso y Grasa (Bajar es Bueno 🟢)
        emoji = "🟢" if diff < 0 else "🔴"
    else:             # Para Músculo y Agua (Subir es Bueno 🟢)
        emoji = "🟢" if diff > 0 else "🔴"
        
    return f" (Δ {diff:+.1f} {emoji})"

def ejecutar_diario():
    try:
        m = obtener_datos_renpho()
        ayer, ya_existia = manejar_historial(m)
        
        # ⚠️ Ponle el hashtag a la línea de abajo (# if ya_existia:...) SOLO si quieres probarlo AHORITA MISMO.
        # Recuerda quitárselo cuando termines la prueba para que vuelva a quedar protegido.
        if ya_existia: return True 
        
        # Calculamos visuales
        d_peso = calcular_delta(m['peso'], ayer['peso'], invert_colors=True) if ayer else ""
        d_grasa = calcular_delta(m['grasa'], ayer['grasa'], invert_colors=True) if ayer else ""
        d_musc = calcular_delta(m['masa_muscular_kg'], ayer['masa_muscular_kg'], invert_colors=False) if ayer else ""
        d_agua = calcular_delta(m['agua'], ayer['agua'], invert_colors=False) if ayer else ""

        analisis = analizar_con_ia(m, ayer)
        
        reporte = (
            f"📊 <b>REPORTE DE SALUD AVANZADO</b>\n\n"
            f"⚖️ <b>Peso:</b> {m['peso']} kg{d_peso}\n"
            f"💪 <b>Masa Muscular:</b> {m['masa_muscular_kg']} kg{d_musc}\n"
            f"🥓 <b>Grasa:</b> {m['grasa']}%{d_grasa} (Visceral: {m['grasa_visceral']})\n"
            f"💧 <b>Agua:</b> {m['agua']}%{d_agua}\n"
            f"📅 <b>Edad Metabólica:</b> {m['edad_metabolica']} años\n\n"
            f"🤖 <b>Análisis IA:</b>\n{analisis}"
        )
        enviar_telegram(reporte)
        log("✅ Flujo diario completado.")
        return True
    except Exception as e:
        enviar_telegram(f"🔴 <b>Error Crítico en Ingesta:</b> {e}")
        return False

if __name__ == "__main__":
    ejecutar_diario()
