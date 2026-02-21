import os
import json
import requests
import pytz
from datetime import datetime, timedelta

from google import genai
from renpho import RenphoClient 

# ==========================================
# 0. CONFIGURACIÓN BASE Y LOGGING
# ==========================================
TZ = pytz.timezone("America/Phoenix") 
DRY_RUN = os.getenv("DRY_RUN", "0") == "1"

def log(msg):
    timestamp = datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")

# ==========================================
# 1. VALIDACIÓN ESTRICTA DE ENTORNO
# ==========================================
REQUIRED_VARS = [
    "RENPHO_EMAIL", "RENPHO_PASSWORD", 
    "GOOGLE_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"
]

env_vars = {var: os.getenv(var) for var in REQUIRED_VARS}

if not all(env_vars.values()):
    faltantes = [var for var, val in env_vars.items() if not val]
    raise RuntimeError(f"❌ Faltan variables de entorno: {', '.join(faltantes)}")

# ==========================================
# 2. FUNCIONES CORE
# ==========================================

def sanitizar_markdown(texto):
    for ch in ["_", "*", "`", "[", "]", "(", ")"]:
        texto = texto.replace(ch, f"\\{ch}")
    return texto

def obtener_datos_renpho():
    log("🔄 Extrayendo métricas completas de Renpho...")
    try:
        cliente = RenphoClient(env_vars["RENPHO_EMAIL"], env_vars["RENPHO_PASSWORD"])
        mediciones = None
        
        try:
            mediciones = cliente.get_all_measurements()
        except:
            pass
            
        if not mediciones:
            user_id = cliente.user_id
            devices = cliente.get_device_info()
            mac = devices[0].get('mac', '') if devices else ''
            mediciones = cliente.get_measurements(table_name=mac, user_id=user_id, total_count=10)

        if not mediciones:
            raise ValueError("No se encontraron mediciones.")

        # Tomar la más reciente
        mediciones = sorted(mediciones, key=lambda x: x.get("time_stamp", 0), reverse=True)
        u = mediciones[0]
        
        # Mapeo según el escáner de data cruda
        res = {
            "peso": u.get("weight"),
            "grasa": u.get("bodyfat"),
            "agua": u.get("water"),
            "bmi": u.get("bmi"),
            "bmr": u.get("bmr"),
            "edad_metabolica": u.get("bodyage"),
            "grasa_visceral": u.get("visfat"),
            "masa_muscular_kg": u.get("sinew"), # Los 72.6kg reales
            "proteina": u.get("protein"),
            "masa_osea": u.get("bone")
        }
        return res

    except Exception as e:
        raise RuntimeError(f"Error en extracción: {e}")

def manejar_historial(metricas):
    directorio_volumen = "/app/data"
    ruta_archivo = os.path.join(directorio_volumen, "metrics.json")
    
    hoy = str(datetime.now(TZ).date())
    ayer = str(datetime.now(TZ).date() - timedelta(days=1))
    data = {}

    os.makedirs(directorio_volumen, exist_ok=True)

    if os.path.exists(ruta_archivo):
        try:
            with open(ruta_archivo, "r") as f:
                data = json.load(f)
        except:
            pass

    datos_ayer = data.get(ayer)

    if hoy in data:
        log("ℹ️ Idempotencia activa.")
        return datos_ayer, True

    data[hoy] = metricas

    with open(ruta_archivo, "w") as f:
        json.dump(data, f, indent=2)
    
    return datos_ayer, False

def analizar_con_ia(m, datos_ayer):
    log("🧠 Generando análisis experto...")
    client = genai.Client(api_key=env_vars["GOOGLE_API_KEY"])
    
    contexto_ayer = ""
    if datos_ayer:
        diff = round(m['peso'] - datos_ayer['peso'], 2)
        contexto_ayer = f"Ayer el peso fue {datos_ayer['peso']}kg (Variación: {diff}kg)."

    prompt = f"""
    Analiza estas métricas de salud:
    - Peso: {m['peso']}kg | BMI: {m['bmi']}
    - Masa Muscular: {m['masa_muscular_kg']}kg
    - Grasa Corporal: {m['grasa']}% | Visceral: {m['grasa_visceral']}
    - Agua: {m['agua']}% | Proteína: {m['proteina']}%
    - Edad Metabólica: {m['edad_metabolica']} años
    {contexto_ayer}

    Actúa como un experto en recomposición corporal. 
    Responde SOLO en este formato:
    📊 *Análisis Clínico:* (Breve impacto de estos valores)
    🎯 *Acción del Día:* (Nutrición/Entrenamiento según proteína/agua/músculo)
    🔥 *Foco:* (1 frase motivadora)
    """
    
    try:
        respuesta = client.models.generate_content(model='gemini-2.0-flash', contents=prompt)
        return respuesta.text.strip()
    except Exception as e:
        return f"Error IA: {e}"

def enviar_telegram(mensaje):
    if DRY_RUN: return log(mensaje)
    url = f"https://api.telegram.org/bot{env_vars['TELEGRAM_BOT_TOKEN']}/sendMessage"
    requests.post(url, json={"chat_id": env_vars["TELEGRAM_CHAT_ID"], "text": mensaje, "parse_mode": "Markdown"})

def main():
    try:
        m = obtener_datos_renpho()
        ayer, ya_existia = manejar_historial(m)
        
        if ya_existia: return
        
        analisis = analizar_con_ia(m, ayer)
        
        reporte = (
            f"📊 *REPORTE DE SALUD AVANZADO*\n\n"
            f"⚖️ *Peso:* `{m['peso']} kg` (BMI: {m['bmi']})\n"
            f"💪 *Masa Muscular:* `{m['masa_muscular_kg']} kg` 👈\n"
            f"🥓 *Grasa:* `{m['grasa']}%` (Visceral: {m['grasa_visceral']})\n"
            f"💧 *Agua:* `{m['agua']}%` | 🥩 *Prot:* `{m['proteina']}%`\n"
            f"📅 *Edad Metabólica:* `{m['edad_metabolica']} años`\n\n"
            f"🤖 *Análisis IA:*\n{sanitizar_markdown(analisis)}"
        )
        
        enviar_telegram(reporte)
        log("✅ Todo listo.")
    except Exception as e:
        enviar_telegram(f"🔴 Error: `{e}`")

if __name__ == "__main__":
    main()
