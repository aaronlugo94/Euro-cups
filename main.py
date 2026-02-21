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
    log("🔄 Extrayendo datos de Renpho (Modo Escáner Profundo)...")
    try:
        cliente = RenphoClient(env_vars["RENPHO_EMAIL"], env_vars["RENPHO_PASSWORD"])
        mediciones = None
        
        try:
            mediciones = cliente.get_all_measurements()
        except Exception as e:
            log(f"get_all_measurements() falló: {e}. Pasando a Intento 2...")
            
        if not mediciones:
            user_id = cliente.user_id
            devices = cliente.get_device_info()
            try:
                if isinstance(devices, list) and len(devices) > 0:
                    mac_dispositivo = devices[0].get('mac', '')
                    mediciones = cliente.get_measurements(table_name=mac_dispositivo, user_id=user_id, total_count=10)
                else:
                    raise ValueError("No se encontraron dispositivos.")
            except Exception as e2:
                raise RuntimeError(f"Fallo al extraer.\nuser_id: `{user_id}`\ndevices: `{devices}`")

        if not mediciones:
            raise ValueError("La API devolvió una lista vacía.")

        # Ordenar por el más reciente
        mediciones = sorted(mediciones, key=lambda x: x.get("time_stamp", 0), reverse=True)
        ultima = mediciones[0]
        
        # 🔥 EL TRUCO: Forzamos un error intencional enviando todo el diccionario convertido a texto
        data_cruda = json.dumps(ultima, indent=2)
        raise ValueError(f"🚨 DATA CRUDA DESCUBIERTA:\n{data_cruda}")

    except Exception as e:
        raise RuntimeError(f"{e}")

def manejar_historial(peso, grasa, musculo):
    # En este modo escáner, el código no llegará aquí porque explotará arriba a propósito.
    return None, False

def analizar_con_ia(peso, grasa, musculo, datos_ayer):
    # En este modo escáner, el código no llegará aquí.
    return "Dummy"

def enviar_telegram(mensaje):
    if DRY_RUN:
        log(f"🛑 DRY_RUN ACTIVO:\n{mensaje}")
        return

    log("📲 Transmitiendo a Telegram...")
    url = f"https://api.telegram.org/bot{env_vars['TELEGRAM_BOT_TOKEN']}/sendMessage"
    
    # Aquí usamos HTML temporalmente porque el JSON crudo puede romper el Markdown de Telegram
    r = requests.post(
        url,
        json={"chat_id": env_vars["TELEGRAM_CHAT_ID"], "text": mensaje},
        timeout=10
    )

    if r.status_code != 200:
        raise RuntimeError(f"Error HTTP {r.status_code} en Telegram: {r.text}")

# ==========================================
# 3. ORQUESTADOR PRINCIPAL
# ==========================================

def main():
    try:
        # Esto va a fallar intencionalmente y nos mandará la data cruda
        peso, grasa, musculo = obtener_datos_renpho()
        
    except Exception as e:
        error_msg = f"🔴 *Reporte del Escáner Renpho*\n\n{str(e)}"
        log("Atrapado el escáner, enviando a Telegram...")
        try:
            enviar_telegram(error_msg)
        except Exception as telegram_error:
            log(f"Fallo catastrófico con Telegram: {telegram_error}")

if __name__ == "__main__":
    main()
