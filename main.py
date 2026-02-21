import os
import json
import requests
import pytz
from datetime import datetime, timedelta

# Nuevas librerías con las rutas EXACTAS
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
    """Blindaje total para Telegram."""
    for ch in ["_", "*", "`", "[", "]", "(", ")"]:
        texto = texto.replace(ch, f"\\{ch}")
    return texto

def obtener_datos_renpho():
    log("🔄 Ejecutando sonda de diagnóstico en RenphoClient...")
    try:
        # Iniciamos sesión
        cliente = RenphoClient(env_vars["RENPHO_EMAIL"], env_vars["RENPHO_PASSWORD"])
        
        # Escaneamos las entrañas del cliente
        metodos = [m for m in dir(cliente) if callable(getattr(cliente, m)) and not m.startswith('_')]
        propiedades = [m for m in dir(cliente) if not callable(getattr(cliente, m)) and not m.startswith('_')]
        
        # Armamos un reporte y forzamos un error para que llegue a Telegram
        diagnostico = (
            f"🔍 *Diagnóstico de la API:*\n\n"
            f"🛠️ *Métodos:* `{metodos}`\n\n"
            f"📦 *Propiedades:* `{propiedades}`"
        )
        raise ValueError(diagnostico)

    except Exception as e:
        # Lo pasamos hacia arriba para que el main lo envíe
        raise RuntimeError(f"{e}")

def manejar_historial(peso, grasa, musculo):
    directorio_volumen = "/app/data"
    ruta_archivo = os.path.join(directorio_volumen, "metrics.json")
    log(f"💾 Gestionando histórico en: {ruta_archivo}")
    
    hoy_date = datetime.now(TZ).date()
    hoy = str(hoy_date)
    ayer = str(hoy_date - timedelta(days=1))
    data = {}

    os.makedirs(directorio_volumen, exist_ok=True)

    if os.path.exists(ruta_archivo):
        try:
            with open(ruta_archivo, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            log("⚠️ Archivo JSON corrupto. Se sobrescribirá.")

    datos_ayer = data.get(ayer)

    if hoy in data:
        log("ℹ️ Ya existe una medición para hoy, omitiendo escritura.")
        return datos_ayer, True

    data[hoy] = {"peso": peso, "grasa": grasa, "musculo": musculo}

    try:
        with open(ruta_archivo, "w") as f:
            json.dump(data, f, indent=2)
        log("✅ Histórico actualizado correctamente.")
    except Exception as e:
        raise RuntimeError(f"Error al escribir en el Volumen: {e}")

    return datos_ayer, False

def analizar_con_ia(peso, grasa, musculo, datos_ayer):
    log("🧠 Ejecutando prompt en Gemini...")
    client = genai.Client(api_key=env_vars["GOOGLE_API_KEY"])
    
    comparativa = ""
    if datos_ayer:
        diff_peso = round(peso - datos_ayer['peso'], 2)
        signo = "+" if diff_peso > 0 else ""
        comparativa = f"\nContexto histórico: Ayer pesaste {datos_ayer['peso']} kg (Diferencia: {signo}{diff_peso} kg)."

    prompt = f"""
    Datos corporales de hoy:
    - Peso: {peso} kg
    - Grasa corporal: {grasa} %
    - Masa muscular: {musculo} kg{comparativa}

    Actúa como entrenador y nutriólogo.
    Responde SOLO en este formato exacto, sin texto adicional:

    📊 Diagnóstico (máx 2 líneas, objetivo y directo)
    🎯 Acción concreta hoy (1 frase)
    🔥 Motivación breve (1 frase)
    """
    
    try:
        respuesta = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=prompt
        )
        return respuesta.text.strip()
    except Exception as e:
        raise RuntimeError(f"Fallo en generación de IA: {e}")

def enviar_telegram(mensaje):
    if DRY_RUN:
        log(f"🛑 DRY_RUN ACTIVO:\n{mensaje}")
        return

    log("📲 Transmitiendo a Telegram...")
    url = f"https://api.telegram.org/bot{env_vars['TELEGRAM_BOT_TOKEN']}/sendMessage"
    r = requests.post(
        url,
        json={"chat_id": env_vars["TELEGRAM_CHAT_ID"], "text": mensaje, "parse_mode": "Markdown"},
        timeout=10
    )

    if r.status_code != 200:
        raise RuntimeError(f"Error HTTP {r.status_code} en Telegram: {r.text}")

# ==========================================
# 3. ORQUESTADOR PRINCIPAL
# ==========================================

def main():
    try:
        # Esto va a "fallar" a propósito para darnos el diagnóstico
        peso, grasa, musculo = obtener_datos_renpho()
        
        datos_ayer, ya_existia = manejar_historial(peso, grasa, musculo)
        
        if ya_existia:
            log("ℹ️ Pipeline detenido por idempotencia.")
            return
        
        analisis_raw = analizar_con_ia(peso, grasa, musculo, datos_ayer)
        analisis_seguro = sanitizar_markdown(analisis_raw)
        
        mensaje_final = (
            f"📈 *Reporte Diario de Composición*\n\n"
            f"⚖️ Peso: `{peso} kg`\n"
            f"🥓 Grasa: `{grasa} %`\n"
            f"💪 Músculo: `{musculo} kg`\n\n"
            f"🤖 *Diagnóstico IA:*\n{analisis_seguro}"
        )
        
        enviar_telegram(mensaje_final)
        log("✅ Pipeline completado exitosamente.")

    except Exception as e:
        # El error de diagnóstico caerá aquí y te llegará por Telegram
        error_msg = f"🔴 *Diagnóstico del Sistema*\n{str(e)}"
        log(error_msg)
        try:
            enviar_telegram(error_msg)
        except:
            log("Fallo catastrófico con Telegram.")

if __name__ == "__main__":
    main()
