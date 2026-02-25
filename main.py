import os
import logging
import pytz
from datetime import datetime
import daily_renpho
import job_dieta

# Unificamos el formato de logs para que Railway lo lea perfecto
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
TZ = pytz.timezone(os.getenv("TZ", "America/Phoenix"))

def main():
    logging.info("🚀 Iniciando Sistema de Control Autónomo...")
    
    try:
        logging.info("--- FASE 1: Ingesta Biométrica Diaria ---")
        # Si ya_existia es True, devuelve True. Si extrae bien, devuelve True. Si falla, devuelve False.
        ingesta_exitosa = daily_renpho.ejecutar_diario()
        
        hoy = datetime.now(TZ)
        if hoy.weekday() == 6: # 6 = Domingo
            logging.info("--- FASE 2: Domingo detectado. Evaluando Lazo Cerrado Metabólico ---")
            if ingesta_exitosa:
                job_dieta.ejecutar_job()
            else:
                logging.warning("⚠️ FASE 2 abortada: La ingesta diaria falló. Se protege el cálculo del menú.")
        else:
            logging.info("--- FASE 2: Omitida. El ajuste de dieta se ejecuta los domingos. ---")
            
        logging.info("✅ Ejecución del orquestador finalizada correctamente.\n")
        
    except Exception as e:
        logging.error(f"❌ Error CRÍTICO en el orquestador principal: {e}", exc_info=True)

if __name__ == "__main__":
    main()
