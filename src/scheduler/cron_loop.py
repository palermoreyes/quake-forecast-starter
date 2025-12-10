"""
cron_loop.py
------------
Bucle principal INTELIGENTE del scheduler. 
Ejecuta las tareas de extracción y carga del IGP sincronizándose 
siempre al minuto 05 de cada hora para garantizar datos frescos.
"""

import time
from datetime import datetime, timedelta
from .jobs import fetch_igp_extract, fetch_igp_load
from .jobs.common import log

def get_seconds_until_next_tick(minute_target=5):
    """Calcula cuántos segundos faltan para el próximo minuto XX:05."""
    now = datetime.now()
    # Objetivo: la hora actual pero en el minuto 05
    target = now.replace(minute=minute_target, second=0, microsecond=0)
    
    # Si ya pasamos el minuto 05 (ej: son las 10:20), apuntamos a la siguiente hora (11:05)
    if target <= now:
        target += timedelta(hours=1)
    
    wait_seconds = (target - now).total_seconds()
    return wait_seconds, target

if __name__ == "__main__":
    log("Scheduler INTELIGENTE iniciado (Sincronizado al minuto 05 de cada hora).")

    # 1. Ejecución inmediata al arrancar 
    # (Vital por si reinicias a las 23:50, para que cargue lo pendiente YA)
    try:
        log("⚡ Ejecución inicial de arranque...")
        fetch_igp_extract.main()
        fetch_igp_load.main()
        log("Ciclo inicial completado.")
    except Exception as e:
        log(f"Error en el ciclo inicial: {e}")

    # 2. Bucle infinito sincronizado
    while True:
        # Calcular cuánto falta para el siguiente XX:05
        wait_sec, next_time = get_seconds_until_next_tick(minute_target=5)
        
        log(f"😴 Durmiendo {int(wait_sec)}s hasta la próxima sincronización: {next_time.strftime('%H:%M:%S')}")
        
        # Dormir exactamente lo necesario
        time.sleep(wait_sec)
        
        # Al despertar (serán las XX:05), ejecutamos
        try:
            log("⏰ Inicio del ciclo programado: Extracción y Carga IGP")
            fetch_igp_extract.main()
            fetch_igp_load.main()
            log("Ciclo completado correctamente.")
        except Exception as e:
            log(f"Error en el ciclo principal: {e}")