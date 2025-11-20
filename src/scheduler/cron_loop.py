"""
cron_loop.py
------------
Bucle principal del scheduler. 
Ejecuta las tareas de extracción y carga del IGP de forma periódica.
"""

import time
from .jobs import fetch_igp_extract, fetch_igp_load
from .jobs.common import log

if __name__ == "__main__":
    log("Scheduler iniciado (intervalo de 120 Minutos).")

    while True:
        try:
            log("Inicio del ciclo: Extracción y Carga IGP")
            fetch_igp_extract.main()
            fetch_igp_load.main()
            log("Ciclo completado correctamente.")
        except Exception as e:
            log(f"Error en el ciclo principal: {e}")
        log("Esperando 7200 segundos (120 Minutos) antes del siguiente ciclo.")
        time.sleep(7200)
