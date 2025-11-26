import os
import pandas as pd
import math
import json
from sqlalchemy import text
from .common import get_engine, log

# CONFIGURACIÓN
TOLERANCIA_KM = 100
EVALUAR_TOP_K = 50

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = math.sin(dLat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dLon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def main():
    log("=== INICIANDO VALIDACIÓN (HISTÓRICO) ===")
    engine = get_engine()
    
    with engine.connect() as conn:
        # 1. Obtener el último Run de Predicción
        q_run = text("SELECT run_id, generated_at, input_max_time FROM prediction_run ORDER BY run_id DESC LIMIT 1")
        run = conn.execute(q_run).mappings().first()
        
        if not run:
            log("No hay predicciones para evaluar.")
            return

        run_id = run['run_id']
        
        # --- NUEVO: VERIFICAR DUPLICADOS ---
        # Antes de calcular nada, preguntamos si ya existe un reporte para este ID
        q_check = text("SELECT id FROM public.validation_realworld WHERE run_id = :rid")
        existing = conn.execute(q_check, {"rid": run_id}).first()
        
        if existing:
            log(f"⚠️ El Run #{run_id} YA FUE VALIDADO anteriormente. Se omite el registro para evitar duplicados.")
            return
        # -----------------------------------
        
        # Calcular ventana de evaluación (Día siguiente al input -> +7 días)
        pred_start = pd.to_datetime(run['input_max_time']) + pd.Timedelta(days=1)
        pred_end = pred_start + pd.Timedelta(days=7) - pd.Timedelta(seconds=1)
        
        log(f"Evaluando Run #{run_id} (Vigencia: {pred_start.date()} al {pred_end.date()})")

        # 2. Obtener Predicciones (Top-K)
        q_preds = text("SELECT rank, lat, lon, place FROM prediction_topk WHERE run_id = :rid")
        df_preds = pd.read_sql(q_preds, conn, params={"rid": run_id})
        
        # 3. Obtener Realidad (Sismos M4+ en la ventana)
        q_real = text("""
            SELECT id, event_time_utc, lat, lon, magnitude, place 
            FROM events_clean 
            WHERE event_time_utc BETWEEN :start AND :end AND magnitude >= 4.0
        """)
        df_real = pd.read_sql(q_real, conn, params={"start": pred_start, "end": pred_end})
        
        total_sismos = len(df_real)
        log(f"Sismos Reales en ventana: {total_sismos}")

        # 4. Algoritmo de Matching
        aciertos_detalle = []
        detectados_count = 0

        if total_sismos > 0:
            for _, sismo in df_real.iterrows():
                detectado = False
                match_info = None
                
                for _, pred in df_preds.iterrows():
                    dist = haversine_km(sismo['lat'], sismo['lon'], pred['lat'], pred['lon'])
                    if dist < TOLERANCIA_KM:
                        detectado = True
                        match_info = f"Alerta #{pred['rank']} a {int(dist)}km"
                        break 
                
                if detectado:
                    detectados_count += 1
                    aciertos_detalle.append({
                        "sismo_lugar": sismo['place'],
                        "magnitud": sismo['magnitude'],
                        "fecha": str(sismo['event_time_utc']),
                        "match": match_info
                    })
        
        recall = (detectados_count / total_sismos * 100) if total_sismos > 0 else 0.0
        
        # 5. GUARDAR EN BASE DE DATOS
        log("Registrando auditoría en 'validation_realworld'...")
        
        with engine.begin() as trans: 
            trans.execute(text("""
                INSERT INTO public.validation_realworld 
                (run_id, window_start, window_end, total_sismos, sismos_detectados, recall_pct, aciertos_json)
                VALUES (:rid, :start, :end, :total, :det, :rec, :json)
            """), {
                "rid": run_id,
                "start": pred_start.date(),
                "end": pred_end.date(),
                "total": total_sismos,
                "det": detectados_count,
                "rec": recall,
                "json": json.dumps(aciertos_detalle)
            })
            
        print("\n" + "="*40)
        print(f" REPORTE GUARDADO EN BD (ID: {run_id})")
        print(f" Ventana: {pred_start.date()} -> {pred_end.date()}")
        print(f" Sismos Totales: {total_sismos}")
        print(f" Detectados:     {detectados_count}")
        print(f" Tasa de Acierto: {recall:.1f}%")
        print("="*40)

if __name__ == "__main__":
    main()