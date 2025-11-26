import os
import tensorflow as tf

# --- CORRECCIÓN AQUÍ: Usamos la ruta absoluta completa ---
from scheduler.jobs.forecast_lstm import Config, build_lstm_model_v2

def main():
    # Instanciamos la config para obtener parámetros como window_days
    cfg = Config()
    
    # Tu modelo V4 usa 2 features de entrada: 
    # 1. Cantidad de sismos normalizada
    # 2. Bandera binaria (Sismo Sí/No)
    n_features = 2 
    
    print("Construyendo modelo V4 para inspección...")
    model = build_lstm_model_v2(cfg, n_features)

    # 1. Imprimir Resumen Técnico (Esto va a tus Anexos)
    print("\n" + "="*60)
    print(" RESUMEN DE ARQUITECTURA (model.summary())")
    print("="*60)
    model.summary()
    
    # 2. Intentar generar diagrama visual
    output_path = "/app/artifacts/arquitectura_red.png"
    try:
        tf.keras.utils.plot_model(
            model,
            to_file=output_path,
            show_shapes=True,
            show_layer_names=True,
            expand_nested=True,
            dpi=96
        )
        print(f"\n[EXITO] Diagrama guardado en: {output_path}")
    except ImportError:
        print("\n[AVISO] Librería 'pydot' o 'graphviz' no encontrada.")
        print("No se generó la imagen PNG, pero usa el resumen de texto de arriba para tu tesis.")
    except Exception as e:
        print(f"\n[AVISO] No se pudo generar la imagen: {e}")
        print("Usa el resumen de texto de arriba.")

if __name__ == "__main__":
    main()