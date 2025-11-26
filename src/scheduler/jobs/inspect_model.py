import os
import tensorflow as tf

from scheduler.jobs.forecast_lstm import Config


def main():
    cfg = Config()

    # Buscar el archivo .keras más reciente (modelo entrenado real)
    models_dir = cfg.models_dir
    models = [
        f for f in os.listdir(models_dir) if f.endswith(".keras")
    ]
    if not models:
        raise RuntimeError(
            f"No se encontraron modelos .keras en {models_dir}"
        )

    latest_model = max(
        models,
        key=lambda f: os.path.getctime(
            os.path.join(models_dir, f)
        ),
    )
    model_path = os.path.join(models_dir, latest_model)

    print("\n" + "=" * 60)
    print(" INSPECCIÓN DE MODELO ENTRENADO")
    print("=" * 60)
    print(f"Modelo cargado desde: {model_path}")

    # Cargar modelo entrenado
    model = tf.keras.models.load_model(model_path)

    # 1. Resumen técnico completo (para anexos de tesis)
    print("\n" + "=" * 60)
    print(" RESUMEN DE ARQUITECTURA (model.summary())")
    print("=" * 60)
    model.summary()

    # 2. Intentar generar diagrama visual
    output_path = "/app/artifacts/arquitectura_red_v3.png"
    try:
        tf.keras.utils.plot_model(
            model,
            to_file=output_path,
            show_shapes=True,
            show_layer_names=True,
            expand_nested=True,
            dpi=120,
        )
        print(f"\n[EXITO] Diagrama de arquitectura guardado en: {output_path}")
    except ImportError:
        print(
            "\n[AVISO] Falta 'pydot' o 'graphviz'. "
            "No se generó el PNG, pero el summary de texto es suficiente para la tesis."
        )
    except Exception as e:
        print(f"\n[AVISO] No se pudo generar la imagen: {e}")
        print(
            "Revisa dependencias de graphviz. Mientras tanto, usa el summary de texto."
        )


if __name__ == "__main__":
    main()
