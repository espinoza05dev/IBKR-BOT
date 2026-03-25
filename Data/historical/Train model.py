"""
train_model.py
Script de entrenamiento. Ejecutar después de download_data.py.

Uso:
    python train_model.py
"""

from Data.historical.Datapipeline import DataPipeline
from IA.ModelTrainer import ModelTrainer


def main():
    print("\n╔══════════════════════════════════════╗")
    print("║       Model Trainer — AutoTrader      ║")
    print("╚══════════════════════════════════════╝\n")

    SYMBOL     = "TTWO"
    INTERVAL   = "1h"
    START      = "2022-01-01"
    TIMESTEPS  = 500_000    # Aumenta a 500_000+ para mejor rendimiento

    # 1. Datos
    pipeline = DataPipeline(source="yfinance")
    train_df, test_df = pipeline.run(SYMBOL, INTERVAL, START)

    # 2. Entrenar
    trainer = ModelTrainer(symbol=SYMBOL)
    trainer.train(train_df, total_timesteps=TIMESTEPS)

    # 3. Evaluar
    metrics = trainer.evaluate(test_df, episodes=10)

    print("\n╔══════════════════════════════════════╗")
    print("║            Resultados finales          ║")
    print("╚══════════════════════════════════════╝")
    for k, v in metrics.items():
        print(f"  {k:30s}: {v}")

    if metrics.get("win_rate", 0) >= 0.55:
        print("\n✓ Modelo aprobado. Puedes iniciar paper trading.")
        print("  Siguiente paso: python 'IBKR Bot.py'")
    else:
        print("\n⚠ Win rate bajo. Considera:")
        print("  - Aumentar TIMESTEPS a 500_000+")
        print("  - Descargar más datos históricos (START más antiguo)")
        print("  - Agregar más contenido a la KnowledgeBase")


if __name__ == "__main__":
    main()