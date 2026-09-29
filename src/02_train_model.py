# src/02_train_model.py

import pandas as pd
import xgboost as xgb
import pickle
import logging
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, make_scorer
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def wmape_ops(y_true, y_pred):
    y_pred_clipped = np.clip(y_pred, a_min=0, a_max=None)
    if np.sum(y_true) == 0:
        return 0
    return (np.sum(np.abs(y_true - y_pred_clipped)) / np.sum(y_true)) * 100


def entrenar_modelo():
    logging.info("🧠 Iniciando Entrenamiento Multidimensional (Agencia + Producto)...")
    df = pd.read_csv("src/dataset_procesado.csv")

    # Purga de valores extremos de operaciones (ahora a nivel producto)
    z_scores = np.abs(stats.zscore(df["ColocacionNumReal"]))
    df = df[(z_scores < 3)]

    # 🔥 AHORA INCLUIMOS EL PRODUCTO EN LA IA
    df_procesado = pd.get_dummies(
        df, columns=["IdSAgencia", "IdTipoProducto", "MesDelAnio", "DiaSemana"]
    )
    features = [
        c
        for c in df_procesado.columns
        if c not in ["Fecha", "ColocacionMontoReal", "ColocacionNumReal"]
    ]

    with open("src/columnas_entrenamiento.pkl", "wb") as f:
        pickle.dump(features, f)

    X = df_procesado[features]
    y_ops = df_procesado["ColocacionNumReal"]

    X_train, X_test, y_train_ops, y_test_ops = train_test_split(
        X, y_ops, test_size=0.15, shuffle=False
    )
    _, X_test_df = train_test_split(df_procesado, test_size=0.15, shuffle=False)
    y_test_monto = X_test_df["ColocacionMontoReal"]

    parametros = {
        "n_estimators": [500, 800],
        "learning_rate": [0.02, 0.05],
        "max_depth": [4, 5],
        "subsample": [0.85, 0.9],
        "colsample_bytree": [0.85, 0.9],
        "reg_alpha": [1, 2],
        "reg_lambda": [5, 10],
    }

    logging.info("⚙️ Entrenando IA de Operaciones por Producto...")
    modelo_base_ops = xgb.XGBRegressor(objective="count:poisson", random_state=42)
    torneo_ops = RandomizedSearchCV(
        modelo_base_ops,
        parametros,
        n_iter=20,
        scoring=make_scorer(wmape_ops, greater_is_better=False),
        cv=3,
        n_jobs=-1,
        random_state=42,
    )
    torneo_ops.fit(X_train, y_train_ops)
    mejor_modelo_ops = torneo_ops.best_estimator_

    pred_ops = np.clip(mejor_modelo_ops.predict(X_test), 0, None)

    logging.info("⚙️ Calculando Ticket Ancla Histórico por Producto...")
    ticket_ancla = X_test_df["Ticket_Promedio_30d"]
    monto_predicho = pred_ops * ticket_ancla

    # Agrupamos la realidad y la predicción POR AGENCIA Y FECHA para ver el error real del negocio
    wmape_global = (
        np.sum(np.abs(y_test_monto - monto_predicho)) / np.sum(y_test_monto)
    ) * 100
    mae_global = mean_absolute_error(y_test_monto, monto_predicho)

    logging.info(f"   📉 Error Financiero Promedio (MAE): S/ {mae_global:,.2f}")
    logging.info(f"   🏆 PRECISION GLOBAL MULTIDIMENSIONAL: {100 - wmape_global:.2f}%")

    with open("src/modelo_ops.pkl", "wb") as f:
        pickle.dump(mejor_modelo_ops, f)
    logging.info("💾 Modelo Corporativo guardado con éxito.")


if __name__ == "__main__":
    entrenar_modelo()
