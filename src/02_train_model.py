import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import joblib
import logging
import os
from config import get_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def run_training():
    logging.info("🧠 Iniciando Entrenamiento V6 (Nivel Agencia Puro)...")
    engine = get_engine()

    df = pd.read_sql(
        "SELECT * FROM ml.fct_features_entrenamiento ORDER BY Fecha", engine
    )
    engine.dispose()

    # ¡ELIMINAMOS EL PRODUCTO DE LA AGRUPACIÓN!
    columnas_agrupacion = [
        "Fecha",
        "Anio",
        "Mes",
        "Dia",
        "DiaSemana",
        "EsQuincena",
        "EsFinDeMes",
        "EsFeriadoNacional",
        "EsFiestaLocal",
        "IdSAgencia",
    ]

    # Agrupamos toda la agencia junta
    df_agrupado = (
        df.groupby(columnas_agrupacion)
        .agg(
            {
                "ColocacionMontoReal": "sum",
                "AnalistasActivos": "max",  # Tomamos el máximo para no duplicar personal
                "ExperienciaPromedioMeses": "mean",
                "ColocacionesMicro": "sum",
                "ColocacionesMacro": "sum",
            }
        )
        .reset_index()
    )

    df_agrupado = df_agrupado.sort_values(by=["IdSAgencia", "Fecha"])

    # Línea base real de la agencia (Promedio de 21 días)
    df_agrupado["LineaBase_30d"] = df_agrupado.groupby("IdSAgencia")[
        "ColocacionMontoReal"
    ].transform(lambda x: x.shift(1).rolling(window=21, min_periods=3).mean())
    df_agrupado["LineaBase_30d"].fillna(0, inplace=True)

    # El Ratio a predecir
    df_agrupado["Ratio_Desempeno"] = np.where(
        df_agrupado["LineaBase_30d"] > 0,
        df_agrupado["ColocacionMontoReal"] / df_agrupado["LineaBase_30d"],
        1.0,
    )
    # Capping anti-optimismo (La agencia no puede colocar más del doble de su promedio normal)
    df_agrupado["Ratio_Desempeno"] = df_agrupado["Ratio_Desempeno"].clip(upper=2.0)

    # Lags Operativos
    df_agrupado["Monto_Ayer"] = df_agrupado.groupby("IdSAgencia")[
        "ColocacionMontoReal"
    ].shift(1)
    df_agrupado["Tendencia_7_Dias"] = df_agrupado.groupby("IdSAgencia")[
        "ColocacionMontoReal"
    ].transform(lambda x: x.shift(1).rolling(window=7, min_periods=1).mean())
    df_agrupado["Analistas_Ayer"] = df_agrupado.groupby("IdSAgencia")[
        "AnalistasActivos"
    ].shift(1)
    df_agrupado["Experiencia_Ayer"] = df_agrupado.groupby("IdSAgencia")[
        "ExperienciaPromedioMeses"
    ].shift(1)
    df_agrupado.fillna(0, inplace=True)

    df_ml = df_agrupado[df_agrupado["LineaBase_30d"] > 0].copy()
    df_ml = df_ml.drop(columns=["AnalistasActivos", "ExperienciaPromedioMeses"])

    # Convertir Agencias a texto para el modelo
    df_ml = pd.get_dummies(df_ml, columns=["IdSAgencia"])

    X = df_ml.drop(
        columns=[
            "Fecha",
            "ColocacionMontoReal",
            "Ratio_Desempeno",
            "LineaBase_30d",
            "ColocacionesMicro",
            "ColocacionesMacro",
        ]
    )
    y = df_ml["Ratio_Desempeno"]

    corte = int(len(df_ml) * 0.85)
    X_train, X_test = X.iloc[:corte], X.iloc[corte:]
    y_train, y_test = y.iloc[:corte], y.iloc[corte:]

    modelo = xgb.XGBRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    modelo.fit(X_train, y_train)

    ruta_modelo = os.path.join(os.path.dirname(__file__), "modelo_xgboost.pkl")
    ruta_columnas = os.path.join(
        os.path.dirname(__file__), "columnas_entrenamiento.pkl"
    )
    joblib.dump(modelo, ruta_modelo)
    joblib.dump(X.columns.tolist(), ruta_columnas)
    logging.info(f"💾 Modelo Agencia-Puro V6 guardado con éxito.")


if __name__ == "__main__":
    run_training()
