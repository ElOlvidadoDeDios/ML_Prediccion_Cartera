import pandas as pd
import joblib
import logging
import os
import holidays
from datetime import timedelta
from config import get_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def run_daily_prediction():
    logging.info("🔮 Iniciando Predicción Diaria V6 (Nivel Agencia)...")
    engine = get_engine()

    df_max = pd.read_sql(
        "SELECT MAX(Fecha) as UltimaFecha FROM ml.fct_historico_colocacion", engine
    )
    ultima_fecha = pd.to_datetime(df_max["UltimaFecha"].iloc[0]).date()
    dia_a_predecir = ultima_fecha + timedelta(days=1)

    query_hist = f"SELECT * FROM ml.fct_historico_colocacion WHERE Fecha >= DATEADD(day, -45, '{ultima_fecha}')"
    df_hist = pd.read_sql(query_hist, engine)
    df_hist["Fecha"] = pd.to_datetime(df_hist["Fecha"]).dt.date

    # 1. Agrupar la historia al nivel Agencia
    col_agrup_hist = ["Fecha", "IdSAgencia"]
    df_hist = (
        df_hist.groupby(col_agrup_hist)
        .agg(
            {
                "ColocacionMontoReal": "sum",
                "AnalistasActivos": "max",
                "ExperienciaPromedioMeses": "mean",
                "ColocacionesMicro": "sum",
                "ColocacionesMacro": "sum",
            }
        )
        .reset_index()
    )

    # 2. Generar el día de hoy (1 fila por agencia)
    agencias = df_hist[["IdSAgencia"]].drop_duplicates()
    df_hoy = agencias.copy()
    df_hoy["Fecha"] = dia_a_predecir
    for col in [
        "ColocacionMontoReal",
        "AnalistasActivos",
        "ExperienciaPromedioMeses",
        "ColocacionesMicro",
        "ColocacionesMacro",
    ]:
        df_hoy[col] = 0

    df_full = pd.concat([df_hist, df_hoy], ignore_index=True)
    df_full["Fecha"] = pd.to_datetime(df_full["Fecha"])

    # Calendario y Festividades
    df_full["Anio"] = df_full["Fecha"].dt.year
    df_full["Mes"] = df_full["Fecha"].dt.month
    df_full["Dia"] = df_full["Fecha"].dt.day
    df_full["DiaSemana"] = df_full["Fecha"].dt.dayofweek
    df_full["EsQuincena"] = df_full["Dia"].apply(lambda x: 1 if x in [15, 16] else 0)
    df_full["EsFinDeMes"] = df_full["Fecha"].dt.is_month_end.astype(int)

    pe_holidays = holidays.PE(years=df_full["Anio"].unique().tolist())
    df_full["EsFeriadoNacional"] = df_full["Fecha"].apply(
        lambda x: 1 if x in pe_holidays else 0
    )

    def es_fiesta_cusco(fecha):
        festividades_cusco = {
            (1, 6),
            (1, 20),
            (2, 5),
            (2, 12),
            (2, 15),
            (3, 27),
            (3, 29),
            (3, 30),
            (4, 1),
            (4, 2),
            (4, 3),
            (4, 4),
            (4, 5),
            (5, 2),
            (5, 3),
            (5, 24),
            (5, 31),
            (6, 1),
            (6, 3),
            (6, 4),
            (6, 9),
            (6, 10),
            (6, 11),
            (6, 12),
            (6, 13),
            (6, 14),
            (6, 15),
            (6, 16),
            (6, 19),
            (6, 21),
            (6, 24),
            (7, 15),
            (7, 16),
            (7, 17),
            (7, 18),
            (8, 1),
            (8, 2),
            (8, 15),
            (8, 24),
            (8, 30),
            (9, 8),
            (9, 14),
            (9, 30),
            (10, 18),
            (10, 31),
            (11, 1),
            (11, 2),
            (12, 22),
            (12, 23),
            (12, 24),
            (12, 31),
        }
        return 1 if (fecha.month, fecha.day) in festividades_cusco else 0

    df_full["EsFiestaLocal"] = df_full["Fecha"].apply(es_fiesta_cusco)

    df_full = df_full.sort_values(by=["IdSAgencia", "Fecha"])

    # Línea base y Lags Operativos
    df_full["LineaBase_30d"] = df_full.groupby("IdSAgencia")[
        "ColocacionMontoReal"
    ].transform(lambda x: x.shift(1).rolling(window=21, min_periods=1).mean())
    df_full["Monto_Ayer"] = df_full.groupby("IdSAgencia")["ColocacionMontoReal"].shift(
        1
    )
    df_full["Tendencia_7_Dias"] = df_full.groupby("IdSAgencia")[
        "ColocacionMontoReal"
    ].transform(lambda x: x.shift(1).rolling(window=7, min_periods=1).mean())
    df_full["Analistas_Ayer"] = df_full.groupby("IdSAgencia")["AnalistasActivos"].shift(
        1
    )
    df_full["Experiencia_Ayer"] = df_full.groupby("IdSAgencia")[
        "ExperienciaPromedioMeses"
    ].shift(1)
    df_full.fillna(0, inplace=True)

    # Filtrar hoy
    df_pred = df_full[df_full["Fecha"].dt.date == dia_a_predecir].copy()
    df_ml = pd.get_dummies(df_pred, columns=["IdSAgencia"])

    modelo = joblib.load(os.path.join(os.path.dirname(__file__), "modelo_xgboost.pkl"))
    columnas_entrenamiento = joblib.load(
        os.path.join(os.path.dirname(__file__), "columnas_entrenamiento.pkl")
    )

    for col in columnas_entrenamiento:
        if col not in df_ml.columns:
            df_ml[col] = 0

    X_pred = df_ml[columnas_entrenamiento]
    ratio_predicho = modelo.predict(X_pred)

    # Guardar resultados
    df_final = df_pred[["Fecha", "IdSAgencia"]].copy()
    df_final["IdTipoProducto"] = "00"  # Código Genérico de Agencia Total
    df_final["MontoPredicho"] = df_pred["LineaBase_30d"] * ratio_predicho
    df_final["MontoPredicho"] = df_final["MontoPredicho"].apply(
        lambda x: round(x, 2) if x > 0 else 0
    )

    tabla_destino = "fct_predicciones_diarias"
    df_final.to_sql(
        name=tabla_destino, schema="ml", con=engine, if_exists="append", index=False
    )
    engine.dispose()
    logging.info("✅ ¡Predicción Diaria completada con éxito!")


if __name__ == "__main__":
    run_daily_prediction()
