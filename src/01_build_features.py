# src/01_build_features.py

import pandas as pd
import pyodbc
import logging
import warnings
import numpy as np
from config import STR_CONN_DESTINO

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def obtener_datos_raw():
    conn = pyodbc.connect(STR_CONN_DESTINO)
    query = """
        SELECT 
            h.*, 
            ISNULL(f.EsFeriadoNacional, 0) AS EsFeriadoNacional, 
            ISNULL(f.EsFiestaLocal, 0) AS EsFiestaLocal
        FROM [ml].[fct_historico_colocacion] h
        LEFT JOIN [ml].[fct_features_entrenamiento] f 
            ON h.Fecha = f.Fecha AND h.IdSAgencia = f.IdSAgencia
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df


def construir_features():
    logging.info("🚀 Construcción de Features (Inyectando Ritmo Operativo)...")
    df = obtener_datos_raw()
    df["Fecha"] = pd.to_datetime(df["Fecha"])

    rango_fechas = pd.date_range(
        start=df["Fecha"].min(), end=df["Fecha"].max(), freq="D"
    )
    combinaciones = df[
        ["IdSAgencia", "IdTipoProducto", "NombreProducto"]
    ].drop_duplicates()

    grid = (
        combinaciones.assign(key=1)
        .merge(pd.DataFrame({"Fecha": rango_fechas, "key": 1}), on="key")
        .drop("key", axis=1)
    )
    fechas_df = (
        df[["Fecha", "EsFeriadoNacional", "EsFiestaLocal"]]
        .drop_duplicates(subset=["Fecha"])
        .dropna()
    )

    df = pd.merge(
        grid,
        df.drop(columns=["EsFeriadoNacional", "EsFiestaLocal", "NombreProducto"]),
        on=["IdSAgencia", "IdTipoProducto", "Fecha"],
        how="left",
    )
    df = pd.merge(df, fechas_df, on="Fecha", how="left")

    df["EsFeriadoNacional"] = df["EsFeriadoNacional"].fillna(0)
    df["EsFiestaLocal"] = df["EsFiestaLocal"].fillna(0)

    columnas_a_cero = [
        "ColocacionNumReal",
        "ColocacionMontoReal",
        "MontoSolicitado",
        "CantidadSolicitudes",
        "RepagoReal",
    ]
    for col in columnas_a_cero:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    df = df.sort_values(by=["IdSAgencia", "IdTipoProducto", "Fecha"]).reset_index(
        drop=True
    )

    # 1. Calendario y Estacionalidad
    df["DiaSemana"] = df["Fecha"].dt.dayofweek
    df["MesDelAnio"] = df["Fecha"].dt.month
    df["DiaDelMes"] = df["Fecha"].dt.day
    df["EsFinDeMes"] = df["Fecha"].dt.is_month_end.astype(int)
    df["DiasParaFinMes"] = df["Fecha"].dt.days_in_month - df["DiaDelMes"]
    df["Fiebre_Cierre"] = df["DiasParaFinMes"].apply(lambda x: 1 if x <= 5 else 0)
    df["EsQuincena"] = df["DiaDelMes"].apply(lambda x: 1 if x in [14, 15, 16] else 0)
    df["EsPrincipioMes"] = df["DiaDelMes"].apply(lambda x: 1 if x <= 7 else 0)
    df["EsDomingo"] = (df["DiaSemana"] == 6).astype(int)

    # 2. Desfases POR PRODUCTO Y AGENCIA
    grupo_prod = df.groupby(["IdSAgencia", "IdTipoProducto"])
    df["Monto_Ayer"] = grupo_prod["ColocacionMontoReal"].shift(1).fillna(0)
    df["Ops_Ayer"] = grupo_prod["ColocacionNumReal"].shift(1).fillna(0)
    df["Monto_Hace_7d"] = grupo_prod["ColocacionMontoReal"].shift(7).fillna(0)
    df["Aceleracion_Semanal"] = df["Monto_Ayer"] - df["Monto_Hace_7d"]

    # 🔥 LA CLAVE: EL RITMO NATURAL DE LA AGENCIA 🔥
    # Esto salva a la IA cuando la bandeja de solicitudes está vacía
    df["Ritmo_Ops_7d"] = (
        grupo_prod["ColocacionNumReal"]
        .transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
        .fillna(0)
    )
    df["Ritmo_Ops_14d"] = (
        grupo_prod["ColocacionNumReal"]
        .transform(lambda x: x.shift(1).rolling(14, min_periods=1).mean())
        .fillna(0)
    )
    df["Ritmo_Ops_30d"] = (
        grupo_prod["ColocacionNumReal"]
        .transform(lambda x: x.shift(1).rolling(30, min_periods=1).mean())
        .fillna(0)
    )

    # Ticket Promedio Histórico
    df_ventas = df[df["ColocacionNumReal"] > 0].copy()
    df_ventas["Ticket_Diario"] = (
        df_ventas["ColocacionMontoReal"] / df_ventas["ColocacionNumReal"]
    )
    grupo_ventas = df_ventas.groupby(["IdSAgencia", "IdTipoProducto"])
    df_ventas["Ticket_Promedio_Real"] = grupo_ventas["Ticket_Diario"].transform(
        lambda x: x.shift(1).rolling(15, min_periods=1).mean()
    )
    df = pd.merge(
        df,
        df_ventas[["IdSAgencia", "IdTipoProducto", "Fecha", "Ticket_Promedio_Real"]],
        on=["IdSAgencia", "IdTipoProducto", "Fecha"],
        how="left",
    )
    df["Ticket_Promedio_30d"] = df.groupby(["IdSAgencia", "IdTipoProducto"])[
        "Ticket_Promedio_Real"
    ].ffill()
    df["Ticket_Promedio_30d"] = df["Ticket_Promedio_30d"].fillna(3000)
    df = df.drop(columns=["Ticket_Promedio_Real"])

    # 3. Contexto General de la Agencia
    grupo_age = df.groupby(["IdSAgencia"])
    df["Bolsa_En_Evaluacion_3d"] = (
        grupo_prod["MontoSolicitado"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).sum())
        .fillna(0)
    )
    df["Monto_Solicitado_7d"] = (
        grupo_prod["MontoSolicitado"]
        .transform(lambda x: x.shift(7).rolling(7, min_periods=1).sum())
        .fillna(0)
    )
    df["Repago_Ayer"] = grupo_age["RepagoReal"].shift(1).fillna(0)
    df["Repago_Semana_Pasada"] = (
        grupo_age["RepagoReal"]
        .transform(lambda x: x.shift(1).rolling(7, min_periods=1).sum())
        .fillna(0)
    )

    # 4. Limpieza final (OJO: Aseguramos NO borrar las columnas Ritmo_Ops)
    columnas_futuras = [
        "NombreProducto",
        "PlazoPromedioMeses",
        "TasaPromedioTEA",
        "ColocacionesHombres",
        "ColocacionesMujeres",
        "ColocacionesSociosNuevos",
        "EdadPromedio",
        "MontoSolicitado",
        "CantidadSolicitudes",
        "RepagoReal",
        "ColocacionesMicro",
        "ColocacionesMacro",
        "LineaBase_30d",
        "Media_Ops_30d",
    ]
    df = df.drop(columns=columnas_futuras, errors="ignore").fillna(0)

    df.to_csv("src/dataset_procesado.csv", index=False)
    logging.info(f"✅ Se procesaron {len(df)} registros listos para la IA.")


if __name__ == "__main__":
    construir_features()
