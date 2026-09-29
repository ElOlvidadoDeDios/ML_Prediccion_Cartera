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
    logging.info("🚀 Construcción de Features (Inyectando calendario de ceros)...")
    df = obtener_datos_raw()
    df["Fecha"] = pd.to_datetime(df["Fecha"])

    # =========================================================================
    # 🔥 LA CURA: LA GRILLA CONTINUA DE TIEMPO 🔥
    # Forzamos la creación de todos los días del calendario para cada producto.
    # Así evitamos que el shift() "viaje en el tiempo" robando ventas de meses pasados.
    # =========================================================================

    rango_fechas = pd.date_range(
        start=df["Fecha"].min(), end=df["Fecha"].max(), freq="D"
    )
    combinaciones = df[
        ["IdSAgencia", "IdTipoProducto", "NombreProducto"]
    ].drop_duplicates()

    # Creamos el esqueleto con todos los días cruzados con todas las agencias y productos
    grid = (
        combinaciones.assign(key=1)
        .merge(pd.DataFrame({"Fecha": rango_fechas, "key": 1}), on="key")
        .drop("key", axis=1)
    )

    # Rescatamos los feriados antes de hacer el cruce para no perderlos
    fechas_df = (
        df[["Fecha", "EsFeriadoNacional", "EsFiestaLocal"]]
        .drop_duplicates(subset=["Fecha"])
        .dropna()
    )

    # Unimos la data real al esqueleto. Los días sin venta quedarán vacíos (NaN)
    df = pd.merge(
        grid,
        df.drop(columns=["EsFeriadoNacional", "EsFiestaLocal", "NombreProducto"]),
        on=["IdSAgencia", "IdTipoProducto", "Fecha"],
        how="left",
    )
    df = pd.merge(df, fechas_df, on="Fecha", how="left")

    # Todo lo vacío significa "Nadie entró a la agencia a pedir/pagar este producto" -> CERO.
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

    # Ordenar vital para que el shift() funcione cronológicamente
    df = df.sort_values(by=["IdSAgencia", "IdTipoProducto", "Fecha"]).reset_index(
        drop=True
    )

    # =========================================================================

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
    df["Monto_Hace_14d"] = grupo_prod["ColocacionMontoReal"].shift(14).fillna(0)

    df["Aceleracion_Semanal"] = df["Monto_Ayer"] - df["Monto_Hace_7d"]

    # Ticket Promedio Histórico POR PRODUCTO
    df["LineaBase_30d"] = (
        grupo_prod["ColocacionMontoReal"]
        .transform(lambda x: x.shift(1).rolling(30, min_periods=1).mean())
        .fillna(0)
    )
    df["Media_Ops_30d"] = (
        grupo_prod["ColocacionNumReal"]
        .transform(lambda x: x.shift(1).rolling(30, min_periods=1).mean())
        .fillna(0)
    )

    df["Ticket_Promedio_30d"] = np.where(
        df["Media_Ops_30d"] > 0, df["LineaBase_30d"] / df["Media_Ops_30d"], 2000
    )

    # 3. Contexto General de la Agencia (Presión comercial y Fondeo)
    grupo_age = df.groupby(["IdSAgencia"])

    # El embudo SÍ usa grupo_prod (porque es solicitud del producto)
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
    df["Monto_Solicitado_14d"] = (
        grupo_prod["MontoSolicitado"]
        .transform(lambda x: x.shift(14).rolling(7, min_periods=1).sum())
        .fillna(0)
    )
    df["Monto_Solicitado_21d"] = (
        grupo_prod["MontoSolicitado"]
        .transform(lambda x: x.shift(21).rolling(7, min_periods=1).sum())
        .fillna(0)
    )

    # El fondeo SÍ usa grupo_age (caja común de la agencia)
    df["Repago_Ayer"] = grupo_age["RepagoReal"].shift(1).fillna(0)
    df["Repago_Semana_Pasada"] = (
        grupo_age["RepagoReal"]
        .transform(lambda x: x.shift(1).rolling(7, min_periods=1).sum())
        .fillna(0)
    )

    # 4. Limpieza final de variables que causan Fuga de Datos
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
        "Media_Ops_30d",
        "ColocacionesMicro",
        "ColocacionesMacro",
    ]

    # ATENCIÓN: Eliminé el .dropna() porque borraba los días vacíos arruinando la grilla.
    df = df.drop(columns=columnas_futuras, errors="ignore").fillna(0)

    df.to_csv("src/dataset_procesado.csv", index=False)
    logging.info(
        f"✅ Se procesaron {len(df)} registros. El XGBoost ya sabe qué es un CERO."
    )


if __name__ == "__main__":
    construir_features()
