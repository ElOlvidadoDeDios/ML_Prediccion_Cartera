# src/01_bulid_features.py

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
    # 🔥 FASE ENTERPRISE: Unimos el Histórico con tus Features de DWH
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
    logging.info("🚀 Construcción de Features (Nivel Producto & Feriados SQL)...")
    df = obtener_datos_raw()
    df["Fecha"] = pd.to_datetime(df["Fecha"])

    # Ya no agrupamos y aplastamos. Solo ordenamos por Agencia y Producto.
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

    df["Monto_Ayer"] = grupo_prod["ColocacionMontoReal"].shift(1)
    df["Ops_Ayer"] = grupo_prod["ColocacionNumReal"].shift(1)
    df["Monto_Hace_7d"] = grupo_prod["ColocacionMontoReal"].shift(7)
    df["Monto_Hace_14d"] = grupo_prod["ColocacionMontoReal"].shift(14)

    df["Aceleracion_Semanal"] = df["Monto_Ayer"] - df["Monto_Hace_7d"]

    # Ticket Promedio Histórico POR PRODUCTO
    df["LineaBase_30d"] = grupo_prod["ColocacionMontoReal"].transform(
        lambda x: x.shift(1).rolling(30, min_periods=1).mean()
    )
    df["Media_Ops_30d"] = grupo_prod["ColocacionNumReal"].transform(
        lambda x: x.shift(1).rolling(30, min_periods=1).mean()
    )

    # Si es un producto nuevo o sin ventas, asume S/ 2000 por defecto
    df["Ticket_Promedio_30d"] = np.where(
        df["Media_Ops_30d"] > 0, df["LineaBase_30d"] / df["Media_Ops_30d"], 2000
    )

    # 3. Contexto General de la Agencia (Presión comercial)
    grupo_age = df.groupby(["IdSAgencia"])
    df["Bolsa_En_Evaluacion_3d"] = grupo_age["MontoSolicitado"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).sum()
    )
    df["Repago_Ayer"] = grupo_age["RepagoReal"].shift(1)

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
    df = df.drop(columns=columnas_futuras).dropna().reset_index(drop=True)

    df.to_csv("src/dataset_procesado.csv", index=False)
    logging.info(f"✅ Se procesaron {len(df)} registros a nivel Producto.")


if __name__ == "__main__":
    construir_features()
