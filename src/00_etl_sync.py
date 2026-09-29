# src/00_etl_sync.py

import pandas as pd
import pyodbc
import logging
import warnings
import time
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from config import STR_CONN_ORIGEN, STR_CONN_DESTINO

load_dotenv()
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

DIAS_LOTE = int(os.getenv("DIAS_POR_LOTE", 30))
PAUSA_SEG = int(os.getenv("TIEMPO_PAUSA_SEGUNDOS", 3))


def obtener_ultima_fecha():
    try:
        conn = pyodbc.connect(STR_CONN_DESTINO)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT ISNULL(MAX(Fecha), '2000-01-01') FROM [ml].[fct_historico_colocacion]"
        )
        ultima_fecha = cursor.fetchone()[0]
        conn.close()

        # Si la tabla está vacía (nos dará '2000-01-01'), usamos la del .env
        if str(ultima_fecha) == "2000-01-01" or ultima_fecha is None:
            return pd.to_datetime(os.getenv("FECHA_INICIO_HISTORICO", "2023-01-01"))

        return pd.to_datetime(ultima_fecha)
    except Exception as e:
        return pd.to_datetime(os.getenv("FECHA_INICIO_HISTORICO", "2023-01-01"))


def extraer_lote(fecha_inicio, fecha_fin):
    conn = pyodbc.connect(STR_CONN_ORIGEN)
    query = f"""
    SET TRANSACTION ISOLATION LEVEL READ UNCOMMITTED;
    SET NOCOUNT ON;
    
    WITH CTE_UNICO AS (
        SELECT
            T_PTM.OTORGA, T_PTM.PAGARE, T_PTM.MONTO_PRESTAMO AS Monto,
            T_PTM.TIPO_PROD, TP.NOM_PROD,
            T_PTM.PLAZO, T_PTM.TEA_INTERES, S.SEXO, S.FECHA_APERTURA,
            DATEDIFF(YEAR, S.FECHA_NAC, T_PTM.OTORGA) AS EdadSocio,
            (SELECT TOP 1 U.ID_USER FROM SEGURIDAD.dbo.ANAREC A WITH(NOLOCK) INNER JOIN SEGURIDAD.dbo.USUARIOS U WITH(NOLOCK) ON A.ID_USER = U.ID_USER WHERE A.ID_ANAREC = T_SOL.ID_ANA) AS ID_USER,
            (SELECT TOP 1 A.ID_AGE FROM SEGURIDAD.dbo.ANAREC A WITH(NOLOCK) WHERE A.ID_ANAREC = T_SOL.ID_ANA) AS ID_AGE_RAW
        FROM TRANSACMIF.dbo.PRESTAMO T_PTM WITH(NOLOCK)
        INNER JOIN TRANSACMIF.dbo.PRESOL T_SOL WITH(NOLOCK) ON T_SOL.CUENTA = T_PTM.CUENTA AND T_SOL.PAGARE = T_PTM.PAGARE
        LEFT JOIN TRANSACMIF.dbo.TIPOPROD TP WITH(NOLOCK) ON T_PTM.TIPO_PROD = TP.TIPO_PROD
        LEFT JOIN TRANSACMIF.dbo.SOCIOS S WITH(NOLOCK) ON S.CUENTA = T_PTM.CUENTA
        WHERE T_PTM.TIPO_PROD <> '52' AND T_SOL.ESTADO = '3'
          AND T_PTM.OTORGA >= '{fecha_inicio}' AND T_PTM.OTORGA < '{fecha_fin}'
    ),
    CTE_LIMPIO AS (
        SELECT 
            OTORGA, Monto, ID_USER, TIPO_PROD, NOM_PROD,
            PLAZO, TEA_INTERES, SEXO, FECHA_APERTURA, EdadSocio,
            CASE
                WHEN ID_AGE_RAW = '98' THEN
                    CASE
                        WHEN RTRIM(ID_USER) LIKE '%10' THEN '10' WHEN RTRIM(ID_USER) LIKE '%11' THEN '11'
                        WHEN RTRIM(ID_USER) LIKE '%12' THEN '12' WHEN RTRIM(ID_USER) LIKE '%13' THEN '13'
                        WHEN RTRIM(ID_USER) LIKE '%6'  THEN '06' WHEN RTRIM(ID_USER) LIKE '%7'  THEN '07'
                        ELSE '98'
                    END
                WHEN ID_AGE_RAW = '01' THEN
                    CASE WHEN RTRIM(ID_USER) LIKE '%9' THEN '09' ELSE '01' END
                ELSE ID_AGE_RAW
            END AS IdSAgencia,
            (SELECT TOP 1 U.FECHA_CREA FROM SEGURIDAD.dbo.USUARIOS U WITH(NOLOCK) WHERE U.ID_USER = CTE_UNICO.ID_USER) AS FECHA_CREA
        FROM CTE_UNICO
        WHERE ID_USER NOT IN ('PRECASTIGO', 'RJULI6', 'RJULIACA', 'RLIMA7', 'RQUILLA3', 'RSICUA4', 'LHR5', 'HTEJ5', 'TKPN5', 'GHVJ5', 'OTA5', 'SDHF5', 'CMN5', 'HQND5', 'RTRES')
    ),
    CTE_REQUERIMIENTOS AS (
        SELECT 
            CAST(FECHA_SOL AS DATE) AS Fecha, 
            COD_AGE AS IdSAgencia_Req, 
            TIPO_PROD AS IdTipoProducto_Req, -- 🔥 NUEVO: Separar por producto
            SUM(MONTO_SOL) AS MontoSolicitado, 
            COUNT(NRO_SOL) AS CantidadSolicitudes
        FROM TRANSACMIF.dbo.PRESOL WITH(NOLOCK)
        WHERE FECHA_SOL >= '{fecha_inicio}' AND FECHA_SOL < '{fecha_fin}'
        GROUP BY CAST(FECHA_SOL AS DATE), COD_AGE, TIPO_PROD
    ),
    -- 🔥 NUEVO: CTE DE REPAGOS (Basado en gc_repago_cpp) 🔥
    CTE_REPAGOS AS (
        SELECT
            CAST(T_MOV.FECHA_MOV AS DATE) AS Fecha,
            CASE
                WHEN T_MOV.COD_AGE = '98' THEN
                    CASE
                        WHEN RTRIM(T_USU.ID_USER) LIKE '%10' THEN '10' WHEN RTRIM(T_USU.ID_USER) LIKE '%11' THEN '11'
                        WHEN RTRIM(T_USU.ID_USER) LIKE '%12' THEN '12' WHEN RTRIM(T_USU.ID_USER) LIKE '%13' THEN '13'
                        WHEN RTRIM(T_USU.ID_USER) LIKE '%6'  THEN '06' WHEN RTRIM(T_USU.ID_USER) LIKE '%7'  THEN '07'
                        ELSE '98'
                    END
                WHEN T_MOV.COD_AGE = '01' THEN
                    CASE WHEN RTRIM(T_USU.ID_USER) LIKE '%9' THEN '09' ELSE '01' END
                ELSE T_MOV.COD_AGE
            END AS IdSAgencia_Rep,
            SUM(T_MOV.CAPITAL) AS RepagoReal
        FROM TRANSACMIF.dbo.PREMOV T_MOV WITH (NOLOCK)
        INNER JOIN SEGURIDAD.dbo.ANAREC T_ANA WITH (NOLOCK)
            ON T_ANA.ID_AGE = T_MOV.COD_AGE AND T_ANA.FLAG_ANAREC = 'A'
        INNER JOIN SEGURIDAD.dbo.USUARIOS T_USU WITH (NOLOCK)
            ON T_USU.ID_USER = T_ANA.ID_USER
        WHERE T_MOV.FECHA_MOV >= '{fecha_inicio}' AND T_MOV.FECHA_MOV < '{fecha_fin}'
          AND T_MOV.TIPO_MOV != '0001'
          AND T_MOV.TIPO_DOC IN ('01', '03')
          AND NOT EXISTS (
              SELECT 1 FROM TRANSACMIF.dbo.PREMOV A WITH (NOLOCK)
              WHERE A.FECHA_MOV = T_MOV.FECHA_MOV AND A.COD_AGE = T_MOV.COD_AGE
                AND A.COD_CAJA = T_MOV.COD_CAJA AND A.TIPO_DOC = T_MOV.TIPO_DOC
                AND A.NRO_DOC = T_MOV.NRO_DOC AND LEFT(A.TIPO_MOV, 2) = '01'
          )
        GROUP BY CAST(T_MOV.FECHA_MOV AS DATE), 
                 CASE
                    WHEN T_MOV.COD_AGE = '98' THEN
                        CASE
                            WHEN RTRIM(T_USU.ID_USER) LIKE '%10' THEN '10' WHEN RTRIM(T_USU.ID_USER) LIKE '%11' THEN '11'
                            WHEN RTRIM(T_USU.ID_USER) LIKE '%12' THEN '12' WHEN RTRIM(T_USU.ID_USER) LIKE '%13' THEN '13'
                            WHEN RTRIM(T_USU.ID_USER) LIKE '%6'  THEN '06' WHEN RTRIM(T_USU.ID_USER) LIKE '%7'  THEN '07'
                            ELSE '98'
                        END
                    WHEN T_MOV.COD_AGE = '01' THEN
                        CASE WHEN RTRIM(T_USU.ID_USER) LIKE '%9' THEN '09' ELSE '01' END
                    ELSE T_MOV.COD_AGE
                 END
    ),
    CTE_FINAL AS (
        SELECT
            CAST(C.OTORGA AS DATE) AS Fecha, C.IdSAgencia,
            C.TIPO_PROD AS IdTipoProducto, ISNULL(C.NOM_PROD, 'No Definido') AS NombreProducto,
            COUNT(*) AS ColocacionNumReal, SUM(C.Monto) AS ColocacionMontoReal,
            COUNT(DISTINCT C.ID_USER) AS AnalistasActivos,
            CAST(AVG(DATEDIFF(MONTH, C.FECHA_CREA, C.OTORGA) * 1.0) AS DECIMAL(10,2)) AS ExperienciaPromedioMeses,
            SUM(CASE WHEN C.Monto <= 5000 THEN 1 ELSE 0 END) AS ColocacionesMicro,
            SUM(CASE WHEN C.Monto >= 50000 THEN 1 ELSE 0 END) AS ColocacionesMacro,
            CAST(AVG(C.PLAZO * 1.0) AS DECIMAL(10,2)) AS PlazoPromedioMeses,
            CAST(AVG(C.TEA_INTERES) AS DECIMAL(10,2)) AS TasaPromedioTEA,
            SUM(CASE WHEN C.SEXO = 'M' THEN 1 ELSE 0 END) AS ColocacionesHombres,
            SUM(CASE WHEN C.SEXO = 'F' THEN 1 ELSE 0 END) AS ColocacionesMujeres,
            SUM(CASE WHEN DATEDIFF(MONTH, C.FECHA_APERTURA, C.OTORGA) <= 1 THEN 1 ELSE 0 END) AS ColocacionesSociosNuevos,
            CAST(AVG(C.EdadSocio * 1.0) AS DECIMAL(10,2)) AS EdadPromedio
        FROM CTE_LIMPIO C
        WHERE C.IdSAgencia IN ('01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13')
        GROUP BY CAST(C.OTORGA AS DATE), C.IdSAgencia, C.TIPO_PROD, C.NOM_PROD
    )
    SELECT 
        F.*, 
        ISNULL(R.MontoSolicitado, 0) AS MontoSolicitado,
        ISNULL(R.CantidadSolicitudes, 0) AS CantidadSolicitudes,
        ISNULL(RP.RepagoReal, 0) AS RepagoReal
    FROM CTE_FINAL F
    LEFT JOIN CTE_REQUERIMIENTOS R 
        ON F.Fecha = R.Fecha 
        AND F.IdSAgencia = R.IdSAgencia_Req 
        AND F.IdTipoProducto = R.IdTipoProducto_Req -- 🔥 NUEVO: Cruce exacto
    LEFT JOIN CTE_REPAGOS RP ON F.Fecha = RP.Fecha AND F.IdSAgencia = RP.IdSAgencia_Rep
    """
    df_lote = pd.read_sql(query, conn)
    conn.close()
    return df_lote


def insertar_lote(df):
    if df.empty:
        return
    df = df.fillna(0)
    conn = pyodbc.connect(STR_CONN_DESTINO)
    cursor = conn.cursor()

    for index, row in df.iterrows():
        cursor.execute(
            """
            INSERT INTO [ml].[fct_historico_colocacion] 
            (Fecha, IdSAgencia, IdTipoProducto, NombreProducto, ColocacionNumReal, ColocacionMontoReal, 
             AnalistasActivos, ExperienciaPromedioMeses, ColocacionesMicro, ColocacionesMacro,
             PlazoPromedioMeses, TasaPromedioTEA, ColocacionesHombres, ColocacionesMujeres, ColocacionesSociosNuevos, EdadPromedio,
             MontoSolicitado, CantidadSolicitudes, RepagoReal)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            row["Fecha"],
            row["IdSAgencia"],
            row["IdTipoProducto"],
            row["NombreProducto"],
            row["ColocacionNumReal"],
            row["ColocacionMontoReal"],
            row["AnalistasActivos"],
            row["ExperienciaPromedioMeses"],
            row["ColocacionesMicro"],
            row["ColocacionesMacro"],
            row["PlazoPromedioMeses"],
            row["TasaPromedioTEA"],
            row["ColocacionesHombres"],
            row["ColocacionesMujeres"],
            row["ColocacionesSociosNuevos"],
            row["EdadPromedio"],
            row["MontoSolicitado"],
            row["CantidadSolicitudes"],
            row["RepagoReal"],
        )
    conn.commit()
    conn.close()


if __name__ == "__main__":
    fecha_actual = pd.to_datetime(datetime.now().date())
    fecha_proceso = obtener_ultima_fecha()

    if fecha_proceso >= fecha_actual:
        logging.info("✅ La base de datos ya está actualizada hasta hoy.")
    else:
        logging.info("☁️ Iniciando sincronización por lotes (Segura)...")

        # Eliminar posible data parcial de la última fecha para no duplicar
        conn = pyodbc.connect(STR_CONN_DESTINO)
        cursor = conn.cursor()
        # 🔥 FORMATO SEGURO SIN GUIONES PARA SQL SERVER (%Y%m%d) 🔥
        cursor.execute(
            f"DELETE FROM [ml].[fct_historico_colocacion] WHERE Fecha = '{fecha_proceso.strftime('%Y%m%d')}'"
        )
        conn.commit()
        conn.close()

        while fecha_proceso <= fecha_actual:
            fecha_fin = fecha_proceso + timedelta(days=DIAS_LOTE)
            # En el log (pantalla) sí lo mostramos con guiones para que se vea bonito
            logging.info(
                f"⏳ Procesando lote: {fecha_proceso.strftime('%Y-%m-%d')} a {fecha_fin.strftime('%Y-%m-%d')}..."
            )

            # 🔥 PERO AL SQL SERVER LE MANDAMOS EL FORMATO SEGURO SIN GUIONES 🔥
            df_lote = extraer_lote(
                fecha_proceso.strftime("%Y%m%d"), fecha_fin.strftime("%Y%m%d")
            )

            if not df_lote.empty:
                insertar_lote(df_lote)
                logging.info(f"   ✓ Guardados {len(df_lote)} registros.")
            else:
                logging.info("   - Sin datos en este periodo.")

            fecha_proceso = fecha_fin

            # Pausa inteligente para no ahorcar a TRANSACMIF
            time.sleep(PAUSA_SEG)

        logging.info("🚀 ¡Sincronización histórica completada con éxito!")
