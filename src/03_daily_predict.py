# src/03_daily_predict.py

import pandas as pd
import xgboost as xgb
import pickle
import logging
import pyodbc
import sys
import os
import shap
import numpy as np
from datetime import datetime
from config import STR_CONN_DESTINO

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

TRADUCTOR_VARIABLES = {
    "Fiebre_Cierre": "Presión Cierre de Mes",
    "EsQuincena": "Efecto Quincena",
    "Ritmo_Ops_7d": "Ritmo de Ventas Reciente",
    "Ritmo_Ops_30d": "Tendencia Mensual del Producto",
    "Bolsa_En_Evaluacion_3d": "Mega-Créditos en Trámite",
    "Monto_Ayer": "Colocación del Día Anterior",
    "Ticket_Promedio_30d": "Ticket Histórico",
    "EsDomingo": "Día No Laborable",
}


def obtener_nombre_legible(feature_name):
    for key, value in TRADUCTOR_VARIABLES.items():
        if key in feature_name:
            return value
    if "IdTipoProducto_" in feature_name:
        return "Naturaleza del Producto"
    if "IdSAgencia_" in feature_name:
        return "Comportamiento de Agencia"
    return feature_name


def predecir_manana():
    logging.info("🔮 Iniciando Predicción (Modelo Ritmo Flash)...")

    try:
        with open("src/modelo_ops.pkl", "rb") as f:
            modelo_ops = pickle.load(f)
        with open("src/columnas_entrenamiento.pkl", "rb") as f:
            columnas_modelo = pickle.load(f)
    except FileNotFoundError:
        logging.error("❌ Faltan archivos .pkl.")
        sys.exit(1)

    df = pd.read_csv(
        "src/dataset_procesado.csv", dtype={"IdSAgencia": str, "IdTipoProducto": str}
    )
    df["IdSAgencia"] = df["IdSAgencia"].apply(lambda x: str(x).zfill(2))
    df["Fecha"] = pd.to_datetime(df["Fecha"])

    fecha_env = os.getenv("FECHA_PREDICCION")
    fecha_objetivo = (
        pd.to_datetime(fecha_env)
        if fecha_env
        else pd.to_datetime(datetime.now().date())
    )
    logging.info(
        f"📅 GENERANDO PREDICCIONES PARA EL DÍA: {fecha_objetivo.strftime('%Y-%m-%d')}"
    )

    idx_ultimos = df.groupby(["IdSAgencia", "IdTipoProducto"])["Fecha"].idxmax()
    df_ultimo = df.loc[idx_ultimos].copy()

    df_ultimo["Fecha_Prediccion"] = fecha_objetivo
    df_ultimo["DiaSemana"] = df_ultimo["Fecha_Prediccion"].dt.dayofweek
    df_ultimo["MesDelAnio"] = df_ultimo["Fecha_Prediccion"].dt.month
    df_ultimo["DiaDelMes"] = df_ultimo["Fecha_Prediccion"].dt.day
    df_ultimo["EsFinDeMes"] = df_ultimo["Fecha_Prediccion"].dt.is_month_end.astype(int)
    df_ultimo["DiasParaFinMes"] = (
        df_ultimo["Fecha_Prediccion"].dt.days_in_month - df_ultimo["DiaDelMes"]
    )
    df_ultimo["Fiebre_Cierre"] = df_ultimo["DiasParaFinMes"].apply(
        lambda x: 1 if x <= 5 else 0
    )
    df_ultimo["EsQuincena"] = df_ultimo["DiaDelMes"].apply(
        lambda x: 1 if x in [14, 15, 16] else 0
    )
    df_ultimo["EsPrincipioMes"] = df_ultimo["DiaDelMes"].apply(
        lambda x: 1 if x <= 7 else 0
    )
    df_ultimo["EsDomingo"] = (df_ultimo["DiaSemana"] == 6).astype(int)

    # 🔥 LIMPIEZA DE MEMORIA FANTASMA 🔥
    dias_desfase = (fecha_objetivo - df_ultimo["Fecha"]).dt.days

    df_ultimo["Monto_Ayer"] = np.where(
        dias_desfase == 1, df_ultimo["ColocacionMontoReal"], 0
    )
    df_ultimo["Ops_Ayer"] = np.where(
        dias_desfase == 1, df_ultimo["ColocacionNumReal"], 0
    )

    # Apagamos los ritmos si el producto lleva muerto más días que su propia ventana de medición
    df_ultimo["Ritmo_Ops_7d"] = np.where(
        dias_desfase <= 7, df_ultimo["Ritmo_Ops_7d"], 0
    )
    df_ultimo["Ritmo_Ops_14d"] = np.where(
        dias_desfase <= 14, df_ultimo["Ritmo_Ops_14d"], 0
    )
    df_ultimo["Ritmo_Ops_30d"] = np.where(
        dias_desfase <= 30, df_ultimo["Ritmo_Ops_30d"], 0
    )

    df_ultimo["Bolsa_En_Evaluacion_3d"] = np.where(
        dias_desfase <= 3, df_ultimo["Bolsa_En_Evaluacion_3d"], 0
    )
    df_ultimo["Monto_Solicitado_7d"] = np.where(
        dias_desfase <= 7, df_ultimo["Monto_Solicitado_7d"], 0
    )
    df_ultimo["Aceleracion_Semanal"] = np.where(
        dias_desfase <= 7, df_ultimo["Aceleracion_Semanal"], 0
    )

    X_pred_raw = pd.get_dummies(
        df_ultimo, columns=["IdSAgencia", "IdTipoProducto", "MesDelAnio", "DiaSemana"]
    )
    X_pred = X_pred_raw.reindex(columns=columnas_modelo, fill_value=0)

    prediccion_operaciones = np.clip(modelo_ops.predict(X_pred), 0, None)
    ticket_ancla = df_ultimo["Ticket_Promedio_30d"]

    # REGLAS DE NEGOCIO
    factor_estacional = np.ones(len(df_ultimo))
    factor_estacional = np.where(df_ultimo["DiaSemana"] == 5, 0.50, factor_estacional)
    factor_estacional = np.where(
        df_ultimo["Fiebre_Cierre"] == 1, factor_estacional * 1.35, factor_estacional
    )

    predicciones_dinero = prediccion_operaciones * ticket_ancla * factor_estacional

    # RED DE SEGURIDAD (Evita predicciones de 5 soles. Si el modelo arroja algo > 0, asume al menos el 5% de un ticket)
    piso_minimo = ticket_ancla * 0.05
    predicciones_dinero = np.where(
        predicciones_dinero > 0, np.maximum(predicciones_dinero, piso_minimo), 0
    )

    predicciones_dinero = np.where(df_ultimo["EsDomingo"] == 1, 0, predicciones_dinero)

    logging.info("🧠 Generando explicaciones de la IA para Gerencia...")
    explainer = shap.TreeExplainer(modelo_ops)
    shap_values = explainer.shap_values(X_pred)

    motivos_positivos = []
    motivos_negativos = []

    for i in range(len(X_pred)):
        idx_max = np.argmax(shap_values[i])
        idx_min = np.argmin(shap_values[i])
        motivos_positivos.append(
            f"Impulsado por: {obtener_nombre_legible(columnas_modelo[idx_max])}"
        )
        motivos_negativos.append(
            f"Frenado por: {obtener_nombre_legible(columnas_modelo[idx_min])}"
        )

    resultados = pd.DataFrame(
        {
            "Fecha": df_ultimo["Fecha_Prediccion"],
            "IdSAgencia": df_ultimo["IdSAgencia"],
            "IdTipoProducto": df_ultimo["IdTipoProducto"],
            "Prediccion_Diaria": predicciones_dinero,
            "Motivo_Positivo": motivos_positivos,
            "Motivo_Negativo": motivos_negativos,
        }
    )

    try:
        conn = pyodbc.connect(STR_CONN_DESTINO)
        cursor = conn.cursor()
        fecha_str = fecha_objetivo.strftime("%Y-%m-%d")

        cursor.execute(
            f"DELETE FROM [ml].[fct_predicciones_diarias] WHERE CAST(Fecha AS DATE) = CAST('{fecha_str}' AS DATE)"
        )

        for _, row in resultados.iterrows():
            if row["Prediccion_Diaria"] > 0:
                cursor.execute(
                    """
                    INSERT INTO [ml].[fct_predicciones_diarias] 
                    (Fecha, IdSAgencia, IdTipoProducto, MontoPredicho, MotivoImpulsor, MotivoFreno)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    row["Fecha"],
                    row["IdSAgencia"],
                    row["IdTipoProducto"],
                    row["Prediccion_Diaria"],
                    row["Motivo_Positivo"],
                    row["Motivo_Negativo"],
                )

        conn.commit()
        conn.close()

        resumen_consola = (
            resultados.groupby("IdSAgencia")["Prediccion_Diaria"].sum().reset_index()
        )
        columna_dinamica = f"Predicción para {fecha_str}"
        resumen_consola[columna_dinamica] = resumen_consola["Prediccion_Diaria"].apply(
            lambda x: f"S/ {x:,.2f}"
        )

        print("\n--- PREDICCIÓN EXPLICADA Y CONSOLIDADA ---")
        print(resumen_consola[["IdSAgencia", columna_dinamica]].to_string(index=False))
        logging.info("✅ ¡Proyección Explicada guardada en SQL con éxito!")
    except Exception as e:
        logging.error(f"❌ Error al guardar en SQL: {e}")


if __name__ == "__main__":
    predecir_manana()
