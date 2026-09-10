import pandas as pd
import holidays
import logging
from config import get_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def fabricar_features_temporales(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Asegurar que la fecha sea tipo datetime
    df["Fecha"] = pd.to_datetime(df["Fecha"])

    # 2. Variables Básicas de Tiempo
    df["Anio"] = df["Fecha"].dt.year
    df["Mes"] = df["Fecha"].dt.month
    df["Dia"] = df["Fecha"].dt.day
    df["DiaSemana"] = df["Fecha"].dt.dayofweek  # 0=Lunes, 6=Domingo

    # 3. Variables de Negocio (Quincenas y Fin de mes)
    df["EsQuincena"] = df["Dia"].apply(lambda x: 1 if x in [15, 16] else 0)
    df["EsFinDeMes"] = df["Fecha"].dt.is_month_end.astype(int)

    # 4. Feriados Nacionales (Perú)
    pe_holidays = holidays.PE(years=df["Anio"].unique().tolist())
    df["EsFeriadoNacional"] = df["Fecha"].apply(lambda x: 1 if x in pe_holidays else 0)

    # 5. Festividades Locales (Cusco)
    # Aquí mapeamos fechas clave que impactan el comercio local
    def es_fiesta_cusco(fecha):
        # Mapeo del calendario festivo de Cusco (Mes, Día)
        festividades_cusco = {
            # Enero
            (1, 6),
            (1, 20),
            # Febrero (Carnavales - Fechas referenciales)
            (2, 5),
            (2, 12),
            (2, 15),
            # Marzo y Abril (Lanzamiento Inti Raymi y Semana Santa)
            (3, 27),
            (3, 29),
            (3, 30),
            (4, 1),
            (4, 2),
            (4, 3),
            (4, 4),
            (4, 5),
            # Mayo (Cruces)
            (5, 2),
            (5, 3),
            (5, 24),
            (5, 31),
            # Junio (Mes Jubilar - Impacto masivo en operaciones)
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
            # Julio (Virgen del Carmen)
            (7, 15),
            (7, 16),
            (7, 17),
            (7, 18),
            # Agosto (Pachamama y Santos)
            (8, 1),
            (8, 2),
            (8, 15),
            (8, 24),
            (8, 30),
            # Septiembre (¡El freno de hoy!)
            (9, 8),
            (9, 14),
            (9, 30),
            # Octubre
            (10, 18),
            (10, 31),
            # Noviembre
            (11, 1),
            (11, 2),
            # Diciembre (Santurantikuy)
            (12, 22),
            (12, 23),
            (12, 24),
            (12, 31),
        }

        # Si la fecha coincide con una festividad cusqueña, marcamos 1
        if (fecha.month, fecha.day) in festividades_cusco:
            return 1
        return 0

    df["EsFiestaLocal"] = df["Fecha"].apply(es_fiesta_cusco)

    return df


def run_feature_engineering():
    logging.info("🚀 Iniciando Construcción de Features...")
    engine = get_engine()

    # 1. Leer el histórico crudo
    query = "SELECT * FROM ml.fct_historico_colocacion"
    logging.info("📖 Leyendo datos históricos desde SQL Server...")
    df_raw = pd.read_sql(query, engine)

    # 2. Fabricar las columnas de contexto
    logging.info(
        "⚙️ Procesando el calendario inteligente (Feriados, Quincenas, Cusco)..."
    )
    df_features = fabricar_features_temporales(df_raw)

    # 3. Guardar la matriz final en la base de datos
    tabla_destino = "fct_features_entrenamiento"
    logging.info(f"💾 Guardando matriz de entrenamiento en ml.{tabla_destino}...")

    df_features.to_sql(
        name=tabla_destino, schema="ml", con=engine, if_exists="replace", index=False
    )

    engine.dispose()
    logging.info(
        f"✅ ¡Éxito! Se procesaron {len(df_features)} registros listos para Machine Learning."
    )


if __name__ == "__main__":
    run_feature_engineering()
