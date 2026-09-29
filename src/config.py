# src/config

import os
from dotenv import load_dotenv

# Cargar variables del .env
load_dotenv()

# Origen (Nube)
ORIGEN_SERVER = os.getenv("DB_TRANSACMIF_SERVER")
ORIGEN_DB = os.getenv("DB_TRANSACMIF_NAME")
ORIGEN_USER = os.getenv("DB_TRANSACMIF_USER")
ORIGEN_PASS = os.getenv("DB_TRANSACMIF_PASS")

# Destino (Local)
DESTINO_SERVER = os.getenv("DB_DWH_SERVER")
DESTINO_DB = os.getenv("DB_DWH_NAME")

# Cadenas de conexión
STR_CONN_ORIGEN = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={ORIGEN_SERVER};DATABASE={ORIGEN_DB};UID={ORIGEN_USER};PWD={ORIGEN_PASS}"
STR_CONN_DESTINO = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={DESTINO_SERVER};DATABASE={DESTINO_DB};Trusted_Connection=yes"
