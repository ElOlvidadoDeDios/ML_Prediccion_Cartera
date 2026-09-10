import os
from dotenv import load_dotenv
from sqlalchemy import create_engine

# Cargar variables de entorno
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))


def get_engine():
    server = os.getenv("DB_SERVER")
    database = os.getenv("DB_DATABASE")

    # Ajusta esto si usas usuario/contraseña en vez de Trusted_Connection
    conn_str = f"mssql+pyodbc://@{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server&Trusted_Connection=yes"
    return create_engine(conn_str)
