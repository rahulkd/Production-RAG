import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

HOST = "localhost"
PORT = 5432
USER = "admin"
PASSWORD = "1234"
DEFAULT_DB = "mydatabase"
NEW_DB = "airflow"

def create_database(db_name: str):
    conn = psycopg2.connect(host=HOST, port=PORT, user=USER, password=PASSWORD, dbname=DEFAULT_DB)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()

    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
    if cur.fetchone():
        print(f"Database '{db_name}' already exists.")
    else:
        cur.execute(f"CREATE DATABASE {db_name}")
        print(f"Database '{db_name}' created successfully.")

    cur.close()
    conn.close()

if __name__ == "__main__":
    create_database(NEW_DB)
