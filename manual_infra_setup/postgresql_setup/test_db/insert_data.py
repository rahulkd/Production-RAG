import psycopg2

DB_URL = "postgresql://admin:1234@localhost:5432/mydatabase"

def insert_sample_row():
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS test_data_table (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100),
            value INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cur.execute(
        "INSERT INTO test_data_table (name, value) VALUES (%s, %s) RETURNING id",
        ("sample_item", 42)
    )
    row_id = cur.fetchone()[0]

    conn.commit()
    cur.close()
    conn.close()

    print(f"Inserted row with id: {row_id}")

if __name__ == "__main__":
    insert_sample_row()
