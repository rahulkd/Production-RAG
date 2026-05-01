import psycopg2

DB_URL = "postgresql://admin:1234@localhost:5432/mydatabase"

def read_rows():
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()

    cur.execute("SELECT id, name, value, created_at FROM test_data_table")
    rows = cur.fetchall()

    cur.close()
    conn.close()

    if not rows:
        print("No rows found in test_data_table")
        return

    print(f"{'id':<5} {'name':<20} {'value':<10} {'created_at'}")
    print("-" * 55)
    for row in rows:
        print(f"{row[0]:<5} {row[1]:<20} {row[2]:<10} {row[3]}")

if __name__ == "__main__":
    read_rows()
