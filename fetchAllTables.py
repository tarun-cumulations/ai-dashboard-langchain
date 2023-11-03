import psycopg2

def fetch_all_table_names(cursor, schema='public'):
    query = f"""
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = '{schema}';
    """
    cursor.execute(query)
    tables = [table[0] for table in cursor.fetchall()]
    return tables

try:
    # Connect to the database (replace these parameters with your own)
    conn = psycopg2.connect(
        host="ai-dashboard.cox4boq5aldo.ap-south-1.rds.amazonaws.com",
        database="postgres",
        user="postgres",
        password="postgres"
    )
    cursor = conn.cursor()

    # Fetch all table names in the 'public' schema
    table_names = fetch_all_table_names(cursor)
    print("Table names:", table_names)
    
    cursor.close()
    conn.close()

except Exception as e:
    print("Something went wrong:", e)
