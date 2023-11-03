import psycopg2

# Function to fetch and print data from a table
def fetch_and_print_data(cursor, table_name):
    query = f"SELECT * FROM {table_name};"
    cursor.execute(query)
    records = cursor.fetchall()
    print(f"Data from {table_name}:")
    for row in records:
        print(row)
    print("---")

try:
    # Connect to the database (replace these parameters with your own)
    conn = psycopg2.connect(
            host="ai-dashboard.cox4boq5aldo.ap-south-1.rds.amazonaws.com",
            database="postgres",
            user="postgres",
            password="postgres"
    )
    cursor = conn.cursor()

    # Fetch and print data from 'movies'
    fetch_and_print_data(cursor, 'movies')

    # Fetch and print data from 'reviews'
    fetch_and_print_data(cursor, 'reviews')

    # Fetch and print data from 'actors'
    fetch_and_print_data(cursor, 'actors')

    # Close the cursor and the connection
    cursor.close()
    conn.close()

except Exception as e:
    print("Something went wrong:", e)
