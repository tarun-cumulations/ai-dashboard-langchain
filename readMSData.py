# Create the connection (replace placeholders with your actual credentials)

import pyodbc

server = 'ai-dashboard-mssql.cox4boq5aldo.ap-south-1.rds.amazonaws.com'
database = 'movies'
username = 'admin'
password = 'cumulations'

conn = pyodbc.connect(f'DRIVER={{/usr/lib/libmsodbcsql-17.so}};SERVER={server};DATABASE={database};UID={username};PWD={password}')

# Create a cursor
cursor = conn.cursor()

# Fetch and print data from 'movies' table
cursor.execute("SELECT * FROM movies")
rows = cursor.fetchall()
print("Data from 'movies' table:")
for row in rows:
    print(row)

# Fetch and print data from 'reviews' table
cursor.execute("SELECT * FROM reviews")
rows = cursor.fetchall()
print("Data from 'reviews' table:")
for row in rows:
    print(row)

# Fetch and print data from 'actors' table
cursor.execute("SELECT * FROM actors")
rows = cursor.fetchall()
print("Data from 'actors' table:")
for row in rows:
    print(row)

# Close the cursor and connection
cursor.close()
conn.close()
