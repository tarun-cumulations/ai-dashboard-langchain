import pyodbc

# Database credentials
server = 'ai-dashboard-mssql.cox4boq5aldo.ap-south-1.rds.amazonaws.com'
database = 'master'
username = 'admin'
password = 'cumulations'

# Create the connection
#conn = pyodbc.connect(f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}')

conn = pyodbc.connect(f'DRIVER={{/usr/lib/libmsodbcsql-17.so}};SERVER={server};DATABASE={database};UID={username};PWD={password}')


# Create a cursor
cursor = conn.cursor()

# Commit the transaction (if you have inserted data, otherwise you can skip this)
conn.commit()

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

# ... (your existing code for fetching and printing data from other tables)

# Close the cursor and connection
cursor.close()
conn.close()
