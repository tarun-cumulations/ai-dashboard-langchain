import pyodbc

# Database credentials
server = 'ai-dashboard-mssql.cox4boq5aldo.ap-south-1.rds.amazonaws.com'
database = 'master'
username = 'admin'
password = 'cumulations'

# Create the connection to the master database with autocommit=True
conn = pyodbc.connect(f'DRIVER={{/usr/lib/libmsodbcsql-17.so}};SERVER={server};DATABASE={database};UID={username};PWD={password}', autocommit=True)

# Create a cursor
cursor = conn.cursor()

# Create a new database called "movies"
try:
    cursor.execute("CREATE DATABASE movies")
    print("Database 'movies' created successfully.")
except pyodbc.Error as err:
    print("Couldn't create database: ", err)

# Close the cursor and connection
cursor.close()
conn.close()
