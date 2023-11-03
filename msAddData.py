import pyodbc

# Database credentials
server = 'ai-dashboard-mssql.cox4boq5aldo.ap-south-1.rds.amazonaws.com'
database = 'movies'  # Now we use the 'movies' database
username = 'admin'
password = 'cumulations'

# Create the connection
conn = pyodbc.connect(f'DRIVER={{/usr/lib/libmsodbcsql-17.so}};SERVER={server};DATABASE={database};UID={username};PWD={password}')

# Create a cursor
cursor = conn.cursor()

# Create tables
# Feel free to add more fields or change types as needed
cursor.execute("""
CREATE TABLE movies (
    id INT PRIMARY KEY,
    title VARCHAR(255),
    genre VARCHAR(255),
    year INT
)
""")
cursor.execute("""
CREATE TABLE reviews (
    id INT PRIMARY KEY,
    movie_id INT FOREIGN KEY REFERENCES movies(id),
    user_id INT,
    rating INT,
    comment TEXT
)
""")
cursor.execute("""
CREATE TABLE actors (
    id INT PRIMARY KEY,
    movie_id INT FOREIGN KEY REFERENCES movies(id),
    name VARCHAR(255),
    role VARCHAR(255)
)
""")

# Insert data
movies_data = [
    (1, 'Inception', 'Sci-Fi', 2010),
    (2, 'The Godfather', 'Crime', 1972),
    (3, 'The Dark Knight', 'Action', 2008),
    (4, 'Forrest Gump', 'Drama', 1994),
    (5, 'Fight Club', 'Drama', 1999),
    (6, 'Pulp Fiction', 'Crime', 1994),
    (7, 'The Matrix', 'Sci-Fi', 1999),
    (8, 'Avengers: Endgame', 'Action', 2019)
]

for movie in movies_data:
    cursor.execute("INSERT INTO movies (id, title, genre, year) VALUES (?, ?, ?, ?)", movie)

reviews_data = [
    (1, 1, 1, 9, 'Mind-blowing!'),
    (2, 2, 2, 10, 'A masterpiece'),
    (3, 3, 3, 9, 'Epic'),
    (4, 4, 4, 8, 'Touching'),
    (5, 5, 5, 8, 'Thought-provoking'),
    (6, 6, 6, 9, 'Classic'),
    (7, 7, 7, 10, 'Revolutionary'),
    (8, 8, 8, 9, 'Amazing finale')
]

for review in reviews_data:
    cursor.execute("INSERT INTO reviews (id, movie_id, user_id, rating, comment) VALUES (?, ?, ?, ?, ?)", review)

actors_data = [
    (1, 1, 'Leonardo DiCaprio', 'Cobb'),
    (2, 2, 'Marlon Brando', 'Vito Corleone'),
    (3, 3, 'Christian Bale', 'Bruce Wayne'),
    (4, 4, 'Tom Hanks', 'Forrest Gump'),
    (5, 5, 'Brad Pitt', 'Tyler Durden'),
    (6, 6, 'John Travolta', 'Vincent Vega'),
    (7, 7, 'Keanu Reeves', 'Neo'),
    (8, 8, 'Robert Downey Jr.', 'Tony Stark')
]

for actor in actors_data:
    cursor.execute("INSERT INTO actors (id, movie_id, name, role) VALUES (?, ?, ?, ?)", actor)

# Commit the changes
conn.commit()

# Close the cursor and connection
cursor.close()
conn.close()
