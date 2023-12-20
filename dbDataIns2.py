import psycopg2

def main():
    try:
        # Connect to PostgreSQL database
        conn = psycopg2.connect(
            host="ai-dashboard2.cox4boq5aldo.ap-south-1.rds.amazonaws.com",
            database="postgres",
            user="postgres",
            password="postgres"
        )
        cursor = conn.cursor()

        # Create 'movies' table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS movies (
            movie_id SERIAL PRIMARY KEY,
            title VARCHAR(100) NOT NULL,
            genre VARCHAR(50) NOT NULL,
            release_year INTEGER NOT NULL
        );
        """)

        # Create 'reviews' table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            review_id SERIAL PRIMARY KEY,
            movie_id INTEGER REFERENCES movies(movie_id),
            user_id INTEGER NOT NULL,
            rating INTEGER NOT NULL,
            comment TEXT
        );
        """)

        # Create 'actors' table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS actors (
            actor_id SERIAL PRIMARY KEY,
            movie_id INTEGER REFERENCES movies(movie_id),
            name VARCHAR(100) NOT NULL,
            role VARCHAR(100) NOT NULL
        );
        """)

        # Insert sample data into 'movies'
        movies_data = [
            ('Inception', 'Sci-Fi', 2010),
            ('The Godfather', 'Crime', 1972),
            ('The Dark Knight', 'Action', 2008),
            ('Forrest Gump', 'Drama', 1994),
            ('Fight Club', 'Drama', 1999),
            ('Pulp Fiction', 'Crime', 1994),
            ('The Matrix', 'Sci-Fi', 1999),
            ('Avengers: Endgame', 'Action', 2019)
        ]

        for movie in movies_data:
            cursor.execute("INSERT INTO movies (title, genre, release_year) VALUES (%s, %s, %s)", movie)

        # Insert sample data into 'reviews'
        reviews_data = [
            (1, 1, 9, 'Mind-blowing!'),
            (2, 2, 10, 'A masterpiece'),
            (3, 3, 9, 'Epic'),
            (4, 4, 8, 'Touching'),
            (5, 5, 8, 'Thought-provoking'),
            (6, 6, 9, 'Classic'),
            (7, 7, 10, 'Revolutionary'),
            (8, 8, 9, 'Amazing finale')
        ]

        for review in reviews_data:
            cursor.execute("INSERT INTO reviews (movie_id, user_id, rating, comment) VALUES (%s, %s, %s, %s)", review)

        # Insert sample data into 'actors'
        actors_data = [
            (1, 'Leonardo DiCaprio', 'Cobb'),
            (2, 'Marlon Brando', 'Vito Corleone'),
            (3, 'Christian Bale', 'Bruce Wayne'),
            (4, 'Tom Hanks', 'Forrest Gump'),
            (5, 'Brad Pitt', 'Tyler Durden'),
            (6, 'John Travolta', 'Vincent Vega'),
            (7, 'Keanu Reeves', 'Neo'),
            (8, 'Robert Downey Jr.', 'Tony Stark')
        ]

        for actor in actors_data:
            cursor.execute("INSERT INTO actors (movie_id, name, role) VALUES (%s, %s, %s)", actor)

        # Commit changes and close the connection
        conn.commit()
        cursor.close()
        conn.close()
        
    except Exception as e:
        print("Something went wrong:", e)

if __name__ == "__main__":
    main()
