import sqlite3
import pandas as pd
import json
import ast
# from models.utils import to_list

def create_tables(conn):
    """Creates all necessary tables for the database schema."""
    cursor = conn.cursor()
    print("Creating database tables...")

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS coffees (
        coffee_id TEXT PRIMARY KEY,
        url TEXT UNIQUE NOT NULL,
        company TEXT,
        coffee_name TEXT,
        roaster_location TEXT,
        roast_level VARCHAR(12),
        review_date DATE,
        rating REAL,
        aroma REAL,
        acidity REAL,
        body REAL,
        flavor REAL,
        aftertaste REAL,
        with_milk REAL,
        blind_assessment TEXT,
        notes TEXT,
        bottom_line TEXT,
        price_per_oz REAL,
        countries TEXT,
        test_method VARCHAR(18),
        process TEXT,
        flavor_profile TEXT,
        flavor_profile_str TEXT,
        varietals TEXT,
        agtron_1 REAL,
        agrton_2 REAL,
        price_tier VARCHAR(8)                   
    )           
    """)

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS synthetic_queries (
        query_id INTEGER PRIMARY KEY AUTOINCREMENT,
        coffee_id TEXT NOT NULL,
        query_text TEXT NOT NULL,
        query_type TEXT,
        FOREIGN KEY (coffee_id) REFERENCES coffees (coffee_id)               
    )            
    ''')

    conn.commit()
    print("Tables created successfully.")


def populate_database(db_path, csv_path, queries_path):
    """Reads data from a CSV and populates an SQLite database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    create_tables(conn)

    # populate coffees table
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Data successfully loaded.")
    df = df.rename(columns={'id': 'coffee_id'})

    print(f"Populating table 'coffees'...")
    df.to_sql("coffees", conn, if_exists="replace", index=False)
    print(f"Successfully inserted {len(df)} records into 'coffees' table.")

    # TODO: populate synthetic queries table
    
    conn.commit()
    conn.close()


if __name__ == "__main__":
    DB_PATH = "data/processed/coffee_database.db"
    PREPROCESSED_PATH_CSV = "data/processed/preprocessed_data.csv"

    populate_database(DB_PATH, PREPROCESSED_PATH_CSV, None)



    



