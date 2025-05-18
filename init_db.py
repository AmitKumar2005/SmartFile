import mysql.connector
from dotenv import load_dotenv
import os

load_dotenv()


def init_db():
    try:
        conn = mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME"),
        )
        cursor = conn.cursor()
        with open("schema.sql", "r") as f:
            sql_commands = f.read().split(";")
            for command in sql_commands:
                if command.strip() and "CREATE DATABASE" not in command:
                    try:
                        cursor.execute(command)
                    except mysql.connector.Error as e:
                        print(f"Skipping command due to error: {e}")
        conn.commit()
        cursor.close()
        conn.close()
        print("Database initialized successfully")
    except Exception as e:
        print(f"Database initialization failed: {e}")
        raise


if __name__ == "__main__":
    init_db()
