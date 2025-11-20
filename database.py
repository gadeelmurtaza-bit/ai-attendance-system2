import sqlite3
import os

DB_FILE = "students.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            roll TEXT UNIQUE,
            image_path TEXT,
            encoding BLOB
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            roll TEXT,
            name TEXT,
            status TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


def add_student(name, roll, image_path, encoding):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("INSERT INTO students (name, roll, image_path, encoding) VALUES (?, ?, ?, ?)",
                (name, roll, image_path, encoding))
    conn.commit()
    conn.close()


def get_all_students():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("SELECT name, roll, image_path, encoding FROM students")
    result = cur.fetchall()
    conn.close()
    return result


def mark_attendance(roll, name, status):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("INSERT INTO attendance (roll, name, status) VALUES (?, ?, ?)",
                (roll, name, status))
    conn.commit()
    conn.close()


def get_attendance():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("SELECT * FROM attendance ORDER BY timestamp DESC")
    data = cur.fetchall()
    conn.close()
    return data
