import sqlite3

# Connect to your database (replace with your actual DB filename)
conn = sqlite3.connect("your_database.db")
cursor = conn.cursor()

# Add the missing column
try:
    cursor.execute("ALTER TABLE comments ADD COLUMN summary TEXT;")
    print("Column 'summary' added successfully!")
except sqlite3.OperationalError as e:
    print("Error:", e)

# Commit changes and close
conn.commit()
conn.close()