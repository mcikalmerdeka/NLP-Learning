# Script to create a database and multiple tables in PostgreSQL
import pandas as pd
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

# Step 1: Load the CSV files
customers      = pd.read_csv('https://raw.githubusercontent.com/mcikalmerdeka/NLP-Learning/main/Business%20Intelligence%20Chatbot%20with%20Langchain/dataset_multiple_tables/customers_dataset.csv')
geolocation    = pd.read_csv('https://raw.githubusercontent.com/mcikalmerdeka/NLP-Learning/main/Business%20Intelligence%20Chatbot%20with%20Langchain/dataset_multiple_tables/geolocation_dataset.csv')
order_items    = pd.read_csv('https://raw.githubusercontent.com/mcikalmerdeka/NLP-Learning/main/Business%20Intelligence%20Chatbot%20with%20Langchain/dataset_multiple_tables/order_items_dataset.csv')
order_payments = pd.read_csv('https://raw.githubusercontent.com/mcikalmerdeka/NLP-Learning/main/Business%20Intelligence%20Chatbot%20with%20Langchain/dataset_multiple_tables/order_payments_dataset.csv')
order_reviews  = pd.read_csv('https://raw.githubusercontent.com/mcikalmerdeka/NLP-Learning/main/Business%20Intelligence%20Chatbot%20with%20Langchain/dataset_multiple_tables/order_reviews_dataset.csv')
orders         = pd.read_csv('https://raw.githubusercontent.com/mcikalmerdeka/NLP-Learning/main/Business%20Intelligence%20Chatbot%20with%20Langchain/dataset_multiple_tables/orders_dataset.csv')
products       = pd.read_csv('https://raw.githubusercontent.com/mcikalmerdeka/NLP-Learning/main/Business%20Intelligence%20Chatbot%20with%20Langchain/dataset_multiple_tables/product_dataset.csv', index_col=0)
sellers        = pd.read_csv('https://raw.githubusercontent.com/mcikalmerdeka/NLP-Learning/main/Business%20Intelligence%20Chatbot%20with%20Langchain/dataset_multiple_tables/sellers_dataset.csv')

# Replace NaN values with None (PostgreSQL's NULL)
all_df = [customers, geolocation, order_items, order_payments, order_reviews, orders, products, sellers]

for i, df in enumerate(all_df):
    all_df[i] = df.where(pd.notnull(df), None)

# Step 2: Connect to PostgreSQL
try:
    conn = psycopg2.connect(
        host="localhost",
        database=os.getenv("DB_NAME_2"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )
    cursor = conn.cursor()
    print("Connected to PostgreSQL successfully!")
except psycopg2.Error as e:
    print(f"Error connecting to PostgreSQL: {e}")
    exit()

# Step 3: Define a function to map Pandas dtypes to PostgreSQL types
def map_dtype_to_sql(dtype):
    if pd.api.types.is_integer_dtype(dtype):
        return "INT"
    elif pd.api.types.is_float_dtype(dtype):
        return "FLOAT"
    elif pd.api.types.is_bool_dtype(dtype):
        return "BOOLEAN"
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        return "TIMESTAMP"
    else:
        return "VARCHAR(255)" # Default to VARCHAR for object/string types
    
# Step 4: Generate the SQL schema from the DataFrame
table_name = ["customers", "geolocation", "order_items", "order_payments", "order_reviews", "orders", "products", "sellers"]

# Create a dictionary to store schemas for each table
schemas = {}

for df, table in zip(all_df, table_name):
    schemas[table] = ", ".join([
        f"{col} {map_dtype_to_sql(dtype)}"
        for col, dtype in zip(df.columns, df.dtypes)
    ])

# Note: the output will look something like this (postgres style):
# id INT, price FLOAT, is_active BOOLEAN, created_at TIMESTAMP, description VARCHAR(255)

# Step 5: Create the table
for table in table_name:
    create_table_query = f"CREATE TABLE IF NOT EXISTS {table} ({schemas[table]});"
    try:
        cursor.execute(create_table_query)
        conn.commit()
        print(f"Table `{table}` created successfully!")
    except psycopg2.Error as e:
        conn.rollback()
        print(f"Error creating table: {e}")

# Step 6: Insert the data into the table
for df, table in zip(all_df, table_name):
    for _, row in df.iterrows():
        placeholders = ", ".join(["%s"] * len(row))
        insert_query = f"INSERT INTO {table} VALUES ({placeholders})" # Insert the data per row
        try:
            cursor.execute(insert_query, tuple(row))
        except psycopg2.Error as e:
            conn.rollback()
            print(f"Error inserting row: {e}")

# Commit the changes and close the connection
conn.commit()
cursor.close()
conn.close()
print("Data inserted successfully!")