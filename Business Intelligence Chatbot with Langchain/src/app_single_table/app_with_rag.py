import os
from dotenv import load_dotenv
import streamlit as st
import psycopg2
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import FAISS

# Configure OpenAI API
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

# Mock database schema and descriptions
schema_descriptions = [
    "Table: sales, Column: ORDERNUMBER, Description: Unique identifier for sales order",
    "Table: sales, Column: QUANTITYORDERED, Description: Number of  quantity of products sold in units",
    "Table: sales, Column: SALES, Description: Amount of sales or revenue in USD",
    "Table: sales, Column: PRODUCTLINE, Description: Line of products",
    "Table: sales, Column: ORDERDATE, Description: Date of the order",
    "Table: sales, Column: STATUS, Description: Current status of the order (e.g., Shipped)",
    "Table: sales, Column: QTR_ID, Description: Quarter of the year when order was placed (1-4)",
    "Table: sales, Column: MONTH_ID, Description: Month of the year when order was placed (1-12)",
    "Table: sales, Column: YEAR_ID, Description: Year when the order was placed",
    "Table: sales, Column: PRODUCTLINE, Description: Line of products",
    "Table: sales, Column: MSRP, Description: Manufacturer's suggested retail price",
    "Table: sales, Column: PRODUCTCODE, Description: Unique code identifying the product",
    "Table: sales, Column: CUSTOMERNAME, Description: Name of the customer who placed the order",
    "Table: sales, Column: PHONE, Description: Customer's phone number",
    "Table: sales, Column: ADDRESSLINE1, Description: First line of customer's address",
    "Table: sales, Column: ADDRESSLINE2, Description: Second line of customer's address (optional)",
    "Table: sales, Column: CITY, Description: Customer's city",
    "Table: sales, Column: STATE, Description: Customer's state or province",
    "Table: sales, Column: POSTALCODE, Description: Customer's postal code",
    "Table: sales, Column: COUNTRY, Description: Customer's country",
    "Table: sales, Column: TERRITORY, Description: Sales territory (e.g., NA, EMEA)",
    "Table: sales, Column: CONTACTLASTNAME, Description: Last name of the contact person",
    "Table: sales, Column: CONTACTFIRSTNAME, Description: First name of the contact person",
    "Table: sales, Column: DEALSIZE, Description: Size of the deal (Small, Medium, Large)"
]

# Function to generate a vector index
def generate_vector_index(text_segments):
    """
    Creates a vector store of text chunks and saves it to a FAISS index.

    Args:
        text_segments (list): List of text chunks to be embedded and indexed.

    Returns:
        None: The FAISS index is saved locally as 'faiss_index_store'.
    """
    embed_model = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_db = FAISS.from_texts(text_segments, embedding=embed_model)
    vector_db.save_local("faiss_index_store")
    return vector_db

vector_db = generate_vector_index(schema_descriptions)

# Utility Functions
def configure_streamlit():
    """Configure the Streamlit app settings."""
    st.set_page_config(page_title="Chat with your database through LLMs")    
    st.header("Chat with your database through LLMs")

def initialize_language_model(model_choice):
    """Initialize the chosen language model."""
    if model_choice == "GPT-4o":
        return ChatOpenAI(api_key=openai_api_key, model="gpt-4o")
    else:
        return ChatAnthropic(api_key=anthropic_api_key, model="claude-3-7-sonnet-20250219")

def get_model_response(prompt, model_choice):
    """Get response from the LLM."""
    try:
        client = initialize_language_model(model_choice)
        response = client.invoke(prompt)
        return response.content
    except Exception as e:
        st.error(f"Error with model API: {e}")
        return None
    
def connect_to_database():
    """Connect to the database using credentials from Streamlit session state."""
    try:
        connection = psycopg2.connect(
            host=st.session_state["Host"],
            database=st.session_state["Database"],
            user=st.session_state["User"],
            password=st.session_state["Password"]
        )
        st.success(f"Connected to the database {st.session_state['Database']} successfully!")
        return connection
    except Exception as e:
        st.error(f"Error with database connection: {e}")
        return None
    
def read_sql_query(query):
    """Execute an SQL query and return the result."""
    connection = connect_to_database()
    if connection:
        try:
            cursor = connection.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            return rows
        except Exception as e:
            st.error(f"Error executing query: {e}")
            return None
        finally:
            cursor.close()
            connection.close()
    return None

# Prompt Definitions
RESPONSE_GENERATION_SYSTEM_PROMPT = """
    You are a customer service agent.

    Previously, you were asked: "{question}"
    The query result from the database is: "{result}".
    
    Please respond to the customer in a humane and friendly and detailed manner.
    For example, if the question is "What is the biggest sales of product A?", 
    you should answer "The biggest sales of product A is 1000 USD".
"""

SQL_GENERATION_SYSTEM_PROMPT = """
    You are an expert in converting English questions to PostgreSQL query!

    The SQL database has the table "sales" with the following columns:
    - ORDERNUMBER: Unique identifier for sales orders
    - QUANTITYORDERED: Number of products ordered in units
    - SALES: Amount of sales or revenue in USD
    - ORDERDATE: Date of the order
    - STATUS: Current status of the order (e.g., Shipped)
    - QTR_ID: Quarter of the year when order was placed (1-4)
    - MONTH_ID: Month of the year when order was placed (1-12)
    - YEAR_ID: Year when the order was placed
    - PRODUCTLINE: Line of products
    - MSRP: Manufacturer's suggested retail price
    - PRODUCTCODE: Unique code identifying the product
    - CUSTOMERNAME: Name of the customer who placed the order
    - PHONE: Customer's phone number
    - ADDRESSLINE1: First line of customer's address
    - ADDRESSLINE2: Second line of customer's address (optional)
    - CITY: Customer's city
    - STATE: Customer's state or province
    - POSTALCODE: Customer's postal code
    - COUNTRY: Customer's country
    - TERRITORY: Sales territory (e.g., NA, EMEA)
    - CONTACTLASTNAME: Last name of the contact person
    - CONTACTFIRSTNAME: First name of the contact person
    - DEALSIZE: Size of the deal (Small, Medium, Large)

    Example SQL command: SELECT COUNT(*) FROM sales;

    The output should not include ``` or the word "sql".
    
    Based on the following database schema:
    {retrieved_schema}
    
    Convert this question: {question}
"""

# Define functions
def retrieve_schema(user_query):
    """Retrieve relevant schema details from vector store."""
    return vector_db.similarity_search(user_query, k=5)

def generate_sql_query(question, retrieved_schema, model_choice):
    """Generate SQL query based on user query and retrieved schema using LLMs."""
    prompt = SQL_GENERATION_SYSTEM_PROMPT.format(retrieved_schema=retrieved_schema, question=question)
    return get_model_response(prompt, model_choice)


# Main Application Logic
def main():
    show_query = True # Show query for debugging
    configure_streamlit()
    
    # Sidebar for UI Configuration
    with st.sidebar:
        # Sidebar for Model Configuration
        st.subheader("Model Settings")
        model_choice = st.selectbox("Select a model", ["GPT-4o", "Claude 3.7 Sonnet"], key="model_choice")

        # Sidebar for Database Configuration
        st.subheader("Database Settings")
        st.text_input("Host", value="localhost", key="Host")
        st.text_input("Port", value="5432", key="Port")
        st.text_input("User", value=os.getenv("DB_USER"), key="User")
        st.text_input("Password", type="password", value=os.getenv("DB_PASSWORD"), key="Password")
        st.text_input("Database", value="postgres", key="Database")
        if st.button("Test Connection"):
            with st.spinner("Testing database connection..."):
                if connect_to_database():
                    st.success("Connection successful!")

    # User input
    question = st.text_input("Input: ", key="input")

    if st.button("Ask the question") and question:
        with st.spinner("Processing your query..."):
            schema_docs = retrieve_schema(question)
            retrieved_schema = "\n".join([doc.page_content for doc in schema_docs])

            st.subheader("Retrieved Schema Details")
            st.write(retrieved_schema)

            # Get SQL query using the selected model
            sql_query = generate_sql_query(question, retrieved_schema, st.session_state.model_choice)
            if sql_query:
                if show_query:
                    st.subheader("Generated SQL Query:")
                    st.write(sql_query)

                # Execute the SQL query
                result = read_sql_query(sql_query)
                if result:
                    if show_query:
                        st.subheader("Query Results:")
                        for row in result:
                            st.write(row)

                    # Generate humane response using the selected model
                    humane_response = get_model_response(
                        RESPONSE_GENERATION_SYSTEM_PROMPT.format(question=question, result=result), 
                        st.session_state.model_choice
                    )
                    st.subheader("AI Response:")
                    st.write(humane_response)
                else:
                    st.error("No results returned from the query.")
            else:
                st.error("Failed to generate SQL query.")

    # Footer
    st.markdown(
        """
        <style>
        .footer {position: fixed;left: 0;bottom: 0;width: 100%;background-color: #000;color: white;text-align: center;}
        </style>
        <div class='footer'>
            <p>mcikalmerdeka@gmail.com</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()