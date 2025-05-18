import os
from dotenv import load_dotenv
import streamlit as st
import psycopg2
from openai import OpenAI
from anthropic import Anthropic

# Configure APIs
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

# Utility Functions
def configure_streamlit():
    """Configure the Streamlit app settings."""
    st.set_page_config(page_title="Chat with your database through LLMs")    
    st.header("Chat with your database through LLMs")

def initialize_language_model(model_choice):
    """Initialize the chosen language model client."""
    if model_choice == "GPT-4o":
        return OpenAI(api_key=openai_api_key)
    else:
        return Anthropic(api_key=anthropic_api_key)

def get_model_response(question, prompt, model_choice):
    """Get response from the selected LLM."""
    try:
        client = initialize_language_model(model_choice)
        
        if model_choice == "GPT-4o":
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": question}
                ]
            )
            return response.choices[0].message.content
        else:
            response = client.messages.create(
                model="claude-3-7-sonnet-20250219",
                system=prompt,
                messages=[
                    {"role": "user", "content": question}
                ]
            )
            return response.content[0].text
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
PROMPT_QUERY = """
    You are an expert in converting English questions to PostgreSQL query!

    The SQL database "postgres" has the table "sales" and the following columns:
    - ORDERNUMBER: Unique identifier for sales orders
    - QUANTITYORDERED: Number of products ordered in each line item
    - PRICEEACH: Unit price of each product
    - ORDERLINENUMBER: Line number of the order item
    - SALES: Total sales amount in USD for the line item
    - ORDERDATE: Date when the order was placed
    - STATUS: Current status of the order (e.g., Shipped)
    - QTR_ID: Quarter of the year when order was placed (1-4)
    - MONTH_ID: Month of the year when order was placed (1-12)
    - YEAR_ID: Year when the order was placed
    - PRODUCTLINE: Category of the product (e.g., Motorcycles)
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
"""

PROMPT_HUMANE_RESPONSE_TEMPLATE = """
    You are a customer service agent.

    Previously, you were asked: "{question}"
    The query result from the database is: "{result}".
    
    Please respond to the customer in a humane and friendly and detailed manner.
    For example, if the question is "What is the biggest sales of product A?", 
    you should answer "The biggest sales of product A is 1000 USD".
"""

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

            # Get SQL query from LLM
            sql_query = get_model_response(question, PROMPT_QUERY, st.session_state.model_choice)
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

                # Generate humane response
                humane_response = get_model_response(
                    question, 
                    PROMPT_HUMANE_RESPONSE_TEMPLATE.format(question=question, result=result),
                    st.session_state.model_choice
                )
                st.subheader("AI Response:")
                st.write(humane_response)
            else:
                st.error("No results returned from the query.")

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
