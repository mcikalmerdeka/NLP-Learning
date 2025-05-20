import os
import streamlit as st
import psycopg2
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv

# Configure APIs
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

# Utility Functions
def configure_streamlit():
    """Configure the Streamlit app settings."""
    st.set_page_config(page_title="Chat with your database through LLMs")    
    st.header("Chat with your database through LLMs")
    
    # Initialize chat history in session state if it doesn't exist
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

def initialize_language_model(model_choice):
    """Initialize the chosen language model client."""
    if model_choice == "GPT-4o" or model_choice == "GPT-4.1":
        return OpenAI(api_key=openai_api_key)
    else:
        return Anthropic(api_key=anthropic_api_key)

def get_model_response(question, prompt, model_choice, history=None):
    """Get response from the selected LLM."""
    try:
        client = initialize_language_model(model_choice)
        
        if model_choice == "GPT-4o":
            messages = [
                {"role": "system", "content": prompt},
            ]
            if history:
                messages.extend(history)
            messages.append({"role": "user", "content": question})
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=4000
            )
            return response.choices[0].message.content
        elif model_choice == "GPT-4.1":
            messages = [
                {"role": "system", "content": prompt},
            ]
            if history:
                messages.extend(history)
            messages.append({"role": "user", "content": question})
            
            response = client.chat.completions.create(
                model="gpt-4.1",
                messages=messages,
                max_tokens=4000
            )
            return response.choices[0].message.content
        else:
            messages = []
            if history:
                messages.extend(history)
            messages.append({"role": "user", "content": question})
            
            response = client.messages.create(
                model="claude-3-7-sonnet-20250219",
                system=prompt,
                messages=messages,
                max_tokens=4000
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

def load_database_schema_description():
    """Load the database schema description from the file."""
    with open(r"E:\NLP Learning\NLP-Learning\Business Intelligence Chatbot with Langchain\datasets\dataset_multiple_tables\database_schema_description.doc", "r") as file:
        return file.read()

def clear_chat_history():
    """Clear the chat history from session state."""
    st.session_state.chat_history = []

# Prompt Definitions
SQL_GENERATION_SYSTEM_PROMPT = """
    You are an expert in converting English questions to PostgreSQL query!

    The SQL database is described in the following file:
    {database_schema_description}

    Please understand the entire database schema, tables, columns and the relationship between the tables.

    Format your SQL query with proper indentation, line breaks, and alignment to make it readable. 
    For example:
    
    SELECT 
        op.payment_type,
        COUNT(o.order_id) FILTER (WHERE EXTRACT(YEAR FROM o.order_purchase_timestamp) = 2016) AS num_usage_2016,
        COUNT(o.order_id) FILTER (WHERE EXTRACT(YEAR FROM o.order_purchase_timestamp) = 2017) AS num_usage_2017,
        COUNT(o.order_id) FILTER (WHERE EXTRACT(YEAR FROM o.order_purchase_timestamp) = 2018) AS num_usage_2018
    FROM 
        orders o
    JOIN 
        order_payments op ON o.order_id = op.order_id 
    GROUP BY 
        op.payment_type
    ORDER BY 
        num_usage_2018 DESC;

    The output should not include ``` or the word "sql".
    And should not include any other like conversational response from your system, just the SQL query.
    Also please be careful with ambiguous column names when joining tables, make sure to use the proper table name or alias in front of the column name.
    
    Remember previous questions and context when generating SQL for follow-up questions.
"""

RESPONSE_GENERATION_SYSTEM_PROMPT = """
    You are a expert data analyst agent.

    Previously, you were asked: "{question}"
    The query result from the database is: "{result}".
    
    Please respond to the user in a humane and friendly and detailed manner.
    For example, if the question is "What is the biggest sales of product A?", 
    you should answer "The biggest sales of product A is 1000 USD".
    
    Remember the conversation history for context when answering follow-up questions.
"""

# Main Application Logic
def main():
    show_query = True # Show query for debugging
    configure_streamlit()
    
    # Add application explanation
    st.expander("ℹ️ About Multi-Table Database Chat").markdown(
    """
        - This app allows you to ask questions about a complex database with multiple related tables.
        - The AI assistant will convert your question into SQL, query the database, and provide a detailed analysis.
        - The system loads the full database schema to understand table relationships.
        - Choose an AI model from the sidebar and connect to your database to get started!
        - The chat has memory, so you can ask follow-up questions.
    """
    )
    
    # Sidebar for UI Configuration
    with st.sidebar:
        # Sidebar for Model Configuration
        st.subheader("Model Settings")
        model_choice = st.selectbox("Select a model", ["GPT-4o", "GPT-4.1", "Claude 3.7 Sonnet"], key="model_choice")

        # Sidebar for Database Configuration
        st.subheader("Database Settings")
        st.text_input("Host", value="localhost", key="Host")
        st.text_input("Port", value="5432", key="Port")
        st.text_input("User", value=os.getenv("DB_USER"), key="User")
        st.text_input("Password", type="password", value=os.getenv("DB_PASSWORD"), key="Password")
        st.text_input("Database", value=os.getenv("DB_NAME_2"), key="Database")
        if st.button("Test Connection"):
            with st.spinner("Testing database connection..."):
                if connect_to_database():
                    st.success("Connection successful!")
                    
        # Clear chat button
        st.subheader("Chat Controls")
        if st.button("Clear Chat History"):
            clear_chat_history()
            st.success("Chat history cleared!")

    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        else:
            st.chat_message("assistant").write(message["content"])

    # User input
    question = st.chat_input("Ask a question about your database")
    
    if question:
        # Display user message
        st.chat_message("user").write(question)
        
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": question})
        
        with st.spinner("Processing your query..."):
            # Extract conversation history for context (excluding the current question)
            model_history = st.session_state.chat_history[:-1] if len(st.session_state.chat_history) > 1 else None

            # Get SQL query from LLM
            sql_query = get_model_response(
                question, 
                SQL_GENERATION_SYSTEM_PROMPT.format(database_schema_description=load_database_schema_description()), 
                st.session_state.model_choice,
                model_history
            )
            
            if sql_query:
                if show_query:
                    st.subheader("Generated SQL Query:")
                    st.code(sql_query, language="sql")

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
                        RESPONSE_GENERATION_SYSTEM_PROMPT.format(question=question, result=result),
                        st.session_state.model_choice,
                        model_history
                    )
                    
                    # Display assistant response
                    st.chat_message("assistant").write(humane_response)
                    
                    # Add assistant response to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": humane_response})
                else:
                    error_message = "No results returned from the query."
                    st.error(error_message)
                    
                    # Add error message to chat history as assistant response
                    st.session_state.chat_history.append({"role": "assistant", "content": error_message})
            else:
                error_message = "Failed to generate SQL query."
                st.error(error_message)
                
                # Add error message to chat history as assistant response
                st.session_state.chat_history.append({"role": "assistant", "content": error_message})

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
