import os
import streamlit as st
import psycopg2
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Configure Streamlit page - must be the first Streamlit command
st.set_page_config(page_title="Chat with your database through LLMs")

# Configure OpenAI API
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

# Mock database schema and descriptions
def load_database_schema_description():
    """Load the database schema description from the file and split into chunks."""
    try:
        import requests
        response = requests.get("https://raw.githubusercontent.com/mcikalmerdeka/NLP-Learning/main/Business%20Intelligence%20Chatbot%20with%20Langchain/datasets/dataset_multiple_tables/database_schema_description.doc")
        response.raise_for_status()
        content = response.text
    except Exception as e:
        st.error(f"Error fetching schema from URL: {e}")
        # Fallback to local file if URL fails
        try:
            with open(r"E:\NLP Learning\NLP-Learning\Business Intelligence Chatbot with Langchain\datasets\dataset_multiple_tables\database_schema_description.doc", "r") as file:
                content = file.read()
        except Exception as e2:
            st.error(f"Error reading local schema file: {e2}")
            return ""
    
    # Split content into meaningful chunks based on sections
    # First split by major headings (# heading)
    major_sections = content.split('\n#')
    
    chunks = []
    if major_sections[0]:  # Handle the first section if not starting with a heading
        chunks.append(major_sections[0])
    
    for section in major_sections[1:]:
        # Add the # back to the section since it was removed in the split
        section = '#' + section
        
        # For larger sections like table descriptions, further split by table
        if "table collections" in section.lower():
            # Split by numbered table sections
            table_sections = section.split('\n ')
            for t_section in table_sections:
                if t_section.strip() and len(t_section) > 50:  # Only add non-empty meaningful chunks
                    chunks.append(t_section.strip())
        else:
            chunks.append(section.strip())
    
    return [chunk for chunk in chunks if chunk and len(chunk) > 50]  # Filter empty or too small chunks

# Function to generate a vector index
def generate_vector_index(text_segments):
    """
    Creates a vector store of text chunks and saves it to a FAISS index.

    Args:
        text_segments (list): List of text chunks to be embedded and indexed.

    Returns:
        FAISS vector store: The loaded or newly created vector store
    """
    embed_model = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_db = FAISS.from_texts(text_segments, embedding=embed_model)
    vector_db.save_local("faiss_index_store")
    return vector_db

# Load or create vector database
def load_or_create_vector_db():
    """Load existing vector database or create a new one if it doesn't exist."""
    # For demo purposes, always create a fresh index to ensure correct chunking
    # But in production, you can load the existing index
    if os.path.exists("faiss_index_store"):
        # Try to remove the old index
        import shutil
        try:
            shutil.rmtree("faiss_index_store")
            st.info("Recreating vector index with proper text chunking...")
        except Exception as e:
            st.warning(f"Could not remove old index: {e}. Will try to create/load it anyway.")
    
    # Create a new vector index with properly chunked text
    schema_chunks = load_database_schema_description()
    if not schema_chunks:
        st.error("No schema chunks found or schema file is empty!")

        # Fallback to a simple tokenization if needed
        content = load_database_schema_description_raw()
        tokens = content.split()
        chunk_size = 100
        schema_chunks = [' '.join(tokens[i:i+chunk_size]) for i in range(0, len(tokens), chunk_size)]
    
    return generate_vector_index(schema_chunks)

# Add a function to get the raw content for fallback
def load_database_schema_description_raw():
    """Load the raw database schema description as a single string."""
    try:
        import requests
        response = requests.get("https://raw.githubusercontent.com/mcikalmerdeka/NLP-Learning/main/Business%20Intelligence%20Chatbot%20with%20Langchain/datasets/dataset_multiple_tables/database_schema_description.doc")
        response.raise_for_status()
        return response.text
    except Exception as e:
        st.error(f"Error fetching schema from URL: {e}")
        # Fallback to local file if URL fails
        try:
            with open(r"E:\NLP Learning\NLP-Learning\Business Intelligence Chatbot with Langchain\datasets\dataset_multiple_tables\database_schema_description.doc", "r") as file:
                return file.read()
        except Exception as e2:
            st.error(f"Error reading local schema file: {e2}")
            return ""

# Load vector database once at startup
vector_db = load_or_create_vector_db()

# Utility Functions
def configure_streamlit():
    """Configure the Streamlit app settings."""
    st.header("Chat with your database through LLMs")

def initialize_language_model(model_choice):
    """Initialize the chosen language model."""
    if model_choice == "GPT-4o":
        return ChatOpenAI(api_key=openai_api_key, model="gpt-4o")
    elif model_choice == "GPT-4.1":
        return ChatOpenAI(api_key=openai_api_key, model="gpt-4.1")
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
    Also please be careful with ambiguous column names when joining tables, make sure to use the proper table name or alias in front of the column name.
    
    Based on the following database schema:
    {retrieved_schema}
    
    Convert this question: {question}
"""

# Define functions for RAG
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
    
    # Add application explanation
    st.expander("ℹ️ About Multi-Table Database Chat with RAG").markdown(
    """
        - This app allows you to ask questions about a complex database with multiple related tables.
        - The AI assistant uses RAG (Retrieval-Augmented Generation) to find relevant schema information.
        - Only the most relevant parts of the schema are used for each query, improving accuracy.
        - Your question is converted into SQL, the database is queried, and you get a detailed analysis.
        - Choose an AI model from the sidebar and connect to your database to get started!
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

    # User input
    question = st.text_input("Input: ", key="input")

    if st.button("Ask the question") and question:
        with st.spinner("Processing your query..."):
            schema_docs = retrieve_schema(question)
            retrieved_schema = "\n".join([doc.page_content for doc in schema_docs])

            st.subheader("Retrieved Schema Details")
            
            # Display retrieved schema in a more formatted way
            with st.expander("View Retrieved Schema Information", expanded=True):
                for i, doc in enumerate(schema_docs):
                    st.markdown(f"### Document {i+1}")
                    st.text(doc.page_content)
                    st.markdown("---")

            # Get SQL query using the selected model
            sql_query = generate_sql_query(question, retrieved_schema, st.session_state.model_choice)
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