# Business Intelligence Chatbot with Langchain

![Project Header](https://raw.githubusercontent.com/mcikalmerdeka/NLP-Learning/refs/heads/main/Business%20Intelligence%20Chatbot%20with%20Langchain/assets/Project%20Header.jpg)

This repo is for using LLMs to chat with your SQL database. Inspired by this [Gemini Chatbot repo](https://github.com/ardyadipta/gemini_chatbot_sql). Instead of using MySQL I used PostgreSQL and instead of using Google Gemini model series, I experimented using OpenAI (GPT-4.1 and GPT-4o) and Anthropic (Claude 3.7 Sonnet) model which I am more familiar with and `also later` wanted to try experiment with local models usage such as Deepseek-r1:1.5b and Qwen3:1.7b that is installed in my PC.

## 🎯 Objective and Project Origin

Around February 2025, I was quite curious if there was a way where we can retrieve data in a database without having to write sql queries, but only using natural language. So initially I tried to search online and the first thing I found was using a library called PandasAI. But there are major limitations on the usage limits in using the API on the free tier and upgrading to another tier costs way too much (around 200 euro per month as of May 2025). So I need to find another way to do database chat but with a more manageable cost and full control over the platform's behavior. But I couldn't find it at that time.

On April, when I was starting to learn about Langchain Framework from the beginning in a more structured way, I learned about the document loader method that can upload content from csv (haven't learned about chunking, embedding, or retrieval at that time) and immediately tried a brute force approach by calling the OpenAI and Anthropic models to analyze the data, and it turns out that it can understand simple lookup questions, but when it comes to data aggregation the model messed up quite bad. That was really dumb experimentation actually but it kinda gives an idea of how the behavior and limitations of the model are when trying to understand tabular data that is being transformed into text.

One month later, I watched the [conference talk](https://youtu.be/wN3T5NCTSAY?t=16827) from Ardya Dipta (Head of Data Science Kalbe Group) at DevFest Jakarta 2024 and was quite interested in trying the approach explained in the demo. The main idea is that the output of the LLM will actually be the SQL query, and how the LLM understand the our database is not by storing all the data per row (like how I did before) because in reality company data have millions of rows and several hundred of columns in total so this will burn out all the token usage fast and the cost will definitely explode. So instead, we just feed it with our database schema information and the secret strategy here is that the schema information need to be as complete as possible like from the column descriptions, relationship to other table in the database, data expected behaviour and example values in a column, and so on.

And in this project we will try to implement that approach.

## 📁 Project Structure

The project now has two main approaches:

### Single Table Approach

Analysis of sales data using a single database table:

- `src/app_single_table/app_with_rag.py`: Main application incorporating RAG for better schema understanding
- `src/app_single_table/app_without_rag.py`: Simplified version without the RAG approach
- `src/app_single_table/database_setup_single_table.py`: Script to set up PostgreSQL database with sample sales data
- `datasets/dataset_single_table/`: Contains the sales data CSV used for this approach

### Multiple Tables Approach

Analysis of e-commerce data (Olist) using multiple related tables:

- `src/app_multiple_tables/app_with_rag.py`: Enhanced version supporting multiple table queries with RAG
- `src/app_multiple_tables/app_without_rag.py`: Basic version for multiple tables without RAG
- `src/app_multiple_tables/database_setup_multiple_tables.py`: Script to set up database with Olist e-commerce data
- `datasets/dataset_multiple_tables/`: Contains the Olist e-commerce datasets with multiple related tables

## 🚀 Features

- **Natural Language to SQL Conversion**: Convert plain English queries into SQL statements
- **RAG-Enhanced Architecture**: Uses retrieval-augmented generation to improve contextual understanding of database schema
- **Multi-Table Support**: Analyze data across multiple related tables (Olist e-commerce dataset)
- **Single Table Analysis**: Simpler analysis for sales data scenarios
- **Streamlit UI**: User-friendly interface for interacting with the database
- **Multiple LLM Support**: Compatible with OpenAI (GPT-4o), Anthropic (Claude 3.7 Sonnet), and potential support for local models
- **Detailed Response Generation**: Formats query results into natural, conversational responses
- **Conversational Memory**: Maintains chat history to support follow-up questions and contextual conversations
- **Clear Chat Functionality**: Easily reset conversations with a "Clear Chat History" button in the sidebar
- **Chat Interface**: Modern chat-style UI for a more natural conversation experience

## 🔧 Setup and Installation

### Prerequisites

- Python 3.11+
- PostgreSQL database
- OpenAI API key and/or Anthropic API key

### Installation

1. Clone this repository:

   ```
   git clone https://github.com/yourusername/Business-Intelligence-Chatbot-with-Langchain.git
   cd Business-Intelligence-Chatbot-with-Langchain
   ```
2. Install dependencies:

   Using pip:

   ```
   pip install -e .
   ```

   Using uv package manager (recommended):

   ```
   uv pip install -e .
   ```

   Required dependencies (from pyproject.toml):

   - faiss-cpu>=1.11.0
   - langchain>=0.3.25
   - langchain-anthropic>=0.3.13
   - langchain-community>=0.3.24
   - langchain-openai>=0.3.16
   - numpy>=2.2.5
   - openai>=1.78.0
   - pandas>=2.2.3
   - psycopg2>=2.9.10
   - python-dotenv>=1.1.0
   - streamlit>=1.45.0
3. Create a `.env` file in the project root with your API and database credentials:

   ```
   OPENAI_API_KEY=your_openai_api_key
   ANTHROPIC_API_KEY=your_anthropic_api_key
   DB_USER=your_database_username
   DB_PASSWORD=your_database_password
   DB_HOST=your_database_host
   DB_PORT=your_database_port
   ```
4. Set up the database:

   For single table approach:

   ```
   python src/app_single_table/database_setup_single_table.py
   ```

   For multiple tables approach:

   ```
   python src/app_multiple_tables/database_setup_multiple_tables.py
   ```

## 💻 Usage

### Single Table Approach

1. Launch the application with RAG:

   ```
   streamlit run src/app_single_table/app_with_rag.py
   ```

   Or without RAG:

   ```
   streamlit run src/app_single_table/app_without_rag.py
   ```
2. Access the application at `http://localhost:8501`
3. Configure your database connection in the sidebar
4. Start asking questions in natural language about your sales data!

Example queries for sales data:

- "What were the total sales in 2003 and 2004?"
- "Show me the top 5 customers by revenue"
- "What is the phone number of customer name Toys of Finland, Co.?"
- "Which product line has the highest average order value?"
- "What about for 2005?" (follow-up question example)
- "Which of them had the highest growth?" (contextual follow-up)

### Multiple Tables Approach

1. Launch the application with RAG:

   ```
   streamlit run src/app_multiple_tables/app_with_rag.py
   ```

   Or without RAG:

   ```
   streamlit run src/app_multiple_tables/app_without_rag.py
   ```
2. Access the application at `http://localhost:8501`
3. Configure your database connection in the sidebar
4. Start asking questions in natural language about the Olist e-commerce data!

Example queries for Olist e-commerce data:

- "What is the average monthly active user count for each year?"
- "Show me the number of customers who made more than one purchase (repeat orders) for each year!"
- "What are the top product categories by revenue per month?"
- "Displays detailed information on the amount of usage for each type of payment for each year!"
- "Which one showed the most significant increase?" (follow-up question)
- "Break down those results by customer location" (contextual follow-up)

## ⚙️ How It Works

1. **Without RAG (Basic Approach)**:

   - User submits a natural language query
   - LLM converts it to SQL based on predefined schema information
   - SQL query is executed against the database
   - Results are passed back to LLM for human-friendly response generation
   - Conversation context is maintained for follow-up questions
2. **With RAG (Enhanced Approach)**:

   - Database schema descriptions are embedded and stored in a FAISS index
   - When a query is received, the system retrieves relevant schema information
   - This context-enriched information is sent to the LLM for SQL generation
   - The generated SQL is executed and results formatted into natural language
   - Chat history provides context for follow-up questions

## 🔮 Future Improvements

- Support for more complex database schemas and relationships
- Integration with additional LLM providers and local models (Deepseek-r1:1.5b and Qwen3:1.7b)
- Advanced data visualization of query results
- Fine-tuning models for improved SQL generation accuracy
- Support for database operations beyond querying (e.g., inserts, updates)
- Enhanced conversation memory management for longer discussions
- User authentication and personalized query history

## 📧 Contact

For questions or feedback, please contact: mcikalmerdeka@gmail.com
