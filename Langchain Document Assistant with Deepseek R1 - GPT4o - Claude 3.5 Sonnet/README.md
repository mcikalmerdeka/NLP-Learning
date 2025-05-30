# DocuChat AI: Intelligent Document Assistant

![Project Header](./assets/Project%20Header.jpg)

A powerful LLM-powered document assistant that extracts information from various documents, supporting files containing both text and image components. This application enables seamless data retrieval and analysis from your content library.

## 🚀 Features

- **Multiple LLM Support**: Integrates with GPT-4o, GPT-4.1, Claude 3.7 Sonnet, and DeepSeek-R1 models
- **Document Processing**: Efficiently processes PDF documents, extracting and indexing their content
- **Semantic Search**: Uses vector embeddings to find the most relevant information from your documents
- **Conversational Interface**: Clean, intuitive chat interface to ask questions about your documents
- **Custom Styling**: Dark mode interface with responsive design
- **Multiple Vector Store Options**: Supports both ChromaDB (current approach) and InMemoryVectorStore (legacy approach)

## 🛠️ Implementation Details

- Built with **LangChain** framework for document processing and retrieval augmented generation (RAG)
- Utilizes **Streamlit** for the web interface
- Implements chunking with `RecursiveCharacterTextSplitter` for optimal document segmentation
- Supports two vector store implementations:
  - **ChromaDB** (current approach): Persistent vector database for better scalability and performance
  - **InMemoryVectorStore** (legacy approach): Simple in-memory storage for lightweight applications
- Supports semantic search with both cloud and local embedding models
- Handles PDF documents using `PyMuPDFLoader` (GPT/Claude version) and `PDFPlumberLoader` (DeepSeek version)
- Includes chat history management for continuous conversations

## 📋 Requirements

- Python 3.11+
- Required packages (from pyproject.toml):
  ```
  langchain-anthropic>=0.3.13
  langchain-community>=0.3.24
  langchain-core>=0.3.59
  langchain-ollama>=0.3.2
  langchain-openai>=0.3.16
  langchain-text-splitters>=0.3.8
  pymupdf>=1.25.5
  python-dotenv>=1.1.0
  streamlit>=1.45.1
  ```
- API keys for OpenAI and Anthropic (for cloud models)
- Ollama installation (for local models)

## 🔧 Setup

1. Clone this repository
2. Install the required packages using uv package manager:
   ```
   uv pip install .
   ```
3. Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ANTHROPIC_API_KEY=your_anthropic_api_key
   ```
4. For local models, install Ollama and pull the DeepSeek models:
   ```
   ollama pull deepseek-r1:1.5b
   ```
5. Create a directory for document storage:
   ```
   mkdir -p document_store/pdfs
   ```

## 🚀 Usage

### GPT-4o and Claude 3.7 Version (ChromaDB)

```
streamlit run rag_gpt_claude.py
```

### GPT-4o and Claude 3.7 Version (Legacy InMemoryVectorStore)

```
streamlit run rag_gpt_claude_old_approach.py
```

### DeepSeek R1 Version

```
streamlit run rag_deepseek_r1.py
```

1. Upload a PDF document using the file uploader
2. Wait for the document to be processed, chunked, and indexed
3. Ask questions about the document content using the chat interface
4. Receive AI-generated responses based on the document's content
5. Use the "Clear Chat History" button in the sidebar to start a new conversation

## 📊 Comparison of Available Models

| Model             | Type  | Best For                                  | Required Setup      | Implementation                            |
| ----------------- | ----- | ----------------------------------------- | ------------------- | ----------------------------------------- |
| GPT-4o            | Cloud | High accuracy, complex queries            | OpenAI API key      | ChatOpenAI with temperature=0             |
| GPT-4.1           | Cloud | Latest OpenAI model with improved reasoning | OpenAI API key    | ChatOpenAI with temperature=0             |
| Claude 3.7 Sonnet | Cloud | Nuanced understanding, detailed responses | Anthropic API key   | ChatAnthropic with temperature=0          |
| DeepSeek R1       | Local | Privacy, offline use, faster responses    | Ollama installation | OllamaLLM with the deepseek-r1:1.5b model |

## 🔍 Embedding Models

The application uses different embedding models based on the version:

- **Cloud Version**: OpenAI's `text-embedding-3-large` model for high-quality embeddings
- **Local Version**: Can use either OpenAI embeddings or local Ollama embeddings with DeepSeek models

## 🔒 Privacy

- Cloud models (GPT-4o, Claude 3.7): Document content is sent to external APIs
- Local models (DeepSeek R1): All processing happens locally with no data leaving your machine

## 🗄️ Vector Storage Options

### ChromaDB (Current Approach)

- **Persistent Storage**: Document embeddings are saved to disk
- **Better Scalability**: Can handle larger document collections
- **Improved Performance**: Optimized for efficient retrieval
- **Implementation**: Used in `rag_gpt_claude.py`

### InMemoryVectorStore (Legacy Approach)

- **Lightweight**: Simple in-memory storage with no persistence
- **Fast for Small Datasets**: Efficient for smaller document collections
- **Implementation**: Used in `rag_gpt_claude_old_approach.py`

## Project Screenshots

In this demo I used the BPS Palu City data of [population and employment](https://github.com/mcikalmerdeka/NLP-Learning/blob/main/Langchain%20Document%20Assistant%20with%20Deepseek%20R1%20-%20GPT4o%20-%20Claude%203.5%20Sonnet/document_store/pdfs/Statistik%20Penduduk%20dan%20Ketenagakerjaan%20Kota%20Palu%202025.pdf) from the official report of "Kota Palu Dalam Angka 2025" which you can access in this `document_store` folder in this repo. You may use several example files that I stored there or use your own PDFs.

### Document Upload Interface

![Upload Document](./assets/Project%20Screenshot%201.png)

*The document upload interface allows users to select and upload PDF files for analysis. The system processes the document and prepares it for question answering.*

### Initial Question and Answer

![Document Chat](./assets/Project%20Screenshot%202.png)

*After document processing, users can ask specific questions about the content. The AI assistant retrieves relevant information from the document and provides comprehensive answers based on the document context.*

### Follow-up Question Capabilities

![Follow-up Questions](./assets/Project%20Screenshot%203.png)

*The system supports follow-up questions, maintaining context from previous interactions. This allows for a natural conversation flow while exploring document content in greater depth.*