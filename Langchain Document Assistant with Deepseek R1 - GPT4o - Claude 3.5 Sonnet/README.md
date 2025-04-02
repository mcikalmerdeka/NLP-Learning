# DocuMind AI: Intelligent Document Assistant

A powerful LLM-powered document assistant that extracts information from various documents, supporting files containing both text and image components. This application enables seamless data retrieval and analysis from your content library.

## 🚀 Features

- **Multiple LLM Support**: Integrates with GPT-4o, Claude 3.5 Sonnet, and DeepSeek-R1 models
- **Document Processing**: Efficiently processes PDF documents, extracting and indexing their content
- **Semantic Search**: Uses vector embeddings to find the most relevant information from your documents
- **Conversational Interface**: Clean, intuitive chat interface to ask questions about your documents
- **Custom Styling**: Dark mode interface with responsive design

## 🛠️ Implementation Details

- Built with **LangChain** for document processing and retrieval augmented generation (RAG)
- Utilizes **Streamlit** for the web interface
- Supports local deployment with **Ollama** for DeepSeek models
- Implements efficient document chunking and indexing for optimal search performance
- Configurable model selection to balance between performance and cost

## 📋 Requirements

- Python 3.8+
- Streamlit
- LangChain and related packages
- API keys for OpenAI and Anthropic (for cloud models)
- Ollama installation (for local models)

## 🔧 Setup

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ANTHROPIC_API_KEY=your_anthropic_api_key
   ```
4. For local models, install Ollama and pull the DeepSeek models:
   ```
   ollama pull deepseek-r1:1.5b
   ollama pull deepseek-r1:3b
   ```

## 🚀 Usage

### GPT-4o and Claude 3.5 Version

```
python rag_gpt_claude.py
```

### DeepSeek R1 Version

```
python rag_deepseek_r1.py
```

1. Upload a PDF document using the file uploader
2. Ask questions about the document content
3. Receive AI-generated responses based on the document's content

## 📊 Comparison of Available Models

| Model             | Type  | Best For                                  | Required Setup      |
| ----------------- | ----- | ----------------------------------------- | ------------------- |
| GPT-4o            | Cloud | High accuracy, complex queries            | OpenAI API key      |
| Claude 3.5 Sonnet | Cloud | Nuanced understanding, detailed responses | Anthropic API key   |
| DeepSeek R1       | Local | Privacy, offline use, faster responses    | Ollama installation |

## 🔒 Privacy

- Cloud models (GPT-4o, Claude 3.5): Document content is sent to external APIs
- Local models (DeepSeek R1): All processing happens locally with no data leaving your machine
