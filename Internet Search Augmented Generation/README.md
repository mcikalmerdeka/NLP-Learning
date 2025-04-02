# Internet Search Augmented Generation (iSAG)

An experimental project that implements internet search capabilities for large language models, similar to browsing features offered by major AI providers like OpenAI, Anthropic, and Google.

## 🚀 Features

- **Real-time Web Search**: Integrates Serper API to perform Google searches
- **Contextual Response Generation**: Enhances LLM responses with fresh web data
- **Search Result Formatting**: Structures search results for optimal context
- **Claude Integration**: Leverages Anthropic's Claude models for response generation
- **Configurable Parameters**: Adjustable search depth and response temperature
- **Fallback Mechanisms**: Graceful degradation if search capabilities fail

## 🛠️ Implementation Details

- Built with Python for easy integration and extension
- Uses **Anthropic's API** for Claude model access
- Implements **Serper API** for Google search capabilities
- Features modular design with clear separation of concerns
- Well-documented code with type hints for better maintainability

## 🧩 How It Works

1. **Query Processing**:

   - User submits a query requiring up-to-date information
   - System determines if web search is needed
2. **Web Search**:

   - Query is sent to Serper API (Google Search)
   - Top results are retrieved (configurable number)
   - Results are formatted into structured context
3. **Enhanced Response Generation**:

   - Search results are incorporated into prompt context
   - Claude model generates response with awareness of current information
   - Citations are included where appropriate
4. **Response Delivery**:

   - Final response combines web knowledge with LLM capabilities
   - User receives up-to-date, factual information

## 📋 Requirements

- Python 3.8+
- Anthropic API key
- Serper API key (for Google Search functionality)
- Required Python packages:
  - anthropic
  - requests
  - python-dotenv

## 🔧 Setup

1. Clone this repository
2. Create a `.env` file with your API keys:
   ```
   ANTHROPIC_API_KEY=your_anthropic_api_key
   SERPER_API_KEY=your_serper_api_key
   ```
3. Install the required packages:
   ```
   pip install anthropic requests python-dotenv
   ```

## 🚀 Usage

```python
from internet_search_llm import SearchAugmentedGeneration

# Initialize the system
sag = SearchAugmentedGeneration()

# Process a query with search augmentation
response = sag.process_query(
    "What are the latest developments in quantum computing?",
    search_enabled=True,  # Enable web search
    num_results=3  # Number of search results to use
)

print(response)
```

## 🔄 Customization Options

You can customize the behavior with these parameters:

- **model**: Choose from available Claude models (default: "claude-3-opus-20240229")
- **temperature**: Adjust creativity of responses (default: 0.7)
- **num_results**: Number of search results to include (default: 3)
- **search_enabled**: Toggle search functionality on/off

```python
# Example with customization
response = sag.process_query(
    "What is the current state of AI regulation?",
    search_enabled=True,
    num_results=5,  # More search results
)

# Using the generate_response method directly
response = sag.generate_response(
    user_query="Explain recent advancements in CRISPR",
    search_results=custom_results,  # Your own search results
    model="claude-3-5-sonnet-20241022",  # Different model
    temperature=0.5  # Less creative
)
```

## 📊 Project Structure

```
├── internet_search_llm.py   # Main implementation
├── .env                     # API keys (create this file)
└── README.md                # Documentation
```

## ⚠️ Considerations

- Serper API has usage limitations based on your plan
- Web search results may contain inaccuracies
- The quality of responses depends on both search results and LLM capabilities
- Always verify critical information from multiple sources

## 🔒 Privacy and Ethics

- User queries are sent to both Serper (Google Search) and Anthropic
- Consider privacy implications when processing sensitive queries
- Be aware of potential biases in search results and LLM responses
- Use responsibly and in accordance with API providers' terms of service

## 📝 Future Enhancements

- Multiple search provider support (Bing, DuckDuckGo)
- Web page content extraction for deeper context
- Image search capabilities
- Support for additional LLM providers
- Enhanced citation formatting
- Result caching for improved performance
