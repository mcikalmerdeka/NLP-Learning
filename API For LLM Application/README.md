# LLM API Gateway

A lightweight API gateway that provides controlled access to large language models (LLMs) running locally with Ollama, featuring API key authentication, credit-based rate limiting, and a clean RESTful interface.

## 🚀 Features

- **Local LLM Integration**: Connects to locally running Ollama models
- **API Key Authentication**: Secures endpoints with API key verification
- **Credit-Based Rate Limiting**: Controls usage with a simple credit system
- **RESTful Design**: Clean, standards-compliant API architecture
- **FastAPI Framework**: High-performance, easy-to-extend web framework
- **Simple Testing Tools**: Includes scripts to verify API functionality

## 🛠️ Implementation Details

- Built with **FastAPI** for efficient API development
- Uses **Ollama** to interface with local LLM models
- Implements a straightforward credit system for usage tracking
- Features error handling for authentication and model failures
- Designed for easy extension with additional endpoints or models

## 🧩 How It Works

1. **Authentication**:
   - Client sends request with API key in header
   - Server verifies key validity and available credits
   - Unauthorized requests are rejected with 401 error

2. **Request Processing**:
   - Validated requests are forwarded to local Ollama instance
   - Prompt is sent to specified model (currently deepseek-r1:1.5b)
   - Response is captured from Ollama

3. **Credit Management**:
   - Each successful API call deducts one credit
   - When credits reach zero, further requests are rejected
   - Simple in-memory credit tracking (resets on server restart)

4. **Response Delivery**:
   - LLM response is extracted and formatted
   - Clean JSON response returned to client

## 📋 Requirements

- Python 3.8+
- Ollama (running locally with models installed)
- Required Python packages:
  - fastapi
  - uvicorn
  - ollama
  - python-dotenv
  - requests

## 🔧 Setup

1. Clone this repository
2. Create a `.env` file with your API key:
   ```
   x_api_key=your_secret_api_key
   ```
3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
4. Install Ollama and pull the required model:
   ```
   ollama pull deepseek-r1:1.5b
   ```

## 🚀 Usage

### Starting the API Server

```bash
uvicorn main:app --reload
```

The server will start at http://127.0.0.1:8000

### Making API Requests

#### Using Python
```python
import requests

url = "http://127.0.0.1:8000/generate"
headers = {
    "x_api_key": "your_secret_api_key",
    "Content-Type": "application/json"
}
data = {
    "prompt": "Explain what machine learning is"
}

response = requests.post(url, headers=headers, json=data)
print(response.json())
```

#### Using cURL
```bash
curl -X POST "http://127.0.0.1:8000/generate" \
     -H "x_api_key: your_secret_api_key" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Explain what machine learning is"}'
```

### Testing the API

Run the included test script:
```bash
python test_api.py
```

## 📊 Project Structure

```
├── main.py                      # FastAPI application
├── test_api.py                  # Test script for API
├── check_model_availability.py  # Script to check Ollama model
├── requirements.txt             # Python dependencies
├── .env                         # Environment variables (API key)
└── README.md                    # Documentation
```

## ⚠️ Considerations

- This implementation uses in-memory storage for API credits (resets on server restart)
- Currently configured for a single model (deepseek-r1:1.5b)
- No persistent logging of API usage
- Local LLM performance depends on your hardware capabilities

## 🔄 Customization Options

### Adding More Models

```python
@app.post("/generate/{model_name}")
def generate(model_name: str, request: PromptRequest, x_api_key: str = Depends(verify_api_key)):
    # Validate model name from a list of allowed models
    allowed_models = ["deepseek-r1:1.5b", "llama3:8b", "orca3:12b"]
    if model_name not in allowed_models:
        raise HTTPException(status_code=400, detail="Invalid model name")
    
    # Reduce credits and call the model
    API_KEYS_CREDITS[x_api_key] -= 1
    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": request.prompt}]
    )
    
    return {"response": response["message"]["content"]}
```

### Enhanced Credit System

```python
# More sophisticated credit system
API_KEYS = {
    "key1": {"credits": 100, "tier": "basic"},
    "key2": {"credits": 500, "tier": "premium"}
}

# Credit usage based on tier
TIER_CREDIT_COST = {
    "basic": 2,
    "premium": 1
}

def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    
    key_data = API_KEYS[x_api_key]
    cost = TIER_CREDIT_COST[key_data["tier"]]
    
    if key_data["credits"] < cost:
        raise HTTPException(status_code=403, detail="Insufficient credits")
    
    return x_api_key
```

## 📝 Future Enhancements

- Persistent storage for API keys and credits
- Multiple model support with model-specific credit costs
- Request logging and analytics
- Streaming response support
- Advanced rate limiting (time-based, IP-based)
- Admin dashboard for API key management 