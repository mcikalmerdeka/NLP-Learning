# AI Chatbot with LangChain

A conversational AI chatbot built with LangChain that can perform calculations with several operations and search the internet using the Tavily API.

## Features

- Advanced arithmetic calculations (add, subtract, multiply, divide, power, modulo)
- Greeting users
- Real-time web searching for current information using Tavily API
- Interactive command-line interface

## Requirements

```
langchain>=0.3.25
langchain-community>=0.3.24
langchain-openai>=0.3.16
langchain-tavily>=0.1.6
langgraph>=0.4.3
python-dotenv>=1.1.0
```

## Setup

1. Clone this repository
2. Create a `.env` file in the root directory with the following content:

```
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

3. Install required packages:

Using uv (recommended):
```bash
# Install uv if you don't have it
curl -sSf https://astral.sh/uv/install.sh | bash

# Install dependencies
uv add -e .
```

or using pip:
```bash
pip install -e .
```

or install packages directly:
```bash
# Using uv
uv add langchain langchain-community langchain-openai langchain-tavily langgraph python-dotenv

# Using pip
pip install langchain langchain-community langchain-openai langchain-tavily langgraph python-dotenv
```

4. Run the chatbot:

```bash
python main.py
```

## Getting API Keys

- **OpenAI API key**: Sign up at [OpenAI Platform](https://platform.openai.com/api-keys)
- **Tavily API key**: Sign up at [Tavily](https://tavily.com/) to access their search API

## Usage

Once running, you can:
- Type your queries or questions utilizing the model knowledge capabilities
- Type 'quit' to exit the program

- Several basic tools usage:

    - Ask for calculations with various operations (addition, substraction, multiplication, division, power, and modulo) and conversational context 
    (e.g., "If the current time is 13.00 PM and the passenger beside me said the train supposed to be at Jakarta at 16.00 PM with the speed of 230 km/hour, what is the distance between those 2 city in km?")

    - Request web searches (e.g., "Who is the current president and vice president of Indonesia?")

