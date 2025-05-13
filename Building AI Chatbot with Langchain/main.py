import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_tavily import TavilySearch

load_dotenv()

# Configure OpenAI and Tavily API
openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

# Define the tools for the agent to use
@tool
def say_hello(name: str) -> str:
    """Useful for saying hello to the user."""
    print("The tool has been called")
    return f"Hello {name}! How can I help you today?"

@tool
def calculator(a: float, b: float, operation: str = "add") -> str:
    """Useful for performing arithmetic calculations with numbers. 
    Supports operations: add, subtract, multiply, divide, power, modulo.""" # Docstring is important so that our LLM knows when to use the tool
    print("The calculator tool has been called") # This is just for debugging purposes
    
    operation = operation.lower()
    if operation == "add" or operation == "+":
        result = a + b
        return f"The sum of {a} and {b} is {result}."
    elif operation == "subtract" or operation == "-":
        result = a - b
        return f"The difference of {a} and {b} is {result}."
    elif operation == "multiply" or operation == "*":
        result = a * b
        return f"The product of {a} and {b} is {result}."
    elif operation == "divide" or operation == "/":
        if b == 0:
            return "Error: Cannot divide by zero."
        result = a / b
        return f"The division of {a} by {b} is {result}."
    elif operation == "power" or operation == "^":
        result = a ** b
        return f"{a} raised to the power of {b} is {result}."
    elif operation == "modulo" or operation == "%":
        if b == 0:
            return "Error: Cannot perform modulo with zero."
        result = a % b
        return f"The remainder of {a} divided by {b} is {result}."
    else:
        return f"Unsupported operation: {operation}. Please use add, subtract, multiply, divide, power, or modulo."

@tool
def tavily_search(query: str) -> str:
    """Useful for searching the web for current information and news."""
    print("The tavily search tool has been called")
    try:
        # Check if API key is available
        if not tavily_api_key:
            return "Error: TAVILY_API_KEY is not set in environment variables. Please add your Tavily API key to use this search feature."
            
        tavily_search_tool = TavilySearch(max_results=5)
        results = tavily_search_tool.invoke(query)
        
        # Format results in a more readable way
        formatted_results = ""
        if isinstance(results, list):
            for i, result in enumerate(results, 1):
                if isinstance(result, dict):
                    title = result.get("title", "No title")
                    content = result.get("content", "No content")
                    url = result.get("url", "No URL")
                    formatted_results += f"{i}. {title}\n{content}\nSource: {url}\n\n"
                else:
                    formatted_results += f"{i}. {str(result)}\n\n"
        else:
            formatted_results = str(results)
            
        return formatted_results
    except Exception as e:
        return f"Error performing search: {str(e)}"

def main():
    # Initialize the model
    model = ChatOpenAI(model="gpt-4o",
                       api_key=openai_api_key,
                       temperature=0)
    
    # Initialize the tools for the agent to use
    tools = [say_hello, calculator, tavily_search]

    # Initialize the agent
    agent_executor = create_react_agent(model, tools)

    # Print the welcome message at the beginning of the conversation
    print("Welcome! I'm your AI assistant. How can I help you today?. Type 'quit' to exit.")
    print("You can ask me to perform calculations or chat with me about anything.")

    while True:
        user_input = input("\nYou: ").strip()

        # Check if the user wants to quit
        if user_input.lower() == "quit":
            print("Exiting...")
            break

        print("\nAssistant: ", end="")
        for chunk in agent_executor.stream(
            {"messages": [HumanMessage(content=user_input)]}
        ):
            if "agent" in chunk and "messages" in chunk["agent"]:

                # Print the assistant's response word by word (looks like ChatGPT)
                for message in chunk["agent"]["messages"]:
                    print(message.content, end="")

        print()

if __name__ == "__main__":
    main()
