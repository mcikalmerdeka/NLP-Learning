from typing import Any, Dict
import os
import sys
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain.schema import Document

# Try to import from the project structure, fallback to path modification if needed
try:
    from graph.state import GraphState
except ModuleNotFoundError:
    # Add the project root to Python path for direct execution
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, '..', '..')
    sys.path.insert(0, os.path.abspath(project_root))
    from graph.state import GraphState

load_dotenv()

# Initialize the TavilySearch client
web_search_tool = TavilySearch(max_results=2, api_key=os.getenv("TAVILY_API_KEY"))

# Define the web search node
def web_search_node(state: GraphState) -> Dict[str, Any]:
    """
    Search the web for the latest information on the given query.
    """
    print("---WEB SEARCH---")

    # Get the question and document from the state
    question = state["question"]

    if "documents" in state: # if the route to web search in first time then give error
        documents = state["documents"]
    else:
        documents = None
    
    # Invoke the web search tool
    tavily_results = web_search_tool.invoke({"query": question})["results"]
    
    # Join the content from all search results
    joined_tavily_result = "\n".join(
        [tavily_result["content"] for tavily_result in tavily_results]
    )

    # Create a Document object from the joined results
    web_results = Document(page_content=joined_tavily_result)

    # Set-up condition to check if the documents are not None
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]

    # Return the updated state
    return {"documents": documents}

if __name__ == "__main__":
    state = {"question": "agentic memory", "document": None}
    web_search_node(state)
