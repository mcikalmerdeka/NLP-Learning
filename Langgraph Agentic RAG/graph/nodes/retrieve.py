from typing import Any, Dict, List

from graph.state import GraphState
from ingestion import retriever

# Define the retrieve node
def retrieve_node(state: GraphState) -> Dict[str, Any]:
    print(f"Retrieving documents for question: {state['question']}")

    # Get the question from the state
    question = state['question']

    # Retrieve the relevant documents based on the question
    documents = retriever.invoke(question)

    # Update the field of document in our current state
    return {"documents": documents}