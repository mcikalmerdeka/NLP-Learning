from typing import Any, Dict
from graph.chains.generation import generation_chain
from graph.state import GraphState


def generate_node(state: GraphState) -> Dict[str, Any]:
    print("---GENERATE NODE---")
    
    # Initialize the state
    question = state["question"]
    documents = state["documents"]

    # Generate the response
    generation = generation_chain.invoke({"question": question, "context": documents})
    return {
        "generation": generation,
        "documents": documents
    }