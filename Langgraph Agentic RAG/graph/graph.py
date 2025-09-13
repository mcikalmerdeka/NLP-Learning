import os
from dotenv import load_dotenv
from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import hallucination_grader
from graph.consts import RETRIEVE, GRADE_DOCUMENTS, WEB_SEARCH, GENERATE
from graph.nodes import retrieve_node, grade_documents_node, web_search_node, generate_node
from graph.chains.router import question_router, RouterQuery
from graph.state import GraphState
from langgraph.graph import START, StateGraph, END

load_dotenv()

# Define the function to handle conditional for grade documents node to web search node or generate node
def decide_to_generate(state: GraphState) -> bool:
    """
    Check if the documents are relevant to the question
    If not, set the web_search flag to True
    """
    print("---ASSES GRADED DOCUMENTS---")

    # Define the condition to check if the documents are relevant to the question
    if state["web_search"]:
        print("---DECISION: NOT ALL DOCUMENTS ARE RELEVANT TO THE QUESTION---")
        print("---ROUTING TO WEB SEARCH NODE---")
        return WEB_SEARCH
    else:
        print("---DECISION: ALL DOCUMENTS ARE RELEVANT TO THE QUESTION---")
        print("---ROUTING TO GENERATE NODE---")
        return GENERATE
    
# Define the function to handle conditional for grade documents node to answer grader node or generate node
def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    print("---CHECKING HALLUCINATIONS---")

    # Get the state question, documents, and generation
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    # Check if the generation is grounded in the documents
    hallucination_score = hallucination_grader.invoke({"documents": documents, "generation": generation})
    if hallucination_score.binary_score:
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")
        answer_score = answer_grader.invoke({"question": question, "generation": generation})

        # Check if the generation addresses the question
        if answer_score.binary_score:
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "Useful" # Return useful if the generation addresses the question
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "Not useful" # Return not useful if the generation does not address the question
        
    # If the generation is not grounded in the documents, return not supported
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RETRYING---")
        return "Not supported"        

# Define the function to route the question to the most relevant path (web search or vectorstore)
def route_question(state: GraphState) -> str:
    """
    Route the initial question to the most relevant path (web search or vectorstore)
    """
    print("---ROUTING QUESTION---")
    question = state["question"]
    source: RouterQuery = question_router.invoke({"question": question})

    # Define the condition to check if the question is to be routed to web search or vectorstore
    if source.datasource == "websearch":
        print("---DECISION: ROUTING TO WEB SEARCH NODE---")
        return "websearch"
    elif source.datasource == "vectorstore":
        print("---DECISION: ROUTING TO RETRIEVE NODE---")
        return "vectorstore"

# Define the graph
graph = StateGraph(GraphState)

# Add the nodes to the graph
graph.add_node(RETRIEVE, retrieve_node)
graph.add_node(GRADE_DOCUMENTS, grade_documents_node)
graph.add_node(GENERATE, generate_node)
graph.add_node(WEB_SEARCH, web_search_node)

# Add the edges to the graph
graph.add_edge(START, RETRIEVE) # The original edge from start to retrieve node
graph.set_conditional_entry_point( # Set the conditional entry point to route the question to the most relevant path
    path=route_question,
    path_map={
        "websearch": WEB_SEARCH, # If the question is to be routed to web search, route to web search node
        "vectorstore": RETRIEVE # If the question is to be routed to vectorstore, route to retrieve node
    }
)
graph.add_edge(RETRIEVE, GRADE_DOCUMENTS)

## Conditional edges for grading the documents
graph.add_conditional_edges(
    source=GRADE_DOCUMENTS,
    path=decide_to_generate,
    path_map={
        WEB_SEARCH: WEB_SEARCH,
        GENERATE: GENERATE
    }
)

## Conditional edges for grading the generation
graph.add_conditional_edges(
    source=GENERATE,
    path=grade_generation_grounded_in_documents_and_question,
    path_map={
        "Useful": END, # If the generation is useful, end the graph
        "Not useful": GENERATE, # If the generation is not useful, retry the generation
        "Not supported": WEB_SEARCH # If the documents are not relevant to the question, retry the web search
    }
)

graph.add_edge(WEB_SEARCH, GENERATE)
graph.add_edge(GENERATE, END)

# Compile the graph and save the graph to png
rag_app = graph.compile()
rag_app.get_graph().draw_mermaid_png(output_file_path="langgraph_agentic_rag/complete_rag_graph.png")