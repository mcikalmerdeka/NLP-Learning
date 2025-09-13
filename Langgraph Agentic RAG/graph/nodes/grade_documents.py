from typing import Any, Dict

from graph.chains.retrieval_grader import retrieval_grader
from graph.state import GraphState

# Define the grade documents node
def grade_documents_node(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")

    # Get the question and the documents from the currentstate
    question = state["question"]
    documents = state["documents"]

    filtered_docs = [] # List to store the relevant documents if any
    web_search = False # Flag to check if we need to run web search, will toggle to true if any document is not relevant

    # Iterate over the documents and grade them
    for doc in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": doc.page_content} # Pass the question and the document to the grader
        )

        # Take the binary score from the grader
        grade = score.binary_score

        # Condition to check if the document is relevant
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(doc)

        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = True
            continue

    # Update the state with the filtered documents and the web search flag
    return {"documents": filtered_docs, "web_search": web_search}