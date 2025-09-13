from typing import List, TypedDict, Annotated
from langchain.schema import Document
import operator

# Define the state of the graph
class GraphState(TypedDict):
    """
    Represent the state of the graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search for extra relevant information or not (boolean)
        documents: list of documents
    """

    question: str
    generation: str
    web_search: bool
    documents: Annotated[List[Document], operator.add]