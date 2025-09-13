"""
Important thing to remember: To run this go to the root directory and run the command:
pytest -s -v
"""
from dotenv import load_dotenv
from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from graph.chains.generation import generation_chain
from ingestion import retriever
from graph.chains.hallucination_grader import hallucination_grader, HallucinationGrader
from graph.chains.router import question_router, RouterQuery

load_dotenv()
import pprint

# Define test for yes answer
def test_retrieval_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_text = docs[0].page_content # Get the most relevant document

    res: GradeDocuments = retrieval_grader.invoke({"question": question, "document": doc_text})

    assert res.binary_score == "yes"

# Define test for no answer
def test_retrieval_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_text = docs[0].page_content # Get the most relevant document

    res: GradeDocuments = retrieval_grader.invoke({"question": "how to make a pizza", "document": doc_text})

    assert res.binary_score == "no"

# Define test for generation chain
def test_generation_chain() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    res: str = generation_chain.invoke({"question": question, "context": docs})
    pprint.pprint(res)

# Define the test for hallucination grader yes (means the answer is grounded in the facts)
def test_hallucination_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)

    generation = generation_chain.invoke({"question": question, "context": docs})

    res: HallucinationGrader = hallucination_grader.invoke({"documents": docs, "generation": generation})

    assert res.binary_score

# Define the test for hallucination grader no (means the answer is not grounded in the facts)
def test_hallucination_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)

    res: HallucinationGrader = hallucination_grader.invoke(
        {
            "documents": docs,
            "generation": "In order to make pizza we need to first start with the dough",
        }
    )
    assert not res.binary_score

# Define the test for question router
def test_router_to_vectorstore() -> None:
    question = "agent memory"
    res: RouterQuery = question_router.invoke({"question": question})
    assert res.datasource == "vectorstore"

# Define the test for question router to web search
def test_router_to_web_search() -> None:
    question = "how to make a pizza"
    res: RouterQuery = question_router.invoke({"question": question})
    assert res.datasource == "websearch"