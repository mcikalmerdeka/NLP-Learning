from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# Define the model
llm = ChatOpenAI(model="gpt-4.1", api_key=os.getenv("OPENAI_API_KEY"), temperature=0)

# Define the grading model
class GradeDocuments(BaseModel):
    """
    Binary score for relevance check on retrieved documents.
    """

    # Define the fields of the model
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' if relevant, 'no' if not relevant")

# Define the grader llm
## What's happen under the hood is that the llm will use function calling and for every call we are going to get a structured output in pydantic object
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# Define the system prompt and the prompt template
system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
    Dont translate the score into 1 or 0, just return the score as a string."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

# Define the final retrieval grader chain
retrieval_grader = grade_prompt | structured_llm_grader