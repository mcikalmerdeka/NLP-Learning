from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize the model
llm = ChatOpenAI(model="gpt-4.1", temperature=0)

# Initialize answer grader class with pydantic structured output
class AnswerGrader(BaseModel):
    """Binary score for answer quality"""

    binary_score: bool = Field(description="Answer addresses / resolves the question, 'yes' if it is, 'no' if it is not")

# Initialize the answer grader llm
structured_llm_grader = llm.with_structured_output(AnswerGrader)

# Define the system prompt with the template
system_prompt = """You are a grader assessing whether an answer addresses / resolves a question \n 
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question.
     If the answer does not address / resolve the question, give a score of 'no'."""

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

# Create the answer grader chain
answer_grader: RunnableSequence = answer_prompt | structured_llm_grader
