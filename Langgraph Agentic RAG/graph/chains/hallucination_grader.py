from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize the model
llm = ChatOpenAI(model="gpt-4.1", temperature=0)

# Initialize hallucination grader class with pydantic structured output
class HallucinationGrader(BaseModel):
    """Binary score for hallucination present in generation answer"""

    binary_score: bool = Field(description="Answer is gorunded in the facts, 'yes' if it is, 'no' if it is not")

# Initialize the hallucination grader llm
structured_llm_grader = llm.with_structured_output(HallucinationGrader)

# Define the system prompt with the template
system_prompt = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

# Create the hallucination grader chain
hallucination_grader: RunnableSequence = hallucination_prompt | structured_llm_grader

