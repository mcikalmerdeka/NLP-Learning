from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()

# Define the router class with pydantic structured output
class RouterQuery(BaseModel):
    """Router user query to the most relevant datasource"""
    
    datasource: Literal["vectorstore", "websearch"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )

# Define the llm with structured output
llm = ChatOpenAI(model="gpt-4.1", api_key=os.getenv("OPENAI_API_KEY"), temperature=0)
structured_llm_router = llm.with_structured_output(RouterQuery)

# Define the system prompt with the template
system_prompt = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on these topics. For all else, use web-search."""

route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{question}"),
    ]
)

# Create the question router chain
question_router = route_prompt | structured_llm_router