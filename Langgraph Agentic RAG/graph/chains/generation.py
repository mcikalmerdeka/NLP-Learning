from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize the model
llm = ChatOpenAI(model="gpt-4.1", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

# Get the prompt template from the hub
# prompt_template = hub.pull("rlm/rag-prompt")

# Modified rlm/rag-prompt template to include instructions of filtering out the irrelevant information
prompt_template = ChatPromptTemplate.from_template("""
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Additional Instructions:
{additional_instructions}
Answer:
""")

prompt_template = prompt_template.partial(
    additional_instructions="""
    Filter out the irrelevant information from the context.
    Exclude the following elements from Tavily search results:
    - Image links and URLs (e.g., .jpg, .png, .gif, .svg)
    - Code blocks and snippets
    - JSON data structures
    - HTML markup and CSS
    - Navigation elements, headers, footers
    - Advertisement content
    - Social media buttons and widgets
    - Metadata like timestamps, author info, tags
    - Cookie notices and privacy policies
    Focus only on the main textual content relevant to answering the question.
    """
)

# Define the generation chain
generation_chain = prompt_template | llm | StrOutputParser()