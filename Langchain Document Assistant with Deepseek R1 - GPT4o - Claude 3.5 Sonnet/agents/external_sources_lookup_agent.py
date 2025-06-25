import os
import sys

# Add the parent directory to the Python path so we can import from tools
# This must be done BEFORE importing tools  
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain import hub
from tools.tools import search_external_resources
from langchain.prompts import PromptTemplate

load_dotenv()

# Create a function to create a tool for searching external resources
def lookup(information_to_lookup: str) -> str:

    # Initialize the LLM
    llm = ChatOpenAI(model="gpt-4.1", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

    # Create a prompt template
    template = """
    Given the information {information_to_lookup}, I want you to search for external resources about it.
    Your answer should only contain the results of the search and nothing else.
    """

    prompt_template = PromptTemplate(
        template=template,
        input_variables=["information_to_lookup"]
    )

    # Create a tool list
    external_sources_lookup_tool = Tool(
        name="Search External Resources",
        func=search_external_resources,

        description="Useful when you need to find up-to-date information about a person, company, or any other topic that is not available in the database"
    )

    tools = [external_sources_lookup_tool]

    # Create a react prompt
    react_prompt = hub.pull("hwchase17/react")

    # Create a react agent with llm
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=react_prompt
    )

    # Create an agent executor from the agent
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

    # Run the agent executor
    result = agent_executor.invoke(
        input={"input": prompt_template.format_prompt(information_to_lookup=information_to_lookup)}
    )

    return result["output"]

# Only use this if you are running the script directly
if __name__ == "__main__":
    # Run the lookup function
    result = lookup("Iran israel war updates")
    print(result)