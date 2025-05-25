import os
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools

# Load the environment variables and configure the OpenAI API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize the Agent with instructions, tools, and a model
agno_assist = Agent(
    name="Agno Assistant",
    model=OpenAIChat(id="gpt-4.1", api_key=openai_api_key),
    description="""
    You are Agno AGI, an autonomous agent that can build agents using the Agno framework Your goal is to help developers understand and use Agno by providing
    explanations, working code examples, and optional visual and audio explanations of key concepts.
    """,
    instructions="Search the web for information about Agno",
    tools=[DuckDuckGoTools()],
    add_datetime_to_instructions=True,
    markdown=True
)

# Run the Agent
agno_assist.print_response("What is Agno?", stream=True)