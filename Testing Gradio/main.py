import gradio as gr
import os
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai import Agent

# ================================= First Example: Simple interface =================================
def greet(name, intensity):
    return "Hello, " + name + "!" * int(intensity)

# Create a Gradio interface
demo = gr.Interface(
    fn=greet,
    inputs=["text", "slider"],
    outputs=["text"],
)

demo.launch(share=True)

# ================================= Second Example: Simple chatbot using OpenAI API =================================

# Initialize Agent with OpenAI model
agent = Agent("gpt-4.1")

# Storage for actual pydantic-ai message history (recommended approach)
message_history_storage = []

def chat_with_gpt(message, history):
    """
    PROPER pydantic-ai approach using message_history parameter.
    This is the recommended way from the documentation.
    """
    global message_history_storage
    
    if not message_history_storage:
        # First conversation - no history
        result = agent.run_sync(message)

        # Store the new messages for next time
        message_history_storage.extend(result.new_messages())
        return result.output
    else:
        # Continue conversation with message history
        result = agent.run_sync(message, message_history=message_history_storage)
        
        # Add new messages to history
        message_history_storage.extend(result.new_messages())
        return result.output

# Create Gradio interface
demo = gr.ChatInterface(
    fn=chat_with_gpt,
    title="OpenAI GPT-4o Chatbot",
    description="Chat with GPT-4o Model"
)

demo.launch(share=True)

# ====================================== Third Example: Email writer and feedback generator  agent ======================================





