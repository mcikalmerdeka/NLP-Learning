import gradio as gr
import os
from openai import OpenAI

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

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Create a function to chat with GPT-4.1
def chat_with_gpt(message, history):
    # Convert history to OpenAI format
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    
    messages.append({"role": "user", "content": message})
    
    # Get response from OpenAI
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=messages,
        max_tokens=1000,
        temperature=0.7
    )
    
    return response.choices[0].message.content

# Create Gradio interface
demo = gr.ChatInterface(
    fn=chat_with_gpt,
    title="OpenAI GPT-4.1 Chatbot",
    description="Chat with GPT-4.1 Model"
)

demo.launch(share=True)