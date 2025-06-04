from __future__ import annotations

import asyncio
from dataclasses import dataclass

import gradio as gr
from pydantic import BaseModel, EmailStr
from pydantic_ai import Agent, format_as_xml
from pydantic_ai.models.openai import OpenAIModel

# # ================================= First Example: Simple interface =================================
# def greet(name, intensity):
#     return "Hello, " + name + "!" * int(intensity)

# # Create a Gradio interface
# demo = gr.Interface(
#     fn=greet,
#     inputs=["text", "slider"],
#     outputs=["text"],
# )

# demo.launch(share=True)

# # ================================= Second Example: Simple chatbot using OpenAI API =================================

# # Initialize Agent with OpenAI model
# agent = Agent("gpt-4.1")

# # Storage for actual pydantic-ai message history (recommended approach)
# message_history_storage = []

# def chat_with_gpt(message, history):
#     """
#     PROPER pydantic-ai approach using message_history parameter.
#     This is the recommended way from the documentation.
#     """
#     global message_history_storage
    
#     if not message_history_storage:
#         # First conversation - no history
#         result = agent.run_sync(message)

#         # Store the new messages for next time
#         message_history_storage.extend(result.new_messages())
#         return result.output
#     else:
#         # Continue conversation with message history
#         result = agent.run_sync(message, message_history=message_history_storage)
        
#         # Add new messages to history
#         message_history_storage.extend(result.new_messages())
#         return result.output

# # Create Gradio interface
# demo = gr.ChatInterface(
#     fn=chat_with_gpt,
#     title="OpenAI GPT-4o Chatbot",
#     description="Chat with GPT-4o Model"
# )

# demo.launch()

# ====================================== Third Example: Email writer and feedback generator  agent ======================================

# Setup input and output data types
@dataclass
class User:
    name: str
    email: EmailStr
    interests: list[str]

@dataclass
class Email:
    subject: str
    body: str

class EmailRequiresWrite(BaseModel):
    feedback: str

class EmailOk(BaseModel):
    pass

# Setup email writer and feedback generator agent
email_writer_agent = Agent(
    model=OpenAIModel(model_name="gpt-4.1"),
    output_type=Email, # by setting the output_type, we are telling the agent what type of data it should constantly return
    system_prompt=(
        "You are an email writer.",
        "Write a welcome email to our new members joining my AI Agent blog.",
        "The first email must exclude interest."
    ),
)

feedback_agent = Agent(
    model=OpenAIModel(model_name="gpt-4.1"),
    output_type=EmailRequiresWrite | EmailOk, # This is a union type, so the agent can return either EmailRequiresWrite or EmailOk

    # Note that the system prompt here is really important since the while loop break is based on the output of the feedback agent.
    # If the system prompt is not strict enough, the while loop will break too early and the email will not be finalized.
    system_prompt=(
        "You are a feedback generator for an email writer.",
        "Review the email and provide feedback on what to improve.",
        "Be strict about including user interests - emails MUST mention the user's interests meaningfully.",
        "Request rewrites if interests are missing, generic, or poorly integrated.",
        "Only approve with EmailOk if the email properly addresses the user's specific interests."
    ),
)

# Create helper function to format user data as XLM (making it more organized)
def create_user_xml(user: User) -> str:
    return f'''
    <user>
        <name>{user.name}</name>
        <email>{user.email}</email>
        <interests>{', '.join(user.interests)}</interests>
    </user>
    '''

# Create async function to handle email generation and feedback workflow 
async def handle_email_flow(name, email, interests=''):
    user = User(name=name, email=email, interests=interests.split(','))
    messages = []
    feedback = None

    # Continue orchestrating email generation and feedback process until the email is finalized
    while True:

        # Create the prompt based on if feedback is available or not
        prompt = f"Write a welcome email: \n{create_user_xml(user)}" if not feedback else f"Refine the email based on the feedback: \n{feedback}\n{create_user_xml(user)}"
        
        # Update the input fields with the current state (showing email generation is in progress)
        email_subject, email_body, feedback = "Generating email...", "", ""

        # Yield in Python makes this function a generator, allowing it to pause execution and return values incrementally.
        # This sends the current email generation state (subject, body, feedback) back to 
        # the Gradio interface immediately, enabling real-time UI updates while the async process continues
        yield email_subject, email_body, feedback

        # Run the email writer agent with the prompt and message history
        result = await email_writer_agent.run(prompt, message_history=messages)
        messages += result.all_messages()
        email = result.output
        email_subject, email_body = email.subject, email.body

        # If feedback is not available, show that the draft is generated and ready for feedback
        if not feedback:
            yield email_subject, email_body, "Draft generated, submitting for feedback..."
        else:
            yield email_subject, email_body, "Refinement complete, submitting reviewing..."

        # Create the feedback prompt based on the email and user data
        feedback_prompt = format_as_xml({'user': user, 'email': email})

        # Run the feedback agent with the feedback prompt
        feedback_result = await feedback_agent.run(feedback_prompt)

        # If the feedback is not good, show the feedback and wait for 7 seconds
        if isinstance(feedback_result.output, EmailRequiresWrite):
            feedback = feedback_result.output.feedback

            yield email_subject, email_body, f"Feedback received: {feedback}"

            await asyncio.sleep(5) # Wait for 5 seconds to allow the user to read the feedback

        # If the feedback is good, break the loop and show the final email
        elif isinstance(feedback_result.output, EmailOk):
            yield email_subject, email_body, "Email finalized successfully"
            break

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## AI Email Feedback Agent")

    # Create the input fields for the user's name, email, and interests
    with gr.Row():
        name_input = gr.Textbox(label="Name", placeholder="Enter your name")
        email_input = gr.Textbox(label="Email", placeholder="Enter your email")
        interests_input = gr.Textbox(label="Interests", placeholder="Enter your interests (comma separated)")

    # Create the output fields for the email subject and body
    email_subject_output = gr.Textbox(label="Email Subject", interactive=False)
    email_body_output = gr.Textbox(label="Email Body", interactive=False, lines=5)

    # Create the feedback display field
    with gr.Row():
        feedback_display = gr.Textbox(label="Feedback", interactive=False, lines=3)

    # Create the generate button
    generate_button = gr.Button("Generate Email")

    # When the generate button is clicked, run the handle_email_flow function
    generate_button.click(
        fn=handle_email_flow,
        inputs=[name_input, email_input, interests_input],
        outputs=[email_subject_output, email_body_output, feedback_display]
    )

# Launch the Gradio interface
if __name__ == "__main__":
    demo.launch()
