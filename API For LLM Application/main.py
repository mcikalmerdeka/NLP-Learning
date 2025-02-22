from fastapi import FastAPI, Depends, HTTPException, Header
from pydantic import BaseModel
import ollama
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set the API key and the rate limit
API_KEYS_CREDITS = {os.getenv("x_api_key"): 5} 

# Create the FastAPI app
app = FastAPI()

# Define the PromptRequest model
class PromptRequest(BaseModel):
    prompt: str

# Function to verify the API key
def veryify_api_key(x_api_key: str = Header(None)):
    credits = API_KEYS_CREDITS.get(x_api_key, 0) # Get the credits for the API key
    if credits <= 0:
        raise HTTPException(status_code=401, detail="Invalid API Key or no credits left")

    return x_api_key

# Define the root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the LLM API"}

# Define the generate endpoint
@app.post("/generate")
def generate(request: PromptRequest, x_api_key: str = Depends(veryify_api_key)):
    
    # Reduce the credits for the API key each time the endpoint is called
    API_KEYS_CREDITS[x_api_key] -= 1
    
    # Call the Ollama API
    response = ollama.chat(
        model="deepseek-r1:1.5b",
        messages=[{"role": "user", "content": request.prompt}]
    )

    # Return the response message only
    return {"response": response["message"]["content"]}
