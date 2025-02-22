import requests
from dotenv import load_dotenv
import os

load_dotenv()

# Define the endpoint URL - remove the query parameter
url = "http://127.0.0.1:8000/generate"

# Define the headers
headers = {
    "x_api_key": os.getenv("x_api_key"),
    "Content-Type": "application/json"
}

# Define the request body
data = {
    "prompt": "Can you explain to me about scikit-learn library?"
}

# Make post request with json data
response = requests.post(url, headers=headers, json=data)
print(response.json())