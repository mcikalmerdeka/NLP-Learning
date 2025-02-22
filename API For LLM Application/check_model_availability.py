import ollama

try:
    user_query = "Can you explain to me about scikit-learn library?"

    response = ollama.chat(
        model="deepseek-r1:1.5b",
        messages=[{"role": "user", "content": user_query}]
    )
    print(response["message"]["content"])
except Exception as e:
    print(f"Error: {e}")