import wikipedia
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Set up Wikipedia API access
def get_wikipedia_content(query):
    try:
        page = wikipedia.page(query)
        return page.content
    except:
        return ""

# Step 2: Initialize the retrieval and language model
model_name = "google/flan-t5-large"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 3: RAG function
def rag(query):
    # Retrieve information specifically about Paris
    content = get_wikipedia_content("Paris")
    
    if not content:
        return "No specific information found about {content}."
    
    sentences = content.split('. ')
    
    query_embedding = sentence_model.encode([query])
    sentence_embeddings = sentence_model.encode(sentences)
    
    similarities = cosine_similarity(query_embedding, sentence_embeddings)[0]
    
    top_indices = similarities.argsort()[-10:][::-1]  # Increased to top 10 sentences
    context = '. '.join([sentences[i] for i in top_indices])
    
    augmented_query = f"""Answer the following question using the provided context about Paris, the capital of France. Be concise and informative.
Context: {context}
Question: {query}
Answer:"""
    
    inputs = tokenizer(augmented_query, return_tensors="pt", max_length=1024, truncation=True)
    outputs = model.generate(
        **inputs,
        max_length=300,
        min_length=100,
        num_beams=5,
        no_repeat_ngram_size=3,
        early_stopping=True
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response

query = "What is the capital of France? Can you give me a short explanation of the history of that capital?"
response = rag(query)
print(response)