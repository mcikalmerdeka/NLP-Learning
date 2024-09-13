import numpy as np
import torch
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity

class RAG:
    def __init__(self, retriever_model_name: str, generator_model_name: str):
        # Initialize tokenizers and models
        self.retriever_tokenizer = AutoTokenizer.from_pretrained(retriever_model_name)
        self.retriever_model = AutoModel.from_pretrained(retriever_model_name)
        
        self.generator_tokenizer = AutoTokenizer.from_pretrained(generator_model_name)
        self.generator_model = AutoModelForCausalLM.from_pretrained(generator_model_name)
        
        self.documents = []
        self.document_embeddings = []

    def add_documents(self, documents: List[str]):
        self.documents.extend(documents)
        new_embeddings = self._embed_documents(documents)
        self.document_embeddings.extend(new_embeddings)

    def _embed_documents(self, documents: List[str]) -> List[np.ndarray]:
        embeddings = []
        for doc in documents:
            inputs = self.retriever_tokenizer(doc, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.retriever_model(**inputs)
            embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
        return embeddings

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        query_embedding = self._embed_documents([query])[0]
        similarities = cosine_similarity([query_embedding], self.document_embeddings)[0]
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        return [(self.documents[i], similarities[i]) for i in top_k_indices]

    def generate(self, query: str, retrieved_docs: List[Tuple[str, float]]) -> str:
        context = " ".join([doc for doc, _ in retrieved_docs])
        prompt = f"Context: {context}\n\nQuery: {query}\n\nAnswer:"
        
        inputs = self.generator_tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self.generator_model.generate(**inputs, max_length=150, num_return_sequences=1)
        
        return self.generator_tokenizer.decode(outputs[0], skip_special_tokens=True)

    def query(self, query: str, k: int = 5) -> str:
        retrieved_docs = self.retrieve(query, k)
        return self.generate(query, retrieved_docs)

# Example usage
if __name__ == "__main__":
    # Initialize RAG with appropriate model names
    rag = RAG("sentence-transformers/all-MiniLM-L6-v2", "gpt2")

    # Add extensive personal documents about Alex
    documents = [
        "My name is Alex Johnson, born on May 15, 1990, in Seattle, Washington.",
        "I work as a senior software engineer at TechCorp, specializing in machine learning and cloud architecture.",
        "I have a golden retriever named Max who is 5 years old. I adopted him from a local shelter when he was a puppy.",
        "My favorite book is 'The Hitchhiker's Guide to the Galaxy' by Douglas Adams. I've read it at least 10 times.",
        "I graduated from Stanford University with a Bachelor's degree in Computer Science in 2012 and a Master's in Artificial Intelligence in 2014.",
        "I'm severely allergic to peanuts and shellfish. I always carry an EpiPen with me just in case.",
        "My emergency contact is my sister, Emily Johnson, reachable at 555-0123. She lives in Portland, Oregon.",
        "I speak fluent English, intermediate Spanish, and basic Mandarin Chinese which I've been learning for the past year.",
        "My current project at work involves developing a machine learning model for predictive maintenance in industrial equipment.",
        "I volunteer at the local animal shelter every other Saturday, mainly helping with dog walks and socialization.",
        "My long-term goal is to start my own tech company focused on sustainable energy solutions, particularly in solar power optimization.",
        "I'm an avid rock climber and go bouldering at the local gym three times a week. I've completed several outdoor climbing trips in Yosemite.",
        "I play guitar in a local indie rock band called 'The Binary Beats'. We perform at local venues about once a month.",
        "I'm a vegetarian and have been for the past 8 years. I love cooking and often experiment with new plant-based recipes.",
        "I've run two marathons, with my best time being 3 hours and 45 minutes in the Chicago Marathon last year.",
        "I invest regularly in index funds and a few selected tech stocks. I'm particularly interested in companies working on renewable energy.",
        "I've been practicing meditation for the past 3 years and try to meditate for at least 20 minutes every morning.",
        "I'm passionate about space exploration and have a small telescope that I use for stargazing. I'm a member of the local astronomy club.",
        "I have a collection of over 500 vinyl records, mostly classic rock and jazz. My prized possession is a first pressing of Pink Floyd's 'Dark Side of the Moon'.",
        "I've traveled to 25 countries so far. My favorite trip was a month-long backpacking journey through Southeast Asia.",
        "I'm currently learning woodworking as a hobby. I've made a few small furniture pieces for my apartment.",
        "I donate blood regularly, every 8 weeks. My blood type is O negative, which makes me a universal donor.",
        "I have a fear of heights, which I'm working on overcoming through my rock climbing hobby.",
        "I'm an early riser, usually waking up at 5:30 AM to start my day with a run or yoga session.",
        "I'm a big fan of sci-fi TV shows. My all-time favorites are 'Firefly', 'Black Mirror', and 'The Expanse'.",
        "I have a scar on my left knee from a skiing accident when I was 16. It required 12 stitches.",
        "I'm colorblind, specifically red-green colorblind. It doesn't affect my work much, but I sometimes need help choosing matching clothes.",
        "I've been a mentor in a local STEM program for high school students for the past 2 years.",
        "I have a small cactus collection in my apartment, with about 15 different species that I've been growing for years.",
        "My go-to coffee order is a double shot espresso, but I limit myself to one per day in the morning."
    ]
    rag.add_documents(documents)

    # Query the RAG system
    queries = [
        "What is Alex's job and specialization?",
        "Can you describe Alex's pet?",
        "What are Alex's food allergies and dietary preferences?",
        "What languages does Alex speak and at what levels?",
        "What are Alex's hobbies and interests?",
        "What is Alex's educational background?",
        "What kind of music is Alex involved with?",
        "What is Alex's fitness routine like?",
        "What are some of Alex's travel experiences?",
        "What is Alex's long-term career goal?",
        "Does Alex have any phobias or medical conditions?",
        "What is Alex's investment strategy?",
        "What kind of volunteer work does Alex do?",
        "What is Alex's daily routine like?",
        "What are some of Alex's favorite books or TV shows?"
    ]

    for query in queries:
        result = rag.query(query)
        print(f"Query: {query}")
        print(f"Answer: {result}")
        print()