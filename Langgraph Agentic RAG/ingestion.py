import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Define the url data source
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# Load the documents and store them in a list
doc_batches = [WebBaseLoader(url).load() for url in urls]
docs = [doc for doc_batch in doc_batches for doc in doc_batch]

# Equivalent for loop version:
# docs = []
# for doc_batch in doc_batches:
#     for doc in doc_batch:
#         docs.append(doc)

# Split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=100)
docs_split = text_splitter.split_documents(docs)

# Store the documents in a vector database (run this once)
vectorstore = Chroma.from_documents(
    documents=docs_split,
    collection_name="rag-chroma",
    embedding=OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key),
    persist_directory="langgraph_agentic_rag/.chroma_db",
)

# # Turn the vectorstore into a retriever (to perform similarity search)
# retriever = vectorstore.as_retriever()

# We can also use this approach after we already created the vectorstore (because the vectorstore variable is commented)
retriever = Chroma(
    collection_name="rag-chroma",
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key),
    persist_directory="langgraph_agentic_rag/.chroma_db",
).as_retriever()