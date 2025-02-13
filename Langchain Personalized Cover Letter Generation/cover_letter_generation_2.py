import os
from typing import Dict, List
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import Document

class CoverLetterGenerator:
    def __init__(self):
        load_dotenv()
        
        # Initialize embeddings and vector store
        self.embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"), model="text-embedding-3-large")
        self.vector_store = None
        
        # Initialize LLM (using Anthropic's Claude, but can be switched to OpenAI)
        self.llm = ChatAnthropic(
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            model_name="claude-3-5-sonnet-20241022"
        )
        
        # Text splitter for document processing
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    def load_documents(self, resume_path: str, template_path: str) -> None:
        """Load and process resume and cover letter template."""
        # Load resume (supports PDF and text files)
        if resume_path.endswith('.pdf'):
            resume_loader = PyPDFLoader(resume_path)
        else:
            resume_loader = TextLoader(resume_path)
        resume_doc = resume_loader.load()
        
        if template_path.endswith('.pdf'):
            template_loader = PyPDFLoader(template_path)
        else:
            template_loader = TextLoader(template_path)
        template_doc = template_loader.load()
        
        # Split documents into chunks
        all_docs = resume_doc + template_doc
        for doc in all_docs:
            doc.metadata['source'] = 'resume' if doc in resume_doc else 'template'
        
        splits = self.text_splitter.split_documents(all_docs)
        
        # Store in vector database
        self.vector_store = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory="./data"
        )

    def generate_cover_letter(self, job_description: str, company_name: str) -> str:
        """Generate a personalized cover letter based on the job description."""
        # Retrieve relevant information from vector store
        relevant_docs = self.vector_store.similarity_search(
            job_description,
            k=1,
            filter={"source": "resume"}
        )
        
        template_docs = self.vector_store.similarity_search(
            "cover letter template",
            k=1,
            filter={"source": "template"}
        )
        
        # Create prompt template
        prompt = PromptTemplate(
            input_variables=["resume_content", "template", "job_description", "company_name"],
            template="""
            Using the following information:
            
            Resume Content: {resume_content}
            Cover Letter Template: {template}
            Job Description: {job_description}
            Company: {company_name}
            
            Generate a personalized cover letter that:
            1. Follows the style and format of the template
            2. Highlights relevant experience from the resume that matches the job description
            3. Demonstrates enthusiasm for the specific company
            4. Maintains a professional tone
            5. Is concise but compelling
            """
        )
        
        # Create a language model chain
        chain = prompt | self.llm
        
        return chain.invoke({
            "resume_content": "\n".join(doc.page_content for doc in relevant_docs),
            "template": template_docs[0].page_content if template_docs else "",
            "job_description": job_description,
            "company_name": company_name
        }).content

    def save_cover_letter(self, cover_letter: str, output_path: str) -> None:
        """Save the generated cover letter to a file."""
        with open(output_path, 'w') as f:
            f.write(cover_letter)

# Initialize the generator
generator = CoverLetterGenerator()

# Load your documents with a single template
generator.load_documents(
    resume_path="document_store/resume_example/Resume_Muhammad Cikal Merdeka.pdf",
    template_path="document_store/cover_letter_example/Cover Letter Muhammad Cikal Merdeka - Siloam Hospitals Group (Tbk)  - Data Analyst.pdf"
)

# Read job description from a file
with open("document_store/job_description.txt", "r", encoding='utf-8') as file:
    job_description = file.read()

# Generate a single optimized cover letter based on the template
cover_letter = generator.generate_cover_letter(
    job_description=job_description,
    company_name="Aigens"
)

# Save the generated cover letter
generator.save_cover_letter(cover_letter, "result_store/generated_cover_letter_2.txt")