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

# Define the CoverLetterGenerator class
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

    # Define the load_documents method
    def load_documents(self, resume_path: str, template_path: str) -> None:
        """Load and process resume and cover letter template."""
        try:
            # Load resume
            print(f"Loading resume from: {resume_path}")
            if resume_path.endswith('.pdf'):
                resume_loader = PyPDFLoader(resume_path)
            else:
                resume_loader = TextLoader(resume_path)
            resume_doc = resume_loader.load()
            
            if not resume_doc:
                raise ValueError(f"Failed to load resume from {resume_path}")
            
            print(f"Successfully loaded resume. Content length: {len(resume_doc[0].page_content)}")
            
            # Load single cover letter template
            print(f"Loading template from: {template_path}")
            if template_path.endswith('.pdf'):
                template_loader = PyPDFLoader(template_path)
            else:
                template_loader = TextLoader(template_path)
            template_doc = template_loader.load()
            
            if not template_doc:
                raise ValueError(f"Failed to load template from {template_path}")
            
            print(f"Successfully loaded template from: {template_path}")
            
            # Process all documents
            all_docs = resume_doc + template_doc
            
            splits = self.text_splitter.split_documents(all_docs)
            
            self.vector_store = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                persist_directory="./data"
            )
                
        except Exception as e:
            print(f"Error loading documents: {str(e)}")
            raise

    # Define the generate_cover_letter method
    def generate_cover_letter(self, job_description: str, company_name: str) -> str:
        """Generate a personalized cover letter based on the template."""
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Please load documents first.")
        
        # Retrieve resume content with more specific query
        relevant_docs = self.vector_store.similarity_search(
            job_description,
            k=1,
            filter={"source": "resume"}
        )
        
        if not relevant_docs:
            raise ValueError("No resume content found in vector store")
        
        print(f"Found {len(relevant_docs)} relevant resume sections")
        
        # Retrieve the cover letter template
        template_doc = self.vector_store.similarity_search(
            "cover letter template",
            k=1,  # Get the single template
            filter={"source": "template"}
        )
        
        # Create a combined prompt that uses the template as an example
        prompt = PromptTemplate(
            input_variables=["resume_content", "template", "job_description", "company_name"],
            template="""
            Using the following information:
            
            Resume Content: {resume_content}
            
            Example Cover Letter Template:
            {template}
            
            Job Description: {job_description}
            Company: {company_name}
            
            Study the style and approach of the example template provided, and generate ONE optimized cover letter that:
            1. Incorporates the best elements from the example template
            2. Highlights relevant experience from the resume that matches the job description
            3. Demonstrates enthusiasm for the specific company
            4. Maintains a professional tone
            5. Has a maximum length of 500 words (a single page maximum)
            """
        )
        
        # Create a language model chain
        chain = prompt | self.llm
        
        return chain.invoke({
            "resume_content": "\n".join(doc.page_content for doc in relevant_docs),
            "template": template_doc[0].page_content if template_doc else "",
            "job_description": job_description,
            "company_name": company_name
        }).content

    # Define the save_cover_letter method
    def save_cover_letter(self, cover_letter: str, output_path: str) -> None:
        """Save the generated cover letter to a file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
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
generator.save_cover_letter(cover_letter, "result_store/generated_cover_letter.txt")