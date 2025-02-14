import os
import datetime
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
    def __init__(self, clear_existing=True):
        load_dotenv()
        
        self.embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"), model="text-embedding-3-large")
        self.vector_store = None
        self.persist_directory = "./data"
        
        # Clear existing vector store if requested
        if clear_existing and os.path.exists(self.persist_directory):
            import shutil
            print(f"Clearing existing vector store at {self.persist_directory}")
            shutil.rmtree(self.persist_directory)
        
        self.llm = ChatAnthropic(
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            model_name="claude-3-5-sonnet-20241022"
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Smaller chunks capture more granular information
            chunk_overlap=400,  # Large overlap ensures context continuity
            length_function=len,
            separators=["\n\n", "\n", " ", ""],  # More granular separation
            is_separator_regex=False
        )

    def load_documents(self, resume_path: str, template_path: str) -> None:
        """Load and process resume and cover letter template."""
        try:
            # Load resume
            print(f"Loading resume from: {resume_path}")
            if resume_path.endswith('.pdf'):
                resume_loader = PyPDFLoader(resume_path)
            else:
                resume_loader = TextLoader(resume_path)
            resume_docs = resume_loader.load()
            
            # Add metadata to resume documents
            for doc in resume_docs:
                doc.metadata["source_type"] = "resume"
            
            if not resume_docs:
                raise ValueError(f"Failed to load resume from {resume_path}")
            
            print(f"Successfully loaded resume. Content length: {len(resume_docs[0].page_content)}")
            
            # Load template
            print(f"Loading template from: {template_path}")
            if template_path.endswith('.pdf'):
                template_loader = PyPDFLoader(template_path)
            else:
                template_loader = TextLoader(template_path)
            template_docs = template_loader.load()
            
            # Add metadata to template documents
            for doc in template_docs:
                doc.metadata["source_type"] = "template"
            
            if not template_docs:
                raise ValueError(f"Failed to load template from {template_path}")
            
            print(f"Successfully loaded template. Content length: {len(template_docs[0].page_content)}")
            
            # Process all documents
            all_docs = resume_docs + template_docs
            splits = self.text_splitter.split_documents(all_docs)
            
            # Create new vector store
            self.vector_store = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            
            print(f"Vector store created with {len(splits)} document chunks")
                
        except Exception as e:
            print(f"Error loading documents: {str(e)}")
            raise
    
    # Define the generate_cover_letter method
    def generate_cover_letter(self, job_description: str, company_name: str) -> str:
        """Generate a personalized cover letter based on the template."""

        # Create experiment_store directory if it doesn't exist
        experiment_dir = "experiment_debugging_store"
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Create a timestamp for the experiment
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_file = os.path.join(experiment_dir, f"debug_log_{timestamp}.txt")
        
        with open(experiment_file, 'w', encoding='utf-8') as f:
            f.write("=== Cover Letter Generation Debug Log ===\n\n")
            
            if not self.vector_store:
                raise ValueError("Vector store not initialized. Please load documents first.")
            
            # Retrieve resume content
            f.write("1. RETRIEVING RESUME CONTENT\n")
            f.write("-" * 50 + "\n")
            relevant_docs = self.vector_store.similarity_search(
                job_description,
                k=2,
                filter={"source_type": "resume"}
            )
            
            if not relevant_docs:
                raise ValueError("No resume content found in vector store")
            
            f.write(f"Found {len(relevant_docs)} relevant resume sections\n\n")
            for i, doc in enumerate(relevant_docs):
                f.write(f"Resume Section {i+1}:\n")
                f.write(doc.page_content + "\n")
                f.write(f"Metadata: {doc.metadata}\n\n")
            
            # Retrieve template
            f.write("\n2. RETRIEVING TEMPLATE\n")
            f.write("-" * 50 + "\n")
            template_docs = self.vector_store.similarity_search(
                "cover letter template",
                k=1,
                filter={"source_type": "template"}
            )
            
            if template_docs:
                f.write("Template Content:\n")
                f.write(template_docs[0].page_content + "\n")
                f.write(f"Metadata: {template_docs[0].metadata}\n")
            
            # Prepare input for LLM
            input_data = {
                "resume_content": "\n".join(doc.page_content for doc in relevant_docs),
                "template": template_docs[0].page_content if template_docs else "",
                "job_description": job_description,
                "company_name": company_name
            }
            
            f.write("\n3. INPUT TO LLM\n")
            f.write("-" * 50 + "\n")
            for key, value in input_data.items():
                f.write(f"\n{key.upper()}:\n")
                f.write(value + "\n")
                f.write("-" * 25 + "\n")
            
            # Generate the cover letter
            f.write("\n4. GENERATING COVER LETTER\n")
            f.write("-" * 50 + "\n")
            
            # Create the prompt
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
            
            # Generate the cover letter
            result = chain.invoke(input_data).content
            
            f.write("\n5. GENERATED COVER LETTER\n")
            f.write("-" * 50 + "\n")
            f.write(result + "\n")
            
            print(f"Debug log saved to: {experiment_file}")
            
            return result

    # Define the save_cover_letter method
    def save_cover_letter(self, cover_letter: str, output_path: str) -> None:
        """Save the generated cover letter to a file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(cover_letter)

# Initialize the generator
generator = CoverLetterGenerator(clear_existing=True)

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
    company_name="PT. BNI Life Insurance"
)

# Save the generated cover letter
generator.save_cover_letter(cover_letter, "result_store/generated_cover_letter.txt")