import os
import datetime
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# """
# This is the more like the experimentation and debugging version of the code implementation for the CoverLetterGenerator class. 
# This version of the code implementation is more verbose and includes debugging information to help identify issues during the cover letter generation process.
# The code includes detailed logging of the steps involved in the cover letter generation process, including the retrieval of resume content, template content, and the final generated cover letter.
# The cover_letter_generation_2.py script is the more refined version of the code implementation without the debugging code.
# """

# Load environment variables
load_dotenv()

# Define the CoverLetterGenerator class
class CoverLetterGenerator:
    def __init__(self, resume_path, template_path, clear_existing=True): 
        
        # Initialize embeddings and vector store
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.persist_directory = "./chroma_db"
        
        # Clear existing vector store if requested
        if clear_existing and os.path.exists(self.persist_directory):
            import shutil
            print(f"Clearing existing vector store at {self.persist_directory}")
            shutil.rmtree(self.persist_directory)
        
        # Initialize LLM (using OpenAI's GPT-4o, but can be switched to Anthropic)
        self.llm = ChatOpenAI(model="gpt-4o")  # Alternatively use Anthropic: ChatAnthropic(model="claude-3-opus-20240229")

        # Load and process documents
        self.resume_retriever = self._prepare_resume(resume_path)
        self.cover_letter_example = self._load_cover_letter_example(template_path)
            
    # Prepare resume retriever
    def _prepare_resume(self, resume_path: str) -> None:
        """Prepare the resume retriever."""

        try:
            # Load resume document
            print(f"Loading resume from: {resume_path}")
            loader = PyPDFLoader(resume_path)
            resume_data = loader.load()
            print(f"Successfully loaded resume. Content length: {len(resume_data[0].page_content)}")

            # Split document into chunks
            print("Splitting resume into chunks")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(resume_data)
            print(f"Resume split into {len(splits)} chunks")

            # Create vector store for resume
            self.vector_store = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                collection_name="resume",
                persist_directory="./chroma_db"
            )

            return self.vector_store.as_retriever(search_kwargs={"k": 3})
        except Exception as e:
            print(f"Error preparing resume retriever: {str(e)}")
            raise

    # Load cover letter example
    def _load_cover_letter_example(self, template_path: str) -> str:
        """Load the cover letter example."""

        try:
            # Load cover letter example
            print(f"Loading cover letter example from: {template_path}")
            loader = PyPDFLoader(template_path)
            example = loader.load()
            print(f"Successfully loaded cover letter example. Content length: {len(example[0].page_content)}")
            
            return example[0].page_content  # Get the text content
        except Exception as e:
            print(f"Error loading cover letter example: {str(e)}")
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
        
        # Write the debug log
        with open(experiment_file, 'w', encoding='utf-8') as f:
            f.write("=== Cover Letter Generation Debug Log ===\n\n")
            
            # Retrieve resume content using the retriever
            f.write("1. RETRIEVING RESUME CONTENT\n")
            f.write("-" * 50 + "\n")
            relevant_docs = self.resume_retriever.invoke(job_description)
            
            if not relevant_docs:
                raise ValueError("No resume content found in vector store")
            
            f.write("Resume Content:\n")
            for doc in relevant_docs:
                f.write(doc.page_content + "\n")
                f.write(f"Metadata: {doc.metadata}\n")
                
            # Get template content
            f.write("\n2. TEMPLATE CONTENT\n")
            f.write("-" * 50 + "\n")
            f.write("Template Content:\n")
            f.write(self.cover_letter_example + "\n")
            
            # Prepare input for LLM
            input_data = {
                "resume_content": "\n".join(doc.page_content for doc in relevant_docs),
                "template": self.cover_letter_example,
                "job_description": job_description,
                "company_name": company_name
            }
            
            f.write("\n3. INPUT TO LLM\n")
            f.write("-" * 50 + "\n")
            f.write(f"Resume Content:\n{input_data['resume_content']}\n")
            f.write(f"Template:\n{input_data['template']}\n")
            f.write(f"Job Description:\n{input_data['job_description']}\n")
            f.write(f"Company Name: {input_data['company_name']}\n")
            
            # Generate the cover letter
            f.write("\n4. GENERATING COVER LETTER\n")
            f.write("-" * 50 + "\n")
            
            # Create the prompt
            template = """You're an expert cover letter writer. Use the following information to create a personalized cover letter:
            
            Resume Context: {context}
            
            Job Description: {job_description}
            
            Follow this exact format and style from the example cover letter:
            {example_style}

            Please ensure the cover letter does not exceed 500 words, which is the maximum word count for a single page Word document.
            
            Generate the cover letter:"""

            # Create the prompt
            prompt = ChatPromptTemplate.from_template(template)

            # Create the processing chain
            chain = (
                {
                    "context": self.resume_retriever,
                    "job_description": RunnablePassthrough(),
                    "example_style": lambda x: self.cover_letter_example
                }
                | prompt
                | self.llm
            )
            
            result = chain.invoke(job_description)
            
            f.write("\n5. GENERATED COVER LETTER\n")
            f.write("-" * 50 + "\n")
            f.write(result.content + "\n")
            
            print(f"Debug log saved to: {experiment_file}")
            
            return result.content

    # Define the save_cover_letter method
    def save_cover_letter(self, cover_letter: str, company_name: str, job_title: str) -> None:
        """Save the generated cover letter to a file."""

        # Create the directory if it doesn't exist
        os.makedirs("result_store", exist_ok=True)
        
        # Create the file path for each cover letter
        file_path = f"result_store/Cover Letter Muhammad Cikal Merdeka - {company_name} - {job_title}.txt"

        with open(file_path, "w", encoding='utf-8') as file:
            file.write(cover_letter)
        return file_path

# Code implementation
if __name__ == "__main__":
    generator = CoverLetterGenerator(
        clear_existing=True,
        resume_path="document_store/resume_example/Resume_Muhammad Cikal Merdeka.pdf",
        template_path="document_store/cover_letter_example/Cover Letter Muhammad Cikal Merdeka - Siloam Hospitals Group (Tbk)  - Data Analyst.pdf"  # Ensure this path is correct
    )
    
    # Read job description from a file
    with open("document_store/job_description.txt", "r", encoding='utf-8') as file:
        job_description = file.read()
    
    # Generate a single optimized cover letter based on the template
    result = generator.generate_cover_letter(job_description, "PT. BNI Life Insurance")
    
    # Save the generated cover letter
    generator.save_cover_letter(result, company_name="PT. BNI Life Insurance", job_title="Data Analytic (Project Based)")