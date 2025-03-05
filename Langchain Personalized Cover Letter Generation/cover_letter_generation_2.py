import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader         # Load PDF documents for processing
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Split text into smaller chunks
from langchain_chroma import Chroma                                  # Create vector stores for documents 
from langchain_openai import OpenAIEmbeddings, ChatOpenAI            # Load OpenAI models for text generation
from langchain_anthropic import ChatAnthropic                        # Load Anthropic models for text generation
from langchain_core.prompts import ChatPromptTemplate                # Create a prompt template 
from langchain_core.runnables import RunnablePassthrough             # Create a runnable to pass through data

# Packages for PDF generation
from reportlab.lib.pagesizes import letter                           # Define page size for PDF
from reportlab.lib.styles import ParagraphStyle                      # Define style for paragraphs 
from reportlab.platypus import SimpleDocTemplate, Paragraph          # Create PDF document and paragraphs
from reportlab.lib.enums import TA_JUSTIFY                           # Define alignment for paragraphs
from reportlab.pdfbase import pdfmetrics                             # Register fonts for PDF generation
from reportlab.pdfbase.ttfonts import TTFont                         # Register TrueType fonts for PDF generation

# """
# This is the final code that i will be using to generate a cover letter for a job application.
# Although in terms of the code implementation the result kinda similar as the previous version, this version of the code has been refined to remove the debugging code and provide a cleaner and more concise implementation.
# """

# Load environment variables
load_dotenv()

# Define the CoverLetterGenerator class
class CoverLetterGenerator:
    def __init__(self):
        # Initialize components with the best available embedding model
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.llm = ChatOpenAI(model="gpt-4o")
        # self.llm = ChatAnthropic(model="claude-3-7-sonnet-latest")
        
        # Load and process documents
        self.resume_retriever = self._prepare_resume()
        self.cover_letter_example = self._load_cover_letter_example()

    # Prepare resume retriever
    def _prepare_resume(self):
        # Load resume document
        loader = PyPDFLoader("document_store/resume_example/Resume_Muhammad Cikal Merdeka.pdf")
        resume_data = loader.load()

        # Split document into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(resume_data)

        # Create vector store for resume
        vector_store = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            collection_name="resume_2",
            persist_directory="./chroma_db_2"
        )

        return vector_store.as_retriever(search_kwargs={"k": 3})

    # Load cover letter example
    def _load_cover_letter_example(self):
        # Load cover letter example
        loader = PyPDFLoader("document_store/cover_letter_example/Cover Letter Muhammad Cikal Merdeka - GoTo Group - Business Intelligence Analyst.pdf")
        example = loader.load()
        return example[0].page_content  # Get the text content

    # Generate cover letter
    def generate_cover_letter(self, job_description: str):

        # Custom prompt template combining both sources
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
        
        return chain.invoke(job_description)
    
    # Save the generated cover letter
    def save_cover_letter(self, cover_letter: str, company_name: str, job_title: str):

        # Create the file path for each cover letter
        file_path = f"result_store/Cover Letter Muhammad Cikal Merdeka - {company_name} - {job_title}.pdf"
        
        # Register Arial font
        pdfmetrics.registerFont(TTFont('Arial', 'arial.ttf'))
        
        # Create document with metadata
        doc = SimpleDocTemplate(
            file_path,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72,
            title=f"Cover Letter Muhammad Cikal Merdeka - {company_name} - {job_title}",
            author="Muhammad Cikal Merdeka",
            subject="Job Application Cover Letter",
        )
        
        # Define style for justified text with 1.15 line spacing
        style = ParagraphStyle(
            'CustomStyle',
            fontName='Arial',
            fontSize=11,
            leading=14.15,
            alignment=TA_JUSTIFY,
            spaceAfter=14.15  # Same as leading for consistent paragraph spacing
        )
        
        # Split content into paragraphs and create story
        story = []
        paragraphs = cover_letter.content.split('\n\n')
        for para in paragraphs:
            if para.strip():
                p = Paragraph(para.replace('\n', ' '), style)
                story.append(p)
        
        # Build PDF
        doc.build(story)
        return file_path

# Code implementation
if __name__ == "__main__":
    generator = CoverLetterGenerator()
    
    # Read job description from a file
    with open("document_store/job_description.txt", "r", encoding='utf-8') as file:
        job_description = file.read()
    
    # Generate a single optimized cover letter based on the template
    result = generator.generate_cover_letter(job_description)
    
    # Save the generated cover letter
    output_path = generator.save_cover_letter(result, company_name="PT Enseval Medika Prima", job_title="Data Analyst")