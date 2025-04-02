# AI-Powered Cover Letter Generator

An intelligent tool that creates tailored cover letters for specific job openings by analyzing user-provided resumes/CVs and preferred writing styles, generating personalized content that aligns perfectly with targeted job descriptions.

## 🚀 Features

- **Resume Analysis**: Automatically extracts relevant skills and experiences from your resume
- **Style Matching**: Maintains your personal writing style based on previous cover letters
- **Job-Specific Customization**: Tailors content to match specific job requirements
- **PDF Generation**: Creates professional-looking, formatted PDF documents
- **Multiple AI Models**: Supports both OpenAI (GPT-4o) and Anthropic (Claude) models
- **Vector Storage**: Efficiently indexes and retrieves relevant resume sections

## 🛠️ Implementation Details

- Built with **LangChain** for document processing and LLM integration
- Uses **ChromaDB** for vector storage and semantic search
- Leverages **OpenAI Embeddings** for text analysis (text-embedding-3-large)
- Implements **ReportLab** for PDF document generation
- Features modular design for easy customization and extension

## 🧩 How It Works

1. **Document Processing**: 
   - Loads your resume from PDF
   - Splits content into manageable chunks
   - Creates vector embeddings for semantic search

2. **Style Learning**:
   - Analyzes your existing cover letter for tone and format
   - Extracts structural elements and writing patterns

3. **Content Generation**:
   - Retrieves relevant resume sections based on job description
   - Generates tailored content using LLM (GPT-4o or Claude)
   - Maintains your personal writing style and format

4. **PDF Creation**:
   - Formats content with professional styling
   - Outputs a ready-to-submit PDF document

## 📋 Requirements

- Python 3.8+
- OpenAI API key for embeddings and generation
- (Optional) Anthropic API key for Claude models
- Required Python packages (see requirements.txt)

## 🔧 Setup

1. Clone this repository
2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
4. Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ANTHROPIC_API_KEY=your_anthropic_api_key  # Optional
   ```
5. Place your documents in the appropriate directories:
   - Resume: `document_store/resume_example/`
   - Cover letter template: `document_store/cover_letter_example/`
   - Job description: `document_store/job_description.txt`

## 🚀 Usage

### Using the Standard Version

```python
from cover_letter_generation import CoverLetterGenerator

# Initialize the generator with your documents
generator = CoverLetterGenerator(
    resume_path="document_store/resume_example/your_resume.pdf",
    template_path="document_store/cover_letter_example/your_template.pdf",
    clear_existing=True  # Set to False to use existing vector store
)

# Read job description
with open("document_store/job_description.txt", "r", encoding='utf-8') as file:
    job_description = file.read()

# Generate cover letter
result = generator.generate_cover_letter(job_description, "Company Name")

# Save to file
generator.save_cover_letter(result, company_name="Company Name", job_title="Job Title")
```

### Using the Enhanced Version (with PDF Generation)

```python
from cover_letter_generation_2 import CoverLetterGenerator

# Initialize the generator (uses default file paths)
generator = CoverLetterGenerator()

# Read job description
with open("document_store/job_description.txt", "r", encoding='utf-8') as file:
    job_description = file.read()

# Generate cover letter
result = generator.generate_cover_letter(job_description)

# Save as PDF
output_path = generator.save_cover_letter(
    result, 
    company_name="Company Name", 
    job_title="Job Title"
)
print(f"Cover letter saved to: {output_path}")
```

## 📊 Project Structure

```
├── cover_letter_generation.py     # Standard implementation with debugging
├── cover_letter_generation_2.py   # Enhanced implementation with PDF output
├── requirements.txt               # Required Python packages
├── .env                           # API keys (create this file)
├── document_store/                # Store for input documents
│   ├── resume_example/            # Your resume(s)
│   ├── cover_letter_example/      # Cover letter template(s)
│   └── job_description.txt        # Job posting description
├── chroma_db/                     # Vector database (created automatically)
├── result_store/                  # Generated cover letters
└── README.md                      # This documentation
```

## ⚠️ Considerations

- Ensure your `.env` file with API keys is not committed to version control
- The quality of generated cover letters depends on:
  - The completeness of your resume
  - The quality of your existing cover letter template
  - The specificity of the job description

## 🔒 Privacy

- All processing happens through API calls to OpenAI/Anthropic
- Your resume and cover letter data will be sent to these services
- Generated content is stored locally in the `result_store` directory 