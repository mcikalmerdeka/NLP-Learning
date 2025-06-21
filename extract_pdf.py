import pypdf

# Function to extract text from a PDF file
def extract_pdf_text(filename: str) -> str:
    try:
        with open(filename, 'rb') as file:
            reader = pypdf.PdfReader(file)
            text = '' # This will store the extracted text from each page
            for page_num, page in enumerate(reader.pages):      # Loop through each page in the PDF, page_num is the page number, page is the page object
                print(f"Extracting page {page_num + 1}...")
                text += f"\n--- PAGE {page_num + 1} ---\n"      # Add a separator between pages
                text += page.extract_text() + '\n'              # Extract text from the page and add it to the text variable
            return text
    except Exception as e:
        print(f'Error: {e}')
        return None

if __name__ == "__main__":
    filename = 'your_pdf_file_name.pdf'
    text = extract_pdf_text(filename)
    if text:
        # Print first 10000 characters to understand the content
        print("\n=== EXTRACTED TEXT ===")
        print(text[:10000])
        
        # Save full text to file for analysis
        with open('pdf_content.txt', 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"\nFull content saved to pdf_content.txt") 