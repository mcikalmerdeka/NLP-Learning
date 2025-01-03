import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import schedule
import time
import os
from dotenv import load_dotenv
from openai import OpenAI
import base64

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_KEY"))

# Step 1: Load and process data
def load_data(file_path):
    data = pd.read_csv(file_path, parse_dates=["Date"])
    data["Week"] = data["Date"].dt.isocalendar().week
    return data

# Step 2: Perform analysis
def analyze_data(data):
    weekly_sales = data.groupby("Week").agg({"Sales": "sum"}).reset_index()
    return weekly_sales

# Step 3: Generate visualizations
def generate_visualizations(data, output_path):
    plt.figure(figsize=(20, 6))
    sns.barplot(data=data, x="Week", y="Sales")
    sns.lineplot(data=data, x="Week", y="Sales", color="red")
    plt.annotate("Sales peak", xy=(data["Sales"].idxmax(), 
                                   data["Sales"].max()), 
                                   xytext=(data["Sales"].idxmax() - 2, data["Sales"].max() + 1000))
    plt.title("Weekly Sales Trend")
    plt.savefig(output_path)

def encode_image_to_base64(image_path):
    """Convert image to base64 string"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

def get_ai_analysis(image_path):
    """Get AI analysis of the sales trend visualization"""
    base64_image = encode_image_to_base64(image_path)
    if not base64_image:
        return "Error: Could not encode image for analysis"
    
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Please analyze this sales trend visualization and provide insights about:\n"
                                  "1. Overall trend\n"
                                  "2. Notable peaks and troughs\n"
                                  "3. Potential seasonal patterns\n"
                                  "4. Recommendations based on the data"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in AI analysis: {e}")
        return f"Error getting AI analysis: {e}"

# Modify the PDFReport class to include AI analysis
class PDFReport(FPDF):
    def header(self):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, "Weekly Sales Report", align="C", ln=True)

    def add_image(self, img_path):
        self.image(img_path, x=10, y=None, w=190)
    
    def add_ai_analysis(self, analysis_text):
        self.add_page()
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, "AI Analysis:", ln=True)
        self.set_font("Arial", size=10)
        
        # Add some padding
        self.ln(5)
        
        # Handle long text properly
        text_width = self.w - 20  # 10mm margins on each side
        self.set_x(10)
        
        # Split analysis into paragraphs and add them to PDF
        paragraphs = analysis_text.split('\n')
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if paragraph:
                self.multi_cell(text_width, 6, paragraph)
                self.ln(3)

def create_pdf_report(analysis_data, img_path, output_path):
    pdf = PDFReport()
    
    # First page with sales data
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, "Sales Analysis:", ln=True)
    
    # Add table headers
    pdf.set_font("Arial", "B", 10)
    pdf.cell(30, 10, "Week", 1)
    pdf.cell(60, 10, "Sales", 1)
    pdf.ln()
    
    # Add table data
    pdf.set_font("Arial", size=10)
    for index, row in analysis_data.iterrows():
        pdf.cell(30, 10, f"{row['Week']}", 1)
        pdf.cell(60, 10, f"{row['Sales']:,.2f}", 1)
        pdf.ln()
    
    # Add visualization
    pdf.add_page()
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Sales Trend Visualization:", ln=True)
    pdf.add_image(img_path)
    
    # Get and add AI analysis
    print("Requesting AI analysis...")
    ai_analysis = get_ai_analysis(img_path)
    if ai_analysis:
        print("Adding AI analysis to report...")
        pdf.add_ai_analysis(ai_analysis)
    else:
        print("No AI analysis available")
    
    pdf.output(output_path)

# The rest of  code remains the same until the generate_report function
def generate_report():
    print("Generating report...")

    try:
        data = load_data("sample_report_data_2.csv")
        print("Data loaded successfully")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    try:
        weekly_sales = analyze_data(data)
        print("Analysis completed")
    except Exception as e:
        print(f"Error analyzing data: {e}")
        return
    
    try:
        img_path = "sales_trend.png"
        generate_visualizations(weekly_sales, img_path)
        print("Visualizations generated")
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        return

    try:
        report_path = "weekly_sales_report.pdf"
        create_pdf_report(weekly_sales, img_path, report_path)
        print(f"Report saved to {report_path}")
    except Exception as e:
        print(f"Error creating PDF report: {e}")
        return

# Uncomment these lines to enable scheduling
# schedule.every().sunday.at("18:00").do(generate_report)
# while True:
#     schedule.run_pending()
#     time.sleep(1)

# Manual trigger for testing
if __name__ == "__main__":
    generate_report()