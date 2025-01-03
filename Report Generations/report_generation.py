import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import schedule
import time

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
    plt.close()

# Step 4: Create PDF report
class PDFReport(FPDF):
    def header(self):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, "Weekly Sales Report", align="C", ln=True)

    def add_image(self, img_path):
        self.image(img_path, x=10, y=None, w=190)

def create_pdf_report(analysis_data, img_path, output_path):
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, "Sales Analysis:", ln=True)
    for index, row in analysis_data.iterrows():
        pdf.cell(0, 10, f"Week {row['Week']}: {row['Sales']} sales", ln=True)
    pdf.add_image(img_path)
    pdf.output(output_path)

# Step 5: Automation script
def generate_report():
    print("Generating report...")

    # Try loading data
    try:
        data = load_data("sample_report_data_2.csv")
        print(data.head())
    except Exception as e:
        print(f"Error loading data: {e}")
    
    # Perform analysis
    try:
        weekly_sales = analyze_data(data)
        print(weekly_sales.head())
    except Exception as e:
        print(f"Error analyzing data: {e}")
    
    # Generate visualizations
    try:
        img_path = "sales_trend.png"
        generate_visualizations(weekly_sales, img_path)
    except Exception as e:
        print(f"Error generating visualizations: {e}")

    # Create PDF report
    try:
        report_path = "weekly_sales_report.pdf"
        create_pdf_report(weekly_sales, img_path, report_path)
        print(f"Report saved to {report_path}")
    except Exception as e:
        print(f"Error creating PDF report: {e}")

# # Schedule the report generation
# schedule.every().sunday.at("18:00").do(generate_report)

# print("Scheduler running. Press Ctrl+C to exit.")
# while True:
#     schedule.run_pending()
#     time.sleep(1)

# Check if the report is generated correctly (manual trigger for testing)
generate_report()
