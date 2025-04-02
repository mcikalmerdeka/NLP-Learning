# AI-Enhanced Report Generation System

An experimental project that automates business report creation with AI-powered data analysis, combining traditional data visualization with advanced large language model insights for comprehensive business intelligence.

## 🚀 Features

- **Automated Report Generation**: Turn raw data into formatted PDF reports
- **Data Visualization**: Create insightful charts and graphs from sales data
- **AI-Powered Analysis**: Uses OpenAI GPT-4 to analyze trends and patterns
- **Scheduled Reports**: Configurable scheduling for automatic report creation
- **Image Analysis**: AI interprets visualizations and provides business insights
- **Detailed Formatting**: Professional PDF layout with tables and sections

## 🛠️ Implementation Details

- Built with **Python** for data processing and report generation
- Uses **Pandas** for data manipulation and analysis
- Implements **Matplotlib** and **Seaborn** for data visualization
- Leverages **OpenAI API** for intelligent data interpretation
- Uses **FPDF** for PDF document creation and formatting
- Features error handling for robust report generation

## 🧩 How It Works

1. **Data Loading and Processing**:

   - Reads CSV/Excel data files
   - Performs data cleaning and preprocessing
   - Aggregates sales data by week or other time periods
2. **Data Visualization**:

   - Creates bar and line charts of sales trends
   - Highlights peaks and important data points
   - Saves visualizations for inclusion in reports
3. **AI Analysis**:

   - Converts visualizations to base64 for API transmission
   - Sends images to OpenAI GPT-4 for interpretation
   - Receives detailed analysis of trends, patterns, and anomalies
4. **Report Generation**:

   - Creates structured PDF document
   - Includes data tables, visualizations, and AI analysis
   - Formats content with professional styling

## 📋 Requirements

- Python 3.8+
- OpenAI API key
- Required Python packages:
  - pandas
  - matplotlib
  - seaborn
  - fpdf
  - python-dotenv
  - openai
  - schedule

## 🔧 Setup

1. Clone this repository
2. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_KEY=your_openai_api_key
   ```
3. Install the required packages:
   ```
   pip install pandas matplotlib seaborn fpdf python-dotenv openai schedule
   ```
4. Prepare your data file (CSV or Excel) with sales information

## 🚀 Usage

### Basic Report Generation

```python
from report_generation import generate_report

# Generate a sales report
generate_report()
```

### AI-Enhanced Report Generation

```python
from report_generation_ai_analysis import generate_report

# Generate a sales report with AI analysis
generate_report()
```

### Scheduled Reports

```python
from report_generation_ai_analysis import generate_report
import schedule
import time

# Schedule weekly reports
schedule.every().sunday.at("18:00").do(generate_report)

# Keep the scheduler running
while True:
    schedule.run_pending()
    time.sleep(1)
```

## 📊 Included Files

- **report_generation.py**: Basic report generation with data visualization
- **report_generation_ai_analysis.py**: Enhanced version with AI-powered insights
- **sample_data_generation.ipynb**: Notebook for creating test datasets
- **sample_report_data_2.csv**: Large dataset (1 million rows) for testing
- **sample_report_data_3.xlsx**: Smaller Excel dataset with formatting
- **weekly_sales_report.pdf**: Example generated report
- **sales_trend.png**: Example visualization output
- **Explanation.txt**: Detailed implementation explanation

## ⚠️ Considerations

- Large datasets may require significant processing time
- OpenAI API calls incur costs based on usage
- The quality of AI analysis depends on visualization clarity
- Schedule module requires the script to be running continuously

## 🔄 Customization Options

The system can be customized in several ways:

1. **Data Sources**:

   ```python
   # Change data source
   data = load_data("your_data_file.csv")
   ```
2. **Visualization Style**:

   ```python
   # Modify visualization parameters
   plt.figure(figsize=(15, 8))  # Change size
   sns.barplot(data=data, x="Week", y="Sales", palette="viridis")  # Change colors
   ```
3. **AI Prompt Customization**:

   ```python
   # Modify AI analysis prompt
   response = client.chat.completions.create(
       model="gpt-4-turbo",
       messages=[
           {
               "role": "user",
               "content": [
                   {
                       "type": "text",
                       "text": "Your custom prompt here..."
                   },
                   {
                       "type": "image_url",
                       "image_url": {
                           "url": f"data:image/png;base64,{base64_image}"
                       }
                   }
               ]
           }
       ]
   )
   ```

## 📝 Future Enhancements

- Multiple visualization types (pie charts, heat maps)
- Email integration for automatic report distribution
- Interactive dashboard version with real-time updates
- Multi-dataset correlation analysis
- Natural language query support for custom reports
