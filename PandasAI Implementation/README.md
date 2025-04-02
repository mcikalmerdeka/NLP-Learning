# PandasAI: Natural Language Database Assistant

An interactive solution that enables users to converse with relational databases through natural language, extracting valuable information and automatically generating comprehensive insights and reports from multiple tables without writing complex queries.

## 🚀 Features

- **Natural Language Queries**: Ask questions about your data in plain English
- **Automated Insights**: Generate visualizations and key metrics without coding
- **Interactive Data Exploration**: Explore complex datasets through conversation
- **Multiple Data Sources**: Connect to CSV files and potentially other database types
- **Web Platform Integration**: Push datasets to PandaBI for enhanced visualization
- **Report Generation**: Create comprehensive visual reports from query results

## 🛠️ Implementation Details

- Built with **PandasAI**, a Python library for natural language interactions with data
- Uses **Pandas** for efficient data manipulation and analysis
- Leverages **OpenAI** language models to interpret natural language queries
- Implements a clean workflow for data loading, transformation, and querying
- Features both local implementation and cloud-based platform options

## 📊 Use Cases

- **Data Analysis**: Quickly extract key metrics and patterns from datasets
- **Business Intelligence**: Generate reports without SQL expertise
- **Research**: Explore complex datasets with simple questions
- **Financial Analysis**: Analyze risk factors and identify patterns
- **Education**: Learn about data exploration without programming knowledge

## 📋 Requirements

- Python 3.8+
- PandasAI v3.0.0b5+
- Pandas
- Access to OpenAI API (via PandasAI API key)

## 🔧 Setup

1. Clone this repository
2. Create a virtual environment:
   ```
   python -m venv pandasai-venv
   source pandasai-venv/bin/activate  # On Windows: pandasai-venv\Scripts\activate
   ```
3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
4. Create a `.env` file with your API key:
   ```
   PANDASAI_API_KEY=your_pandasai_api_key
   ```

## 🚀 Usage

### Local Implementation

```python
import pandasai as pdai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load your dataset
df = pdai.read_csv("path/to/your/data.csv")

# Ask questions about your data
response = df.chat("What is the percentage of customers that defaulted?")
print(response)
```

### PandaBI Platform

```python
# Load your CSV file
file = pdai.read_csv("path/to/your/data.csv")

# Save your dataset configuration
df = pdai.create(
  path="your-namespace/your-dataset-name",
  df=file,
  description="Dataset description",
)

# Push your dataset to PandaBI platform
df.push()
```

Then access the platform at https://app.pandabi.ai to query your data with a visual interface.

## ⚠️ Limitations

The free tier of PandasAI has several limitations:

- Maximum file size of 10MB
- Limited to 100 queries per month
- No access to advanced features

For production use, consider the paid plans which offer:

- Larger file sizes (100MB-1GB)
- More monthly queries (1,000-10,000)
- Advanced visualization and collaboration features

## 📚 Resources

- [PandasAI Documentation](https://docs.getpanda.ai/v3/getting-started)
- [PandaBI Platform](https://app.pandabi.ai)
- [GitHub Repository](https://github.com/sinaptik-ai/pandas-ai)
