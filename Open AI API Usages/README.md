# OpenAI API Usages

An experimental repository exploring various applications of the OpenAI API for natural language processing, text generation, and visual data analysis tasks.

## 🚀 Overview

This repository contains Jupyter notebooks and datasets demonstrating practical applications of OpenAI's API. The examples cover a range of use cases from simple text generation to complex data analysis and visualization interpretation.

## 📋 Contents

- **Notebooks**:

  - `simple_chat_completions.ipynb`: Basic OpenAI API integration for text generation
  - `several_useful_applications.ipynb`: Multiple practical applications in a single utility class
  - `chart_insights.ipynb`: Visual data analysis with GPT-4 Vision capabilities
  - `openai_api_implementation.ipynb`: Comprehensive guide and tutorial for OpenAI API usage
- **Datasets**:

  - `Clicked Ads Dataset.csv`: Sample dataset for ad click analysis

## 🛠️ Implementations

### Simple Chat Completions

Demonstrates the fundamental usage of OpenAI's API for text generation with different parameter configurations:

- Basic API setup and authentication
- Parameter tuning (temperature, top_p, max_tokens)
- Response handling and parsing

Example application: Generating creative taglines for businesses.

### Several Useful Applications

A comprehensive class (`DataAnalysisGPT`) with multiple practical applications:

1. **SQL Query Generation**: Converting natural language to SQL queries

   ```python
   analyzer.generate_sql_query("Find the average sales by product category for the last quarter")
   ```
2. **Code Explanation**: Explaining complex data analysis code

   ```python
   analyzer.explain_code("your_python_code_here")
   ```
3. **Visualization Suggestions**: Recommending appropriate visualizations based on data

   ```python
   analyzer.suggest_visualizations(df_info)
   ```
4. **Data Cleaning Code**: Generating code to clean datasets with specific issues

   ```python
   analyzer.generate_data_cleaning_code(df_head, issues)
   ```
5. **Statistical Interpretation**: Explaining statistical results in plain language

   ```python
   analyzer.interpret_statistical_results(results)
   ```
6. **EDA Code Generation**: Creating exploratory data analysis code

   ```python
   analyzer.generate_eda_code(df_info)
   ```

### Chart Insights

Demonstrates OpenAI's Vision capabilities to analyze and interpret data visualizations:

- Processing Matplotlib/Seaborn visualizations
- Sending images to GPT-4 for analysis
- Getting detailed interpretations of patterns and trends
- Working with a real-world advertising dataset

### OpenAI API Implementation Guide

A comprehensive tutorial covering various aspects of the OpenAI API:

- Detailed explanations of API parameters (temperature, top_p, frequency_penalty, etc.)
- Multiple use case examples with explanations
- Best practices for prompt engineering
- Cost optimization strategies
- Error handling techniques

## 🔧 Setup

1. Clone this repository
2. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_KEY=your_openai_api_key
   ```
3. Install required dependencies:
   ```
   pip install openai pandas matplotlib seaborn python-dotenv
   ```
4. Run the Jupyter notebooks

## 🧩 Model Options

The repository uses the following OpenAI models:

- `gpt-3.5-turbo`: Fast and cost-effective for simpler tasks
- `gpt-4o-mini`: Balanced performance and cost
- `gpt-4o`: High performance vision-capable model
- `gpt-4o-2024-05-13`: Date-specific model version for consistent results

## 📊 Example Use Cases

- **Content Creation**: Generate marketing content, taglines, or creative text
- **Data Analysis Support**: Get help with data cleaning, visualization, and interpretation
- **Code Generation**: Create SQL queries or data cleaning code automatically
- **Visual Understanding**: Analyze charts and graphs for insights
- **Statistical Interpretation**: Explain complex statistical results in simple terms

## ⚠️ Considerations

- API usage incurs costs based on your OpenAI account settings
- Model capabilities and performance vary by model version
- For production use, implement proper error handling and rate limiting
- Vision capabilities are only available in certain models (gpt-4o and up)
- Consider token limits when designing prompts and handling responses

## 📝 License

This project is intended for educational and experimental purposes.

## 📚 Resources

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [OpenAI Python SDK](https://github.com/openai/openai-python)
- [OpenAI Cookbook](https://github.com/openai/openai-cookbook)
