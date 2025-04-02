# Anthropic API Usages

An experimental repository exploring various applications of the Anthropic Claude API for natural language processing and data analysis tasks.

## 🚀 Overview

This repository contains Jupyter notebooks and datasets demonstrating practical applications of Anthropic's Claude API. The examples cover a range of use cases from simple text generation to complex data analysis and visualization interpretation.

## 📋 Contents

- **Notebooks**:
  - `simple_chat_completions.ipynb`: Basic Claude API integration for text generation
  - `several_useful_applications.ipynb`: Multiple practical applications in a single utility class
  - `chart_insights.ipynb`: Visual data analysis with Claude's vision capabilities

- **Datasets**:
  - `Clicked Ads Dataset.csv`: Sample dataset for ad click analysis

## 🛠️ Implementations

### Simple Chat Completions

Demonstrates the fundamental usage of Claude's API for text generation with different parameter configurations:

- Basic API setup and authentication
- Parameter tuning (temperature, top_p, top_k)
- System prompt customization
- Response handling

Example application: Generating creative taglines for businesses.

### Several Useful Applications

A comprehensive class (`ClaudeAnalysis`) with multiple practical applications:

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

7. **Visualization Analysis**: Analyzing data visualizations through image recognition
   ```python
   analyzer.analyze_visualization(plt_figure)
   ```

### Chart Insights

Demonstrates Claude's vision capabilities to analyze and interpret data visualizations:

- Processing Matplotlib/Seaborn visualizations
- Sending images to Claude for analysis
- Getting detailed interpretations of patterns and trends
- Working with a real-world advertising dataset

## 🔧 Setup

1. Clone this repository
2. Create a `.env` file with your Anthropic API key:
   ```
   ANTHROPIC_KEY=your_anthropic_api_key
   ```
3. Install required dependencies:
   ```
   pip install anthropic pandas matplotlib seaborn python-dotenv
   ```
4. Run the Jupyter notebooks

## 🧩 Model Names

The repository uses the following Claude models:

- `claude-3-haiku-20240307` (fastest)
- `claude-3-sonnet-20240229` (balanced)
- `claude-3-opus-20240229` (most capable)

## 📊 Example Use Cases

- **Content Creation**: Generate marketing content, taglines, or creative text
- **Data Analysis Support**: Get help with data cleaning, visualization, and interpretation
- **Code Generation**: Create SQL queries or data cleaning code automatically
- **Visual Understanding**: Analyze charts and graphs for insights
- **Statistical Interpretation**: Explain complex statistical results in simple terms

## ⚠️ Considerations

- The examples use specific Claude model versions that may change over time
- API usage incurs costs based on your Anthropic account settings
- For production use, implement proper error handling and rate limiting
- Vision capabilities are only available in certain Claude models

## 📝 License

This project is intended for educational and experimental purposes.

## 📚 Resources

- [Anthropic Claude API Documentation](https://docs.anthropic.com/claude/reference/getting-started-with-the-api)
- [Claude API Python SDK](https://github.com/anthropics/anthropic-sdk-python) 