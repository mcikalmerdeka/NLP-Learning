# Wikipedia Data Scrapping

An experimental repository demonstrating techniques for extracting and processing data from Wikipedia pages using web scraping methods.

## 🚀 Overview

This repository showcases how to extract structured data from Wikipedia tables using the BeautifulSoup library in Python. The example focuses on scraping a list of largest companies in the United States by revenue, parsing the HTML structure, and transforming the data into a clean, usable format for analysis.

## 📋 Contents

- **Notebooks**:

  - `wikipedia_scrapping.ipynb`: Jupyter notebook containing the complete scraping process, from HTML extraction to data cleaning and CSV export
- **Data**:

  - `Extracted Wikipedia Data.csv`: The cleaned dataset extracted from Wikipedia, containing information about the top 100 US companies by revenue

## 🛠️ Implementation

The `wikipedia_scrapping.ipynb` notebook demonstrates the complete web scraping workflow:

1. **HTML Retrieval**: Using the `requests` library to fetch the Wikipedia page content
2. **HTML Parsing**: Employing `BeautifulSoup` to parse the HTML and locate the target table
3. **Data Extraction**: Navigating the HTML structure to extract table headers and row data
4. **Data Cleaning**: Processing the raw text to remove unwanted characters and formatting
5. **Data Transformation**: Converting the extracted data into a structured pandas DataFrame
6. **Data Export**: Saving the cleaned data to a CSV file for further analysis

The example extracts the following information for each company:

- Rank
- Company name
- Industry
- Revenue (in USD millions)
- Revenue growth
- Number of employees
- Headquarters location

## 🔧 Key Techniques Demonstrated

### HTML Parsing and Element Selection

```python
# Parsing the HTML content
soup = BeautifulSoup(page.text, "html")

# Finding the main table with data
table = soup.find_all("table", {"class": "wikitable sortable"})[0]

# Extracting column names
column_names = table.find_all("th")
column_data = table.find_all("tr")[1:]
```

### Text Extraction and Cleaning

```python
# Extracting and cleaning text data from HTML elements
for row in column_data:
    row_data = row.find_all("td")
    individual_row_data = [data.text.strip() for data in row_data]
    # Process or store the extracted data
```

### Data Transformation

```python
# Creating a structured DataFrame from extracted data
df = pd.DataFrame(columns=headers)
for row in column_data[1:]:
    row_data = row.find_all("td")
    individual_row_data = [data.text.strip() for data in row_data]
    length = len(df)
    df.loc[length] = individual_row_data
```

### Location Parsing

```python
# Extracting city and state from headquarters information
df[['City', 'State']] = df['Headquarters'].str.extract(r'([\w\s]+),\s*([\w\s\.]+)')
```

## 🧪 Use Cases

The techniques demonstrated in this repository can be applied to:

- Extract data from any Wikipedia table
- Create datasets from publicly available information
- Monitor and track changes in company rankings and metrics
- Perform market analysis based on revenue, growth, and employee data
- Compare industries based on various financial metrics

## ⚠️ Limitations and Considerations

- **Web Page Structure**: Scraping depends on the page's HTML structure, which may change over time
- **Rate Limiting**: Frequent scraping requests may be blocked by Wikipedia
- **Data Accuracy**: Information on Wikipedia may not always be up-to-date or accurate
- **Ethical Use**: Respect Wikipedia's terms of service and use the data responsibly
- **Alternative APIs**: Consider using Wikipedia's official API for more reliable data access

## 🔧 Setup and Usage

1. Clone this repository
2. Install required dependencies:
   ```
   pip install beautifulsoup4 pandas requests
   ```
3. Run the Jupyter notebook to see the scraping process in action
4. Modify the code to target different Wikipedia pages of interest

## 📚 Resources

- [BeautifulSoup Documentation](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Wikipedia&#39;s Robot Policy](https://en.wikipedia.org/wiki/Wikipedia:Bots)
- [HTTP Requests in Python](https://docs.python-requests.org/en/latest/)
