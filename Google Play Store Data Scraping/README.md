# Google Play Store Data Scraping

An experimental repository demonstrating techniques for extracting and analyzing app data and reviews from the Google Play Store.

## 🚀 Overview

This repository showcases methods for scraping data from the Google Play Store using Python. It focuses on extracting app reviews to analyze user sentiment, identify common feature requests, and understand user experiences with mobile applications.

## 📋 Contents

- **Notebooks**:
  - `playstore_data_scraping.ipynb`: Jupyter notebook containing the complete implementation for scraping Google Play Store app reviews using the google-play-scraper library

## 🛠️ Implementation

The implementation demonstrates a workflow for scraping and analyzing Google Play Store app reviews:

1. **Setup and Installation**: Setting up the required libraries (`google-play-scraper`)
2. **Data Extraction**: Fetching app reviews using the app ID
3. **Data Organization**: Converting extracted JSON data into a pandas DataFrame
4. **Analysis Preparation**: Initial exploration of the data structure

The code extracts the following information for each review:

- Review ID
- Username
- User profile image URL
- Review content/text
- Rating score (1-5)
- Thumbs up count
- App version when review was created
- Review timestamp
- Developer reply content (if any)
- Developer reply timestamp (if any)
- App version

## 🔧 Key Techniques Demonstrated

### Library Installation

```python
# Install the required library
pip install google-play-scraper
```

### Required Imports

```python
from google_play_scraper import app, Sort, reviews
import numpy as np
import pandas as pd
```

### Reviews Extraction

```python
# Extract reviews with pagination
result, continuation_token = reviews(
    'com.tomoro.indonesia.android',  # App ID from Play Store URL
    lang='id',                        # Language of the reviews
    country='id',                     # Country for the reviews
    sort=Sort.NEWEST,                 # Sort order
    count=2000,                       # Number of reviews to extract
    filter_score_with=None            # Option to filter by rating
)
```

### Converting to DataFrame

```python
# Create DataFrame from the extracted reviews
df = pd.DataFrame(result)

# Check the data structure
df.head()
```

## 📊 Findings and Limitations

- The extraction method has a cap of approximately 654 reviews even when requesting a higher number (2000+)
- The extracted data is well-structured in JSON format, making it easy to convert to a pandas DataFrame
- The API provides comprehensive metadata for each review including timestamps, app versions, and developer replies
- The technique is effective for obtaining the most recent reviews for analysis but may not capture the complete review history

## 🧪 Use Cases

The techniques demonstrated in this repository can be applied to:

- Monitor user sentiment and satisfaction over time
- Identify common issues and bugs reported by users
- Track the impact of app updates on user reviews
- Create datasets for training natural language processing models
- Compare user experiences across different apps in the same category
- Inform product development and feature prioritization

## ⚠️ Limitations and Considerations

- **API Limits**: The google-play-scraper library has limitations on the number of reviews that can be extracted
- **Legal Considerations**: Always review Google Play Store's Terms of Service before scraping
- **Data Comprehensiveness**: The method doesn't guarantee extraction of all historical reviews
- **Rate Limiting**: Excessive requests may lead to temporary blocking
- **Data Freshness**: Reviews are constantly being added, so results represent a snapshot in time

## 🔧 Setup and Usage

1. Install Python 3.x
2. Clone this repository
3. Install required dependencies:
   ```
   pip install google-play-scraper pandas numpy
   ```
4. Open and run the Jupyter Notebook

## 📚 Resources

- [google-play-scraper Documentation](https://github.com/JoMingyu/google-play-scraper)
- [pandas Documentation](https://pandas.pydata.org/docs/)
- [Google Play Store Developer Policies](https://play.google.com/about/developer-content-policy/)