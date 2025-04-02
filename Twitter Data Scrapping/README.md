# Twitter Data Scrapping

A practical repository exploring different techniques for extracting data from Twitter (X) using various methods including official Twitter API via Tweepy and alternative approaches like Nitter scraping.

## 🚀 Overview

This repository demonstrates multiple approaches to gather data from Twitter for analysis purposes. The examples showcase both authenticated API access through Tweepy and web scraping alternatives that don't require API access. The repository includes examples of tweet extraction, filtering, and simple sentiment analysis.

## 📋 Contents

- **Notebooks**:

  - `tweepy_testing.ipynb`: Implementation using the official Twitter API with Tweepy
  - `twitter_scraping.ipynb`: Alternative approach using Nitter for web scraping
- **Source Code**:

  - `Nitter Source Code.py`: Implementation of the Nitter scraper class
- **Data Samples**:

  - `filtered_tweets_tweepy.csv`: Example dataset of filtered tweets extracted with Tweepy

## 🛠️ Implementations

### Official Twitter API (Tweepy)

The `tweepy_testing.ipynb` notebook demonstrates how to use the official Twitter API through the Tweepy library:

- Authentication with Twitter API credentials
- Searching for tweets based on keywords
- Filtering tweets based on criteria
- Basic sentiment analysis using NLTK and TextBlob
- Data extraction and storage in CSV format

The notebook includes a practical case study analyzing tweets related to a news story about the Indonesian Minister of Science, Technology, and Higher Education.

### Alternative Approach (Nitter)

The `twitter_scraping.ipynb` notebook and `Nitter Source Code.py` demonstrate an alternative approach using Nitter:

- Web scraping tweets without requiring API credentials
- Extracting tweet content, user information, statistics, and media
- Working with multiple Nitter instances
- Error handling and retry mechanisms

This approach is useful when:

- You don't have Twitter API access
- You've reached API rate limits
- You need data that's not easily accessible through the official API

## 🔧 Setup

1. Clone this repository
2. Create a `.env` file with your Twitter API credentials (if using Tweepy):
   ```
   API_KEY="your_api_key"
   API_SECRET_KEY="your_api_secret_key"
   BEARER_TOKEN="your_bearer_token"
   ACCESS_TOKEN="your_access_token"
   ACCESS_TOKEN_SECRET="your_access_token_secret"
   ```
3. Install required dependencies:
   ```
   pip install tweepy ntscraper nltk textblob pandas wordcloud matplotlib pillow python-dotenv
   ```
4. Run the Jupyter notebooks

## 🧪 Scraping Methods

### Tweepy (Official API)

```python
# Authenticate with Twitter API
client = tweepy.Client(bearer_token=bearer_token)

# Search for recent tweets
query = "your search query -is:retweet"
tweets = client.search_recent_tweets(
    query=query,
    max_results=100,
    tweet_fields=['created_at', 'author_id', 'public_metrics']
)
```

### Nitter (Web Scraping)

```python
# Initialize the Nitter scraper
scraper = Nitter()

# Get tweets from a user
tweets = scraper.get_tweets("username", mode="user", number=10)

# Search for tweets containing a term
tweets = scraper.get_tweets("search term", mode="term", number=10)
```

## 📊 Data Processing Examples

The repository includes examples of data processing:

- Text cleaning and preprocessing
- Sentiment analysis using VADER and TextBlob
- Data filtering and transformation
- CSV export for further analysis

## ⚠️ Limitations and Considerations

- **API Restrictions**: Twitter's official API has rate limits and requires developer access
- **Terms of Service**: Always ensure your usage complies with Twitter's Terms of Service
- **Data Reliability**: Web scraping methods may break if Twitter/Nitter changes their structure
- **Ethical Use**: Be mindful of privacy concerns when collecting and analyzing social media data
- **Authentication**: Keep your API credentials secure and never commit them to public repositories

## 📝 License

This project is intended for educational and experimental purposes only.

## 📚 Resources

- [Twitter API Documentation](https://developer.twitter.com/en/docs)
- [Tweepy Documentation](https://docs.tweepy.org/)
- [NLTK Documentation](https://www.nltk.org/)
- [TextBlob Documentation](https://textblob.readthedocs.io/)
