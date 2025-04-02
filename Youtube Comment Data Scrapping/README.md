# YouTube Comment Data Scrapping

An experimental repository demonstrating techniques for extracting and analyzing comments from YouTube videos using the YouTube Data API.

## 🚀 Overview

This repository showcases how to scrape comments from YouTube videos using the YouTube Data API with Python. It includes methods for extracting comment data such as author names, timestamps, like counts, and comment text, enabling analysis of user engagement and sentiment.

## 📋 Contents

- **Notebooks**:
  - `YouTube_Comments_Advanced_Scrapping.ipynb`: Jupyter notebook containing the complete scraping process for YouTube comments, with examples of data extraction, pagination, and organization

## 🛠️ Implementation

The `YouTube_Comments_Advanced_Scrapping.ipynb` notebook demonstrates a complete YouTube comment scraping workflow:

1. **API Authentication**: Using Google API credentials to access the YouTube Data API
2. **Comment Retrieval**: Fetching comments for specified videos using their video IDs
3. **Pagination Handling**: Implementing pagination to retrieve more than the default limit of comments
4. **Data Organization**: Structuring the extracted comments into a pandas DataFrame
5. **Basic Analysis**: Sorting and filtering comments based on metrics like like count

The implementation extracts the following information for each comment:

- Author username
- Comment timestamp
- Like count
- Comment text
- Public visibility status
- Video ID (when scraping multiple videos)

## 🔧 Key Techniques Demonstrated

### API Authentication

```python
import googleapiclient.discovery
import pandas as pd

api_service_name = "youtube"
api_version = "v3"
DEVELOPER_KEY = "YOUR_API_KEY"

youtube = googleapiclient.discovery.build(
    api_service_name, api_version, developerKey=DEVELOPER_KEY)
```

### Comment Extraction

```python
request = youtube.commentThreads().list(
    part="snippet",
    videoId="VIDEO_ID",
    maxResults=100
)

comments = []
response = request.execute()

for item in response['items']:
    comment = item['snippet']['topLevelComment']['snippet']
    public = item['snippet']['isPublic']
    comments.append([
        comment['authorDisplayName'],
        comment['publishedAt'],
        comment['likeCount'],
        comment['textOriginal'],
        public
    ])
```

### Pagination Implementation

```python
while (1 == 1):
  try:
   nextPageToken = response['nextPageToken']
  except KeyError:
   break
  nextPageToken = response['nextPageToken']

  # Create a new request object with the next page token
  nextRequest = youtube.commentThreads().list(
      part="snippet", 
      videoId="VIDEO_ID", 
      maxResults=100, 
      pageToken=nextPageToken
  )

  # Execute the next request
  response = nextRequest.execute()

  # Process comments...
```

### Multiple Video Scraping

```python
def getcomments(video):
  request = youtube.commentThreads().list(
      part="snippet",
      videoId=video,
      maxResults=100
  )
  
  # Process comments and handle pagination
  # ...
  
  return pd.DataFrame(comments, columns=['author', 'updated_at', 'like_count', 'text', 'video_id', 'public'])

# Use the function to get comments from different videos
df1 = getcomments('VIDEO_ID_1')
df2 = getcomments('VIDEO_ID_2')
```

## 🧪 Use Cases

The techniques demonstrated in this repository can be applied to:

- Monitor audience sentiment and feedback
- Identify most-engaged users and top comments
- Gather market research from user responses
- Analyze content performance across different videos
- Track engagement patterns over time
- Create datasets for training sentiment analysis models

## ⚠️ Limitations and Considerations

- **API Quota**: YouTube Data API has daily quota limits that restrict the number of API calls
- **Comment Completeness**: As noted in the notebook, the API may not return all comments (only retrieved ~45-60% of total comments in testing)
- **Rate Limiting**: Excessive requests may trigger rate limiting or temporary blocks
- **API Key Security**: API keys should be kept secure and not exposed in public repositories
- **Terms of Service**: Usage must comply with YouTube's Terms of Service and API policies

## 🔧 Setup and Usage

1. Create a Google Cloud project and enable the YouTube Data API
2. Generate an API key in the Google Cloud Console
3. Clone this repository
4. Install required dependencies:
   ```
   pip install google-api-python-client pandas
   ```
5. Replace the placeholder API key with your actual key
6. Run the Jupyter notebook to see the scraping process in action

## 📚 Resources

- [YouTube Data API Documentation](https://developers.google.com/youtube/v3/docs)
- [Google API Python Client](https://github.com/googleapis/google-api-python-client)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [YouTube API Terms of Service](https://developers.google.com/youtube/terms/api-services-terms-of-service)