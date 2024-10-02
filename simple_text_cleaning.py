import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Improved function for text preprocessing and cleaning
def clean_text(text: str, remove_stopwords: bool = False, remove_numbers: bool = True) -> str:
    """
    Cleans and normalizes the input text by:
    1. Converting the text to lowercase (casefold for better internationalization).
    2. Removing URLs (both http/https and www patterns).
    3. Removing email addresses.
    4. Optionally removing numbers (positive or negative).
    5. Removing all non-alphanumeric characters except whitespace.
    6. Removing multiple spaces.
    7. Optionally removing stopwords.
    8. Removing leading and trailing whitespace.

    Args:
    text (str): The input text to be processed.
    remove_stopwords (bool): Whether to remove stopwords from the text.
    remove_numbers (bool): Whether to remove numbers from the text.

    Returns:
    str: The cleaned and normalized text.
    """
    # Step 1: Convert to lowercase (use casefold for better normalization in some languages)
    text = text.casefold()

    # Step 2: Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Step 3: Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # Step 4: Optionally remove numbers
    if remove_numbers:
        text = re.sub(r'[-+]?[0-9]+', '', text)

    # Step 5: Remove non-alphanumeric characters (punctuation etc.)
    text = re.sub(r'[^\w\s]', '', text)

    # Step 6: Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)

    # Step 7: Optionally remove stopwords
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(text)
        text = ' '.join([word for word in words if word not in stop_words])

    # Step 8: Remove leading/trailing whitespace
    text = text.strip()

    return text

# Example raw data utilizing all the processes
sample_text = "Send your response to john.doe123@example.com, check my website https://example.com. I earned $100 today!"

# Example usage with stopword removal
processed_text = clean_text(sample_text, remove_stopwords=True, remove_numbers=True)

print(f"raw data : {sample_text}")
print(f"processed data : {processed_text}")