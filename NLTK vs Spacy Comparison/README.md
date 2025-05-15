# NLTK vs spaCy: Comprehensive Code Comparison

This guide offers side-by-side code comparisons between NLTK and spaCy for a wide range of NLP tasks, from basic to advanced.

## Introduction and Quick Summary

**🧠 General Overview**

Both NLTK (Natural Language Toolkit) and spaCy are popular Python libraries for Natural Language Processing (NLP), but they are designed with different philosophies and strengths. Here's a breakdown of their capabilities and a comparison for common NLP tasks:

| Feature            | **NLTK**                                    | **spaCy**                                                       |
| ------------------ | ------------------------------------------- | --------------------------------------------------------------- |
| Philosophy         | Education, research, prototyping            | Production-ready, industrial-strength                           |
| Language support   | Many languages (but variable depth)         | Focused, high-quality support (mainly English and a few others) |
| Speed              | Slower                                      | Much faster                                                     |
| Ease of use        | Requires combining many components manually | More streamlined pipeline                                       |
| Pre-trained models | Few, mostly basic                           | Comes with powerful pre-trained pipelines                       |
| Tokenizer          | Rule-based                                  | Statistical + rule-based (more robust)                          |

**🧪 Comparison by NLP Tasks**

| NLP Task                           | **NLTK**                                             | **spaCy**                                                            |
| ---------------------------------- | ---------------------------------------------------- | -------------------------------------------------------------------- |
| **Tokenization**                   | `word_tokenize`, `sent_tokenize`; basic rule-based   | `nlp(text)`, efficient and handles complex cases better              |
| **Part-of-Speech (POS) Tagging**   | Uses Penn Treebank tagger (`pos_tag`)                | `token.pos_` with more accurate and faster POS tagging               |
| **Named Entity Recognition (NER)** | Available, but requires training or external models  | Built-in with pre-trained models (`ent.label_`, `ent.text`)          |
| **Dependency Parsing**             | Limited; no native dependency parsing                | Built-in, efficient, production-ready                                |
| **Lemmatization**                  | WordNet-based lemmatizer (rule + dictionary)         | Built-in lemmatizer (context-aware via models)                       |
| **Stemming**                       | Available (`Porter`, `Lancaster`, `Snowball`)        | Not built-in; prefers lemmatization                                  |
| **Text Classification**            | Custom implementation or training needed             | Optional pipeline component (custom or trained)                      |
| **Vector Representation**          | Not built-in (use `gensim`, etc.)                    | Word vectors via models (`token.vector`) if model supports it        |
| **Stop Words**                     | Predefined stopword list                             | Predefined, can be customized                                        |
| **Language Models**                | Not included directly; integrates poorly with others | Comes with trained statistical models (e.g., `en_core_web_sm`)       |
| **Integration with ML tools**      | Weak integration with scikit-learn, TensorFlow       | Stronger for production NLP, but less customizable for deep ML tasks |
| **Visualization**                  | None built-in                                        | Has `displacy` for dependency and entity visualization               |

**⚖️ When to Use What**
Use NLTK if:

- You're learning or teaching NLP concepts.

- You need fine control over preprocessing steps.

- You're doing linguistics research or working with less common languages.

Use spaCy if:

- You need a fast and production-ready NLP pipeline.

- You're building an NLP app or API.

- You want better support for dependency parsing, NER, and vector representations.


## Table of Contents

1. [Setup and Installation](#setup-and-installation)
2. [Basic NLP Tasks](#basic-nlp-tasks)
   - [Tokenization](#1-tokenization)
   - [Part-of-Speech Tagging](#2-part-of-speech-tagging)
   - [Lemmatization](#3-lemmatization)
   - [Stemming](#4-stemming)
   - [Stop Words Removal](#5-stop-words-removal)
3. [Intermediate NLP Tasks](#intermediate-nlp-tasks)
   - [Named Entity Recognition](#6-named-entity-recognition)
   - [Dependency Parsing](#7-dependency-parsing)
   - [Sentence Segmentation](#8-sentence-segmentation)
   - [Text Preprocessing Pipeline](#9-text-preprocessing-pipeline)
   - [N-grams Generation](#10-n-grams-generation)
4. [Advanced NLP Tasks](#advanced-nlp-tasks)
   - [Word Embeddings](#11-word-embeddings)
   - [Text Classification](#12-text-classification)
   - [Sentiment Analysis](#13-sentiment-analysis)
   - [Topic Modeling](#14-topic-modeling)
   - [Text Summarization](#15-text-summarization)
   - [Language Detection](#16-language-detection)
   - [Chunking and Shallow Parsing](#17-chunking-and-shallow-parsing)
   - [Custom NLP Pipeline Components](#18-custom-nlp-pipeline-components)
5. [Practical Use Cases](#practical-use-cases)
   - [Text Similarity Comparison](#19-text-similarity-comparison)
   - [Keyword Extraction](#20-keyword-extraction)
   - [Question Answering](#21-question-answering)
6. [Performance Benchmarks](#performance-benchmarks)
7. [When to Use Which Library](#when-to-use-which-library)

## Setup and Installation

Let's start with how to set up each library:

# Core NLP Libraries

uv add nltk spacy

# Install scikit-learn for machine learning components

uv add scikit-learn

# Additional helpful libraries

uv add gensim textblob pandas numpy

# For visualization (optional, but recommended)

uv add matplotlib seaborn

# Download spaCy language models

python -m spacy download en_core_web_sm   # Small model (faster)
python -m spacy download en_core_web_md   # Medium model (includes word vectors)

# Download NLTK data

python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('wordnet'); nltk.download('stopwords'); nltk.download('maxent_ne_chunker'); nltk.download('words'); nltk.download('vader_lexicon')"

```python
# NLTK Installation and Setup
# --------------------------
# Install: pip install nltk
import nltk

# Download necessary data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# spaCy Installation and Setup
# ---------------------------
# Install: pip install spacy
# Download model: python -m spacy download en_core_web_sm
import spacy

# Load model
nlp = spacy.load('en_core_web_sm')
```

## Basic NLP Tasks

### 1. Tokenization

```python
def tokenize_comparison(text):
    """Compare tokenization between NLTK and spaCy"""
  
    # Sample text
    print(f"Original text: {text}")
  
    # NLTK tokenization
    from nltk.tokenize import word_tokenize
    nltk_tokens = word_tokenize(text)
    print(f"\nNLTK tokens: {nltk_tokens}")
    print(f"NLTK token count: {len(nltk_tokens)}")
  
    # spaCy tokenization
    import spacy
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    spacy_tokens = [token.text for token in doc]
    print(f"\nspaCy tokens: {spacy_tokens}")
    print(f"spaCy token count: {len(spacy_tokens)}")
  
    # Compare differences
    nltk_set = set(nltk_tokens)
    spacy_set = set(spacy_tokens)
    print(f"\nTokens in NLTK but not in spaCy: {nltk_set - spacy_set}")
    print(f"Tokens in spaCy but not in NLTK: {spacy_set - nltk_set}")

# Example usage
text = "Mr. Smith paid $30.00 for the U.S.A. trip-package. It's worth it!"
tokenize_comparison(text)
```

### 2. Part-of-Speech Tagging

```python
def pos_tagging_comparison(text):
    """Compare part-of-speech tagging between NLTK and spaCy"""
  
    print(f"Original text: {text}")
  
    # NLTK POS Tagging
    from nltk.tokenize import word_tokenize
    from nltk import pos_tag
  
    nltk_tokens = word_tokenize(text)
    nltk_pos_tags = pos_tag(nltk_tokens)
  
    print("\nNLTK POS Tags:")
    for token, tag in nltk_pos_tags:
        print(f"{token}: {tag}")
  
    # spaCy POS Tagging
    import spacy
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
  
    print("\nspaCy POS Tags:")
    for token in doc:
        # Print both the simple POS and the detailed tag
        print(f"{token.text}: {token.pos_} (fine-grained: {token.tag_})")
  
    # Note: NLTK uses Penn Treebank tagset, while spaCy uses Universal Dependencies

# Example usage
text = "The quick brown fox jumps over the lazy dog."
pos_tagging_comparison(text)
```

### 3. Lemmatization

```python
def lemmatization_comparison(text):
    """Compare lemmatization between NLTK and spaCy"""
  
    print(f"Original text: {text}")
  
    # NLTK Lemmatization
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    from nltk import pos_tag
  
    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()
  
    # NLTK requires POS information for better lemmatization
    # We need to convert Penn Treebank tags to WordNet tags
    def get_wordnet_pos(treebank_tag):
        if treebank_tag.startswith('J'):
            return 'a'  # adjective
        elif treebank_tag.startswith('V'):
            return 'v'  # verb
        elif treebank_tag.startswith('N'):
            return 'n'  # noun
        elif treebank_tag.startswith('R'):
            return 'r'  # adverb
        else:
            return 'n'  # default to noun
  
    # Tokenize and get POS tags
    nltk_tokens = word_tokenize(text)
    nltk_pos = pos_tag(nltk_tokens)
  
    # Lemmatize with POS tags
    nltk_lemmas = [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) 
                   for word, pos in nltk_pos]
  
    # Simple lemmatization (without POS, defaults to nouns)
    nltk_simple_lemmas = [lemmatizer.lemmatize(word) for word in nltk_tokens]
  
    print("\nNLTK lemmas (with POS):")
    for original, lemma in zip(nltk_tokens, nltk_lemmas):
        print(f"{original} -> {lemma}")
  
    print("\nNLTK simple lemmas (without POS):")
    for original, lemma in zip(nltk_tokens, nltk_simple_lemmas):
        print(f"{original} -> {lemma}")
  
    # spaCy Lemmatization
    import spacy
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
  
    print("\nspaCy lemmas:")
    for token in doc:
        print(f"{token.text} -> {token.lemma_}")

# Example usage
text = "The cats are running and jumping over many boxes."
lemmatization_comparison(text)
```

### 4. Stemming

```python
def stemming_comparison(text):
    """Compare stemming in NLTK (spaCy doesn't have built-in stemming)"""
  
    print(f"Original text: {text}")
  
    # NLTK Stemming
    from nltk.tokenize import word_tokenize
    from nltk.stem.porter import PorterStemmer
    from nltk.stem.lancaster import LancasterStemmer
    from nltk.stem.snowball import SnowballStemmer
  
    # Initialize stemmers
    porter = PorterStemmer()
    lancaster = LancasterStemmer()
    snowball = SnowballStemmer('english')
  
    # Tokenize
    tokens = word_tokenize(text)
  
    # Apply stemmers
    porter_stems = [porter.stem(word) for word in tokens]
    lancaster_stems = [lancaster.stem(word) for word in tokens]
    snowball_stems = [snowball.stem(word) for word in tokens]
  
    print("\nNLTK Porter stemmer:")
    for original, stem in zip(tokens, porter_stems):
        print(f"{original} -> {stem}")
  
    print("\nNLTK Lancaster stemmer:")
    for original, stem in zip(tokens, lancaster_stems):
        print(f"{original} -> {stem}")
  
    print("\nNLTK Snowball stemmer:")
    for original, stem in zip(tokens, snowball_stems):
        print(f"{original} -> {stem}")
  
    # spaCy doesn't have built-in stemming
    print("\nspaCy doesn't have built-in stemming functionality.")
    print("Instead, spaCy focuses on lemmatization, which is generally more accurate.")
    print("You can use NLTK's stemmers with spaCy tokens if needed.")

# Example usage
text = "The runners running quickly jumped over hurdles and boxes."
stemming_comparison(text)
```

### 5. Stop Words Removal

```python
def stopwords_comparison(text):
    """Compare stop words removal between NLTK and spaCy"""
  
    print(f"Original text: {text}")
  
    # NLTK Stop Words
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
  
    nltk_stopwords = set(stopwords.words('english'))
    nltk_tokens = word_tokenize(text.lower())
    nltk_filtered = [word for word in nltk_tokens if word not in nltk_stopwords]
  
    print(f"\nNLTK stop words count: {len(nltk_stopwords)}")
    print(f"NLTK filtered text: {' '.join(nltk_filtered)}")
  
    # spaCy Stop Words
    import spacy
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text.lower())
  
    spacy_filtered = [token.text for token in doc if not token.is_stop]
  
    print(f"\nspaCy stop words count: {len(nlp.Defaults.stop_words)}")
    print(f"spaCy filtered text: {' '.join(spacy_filtered)}")
  
    # Compare stop word lists
    spacy_stopwords = nlp.Defaults.stop_words
    print("\nSome stop words in NLTK but not in spaCy:")
    nltk_only = list(nltk_stopwords - spacy_stopwords)
    print(nltk_only[:10] if len(nltk_only) > 10 else nltk_only)
  
    print("\nSome stop words in spaCy but not in NLTK:")
    spacy_only = list(spacy_stopwords - nltk_stopwords)
    print(spacy_only[:10] if len(spacy_only) > 10 else spacy_only)

# Example usage
text = "This is a simple example of how to remove stop words from a sentence."
stopwords_comparison(text)
```

## Intermediate NLP Tasks

### 6. Named Entity Recognition

```python
def ner_comparison(text):
    """Compare Named Entity Recognition between NLTK and spaCy"""
  
    print(f"Original text: {text}")
  
    # NLTK Named Entity Recognition
    import nltk
    from nltk.tokenize import word_tokenize
  
    # Tokenize and POS tag
    tokens = word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
  
    # Apply NER
    nltk_entities = nltk.chunk.ne_chunk(pos_tags)
  
    print("\nNLTK Named Entities:")
    for subtree in nltk_entities:
        if isinstance(subtree, nltk.Tree):
            entity_type = subtree.label()
            entity_text = " ".join([word for word, tag in subtree.leaves()])
            print(f"{entity_text}: {entity_type}")
  
    # spaCy Named Entity Recognition
    import spacy
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
  
    print("\nspaCy Named Entities:")
    for ent in doc.ents:
        print(f"{ent.text}: {ent.label_} ({spacy.explain(ent.label_)})")
  
    # Visual display with spaCy
    print("\nspaCy offers visual display of NER with displacy:")
    print("from spacy import displacy\ndisplacy.render(doc, style='ent')")

# Example usage
text = "Apple Inc. was founded by Steve Jobs in California. The company is worth $2 trillion as of 2023."
ner_comparison(text)
```

### 7. Dependency Parsing

```python
def dependency_parsing_comparison(text):
    """Compare dependency parsing between NLTK and spaCy"""
  
    print(f"Original text: {text}")
  
    # NLTK Dependency Parsing
    print("\nNLTK doesn't have built-in dependency parsing.")
    print("You would typically use a separate library like 'nltk.parse.corenlp' which requires")
    print("setting up Stanford CoreNLP server separately.")
  
    # spaCy Dependency Parsing
    import spacy
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
  
    print("\nspaCy Dependency Parse:")
    for token in doc:
        print(f"{token.text} --{token.dep_}--> {token.head.text}")
  
    # Visual display with spaCy
    print("\nspaCy offers visual display of dependencies with displacy:")
    print("from spacy import displacy\ndisplacy.render(doc, style='dep')")

# Example usage
text = "The quick brown fox jumps over the lazy dog."
dependency_parsing_comparison(text)
```

### 8. Sentence Segmentation

```python
def sentence_segmentation_comparison(text):
    """Compare sentence segmentation between NLTK and spaCy"""
  
    print(f"Original text: {text}")
  
    # NLTK Sentence Tokenization
    from nltk.tokenize import sent_tokenize
  
    nltk_sentences = sent_tokenize(text)
  
    print("\nNLTK sentence segmentation:")
    for i, sent in enumerate(nltk_sentences, 1):
        print(f"Sentence {i}: {sent}")
  
    # spaCy Sentence Segmentation
    import spacy
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
  
    spacy_sentences = list(doc.sents)
  
    print("\nspaCy sentence segmentation:")
    for i, sent in enumerate(spacy_sentences, 1):
        print(f"Sentence {i}: {sent}")
  
    # Compare
    print(f"\nNLTK found {len(nltk_sentences)} sentences.")
    print(f"spaCy found {len(spacy_sentences)} sentences.")

# Example usage
text = "Mr. Smith went to Washington, D.C. He had a meeting at the White House. Later, he visited the Smithsonian Museum."
sentence_segmentation_comparison(text)
```

### 9. Text Preprocessing Pipeline

```python
def preprocessing_pipeline_comparison(text):
    """Compare full preprocessing pipelines in NLTK and spaCy"""
  
    print(f"Original text: {text}")
  
    # NLTK Preprocessing Pipeline
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    import string
  
    def nltk_preprocess(text):
        # Convert to lowercase
        text = text.lower()
      
        # Tokenize
        tokens = word_tokenize(text)
      
        # Remove punctuation
        tokens = [token for token in tokens if token not in string.punctuation]
      
        # Remove stop words
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
      
        # Lemmatize
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
      
        return tokens
  
    nltk_processed = nltk_preprocess(text)
    print("\nNLTK preprocessing result:")
    print(nltk_processed)
  
    # spaCy Preprocessing Pipeline
    import spacy
  
    def spacy_preprocess(text):
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(text.lower())
      
        # Filter tokens: no punctuation, no stop words
        tokens = [token.lemma_ for token in doc 
                  if not token.is_punct and not token.is_stop]
      
        return tokens
  
    spacy_processed = spacy_preprocess(text)
    print("\nspaCy preprocessing result:")
    print(spacy_processed)
  
    # Compare results
    nltk_set = set(nltk_processed)
    spacy_set = set(spacy_processed)
  
    print("\nTokens in NLTK result but not in spaCy:")
    print(nltk_set - spacy_set)
  
    print("\nTokens in spaCy result but not in NLTK:")
    print(spacy_set - nltk_set)

# Example usage
text = "The cats are running quickly! They're chasing mice in the garden."
preprocessing_pipeline_comparison(text)
```

### 10. N-grams Generation

```python
def ngram_comparison(text, n=2):
    """Compare n-gram generation between NLTK and spaCy"""
  
    print(f"Original text: {text}")
    print(f"Generating {n}-grams")
  
    # NLTK N-grams
    from nltk.tokenize import word_tokenize
    from nltk.util import ngrams
  
    nltk_tokens = word_tokenize(text)
    nltk_ngrams = list(ngrams(nltk_tokens, n))
  
    print("\nNLTK n-grams:")
    for gram in nltk_ngrams:
        print(f"{'_'.join(gram)}")
  
    # spaCy doesn't have built-in n-grams
    print("\nspaCy doesn't have a built-in n-gram function.")
    print("However, you can create n-grams from spaCy tokens:")
  
    import spacy
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
  
    # Custom function for n-grams with spaCy
    def spacy_ngrams(doc, n):
        return [doc[i:i+n] for i in range(len(doc)-n+1)]
  
    spacy_ngrams_list = spacy_ngrams(doc, n)
  
    print("\nspaCy custom n-grams:")
    for gram in spacy_ngrams_list:
        print(f"{'_'.join([token.text for token in gram])}")

# Example usage
text = "Natural language processing is fascinating and useful."
ngram_comparison(text, 2)  # Bigrams
```

## Advanced NLP Tasks

### 11. Word Embeddings

```python
def word_embeddings_comparison(words):
    """Compare word embeddings between NLTK/Gensim and spaCy"""
  
    print(f"Getting embeddings for words: {words}")
  
    # NLTK doesn't have built-in word embeddings
    # Typically, you'd use Gensim with NLTK
    print("\nNLTK doesn't have built-in word embeddings.")
    print("Usually used with Gensim:")
  
    try:
        import gensim.downloader as api
        # Download a small model for demonstration
        model = api.load('glove-wiki-gigaword-50')
      
        print("\nGensim (commonly used with NLTK) word embeddings:")
        for word in words:
            if word in model:
                # Just show first 5 dimensions of the embedding vector
                print(f"{word}: {model[word][:5]}... (vector length: {len(model[word])})")
            else:
                print(f"{word}: Not found in vocabulary")
      
        # Similarity demo with Gensim
        if 'cat' in model and 'dog' in model:
            similarity = model.similarity('cat', 'dog')
            print(f"\nGensim similarity between 'cat' and 'dog': {similarity:.4f}")
    except:
        print("Could not load Gensim model. Install with: pip install gensim")
  
    # spaCy Word Embeddings
    import spacy
  
    try:
        # Need a model with word vectors
        nlp = spacy.load('en_core_web_md')  # Medium model with word vectors
      
        print("\nspaCy word embeddings:")
        for word in words:
            doc = nlp(word)
            # Just show first 5 dimensions
            print(f"{word}: {doc.vector[:5]}... (vector length: {len(doc.vector)})")
      
        # Similarity demo with spaCy
        doc1 = nlp("cat")
        doc2 = nlp("dog")
        similarity = doc1.similarity(doc2)
        print(f"\nspaCy similarity between 'cat' and 'dog': {similarity:.4f}")
    except:
        print("Could not load spaCy model with vectors.")
        print("Install with: python -m spacy download en_core_web_md")

# Example usage
word_list = ["computer", "keyboard", "algorithm", "python", "nonsenseword123"]
word_embeddings_comparison(word_list)
```

### 12. Text Classification

```python
def text_classification_comparison():
    """Compare text classification approaches between NLTK and spaCy"""
  
    # Sample data
    texts = ["I love this product, it's amazing!", 
             "This is terrible, don't buy it.",
             "Somewhat disappointed with the quality.",
             "Absolutely fantastic service!"]
    labels = ["positive", "negative", "negative", "positive"]
  
    print("Text Classification Example with NLTK and spaCy")
    print(f"Sample texts: {texts}")
    print(f"Sample labels: {labels}")
  
    # NLTK Text Classification
    print("\n--- NLTK Text Classification ---")
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
  
    # Define preprocessing function for NLTK
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
  
    def nltk_preprocessor(text):
        tokens = word_tokenize(text.lower())
        tokens = [lemmatizer.lemmatize(token) for token in tokens 
                  if token.isalpha() and token not in stop_words]
        return ' '.join(tokens)
  
    # Create preprocessing and classification pipeline
    nltk_pipeline = Pipeline([
        ('vectorizer', CountVectorizer(preprocessor=nltk_preprocessor)),
        ('classifier', MultinomialNB())
    ])
  
    # Train the classifier
    nltk_pipeline.fit(texts, labels)
  
    # Test on new data
    test_texts = ["I really enjoyed using this product",
                  "This was a waste of money"]
  
    nltk_predictions = nltk_pipeline.predict(test_texts)
  
    print("NLTK Classification results:")
    for text, prediction in zip(test_texts, nltk_predictions):
        print(f"Text: '{text}' -> Predicted: {prediction}")
  
    # spaCy Text Classification
    print("\n--- spaCy Text Classification ---")
    import spacy
    from spacy.training import Example
    from spacy.util import minibatch, compounding
    import random
  
    print("Note: spaCy requires more setup for text classification.")
    print("Below is a pseudocode/simplified version of the process:")
  
    """
    # Initialize spaCy
    nlp = spacy.load('en_core_web_sm')
  
    # Add text classifier to pipeline
    textcat = nlp.add_pipe('textcat')
    textcat.add_label("positive")
    textcat.add_label("negative")
  
    # Prepare training data
    train_data = []
    for text, label in zip(texts, labels):
        cats = {"positive": label == "positive", "negative": label == "negative"}
        doc = nlp.make_doc(text)
        train_data.append(Example.from_dict(doc, {"cats": cats}))
  
    # Train the model
    optimizer = nlp.begin_training()
    for i in range(10):  # 10 iterations
        losses = {}
        batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
        for batch in batches:
            nlp.update(batch, drop=0.5, losses=losses)
  
    # Test on new data
    for text in test_texts:
        doc = nlp(text)
        prediction = max(doc.cats.items(), key=lambda x: x[1])[0]
        print(f"Text: '{text}' -> Predicted: {prediction}")
    """
  
    print("\nFor production use, spaCy offers a more integrated pipeline approach")
    print("that can incorporate word vectors and neural network classifiers.")

# Example usage
text_classification_comparison()
```

### 13. Sentiment Analysis

```python
def sentiment_analysis_comparison(texts):
    """Compare sentiment analysis approaches between NLTK and spaCy"""
  
    print("Sentiment Analysis Examples:")
    for text in texts:
        print(f"- \"{text}\"")
  
    # NLTK Sentiment Analysis with VADER
    try:
        from nltk.sentiment import SentimentIntensityAnalyzer
      
        print("\n--- NLTK Sentiment Analysis with VADER ---")
      
        # Initialize VADER
        sia = SentimentIntensityAnalyzer()
      
        # Analyze each text
        for text in texts:
            scores = sia.polarity_scores(text)
            compound = scores['compound']
          
            # Interpret compound score
            if compound >= 0.05:
                sentiment = "Positive"
            elif compound <= -0.05:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"
          
            print(f"Text: \"{text}\"")
            print(f"Scores: {scores}")
            print(f"Sentiment: {sentiment}\n")
  
    except:
        print("NLTK VADER not available. Install with:")
        print("nltk.download('vader_lexicon')")
  
    # spaCy Sentiment Analysis
    print("\n--- spaCy Sentiment Analysis ---")
    print("spaCy doesn't have built-in sentiment analysis.")
    print("You would typically use a custom pipeline component or")
    print("integrate with other libraries like TextBlob:")
  
    try:
        import spacy
        from textblob import TextBlob
      
        nlp = spacy.load('en_core_web_sm')
      
        # Add sentiment analysis with TextBlob
        class TextBlobSentiment:
            def __call__(self, doc):
                blob = TextBlob(doc.text)
                # TextBlob polarity: -1.0 to 1.0
                doc._.sentiment = blob.sentiment.polarity
                return doc
      
        # Register extension
        if not spacy.tokens.Doc.has_extension("sentiment"):
            spacy.tokens.Doc.set_extension("sentiment", default=None)
      
        # Add to pipeline
        nlp.add_pipe(TextBlobSentiment(), last=True)
      
        # Process texts
        for text in texts:
            doc = nlp(text)
            polarity = doc._.sentiment
          
            # Interpret polarity
            if polarity >= 0.05:
                sentiment = "Positive"
            elif polarity <= -0.05:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"
          
            print(f"Text: \"{text}\"")
            print(f"Polarity: {polarity:.2f}")
            print(f"Sentiment: {sentiment}\n")
  
    except:
        print("TextBlob not available. Install with: pip install textblob")

# Example usage
sample_texts = [
    "I absolutely love this product! It's amazing.",
    "This is the worst experience I've ever had.",
    "The service was okay, nothing special."
]
sentiment_analysis_comparison(sample_texts)
```

### 14. Topic Modeling

```python
def topic_modeling_comparison():
    """Compare topic modeling approaches with NLTK/Gensim and spaCy"""
  
    # Sample corpus
    documents = [
        "Machine learning is a subset of artificial intelligence.",
        "Neural networks are popular in deep learning applications.",
        "Natural language processing helps computers understand human language.",
        "Computer vision enables machines to see and interpret visual information.",
        "Data mining extracts patterns from large datasets.",
        "Reinforcement learning trains agents through reward systems.",
        "The stock market has been volatile this week.",
        "Investors are concerned about economic growth.",
        "Companies reported mixed quarterly earnings.",
        "Financial analysts predict market recovery soon."
    ]
  
    print("Topic Modeling with 10 documents on AI and Finance topics")
  
    # NLTK + Gensim Topic Modeling
    try:
        import gensim
        from gensim import corpora
        from nltk.tokenize import word_tokenize
        from nltk.corpus import stopwords
      
        print("\n--- NLTK + Gensim Topic Modeling (LDA) ---")
      
        # Preprocessing
        stop_words = set(stopwords.words('english'))
      
        texts = []
        for doc in documents:
            # Tokenize and remove stopwords
            tokens = [word.lower() for word in word_tokenize(doc) 
                      if word.isalpha() and word.lower() not in stop_words]
            texts.append(tokens)
      
        # Create dictionary and corpus
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text
```
