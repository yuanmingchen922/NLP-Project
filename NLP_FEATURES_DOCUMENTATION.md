# Yelp Review Analysis System - NLP Features Documentation

**Version:** 1.1.0
**Date:** 2025-12-01
**Author:** Mingchen Yuan

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Complete Workflow](#complete-workflow)
4. [NLP Features](#nlp-features)
5. [Technologies & Methods](#technologies--methods)
6. [API Reference](#api-reference)
7. [Usage Guide](#usage-guide)
8. [Data Pipeline](#data-pipeline)

---

## Project Overview

### Purpose
A comprehensive Natural Language Processing system for analyzing Yelp business reviews. The system provides multiple NLP-powered features to extract insights from customer reviews, including sentiment analysis, entity recognition, keyword extraction, quality scoring, and business intelligence.

### Dataset
- **Source:** Yelp Academic Dataset
- **Size:** 100,000 sampled reviews (from 6.9M total)
- **Businesses:** 9,973 unique businesses
- **Format:** JSON files merged into processed CSV

### Key Objectives
1. Analyze customer sentiment across multiple dimensions
2. Extract meaningful insights from unstructured review text
3. Provide business intelligence through advanced analytics
4. Detect review quality and spam patterns
5. Enable intelligent search and recommendation

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (HTML/JS/CSS)                  │
│  - Search Interface  - Dashboard  - Visualizations          │
└────────────────────┬────────────────────────────────────────┘
                     │
                     │ HTTP/JSON
                     │
┌────────────────────▼────────────────────────────────────────┐
│                   Flask REST API (app.py)                   │
│  - 30+ API Endpoints  - Caching  - Request Handling         │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
┌───────▼────────┐      ┌─────────▼──────────┐
│  NLP Models    │      │  Data Processing   │
│  - Sentiment   │      │  - Pandas          │
│  - NER         │      │  - NumPy           │
│  - Keywords    │      │  - Sklearn         │
│  - Quality     │      │  - Statistics      │
│  - Spam        │      └────────────────────┘
│  - Aspects     │
└────────────────┘
        │
┌───────▼────────────────────────────────────────┐
│  External AI Services (Optional)               │
│  - OpenAI GPT-4o-mini (dish extraction)        │
└────────────────────────────────────────────────┘
```

---

## Complete Workflow

### 1. Data Ingestion Workflow

```
Raw Yelp JSON Files
       │
       ▼
┌──────────────────┐
│  Data Sampling   │ (100K reviews selected)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Data Merging    │ (Join reviews + business + user data)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Preprocessing   │ (Text cleaning, tokenization)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Feature         │ (TF-IDF, embeddings)
│  Extraction      │
└────────┬─────────┘
         │
         ▼
    Processed CSV
```

### 2. Model Training Workflow

```
Processed Data
       │
       ├─────────────────────────────────────┐
       │                                     │
       ▼                                     ▼
┌─────────────────┐              ┌──────────────────┐
│  Traditional ML │              │  Deep Learning   │
│  - Logistic Reg │              │  - BERT          │
│  - SVM          │              │  (Optional)      │
│  - Naive Bayes  │              └──────────────────┘
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Model Saving   │ (sentiment_lr.pkl)
└─────────────────┘
```

### 3. Runtime Analysis Workflow

```
User Search Query
       │
       ▼
┌──────────────────┐
│  Search Engine   │ (Text matching, filtering)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Business        │ (Load relevant reviews)
│  Selection       │
└────────┬─────────┘
         │
         ├──────────────┬──────────────┬──────────────┐
         │              │              │              │
         ▼              ▼              ▼              ▼
┌────────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│ Sentiment  │  │   NER    │  │ Keywords │  │ Quality  │
│ Analysis   │  │ Extract  │  │ Extract  │  │ Analysis │
└─────┬──────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘
      │              │              │              │
      └──────────────┴──────────────┴──────────────┘
                     │
                     ▼
            ┌────────────────┐
            │  Cache Result  │ (1-hour TTL)
            └────────┬───────┘
                     │
                     ▼
            ┌────────────────┐
            │  JSON Response │
            └────────┬───────┘
                     │
                     ▼
            Frontend Rendering
```

---

## NLP Features

### Feature 1: Sentiment Analysis

#### Design Process
1. **Problem Definition:** Classify reviews into positive/negative/neutral sentiment
2. **Approach Selection:** Traditional ML (fast, interpretable) vs Deep Learning (accurate but heavy)
3. **Label Definition:** 1-2 stars → Negative, 3 stars → Neutral, 4-5 stars → Positive
4. **Model Selection:** Logistic Regression chosen for balance of speed and accuracy

#### Technologies Used
- **Scikit-learn:** LogisticRegression, SVM, MultinomialNB
- **Feature Engineering:** TF-IDF vectorization (max 5000 features)
- **Preprocessing:** NLTK for stopword removal, lemmatization
- **Optional:** BERT for sequence classification (deep learning variant)

#### Implementation Details
```python
# Model: TraditionalSentimentAnalyzer
# Location: src/models/sentiment_analysis.py
# Training: train_all_models.py
# API Endpoint: POST /api/analyze/sentiment
```

**Key Parameters:**
- Model Type: 'logistic' (default), 'svm', 'naive_bayes'
- Max Iterations: 1000
- TF-IDF Max Features: 5000
- Test Split: 20%

#### How to Use
```python
# Via API
POST /api/analyze/sentiment
{
  "text": "The food was amazing and service was great!"
}

# Response
{
  "sentiment": "positive",
  "sentiment_score": 2,
  "confidence": 0.92,
  "label": "Positive"
}
```

#### Performance Metrics
- **Accuracy:** ~85-90% on test set
- **F1-Score:** 0.87 (weighted average)
- **Inference Time:** <50ms per review

---

### Feature 2: Named Entity Recognition (NER)

#### Design Process
1. **Goal:** Extract business-relevant entities (people, locations, products, times)
2. **Tool Selection:** spaCy chosen for speed and accuracy
3. **Entity Filtering:** Focus on PERSON, ORG, GPE, LOC, PRODUCT, TIME, DATE
4. **Post-processing:** Aggregate and count entity mentions

#### Technologies Used
- **spaCy:** en_core_web_sm model (12MB lightweight model)
- **Entity Types:** 18 standard NER labels
- **Processing:** Real-time text processing with caching

#### Implementation Details
```python
# Model: NamedEntityRecognizer
# Location: src/models/advanced_nlp.py
# API Endpoint: POST /api/analyze/entities
```

**Recognized Entity Types:**
- PERSON: People mentioned (waiters, chefs, owners)
- ORG: Organizations (chain names, suppliers)
- GPE: Locations (cities, countries)
- PRODUCT: Menu items, dishes
- TIME/DATE: Visit times, opening hours

#### How to Use
```python
# Via API
POST /api/analyze/entities
{
  "text": "John the waiter served us delicious pasta at Mario's in New York"
}

# Response
{
  "entities": {
    "PERSON": ["John", "Mario"],
    "GPE": ["New York"],
    "PRODUCT": ["pasta"]
  },
  "summary": {
    "PERSON": 2,
    "GPE": 1,
    "PRODUCT": 1
  }
}
```

---

### Feature 3: Keyword Extraction

#### Design Process
1. **Method Selection:** TF-IDF + Noun Phrase Extraction
2. **Combination Strategy:** Merge statistical (TF-IDF) and linguistic (noun chunks) approaches
3. **Ranking:** Frequency-based scoring
4. **Filtering:** Remove stopwords, short words (<3 chars)

#### Technologies Used
- **spaCy:** Noun chunk detection, POS tagging
- **Counter:** Frequency analysis
- **Lemmatization:** Normalize word forms

#### Implementation Details
```python
# Model: KeywordExtractor
# Location: src/models/advanced_nlp.py
# API Endpoint: POST /api/analyze/keywords
```

**Extraction Strategy:**
1. Extract noun chunks (multi-word phrases)
2. Extract individual nouns, proper nouns, adjectives
3. Filter stopwords and short tokens
4. Count frequencies
5. Return top N by frequency

#### How to Use
```python
# Via API
POST /api/analyze/keywords
{
  "text": "Amazing pizza with fresh ingredients. Great service!",
  "top_n": 5
}

# Response
{
  "keywords": [
    ["pizza", 3],
    ["fresh ingredients", 2],
    ["service", 2],
    ["amazing", 1],
    ["great", 1]
  ]
}
```

---

### Feature 4: Text Summarization

#### Design Process
1. **Approach:** Extractive summarization (select important sentences)
2. **Scoring Method:** TF-IDF weighted sentence importance
3. **Sentence Selection:** Top N sentences by score
4. **Length Control:** Configurable max sentences

#### Technologies Used
- **spaCy:** Sentence segmentation
- **TF-IDF:** Sentence importance scoring
- **Normalization:** Min-max scaling of scores

#### Implementation Details
```python
# Model: TextSummarizer
# Location: src/models/advanced_nlp.py
# API Endpoint: POST /api/analyze/summarize
```

**Algorithm:**
1. Split text into sentences
2. Calculate TF-IDF scores for each word
3. Score each sentence as sum of word TF-IDF scores
4. Normalize scores to 0-1 range
5. Select top N sentences
6. Return in original order

#### How to Use
```python
# Via API
POST /api/analyze/summarize
{
  "text": "Long review text...",
  "max_sentences": 3
}

# Response
{
  "summary": "The food was excellent. Service was fast. Would recommend.",
  "original_length": 250,
  "summary_length": 45,
  "compression_ratio": 0.18
}
```

---

### Feature 5: Aspect-Based Sentiment Analysis

#### Design Process
1. **Aspect Definition:** Identify 5 core aspects: food_quality, service, atmosphere, price_value, cleanliness
2. **Keyword Mapping:** Define keyword dictionaries for each aspect
3. **Context Window:** Analyze 8 words around each keyword
4. **Sentiment Scoring:** Apply sentiment to each aspect mention
5. **Aggregation:** Average scores across all reviews for a business

#### Technologies Used
- **Custom Aspect Keywords:** Hand-crafted dictionaries
- **Context Analysis:** Window-based sentiment attribution
- **Negation Handling:** Detect "not good" → negative
- **Intensity Modifiers:** "very delicious" > "delicious"

#### Implementation Details
```python
# Model: EnhancedSentimentAspectAnalyzer
# Location: src/models/advanced_nlp.py (line 429+)
# API Endpoint: POST /api/analyze/aspects
#              GET /api/business/<id>/aspects-enhanced
```

**Aspects and Keywords:**
```python
ENHANCED_ASPECT_KEYWORDS = {
    'food_quality': ['food', 'meal', 'dish', 'taste', 'flavor', 'delicious', 'fresh'],
    'service': ['service', 'staff', 'waiter', 'waitress', 'server', 'friendly'],
    'atmosphere': ['atmosphere', 'ambiance', 'decor', 'music', 'vibe', 'cozy'],
    'price_value': ['price', 'expensive', 'cheap', 'value', 'worth', 'affordable'],
    'cleanliness': ['clean', 'dirty', 'hygiene', 'sanitary', 'neat', 'tidy']
}
```

#### How to Use
```python
# Via API (single review)
POST /api/analyze/aspects
{
  "text": "Great food but service was slow. Clean place though."
}

# Response
{
  "aspects": {
    "food_quality": {"sentiment": "positive", "score": 0.8},
    "service": {"sentiment": "negative", "score": -0.5},
    "cleanliness": {"sentiment": "positive", "score": 0.6}
  }
}

# Via API (business aggregation)
GET /api/business/abc123/aspects-enhanced

# Response
{
  "food_quality": {
    "average_rating": 4.5,
    "average_score": 0.8,
    "mention_count": 120,
    "coverage_percentage": 85.0,
    "sentiment_distribution": {
      "positive": 102,
      "negative": 10,
      "neutral": 8
    }
  },
  ...
}
```

---

### Feature 6: Dish/Menu Item Extraction

#### Design Process
1. **Multi-Strategy Approach:**
   - **Strategy A:** Rule-based regex patterns
   - **Strategy B:** Frequency analysis + spaCy noun phrases
   - **Strategy C:** AI fallback using OpenAI GPT-4o-mini
2. **Combination Logic:** Try A+B first, fallback to C if <3 dishes found
3. **Ranking:** Sort by (frequency × average_rating)

#### Technologies Used
- **Regex Patterns:** Detect common dish patterns
- **spaCy:** Noun phrase extraction
- **Food Categories:** Pre-defined food type lists
- **OpenAI API:** GPT-4o-mini for intelligent extraction (fallback)
- **Pandas:** Aggregation and ranking

#### Implementation Details
```python
# Model: DishExtractor
# Location: src/models/advanced_nlp.py (line 500+)
# API Endpoint: GET /api/business/<id>/dishes
```

**Extraction Methods:**

**Method A - Rule-Based:**
```python
# Patterns: "the X", "ordered X", "tried the X"
# Food categories: ['pizza', 'burger', 'pasta', 'salad', ...]
```

**Method B - Frequency:**
```python
# Extract noun phrases with spaCy
# Filter by food-related terms
# Rank by frequency × rating
```

**Method C - AI Fallback:**
```python
# Sample 5 high-rated + 5 low-rated reviews
# Send to OpenAI: "Extract mentioned dishes"
# Parse JSON response
```

#### How to Use
```python
# Via API
GET /api/business/abc123/dishes?top_n=10

# Response
{
  "business_id": "abc123",
  "dishes": [
    {
      "name": "Margherita Pizza",
      "mention_count": 45,
      "average_rating": 4.8,
      "relevance_score": 216.0,
      "sentiment": "very positive"
    },
    {
      "name": "Caesar Salad",
      "mention_count": 32,
      "average_rating": 4.2,
      "relevance_score": 134.4,
      "sentiment": "positive"
    }
  ],
  "extraction_method": "combined",
  "total_dishes": 10
}
```

---

### Feature 7: Review Quality Scoring

#### Design Process
1. **Quality Dimensions:**
   - Length (optimal: 50-300 words)
   - Detail (number of specific nouns/verbs)
   - Grammar (sentence structure, punctuation)
   - Objectivity (balance of opinion vs. facts)
   - Informativeness (unique vocabulary richness)
2. **Weighted Scoring:** Combine 5 dimensions with weights
3. **Classification:** Map 0-100 score to quality labels

#### Technologies Used
- **spaCy:** Grammar and syntax analysis
- **TextBlob:** Subjectivity detection
- **Custom Metrics:** Length, vocabulary diversity
- **Scoring Weights:**
  - Length: 15%
  - Detail: 25%
  - Grammar: 20%
  - Objectivity: 20%
  - Informativeness: 20%

#### Implementation Details
```python
# Model: ReviewQualityAnalyzer
# Location: src/models/professional_features.py
# API Endpoint: POST /api/analyze/quality
```

**Quality Calculation:**
```python
overall_score = (
    length_score * 0.15 +
    detail_score * 0.25 +
    grammar_score * 0.20 +
    objectivity_score * 0.20 +
    informative_score * 0.20
)
```

**Quality Labels:**
- 75-100: Excellent
- 60-74: Good
- 45-59: Fair
- 30-44: Poor
- 0-29: Very Poor

#### How to Use
```python
# Via API
POST /api/analyze/quality
{
  "text": "Review text here..."
}

# Response
{
  "overall_score": 78.5,
  "quality_label": "Excellent",
  "quality_class": "high",
  "metrics": {
    "length_score": 85.0,
    "detail_score": 75.0,
    "grammar_score": 90.0,
    "objectivity_score": 70.0,
    "informative_score": 72.5
  },
  "insights": [
    "Well-structured review with good length",
    "Highly detailed and specific",
    "Good grammar and punctuation"
  ]
}
```

---

### Feature 8: Spam Detection

#### Design Process
1. **Spam Indicators:**
   - Excessive capitalization
   - Repetitive patterns
   - Promotional keywords
   - Suspicious URLs/emails
   - Generic/template language
2. **Scoring System:** Each indicator contributes to spam score
3. **Threshold:** >60 = likely spam

#### Technologies Used
- **Regex Patterns:** Detect caps, repetition, URLs
- **Keyword Matching:** Promotional terms
- **Statistical Analysis:** Character/word repetition rates
- **Custom Heuristics:** Template detection

#### Implementation Details
```python
# Model: SpamDetector
# Location: src/models/professional_features.py
# API Endpoint: POST /api/analyze/spam
```

**Spam Indicators:**
```python
spam_indicators = {
    'excessive_caps': 15,      # >30% CAPS
    'repetition': 20,          # Same word 5+ times
    'promotional': 25,         # Keywords: "buy", "discount", "click here"
    'suspicious_urls': 30,     # Multiple URLs/emails
    'generic_language': 10     # Template phrases
}
```

#### How to Use
```python
# Via API
POST /api/analyze/spam
{
  "text": "BUY NOW!!! CLICK HERE for AMAZING DEALS!!!"
}

# Response
{
  "spam_score": 85.0,
  "is_spam": true,
  "spam_indicators": [
    "Excessive capitalization detected",
    "Contains promotional keywords",
    "Repetitive patterns found"
  ],
  "confidence": "high"
}
```

---

### Feature 9: Credibility Analysis

#### Design Process
1. **Credibility Factors:**
   - Verified purchase indicators
   - Account age/activity (if available)
   - Review consistency
   - Specific details vs. vague claims
   - Balanced perspective
2. **Multi-dimensional scoring**
3. **Trust level classification**

#### Technologies Used
- **Pattern Matching:** Verified purchase phrases
- **Consistency Analysis:** Cross-review comparison
- **Detail Detection:** Specific nouns/numbers
- **Balance Scoring:** Pro/con ratio

#### Implementation Details
```python
# Model: CredibilityAnalyzer
# Location: src/models/professional_features.py
# API Endpoint: POST /api/analyze/credibility
```

#### How to Use
```python
# Via API
POST /api/analyze/credibility
{
  "text": "I visited this restaurant last Tuesday...",
  "metadata": {
    "user_review_count": 25,
    "account_age_days": 730
  }
}

# Response
{
  "credibility_score": 82.0,
  "trust_level": "High",
  "factors": {
    "specificity": 85.0,
    "balance": 75.0,
    "user_history": 90.0
  },
  "red_flags": []
}
```

---

### Feature 10: Business Analytics & Trends

#### Design Process
1. **Time-Series Analysis:** Group reviews by month/quarter/year
2. **Trend Detection:** Linear regression on sentiment over time
3. **Improvement Scoring:** 0-100 score based on trend direction
4. **Comparative Analysis:** Business vs. category averages

#### Technologies Used
- **Pandas:** Time-series grouping and aggregation
- **SciPy:** Linear regression (scipy.stats.linregress)
- **NumPy:** Statistical calculations
- **Custom Metrics:** Improvement score algorithm

#### Implementation Details
```python
# Model: SentimentTrendAnalyzer, BusinessScorecard
# Location: src/models/business_analytics.py
# API Endpoints:
#   - GET /api/business/trends/<id>
#   - GET /api/business/scorecard/<id>
#   - GET /api/business/sentiment-shifts/<id>
#   - GET /api/business/<id>/improvement-trends
```

**Improvement Score Calculation:**
```python
improvement_score = (
    trend_direction * 40 +      # Slope (positive/negative)
    performance_improvement * 40 +  # Recent vs. historical
    consistency * 20            # Variance
)
```

**Trend Classifications:**
- Strongly Improving: Score 80-100
- Improving: Score 60-79
- Stable: Score 40-59
- Declining: Score 0-39

#### How to Use
```python
# Via API - Scorecard
GET /api/business/scorecard/abc123

# Response
{
  "business_id": "abc123",
  "overall_score": 85.5,
  "metrics": {
    "sentiment_score": 88.0,
    "quality_score": 82.0,
    "consistency": 86.5,
    "recency": 90.0
  },
  "grade": "A",
  "percentile": 92
}

# Via API - Trends
GET /api/business/trends/abc123?period=quarter

# Response
{
  "business_id": "abc123",
  "trend_data": [
    {"period": "2024-Q1", "avg_rating": 4.2, "review_count": 45},
    {"period": "2024-Q2", "avg_rating": 4.5, "review_count": 52},
    {"period": "2024-Q3", "avg_rating": 4.7, "review_count": 58}
  ],
  "trend_direction": "improving",
  "improvement_score": 87.5
}
```

---

### Feature 11: Business Search & Filtering

#### Design Process
1. **Search Strategy:** Text matching on name, address, categories
2. **Ranking:** Sort by relevance, rating, review count
3. **Filtering:** By category, rating range, location
4. **Optimization:** Case-insensitive, partial matching

#### Technologies Used
- **Pandas:** DataFrame filtering and sorting
- **String Matching:** Contains, case-insensitive comparison
- **Multi-field Search:** Combine name + address + category

#### Implementation Details
```python
# API Endpoint: GET /api/businesses
# Location: src/api/app.py (line 221)
```

#### How to Use
```python
# Via API
GET /api/businesses?search=pizza&limit=10

# Response
{
  "businesses": [
    {
      "business_id": "abc123",
      "name": "Mario's Pizza",
      "rating": 4.5,
      "review_count": 245,
      "address": "123 Main St",
      "categories": "Pizza, Italian"
    }
  ],
  "count": 10,
  "total": 156
}
```

---

## Technologies & Methods

### Core NLP Libraries

| Library | Version | Purpose | Features Used |
|---------|---------|---------|---------------|
| **spaCy** | 3.7+ | NLP processing | NER, POS tagging, dependency parsing, noun chunks |
| **NLTK** | 3.8+ | Text preprocessing | Stopwords, tokenization, lemmatization |
| **Scikit-learn** | 1.3+ | Machine learning | LogisticRegression, SVM, TF-IDF, train/test split |
| **Transformers** | 4.30+ | Deep learning (optional) | BERT tokenizer, sequence classification |
| **TextBlob** | 0.17+ | Simple NLP | Sentiment polarity, subjectivity |

### Data Processing

| Library | Purpose | Key Operations |
|---------|---------|----------------|
| **Pandas** | Data manipulation | DataFrame operations, groupby, merge, filtering |
| **NumPy** | Numerical computing | Array operations, statistical functions |
| **SciPy** | Scientific computing | Linear regression, statistical tests |

### Web Framework & API

| Library | Purpose | Features |
|---------|---------|----------|
| **Flask** | Web server | REST API, routing, JSON responses |
| **Flask-CORS** | Cross-origin | Enable frontend-backend communication |

### External Services

| Service | Purpose | Usage |
|---------|---------|-------|
| **OpenAI API** | AI assistance | Dish extraction fallback (GPT-4o-mini) |
| **SerpAPI** | (Optional) Real-time data | Fetch live Yelp data |

### Frontend

| Technology | Purpose |
|-----------|---------|
| **Bootstrap 5** | UI framework |
| **Chart.js** | Data visualization |
| **Vanilla JavaScript** | Interactivity |

---

## API Reference

### Search & Business Info

#### 1. Search Businesses
```http
GET /api/businesses?search={query}&limit={n}
```
**Response:** List of matching businesses

#### 2. Get Business Details
```http
GET /api/business/<business_id>
```
**Response:** Business information (name, rating, address, etc.)

#### 3. Get Business Reviews
```http
GET /api/business/<business_id>/reviews?limit={n}
```
**Response:** Recent reviews sorted by date

#### 4. Get Business Statistics
```http
GET /api/business/<business_id>/stats
```
**Response:** Aggregated statistics (total reviews, avg rating, distribution)

---

### NLP Analysis Endpoints

#### 5. Sentiment Analysis
```http
POST /api/analyze/sentiment
Body: {"text": "review text"}
```
**Response:** Sentiment label, score, confidence

#### 6. Named Entity Recognition
```http
POST /api/analyze/entities
Body: {"text": "review text"}
```
**Response:** Extracted entities by type

#### 7. Keyword Extraction
```http
POST /api/analyze/keywords
Body: {"text": "review text", "top_n": 10}
```
**Response:** Top keywords with frequencies

#### 8. Text Summarization
```http
POST /api/analyze/summarize
Body: {"text": "long review", "max_sentences": 3}
```
**Response:** Summarized text

#### 9. Aspect-Based Sentiment
```http
POST /api/analyze/aspects
Body: {"text": "review text"}
```
**Response:** Sentiment per aspect (food, service, etc.)

#### 10. Quality Scoring
```http
POST /api/analyze/quality
Body: {"text": "review text"}
```
**Response:** Quality score (0-100) with breakdown

#### 11. Spam Detection
```http
POST /api/analyze/spam
Body: {"text": "review text"}
```
**Response:** Spam score and indicators

#### 12. Credibility Analysis
```http
POST /api/analyze/credibility
Body: {"text": "review text", "metadata": {...}}
```
**Response:** Credibility score and trust level

#### 13. Comprehensive Analysis
```http
POST /api/analyze/comprehensive
Body: {"text": "review text"}
```
**Response:** All analyses combined (sentiment, entities, quality, etc.)

---

### Business Intelligence

#### 14. Business Scorecard
```http
GET /api/business/scorecard/<business_id>
```
**Response:** Overall business score, grade, metrics

#### 15. Sentiment Trends
```http
GET /api/business/trends/<business_id>?period={month|quarter|year}
```
**Response:** Time-series sentiment data

#### 16. Improvement Trends
```http
GET /api/business/<business_id>/improvement-trends?period={period}
```
**Response:** Improvement score, trend direction, visualization data

#### 17. Popular Dishes
```http
GET /api/business/<business_id>/dishes?top_n={n}
```
**Response:** Top dishes with ratings and mentions

#### 18. Enhanced Aspects
```http
GET /api/business/<business_id>/aspects-enhanced
```
**Response:** Aggregated aspect analysis across all reviews

#### 19. Business Comparison
```http
POST /api/business/compare
Body: {"business_ids": ["id1", "id2"]}
```
**Response:** Side-by-side comparison metrics

---

### System Endpoints

#### 20. System Overview
```http
GET /api/stats/overview
```
**Response:** Dataset statistics (total reviews, businesses, ratings distribution)

#### 21. Top Businesses
```http
GET /api/insights/top-businesses?limit={n}
```
**Response:** Highest-rated businesses

#### 22. Word Cloud Data
```http
GET /api/insights/word-cloud?sentiment={positive|negative|all}
```
**Response:** Word frequencies for visualization

---

## Usage Guide

### Installation

```bash
# 1. Clone repository
git clone https://github.com/yuanmingchen922/NLP-Project.git
cd NLP-Project

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download spaCy model
python -m spacy download en_core_web_sm

# 5. Set up environment variables (optional)
cp .env.example .env
# Edit .env and add your API keys
```

### Running the System

```bash
# Start the Flask server
python run.py

# Server will start at:
# http://127.0.0.1:5001
```

### Training Models

```bash
# Train all models (requires raw data)
python train_all_models.py

# Models will be saved to:
# data/models/sentiment_lr.pkl
```

### Using the Web Interface

1. **Search:** Enter business name or category in search box
2. **View Results:** Click on business to see details
3. **Analyze Reviews:** Click "View Reviews" to see customer reviews
4. **View Popular Dishes:** Click "Popular Dishes" to see menu items
5. **Check Trends:** View rating trends over time

### Using the API

```python
import requests

# Example: Analyze sentiment
response = requests.post('http://127.0.0.1:5001/api/analyze/sentiment',
    json={'text': 'The food was amazing!'})
print(response.json())

# Example: Get business info
response = requests.get('http://127.0.0.1:5001/api/business/abc123')
print(response.json())
```

---

## Data Pipeline

### 1. Data Collection
- Source: Yelp Academic Dataset
- Files: review.json, business.json, user.json
- Sampling: 100K reviews randomly selected

### 2. Data Preprocessing
```python
# Text Cleaning Steps:
1. Convert to lowercase
2. Remove URLs, emails
3. Remove special characters (keep letters, numbers, spaces)
4. Remove extra whitespace
5. Tokenization
6. Stopword removal (optional)
7. Lemmatization (optional)
```

### 3. Feature Engineering
```python
# TF-IDF Features
- Max features: 5000
- N-gram range: (1, 2)
- Min document frequency: 2

# Additional Features
- Review length (words)
- User review count
- Business average rating
- Review date (for trends)
```

### 4. Model Training
```python
# Train/Test Split
- Test size: 20%
- Stratified by sentiment label
- Random state: 42

# Model Selection
- Cross-validation: 5-fold
- Metric: F1-score (weighted)
```

### 5. Model Deployment
```python
# Serialization
- Format: Pickle (.pkl)
- Location: data/models/
- Loading: On server startup

# Caching
- In-memory cache
- TTL: 1 hour
- Max entries: 1000
```

---

## Performance Optimization

### Caching Strategy
```python
# Cache Implementation
_cache_store = {}  # {key: (data, timestamp)}
CACHE_TTL_SECONDS = 3600  # 1 hour

# Cache Keys
- reviews_{business_id}_{limit}
- dishes_{business_id}_{top_n}
- aspects_{business_id}
- trends_{business_id}_{period}
```

### Performance Targets
- First request (cache miss): < 3 seconds
- Cached request: < 100ms
- Chart rendering: < 500ms
- Cache hit rate: > 70%

---

## Future Enhancements

### Planned Features (v1.2+)
1. **Redis Cache:** Persistent caching across server restarts
2. **Real-time Updates:** WebSocket for live data
3. **Multi-Business Comparison:** Compare multiple businesses side-by-side
4. **Image Analysis:** Dish photo recognition
5. **Multi-language Support:** Analyze reviews in other languages
6. **Export Reports:** PDF/Excel report generation
7. **Aspect-Based Search:** "Find restaurants with great atmosphere"
8. **Recommendation System:** Personalized business recommendations

---

## Troubleshooting

### Common Issues

**Issue:** spaCy model not found
```bash
# Solution
python -m spacy download en_core_web_sm
```

**Issue:** OpenAI API errors
```bash
# Check API key in .env file
OPENAI_API_KEY=your_key_here
```

**Issue:** Port 5001 already in use
```python
# Change port in config.py
FLASK_PORT = 5002
```

**Issue:** Pandas DataFrame errors
```bash
# Ensure processed data exists
ls data/processed/reviews_merged.csv
```

---

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

---

## License

This project is for academic purposes. Please cite the Yelp Academic Dataset if using this code.

---

## Contact

**Developer:** Mingchen Yuan
**GitHub:** https://github.com/yuanmingchen922/NLP-Project
**Version:** 1.1.0
**Last Updated:** 2025-12-01

---

## Acknowledgments

- **Yelp Academic Dataset:** For providing the review data
- **spaCy:** For excellent NLP tools
- **OpenAI:** For GPT-4o-mini API
- **Bootstrap:** For UI components
- **Chart.js:** For visualizations
