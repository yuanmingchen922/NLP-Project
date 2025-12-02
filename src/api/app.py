"""
Yelp Review Analysis System - Flask Web Application
Advanced NLP-powered review analysis platform
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging
from difflib import SequenceMatcher, get_close_matches
from openai import OpenAI
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import (
    PROCESSED_DATA_DIR, MODEL_DIR, FLASK_HOST, FLASK_PORT, DEBUG,
    OPENAI_API_KEY, OPENAI_MODEL, OPENAI_TEMPERATURE, OPENAI_MAX_TOKENS
)
from src.data_processing.preprocessor import TextPreprocessor, SentimentPreprocessor
from src.data_processing.feature_extraction import extract_text_statistics
from src.models.advanced_nlp import (
    NamedEntityRecognizer,
    KeywordExtractor,
    TextSummarizer,
    SentimentAspectAnalyzer,
    analyze_review_comprehensive
)
from src.models.professional_features import (
    ReviewQualityAnalyzer,
    SpamDetector,
    CredibilityAnalyzer,
    analyze_review_comprehensive_professional
)
from src.models.business_analytics import (
    SentimentTrendAnalyzer,
    BusinessInsightsAnalyzer,
    generate_executive_summary
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask application
app = Flask(__name__,
            template_folder='../../frontend/templates',
            static_folder='../../frontend/static')
CORS(app)

# Global variables for data and models
reviews_df = None
businesses_df = None
models = {}

# Initialize advanced NLP models
advanced_nlp_models = {
    'ner': None,
    'keyword_extractor': None,
    'summarizer': None,
    'aspect_analyzer': None
}

# Initialize professional features
professional_models = {
    'quality_analyzer': None,
    'spam_detector': None,
    'credibility_analyzer': None
}

# Initialize business analytics
business_analytics = {
    'trend_analyzer': None,
    'insights_analyzer': None
}

# Initialize OpenAI client
openai_client = None
if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    logger.info("OpenAI client initialized")
else:
    logger.warning("OpenAI API key not found. Chatbot will use fallback mode.")


def load_data():
    """Load dataset"""
    global reviews_df, businesses_df

    logger.info("Loading data...")

    try:
        # Load reviews data
        reviews_path = PROCESSED_DATA_DIR / 'reviews_merged.csv'
        if reviews_path.exists():
            reviews_df = pd.read_csv(reviews_path)
            # Normalize column names - handle both 'rating' and 'stars'
            if 'stars' in reviews_df.columns and 'rating' not in reviews_df.columns:
                reviews_df['rating'] = reviews_df['stars']
            logger.info(f"Loaded {len(reviews_df)} reviews")
        else:
            logger.warning(f"Reviews file not found: {reviews_path}")
            reviews_df = pd.DataFrame()

        # Load business data
        businesses_path = PROCESSED_DATA_DIR / 'businesses.csv'
        if businesses_path.exists():
            businesses_df = pd.read_csv(businesses_path)
            logger.info(f"Loaded {len(businesses_df)} businesses")
        else:
            logger.warning(f"Businesses file not found: {businesses_path}")
            businesses_df = pd.DataFrame()

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        reviews_df = pd.DataFrame()
        businesses_df = pd.DataFrame()


def load_models():
    """Load trained models"""
    global models, advanced_nlp_models

    logger.info("Loading models...")

    try:
        # Load sentiment analysis model if exists
        import joblib
        sentiment_model_path = MODEL_DIR / 'sentiment_lr.pkl'
        tfidf_path = MODEL_DIR / 'tfidf_vectorizer.pkl'

        if sentiment_model_path.exists() and tfidf_path.exists():
            from src.models.sentiment_analysis import TraditionalSentimentAnalyzer
            models['sentiment'] = TraditionalSentimentAnalyzer(model_type='logistic')
            models['sentiment'].load(sentiment_model_path, tfidf_path)
            logger.info("Loaded sentiment analysis model")

    except Exception as e:
        logger.error(f"Error loading models: {e}")

    # Initialize advanced NLP models
    try:
        logger.info("Initializing advanced NLP models...")
        advanced_nlp_models['ner'] = NamedEntityRecognizer()
        advanced_nlp_models['keyword_extractor'] = KeywordExtractor()
        advanced_nlp_models['summarizer'] = TextSummarizer()
        advanced_nlp_models['aspect_analyzer'] = SentimentAspectAnalyzer()
        logger.info("Advanced NLP models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading advanced NLP models: {e}")

    # Initialize professional features
    try:
        logger.info("Initializing professional analysis models...")
        professional_models['quality_analyzer'] = ReviewQualityAnalyzer()
        professional_models['spam_detector'] = SpamDetector()
        professional_models['credibility_analyzer'] = CredibilityAnalyzer()
        logger.info("Professional analysis models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading professional models: {e}")

    # Initialize business analytics
    try:
        logger.info("Initializing business analytics...")
        if not reviews_df.empty:
            business_analytics['trend_analyzer'] = SentimentTrendAnalyzer(reviews_df)
            business_analytics['insights_analyzer'] = BusinessInsightsAnalyzer(reviews_df, businesses_df)
            logger.info("Business analytics initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing business analytics: {e}")


# Initialize
load_data()
load_models()


@app.route('/')
def index():
    """Home page"""
    stats = {}
    if not reviews_df.empty:
        stats = {
            'total_reviews': int(len(reviews_df)),
            'total_businesses': int(reviews_df['business_id'].nunique()),
            'total_users': int(reviews_df['user_id'].nunique()) if 'user_id' in reviews_df.columns else 0,
            'avg_rating': float(reviews_df['rating'].mean()) if 'rating' in reviews_df.columns else 0
        }

    return render_template('index.html', stats=stats)


@app.route('/analyze')
def analyze_page():
    """Analysis page"""
    return render_template('analyze.html')


@app.route('/dashboard')
def dashboard():
    """Dashboard page"""
    return render_template('dashboard.html')


@app.route('/insights')
def insights_page():
    """Advanced insights page"""
    return render_template('insights.html')


@app.route('/chatbot')
def chatbot_page():
    """Chatbot assistant page"""
    return render_template('chatbot.html')


@app.route('/api/businesses', methods=['GET'])
def get_businesses():
    """Get business list with advanced filtering"""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    search = request.args.get('search', '', type=str)
    min_rating = request.args.get('min_rating', type=float)
    category = request.args.get('category', '', type=str)
    city = request.args.get('city', '', type=str)

    if businesses_df.empty:
        return jsonify({'error': 'No business data available'}), 404

    # Apply filters
    df = businesses_df.copy()
    fuzzy_match_used = False
    corrected_search = None

    if search:
        # First try exact match (case insensitive)
        df = df[df['name'].str.contains(search, case=False, na=False)]

        # If no results found, try fuzzy matching
        if len(df) == 0:
            # Get all unique business names
            all_names = businesses_df['name'].dropna().unique().tolist()

            # Find close matches using difflib
            # cutoff=0.6 means 60% similarity threshold
            close_matches = get_close_matches(search, all_names, n=20, cutoff=0.6)

            if close_matches:
                fuzzy_match_used = True
                corrected_search = close_matches[0]  # Best match
                # Search for all close matches
                df = businesses_df[businesses_df['name'].isin(close_matches)]
            else:
                # If still no matches, try partial fuzzy matching with lower threshold
                # Calculate similarity scores for all names
                matches_with_scores = []
                for name in all_names:
                    similarity = SequenceMatcher(None, search.lower(), name.lower()).ratio()
                    if similarity >= 0.5:  # 50% similarity
                        matches_with_scores.append((name, similarity))

                if matches_with_scores:
                    # Sort by similarity score
                    matches_with_scores.sort(key=lambda x: x[1], reverse=True)
                    # Take top 20 matches
                    top_matches = [name for name, score in matches_with_scores[:20]]
                    fuzzy_match_used = True
                    corrected_search = top_matches[0] if top_matches else None
                    df = businesses_df[businesses_df['name'].isin(top_matches)]
                else:
                    df = businesses_df[businesses_df['name'].str.contains(search, case=False, na=False)]

    if min_rating is not None and 'stars' in df.columns:
        df = df[df['stars'] >= min_rating]

    if category and 'categories' in df.columns:
        df = df[df['categories'].str.contains(category, case=False, na=False)]

    if city and 'city' in df.columns:
        df = df[df['city'].str.contains(city, case=False, na=False)]

    # Sort by rating
    if 'stars' in df.columns:
        df = df.sort_values('stars', ascending=False)

    # Pagination
    total = len(df)
    start = (page - 1) * per_page
    end = start + per_page
    businesses = df.iloc[start:end]

    # Add Google Maps and Search links to each business
    business_list = businesses.to_dict('records')

    # Replace NaN values with None (null in JSON) to prevent JSON parsing errors in frontend
    for business in business_list:
        for key, value in list(business.items()):
            if pd.isna(value):
                business[key] = None

        # Generate Google Maps link
        if pd.notna(business.get('latitude')) and pd.notna(business.get('longitude')):
            lat = business['latitude']
            lon = business['longitude']
            business['google_maps_url'] = f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"
        elif pd.notna(business.get('address')) and pd.notna(business.get('city')):
            # Fallback to address-based search
            address = f"{business.get('address', '')}, {business.get('city', '')}, {business.get('state', '')} {business.get('postal_code', '')}"
            from urllib.parse import quote
            business['google_maps_url'] = f"https://www.google.com/maps/search/?api=1&query={quote(address)}"
        else:
            business['google_maps_url'] = None

        # Generate Google Search link for finding website
        if pd.notna(business.get('name')):
            from urllib.parse import quote
            search_query = f"{business['name']}"
            if pd.notna(business.get('city')):
                search_query += f" {business['city']}"
            if pd.notna(business.get('state')):
                search_query += f" {business['state']}"
            business['google_search_url'] = f"https://www.google.com/search?q={quote(search_query)}"
        else:
            business['google_search_url'] = None

        # Format full address string
        address_parts = []
        if pd.notna(business.get('address')):
            address_parts.append(str(business['address']))
        if pd.notna(business.get('city')):
            address_parts.append(str(business['city']))
        if pd.notna(business.get('state')):
            address_parts.append(str(business['state']))
        if pd.notna(business.get('postal_code')):
            address_parts.append(str(business['postal_code']))
        business['full_address'] = ', '.join(address_parts) if address_parts else 'Address not available'

        # Add review summary and count
        business_id = business.get('business_id')
        if business_id and not reviews_df.empty:
            # Get reviews for this business
            business_reviews = reviews_df[reviews_df['business_id'] == business_id]

            if not business_reviews.empty:
                review_count = len(business_reviews)
                business['review_count'] = review_count

                # Get top 3 most useful reviews or most recent ones
                if 'useful' in business_reviews.columns:
                    top_reviews = business_reviews.nlargest(3, 'useful')
                else:
                    top_reviews = business_reviews.head(3)

                # Generate summary
                if 'text' in top_reviews.columns:
                    review_texts = top_reviews['text'].dropna().tolist()

                    if review_texts:
                        # Use summarizer if available, otherwise create a simple summary
                        if advanced_nlp_models.get('summarizer') is not None:
                            try:
                                # Combine review texts and summarize
                                combined_text = ' '.join(review_texts[:3])  # Use top 3 reviews
                                summary = advanced_nlp_models['summarizer'].summarize(combined_text, num_sentences=3)
                                business['review_summary'] = summary
                            except Exception as e:
                                logger.warning(f"Error generating summary for business {business_id}: {e}")
                                # Fallback to excerpt of first review
                                business['review_summary'] = review_texts[0][:200] + '...' if len(review_texts[0]) > 200 else review_texts[0]
                        else:
                            # Simple fallback - use excerpt of first review
                            business['review_summary'] = review_texts[0][:200] + '...' if len(review_texts[0]) > 200 else review_texts[0]
                    else:
                        business['review_summary'] = 'No review text available'
                else:
                    business['review_summary'] = 'No review text available'

                # Add average rating from reviews
                if 'stars' in business_reviews.columns:
                    avg_review_rating = float(business_reviews['stars'].mean())
                    business['avg_review_rating'] = round(avg_review_rating, 1)
            else:
                # No reviews found - use OpenAI to generate a summary based on business info
                business['review_count'] = 0
                if openai_client:
                    try:
                        prompt = f"Generate a brief, realistic 2-sentence summary for a business with the following information:\nName: {business.get('name', 'Unknown')}\nCategory: {business.get('categories', 'General')}\nRating: {business.get('stars', 'N/A')} stars\nLocation: {business.get('city', '')}, {business.get('state', '')}\n\nCreate a professional summary that highlights what customers might expect from this business."

                        response = openai_client.chat.completions.create(
                            model=OPENAI_MODEL,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.7,
                            max_tokens=150
                        )
                        business['review_summary'] = response.choices[0].message.content.strip()
                        logger.info(f"Generated AI summary for business {business_id}")
                    except Exception as e:
                        logger.warning(f"Error generating AI summary: {e}")
                        business['review_summary'] = f"This {business.get('categories', 'business')} is rated {business.get('stars', 'N/A')} stars. Located in {business.get('city', 'the area')}."
                else:
                    business['review_summary'] = f"This {business.get('categories', 'business')} is rated {business.get('stars', 'N/A')} stars. Located in {business.get('city', 'the area')}."
        else:
            business['review_count'] = 0
            # Use OpenAI fallback for businesses with no review data
            if openai_client:
                try:
                    prompt = f"Generate a brief, realistic 2-sentence summary for a business with the following information:\nName: {business.get('name', 'Unknown')}\nCategory: {business.get('categories', 'General')}\nRating: {business.get('stars', 'N/A')} stars\nLocation: {business.get('city', '')}, {business.get('state', '')}\n\nCreate a professional summary that highlights what customers might expect from this business."

                    response = openai_client.chat.completions.create(
                        model=OPENAI_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.7,
                        max_tokens=150
                    )
                    business['review_summary'] = response.choices[0].message.content.strip()
                except Exception as e:
                    logger.warning(f"Error generating AI summary: {e}")
                    business['review_summary'] = 'No reviews available'
            else:
                business['review_summary'] = 'No reviews available'

    result = {
        'businesses': business_list,
        'total': total,
        'page': page,
        'per_page': per_page
    }

    # Add fuzzy match information if used
    if fuzzy_match_used and corrected_search:
        result['fuzzy_match'] = True
        result['original_search'] = search
        result['corrected_search'] = corrected_search
        result['message'] = f'Showing results for "{corrected_search}" (similar to "{search}")'
    else:
        result['fuzzy_match'] = False

    return jsonify(result)


@app.route('/api/business/<business_id>', methods=['GET'])
def get_business(business_id):
    """Get business details"""
    if businesses_df.empty:
        return jsonify({'error': 'No business data available'}), 404

    business = businesses_df[businesses_df['business_id'] == business_id]

    if business.empty:
        return jsonify({'error': 'Business not found'}), 404

    return jsonify(business.iloc[0].to_dict())


@app.route('/api/business/<business_id>/reviews', methods=['GET'])
def get_business_reviews(business_id):
    """Get business reviews with caching and sorting"""
    try:
        limit = request.args.get('limit', default=10, type=int)
        limit = min(limit, 50)  # Maximum 50 reviews

        # Check cache first
        cache_key = f"reviews_{business_id}_{limit}"
        cached_data = get_from_cache(cache_key)

        if cached_data is not None:
            logger.info(f"Cache hit for reviews: {business_id}")
            return jsonify(cached_data)

        logger.info(f"Cache miss for reviews: {business_id}, fetching...")

        if reviews_df.empty:
            return jsonify({'error': 'No review data available'}), 404

        # Get reviews for this business
        business_reviews = reviews_df[reviews_df['business_id'] == business_id]

        if len(business_reviews) == 0:
            return jsonify({'error': 'No reviews found for this business'}), 404

        # Sort by date (most recent first) and get top N
        business_reviews = business_reviews.sort_values('date', ascending=False).head(limit)

        # Format reviews
        reviews = []
        for _, review in business_reviews.iterrows():
            reviews.append({
                'review_id': review.get('review_id', ''),
                'rating': float(review.get('rating', review.get('stars', 0))),
                'text': review.get('text', ''),
                'date': str(review.get('date', '')),
                'useful': int(review.get('useful', 0)),
                'user_review_count': int(review.get('user_review_count', 0))
            })

        result = {
            'business_id': business_id,
            'reviews': reviews,
            'count': len(reviews),
            'total_reviews': len(reviews_df[reviews_df['business_id'] == business_id]),
            'timestamp': pd.Timestamp.now().isoformat()
        }

        # Cache the result
        set_in_cache(cache_key, result)

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error fetching reviews: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/business/<business_id>/stats', methods=['GET'])
def get_business_stats(business_id):
    """Get detailed business statistics"""
    if reviews_df.empty:
        return jsonify({'error': 'No review data available'}), 404

    business_reviews = reviews_df[reviews_df['business_id'] == business_id]

    if business_reviews.empty:
        return jsonify({'error': 'No reviews found'}), 404

    stats = {
        'total_reviews': int(len(business_reviews)),
        'avg_rating': float(business_reviews['rating'].mean()),
        'rating_distribution': business_reviews['rating'].value_counts().to_dict(),
        'avg_text_length': float(business_reviews['text'].str.len().mean()),
        'useful_votes': int(business_reviews['useful'].sum()) if 'useful' in business_reviews.columns else 0,
        'funny_votes': int(business_reviews['funny'].sum()) if 'funny' in business_reviews.columns else 0,
        'cool_votes': int(business_reviews['cool'].sum()) if 'cool' in business_reviews.columns else 0
    }

    return jsonify(stats)


@app.route('/api/analyze/sentiment', methods=['POST'])
def analyze_sentiment():
    """Analyze text sentiment"""
    data = request.get_json()

    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    text = data['text']

    # Preprocess
    preprocessor = SentimentPreprocessor(method='nltk')
    processed_text = preprocessor.process(text)

    result = {
        'original_text': text,
        'processed_text': processed_text,
        'text_length': len(text),
        'word_count': len(text.split())
    }

    # Use model if available
    if 'sentiment' in models:
        try:
            prediction = models['sentiment'].predict([processed_text])[0]
            probabilities = models['sentiment'].predict_proba([processed_text])[0]

            sentiment_labels = ['Negative', 'Neutral', 'Positive']

            result['sentiment'] = sentiment_labels[prediction]
            result['confidence'] = float(probabilities[prediction])
            result['probabilities'] = {
                label: float(prob)
                for label, prob in zip(sentiment_labels, probabilities)
            }
        except Exception as e:
            logger.error(f"Error in sentiment prediction: {e}")
            result['error'] = str(e)
    else:
        result['message'] = 'Sentiment model not loaded. Please train the model first.'

    return jsonify(result)


@app.route('/api/analyze/statistics', methods=['POST'])
def analyze_statistics():
    """Analyze text statistics"""
    data = request.get_json()

    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    text = data['text']

    # Extract statistical features
    stats = extract_text_statistics(text)

    return jsonify(stats)


@app.route('/api/stats/overview', methods=['GET'])
def get_stats_overview():
    """Get overall statistics"""
    if reviews_df.empty:
        return jsonify({'error': 'No data available'}), 404

    stats = {
        'total_reviews': int(len(reviews_df)),
        'total_businesses': int(reviews_df['business_id'].nunique()),
        'total_users': int(reviews_df['user_id'].nunique()),
        'avg_rating': float(reviews_df['rating'].mean()) if 'rating' in reviews_df.columns else 0
    }

    # Rating distribution
    if 'rating' in reviews_df.columns:
        rating_dist = reviews_df['rating'].value_counts().sort_index()
        stats['rating_distribution'] = {float(k): int(v) for k, v in rating_dist.to_dict().items()}

    # Reviews by date
    if 'date' in reviews_df.columns:
        try:
            reviews_df['date'] = pd.to_datetime(reviews_df['date'])
            reviews_by_date = reviews_df.groupby(
                reviews_df['date'].dt.to_period('M')
            ).size()
            stats['reviews_by_month'] = {
                str(k): int(v) for k, v in reviews_by_date.to_dict().items()
            }
        except:
            pass

    return jsonify(stats)


@app.route('/api/search/reviews', methods=['GET'])
def search_reviews():
    """Search reviews"""
    query = request.args.get('q', '', type=str)
    limit = request.args.get('limit', 20, type=int)
    min_rating = request.args.get('min_rating', type=float)
    max_rating = request.args.get('max_rating', type=float)

    if not query or reviews_df.empty:
        return jsonify({'reviews': []})

    # Text search
    results = reviews_df[
        reviews_df['text'].str.contains(query, case=False, na=False)
    ]

    # Rating filters
    if min_rating is not None:
        results = results[results['rating'] >= min_rating]
    if max_rating is not None:
        results = results[results['rating'] <= max_rating]

    results = results.head(limit)

    return jsonify({
        'reviews': results.to_dict('records'),
        'count': len(results)
    })


@app.route('/api/insights/top-businesses', methods=['GET'])
def get_top_businesses():
    """Get top-rated businesses"""
    limit = request.args.get('limit', 10, type=int)

    if businesses_df.empty:
        return jsonify({'error': 'No business data available'}), 404

    # Filter businesses with enough reviews
    if 'review_count' in businesses_df.columns and 'stars' in businesses_df.columns:
        top = businesses_df[businesses_df['review_count'] >= 10].nlargest(limit, 'stars')
    else:
        top = businesses_df.head(limit)

    return jsonify({
        'businesses': top.to_dict('records'),
        'count': len(top)
    })


@app.route('/api/insights/word-cloud', methods=['GET'])
def get_word_cloud_data():
    """Get word frequency data for word cloud"""
    if reviews_df.empty:
        return jsonify({'error': 'No data available'}), 404

    from collections import Counter
    import re

    # Sample reviews
    sample_texts = reviews_df['text'].sample(min(1000, len(reviews_df))).tolist()

    # Combine and tokenize
    combined_text = ' '.join(sample_texts).lower()
    words = re.findall(r'\b[a-z]{3,}\b', combined_text)

    # Remove common stop words
    stop_words = {'the', 'and', 'was', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'her', 'his',
                  'has', 'had', 'this', 'that', 'with', 'from', 'they', 'were', 'been', 'have'}
    words = [w for w in words if w not in stop_words]

    # Get top 100 words
    word_freq = Counter(words).most_common(100)

    return jsonify({
        'words': [{'text': word, 'count': count} for word, count in word_freq]
    })


@app.route('/api/analyze/entities', methods=['POST'])
def analyze_entities():
    """Extract named entities from text"""
    try:
        data = request.get_json()
        text = data.get('text', '')

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        if not advanced_nlp_models['ner']:
            return jsonify({'error': 'NER model not loaded'}), 503

        entities = advanced_nlp_models['ner'].extract_business_entities(text)
        entity_summary = advanced_nlp_models['ner'].get_entity_summary(text)

        return jsonify({
            'entities': entities,
            'entity_summary': entity_summary,
            'total_entities': sum(entity_summary.values())
        })

    except Exception as e:
        logger.error(f"Error in entity extraction: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze/keywords', methods=['POST'])
def analyze_keywords():
    """Extract keywords from text"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        top_n = data.get('top_n', 10)

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        if not advanced_nlp_models['keyword_extractor']:
            return jsonify({'error': 'Keyword extractor not loaded'}), 503

        keywords = advanced_nlp_models['keyword_extractor'].extract_keywords(text, top_n=top_n)
        aspects = advanced_nlp_models['keyword_extractor'].extract_aspects(text)
        noun_phrases = advanced_nlp_models['keyword_extractor'].extract_noun_phrases(text)

        return jsonify({
            'keywords': [{'word': word, 'score': score} for word, score in keywords],
            'aspects': aspects,
            'noun_phrases': noun_phrases[:10]  # Top 10 noun phrases
        })

    except Exception as e:
        logger.error(f"Error in keyword extraction: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze/summarize', methods=['POST'])
def summarize_text():
    """Generate text summary"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        num_sentences = data.get('num_sentences', 3)

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        if not advanced_nlp_models['summarizer']:
            return jsonify({'error': 'Summarizer not loaded'}), 503

        summary = advanced_nlp_models['summarizer'].summarize(text, num_sentences=num_sentences)
        key_points = advanced_nlp_models['summarizer'].extract_key_points(text, max_points=5)

        return jsonify({
            'summary': summary,
            'key_points': key_points,
            'original_length': len(text.split()),
            'summary_length': len(summary.split())
        })

    except Exception as e:
        logger.error(f"Error in summarization: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze/aspects', methods=['POST'])
def analyze_aspects():
    """Analyze sentiment by aspects (food, service, etc.)"""
    try:
        data = request.get_json()
        text = data.get('text', '')

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        if not advanced_nlp_models['aspect_analyzer']:
            return jsonify({'error': 'Aspect analyzer not loaded'}), 503

        aspects = {
            'food': ['food', 'meal', 'dish', 'taste', 'flavor'],
            'service': ['service', 'waiter', 'staff', 'server'],
            'atmosphere': ['atmosphere', 'ambiance', 'decor', 'place'],
            'price': ['price', 'cost', 'expensive', 'value']
        }

        results = {}
        for aspect, keywords in aspects.items():
            analysis = advanced_nlp_models['aspect_analyzer'].analyze_aspect_sentiment(text, keywords)
            results[aspect] = analysis

        return jsonify({
            'aspect_sentiments': results,
            'text_length': len(text.split())
        })

    except Exception as e:
        logger.error(f"Error in aspect analysis: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze/comprehensive', methods=['POST'])
def analyze_comprehensive():
    """Perform comprehensive NLP analysis"""
    try:
        data = request.get_json()
        text = data.get('text', '')

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        # Perform comprehensive analysis
        results = analyze_review_comprehensive(text)

        return jsonify(results)

    except Exception as e:
        logger.error(f"Error in comprehensive analysis: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze/quality', methods=['POST'])
def analyze_quality():
    """Analyze review quality"""
    try:
        data = request.get_json()
        text = data.get('text', '')

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        if not professional_models['quality_analyzer']:
            return jsonify({'error': 'Quality analyzer not loaded'}), 503

        result = professional_models['quality_analyzer'].analyze_quality(text)

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in quality analysis: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze/spam', methods=['POST'])
def analyze_spam():
    """Detect spam in reviews"""
    try:
        data = request.get_json()
        text = data.get('text', '')

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        if not professional_models['spam_detector']:
            return jsonify({'error': 'Spam detector not loaded'}), 503

        result = professional_models['spam_detector'].detect_spam(text)

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in spam detection: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze/credibility', methods=['POST'])
def analyze_credibility():
    """Analyze review credibility"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        rating = data.get('rating', None)

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        if not professional_models['credibility_analyzer']:
            return jsonify({'error': 'Credibility analyzer not loaded'}), 503

        result = professional_models['credibility_analyzer'].analyze_credibility(text, rating)

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in credibility analysis: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze/professional', methods=['POST'])
def analyze_professional():
    """Comprehensive professional analysis"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        rating = data.get('rating', None)

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        result = analyze_review_comprehensive_professional(text, rating)

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in professional analysis: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/business/scorecard/<business_id>', methods=['GET'])
def get_business_scorecard(business_id):
    """Get business scorecard with key metrics"""
    try:
        if not business_analytics['insights_analyzer']:
            return jsonify({'error': 'Business analytics not available'}), 503

        result = business_analytics['insights_analyzer'].generate_business_scorecard(business_id)

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error generating scorecard: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/business/trends/<business_id>', methods=['GET'])
def get_business_trends(business_id):
    """Get sentiment trends for a business"""
    try:
        time_period = request.args.get('period', 'month')

        if not business_analytics['trend_analyzer']:
            return jsonify({'error': 'Trend analyzer not available'}), 503

        result = business_analytics['trend_analyzer'].analyze_temporal_trends(
            business_id=business_id,
            time_period=time_period
        )

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error analyzing trends: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/business/sentiment-shifts/<business_id>', methods=['GET'])
def get_sentiment_shifts(business_id):
    """Get significant sentiment shifts for a business"""
    try:
        threshold = request.args.get('threshold', 0.5, type=float)

        if not business_analytics['trend_analyzer']:
            return jsonify({'error': 'Trend analyzer not available'}), 503

        result = business_analytics['trend_analyzer'].identify_sentiment_shifts(
            business_id=business_id,
            threshold=threshold
        )

        return jsonify({'shifts': result})

    except Exception as e:
        logger.error(f"Error identifying shifts: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/business/compare', methods=['POST'])
def compare_businesses():
    """Compare multiple businesses"""
    try:
        data = request.get_json()
        business_ids = data.get('business_ids', [])
        metric = data.get('metric', 'rating')

        if not business_ids or len(business_ids) < 2:
            return jsonify({'error': 'Provide at least 2 business IDs'}), 400

        if not business_analytics['insights_analyzer']:
            return jsonify({'error': 'Business analytics not available'}), 503

        result = business_analytics['insights_analyzer'].compare_competitors(
            business_ids=business_ids,
            metric=metric
        )

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error comparing businesses: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/business/aspects/<business_id>', methods=['GET'])
def get_aspect_importance(business_id):
    """Get aspect importance analysis for a business"""
    try:
        if not business_analytics['insights_analyzer']:
            return jsonify({'error': 'Business analytics not available'}), 503

        result = business_analytics['insights_analyzer'].analyze_aspect_importance(business_id)

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error analyzing aspects: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# VERSION 1.1: CACHE SYSTEM AND ENHANCED ENDPOINTS
# ============================================================================

# In-memory cache with TTL
_cache_store = {}  # {cache_key: (data, timestamp)}
CACHE_TTL_SECONDS = 3600  # 1 hour
MAX_CACHE_ENTRIES = 1000  # Auto-evict if exceeded


def get_from_cache(key: str):
    """Get item from cache if not expired"""
    import time

    if key not in _cache_store:
        return None

    data, timestamp = _cache_store[key]
    age = time.time() - timestamp

    if age > CACHE_TTL_SECONDS:
        # Expired, remove from cache
        del _cache_store[key]
        return None

    return data


def set_in_cache(key: str, data):
    """Set item in cache with current timestamp"""
    import time

    # Auto-evict oldest entries if cache is too large
    if len(_cache_store) >= MAX_CACHE_ENTRIES:
        # Remove oldest 10% of entries
        sorted_keys = sorted(_cache_store.keys(),
                           key=lambda k: _cache_store[k][1])
        for k in sorted_keys[:MAX_CACHE_ENTRIES // 10]:
            del _cache_store[k]

    _cache_store[key] = (data, time.time())


def clear_cache():
    """Clear all cache entries"""
    global _cache_store
    _cache_store = {}


@app.route('/api/business/<business_id>/aspects-enhanced', methods=['GET'])
def get_enhanced_aspects(business_id):
    """
    Get enhanced aspect-based sentiment analysis (Version 1.1)
    Returns 8 aspects with detailed sentiment analysis
    """
    try:
        # Check cache first
        cache_key = f"aspects_enhanced_{business_id}"
        cached_data = get_from_cache(cache_key)

        if cached_data is not None:
            logger.info(f"Cache hit for aspects-enhanced: {business_id}")
            return jsonify(cached_data)

        logger.info(f"Cache miss for aspects-enhanced: {business_id}, computing...")

        # Import enhanced analyzer
        from src.models.advanced_nlp import analyze_business_aspects

        if reviews_df.empty:
            return jsonify({'error': 'No review data available'}), 404

        # Analyze aspects
        aspects = analyze_business_aspects(business_id, reviews_df)

        if not aspects:
            return jsonify({'error': 'No reviews found for this business'}), 404

        # Prepare radar chart data
        radar_data = {
            'labels': [],
            'scores': [],
            'ratings': []
        }

        for aspect_key, aspect_data in aspects.items():
            radar_data['labels'].append(aspect_data['label'])
            # Convert -1 to +1 score to 0-5 scale for radar chart
            normalized_score = (aspect_data['average_score'] + 1) / 2 * 5
            radar_data['scores'].append(round(normalized_score, 2))
            radar_data['ratings'].append(aspect_data['average_rating'])

        result = {
            'business_id': business_id,
            'aspects': aspects,
            'radar_chart_data': radar_data,
            'timestamp': pd.Timestamp.now().isoformat()
        }

        # Cache the result
        set_in_cache(cache_key, result)

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in enhanced aspects analysis: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/business/<business_id>/dishes', methods=['GET'])
def get_business_dishes(business_id):
    """
    Get popular dishes/menu items for a business (Version 1.1)
    Uses rule-based + frequency + AI fallback extraction
    """
    try:
        top_n = request.args.get('top_n', default=10, type=int)
        top_n = min(top_n, 20)  # Limit to 20 max

        # Check cache first
        cache_key = f"dishes_{business_id}_{top_n}"
        cached_data = get_from_cache(cache_key)

        if cached_data is not None:
            logger.info(f"Cache hit for dishes: {business_id}")
            return jsonify(cached_data)

        logger.info(f"Cache miss for dishes: {business_id}, extracting...")

        # Import dish extractor
        from src.models.advanced_nlp import extract_business_dishes

        if reviews_df.empty:
            return jsonify({'error': 'No review data available'}), 404

        # Extract dishes
        dishes = extract_business_dishes(business_id, reviews_df, top_n=top_n)

        result = {
            'business_id': business_id,
            'dishes': dishes,
            'count': len(dishes),
            'timestamp': pd.Timestamp.now().isoformat()
        }

        # Cache the result
        set_in_cache(cache_key, result)

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error extracting dishes: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/business/<business_id>/improvement-trends', methods=['GET'])
def get_improvement_trends(business_id):
    """
    Get improvement trend analysis with 0-100 score (Version 1.1)
    Shows how business is improving over time
    """
    try:
        period = request.args.get('period', default='quarter', type=str)

        # Validate period
        if period not in ['month', 'quarter', 'year']:
            return jsonify({'error': 'Invalid period. Use: month, quarter, or year'}), 400

        # Check cache first
        cache_key = f"trends_{business_id}_{period}"
        cached_data = get_from_cache(cache_key)

        if cached_data is not None:
            logger.info(f"Cache hit for improvement trends: {business_id}")
            return jsonify(cached_data)

        logger.info(f"Cache miss for improvement trends: {business_id}, analyzing...")

        if not business_analytics['trend_analyzer']:
            return jsonify({'error': 'Trend analyzer not available'}), 503

        # Analyze trends
        trends = business_analytics['trend_analyzer'].analyze_improvement_trends(
            business_id=business_id,
            period=period
        )

        if 'error' in trends:
            return jsonify(trends), 400

        result = {
            'business_id': business_id,
            'period': period,
            'trends': trends,
            'timestamp': pd.Timestamp.now().isoformat()
        }

        # Cache the result
        set_in_cache(cache_key, result)

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error analyzing improvement trends: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/cache/stats', methods=['GET'])
def get_cache_stats():
    """Get cache statistics for debugging"""
    import time

    total_entries = len(_cache_store)
    expired_count = 0
    valid_count = 0

    for key, (data, timestamp) in _cache_store.items():
        age = time.time() - timestamp
        if age > CACHE_TTL_SECONDS:
            expired_count += 1
        else:
            valid_count += 1

    return jsonify({
        'total_entries': total_entries,
        'valid_entries': valid_count,
        'expired_entries': expired_count,
        'max_entries': MAX_CACHE_ENTRIES,
        'ttl_seconds': CACHE_TTL_SECONDS
    })


@app.route('/api/cache/clear', methods=['POST'])
def clear_cache_endpoint():
    """Clear all cache entries"""
    try:
        entries_cleared = len(_cache_store)
        clear_cache()

        return jsonify({
            'success': True,
            'entries_cleared': entries_cleared,
            'message': 'Cache cleared successfully'
        })

    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/business/executive-summary', methods=['GET'])
def get_executive_summary():
    """Get executive summary with comprehensive business insights"""
    try:
        business_id = request.args.get('business_id', None)

        if reviews_df.empty:
            return jsonify({'error': 'No data available'}), 404

        result = generate_executive_summary(reviews_df, business_id)

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error generating executive summary: {e}")
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    """404 error handler"""
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """500 error handler"""
    return jsonify({'error': 'Internal server error'}), 500


# ==================== Restaurant Chatbot API ====================

# Session storage for conversation history (in production, use Redis or database)
conversation_sessions = {}

def chat_with_gpt(messages):
    """
    Main GPT chat function with function calling for database queries
    """
    if not openai_client:
        return {
            'type': 'error',
            'message': 'ChatGPT is not available. Please configure OpenAI API key.'
        }

    # Define available functions for GPT to call
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_restaurants",
                "description": "Search for restaurants in the database by cuisine type, location, name, or other criteria. Returns matching restaurants with their details.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cuisine": {
                            "type": "string",
                            "description": "Type of cuisine (e.g., Italian, Chinese, Mexican, American)"
                        },
                        "location": {
                            "type": "string",
                            "description": "City or state to search in"
                        },
                        "name": {
                            "type": "string",
                            "description": "Restaurant name or partial name to search for"
                        },
                        "min_rating": {
                            "type": "number",
                            "description": "Minimum star rating (1-5)"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results to return (default 10)"
                        }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_restaurant_details",
                "description": "Get detailed information about a specific restaurant including attributes, hours, and sample reviews",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "restaurant_name": {
                            "type": "string",
                            "description": "Name of the restaurant"
                        }
                    },
                    "required": ["restaurant_name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_restaurant_reviews",
                "description": "Get reviews for a specific restaurant to understand customer opinions and experiences",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "restaurant_name": {
                            "type": "string",
                            "description": "Name of the restaurant"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Number of reviews to retrieve (default 10)"
                        }
                    },
                    "required": ["restaurant_name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_yelp_live_data",
                "description": "Fetch live, real-time data from Yelp. Use this to search for businesses by name, type, or location. Returns up to 5 businesses with current ratings, reviews, service options, and contact info. For the top result, includes detailed recent reviews. Use this for location-based recommendations (e.g., 'restaurants near me', 'pizza in Austin') or specific business lookups.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "business_name": {
                            "type": "string",
                            "description": "Name or type of business to search for (e.g., 'Starbucks', 'Italian restaurant', 'Pizza'). Can be omitted if just searching by location."
                        },
                        "location": {
                            "type": "string",
                            "description": "City, state, or specific location to search in (e.g., 'Austin, TX', 'San Francisco, CA', 'Pittsburgh, PA'). Required if business_name is not provided."
                        }
                    },
                    "required": []
                }
            }
        }
    ]

    # System prompt for Yelp Insight Assistant
    system_message = {
        "role": "system",
        "content": """You are a Yelp Insight Assistant.

You answer restaurant and business questions using the most recent Yelp data from both our comprehensive database and live Yelp searches.

WHEN TO USE EACH DATA SOURCE:

**ALWAYS use fetch_yelp_live_data() for:**
- Specific business names (e.g., "Tell me about Starbucks", "How is Primanti Bros?")
- Location-based searches (e.g., "pizza in Austin", "restaurants in San Francisco", "coffee shops near Pittsburgh")
- User asks for current/live/recent information
- Any specific location mentioned in the query

**Use search_restaurants() ONLY for:**
- General database queries without specific locations
- Filtering by attributes in our database (WiFi, parking, etc.)
- Exploratory analysis of our dataset

CRITICAL RULES:
- Do NOT hallucinate unavailable data
- ALWAYS call fetch_yelp_live_data() when user mentions a location or business name
- If live data returns no results, be honest and suggest alternatives
- Summarize reviews highlighting: sentiment, pricing, service quality, food quality, atmosphere
- Be conversational and helpful, like ChatGPT
- Explain WHY restaurants are good matches based on actual data

FORMATTING RULES:
- **Business Names**: ALWAYS format as [Business Name](google_maps_link) for navigation
  - Use the "google_maps_link" field from the data (NOT the "link" field which goes to Yelp)
  - Example: [Starbucks](https://www.google.com/maps/search/?api=1&query=Starbucks+123+Main+St)
- **Rating Stars**: Display exact star count based on rating (max 5 stars)
  - 4.5 rating = ⭐⭐⭐⭐✨ (4 full stars + 1 half star)
  - 4.0 rating = ⭐⭐⭐⭐ (4 full stars)
  - 3.7 rating = ⭐⭐⭐✨ (3 full stars + 1 half star)
  - Use ⭐ for full stars, ✨ for half stars (0.3-0.7 range rounds to half)
  - Always show numerical rating too: ⭐⭐⭐⭐✨ (4.5/5)
- **Price Display**: Use the "price_estimate" field which shows concrete ranges
  - Display like: "$10-20 per person" (no emoji)
  - Fallback to price tier only if price_estimate unavailable: $$
- **Other emojis**: Use sparingly for context (📍 location, 🍽️ food type, 📶 WiFi, 🅿️ parking)

Data Sources:
- Local database: 150,346 businesses, 99,998 reviews (limited locations)
- Live Yelp data: Real-time business info from ANY location, current ratings, recent reviews (via SerpApi)

Format your responses in markdown for better readability."""
    }

    # Prepare messages with system prompt
    full_messages = [system_message] + messages

    try:
        # Call GPT with function calling
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=full_messages,
            tools=tools,
            tool_choice="auto",
            temperature=OPENAI_TEMPERATURE,
            max_tokens=OPENAI_MAX_TOKENS
        )

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        # If GPT wants to call functions
        if tool_calls:
            # Append assistant's tool call message
            full_messages.append(response_message)

            # Execute each function call
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                logger.info(f"GPT calling function: {function_name} with args: {function_args}")

                # Call the appropriate function
                if function_name == "search_restaurants":
                    function_response = search_restaurants_func(**function_args)
                elif function_name == "get_restaurant_details":
                    function_response = get_restaurant_details_func(**function_args)
                elif function_name == "get_restaurant_reviews":
                    function_response = get_restaurant_reviews_func(**function_args)
                elif function_name == "fetch_yelp_live_data":
                    function_response = fetch_yelp_live_data(**function_args)
                else:
                    function_response = {"error": "Unknown function"}

                # Append function response
                full_messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": json.dumps(function_response)
                })

            # Get final response from GPT with function results
            second_response = openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=full_messages,
                temperature=OPENAI_TEMPERATURE,
                max_tokens=OPENAI_MAX_TOKENS
            )

            final_message = second_response.choices[0].message.content
        else:
            # No function calls, use direct response
            final_message = response_message.content

        return {
            'type': 'chat',
            'message': final_message
        }

    except Exception as e:
        logger.error(f"GPT chat error: {str(e)}")
        return {
            'type': 'error',
            'message': f'Sorry, I encountered an error: {str(e)}'
        }

@app.route('/api/chatbot', methods=['POST'])
def chatbot():
    """
    Intelligent Restaurant Review Chatbot powered by GPT-4
    Uses function calling to query database and maintain conversation context
    """
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        session_id = data.get('session_id', 'default')

        if not user_message:
            return jsonify({'error': 'Message is required'}), 400

        # Initialize or get conversation history
        if session_id not in conversation_sessions:
            conversation_sessions[session_id] = []

        # Add user message to history
        conversation_sessions[session_id].append({
            "role": "user",
            "content": user_message
        })

        # Keep only last 10 messages to avoid token limits
        if len(conversation_sessions[session_id]) > 10:
            conversation_sessions[session_id] = conversation_sessions[session_id][-10:]

        # Call GPT with function calling capabilities
        response = chat_with_gpt(conversation_sessions[session_id])

        # Add assistant response to history
        conversation_sessions[session_id].append({
            "role": "assistant",
            "content": response['message']
        })

        return jsonify(response)

    except Exception as e:
        logger.error(f"Chatbot error: {str(e)}")
        return jsonify({
            'type': 'error',
            'message': 'Sorry, I encountered an error. Please try again.'
        }), 500


# ==================== Database Query Functions for GPT ====================

def fetch_yelp_live_data(business_name="", location=""):
    """
    Fetch live Yelp data using SerpApi
    Returns real-time business information and reviews from Yelp

    Args:
        business_name: Name or type of business to search for (e.g., "Pizza", "Starbucks")
        location: Location to search in (e.g., "Pittsburgh, PA", "Austin, TX")
    """
    try:
        from config import SERPAPI_KEY, SERPAPI_ENABLED
        import requests

        if not SERPAPI_ENABLED:
            logger.info("SerpApi not enabled, skipping live fetch")
            return {
                "success": False,
                "message": "Live Yelp data fetching is not enabled (no SerpApi key configured)"
            }

        # Build search parameters
        if not business_name and not location:
            return {
                "success": False,
                "message": "Please provide either a business name or location to search"
            }

        params = {
            "engine": "yelp",
            "api_key": SERPAPI_KEY
        }

        # Add search parameters
        if business_name:
            params["find_desc"] = business_name
        if location:
            params["find_loc"] = location
        else:
            params["find_loc"] = "United States"  # Default location

        search_query = f"{business_name} in {location}" if business_name and location else (business_name or location)
        logger.info(f"Fetching live Yelp data for: {search_query}")

        response = requests.get("https://serpapi.com/search.json", params=params, timeout=15)

        if response.status_code == 200:
            data = response.json()

            # Extract organic results (business listings)
            organic_results = data.get('organic_results', [])

            if not organic_results:
                return {
                    "success": False,
                    "message": f"No live Yelp results found for '{search_query}'"
                }

            # Process multiple results (up to 5 businesses)
            businesses_list = []
            for i, business in enumerate(organic_results[:5]):
                # Get place_id from place_ids array (SerpApi returns it as an array)
                place_ids = business.get('place_ids', [])
                place_id = place_ids[0] if place_ids else None

                if not place_id:
                    logger.warning(f"No place_id found for business: {business.get('title', 'Unknown')}")
                    continue

                # Extract business information
                name = business.get('title', '')
                address = business.get('address', 'N/A')

                # Create Google Maps navigation link
                import urllib.parse
                google_maps_query = f"{name} {address}".replace('\n', ' ')
                google_maps_link = f"https://www.google.com/maps/search/?api=1&query={urllib.parse.quote(google_maps_query)}"

                # Convert Yelp price tier to estimated price range
                price_tier = business.get('price', '')
                price_estimate = {
                    '$': '$5-10 per person',
                    '$$': '$10-20 per person',
                    '$$$': '$20-35 per person',
                    '$$$$': '$35+ per person'
                }.get(price_tier, price_tier or 'Price not available')

                business_info = {
                    "name": name,
                    "rating": business.get('rating', 0),
                    "review_count": business.get('reviews', 0),
                    "price": price_tier,  # Original Yelp tier ($, $$, etc.)
                    "price_estimate": price_estimate,  # Human-readable price range
                    "categories": [cat['title'] if isinstance(cat, dict) else cat for cat in business.get('categories', [])],
                    "neighborhoods": business.get('neighborhoods', []),
                    "address": address,
                    "phone": business.get('phone', 'N/A'),
                    "service_options": business.get('service_options', {}),
                    "link": business.get('link', ''),  # Yelp link
                    "google_maps_link": google_maps_link,  # Google Maps navigation link
                    "place_id": place_id
                }

                # For the first result, fetch detailed reviews
                if i == 0:
                    reviews_params = {
                        "engine": "yelp_reviews",
                        "place_id": place_id,
                        "api_key": SERPAPI_KEY
                    }

                    try:
                        reviews_response = requests.get("https://serpapi.com/search.json", params=reviews_params, timeout=15)
                        if reviews_response.status_code == 200:
                            reviews_data = reviews_response.json()
                            business_info['reviews'] = []

                            # Extract reviews
                            for review in reviews_data.get('reviews', [])[:10]:
                                business_info['reviews'].append({
                                    "rating": review.get('rating', 0),
                                    "text": review.get('comment', {}).get('text', '') or review.get('text', ''),
                                    "date": review.get('date', ''),
                                    "user": review.get('user', {}).get('name', 'Anonymous')
                                })

                            logger.info(f"Fetched {len(business_info['reviews'])} reviews for {business_info['name']}")
                    except Exception as e:
                        logger.warning(f"Failed to fetch reviews for {business_info['name']}: {e}")
                        business_info['reviews'] = []

                businesses_list.append(business_info)

            return {
                "success": True,
                "search_query": search_query,
                "total_results": len(businesses_list),
                "businesses": businesses_list
            }

        else:
            logger.error(f"SerpApi request failed: {response.status_code}")
            return {
                "success": False,
                "message": f"Failed to fetch live data (status: {response.status_code})"
            }

    except Exception as e:
        logger.error(f"fetch_yelp_live_data error: {str(e)}", exc_info=True)
        return {
            "success": False,
            "message": f"Error fetching live data: {str(e)}"
        }


def search_restaurants_func(cuisine=None, location=None, name=None, min_rating=None, limit=10):
    """
    Search restaurants in database based on criteria
    Called by GPT via function calling
    """
    try:
        if businesses_df.empty:
            return {"error": "No restaurant data available", "restaurants": []}

        filtered_df = businesses_df.copy()

        # Filter by name if provided
        if name:
            filtered_df = filtered_df[
                filtered_df['name'].str.contains(name, case=False, na=False, regex=False)
            ]

        # Filter by cuisine if provided
        if cuisine:
            filtered_df = filtered_df[
                filtered_df['categories'].str.contains(cuisine, case=False, na=False, regex=False)
            ]

        # Filter by location if provided
        if location:
            filtered_df = filtered_df[
                (filtered_df['city'].str.contains(location, case=False, na=False, regex=False)) |
                (filtered_df['state'].str.contains(location, case=False, na=False, regex=False))
            ]

        # Filter by minimum rating if provided
        if min_rating is not None and 'stars' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['stars'] >= min_rating]

        # Sort by rating and review count
        if 'stars' in filtered_df.columns and 'review_count' in filtered_df.columns:
            filtered_df = filtered_df.sort_values(
                ['stars', 'review_count'],
                ascending=[False, False]
            )

        # Limit results
        results = filtered_df.head(limit)

        # Format results
        restaurants = []
        for _, row in results.iterrows():
            restaurants.append({
                'name': row.get('name', 'Unknown'),
                'rating': float(row.get('stars', 0)),
                'review_count': int(row.get('review_count', 0)),
                'categories': row.get('categories', ''),
                'city': row.get('city', ''),
                'state': row.get('state', ''),
                'address': row.get('full_address', ''),
                'price_range': int(row.get('RestaurantsPriceRange2', 2))
            })

        return {
            "count": len(restaurants),
            "restaurants": restaurants
        }

    except Exception as e:
        logger.error(f"search_restaurants_func error: {str(e)}")
        return {"error": str(e), "restaurants": []}


def get_restaurant_details_func(restaurant_name):
    """
    Get detailed information about a specific restaurant
    Called by GPT via function calling
    """
    try:
        if businesses_df.empty:
            return {"error": "No restaurant data available"}

        # Search for restaurant
        restaurant = businesses_df[
            businesses_df['name'].str.contains(restaurant_name, case=False, na=False, regex=False)
        ]

        if restaurant.empty:
            return {"error": f"Restaurant '{restaurant_name}' not found"}

        # Get first match
        rest_info = restaurant.iloc[0]

        # Build details
        details = {
            'name': rest_info.get('name', ''),
            'rating': float(rest_info.get('stars', 0)),
            'review_count': int(rest_info.get('review_count', 0)),
            'categories': rest_info.get('categories', ''),
            'city': rest_info.get('city', ''),
            'state': rest_info.get('state', ''),
            'address': rest_info.get('full_address', ''),
            'price_range': int(rest_info.get('RestaurantsPriceRange2', 2)),
            'hours': str(rest_info.get('hours', 'Not specified')),
            'attributes': {}
        }

        # Parse attributes if available
        attributes_str = str(rest_info.get('attributes', ''))
        if attributes_str and attributes_str != 'nan':
            try:
                # Try to extract key attributes
                details['attributes']['raw'] = attributes_str
            except:
                pass

        # Add some common attributes
        for attr in ['WiFi', 'BikeParking', 'RestaurantsDelivery', 'RestaurantsTakeOut',
                     'OutdoorSeating', 'GoodForKids', 'RestaurantsReservations']:
            if attr in rest_info:
                details['attributes'][attr] = str(rest_info.get(attr, 'Unknown'))

        return details

    except Exception as e:
        logger.error(f"get_restaurant_details_func error: {str(e)}")
        return {"error": str(e)}


def get_restaurant_reviews_func(restaurant_name, limit=10):
    """
    Get reviews for a specific restaurant
    Called by GPT via function calling
    """
    try:
        if businesses_df.empty:
            return {"error": "No restaurant data available", "reviews": []}

        # Find restaurant
        restaurant = businesses_df[
            businesses_df['name'].str.contains(restaurant_name, case=False, na=False, regex=False)
        ]

        if restaurant.empty:
            return {"error": f"Restaurant '{restaurant_name}' not found", "reviews": []}

        rest_info = restaurant.iloc[0]
        business_id = rest_info.get('business_id')

        if not business_id or reviews_df.empty:
            return {"restaurant": rest_info.get('name', ''), "reviews": []}

        # Get reviews for this business
        biz_reviews = reviews_df[reviews_df['business_id'] == business_id].head(limit)

        reviews = []
        for _, review in biz_reviews.iterrows():
            reviews.append({
                'rating': int(review.get('rating', review.get('stars', 0))),
                'text': str(review.get('text', ''))[:500],  # Limit text length
                'date': str(review.get('date', ''))
            })

        return {
            "restaurant": rest_info.get('name', ''),
            "review_count": len(reviews),
            "reviews": reviews
        }

    except Exception as e:
        logger.error(f"get_restaurant_reviews_func error: {str(e)}")
        return {"error": str(e), "reviews": []}


def handle_recommendation(message):
    """Module 1: Restaurant Recommendation based on user preferences using GPT"""
    try:
        if businesses_df.empty:
            return {'type': 'error', 'message': 'No restaurant data available'}

        # Extract preferences from message
        budget = extract_budget(message)
        cuisine = extract_cuisine(message)
        location = extract_location(message)

        # Filter businesses
        filtered_df = businesses_df.copy()

        # Filter by cuisine if mentioned
        if cuisine:
            filtered_df = filtered_df[filtered_df['categories'].str.contains(cuisine, case=False, na=False)]

        # Filter by location if mentioned
        if location:
            filtered_df = filtered_df[
                (filtered_df['city'].str.contains(location, case=False, na=False)) |
                (filtered_df['state'].str.contains(location, case=False, na=False))
            ]

        # Filter by high ratings (>= 3.5 stars for more options)
        if 'stars' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['stars'] >= 3.5]

        # Sort by rating and review count
        if 'stars' in filtered_df.columns and 'review_count' in filtered_df.columns:
            filtered_df = filtered_df.sort_values(['stars', 'review_count'], ascending=[False, False])

        # Get top 10 restaurants for GPT to analyze
        candidates = filtered_df.head(10).to_dict('records')

        if not candidates:
            candidates = businesses_df.nlargest(10, 'stars').to_dict('records') if 'stars' in businesses_df.columns else []

        # Prepare restaurant data for GPT
        restaurant_data = []
        for rest in candidates[:10]:
            restaurant_data.append({
                'name': rest.get('name', 'Unknown'),
                'rating': float(rest.get('stars', 0)),
                'review_count': int(rest.get('review_count', 0)),
                'categories': rest.get('categories', ''),
                'city': rest.get('city', ''),
                'state': rest.get('state', ''),
                'price_range': int(rest.get('RestaurantsPriceRange2', 2))
            })

        # Use GPT to generate recommendations
        if openai_client:
            system_prompt = """You are a restaurant recommendation expert. Based on the user's preferences and the restaurant data provided, suggest the top 3-5 restaurants that best match their needs.

For each recommendation:
1. Explain WHY it matches their criteria (budget, cuisine, ratings, location)
2. Highlight key strengths based on ratings and review counts
3. Mention any relevant details (price range, popular dishes if applicable)
4. Keep your tone friendly and conversational

Format your response in markdown with restaurant names in bold and use emojis where appropriate (⭐📍🍽️💰)."""

            user_prompt = f"""User request: "{message}"

Available restaurants:
{json.dumps(restaurant_data, indent=2)}

Please recommend the top 3-5 restaurants that best match the user's needs. Consider:
- Budget: around ${budget} per person
- Cuisine preference: {cuisine if cuisine else 'any'}
- Location: {location if location else 'any'}
- High ratings and good reviews"""

            try:
                response = openai_client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=OPENAI_TEMPERATURE,
                    max_tokens=OPENAI_MAX_TOKENS
                )

                gpt_response = response.choices[0].message.content

                return {
                    'type': 'recommendation',
                    'message': gpt_response,
                    'restaurants': candidates[:5]
                }
            except Exception as e:
                logger.error(f"OpenAI API error: {str(e)}")
                # Fallback to simple response
                pass

        # Fallback if GPT not available
        response_text = f"Based on your preferences, here are my top recommendations:\n\n"
        for i, rest in enumerate(candidates[:3], 1):
            response_text += f"{i}. **{rest['name']}** ({rest.get('stars', 'N/A')}⭐)\n"
            response_text += f"   📍 {rest.get('city', '')}, {rest.get('state', '')}\n"
            response_text += f"   🍽️ {rest.get('categories', '')}\n"
            response_text += f"   💰 Estimated: ${budget} per person\n"
            response_text += f"   ✨ {rest.get('review_count', 0)} reviews\n\n"

        return {
            'type': 'recommendation',
            'message': response_text,
            'restaurants': candidates[:5]
        }

    except Exception as e:
        logger.error(f"Recommendation error: {str(e)}")
        return {'type': 'error', 'message': 'Error generating recommendations'}


def handle_factual_question(message):
    """Module 2: Extract factual information from restaurant data using GPT"""
    try:
        # Extract restaurant name from message
        restaurant_name = extract_restaurant_name(message)

        if not restaurant_name:
            return {
                'type': 'factual',
                'message': 'Please specify which restaurant you\'re asking about. For example: "Does Starbucks have WiFi?"'
            }

        # Find restaurant
        if businesses_df.empty:
            return {'type': 'error', 'message': 'No restaurant data available'}

        restaurant = businesses_df[businesses_df['name'].str.contains(restaurant_name, case=False, na=False)]

        if restaurant.empty:
            return {
                'type': 'factual',
                'message': f'I couldn\'t find "{restaurant_name}" in our database. Please try a different restaurant name.'
            }

        # Get first match
        rest_info = restaurant.iloc[0]

        # Prepare restaurant information
        restaurant_data = {
            'name': rest_info.get('name', ''),
            'rating': float(rest_info.get('stars', 0)),
            'review_count': int(rest_info.get('review_count', 0)),
            'categories': rest_info.get('categories', ''),
            'city': rest_info.get('city', ''),
            'state': rest_info.get('state', ''),
            'address': rest_info.get('full_address', ''),
            'attributes': str(rest_info.get('attributes', '')),
            'hours': str(rest_info.get('hours', 'Not specified'))
        }

        # Get sample reviews if available
        sample_reviews = []
        if not reviews_df.empty and 'business_id' in rest_info:
            biz_reviews = reviews_df[reviews_df['business_id'] == rest_info['business_id']].head(5)
            for _, review in biz_reviews.iterrows():
                sample_reviews.append({
                    'rating': review.get('rating', review.get('stars', 0)),
                    'text': review.get('text', '')[:200]  # First 200 chars
                })

        # Use GPT to answer the factual question
        if openai_client:
            system_prompt = """You are a helpful restaurant information assistant. Answer factual questions about restaurants based on the data provided (attributes, reviews, and general information).

Instructions:
1. Answer the specific question asked concisely and accurately
2. If the information is explicitly available in the attributes, provide it
3. If not available in attributes but can be inferred from reviews, mention that
4. If information is not available, be honest and say so
5. Always include key restaurant details (location, rating) at the end
6. Use emojis appropriately (📶 WiFi, 🅿️ Parking, 🚻 Restrooms, 🌳 Outdoor Seating, 🚗 Delivery, 🥡 Takeout, 🕐 Hours)
7. Format in markdown with bold restaurant name"""

            user_prompt = f"""User question: "{message}"

Restaurant Information:
{json.dumps(restaurant_data, indent=2)}

Sample Reviews:
{json.dumps(sample_reviews, indent=2) if sample_reviews else "No reviews available"}

Please answer the user's specific question about this restaurant."""

            try:
                response = openai_client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.3,  # Lower temperature for factual accuracy
                    max_tokens=OPENAI_MAX_TOKENS
                )

                gpt_response = response.choices[0].message.content

                return {
                    'type': 'factual',
                    'message': gpt_response,
                    'restaurant': rest_info.to_dict()
                }
            except Exception as e:
                logger.error(f"OpenAI API error: {str(e)}")
                # Fallback to simple response
                pass

        # Fallback response if GPT not available
        question_lower = message.lower()
        response_text = f"**{rest_info['name']}**\n\n"

        # Check attributes
        if 'wifi' in question_lower:
            wifi_info = check_attribute(rest_info, 'WiFi')
            response_text += f"📶 **WiFi:** {wifi_info}\n"

        if 'parking' in question_lower:
            parking_info = check_attribute(rest_info, 'Parking')
            response_text += f"🅿️ **Parking:** {parking_info}\n"

        if 'restroom' in question_lower or 'bathroom' in question_lower:
            response_text += f"🚻 **Restrooms:** Information not specified\n"

        if 'outdoor' in question_lower or 'patio' in question_lower:
            outdoor_info = check_attribute(rest_info, 'OutdoorSeating')
            response_text += f"🌳 **Outdoor Seating:** {outdoor_info}\n"

        # Add general info
        response_text += f"\n📍 **Location:** {rest_info.get('city', '')}, {rest_info.get('state', '')}\n"
        response_text += f"⭐ **Rating:** {rest_info.get('stars', 'N/A')} ({rest_info.get('review_count', 0)} reviews)\n"

        return {
            'type': 'factual',
            'message': response_text,
            'restaurant': rest_info.to_dict()
        }

    except Exception as e:
        logger.error(f"Factual question error: {str(e)}")
        return {'type': 'error', 'message': 'Error answering your question'}


def handle_review_summary(message):
    """Module 3: Summarize restaurant reviews using GPT and suggest similar places"""
    try:
        # Extract restaurant name
        restaurant_name = extract_restaurant_name(message)

        if not restaurant_name:
            return {
                'type': 'summary',
                'message': 'Please specify which restaurant you want to know about. For example: "How is Starbucks?"'
            }

        # Find restaurant
        if businesses_df.empty:
            return {'type': 'error', 'message': 'No restaurant data available'}

        restaurant = businesses_df[businesses_df['name'].str.contains(restaurant_name, case=False, na=False)]

        if restaurant.empty:
            return {
                'type': 'summary',
                'message': f'I couldn\'t find "{restaurant_name}" in our database.'
            }

        rest_info = restaurant.iloc[0]

        # Prepare restaurant data
        restaurant_data = {
            'name': rest_info.get('name', ''),
            'rating': float(rest_info.get('stars', 0)),
            'review_count': int(rest_info.get('review_count', 0)),
            'categories': rest_info.get('categories', ''),
            'city': rest_info.get('city', ''),
            'state': rest_info.get('state', ''),
            'price_range': int(rest_info.get('RestaurantsPriceRange2', 2))
        }

        # Get actual reviews if available
        review_samples = []
        rating_distribution = {'5': 0, '4': 0, '3': 0, '2': 0, '1': 0}

        if not reviews_df.empty and 'business_id' in rest_info:
            biz_reviews = reviews_df[reviews_df['business_id'] == rest_info['business_id']]

            if not biz_reviews.empty:
                # Get rating distribution
                for rating in [5, 4, 3, 2, 1]:
                    count = len(biz_reviews[biz_reviews['rating'] == rating]) if 'rating' in biz_reviews.columns else \
                            len(biz_reviews[biz_reviews['stars'] == rating])
                    rating_distribution[str(rating)] = count

                # Sample reviews from different ratings
                for rating in [5, 4, 3, 2, 1]:
                    rating_reviews = biz_reviews[biz_reviews.get('rating', biz_reviews.get('stars', 0)) == rating]
                    if not rating_reviews.empty:
                        sample = rating_reviews.head(2)
                        for _, review in sample.iterrows():
                            review_samples.append({
                                'rating': rating,
                                'text': review.get('text', '')[:300]  # First 300 chars
                            })

        # Find similar restaurants
        similar = find_similar_restaurants(rest_info)
        similar_data = []
        if similar:
            for sim in similar[:3]:
                similar_data.append({
                    'name': sim.get('name', ''),
                    'rating': float(sim.get('stars', 0)),
                    'categories': sim.get('categories', ''),
                    'city': sim.get('city', '')
                })

        # Use GPT to generate comprehensive summary
        if openai_client:
            system_prompt = """You are a restaurant review analysis expert. Analyze the restaurant data and reviews to provide a comprehensive, honest summary.

Your summary should include:
1. Overall food quality and taste (based on ratings and review content)
2. Key strengths and weaknesses (specific, based on actual reviews)
3. Average price per person (calculate from price_range: 1=$10, 2=$20, 3=$30, 4=$40+)
4. Notable patterns in reviews (what customers frequently mention)
5. Similar restaurant recommendations with brief explanations

Format guidelines:
- Use markdown with bold headings
- Use emojis appropriately (⭐📝🍽️✨⚠️💰🔍)
- Be honest - if reviews are mixed, say so
- Base insights on actual review content when available
- Keep it concise but informative (max 400 words)
- For similar restaurants, explain WHY they're similar (same cuisine, same area, similar quality)"""

            user_prompt = f"""Analyze this restaurant:

Restaurant Data:
{json.dumps(restaurant_data, indent=2)}

Rating Distribution:
{json.dumps(rating_distribution, indent=2)}

Sample Reviews:
{json.dumps(review_samples[:10], indent=2) if review_samples else "No reviews available - base analysis on rating data"}

Similar Restaurants:
{json.dumps(similar_data, indent=2) if similar_data else "No similar restaurants found"}

Provide a comprehensive review summary including strengths, weaknesses, pricing, patterns, and similar restaurant recommendations."""

            try:
                response = openai_client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=OPENAI_TEMPERATURE,
                    max_tokens=OPENAI_MAX_TOKENS
                )

                gpt_response = response.choices[0].message.content

                return {
                    'type': 'summary',
                    'message': gpt_response,
                    'restaurant': rest_info.to_dict(),
                    'similar_restaurants': similar[:3] if similar else []
                }
            except Exception as e:
                logger.error(f"OpenAI API error: {str(e)}")
                # Fallback to simple response
                pass

        # Fallback response if GPT not available
        response_text = f"**{rest_info['name']}** - Review Summary\n\n"
        stars = rest_info.get('stars', 0)
        rating_desc = get_rating_description(stars)
        response_text += f"⭐ **Overall Rating:** {stars}/5.0 - {rating_desc}\n"
        response_text += f"📝 **Total Reviews:** {rest_info.get('review_count', 0)}\n"
        response_text += f"🍽️ **Cuisine:** {rest_info.get('categories', '')}\n\n"

        # Estimated price
        price_range = rest_info.get('RestaurantsPriceRange2', 2)
        if pd.notna(price_range):
            price_per_person = int(price_range) * 15
            response_text += f"💰 **Estimated Price:** ${price_per_person} per person\n\n"

        # Similar restaurants
        if similar:
            response_text += "🔍 **Similar Restaurants:**\n"
            for i, sim in enumerate(similar[:3], 1):
                response_text += f"{i}. {sim['name']} ({sim.get('stars', 'N/A')}⭐) - {sim.get('city', '')}\n"

        return {
            'type': 'summary',
            'message': response_text,
            'restaurant': rest_info.to_dict(),
            'similar_restaurants': similar[:3] if similar else []
        }

    except Exception as e:
        logger.error(f"Summary error: {str(e)}")
        return {'type': 'error', 'message': 'Error generating summary'}


# Helper functions
def extract_budget(message):
    """Extract budget from message"""
    import re
    match = re.search(r'\$(\d+)', message)
    return int(match.group(1)) if match else 30  # Default $30


def extract_cuisine(message):
    """Extract cuisine type from message"""
    cuisines = ['italian', 'chinese', 'japanese', 'korean', 'mexican', 'thai', 'indian',
                'french', 'american', 'pizza', 'burger', 'sushi', 'coffee', 'cafe', 'bakery']
    message_lower = message.lower()
    for cuisine in cuisines:
        if cuisine in message_lower:
            return cuisine.capitalize()
    return None


def extract_location(message):
    """Extract location from message"""
    # Simple extraction - look for city names
    words = message.split()
    for word in words:
        if word[0].isupper() and len(word) > 3:
            return word
    return None


def extract_restaurant_name(message):
    """Extract restaurant name from message"""
    # Look for restaurant names after keywords
    import re
    patterns = [
        r'(?:about|is|does|at)\s+([A-Z][a-zA-Z\s\']+?)(?:\s+have|\s+offer|\?|$)',
        r'([A-Z][a-zA-Z\s\']+?)\s+(?:have|offer|provide)',
        r'(?:How is|Tell me about)\s+([A-Z][a-zA-Z\s\']+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, message)
        if match:
            return match.group(1).strip()

    # Fallback: look for capitalized words
    words = message.split()
    capitalized = [w for w in words if w and w[0].isupper() and w not in ['I', 'Does', 'Is', 'How', 'Tell', 'What', 'A', 'The']]
    if capitalized:
        return ' '.join(capitalized[:2])  # Take first 1-2 capitalized words

    return None


def check_attribute(restaurant_info, attr_name):
    """Check if restaurant has a specific attribute"""
    attributes = str(restaurant_info.get('attributes', ''))
    if attr_name in attributes:
        if 'True' in attributes:
            return "✅ Yes, available"
        elif 'False' in attributes:
            return "❌ Not available"
    return "ℹ️ Information not specified"


def get_rating_description(stars):
    """Get description based on star rating"""
    if stars >= 4.5:
        return "Excellent"
    elif stars >= 4.0:
        return "Very Good"
    elif stars >= 3.5:
        return "Good"
    elif stars >= 3.0:
        return "Average"
    else:
        return "Below Average"


def find_similar_restaurants(restaurant_info):
    """Find similar restaurants based on category and location"""
    try:
        if businesses_df.empty:
            return []

        # Filter by same category
        categories = restaurant_info.get('categories', '')
        same_category = businesses_df[
            (businesses_df['categories'].str.contains(categories.split(',')[0] if categories else '', case=False, na=False, regex=False)) &
            (businesses_df['business_id'] != restaurant_info['business_id'])
        ]

        # Filter by location
        same_location = same_category[
            (same_category['city'] == restaurant_info['city']) |
            (same_category['state'] == restaurant_info['state'])
        ]

        # Sort by rating
        if 'stars' in same_location.columns:
            same_location = same_location.nlargest(5, 'stars')

        return same_location.head(5).to_dict('records')

    except Exception as e:
        logger.error(f"Similar restaurants error: {str(e)}")
        return []


if __name__ == '__main__':
    logger.info(f"Starting Flask app on {FLASK_HOST}:{FLASK_PORT}")
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=DEBUG)
