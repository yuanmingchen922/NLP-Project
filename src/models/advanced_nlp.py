"""
Advanced NLP Features Module
Includes: Named Entity Recognition, Keyword Extraction, Text Summarization
"""

import spacy
import logging
from collections import Counter
from typing import List, Dict, Tuple
import re

logger = logging.getLogger(__name__)


class NamedEntityRecognizer:
    """
    Named Entity Recognition using spaCy
    Extracts persons, organizations, locations, etc. from text
    """

    def __init__(self, model_name='en_core_web_sm'):
        """
        Initialize NER with spaCy model

        Args:
            model_name: spaCy model name
        """
        try:
            self.nlp = spacy.load(model_name)
            logger.info(f"Loaded spaCy model: {model_name}")
        except OSError:
            logger.error(f"spaCy model '{model_name}' not found. Please install it:")
            logger.error(f"  python -m spacy download {model_name}")
            raise

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text

        Args:
            text: Input text

        Returns:
            Dict mapping entity types to lists of entities
        """
        doc = self.nlp(text)

        entities = {}
        for ent in doc.ents:
            entity_type = ent.label_
            entity_text = ent.text

            if entity_type not in entities:
                entities[entity_type] = []

            entities[entity_type].append(entity_text)

        return entities

    def extract_business_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract business-relevant entities (food items, services, etc.)

        Args:
            text: Review text

        Returns:
            Dict with business-relevant entities
        """
        entities = self.extract_entities(text)

        # Focus on relevant entity types for business reviews
        relevant_types = ['PERSON', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'TIME', 'DATE']
        filtered_entities = {
            k: v for k, v in entities.items() if k in relevant_types
        }

        return filtered_entities

    def get_entity_summary(self, text: str) -> Dict[str, int]:
        """
        Get count of entities by type

        Args:
            text: Input text

        Returns:
            Dict mapping entity types to counts
        """
        entities = self.extract_entities(text)
        return {ent_type: len(ents) for ent_type, ents in entities.items()}


class KeywordExtractor:
    """
    Keyword extraction using TF-IDF and noun phrase extraction
    """

    def __init__(self, model_name='en_core_web_sm'):
        """
        Initialize keyword extractor

        Args:
            model_name: spaCy model name
        """
        try:
            self.nlp = spacy.load(model_name)
            logger.info(f"Loaded spaCy model for keyword extraction: {model_name}")
        except OSError:
            logger.error(f"spaCy model '{model_name}' not found")
            raise

    def extract_keywords(self, text: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Extract keywords from text using noun phrases and frequency

        Args:
            text: Input text
            top_n: Number of top keywords to return

        Returns:
            List of (keyword, score) tuples
        """
        doc = self.nlp(text.lower())

        # Extract noun chunks
        noun_chunks = [chunk.text for chunk in doc.noun_chunks]

        # Extract individual nouns and adjectives
        keywords = [token.lemma_ for token in doc
                    if token.pos_ in ['NOUN', 'PROPN', 'ADJ']
                    and not token.is_stop
                    and len(token.text) > 2]

        # Combine noun chunks and keywords
        all_keywords = noun_chunks + keywords

        # Count frequencies
        keyword_freq = Counter(all_keywords)

        # Return top N keywords with frequencies
        return keyword_freq.most_common(top_n)

    def extract_noun_phrases(self, text: str) -> List[str]:
        """
        Extract noun phrases from text

        Args:
            text: Input text

        Returns:
            List of noun phrases
        """
        doc = self.nlp(text)
        noun_phrases = [chunk.text for chunk in doc.noun_chunks]
        return noun_phrases

    def extract_aspects(self, text: str) -> Dict[str, List[str]]:
        """
        Extract business aspects (food, service, atmosphere, etc.)

        Args:
            text: Review text

        Returns:
            Dict mapping aspects to related words
        """
        doc = self.nlp(text.lower())

        # Define aspect keywords
        aspect_keywords = {
            'food': ['food', 'meal', 'dish', 'taste', 'flavor', 'cuisine', 'menu', 'plate', 'order'],
            'service': ['service', 'waiter', 'waitress', 'staff', 'server', 'employee', 'manager'],
            'atmosphere': ['atmosphere', 'ambiance', 'decor', 'setting', 'environment', 'vibe', 'place'],
            'price': ['price', 'cost', 'expensive', 'cheap', 'value', 'worth', 'money', 'budget'],
            'cleanliness': ['clean', 'dirty', 'hygiene', 'sanitary', 'neat', 'tidy']
        }

        # Find mentions of each aspect
        aspects = {}
        for aspect, keywords in aspect_keywords.items():
            mentions = []
            for token in doc:
                if token.lemma_ in keywords or token.text in keywords:
                    # Get surrounding context (3 words before and after)
                    start_idx = max(0, token.i - 3)
                    end_idx = min(len(doc), token.i + 4)
                    context = doc[start_idx:end_idx].text
                    mentions.append(context)

            if mentions:
                aspects[aspect] = mentions

        return aspects


class TextSummarizer:
    """
    Text summarization using extractive approach
    """

    def __init__(self, model_name='en_core_web_sm'):
        """
        Initialize text summarizer

        Args:
            model_name: spaCy model name
        """
        try:
            self.nlp = spacy.load(model_name)
            logger.info(f"Loaded spaCy model for summarization: {model_name}")
        except OSError:
            logger.error(f"spaCy model '{model_name}' not found")
            raise

    def summarize(self, text: str, num_sentences: int = 3) -> str:
        """
        Create extractive summary by selecting most important sentences

        Args:
            text: Input text
            num_sentences: Number of sentences to include in summary

        Returns:
            Summary text
        """
        doc = self.nlp(text)

        # Split into sentences
        sentences = [sent.text.strip() for sent in doc.sents]

        if len(sentences) <= num_sentences:
            return text

        # Score sentences based on word frequency
        word_freq = Counter()
        for token in doc:
            if not token.is_stop and not token.is_punct and len(token.text) > 2:
                word_freq[token.lemma_] += 1

        # Normalize frequencies
        max_freq = max(word_freq.values()) if word_freq else 1
        for word in word_freq:
            word_freq[word] /= max_freq

        # Score each sentence
        sentence_scores = {}
        for sent in doc.sents:
            score = 0
            word_count = 0
            for token in sent:
                if token.lemma_ in word_freq:
                    score += word_freq[token.lemma_]
                    word_count += 1

            if word_count > 0:
                sentence_scores[sent.text.strip()] = score / word_count

        # Select top sentences (maintaining original order)
        top_sentences = sorted(sentence_scores.items(),
                               key=lambda x: x[1],
                               reverse=True)[:num_sentences]

        # Sort by original position to maintain coherence
        summary_sentences = []
        for sent in sentences:
            if any(sent == top_sent[0] for top_sent in top_sentences):
                summary_sentences.append(sent)
                if len(summary_sentences) == num_sentences:
                    break

        return ' '.join(summary_sentences)

    def extract_key_points(self, text: str, max_points: int = 5) -> List[str]:
        """
        Extract key points from text

        Args:
            text: Input text
            max_points: Maximum number of key points

        Returns:
            List of key points
        """
        doc = self.nlp(text)

        sentences = [sent.text.strip() for sent in doc.sents]

        if len(sentences) <= max_points:
            return sentences

        # Score sentences by importance
        sentence_scores = []
        for sent in doc.sents:
            # Factors for importance:
            # 1. Length (not too short, not too long)
            # 2. Presence of numbers
            # 3. Presence of strong adjectives
            # 4. Position (first and last sentences often important)

            score = 0
            sent_text = sent.text.strip()

            # Length score
            word_count = len([t for t in sent if not t.is_punct])
            if 5 <= word_count <= 20:
                score += 1

            # Numbers indicate specific facts
            if any(token.like_num for token in sent):
                score += 0.5

            # Strong adjectives/adverbs
            strong_words = ['excellent', 'terrible', 'amazing', 'awful', 'best', 'worst',
                            'great', 'horrible', 'fantastic', 'disappointing']
            if any(token.lemma_ in strong_words for token in sent):
                score += 1

            # Position bonus
            sent_idx = sentences.index(sent_text)
            if sent_idx == 0 or sent_idx == len(sentences) - 1:
                score += 0.5

            sentence_scores.append((sent_text, score))

        # Sort by score and return top points
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        key_points = [sent for sent, _ in sentence_scores[:max_points]]

        return key_points


class SentimentAspectAnalyzer:
    """
    Aspect-based sentiment analysis
    Analyzes sentiment for specific aspects (food, service, etc.)
    """

    def __init__(self, model_name='en_core_web_sm'):
        """
        Initialize aspect-based sentiment analyzer

        Args:
            model_name: spaCy model name
        """
        try:
            self.nlp = spacy.load(model_name)
            logger.info(f"Loaded spaCy model for aspect analysis: {model_name}")
        except OSError:
            logger.error(f"spaCy model '{model_name}' not found")
            raise

        # Sentiment lexicon
        self.positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'delicious', 'tasty', 'perfect', 'love', 'best', 'awesome', 'nice',
            'friendly', 'helpful', 'fresh', 'clean', 'beautiful', 'recommend'
        }

        self.negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'disgusting', 'worst',
            'poor', 'disappointing', 'bland', 'cold', 'slow', 'rude', 'dirty',
            'expensive', 'overpriced', 'mediocre', 'unacceptable', 'never'
        }

    def analyze_aspect_sentiment(self, text: str, aspect_keywords: List[str]) -> Dict[str, any]:
        """
        Analyze sentiment for a specific aspect

        Args:
            text: Review text
            aspect_keywords: Keywords related to the aspect

        Returns:
            Dict with sentiment analysis results
        """
        doc = self.nlp(text.lower())

        # Find aspect mentions and surrounding context
        aspect_sentiments = []
        for token in doc:
            if token.lemma_ in aspect_keywords or token.text in aspect_keywords:
                # Get context window
                start_idx = max(0, token.i - 5)
                end_idx = min(len(doc), token.i + 6)
                context = doc[start_idx:end_idx]

                # Count positive and negative words in context
                pos_count = sum(1 for t in context if t.lemma_ in self.positive_words)
                neg_count = sum(1 for t in context if t.lemma_ in self.negative_words)

                # Determine sentiment
                if pos_count > neg_count:
                    sentiment = 'positive'
                    score = pos_count / (pos_count + neg_count) if (pos_count + neg_count) > 0 else 0
                elif neg_count > pos_count:
                    sentiment = 'negative'
                    score = neg_count / (pos_count + neg_count) if (pos_count + neg_count) > 0 else 0
                else:
                    sentiment = 'neutral'
                    score = 0.5

                aspect_sentiments.append({
                    'aspect': token.text,
                    'sentiment': sentiment,
                    'score': score,
                    'context': context.text
                })

        # Aggregate results
        if aspect_sentiments:
            avg_score = sum(s['score'] for s in aspect_sentiments) / len(aspect_sentiments)
            sentiments = [s['sentiment'] for s in aspect_sentiments]
            overall_sentiment = max(set(sentiments), key=sentiments.count)

            return {
                'overall_sentiment': overall_sentiment,
                'confidence': avg_score,
                'mentions': len(aspect_sentiments),
                'details': aspect_sentiments
            }
        else:
            return {
                'overall_sentiment': 'not_mentioned',
                'confidence': 0,
                'mentions': 0,
                'details': []
            }


# ============================================================================
# ENHANCED ASPECT-BASED SENTIMENT ANALYSIS (Version 1.1)
# ============================================================================

# Enhanced aspect keywords with 8 aspects (5 core + 3 extended)
ENHANCED_ASPECT_KEYWORDS = {
    'food_quality': {
        'keywords': [
            'food', 'dish', 'meal', 'taste', 'flavor', 'delicious', 'tasty',
            'fresh', 'quality', 'cook', 'cooked', 'prepare', 'prepared',
            'ingredient', 'spice', 'season', 'seasoned', 'recipe', 'authentic',
            'homemade', 'gourmet', 'cuisine', 'menu', 'entree', 'appetizer'
        ],
        'emoji': '🍽️',
        'label': 'Food Quality'
    },
    'service': {
        'keywords': [
            'service', 'server', 'waiter', 'waitress', 'staff', 'employee',
            'friendly', 'helpful', 'attentive', 'professional', 'polite',
            'courteous', 'welcoming', 'greeting', 'prompt', 'quick',
            'manager', 'host', 'hostess', 'bartender', 'team'
        ],
        'emoji': '👥',
        'label': 'Service'
    },
    'atmosphere': {
        'keywords': [
            'atmosphere', 'ambiance', 'ambience', 'environment', 'vibe',
            'decor', 'decoration', 'interior', 'design', 'setting',
            'music', 'lighting', 'noise', 'loud', 'quiet', 'cozy',
            'romantic', 'casual', 'formal', 'modern', 'rustic', 'elegant'
        ],
        'emoji': '🏪',
        'label': 'Atmosphere'
    },
    'price_value': {
        'keywords': [
            'price', 'cost', 'expensive', 'cheap', 'affordable', 'value',
            'worth', 'money', 'overpriced', 'reasonable', 'fair',
            'budget', 'deal', 'special', 'discount', 'promotion',
            'dollar', 'pay', 'paid', 'charge', 'bill', 'tab'
        ],
        'emoji': '💰',
        'label': 'Price/Value'
    },
    'cleanliness': {
        'keywords': [
            'clean', 'cleanliness', 'hygiene', 'sanitary', 'tidy',
            'spotless', 'dirty', 'messy', 'filth', 'stain',
            'bathroom', 'restroom', 'table', 'floor', 'kitchen',
            'organized', 'maintained', 'well-kept'
        ],
        'emoji': '✨',
        'label': 'Cleanliness'
    },
    'location_parking': {
        'keywords': [
            'location', 'parking', 'park', 'lot', 'garage', 'street',
            'access', 'accessible', 'convenient', 'nearby', 'close',
            'distance', 'walk', 'drive', 'find', 'spot', 'space',
            'valet', 'validated', 'free parking', 'paid parking'
        ],
        'emoji': '📍',
        'label': 'Location/Parking'
    },
    'portion_size': {
        'keywords': [
            'portion', 'size', 'serving', 'amount', 'quantity',
            'large', 'small', 'huge', 'tiny', 'generous', 'skimpy',
            'big', 'little', 'massive', 'enough', 'filling',
            'leftovers', 'share', 'sharing', 'hungry', 'full'
        ],
        'emoji': '🍱',
        'label': 'Portion Size'
    },
    'wait_time': {
        'keywords': [
            'wait', 'waiting', 'time', 'long', 'short', 'quick', 'fast',
            'slow', 'delay', 'delayed', 'hour', 'minute', 'immediately',
            'reservation', 'line', 'queue', 'seated', 'table ready',
            'patient', 'impatient', 'prompt', 'speedy'
        ],
        'emoji': '⏱️',
        'label': 'Wait Time'
    }
}

# Negation words for handling phrases like "not good"
NEGATION_WORDS = {
    'not', 'no', 'never', 'nothing', 'nobody', 'none', 'neither', 'nowhere',
    'hardly', 'barely', 'scarcely', "n't", 'without', 'lack', 'lacking'
}

# Intensity modifiers
INTENSIFIERS = {
    'very': 1.5, 'extremely': 2.0, 'incredibly': 2.0, 'absolutely': 1.8,
    'really': 1.3, 'so': 1.3, 'quite': 1.2, 'super': 1.5, 'totally': 1.5,
    'completely': 1.8, 'utterly': 2.0, 'highly': 1.5
}

DAMPENERS = {
    'somewhat': 0.7, 'slightly': 0.5, 'a bit': 0.6, 'kind of': 0.7,
    'sort of': 0.7, 'fairly': 0.8, 'rather': 0.8, 'pretty': 0.9
}


class EnhancedSentimentAspectAnalyzer:
    """
    Enhanced aspect-based sentiment analyzer with:
    - 8 aspects (5 core + 3 extended)
    - Negation handling ("not good" → negative)
    - Intensity modifiers ("very delicious" > "delicious")
    - Expanded context window (8 words)
    - Confidence scoring
    """

    def __init__(self, model_name: str = 'en_core_web_sm'):
        """Initialize with spaCy model"""
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            logger.warning(f"Downloading spaCy model: {model_name}")
            import subprocess
            subprocess.run(['python', '-m', 'spacy', 'download', model_name])
            self.nlp = spacy.load(model_name)

        # Enhanced sentiment lexicons
        self.positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'delicious', 'tasty', 'perfect', 'love', 'loved', 'best', 'awesome',
            'nice', 'friendly', 'helpful', 'fresh', 'clean', 'beautiful',
            'recommend', 'outstanding', 'superb', 'exceptional', 'fabulous',
            'brilliant', 'magnificent', 'marvelous', 'terrific', 'splendid',
            'divine', 'heavenly', 'yummy', 'scrumptious', 'phenomenal',
            'impressive', 'enjoyable', 'pleasant', 'satisfying', 'quality'
        }

        self.negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'disgusting', 'worst',
            'poor', 'disappointing', 'disappointed', 'bland', 'cold', 'slow',
            'rude', 'dirty', 'expensive', 'overpriced', 'mediocre',
            'unacceptable', 'never', 'nasty', 'gross', 'lousy', 'subpar',
            'inferior', 'dreadful', 'atrocious', 'appalling', 'horrendous',
            'pathetic', 'useless', 'worthless', 'shocking', 'offensive',
            'unpleasant', 'unsatisfactory', 'inadequate', 'flawed'
        }

    def _detect_negation(self, context_tokens, target_idx):
        """
        Detect if there's a negation word before the target word

        Args:
            context_tokens: List of tokens in context
            target_idx: Index of target sentiment word

        Returns:
            bool: True if negated
        """
        # Check 3 words before target
        start = max(0, target_idx - 3)
        for i in range(start, target_idx):
            if context_tokens[i].text.lower() in NEGATION_WORDS or \
               context_tokens[i].lemma_ in NEGATION_WORDS:
                return True
        return False

    def _detect_intensity(self, context_tokens, target_idx):
        """
        Detect intensity modifiers before the target word

        Args:
            context_tokens: List of tokens in context
            target_idx: Index of target sentiment word

        Returns:
            float: Intensity multiplier (default 1.0)
        """
        # Check 2 words before target
        start = max(0, target_idx - 2)
        for i in range(start, target_idx):
            word = context_tokens[i].text.lower()
            if word in INTENSIFIERS:
                return INTENSIFIERS[word]
            if word in DAMPENERS:
                return DAMPENERS[word]
        return 1.0

    def analyze_aspect_sentiment(self, text: str, aspect: str,
                                 aspect_keywords: List[str]) -> Dict[str, any]:
        """
        Analyze sentiment for a specific aspect with enhanced features

        Args:
            text: Review text
            aspect: Aspect name (e.g., 'food_quality')
            aspect_keywords: Keywords related to the aspect

        Returns:
            Dict with enhanced sentiment analysis results
        """
        doc = self.nlp(text.lower())

        # Find aspect mentions and analyze surrounding context
        aspect_sentiments = []
        aspect_mentioned = False

        for token in doc:
            if token.lemma_ in aspect_keywords or token.text in aspect_keywords:
                aspect_mentioned = True

                # Expanded context window (8 words on each side)
                start_idx = max(0, token.i - 8)
                end_idx = min(len(doc), token.i + 9)
                context = doc[start_idx:end_idx]

                # Analyze sentiment words in context with modifiers
                pos_score = 0
                neg_score = 0

                for i, t in enumerate(context):
                    base_weight = 1.0
                    is_negated = self._detect_negation(list(context), i)
                    intensity = self._detect_intensity(list(context), i)

                    if t.lemma_ in self.positive_words or t.text in self.positive_words:
                        score = base_weight * intensity
                        if is_negated:
                            neg_score += score  # "not good" becomes negative
                        else:
                            pos_score += score

                    elif t.lemma_ in self.negative_words or t.text in self.negative_words:
                        score = base_weight * intensity
                        if is_negated:
                            pos_score += score  # "not bad" becomes positive
                        else:
                            neg_score += score

                # Determine sentiment with confidence
                total_score = pos_score + neg_score
                if total_score > 0:
                    if pos_score > neg_score:
                        sentiment = 'positive'
                        confidence = pos_score / total_score
                        raw_score = (pos_score - neg_score) / total_score
                    elif neg_score > pos_score:
                        sentiment = 'negative'
                        confidence = neg_score / total_score
                        raw_score = -(neg_score - pos_score) / total_score
                    else:
                        sentiment = 'neutral'
                        confidence = 0.5
                        raw_score = 0
                else:
                    sentiment = 'neutral'
                    confidence = 0.3  # Low confidence neutral
                    raw_score = 0

                aspect_sentiments.append({
                    'aspect_word': token.text,
                    'sentiment': sentiment,
                    'confidence': confidence,
                    'score': raw_score,  # -1 to +1 scale
                    'context': context.text
                })

        # Aggregate results
        if aspect_sentiments and aspect_mentioned:
            avg_score = sum(s['score'] for s in aspect_sentiments) / len(aspect_sentiments)
            avg_confidence = sum(s['confidence'] for s in aspect_sentiments) / len(aspect_sentiments)

            # Determine overall sentiment
            if avg_score > 0.1:
                overall_sentiment = 'positive'
            elif avg_score < -0.1:
                overall_sentiment = 'negative'
            else:
                overall_sentiment = 'neutral'

            # Count sentiment distribution
            sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
            for s in aspect_sentiments:
                sentiment_counts[s['sentiment']] += 1

            return {
                'aspect': aspect,
                'overall_sentiment': overall_sentiment,
                'average_score': avg_score,  # -1 to +1
                'confidence': avg_confidence,
                'mention_count': len(aspect_sentiments),
                'sentiment_distribution': sentiment_counts,
                'details': aspect_sentiments[:5]  # Limit details to top 5
            }
        else:
            return {
                'aspect': aspect,
                'overall_sentiment': 'not_mentioned',
                'average_score': 0,
                'confidence': 0,
                'mention_count': 0,
                'sentiment_distribution': {'positive': 0, 'negative': 0, 'neutral': 0},
                'details': []
            }

    def analyze_all_aspects(self, text: str) -> Dict[str, Dict]:
        """
        Analyze all 8 aspects for a single review

        Args:
            text: Review text

        Returns:
            Dict mapping aspect names to sentiment results
        """
        results = {}
        for aspect, config in ENHANCED_ASPECT_KEYWORDS.items():
            results[aspect] = self.analyze_aspect_sentiment(
                text, aspect, config['keywords']
            )
            # Add emoji and label
            results[aspect]['emoji'] = config['emoji']
            results[aspect]['label'] = config['label']

        return results


def analyze_business_aspects(business_id: str, reviews_df) -> Dict[str, Dict]:
    """
    Analyze all aspects for a business across all its reviews

    Args:
        business_id: Business ID
        reviews_df: DataFrame containing reviews

    Returns:
        Dict with aggregated aspect analysis for the business
    """
    # Get all reviews for this business
    business_reviews = reviews_df[reviews_df['business_id'] == business_id]

    if len(business_reviews) == 0:
        return {}

    # Initialize analyzer
    analyzer = EnhancedSentimentAspectAnalyzer()

    # Aggregate results for each aspect
    aspect_results = {aspect: {
        'scores': [],
        'sentiments': [],
        'ratings': [],  # From review stars
        'mention_count': 0
    } for aspect in ENHANCED_ASPECT_KEYWORDS.keys()}

    # Analyze each review
    for _, review in business_reviews.iterrows():
        text = review.get('text', '')
        rating = review.get('rating', review.get('stars', 0))

        if not text:
            continue

        review_aspects = analyzer.analyze_all_aspects(text)

        for aspect, result in review_aspects.items():
            if result['overall_sentiment'] != 'not_mentioned':
                aspect_results[aspect]['scores'].append(result['average_score'])
                aspect_results[aspect]['sentiments'].append(result['overall_sentiment'])
                aspect_results[aspect]['ratings'].append(rating)
                aspect_results[aspect]['mention_count'] += result['mention_count']

    # Calculate aggregated metrics
    aggregated = {}
    total_reviews = len(business_reviews)

    for aspect, data in aspect_results.items():
        if data['mention_count'] > 0:
            avg_score = sum(data['scores']) / len(data['scores'])
            avg_rating = sum(data['ratings']) / len(data['ratings'])

            # Sentiment distribution
            sentiment_dist = {
                'positive': data['sentiments'].count('positive'),
                'negative': data['sentiments'].count('negative'),
                'neutral': data['sentiments'].count('neutral')
            }

            # Coverage percentage
            coverage = (len(data['scores']) / total_reviews) * 100

            aggregated[aspect] = {
                'average_score': round(avg_score, 3),  # -1 to +1
                'average_rating': round(avg_rating, 2),  # 1 to 5 stars
                'sentiment_distribution': sentiment_dist,
                'mention_count': data['mention_count'],
                'coverage_percentage': round(coverage, 1),
                'emoji': ENHANCED_ASPECT_KEYWORDS[aspect]['emoji'],
                'label': ENHANCED_ASPECT_KEYWORDS[aspect]['label']
            }
        else:
            # Aspect not mentioned in any review
            aggregated[aspect] = {
                'average_score': 0,
                'average_rating': 0,
                'sentiment_distribution': {'positive': 0, 'negative': 0, 'neutral': 0},
                'mention_count': 0,
                'coverage_percentage': 0,
                'emoji': ENHANCED_ASPECT_KEYWORDS[aspect]['emoji'],
                'label': ENHANCED_ASPECT_KEYWORDS[aspect]['label']
            }

    return aggregated


# ============================================================================
# DISH/MENU ITEM EXTRACTION (Version 1.1)
# ============================================================================

class DishExtractor:
    """
    Extract popular dishes and menu items from reviews using multiple strategies:
    - Strategy A: Rule-based extraction (regex + spaCy noun phrases)
    - Strategy B: Frequency ranking with sentiment
    - Strategy C: AI fallback using OpenAI GPT-4o-mini
    """

    def __init__(self):
        """Initialize dish extractor"""
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            logger.warning("Downloading spaCy model: en_core_web_sm")
            import subprocess
            subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'])
            self.nlp = spacy.load('en_core_web_sm')

        # Common food categories and dish patterns
        self.food_categories = {
            'burger', 'sandwich', 'pizza', 'pasta', 'salad', 'soup', 'steak',
            'chicken', 'fish', 'seafood', 'sushi', 'taco', 'burrito', 'noodle',
            'rice', 'curry', 'ramen', 'pho', 'dumpling', 'wing', 'fries',
            'dessert', 'cake', 'pie', 'ice cream', 'coffee', 'tea', 'smoothie',
            'appetizer', 'entree', 'main course', 'side dish'
        }

        # Dish indicators (words that often precede dish names)
        self.dish_indicators = {
            'ordered', 'tried', 'got', 'had', 'ate', 'loved', 'enjoyed',
            'recommend', 'try', 'get', 'order', 'delicious', 'amazing',
            'favorite', 'best', 'worst'
        }

    def _extract_rule_based(self, reviews_df) -> Dict[str, List]:
        """
        Strategy A: Rule-based extraction using regex and spaCy

        Returns:
            Dict mapping dish names to list of mentions with sentiment
        """
        dish_mentions = {}

        for _, review in reviews_df.iterrows():
            text = review.get('text', '')
            rating = review.get('rating', review.get('stars', 0))

            if not text:
                continue

            # Process with spaCy
            doc = self.nlp(text)

            # Extract noun phrases that might be dishes
            for chunk in doc.noun_chunks:
                # Filter to likely food items
                chunk_lower = chunk.text.lower()

                # Check if contains food category words
                is_food = any(cat in chunk_lower for cat in self.food_categories)

                # Check if near dish indicators
                if is_food or self._is_near_indicator(chunk, doc):
                    dish_name = chunk.text.title()

                    # Clean up the dish name
                    if len(dish_name.split()) <= 5 and len(dish_name) <= 50:
                        if dish_name not in dish_mentions:
                            dish_mentions[dish_name] = []

                        dish_mentions[dish_name].append({
                            'rating': rating,
                            'context': text[max(0, chunk.start_char-50):min(len(text), chunk.end_char+50)]
                        })

        return dish_mentions

    def _is_near_indicator(self, chunk, doc):
        """Check if noun chunk is near a dish indicator word"""
        chunk_idx = chunk.start

        # Check 3 words before
        for i in range(max(0, chunk_idx-3), chunk_idx):
            if doc[i].lemma_ in self.dish_indicators:
                return True

        return False

    def _rank_by_frequency(self, dish_mentions: Dict[str, List],
                          top_n: int = 10) -> List[Dict]:
        """
        Strategy B: Rank dishes by frequency and average rating

        Args:
            dish_mentions: Dict mapping dish names to mentions
            top_n: Number of top dishes to return

        Returns:
            List of top dishes with statistics
        """
        ranked_dishes = []

        for dish_name, mentions in dish_mentions.items():
            if len(mentions) < 2:  # Filter dishes mentioned only once
                continue

            ratings = [m['rating'] for m in mentions]
            avg_rating = sum(ratings) / len(ratings)
            mention_count = len(mentions)

            # Calculate relevance score (frequency × average rating)
            relevance_score = mention_count * avg_rating

            ranked_dishes.append({
                'name': dish_name,
                'mention_count': mention_count,
                'average_rating': round(avg_rating, 2),
                'relevance_score': round(relevance_score, 2),
                'sentiment': 'positive' if avg_rating >= 4 else 'negative' if avg_rating < 3 else 'neutral'
            })

        # Sort by relevance score
        ranked_dishes.sort(key=lambda x: x['relevance_score'], reverse=True)

        return ranked_dishes[:top_n]

    def _extract_with_openai(self, reviews_df, top_n: int = 10) -> List[Dict]:
        """
        Strategy C: Use OpenAI to extract dishes when rule-based fails

        Args:
            reviews_df: DataFrame with reviews
            top_n: Number of dishes to extract

        Returns:
            List of extracted dishes
        """
        from config import OPENAI_API_KEY, OPENAI_MODEL
        from openai import OpenAI

        if not OPENAI_API_KEY:
            logger.warning("OpenAI API key not configured, skipping AI extraction")
            return []

        try:
            client = OpenAI(api_key=OPENAI_API_KEY)

            # Get sample of reviews (top rated and low rated for balance)
            top_reviews = reviews_df.nlargest(5, 'rating' if 'rating' in reviews_df.columns else 'stars')
            low_reviews = reviews_df.nsmallest(3, 'rating' if 'rating' in reviews_df.columns else 'stars')
            sample_reviews = pd.concat([top_reviews, low_reviews])

            # Combine review texts
            review_texts = []
            for _, review in sample_reviews.iterrows():
                text = review.get('text', '')
                rating = review.get('rating', review.get('stars', 0))
                review_texts.append(f"[{rating}★] {text[:300]}")

            combined_text = '\n\n'.join(review_texts[:8])  # Limit to 8 reviews

            # Create prompt
            prompt = f"""Extract the top {top_n} most mentioned dishes or menu items from these restaurant reviews.
For each dish, provide:
1. Name of the dish
2. How often it's mentioned (estimate)
3. Overall sentiment (positive/negative/neutral)

Reviews:
{combined_text}

Return ONLY a JSON array of objects with keys: name, mention_count, sentiment
Example: [{{"name": "Margherita Pizza", "mention_count": 3, "sentiment": "positive"}}]
"""

            # Call OpenAI
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that extracts dish names from restaurant reviews. Always return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )

            # Parse response
            import json
            content = response.choices[0].message.content.strip()

            # Try to extract JSON from response
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()

            dishes = json.loads(content)

            # Add average_rating and relevance_score (estimated)
            for dish in dishes:
                dish['average_rating'] = 4.0 if dish['sentiment'] == 'positive' else 2.5 if dish['sentiment'] == 'negative' else 3.0
                dish['relevance_score'] = dish['mention_count'] * dish['average_rating']

            return dishes[:top_n]

        except Exception as e:
            logger.error(f"OpenAI extraction failed: {e}")
            return []

    def extract_dishes_combined(self, reviews_df, top_n: int = 10) -> List[Dict]:
        """
        Combined extraction strategy: Try A+B, fall back to C if needed

        Args:
            reviews_df: DataFrame with reviews for a business
            top_n: Number of top dishes to return

        Returns:
            List of top dishes with statistics
        """
        # Strategy A: Rule-based extraction
        dish_mentions = self._extract_rule_based(reviews_df)

        # Strategy B: Frequency ranking
        ranked_dishes = self._rank_by_frequency(dish_mentions, top_n=top_n)

        # Check if we have enough dishes
        if len(ranked_dishes) >= 3:
            logger.info(f"Rule-based extraction successful: {len(ranked_dishes)} dishes found")
            return ranked_dishes

        # Strategy C: Fallback to OpenAI if too few dishes found
        logger.info(f"Only {len(ranked_dishes)} dishes found via rules, trying OpenAI fallback")
        ai_dishes = self._extract_with_openai(reviews_df, top_n=top_n)

        if ai_dishes:
            logger.info(f"OpenAI extraction successful: {len(ai_dishes)} dishes found")
            return ai_dishes

        # Return whatever we got
        logger.info(f"Returning {len(ranked_dishes)} dishes from rule-based extraction")
        return ranked_dishes


def extract_business_dishes(business_id: str, reviews_df, top_n: int = 10) -> List[Dict]:
    """
    Extract top dishes for a specific business

    Args:
        business_id: Business ID
        reviews_df: DataFrame containing all reviews
        top_n: Number of top dishes to return

    Returns:
        List of top dishes with statistics
    """
    # Filter reviews for this business
    business_reviews = reviews_df[reviews_df['business_id'] == business_id]

    if len(business_reviews) == 0:
        return []

    # Extract dishes
    extractor = DishExtractor()
    dishes = extractor.extract_dishes_combined(business_reviews, top_n=top_n)

    return dishes


# Convenience function to analyze a review comprehensively
def analyze_review_comprehensive(text: str) -> Dict[str, any]:
    """
    Perform comprehensive NLP analysis on a review

    Args:
        text: Review text

    Returns:
        Dict with all analysis results
    """
    try:
        # Initialize analyzers
        ner = NamedEntityRecognizer()
        keyword_extractor = KeywordExtractor()
        summarizer = TextSummarizer()
        aspect_analyzer = SentimentAspectAnalyzer()

        # Perform analyses
        results = {
            'entities': ner.extract_business_entities(text),
            'entity_summary': ner.get_entity_summary(text),
            'keywords': keyword_extractor.extract_keywords(text, top_n=10),
            'aspects': keyword_extractor.extract_aspects(text),
            'summary': summarizer.summarize(text, num_sentences=2),
            'key_points': summarizer.extract_key_points(text, max_points=3),
        }

        # Aspect-based sentiment
        aspect_sentiments = {}
        for aspect, keywords in {
            'food': ['food', 'meal', 'dish', 'taste'],
            'service': ['service', 'waiter', 'staff'],
            'atmosphere': ['atmosphere', 'ambiance', 'place']
        }.items():
            aspect_sentiments[aspect] = aspect_analyzer.analyze_aspect_sentiment(text, keywords)

        results['aspect_sentiments'] = aspect_sentiments

        return results

    except Exception as e:
        logger.error(f"Error in comprehensive analysis: {e}")
        return {'error': str(e)}
