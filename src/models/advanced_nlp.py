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
