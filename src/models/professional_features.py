"""
Professional NLP Features for Review Analysis
Includes: Review Quality Scoring, Spam Detection, Credibility Analysis, Language Quality
"""

import re
import spacy
from typing import Dict, List, Tuple, Any
from collections import Counter
import numpy as np
from textblob import TextBlob
import logging

logger = logging.getLogger(__name__)


class ReviewQualityAnalyzer:
    """
    Analyzes the quality and credibility of reviews
    Evaluates based on multiple factors: length, detail, language quality, objectivity
    """

    def __init__(self, model_name='en_core_web_sm'):
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            logger.warning(f"spaCy model {model_name} not found. Installing...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", model_name])
            self.nlp = spacy.load(model_name)

    def analyze_quality(self, text: str) -> Dict[str, Any]:
        """
        Comprehensive quality analysis of a review
        Returns quality score (0-100) and detailed metrics
        """
        doc = self.nlp(text)

        # Calculate various quality metrics
        length_score = self._calculate_length_score(text)
        detail_score = self._calculate_detail_score(doc)
        grammar_score = self._calculate_grammar_score(text)
        objectivity_score = self._calculate_objectivity_score(text)
        informative_score = self._calculate_informative_score(doc)

        # Weighted overall score
        overall_score = (
            length_score * 0.15 +
            detail_score * 0.25 +
            grammar_score * 0.20 +
            objectivity_score * 0.20 +
            informative_score * 0.20
        )

        # Quality classification
        if overall_score >= 75:
            quality_label = "Excellent"
            quality_class = "high"
        elif overall_score >= 60:
            quality_label = "Good"
            quality_class = "medium-high"
        elif overall_score >= 45:
            quality_label = "Fair"
            quality_class = "medium"
        elif overall_score >= 30:
            quality_label = "Poor"
            quality_class = "low"
        else:
            quality_label = "Very Poor"
            quality_class = "very-low"

        return {
            "overall_score": round(overall_score, 2),
            "quality_label": quality_label,
            "quality_class": quality_class,
            "metrics": {
                "length_score": round(length_score, 2),
                "detail_score": round(detail_score, 2),
                "grammar_score": round(grammar_score, 2),
                "objectivity_score": round(objectivity_score, 2),
                "informative_score": round(informative_score, 2)
            },
            "insights": self._generate_insights(
                length_score, detail_score, grammar_score,
                objectivity_score, informative_score
            )
        }

    def _calculate_length_score(self, text: str) -> float:
        """Score based on review length (optimal: 50-300 words)"""
        word_count = len(text.split())

        if word_count < 10:
            return 20.0
        elif word_count < 30:
            return 50.0
        elif word_count < 50:
            return 70.0
        elif word_count <= 300:
            return 100.0
        elif word_count <= 500:
            return 85.0
        else:
            return 70.0  # Very long reviews can be less focused

    def _calculate_detail_score(self, doc) -> float:
        """Score based on specific details (entities, numbers, adjectives)"""
        entities = len([ent for ent in doc.ents])
        numbers = len([token for token in doc if token.like_num])
        adjectives = len([token for token in doc if token.pos_ == 'ADJ'])
        nouns = len([token for token in doc if token.pos_ in ['NOUN', 'PROPN']])

        # Normalize based on text length
        text_length = len(doc)
        if text_length == 0:
            return 0.0

        entity_ratio = (entities / text_length) * 100
        number_ratio = (numbers / text_length) * 100
        adjective_ratio = (adjectives / text_length) * 100
        noun_ratio = (nouns / text_length) * 100

        # Detail score based on presence of specific information
        score = min(100, (
            entity_ratio * 30 +
            number_ratio * 20 +
            adjective_ratio * 25 +
            noun_ratio * 25
        ))

        return score

    def _calculate_grammar_score(self, text: str) -> float:
        """Score based on grammar and spelling"""
        # Check for basic grammar patterns
        score = 100.0

        # Penalize all caps
        if text.isupper():
            score -= 30

        # Penalize excessive punctuation
        punct_ratio = len(re.findall(r'[!?]{2,}', text)) / max(1, len(text.split()))
        score -= min(20, punct_ratio * 100)

        # Check for sentence structure
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) == 0:
            score -= 40
        else:
            # Check if sentences start with capital letters
            properly_capitalized = sum(1 for s in sentences if s and s[0].isupper())
            capitalization_ratio = properly_capitalized / len(sentences)
            score = score * capitalization_ratio

        return max(0, min(100, score))

    def _calculate_objectivity_score(self, text: str) -> float:
        """Score based on objectivity (balance of facts vs emotions)"""
        blob = TextBlob(text)

        # Check subjectivity (0 = objective, 1 = subjective)
        subjectivity = blob.sentiment.subjectivity

        # Convert to objectivity score (higher is better for balanced reviews)
        # Optimal reviews have moderate subjectivity (0.3-0.6)
        if 0.3 <= subjectivity <= 0.6:
            objectivity_score = 100.0
        elif subjectivity < 0.3:
            # Too objective, might lack personal experience
            objectivity_score = 70.0
        else:
            # Too subjective
            objectivity_score = max(30, 100 - (subjectivity - 0.6) * 150)

        return objectivity_score

    def _calculate_informative_score(self, doc) -> float:
        """Score based on informational content"""
        # Check for informative patterns
        score = 50.0  # Base score

        # Bonus for specific aspects mentioned
        aspect_keywords = {
            'food': ['food', 'meal', 'dish', 'cuisine', 'taste', 'flavor'],
            'service': ['service', 'staff', 'waiter', 'waitress', 'server'],
            'atmosphere': ['atmosphere', 'ambiance', 'decor', 'environment'],
            'price': ['price', 'cost', 'expensive', 'cheap', 'value'],
            'location': ['location', 'parking', 'access', 'area']
        }

        text_lower = doc.text.lower()
        aspects_mentioned = 0

        for aspect, keywords in aspect_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                aspects_mentioned += 1

        score += aspects_mentioned * 10

        # Bonus for specific details (times, dates, prices)
        if re.search(r'\$\d+|\d+\s*(?:am|pm|dollars|minutes|hours)', text_lower):
            score += 15

        return min(100, score)

    def _generate_insights(self, length_score, detail_score, grammar_score,
                          objectivity_score, informative_score) -> List[str]:
        """Generate actionable insights based on scores"""
        insights = []

        if length_score < 50:
            insights.append("Review is too short. More details would improve quality.")
        elif length_score > 90:
            insights.append("Good length with adequate detail.")

        if detail_score < 50:
            insights.append("Lacks specific details. Include more concrete examples.")
        elif detail_score > 70:
            insights.append("Contains good specific details and examples.")

        if grammar_score < 60:
            insights.append("Grammar and formatting could be improved.")
        elif grammar_score > 80:
            insights.append("Well-written with good grammar and structure.")

        if objectivity_score < 50:
            insights.append("Review is highly subjective. Balance with objective facts.")
        elif objectivity_score > 75:
            insights.append("Good balance of personal experience and objective information.")

        if informative_score < 50:
            insights.append("Limited informational value. Mention specific aspects.")
        elif informative_score > 75:
            insights.append("Informative with specific aspects covered.")

        return insights


class SpamDetector:
    """
    Detects spam and fake reviews using pattern analysis
    """

    def __init__(self):
        self.spam_patterns = [
            r'(?:click|visit|check\s+out).*(?:link|website|url)',
            r'(?:buy|order|purchase)\s+(?:now|today|here)',
            r'(?:free|discount|offer|deal)\s+(?:now|today)',
            r'(?:call|contact|email)\s+(?:me|us)',
            r'\b(?:viagra|cialis|pharmacy|pills)\b',
            r'(?:\d{3}[-.]?\d{3}[-.]?\d{4})',  # Phone numbers
            r'(?:www\.|http|\.com|\.net)',  # URLs
        ]

        self.suspicious_patterns = [
            r'^.{1,15}$',  # Very short
            r'([A-Z]{2,}\s+){3,}',  # Excessive caps
            r'([!?]{3,})',  # Excessive punctuation
            r'(\w)\1{4,}',  # Repeated characters
        ]

    def detect_spam(self, text: str) -> Dict[str, Any]:
        """
        Analyze text for spam indicators
        Returns spam probability and flags
        """
        spam_score = 0
        flags = []

        # Check for spam patterns
        for pattern in self.spam_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                spam_score += 25
                flags.append(f"Contains promotional content")

        # Check suspicious patterns
        for pattern in self.suspicious_patterns:
            if re.search(pattern, text):
                spam_score += 10
                flags.append(f"Contains suspicious formatting")

        # Check for generic/template language
        generic_phrases = [
            'this is the best', 'highly recommend', 'amazing experience',
            'you should definitely', 'best ever', 'worst ever'
        ]

        generic_count = sum(1 for phrase in generic_phrases
                          if phrase in text.lower())
        if generic_count >= 3:
            spam_score += 15
            flags.append("Contains generic template phrases")

        # Check review uniqueness (low vocabulary diversity)
        words = re.findall(r'\w+', text.lower())
        if len(words) > 10:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.5:
                spam_score += 20
                flags.append("Low vocabulary diversity")

        spam_score = min(100, spam_score)

        # Classification
        if spam_score >= 70:
            classification = "Likely Spam"
            risk_level = "high"
        elif spam_score >= 40:
            classification = "Suspicious"
            risk_level = "medium"
        elif spam_score >= 20:
            classification = "Potentially Suspicious"
            risk_level = "low"
        else:
            classification = "Legitimate"
            risk_level = "none"

        return {
            "spam_score": spam_score,
            "classification": classification,
            "risk_level": risk_level,
            "flags": list(set(flags)),  # Remove duplicates
            "is_spam": spam_score >= 40
        }


class CredibilityAnalyzer:
    """
    Analyzes review credibility based on multiple factors
    """

    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def analyze_credibility(self, text: str, rating: int = None) -> Dict[str, Any]:
        """
        Analyze review credibility
        """
        doc = self.nlp(text)

        credibility_score = 100.0
        flags = []

        # Check sentiment-rating alignment
        if rating:
            sentiment_polarity = TextBlob(text).sentiment.polarity
            expected_sentiment = (rating - 3) / 2  # Normalize rating to -1 to 1

            sentiment_diff = abs(sentiment_polarity - expected_sentiment)
            if sentiment_diff > 0.6:
                credibility_score -= 30
                flags.append("Sentiment doesn't match rating")

        # Check for extreme language
        extreme_words = ['worst', 'best', 'terrible', 'amazing', 'awful', 'perfect']
        extreme_count = sum(1 for word in extreme_words if word in text.lower())

        if extreme_count > 3:
            credibility_score -= 15
            flags.append("Excessive extreme language")

        # Check for first-person experience
        first_person = len([token for token in doc if token.text.lower() in ['i', 'we', 'my', 'our']])
        if first_person == 0 and len(text.split()) > 20:
            credibility_score -= 10
            flags.append("Lacks personal experience indicators")

        # Check for specific details
        entities = len(list(doc.ents))
        numbers = len([token for token in doc if token.like_num])

        if entities + numbers < 2 and len(text.split()) > 30:
            credibility_score -= 15
            flags.append("Lacks specific details")

        credibility_score = max(0, credibility_score)

        # Classification
        if credibility_score >= 80:
            classification = "Highly Credible"
            level = "high"
        elif credibility_score >= 60:
            classification = "Credible"
            level = "medium-high"
        elif credibility_score >= 40:
            classification = "Moderately Credible"
            level = "medium"
        else:
            classification = "Low Credibility"
            level = "low"

        return {
            "credibility_score": round(credibility_score, 2),
            "classification": classification,
            "credibility_level": level,
            "flags": flags,
            "is_credible": credibility_score >= 60
        }


def analyze_review_comprehensive_professional(text: str, rating: int = None) -> Dict[str, Any]:
    """
    Comprehensive professional analysis of a review
    Combines quality, spam detection, and credibility analysis
    """
    quality_analyzer = ReviewQualityAnalyzer()
    spam_detector = SpamDetector()
    credibility_analyzer = CredibilityAnalyzer()

    quality_result = quality_analyzer.analyze_quality(text)
    spam_result = spam_detector.detect_spam(text)
    credibility_result = credibility_analyzer.analyze_credibility(text, rating)

    # Calculate overall trustworthiness score
    trustworthiness = (
        quality_result['overall_score'] * 0.4 +
        (100 - spam_result['spam_score']) * 0.3 +
        credibility_result['credibility_score'] * 0.3
    )

    return {
        "quality": quality_result,
        "spam_detection": spam_result,
        "credibility": credibility_result,
        "overall_trustworthiness": round(trustworthiness, 2),
        "recommendation": _generate_recommendation(trustworthiness, spam_result['is_spam'])
    }


def _generate_recommendation(trustworthiness: float, is_spam: bool) -> str:
    """Generate recommendation based on analysis"""
    if is_spam:
        return "This review appears to be spam and should be flagged for review."
    elif trustworthiness >= 75:
        return "This is a high-quality, trustworthy review."
    elif trustworthiness >= 60:
        return "This review appears legitimate with good quality."
    elif trustworthiness >= 40:
        return "This review has some concerns regarding quality or credibility."
    else:
        return "This review has significant quality or credibility issues."
