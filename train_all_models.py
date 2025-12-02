"""
Train All NLP Models
Complete training script for sentiment analysis, rating prediction, and topic modeling
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config import PROCESSED_DATA_DIR, MODEL_DIR
from src.data_processing.feature_extraction import TfidfFeatureExtractor
from src.models.sentiment_analysis import TraditionalSentimentAnalyzer
from src.models.rating_prediction import TraditionalRatingPredictor
from src.models.topic_modeling import TopicModeler
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_sentiment_models():
    """Train sentiment analysis models"""
    logger.info("="*60)
    logger.info("TRAINING SENTIMENT ANALYSIS MODELS")
    logger.info("="*60)

    # Load data
    logger.info("Loading review data...")
    reviews_path = PROCESSED_DATA_DIR / 'reviews.csv'

    if not reviews_path.exists():
        logger.error(f"Data file not found: {reviews_path}")
        logger.error("Please run: python src/data_processing/data_loader.py first")
        return False

    df = pd.read_csv(reviews_path)
    logger.info(f"Loaded {len(df)} reviews")

    # Use a subset for faster training
    sample_size = min(50000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)
    logger.info(f"Using {sample_size} samples for training")

    texts = df_sample['text'].tolist()

    # Handle both 'rating' and 'stars' column names
    if 'rating' in df_sample.columns:
        ratings = df_sample['rating'].tolist()
    elif 'stars' in df_sample.columns:
        ratings = df_sample['stars'].tolist()
    else:
        logger.error("No rating or stars column found!")
        return False

    # Train Logistic Regression model
    logger.info("\n--- Training Logistic Regression Model ---")
    tfidf_lr = TfidfFeatureExtractor(max_features=5000, ngram_range=(1, 2))
    analyzer_lr = TraditionalSentimentAnalyzer(model_type='logistic', feature_extractor=tfidf_lr)

    results_lr = analyzer_lr.train(texts, ratings)
    logger.info(f"Accuracy: {results_lr['accuracy']:.4f}")
    logger.info(f"F1 Score: {results_lr['f1_score']:.4f}")

    # Save model
    model_path = MODEL_DIR / 'sentiment_lr.pkl'
    vectorizer_path = MODEL_DIR / 'tfidf_vectorizer.pkl'

    joblib.dump(analyzer_lr.model, model_path)
    joblib.dump(tfidf_lr.vectorizer, vectorizer_path)
    logger.info(f"Saved model to {model_path}")
    logger.info(f"Saved vectorizer to {vectorizer_path}")

    # Train SVM model
    logger.info("\n--- Training SVM Model ---")
    tfidf_svm = TfidfFeatureExtractor(max_features=5000, ngram_range=(1, 2))
    analyzer_svm = TraditionalSentimentAnalyzer(model_type='svm', feature_extractor=tfidf_svm)

    results_svm = analyzer_svm.train(texts, ratings)
    logger.info(f"Accuracy: {results_svm['accuracy']:.4f}")
    logger.info(f"F1 Score: {results_svm['f1_score']:.4f}")

    # Save SVM model
    svm_model_path = MODEL_DIR / 'sentiment_svm.pkl'
    svm_vectorizer_path = MODEL_DIR / 'tfidf_vectorizer_svm.pkl'

    joblib.dump(analyzer_svm.model, svm_model_path)
    joblib.dump(tfidf_svm.vectorizer, svm_vectorizer_path)
    logger.info(f"Saved SVM model to {svm_model_path}")

    # Train Naive Bayes model
    logger.info("\n--- Training Naive Bayes Model ---")
    tfidf_nb = TfidfFeatureExtractor(max_features=5000, ngram_range=(1, 2))
    analyzer_nb = TraditionalSentimentAnalyzer(model_type='naive_bayes', feature_extractor=tfidf_nb)

    results_nb = analyzer_nb.train(texts, ratings)
    logger.info(f"Accuracy: {results_nb['accuracy']:.4f}")
    logger.info(f"F1 Score: {results_nb['f1_score']:.4f}")

    # Save Naive Bayes model
    nb_model_path = MODEL_DIR / 'sentiment_nb.pkl'
    nb_vectorizer_path = MODEL_DIR / 'tfidf_vectorizer_nb.pkl'

    joblib.dump(analyzer_nb.model, nb_model_path)
    joblib.dump(tfidf_nb.vectorizer, nb_vectorizer_path)
    logger.info(f"Saved Naive Bayes model to {nb_model_path}")

    logger.info("\n✅ Sentiment analysis models training complete!")
    return True

def train_rating_prediction_models():
    """Train rating prediction models"""
    logger.info("\n" + "="*60)
    logger.info("TRAINING RATING PREDICTION MODELS")
    logger.info("="*60)

    # Load data
    logger.info("Loading review data...")
    reviews_path = PROCESSED_DATA_DIR / 'reviews.csv'

    if not reviews_path.exists():
        logger.error(f"Data file not found: {reviews_path}")
        return False

    df = pd.read_csv(reviews_path)
    logger.info(f"Loaded {len(df)} reviews")

    # Use a subset for faster training
    sample_size = min(50000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)
    logger.info(f"Using {sample_size} samples for training")

    texts = df_sample['text'].tolist()

    # Handle both 'rating' and 'stars' column names
    if 'rating' in df_sample.columns:
        ratings = df_sample['rating'].tolist()
    elif 'stars' in df_sample.columns:
        ratings = df_sample['stars'].tolist()
    else:
        logger.error("No rating or stars column found!")
        return False

    # Train Random Forest model
    logger.info("\n--- Training Random Forest Model ---")
    tfidf_rf = TfidfFeatureExtractor(max_features=3000, ngram_range=(1, 2))
    predictor_rf = TraditionalRatingPredictor(model_type='random_forest', feature_extractor=tfidf_rf)

    results_rf = predictor_rf.train(texts, ratings)
    logger.info(f"RMSE: {results_rf['rmse']:.4f}")
    logger.info(f"MAE: {results_rf['mae']:.4f}")
    logger.info(f"R² Score: {results_rf['r2_score']:.4f}")

    # Save model
    rf_model_path = MODEL_DIR / 'rating_rf.pkl'
    rf_vectorizer_path = MODEL_DIR / 'tfidf_vectorizer_rating.pkl'

    joblib.dump(predictor_rf.model, rf_model_path)
    joblib.dump(tfidf_rf.vectorizer, rf_vectorizer_path)
    logger.info(f"Saved model to {rf_model_path}")

    logger.info("\n✅ Rating prediction models training complete!")
    return True

def train_topic_models():
    """Train topic modeling (LDA)"""
    logger.info("\n" + "="*60)
    logger.info("TRAINING TOPIC MODELING (LDA)")
    logger.info("="*60)

    # Load data
    logger.info("Loading review data...")
    reviews_path = PROCESSED_DATA_DIR / 'reviews.csv'

    if not reviews_path.exists():
        logger.error(f"Data file not found: {reviews_path}")
        return False

    df = pd.read_csv(reviews_path)
    logger.info(f"Loaded {len(df)} reviews")

    # Use a subset for faster training
    sample_size = min(20000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)
    logger.info(f"Using {sample_size} samples for training")

    texts = df_sample['text'].tolist()

    # Train LDA model
    logger.info("\n--- Training LDA Topic Model ---")
    modeler = TopicModeler(num_topics=10, num_words=15, passes=10)

    results = modeler.train(texts)
    logger.info(f"Perplexity: {results['perplexity']:.4f}")
    logger.info(f"Coherence Score: {results['coherence_score']:.4f}")

    # Display topics
    logger.info("\nDiscovered Topics:")
    for topic_id in range(10):
        words = modeler.get_topic_words(topic_id, top_n=10)
        word_str = ', '.join([word for word, _ in words])
        logger.info(f"Topic {topic_id}: {word_str}")

    # Save model
    lda_model_path = MODEL_DIR / 'lda_model.pkl'
    lda_dict_path = MODEL_DIR / 'lda_dictionary.pkl'

    joblib.dump(modeler.model, lda_model_path)
    joblib.dump(modeler.dictionary, lda_dict_path)
    logger.info(f"Saved LDA model to {lda_model_path}")
    logger.info(f"Saved dictionary to {lda_dict_path}")

    logger.info("\n✅ Topic modeling training complete!")
    return True

def main():
    """Main training function"""
    logger.info("="*60)
    logger.info("YELP REVIEW ANALYSIS - MODEL TRAINING")
    logger.info("="*60)

    # Create models directory if it doesn't exist
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Check if data exists
    if not (PROCESSED_DATA_DIR / 'reviews.csv').exists():
        logger.error("\n❌ ERROR: Data not found!")
        logger.error("Please run data preprocessing first:")
        logger.error("  python src/data_processing/data_loader.py")
        return

    success_count = 0
    total_tasks = 3

    # Train all models
    if train_sentiment_models():
        success_count += 1

    if train_rating_prediction_models():
        success_count += 1

    if train_topic_models():
        success_count += 1

    # Summary
    logger.info("\n" + "="*60)
    logger.info("TRAINING SUMMARY")
    logger.info("="*60)
    logger.info(f"Completed: {success_count}/{total_tasks} tasks")

    if success_count == total_tasks:
        logger.info("\n✅ ALL MODELS TRAINED SUCCESSFULLY!")
        logger.info("\nTrained Models:")
        logger.info("  • Sentiment Analysis: Logistic Regression, SVM, Naive Bayes")
        logger.info("  • Rating Prediction: Random Forest")
        logger.info("  • Topic Modeling: LDA (10 topics)")
        logger.info(f"\nModels saved to: {MODEL_DIR}")
        logger.info("\nYou can now run the web application:")
        logger.info("  python run.py")
    else:
        logger.warning("\n⚠️  Some models failed to train. Check errors above.")

if __name__ == '__main__':
    main()
