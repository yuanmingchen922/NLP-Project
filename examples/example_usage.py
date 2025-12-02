"""
Yelp评论分析系统 - 使用示例
演示如何使用各个模块
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

import logging
logging.basicConfig(level=logging.INFO)


def example_text_preprocessing():
    """示例：文本预处理"""
    print("\n" + "="*60)
    print("示例1: 文本预处理")
    print("="*60)

    from src.data_processing.preprocessor import TextPreprocessor, SentimentPreprocessor

    # 创建预处理器
    processor = TextPreprocessor(method='nltk')

    # 测试文本
    text = "This restaurant is AMAZING!!! The food was delicious and the service was great!"

    # 预处理
    processed = processor.process(text)

    print(f"\n原文: {text}")
    print(f"处理后: {processed}")


def example_feature_extraction():
    """示例：特征提取"""
    print("\n" + "="*60)
    print("示例2: 特征提取")
    print("="*60)

    from src.data_processing.feature_extraction import (
        TfidfFeatureExtractor,
        extract_text_statistics
    )

    texts = [
        "Great food and excellent service!",
        "Terrible experience. Never coming back.",
        "The place is okay, nothing special."
    ]

    # TF-IDF特征
    print("\n使用TF-IDF提取特征...")
    tfidf = TfidfFeatureExtractor(max_features=20)
    features = tfidf.fit_transform(texts)
    print(f"特征矩阵形状: {features.shape}")
    print(f"特征词: {tfidf.get_feature_names()[:10]}")

    # 文本统计特征
    print("\n提取文本统计特征...")
    stats = extract_text_statistics(texts[0])
    print(f"统计特征: {stats}")


def example_sentiment_analysis():
    """示例：情感分析"""
    print("\n" + "="*60)
    print("示例3: 情感分析")
    print("="*60)

    from src.models.sentiment_analysis import TraditionalSentimentAnalyzer
    from src.data_processing.feature_extraction import TfidfFeatureExtractor
    import pandas as pd
    from config import PROCESSED_DATA_DIR

    print("\n加载数据...")
    try:
        df = pd.read_csv(PROCESSED_DATA_DIR / 'reviews.csv')
        df = df.sample(n=min(1000, len(df)), random_state=42)

        texts = df['text'].tolist()
        ratings = df['rating'].tolist() if 'rating' in df.columns else df['stars'].tolist()

        print(f"数据量: {len(texts)}")

        # 创建并训练模型
        print("\n训练Logistic Regression模型...")
        tfidf = TfidfFeatureExtractor(max_features=1000)
        model = TraditionalSentimentAnalyzer(model_type='logistic', feature_extractor=tfidf)
        results = model.train(texts, ratings)

        print(f"\n准确率: {results['accuracy']:.4f}")
        print(f"F1分数: {results['f1_score']:.4f}")

        # 测试预测
        test_texts = [
            "This place is amazing! Best food ever!",
            "Terrible experience. Will never come back.",
            "It's okay, nothing special."
        ]

        print("\n测试预测:")
        predictions = model.predict(test_texts)
        probas = model.predict_proba(test_texts)

        sentiment_labels = ['Negative', 'Neutral', 'Positive']
        for text, pred, proba in zip(test_texts, predictions, probas):
            print(f"\nText: {text}")
            print(f"Prediction: {sentiment_labels[pred]}")
            print(f"Confidence: {proba[pred]:.2%}")

    except FileNotFoundError:
        print("未找到数据文件。请先运行 python src/data_processing/data_loader.py")


def example_topic_modeling():
    """示例：主题建模"""
    print("\n" + "="*60)
    print("示例4: 主题建模")
    print("="*60)

    from src.models.topic_modeling import TopicModeler
    import pandas as pd
    from config import PROCESSED_DATA_DIR

    print("\n加载数据...")
    try:
        df = pd.read_csv(PROCESSED_DATA_DIR / 'reviews.csv')
        df = df.sample(n=min(500, len(df)), random_state=42)

        texts = df['text'].tolist()
        print(f"数据量: {len(texts)}")

        # 训练LDA模型
        print("\n训练LDA主题模型...")
        modeler = TopicModeler(num_topics=5, num_words=10, passes=5)
        results = modeler.train(texts)

        print(f"\n困惑度: {results['perplexity']:.4f}")
        print(f"一致性分数: {results['coherence_score']:.4f}")

        # 测试文档主题预测
        test_text = "The food was delicious and the service was excellent!"
        topics = modeler.get_document_topics(test_text)

        print(f"\n测试文本: {test_text}")
        print(f"主题分布:")
        for topic_id, prob in topics[:3]:
            keywords = [word for word, _ in modeler.get_topic_words(topic_id, 5)]
            print(f"  主题{topic_id}: {prob:.2%} - 关键词: {', '.join(keywords)}")

    except FileNotFoundError:
        print("未找到数据文件。请先运行 python src/data_processing/data_loader.py")


def example_recommendation():
    """示例：推荐系统"""
    print("\n" + "="*60)
    print("示例5: 推荐系统")
    print("="*60)

    from src.models.recommendation import ContentBasedRecommender
    import pandas as pd
    from config import PROCESSED_DATA_DIR

    print("\n加载数据...")
    try:
        df = pd.read_csv(PROCESSED_DATA_DIR / 'reviews_merged.csv')
        df = df.sample(n=min(2000, len(df)), random_state=42)

        print(f"数据量: {len(df)}")

        # 构建基于内容的推荐系统
        print("\n构建基于内容的推荐系统...")
        recommender = ContentBasedRecommender(max_features=500)
        recommender.build_business_profiles(df)

        # 获取一个商户ID
        business_id = df['business_id'].iloc[0]

        # 获取相似商户
        print(f"\n为商户 '{business_id}' 推荐相似商户...")
        similar = recommender.recommend_similar_businesses(business_id, top_n=5)

        for i, (bid, score) in enumerate(similar, 1):
            print(f"{i}. 商户ID: {bid}, 相似度: {score:.4f}")

        # 获取商户关键词
        print(f"\n商户关键词:")
        keywords = recommender.get_business_keywords(business_id, top_n=10)
        for word, score in keywords:
            print(f"  {word}: {score:.4f}")

    except FileNotFoundError:
        print("未找到数据文件。请先运行 python src/data_processing/data_loader.py")


def example_web_api():
    """示例：Web API调用"""
    print("\n" + "="*60)
    print("示例6: Web API调用")
    print("="*60)

    import requests

    base_url = "http://localhost:5000/api"

    print("\n请确保Web应用正在运行 (python run.py)")
    print("如果应用未运行，此示例将失败。\n")

    try:
        # 测试情感分析API
        print("测试情感分析API...")
        response = requests.post(
            f"{base_url}/analyze/sentiment",
            json={'text': 'This place is amazing! Best food ever!'},
            timeout=5
        )

        if response.status_code == 200:
            result = response.json()
            print(f"情感: {result.get('sentiment', 'N/A')}")
            print(f"置信度: {result.get('confidence', 0):.2%}")
        else:
            print(f"请求失败: {response.status_code}")

        # 测试商户搜索API
        print("\n测试商户搜索API...")
        response = requests.get(
            f"{base_url}/businesses?search=restaurant&per_page=3",
            timeout=5
        )

        if response.status_code == 200:
            result = response.json()
            print(f"找到 {result.get('total', 0)} 个商户")
            for business in result.get('businesses', [])[:3]:
                print(f"  - {business.get('name', 'Unknown')}")

    except requests.exceptions.ConnectionError:
        print("无法连接到Web应用。请先运行: python run.py")
    except Exception as e:
        print(f"错误: {e}")


def main():
    """运行所有示例"""
    print("\n" + "="*60)
    print("Yelp评论分析系统 - 使用示例")
    print("="*60)

    # 运行各个示例
    example_text_preprocessing()
    example_feature_extraction()
    example_sentiment_analysis()
    example_topic_modeling()
    example_recommendation()
    example_web_api()

    print("\n" + "="*60)
    print("所有示例运行完成！")
    print("="*60)


if __name__ == '__main__':
    main()
