"""
推荐系统模块
包含基于内容的推荐和协同过滤
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import logging

logger = logging.getLogger(__name__)


class ContentBasedRecommender:
    """
    基于内容的推荐系统
    使用商户的评论文本计算相似度
    """

    def __init__(self, max_features=5000):
        """
        Args:
            max_features: TF-IDF最大特征数
        """
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
        self.business_profiles = None
        self.business_ids = None
        self.similarity_matrix = None

    def build_business_profiles(self, reviews_df):
        """
        构建商户内容档案

        Args:
            reviews_df: 评论数据框（包含business_id和text列）

        Returns:
            pd.DataFrame: 商户档案
        """
        logger.info("Building business profiles...")

        # 合并每个商户的所有评论
        business_texts = reviews_df.groupby('business_id')['text'].apply(
            lambda x: ' '.join(x)
        ).reset_index()

        business_texts.columns = ['business_id', 'combined_text']

        self.business_ids = business_texts['business_id'].values

        # 提取TF-IDF特征
        logger.info("Extracting TF-IDF features...")
        self.business_profiles = self.vectorizer.fit_transform(
            business_texts['combined_text']
        )

        # 计算相似度矩阵
        logger.info("Computing similarity matrix...")
        self.similarity_matrix = cosine_similarity(self.business_profiles)

        logger.info(f"Built profiles for {len(self.business_ids)} businesses")

        return business_texts

    def recommend_similar_businesses(self, business_id, top_n=10):
        """
        推荐与给定商户相似的商户

        Args:
            business_id: 商户ID
            top_n: 返回top N个推荐

        Returns:
            list: (business_id, similarity_score) 元组列表
        """
        if self.similarity_matrix is None:
            raise ValueError("Must build business profiles first!")

        # 找到business_id的索引
        try:
            idx = np.where(self.business_ids == business_id)[0][0]
        except IndexError:
            logger.warning(f"Business {business_id} not found")
            return []

        # 获取相似度分数
        sim_scores = list(enumerate(self.similarity_matrix[idx]))

        # 排序（降序）
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # 排除自己，取top N
        sim_scores = sim_scores[1:top_n + 1]

        # 返回business_id和分数
        recommendations = [
            (self.business_ids[i], score) for i, score in sim_scores
        ]

        return recommendations

    def get_business_keywords(self, business_id, top_n=10):
        """
        获取商户的关键词

        Args:
            business_id: 商户ID
            top_n: 返回top N个关键词

        Returns:
            list: (keyword, score) 元组列表
        """
        try:
            idx = np.where(self.business_ids == business_id)[0][0]
        except IndexError:
            logger.warning(f"Business {business_id} not found")
            return []

        # 获取该商户的TF-IDF向量
        feature_array = self.business_profiles[idx].toarray().flatten()

        # 获取特征名称
        feature_names = self.vectorizer.get_feature_names_out()

        # 排序
        top_indices = feature_array.argsort()[-top_n:][::-1]

        keywords = [
            (feature_names[i], feature_array[i]) for i in top_indices
        ]

        return keywords


class CollaborativeFilteringRecommender:
    """
    协同过滤推荐系统
    基于用户-商户评分矩阵
    """

    def __init__(self, n_factors=50):
        """
        Args:
            n_factors: 矩阵分解的因子数
        """
        self.n_factors = n_factors
        self.user_ids = None
        self.business_ids = None
        self.rating_matrix = None
        self.user_factors = None
        self.business_factors = None
        self.user_bias = None
        self.business_bias = None
        self.global_mean = None

    def build_rating_matrix(self, reviews_df):
        """
        构建用户-商户评分矩阵

        Args:
            reviews_df: 评论数据框（包含user_id, business_id, rating列）

        Returns:
            scipy.sparse.csr_matrix: 稀疏评分矩阵
        """
        logger.info("Building rating matrix...")

        # 获取唯一的用户和商户
        self.user_ids = reviews_df['user_id'].unique()
        self.business_ids = reviews_df['business_id'].unique()

        # 创建ID到索引的映射
        user_to_idx = {user_id: idx for idx, user_id in enumerate(self.user_ids)}
        business_to_idx = {business_id: idx for idx, business_id in enumerate(self.business_ids)}

        # 创建稀疏矩阵
        rows = reviews_df['user_id'].map(user_to_idx)
        cols = reviews_df['business_id'].map(business_to_idx)
        data = reviews_df['rating'].values

        self.rating_matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(len(self.user_ids), len(self.business_ids))
        )

        # 计算全局平均评分
        self.global_mean = np.mean(data)

        logger.info(f"Rating matrix shape: {self.rating_matrix.shape}")
        logger.info(f"Sparsity: {1 - self.rating_matrix.nnz / (self.rating_matrix.shape[0] * self.rating_matrix.shape[1]):.4f}")

        return self.rating_matrix

    def train_svd(self):
        """
        使用SVD进行矩阵分解
        """
        logger.info("Performing SVD...")

        # 中心化评分矩阵
        rating_mean = self.rating_matrix.mean()
        rating_centered = self.rating_matrix - rating_mean

        # SVD分解
        U, sigma, Vt = svds(rating_centered, k=self.n_factors)

        # 保存因子
        self.user_factors = U
        self.business_factors = Vt.T

        logger.info("SVD training completed")

    def predict_rating(self, user_id, business_id):
        """
        预测用户对商户的评分

        Args:
            user_id: 用户ID
            business_id: 商户ID

        Returns:
            float: 预测评分
        """
        try:
            user_idx = np.where(self.user_ids == user_id)[0][0]
            business_idx = np.where(self.business_ids == business_id)[0][0]
        except IndexError:
            # 如果用户或商户不存在，返回全局平均
            return self.global_mean

        # 预测评分
        prediction = np.dot(
            self.user_factors[user_idx],
            self.business_factors[business_idx]
        ) + self.global_mean

        # 限制在1-5之间
        return np.clip(prediction, 1, 5)

    def recommend_for_user(self, user_id, top_n=10, exclude_rated=True):
        """
        为用户推荐商户

        Args:
            user_id: 用户ID
            top_n: 返回top N个推荐
            exclude_rated: 是否排除已评分的商户

        Returns:
            list: (business_id, predicted_rating) 元组列表
        """
        try:
            user_idx = np.where(self.user_ids == user_id)[0][0]
        except IndexError:
            logger.warning(f"User {user_id} not found")
            return []

        # 预测该用户对所有商户的评分
        predictions = np.dot(
            self.user_factors[user_idx],
            self.business_factors.T
        ) + self.global_mean

        # 如果需要排除已评分的商户
        if exclude_rated:
            rated_businesses = self.rating_matrix[user_idx].nonzero()[1]
            predictions[rated_businesses] = -np.inf

        # 获取top N
        top_indices = predictions.argsort()[-top_n:][::-1]

        recommendations = [
            (self.business_ids[idx], predictions[idx])
            for idx in top_indices
            if predictions[idx] != -np.inf
        ]

        return recommendations


class HybridRecommender:
    """
    混合推荐系统
    结合基于内容和协同过滤
    """

    def __init__(self, content_weight=0.5, cf_weight=0.5):
        """
        Args:
            content_weight: 基于内容推荐的权重
            cf_weight: 协同过滤的权重
        """
        self.content_weight = content_weight
        self.cf_weight = cf_weight
        self.content_recommender = None
        self.cf_recommender = None

    def train(self, reviews_df):
        """
        训练两个推荐系统

        Args:
            reviews_df: 评论数据框
        """
        logger.info("Training hybrid recommender...")

        # 训练基于内容的推荐
        logger.info("Training content-based recommender...")
        self.content_recommender = ContentBasedRecommender()
        self.content_recommender.build_business_profiles(reviews_df)

        # 训练协同过滤
        logger.info("Training collaborative filtering...")
        self.cf_recommender = CollaborativeFilteringRecommender()
        self.cf_recommender.build_rating_matrix(reviews_df)
        self.cf_recommender.train_svd()

        logger.info("Hybrid recommender training completed")

    def recommend(self, user_id, business_id=None, top_n=10):
        """
        混合推荐

        Args:
            user_id: 用户ID
            business_id: 可选，用于基于内容的推荐
            top_n: 返回top N个推荐

        Returns:
            list: (business_id, score) 元组列表
        """
        recommendations = {}

        # 协同过滤推荐
        cf_recs = self.cf_recommender.recommend_for_user(user_id, top_n=top_n * 2)
        for business_id, score in cf_recs:
            recommendations[business_id] = score * self.cf_weight

        # 如果提供了business_id，添加基于内容的推荐
        if business_id:
            content_recs = self.content_recommender.recommend_similar_businesses(
                business_id, top_n=top_n * 2
            )
            for bid, score in content_recs:
                if bid in recommendations:
                    recommendations[bid] += score * self.content_weight
                else:
                    recommendations[bid] = score * self.content_weight

        # 排序并返回top N
        sorted_recs = sorted(
            recommendations.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]

        return sorted_recs


def evaluate_recommender(recommender, test_df, top_n=10):
    """
    评估推荐系统

    Args:
        recommender: 推荐器
        test_df: 测试数据
        top_n: Top N推荐

    Returns:
        dict: 评估指标
    """
    logger.info("Evaluating recommender...")

    precisions = []
    recalls = []

    for user_id in test_df['user_id'].unique()[:100]:  # 只评估100个用户
        # 获取用户的真实评分高于4的商户
        user_test = test_df[test_df['user_id'] == user_id]
        relevant_items = set(user_test[user_test['rating'] >= 4]['business_id'])

        if len(relevant_items) == 0:
            continue

        # 获取推荐
        if isinstance(recommender, CollaborativeFilteringRecommender):
            recs = recommender.recommend_for_user(user_id, top_n=top_n)
        else:
            continue

        recommended_items = set([business_id for business_id, _ in recs])

        # 计算precision和recall
        hits = len(relevant_items & recommended_items)
        precision = hits / len(recommended_items) if recommended_items else 0
        recall = hits / len(relevant_items) if relevant_items else 0

        precisions.append(precision)
        recalls.append(recall)

    avg_precision = np.mean(precisions) if precisions else 0
    avg_recall = np.mean(recalls) if recalls else 0

    logger.info(f"Precision@{top_n}: {avg_precision:.4f}")
    logger.info(f"Recall@{top_n}: {avg_recall:.4f}")

    return {
        'precision': avg_precision,
        'recall': avg_recall
    }


def main():
    """示例：推荐系统"""
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))

    from config import PROCESSED_DATA_DIR

    # 加载数据
    logger.info("Loading data...")
    df = pd.read_csv(PROCESSED_DATA_DIR / 'reviews_merged.csv')

    # 采样
    df = df.sample(n=min(20000, len(df)), random_state=42)

    # 训练测试分割
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)

    print("\n" + "="*60)
    print("训练基于内容的推荐系统")
    print("="*60)

    # 基于内容的推荐
    content_rec = ContentBasedRecommender()
    content_rec.build_business_profiles(train_df)

    # 测试推荐
    test_business_id = train_df['business_id'].iloc[0]
    similar = content_rec.recommend_similar_businesses(test_business_id, top_n=5)
    print(f"\n与商户 {test_business_id} 相似的商户:")
    for bid, score in similar:
        print(f"  {bid}: {score:.4f}")

    # 获取关键词
    keywords = content_rec.get_business_keywords(test_business_id, top_n=10)
    print(f"\n商户关键词:")
    for word, score in keywords:
        print(f"  {word}: {score:.4f}")

    print("\n" + "="*60)
    print("训练协同过滤推荐系统")
    print("="*60)

    # 协同过滤
    cf_rec = CollaborativeFilteringRecommender(n_factors=20)
    cf_rec.build_rating_matrix(train_df)
    cf_rec.train_svd()

    # 测试推荐
    test_user_id = train_df['user_id'].iloc[0]
    recommendations = cf_rec.recommend_for_user(test_user_id, top_n=5)
    print(f"\n为用户 {test_user_id} 推荐:")
    for bid, score in recommendations:
        print(f"  {bid}: {score:.2f} stars")

    # 评估
    metrics = evaluate_recommender(cf_rec, test_df, top_n=10)
    print(f"\n评估结果:")
    print(f"  Precision@10: {metrics['precision']:.4f}")
    print(f"  Recall@10: {metrics['recall']:.4f}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
