"""
特征提取模块
包含TF-IDF, Word2Vec, BERT embeddings等
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models import Word2Vec
import torch
from transformers import BertTokenizer, BertModel
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    特征提取器基类
    """

    def __init__(self):
        self.is_fitted = False

    def fit(self, texts):
        """拟合特征提取器"""
        raise NotImplementedError

    def transform(self, texts):
        """转换文本为特征"""
        raise NotImplementedError

    def fit_transform(self, texts):
        """拟合并转换"""
        self.fit(texts)
        return self.transform(texts)


class TfidfFeatureExtractor(FeatureExtractor):
    """
    TF-IDF特征提取器
    """

    def __init__(self, max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.8):
        """
        Args:
            max_features: 最大特征数
            ngram_range: N-gram范围
            min_df: 最小文档频率
            max_df: 最大文档频率
        """
        super().__init__()
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            strip_accents='unicode',
            lowercase=True
        )

    def fit(self, texts):
        """拟合TF-IDF"""
        logger.info("Fitting TF-IDF vectorizer...")
        self.vectorizer.fit(texts)
        self.is_fitted = True
        logger.info(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")

    def transform(self, texts):
        """转换文本为TF-IDF特征"""
        if not self.is_fitted:
            raise ValueError("Vectorizer not fitted yet!")
        return self.vectorizer.transform(texts)

    def get_feature_names(self):
        """获取特征名称"""
        return self.vectorizer.get_feature_names_out()


class Word2VecFeatureExtractor(FeatureExtractor):
    """
    Word2Vec特征提取器
    """

    def __init__(self, vector_size=100, window=5, min_count=2, workers=4):
        """
        Args:
            vector_size: 词向量维度
            window: 上下文窗口大小
            min_count: 最小词频
            workers: 线程数
        """
        super().__init__()
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.model = None

    def fit(self, texts):
        """训练Word2Vec模型"""
        logger.info("Training Word2Vec model...")

        # 分词
        sentences = [text.split() for text in texts]

        # 训练模型
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            epochs=10
        )

        self.is_fitted = True
        logger.info(f"Vocabulary size: {len(self.model.wv)}")

    def transform(self, texts):
        """转换文本为Word2Vec特征（平均词向量）"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet!")

        vectors = []
        for text in texts:
            words = text.split()
            # 获取所有词的向量
            word_vectors = [self.model.wv[word] for word in words if word in self.model.wv]

            if word_vectors:
                # 平均词向量作为文本表示
                vectors.append(np.mean(word_vectors, axis=0))
            else:
                # 如果没有词在词汇表中，使用零向量
                vectors.append(np.zeros(self.vector_size))

        return np.array(vectors)

    def get_word_vector(self, word):
        """获取单个词的向量"""
        if word in self.model.wv:
            return self.model.wv[word]
        return None

    def most_similar(self, word, topn=10):
        """找到最相似的词"""
        if word in self.model.wv:
            return self.model.wv.most_similar(word, topn=topn)
        return []


class BertFeatureExtractor(FeatureExtractor):
    """
    BERT特征提取器
    """

    def __init__(self, model_name='bert-base-uncased', max_length=512, batch_size=16, device=None):
        """
        Args:
            model_name: BERT模型名称
            max_length: 最大序列长度
            batch_size: 批次大小
            device: 设备（cuda或cpu）
        """
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size

        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # 加载模型和tokenizer
        logger.info(f"Loading BERT model: {model_name}")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        self.is_fitted = True  # BERT是预训练的，不需要fit

    def fit(self, texts):
        """BERT不需要fit"""
        pass

    def transform(self, texts, pooling='cls'):
        """
        转换文本为BERT embeddings

        Args:
            texts: 文本列表
            pooling: 池化方法，'cls'使用[CLS] token, 'mean'使用平均池化

        Returns:
            np.array: 嵌入向量
        """
        embeddings = []

        # 批处理
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Extracting BERT features"):
            batch_texts = texts[i:i + self.batch_size]

            # Tokenize
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )

            # 移到设备
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)

            # 获取embeddings
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

            # 池化
            if pooling == 'cls':
                # 使用[CLS] token的表示
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            elif pooling == 'mean':
                # 平均池化
                last_hidden = outputs.last_hidden_state
                # 考虑attention mask
                mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
                sum_embeddings = torch.sum(last_hidden * mask_expanded, 1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                batch_embeddings = (sum_embeddings / sum_mask).cpu().numpy()
            else:
                raise ValueError(f"Unknown pooling method: {pooling}")

            embeddings.append(batch_embeddings)

        return np.vstack(embeddings)


class NGramFeatureExtractor(FeatureExtractor):
    """
    N-gram特征提取器（用于传统ML方法）
    """

    def __init__(self, ngram_range=(1, 3), max_features=10000, binary=False):
        """
        Args:
            ngram_range: N-gram范围
            max_features: 最大特征数
            binary: 是否使用二值特征
        """
        super().__init__()
        self.vectorizer = CountVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            binary=binary,
            lowercase=True
        )

    def fit(self, texts):
        """拟合N-gram vectorizer"""
        logger.info("Fitting N-gram vectorizer...")
        self.vectorizer.fit(texts)
        self.is_fitted = True
        logger.info(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")

    def transform(self, texts):
        """转换文本为N-gram特征"""
        if not self.is_fitted:
            raise ValueError("Vectorizer not fitted yet!")
        return self.vectorizer.transform(texts)


def extract_text_statistics(text):
    """
    提取文本统计特征

    Args:
        text: 输入文本

    Returns:
        dict: 统计特征
    """
    words = text.split()

    features = {
        'num_chars': len(text),
        'num_words': len(words),
        'num_sentences': text.count('.') + text.count('!') + text.count('?'),
        'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
        'num_exclamation': text.count('!'),
        'num_question': text.count('?'),
        'num_uppercase': sum(1 for c in text if c.isupper()),
        'num_digits': sum(1 for c in text if c.isdigit()),
    }

    return features


def extract_batch_statistics(texts):
    """
    批量提取文本统计特征

    Args:
        texts: 文本列表

    Returns:
        pd.DataFrame: 统计特征数据框
    """
    features_list = [extract_text_statistics(text) for text in texts]
    return pd.DataFrame(features_list)


def main():
    """
    测试特征提取器
    """
    # 测试数据
    test_texts = [
        "This is a great restaurant with excellent food!",
        "The service was terrible and the food was cold.",
        "Amazing experience! Highly recommended!",
        "Not bad, but could be better.",
        "Worst place ever. Never going back."
    ]

    print("="*60)
    print("TF-IDF特征提取测试")
    print("="*60)

    tfidf_extractor = TfidfFeatureExtractor(max_features=50)
    tfidf_features = tfidf_extractor.fit_transform(test_texts)
    print(f"Shape: {tfidf_features.shape}")
    print(f"Features: {tfidf_extractor.get_feature_names()[:10]}")

    print("\n" + "="*60)
    print("Word2Vec特征提取测试")
    print("="*60)

    w2v_extractor = Word2VecFeatureExtractor(vector_size=50)
    w2v_features = w2v_extractor.fit_transform(test_texts)
    print(f"Shape: {w2v_features.shape}")

    # 测试词相似度
    similar_words = w2v_extractor.most_similar('food', topn=3)
    print(f"Similar to 'food': {similar_words}")

    print("\n" + "="*60)
    print("文本统计特征测试")
    print("="*60)

    stat_features = extract_batch_statistics(test_texts)
    print(stat_features)

    print("\n" + "="*60)
    print("BERT特征提取测试")
    print("="*60)

    try:
        bert_extractor = BertFeatureExtractor(batch_size=2)
        bert_features = bert_extractor.transform(test_texts[:2])  # 只测试前2个
        print(f"Shape: {bert_features.shape}")
    except Exception as e:
        print(f"BERT test failed: {e}")


if __name__ == '__main__':
    main()
