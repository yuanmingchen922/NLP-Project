"""
情感分析模块
包含传统ML方法和深度学习方法
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from tqdm import tqdm
import joblib
import logging

logger = logging.getLogger(__name__)


class TraditionalSentimentAnalyzer:
    """
    基于传统ML方法的情感分析器
    支持Logistic Regression, SVM, Naive Bayes
    """

    def __init__(self, model_type='logistic', feature_extractor=None):
        """
        Args:
            model_type: 'logistic', 'svm', 或 'naive_bayes'
            feature_extractor: 特征提取器（如TF-IDF）
        """
        self.model_type = model_type
        self.feature_extractor = feature_extractor
        self.model = None

        # 初始化模型
        if model_type == 'logistic':
            self.model = LogisticRegression(max_iter=1000, random_state=42)
        elif model_type == 'svm':
            self.model = LinearSVC(max_iter=1000, random_state=42)
        elif model_type == 'naive_bayes':
            self.model = MultinomialNB()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def prepare_labels(self, ratings):
        """
        将评分转换为情感标签
        1-2星 -> 负面 (0)
        3星 -> 中性 (1)
        4-5星 -> 正面 (2)

        Args:
            ratings: 评分数组

        Returns:
            np.array: 标签数组
        """
        labels = np.zeros(len(ratings), dtype=int)
        labels[ratings <= 2] = 0  # 负面
        labels[ratings == 3] = 1  # 中性
        labels[ratings >= 4] = 2  # 正面
        return labels

    def train(self, texts, ratings):
        """
        训练模型

        Args:
            texts: 文本列表
            ratings: 评分列表

        Returns:
            dict: 训练结果
        """
        logger.info(f"Training {self.model_type} model...")

        # 准备标签
        labels = self.prepare_labels(np.array(ratings))

        # 提取特征
        if self.feature_extractor is None:
            raise ValueError("Feature extractor not provided!")

        logger.info("Extracting features...")
        X = self.feature_extractor.fit_transform(texts)

        # 训练测试分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.2, random_state=42, stratify=labels
        )

        # 训练模型
        logger.info("Training...")
        self.model.fit(X_train, y_train)

        # 评估
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")

        # 详细报告
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Negative', 'Neutral', 'Positive']))

        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }

    def predict(self, texts):
        """
        预测情感

        Args:
            texts: 文本列表

        Returns:
            np.array: 预测标签
        """
        X = self.feature_extractor.transform(texts)
        return self.model.predict(X)

    def predict_proba(self, texts):
        """
        预测概率

        Args:
            texts: 文本列表

        Returns:
            np.array: 预测概率
        """
        X = self.feature_extractor.transform(texts)
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # SVM没有predict_proba，使用decision_function
            scores = self.model.decision_function(X)
            # 转换为概率（简化版本）
            return np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)

    def save(self, model_path, feature_extractor_path=None):
        """保存模型"""
        logger.info(f"Saving model to {model_path}")
        joblib.dump(self.model, model_path)
        if feature_extractor_path and self.feature_extractor:
            joblib.dump(self.feature_extractor, feature_extractor_path)

    def load(self, model_path, feature_extractor_path=None):
        """加载模型"""
        logger.info(f"Loading model from {model_path}")
        self.model = joblib.load(model_path)
        if feature_extractor_path:
            self.feature_extractor = joblib.load(feature_extractor_path)


class ReviewDataset(Dataset):
    """
    评论数据集（用于BERT）
    """

    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


class BertSentimentAnalyzer:
    """
    基于BERT的情感分析器
    """

    def __init__(self, model_name='bert-base-uncased', num_labels=3, max_length=512,
                 batch_size=16, learning_rate=2e-5, epochs=3):
        """
        Args:
            model_name: BERT模型名称
            num_labels: 标签数量（3: 负面/中性/正面）
            max_length: 最大序列长度
            batch_size: 批次大小
            learning_rate: 学习率
            epochs: 训练轮数
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs

        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        # 加载tokenizer和模型
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        self.model.to(self.device)

    def prepare_labels(self, ratings):
        """将评分转换为情感标签"""
        labels = np.zeros(len(ratings), dtype=int)
        labels[ratings <= 2] = 0  # 负面
        labels[ratings == 3] = 1  # 中性
        labels[ratings >= 4] = 2  # 正面
        return labels

    def train(self, texts, ratings):
        """
        训练BERT模型

        Args:
            texts: 文本列表
            ratings: 评分列表

        Returns:
            dict: 训练结果
        """
        logger.info("Preparing data for BERT training...")

        # 准备标签
        labels = self.prepare_labels(np.array(ratings))

        # 训练测试分割
        texts_train, texts_test, labels_train, labels_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )

        # 创建数据集
        train_dataset = ReviewDataset(texts_train, labels_train, self.tokenizer, self.max_length)
        test_dataset = ReviewDataset(texts_test, labels_test, self.tokenizer, self.max_length)

        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)

        # 优化器
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)

        # 训练循环
        self.model.train()
        for epoch in range(self.epochs):
            logger.info(f"Epoch {epoch + 1}/{self.epochs}")

            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")

            for batch in progress_bar:
                # 移到设备
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                # 前向传播
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                total_loss += loss.item()

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                progress_bar.set_postfix({'loss': loss.item()})

            avg_loss = total_loss / len(train_loader)
            logger.info(f"Average loss: {avg_loss:.4f}")

        # 评估
        logger.info("Evaluating model...")
        results = self.evaluate(test_loader)

        return results

    def evaluate(self, data_loader):
        """评估模型"""
        self.model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 计算指标
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')

        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")

        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds,
                                     target_names=['Negative', 'Neutral', 'Positive']))

        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'confusion_matrix': confusion_matrix(all_labels, all_preds)
        }

    def predict(self, texts):
        """预测情感"""
        self.model.eval()

        # 创建数据集（标签为占位符）
        dataset = ReviewDataset(texts, [0] * len(texts), self.tokenizer, self.max_length)
        data_loader = DataLoader(dataset, batch_size=self.batch_size)

        predictions = []

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)

                predictions.extend(preds.cpu().numpy())

        return np.array(predictions)

    def predict_proba(self, texts):
        """预测概率"""
        self.model.eval()

        dataset = ReviewDataset(texts, [0] * len(texts), self.tokenizer, self.max_length)
        data_loader = DataLoader(dataset, batch_size=self.batch_size)

        probabilities = []

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=1)

                probabilities.extend(probs.cpu().numpy())

        return np.array(probabilities)

    def save(self, save_path):
        """保存模型"""
        logger.info(f"Saving model to {save_path}")
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    def load(self, load_path):
        """加载模型"""
        logger.info(f"Loading model from {load_path}")
        self.model = BertForSequenceClassification.from_pretrained(load_path)
        self.tokenizer = BertTokenizer.from_pretrained(load_path)
        self.model.to(self.device)


def main():
    """
    示例：训练情感分析模型
    """
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))

    from config import PROCESSED_DATA_DIR, MODEL_DIR
    from src.data_processing.feature_extraction import TfidfFeatureExtractor

    # 加载数据
    logger.info("Loading data...")
    df = pd.read_csv(PROCESSED_DATA_DIR / 'reviews.csv')

    # 采样（用于快速测试）
    df = df.sample(n=min(10000, len(df)), random_state=42)

    texts = df['text'].tolist()
    ratings = df['rating'].tolist()

    # 训练传统模型
    print("\n" + "="*60)
    print("训练 Logistic Regression 模型")
    print("="*60)

    tfidf = TfidfFeatureExtractor(max_features=5000)
    lr_model = TraditionalSentimentAnalyzer(model_type='logistic', feature_extractor=tfidf)
    lr_results = lr_model.train(texts, ratings)

    # 保存模型
    lr_model.save(
        MODEL_DIR / 'sentiment_lr.pkl',
        MODEL_DIR / 'tfidf_vectorizer.pkl'
    )

    # 测试预测
    test_texts = [
        "This place is amazing! Best food ever!",
        "Terrible experience. Will never come back.",
        "It's okay, nothing special."
    ]

    predictions = lr_model.predict(test_texts)
    probas = lr_model.predict_proba(test_texts)

    print("\n测试预测:")
    sentiment_labels = ['Negative', 'Neutral', 'Positive']
    for text, pred, proba in zip(test_texts, predictions, probas):
        print(f"\nText: {text}")
        print(f"Prediction: {sentiment_labels[pred]}")
        print(f"Probabilities: {proba}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
