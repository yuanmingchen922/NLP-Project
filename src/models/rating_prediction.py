"""
评分预测模块
从评论文本预测1-5星评分
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW
from tqdm import tqdm
import joblib
import logging

logger = logging.getLogger(__name__)


class TraditionalRatingPredictor:
    """
    基于传统ML方法的评分预测器
    """

    def __init__(self, model_type='random_forest', feature_extractor=None):
        """
        Args:
            model_type: 'ridge', 'random_forest', 'gradient_boosting'
            feature_extractor: 特征提取器
        """
        self.model_type = model_type
        self.feature_extractor = feature_extractor
        self.model = None

        # 初始化模型
        if model_type == 'ridge':
            self.model = Ridge(alpha=1.0, random_state=42)
        elif model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def train(self, texts, ratings, additional_features=None):
        """
        训练模型

        Args:
            texts: 文本列表
            ratings: 评分列表
            additional_features: 额外特征（如文本统计特征）

        Returns:
            dict: 训练结果
        """
        logger.info(f"Training {self.model_type} model...")

        # 提取特征
        if self.feature_extractor is None:
            raise ValueError("Feature extractor not provided!")

        logger.info("Extracting text features...")
        X_text = self.feature_extractor.fit_transform(texts)

        # 合并额外特征
        if additional_features is not None:
            logger.info("Combining with additional features...")
            from scipy.sparse import hstack, issparse
            if issparse(X_text):
                X = hstack([X_text, additional_features])
            else:
                X = np.hstack([X_text, additional_features])
        else:
            X = X_text

        # 训练测试分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, ratings, test_size=0.2, random_state=42
        )

        # 训练模型
        logger.info("Training...")
        self.model.fit(X_train, y_train)

        # 评估
        y_pred = self.model.predict(X_test)

        # 限制预测范围在1-5之间
        y_pred = np.clip(y_pred, 1, 5)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)

        logger.info(f"RMSE: {rmse:.4f}")
        logger.info(f"MAE: {mae:.4f}")
        logger.info(f"R² Score: {r2:.4f}")

        return {
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2,
            'predictions': y_pred,
            'true_values': y_test
        }

    def predict(self, texts, additional_features=None):
        """
        预测评分

        Args:
            texts: 文本列表
            additional_features: 额外特征

        Returns:
            np.array: 预测评分
        """
        X_text = self.feature_extractor.transform(texts)

        if additional_features is not None:
            from scipy.sparse import hstack, issparse
            if issparse(X_text):
                X = hstack([X_text, additional_features])
            else:
                X = np.hstack([X_text, additional_features])
        else:
            X = X_text

        predictions = self.model.predict(X)
        return np.clip(predictions, 1, 5)

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


class RatingDataset(Dataset):
    """评分预测数据集（用于BERT）"""

    def __init__(self, texts, ratings, tokenizer, max_length=512):
        self.texts = texts
        self.ratings = ratings
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        rating = float(self.ratings[idx])

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
            'rating': torch.tensor(rating, dtype=torch.float)
        }


class BertRatingPredictor(nn.Module):
    """基于BERT的评分预测器"""

    def __init__(self, bert_model_name='bert-base-uncased', dropout=0.3):
        super(BertRatingPredictor, self).__init__()

        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        rating = self.regressor(pooled_output)
        # 使用sigmoid将输出映射到0-1，然后缩放到1-5
        rating = torch.sigmoid(rating) * 4 + 1
        return rating


class BertRatingPredictorTrainer:
    """BERT评分预测训练器"""

    def __init__(self, model_name='bert-base-uncased', max_length=512,
                 batch_size=16, learning_rate=2e-5, epochs=3):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertRatingPredictor(model_name)
        self.model.to(self.device)

    def train(self, texts, ratings):
        """训练模型"""
        logger.info("Preparing data for BERT training...")

        # 训练测试分割
        texts_train, texts_test, ratings_train, ratings_test = train_test_split(
            texts, ratings, test_size=0.2, random_state=42
        )

        # 创建数据集
        train_dataset = RatingDataset(texts_train, ratings_train, self.tokenizer, self.max_length)
        test_dataset = RatingDataset(texts_test, ratings_test, self.tokenizer, self.max_length)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)

        # 优化器和损失函数
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        # 训练循环
        self.model.train()
        for epoch in range(self.epochs):
            logger.info(f"Epoch {epoch + 1}/{self.epochs}")

            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")

            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                ratings = batch['rating'].to(self.device)

                # 前向传播
                predictions = self.model(input_ids, attention_mask).squeeze()
                loss = criterion(predictions, ratings)

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
        all_ratings = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                ratings = batch['rating'].to(self.device)

                predictions = self.model(input_ids, attention_mask).squeeze()

                all_preds.extend(predictions.cpu().numpy())
                all_ratings.extend(ratings.cpu().numpy())

        all_preds = np.array(all_preds)
        all_ratings = np.array(all_ratings)

        # 计算指标
        mse = mean_squared_error(all_ratings, all_preds)
        mae = mean_absolute_error(all_ratings, all_preds)
        r2 = r2_score(all_ratings, all_preds)
        rmse = np.sqrt(mse)

        logger.info(f"RMSE: {rmse:.4f}")
        logger.info(f"MAE: {mae:.4f}")
        logger.info(f"R² Score: {r2:.4f}")

        return {
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2,
            'predictions': all_preds,
            'true_values': all_ratings
        }

    def predict(self, texts):
        """预测评分"""
        self.model.eval()

        dataset = RatingDataset(texts, [3.0] * len(texts), self.tokenizer, self.max_length)
        data_loader = DataLoader(dataset, batch_size=self.batch_size)

        predictions = []

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                preds = self.model(input_ids, attention_mask).squeeze()
                predictions.extend(preds.cpu().numpy())

        return np.array(predictions)

    def save(self, save_path):
        """保存模型"""
        logger.info(f"Saving model to {save_path}")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
        }, save_path)
        self.tokenizer.save_pretrained(save_path)

    def load(self, load_path):
        """加载模型"""
        logger.info(f"Loading model from {load_path}")
        checkpoint = torch.load(load_path / 'pytorch_model.bin')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.tokenizer = BertTokenizer.from_pretrained(load_path)
        self.model.to(self.device)


def main():
    """示例：训练评分预测模型"""
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))

    from config import PROCESSED_DATA_DIR, MODEL_DIR
    from src.data_processing.feature_extraction import TfidfFeatureExtractor, extract_batch_statistics

    # 加载数据
    logger.info("Loading data...")
    df = pd.read_csv(PROCESSED_DATA_DIR / 'reviews.csv')

    # 采样
    df = df.sample(n=min(10000, len(df)), random_state=42)

    texts = df['text'].tolist()
    ratings = df['rating'].tolist()

    # 提取文本统计特征
    logger.info("Extracting text statistics...")
    text_stats = extract_batch_statistics(texts)

    # 训练传统模型
    print("\n" + "="*60)
    print("训练 Random Forest 评分预测模型")
    print("="*60)

    tfidf = TfidfFeatureExtractor(max_features=5000)
    rf_model = TraditionalRatingPredictor(model_type='random_forest', feature_extractor=tfidf)
    rf_results = rf_model.train(texts, ratings, additional_features=text_stats.values)

    # 保存模型
    rf_model.save(
        MODEL_DIR / 'rating_rf.pkl',
        MODEL_DIR / 'rating_tfidf.pkl'
    )

    # 测试预测
    test_texts = [
        "This place is amazing! Best food ever!",
        "Terrible experience. Will never come back.",
        "It's okay, nothing special."
    ]

    test_stats = extract_batch_statistics(test_texts)
    predictions = rf_model.predict(test_texts, additional_features=test_stats.values)

    print("\n测试预测:")
    for text, pred in zip(test_texts, predictions):
        print(f"\nText: {text}")
        print(f"Predicted Rating: {pred:.2f} stars")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
