"""
Yelp数据加载器
处理JSONL格式的Yelp数据集，支持采样和数据转换
"""

import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import random
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YelpDataLoader:
    """
    Yelp数据集加载器
    支持加载review, business, user, tip, checkin数据
    """

    def __init__(self, dataset_dir):
        """
        初始化数据加载器

        Args:
            dataset_dir: Yelp数据集目录路径
        """
        self.dataset_dir = Path(dataset_dir)

        # 数据文件路径
        self.files = {
            'review': self.dataset_dir / 'yelp_academic_dataset_review.json',
            'business': self.dataset_dir / 'yelp_academic_dataset_business.json',
            'user': self.dataset_dir / 'yelp_academic_dataset_user.json',
            'tip': self.dataset_dir / 'yelp_academic_dataset_tip.json',
            'checkin': self.dataset_dir / 'yelp_academic_dataset_checkin.json'
        }

        # 验证文件存在
        for name, path in self.files.items():
            if not path.exists():
                logger.warning(f"{name} file not found: {path}")

    def load_jsonl(self, file_path, sample_size=None, random_seed=42):
        """
        加载JSONL文件（每行一个JSON对象）

        Args:
            file_path: 文件路径
            sample_size: 采样大小，None表示加载全部
            random_seed: 随机种子

        Returns:
            list: JSON对象列表
        """
        data = []

        if not Path(file_path).exists():
            logger.error(f"File not found: {file_path}")
            return data

        # 如果需要采样，先计算总行数
        if sample_size:
            logger.info(f"Counting lines in {file_path}...")
            with open(file_path, 'r', encoding='utf-8') as f:
                total_lines = sum(1 for _ in f)

            # 计算采样间隔
            if sample_size >= total_lines:
                sample_size = None  # 如果采样数>=总数，加载全部
            else:
                # 使用reservoir sampling或均匀采样
                random.seed(random_seed)
                sample_indices = set(random.sample(range(total_lines), sample_size))

        logger.info(f"Loading data from {file_path}...")

        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f, desc="Loading")):
                # 如果需要采样且当前行不在采样索引中，跳过
                if sample_size and i not in sample_indices:
                    continue

                try:
                    obj = json.loads(line)
                    data.append(obj)
                except json.JSONDecodeError as e:
                    logger.warning(f"Error parsing line {i}: {e}")
                    continue

                # 如果已达到采样大小，可以提前结束
                if sample_size and len(data) >= sample_size:
                    break

        logger.info(f"Loaded {len(data)} records")
        return data

    def load_reviews(self, sample_size=None, min_text_length=10):
        """
        加载评论数据

        Args:
            sample_size: 采样大小
            min_text_length: 最小文本长度（过滤太短的评论）

        Returns:
            pd.DataFrame: 评论数据框
        """
        reviews = self.load_jsonl(self.files['review'], sample_size=sample_size)
        df = pd.DataFrame(reviews)

        if len(df) == 0:
            return df

        # 过滤太短的评论
        if 'text' in df.columns:
            original_len = len(df)
            df = df[df['text'].str.len() >= min_text_length]
            logger.info(f"Filtered {original_len - len(df)} reviews with text length < {min_text_length}")

        # 转换日期
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])

        return df

    def load_businesses(self):
        """
        加载商户数据

        Returns:
            pd.DataFrame: 商户数据框
        """
        businesses = self.load_jsonl(self.files['business'])
        df = pd.DataFrame(businesses)

        # 处理categories字段（逗号分隔的字符串 -> 列表）
        if 'categories' in df.columns:
            df['categories'] = df['categories'].apply(
                lambda x: x.split(', ') if isinstance(x, str) else []
            )

        return df

    def load_users(self, sample_size=None):
        """
        加载用户数据

        Args:
            sample_size: 采样大小

        Returns:
            pd.DataFrame: 用户数据框
        """
        users = self.load_jsonl(self.files['user'], sample_size=sample_size)
        df = pd.DataFrame(users)

        # 转换日期
        if 'yelping_since' in df.columns:
            df['yelping_since'] = pd.to_datetime(df['yelping_since'])

        # 处理friends字段
        if 'friends' in df.columns:
            df['friends'] = df['friends'].apply(
                lambda x: x.split(', ') if isinstance(x, str) and x != 'None' else []
            )
            df['num_friends'] = df['friends'].apply(len)

        return df

    def load_tips(self, sample_size=None):
        """
        加载小贴士数据

        Returns:
            pd.DataFrame: 小贴士数据框
        """
        tips = self.load_jsonl(self.files['tip'], sample_size=sample_size)
        df = pd.DataFrame(tips)

        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])

        return df

    def create_merged_dataset(self, review_sample_size=100000, save_path=None):
        """
        创建合并的数据集（评论 + 商户信息 + 用户信息）

        Args:
            review_sample_size: 评论采样大小
            save_path: 保存路径，None表示不保存

        Returns:
            pd.DataFrame: 合并后的数据框
        """
        logger.info("Loading reviews...")
        reviews_df = self.load_reviews(sample_size=review_sample_size)

        logger.info("Loading businesses...")
        business_df = self.load_businesses()

        logger.info("Loading users...")
        # 只加载在评论中出现的用户
        user_ids = reviews_df['user_id'].unique()
        users_df = self.load_users()
        users_df = users_df[users_df['user_id'].isin(user_ids)]

        # 合并数据
        logger.info("Merging datasets...")
        merged_df = reviews_df.merge(
            business_df[['business_id', 'name', 'city', 'state', 'stars', 'categories']],
            on='business_id',
            how='left',
            suffixes=('_review', '_business')
        )

        merged_df = merged_df.merge(
            users_df[['user_id', 'review_count', 'average_stars', 'num_friends']],
            on='user_id',
            how='left',
            suffixes=('', '_user')
        )

        # 重命名列以避免混淆
        merged_df = merged_df.rename(columns={
            'stars_review': 'rating',
            'stars_business': 'business_avg_rating',
            'name': 'business_name',
            'review_count': 'user_review_count',
            'average_stars': 'user_avg_rating'
        })

        if save_path:
            logger.info(f"Saving merged dataset to {save_path}...")
            merged_df.to_csv(save_path, index=False)

        logger.info(f"Created merged dataset with {len(merged_df)} records")
        return merged_df


def main():
    """
    主函数：示例使用
    """
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from config import YELP_DATASET_DIR, PROCESSED_DATA_DIR, SAMPLE_SIZE

    # 创建数据加载器
    loader = YelpDataLoader(YELP_DATASET_DIR)

    # 创建合并数据集
    merged_df = loader.create_merged_dataset(
        review_sample_size=SAMPLE_SIZE,
        save_path=PROCESSED_DATA_DIR / 'reviews_merged.csv'
    )

    # 显示数据信息
    print("\n" + "="*50)
    print("数据集信息:")
    print("="*50)
    print(f"总记录数: {len(merged_df)}")
    print(f"\n列名:\n{merged_df.columns.tolist()}")
    print(f"\n数据类型:\n{merged_df.dtypes}")
    print(f"\n缺失值:\n{merged_df.isnull().sum()}")
    print(f"\n评分分布:\n{merged_df['rating'].value_counts().sort_index()}")
    print(f"\n前5条记录:\n{merged_df.head()}")

    # 保存分离的数据集
    logger.info("Saving separate datasets...")
    reviews_df = loader.load_reviews(sample_size=SAMPLE_SIZE)
    reviews_df.to_csv(PROCESSED_DATA_DIR / 'reviews.csv', index=False)

    business_df = loader.load_businesses()
    business_df.to_csv(PROCESSED_DATA_DIR / 'businesses.csv', index=False)

    logger.info("Data loading completed!")


if __name__ == '__main__':
    main()
