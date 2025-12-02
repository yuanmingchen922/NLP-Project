"""
分析商家和评论数据匹配情况
"""
import pandas as pd
from pathlib import Path

# 数据路径
DATA_DIR = Path(__file__).parent / 'data' / 'processed'
reviews_path = DATA_DIR / 'reviews_merged.csv'
businesses_path = DATA_DIR / 'businesses.csv'

print("="*60)
print("数据匹配分析")
print("="*60)

# 加载数据
print("\n加载数据...")
reviews_df = pd.read_csv(reviews_path)
businesses_df = pd.read_csv(businesses_path)

print(f"评论总数: {len(reviews_df)}")
print(f"商家总数: {len(businesses_df)}")

# 检查business_id列
print(f"\n评论数据的列: {reviews_df.columns.tolist()}")
print(f"商家数据的列: {businesses_df.columns.tolist()}")

# 获取唯一的business_id
unique_review_business_ids = set(reviews_df['business_id'].unique())
unique_business_ids = set(businesses_df['business_id'].unique())

print(f"\n有评论的商家数量: {len(unique_review_business_ids)}")
print(f"商家数据中的商家数量: {len(unique_business_ids)}")

# 检查匹配情况
businesses_with_reviews = unique_business_ids & unique_review_business_ids
businesses_without_reviews = unique_business_ids - unique_review_business_ids

print(f"\n有评论的商家数: {len(businesses_with_reviews)}")
print(f"没有评论的商家数: {len(businesses_without_reviews)}")
print(f"匹配率: {len(businesses_with_reviews) / len(unique_business_ids) * 100:.2f}%")

# 统计每个商家的评论数
review_counts = reviews_df['business_id'].value_counts()
print(f"\n评论数统计:")
print(f"平均每个商家的评论数: {review_counts.mean():.2f}")
print(f"中位数评论数: {review_counts.median():.0f}")
print(f"最多评论数: {review_counts.max()}")
print(f"最少评论数: {review_counts.min()}")

# 显示评论数分布
print(f"\n评论数分布:")
print(f"1-10条评论的商家: {len(review_counts[review_counts <= 10])}")
print(f"11-50条评论的商家: {len(review_counts[(review_counts > 10) & (review_counts <= 50)])}")
print(f"51-100条评论的商家: {len(review_counts[(review_counts > 50) & (review_counts <= 100)])}")
print(f"100+条评论的商家: {len(review_counts[review_counts > 100])}")

print("\n" + "="*60)
