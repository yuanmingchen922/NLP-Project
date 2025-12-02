"""
重新处理数据，优先保留有评论的商家
只保留有评论数据的商家，确保搜索结果都有评论摘要
"""
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm

# 配置
BASE_DIR = Path(__file__).parent
YELP_DATASET_DIR = BASE_DIR / 'yelp_dataset'
PROCESSED_DATA_DIR = BASE_DIR / 'data' / 'processed'

REVIEW_FILE = YELP_DATASET_DIR / 'yelp_academic_dataset_review.json'
BUSINESS_FILE = YELP_DATASET_DIR / 'yelp_academic_dataset_business.json'
USER_FILE = YELP_DATASET_DIR / 'yelp_academic_dataset_user.json'

# 目标：保留100,000条评论和所有对应的商家
TARGET_REVIEWS = 100000

print("="*60)
print("重新处理Yelp数据 - 优先保留有评论的商家")
print("="*60)

# Step 1: 加载评论数据
print("\n步骤1: 加载评论数据...")
reviews = []
business_ids_with_reviews = set()

with open(REVIEW_FILE, 'r', encoding='utf-8') as f:
    for i, line in enumerate(tqdm(f, desc="读取评论", total=TARGET_REVIEWS)):
        if i >= TARGET_REVIEWS:
            break
        review = json.loads(line)
        reviews.append({
            'review_id': review['review_id'],
            'user_id': review['user_id'],
            'business_id': review['business_id'],
            'stars': review['stars'],
            'useful': review['useful'],
            'funny': review['funny'],
            'cool': review['cool'],
            'text': review['text'],
            'date': review['date']
        })
        business_ids_with_reviews.add(review['business_id'])

print(f"加载了 {len(reviews)} 条评论")
print(f"涉及 {len(business_ids_with_reviews)} 个商家")

# Step 2: 只加载有评论的商家
print("\n步骤2: 加载对应的商家数据...")
businesses = []

with open(BUSINESS_FILE, 'r', encoding='utf-8') as f:
    for line in tqdm(f, desc="读取商家"):
        business = json.loads(line)
        # 只保留有评论的商家
        if business['business_id'] in business_ids_with_reviews:
            businesses.append({
                'business_id': business['business_id'],
                'name': business['name'],
                'address': business.get('address', ''),
                'city': business.get('city', ''),
                'state': business.get('state', ''),
                'postal_code': business.get('postal_code', ''),
                'latitude': business.get('latitude', None),
                'longitude': business.get('longitude', None),
                'stars': business.get('stars', 0),
                'review_count': business.get('review_count', 0),
                'is_open': business.get('is_open', 1),
                'categories': business.get('categories', ''),
            })

print(f"加载了 {len(businesses)} 个商家（全部有评论）")

# Step 3: 加载用户数据（用于合并）
print("\n步骤3: 加载用户数据...")
user_ids = set(r['user_id'] for r in reviews)
users = {}

with open(USER_FILE, 'r', encoding='utf-8') as f:
    for line in tqdm(f, desc="读取用户"):
        user = json.loads(line)
        if user['user_id'] in user_ids:
            users[user['user_id']] = {
                'user_review_count': user.get('review_count', 0),
                'user_avg_rating': user.get('average_stars', 0),
                'num_friends': len(user.get('friends', '').split(',')) if user.get('friends') else 0
            }

print(f"加载了 {len(users)} 个用户")

# Step 4: 合并数据
print("\n步骤4: 合并数据...")
reviews_df = pd.DataFrame(reviews)
businesses_df = pd.DataFrame(businesses)

# 创建商家字典以便快速查找
business_dict = {b['business_id']: b for b in businesses}

# 为每条评论添加商家和用户信息
for i, review in enumerate(tqdm(reviews, desc="合并评论数据")):
    business = business_dict.get(review['business_id'], {})
    user = users.get(review['user_id'], {})

    reviews[i]['business_name'] = business.get('name', '')
    reviews[i]['city'] = business.get('city', '')
    reviews[i]['state'] = business.get('state', '')
    reviews[i]['business_avg_rating'] = business.get('stars', 0)
    reviews[i]['categories'] = business.get('categories', '')
    reviews[i]['user_review_count'] = user.get('user_review_count', 0)
    reviews[i]['user_avg_rating'] = user.get('user_avg_rating', 0)
    reviews[i]['num_friends'] = user.get('num_friends', 0)

# 重新创建DataFrame
reviews_merged_df = pd.DataFrame(reviews)

# 重命名stars为rating以保持一致性
if 'stars' in reviews_merged_df.columns:
    reviews_merged_df['rating'] = reviews_merged_df['stars']

# Step 5: 保存数据
print("\n步骤5: 保存处理后的数据...")
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

reviews_merged_df.to_csv(PROCESSED_DATA_DIR / 'reviews_merged.csv', index=False)
businesses_df.to_csv(PROCESSED_DATA_DIR / 'businesses.csv', index=False)

print(f"\n✅ 保存成功!")
print(f"   - 评论数据: {len(reviews_merged_df)} 条")
print(f"   - 商家数据: {len(businesses_df)} 个")
print(f"   - 匹配率: 100% (所有商家都有评论)")

# 统计信息
review_counts = reviews_merged_df['business_id'].value_counts()
print(f"\n📊 评论数统计:")
print(f"   - 平均每个商家: {review_counts.mean():.2f} 条评论")
print(f"   - 中位数: {review_counts.median():.0f} 条")
print(f"   - 最多: {review_counts.max()} 条")
print(f"   - 最少: {review_counts.min()} 条")

print("\n" + "="*60)
print("数据重新处理完成！")
print("="*60)
