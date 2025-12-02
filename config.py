"""
配置文件
Configuration file for Yelp Review System
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 项目根目录
BASE_DIR = Path(__file__).parent

# 加载.env文件
load_dotenv(BASE_DIR / '.env')

# 数据目录
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MODEL_DIR = DATA_DIR / 'models'

# Yelp数据集路径
YELP_DATASET_DIR = BASE_DIR / 'yelp_dataset'
REVIEW_FILE = YELP_DATASET_DIR / 'yelp_academic_dataset_review.json'
BUSINESS_FILE = YELP_DATASET_DIR / 'yelp_academic_dataset_business.json'
USER_FILE = YELP_DATASET_DIR / 'yelp_academic_dataset_user.json'
TIP_FILE = YELP_DATASET_DIR / 'yelp_academic_dataset_tip.json'
CHECKIN_FILE = YELP_DATASET_DIR / 'yelp_academic_dataset_checkin.json'

# 采样参数
SAMPLE_SIZE = 100000  # 从690万评论中采样10万条用于开发
TEST_SIZE = 0.2
RANDOM_STATE = 42

# 文本预处理参数
MAX_TEXT_LENGTH = 512  # BERT最大长度
MIN_TEXT_LENGTH = 10   # 最小文本长度（过滤太短的评论）

# 模型参数
# 情感分析
SENTIMENT_MODEL_NAME = 'bert-base-uncased'
SENTIMENT_BATCH_SIZE = 16
SENTIMENT_EPOCHS = 3
SENTIMENT_LR = 2e-5

# 评分预测
RATING_FEATURES = ['text_embedding', 'review_length', 'user_review_count', 'business_stars']

# 主题建模
NUM_TOPICS = 10  # LDA主题数量
NUM_WORDS = 10   # 每个主题显示的词数

# 推荐系统
TOP_K_RECOMMENDATIONS = 10
MIN_REVIEWS_FOR_RECOMMENDATION = 5

# 数据库配置
DATABASE_URI = f"sqlite:///{DATA_DIR / 'yelp_reviews.db'}"

# Flask配置
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5001
DEBUG = True

# API配置
API_PREFIX = '/api'

# OpenAI配置
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')  # 从环境变量读取
OPENAI_MODEL = 'gpt-4o-mini'  # 使用cost-effective的模型
OPENAI_TEMPERATURE = 0.7
OPENAI_MAX_TOKENS = 1000

# SerpApi配置 (用于实时Yelp数据抓取)
SERPAPI_KEY = os.getenv('SERPAPI_KEY', '')  # 从环境变量读取
SERPAPI_ENABLED = bool(SERPAPI_KEY)  # 如果有API key则启用实时抓取

# 日志配置
LOG_DIR = BASE_DIR / 'logs'
LOG_LEVEL = 'INFO'

# 创建必要的目录
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
