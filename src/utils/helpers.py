"""
辅助函数
常用的工具函数
"""

import time
from functools import wraps
import logging

logger = logging.getLogger(__name__)


def timing_decorator(func):
    """
    计时装饰器
    用于测量函数执行时间
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed = end_time - start_time

        logger.info(f"{func.__name__} executed in {elapsed:.4f} seconds")
        return result

    return wrapper


def format_number(num):
    """格式化数字（添加千位分隔符）"""
    return f"{num:,}"


def truncate_text(text, max_length=100, suffix='...'):
    """截断文本"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + suffix


def calculate_percentage(part, total):
    """计算百分比"""
    if total == 0:
        return 0
    return (part / total) * 100


def safe_divide(numerator, denominator, default=0):
    """安全除法（避免除零错误）"""
    try:
        return numerator / denominator
    except ZeroDivisionError:
        return default


def batch_iterator(iterable, batch_size):
    """
    批次迭代器
    将可迭代对象分批处理

    Example:
        for batch in batch_iterator(range(100), batch_size=10):
            process(batch)
    """
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


class ProgressTracker:
    """
    进度跟踪器
    用于跟踪长时间运行的任务
    """

    def __init__(self, total, description="Processing"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()

    def update(self, n=1):
        """更新进度"""
        self.current += n
        percentage = (self.current / self.total) * 100
        elapsed = time.time() - self.start_time
        eta = (elapsed / self.current) * (self.total - self.current) if self.current > 0 else 0

        logger.info(f"{self.description}: {self.current}/{self.total} ({percentage:.1f}%) "
                    f"Elapsed: {elapsed:.1f}s ETA: {eta:.1f}s")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        total_time = time.time() - self.start_time
        logger.info(f"{self.description} completed in {total_time:.1f}s")


def get_sentiment_emoji(sentiment):
    """根据情感返回emoji"""
    emojis = {
        'Positive': '😊',
        'Neutral': '😐',
        'Negative': '😞'
    }
    return emojis.get(sentiment, '❓')


def get_rating_stars(rating):
    """将评分转换为星星表示"""
    full_stars = int(rating)
    half_star = 1 if (rating - full_stars) >= 0.5 else 0
    empty_stars = 5 - full_stars - half_star

    return '⭐' * full_stars + '✨' * half_star + '☆' * empty_stars


def main():
    """测试辅助函数"""
    # 测试计时装饰器
    @timing_decorator
    def slow_function():
        time.sleep(1)
        return "Done"

    result = slow_function()
    print(f"Result: {result}")

    # 测试批次迭代器
    print("\nBatch iterator test:")
    for i, batch in enumerate(batch_iterator(range(25), batch_size=10)):
        print(f"Batch {i + 1}: {batch}")

    # 测试进度跟踪器
    print("\nProgress tracker test:")
    with ProgressTracker(100, "Test task") as tracker:
        for i in range(10):
            time.sleep(0.1)
            tracker.update(10)

    # 测试其他函数
    print(f"\nFormat number: {format_number(1234567)}")
    print(f"Truncate text: {truncate_text('This is a very long text that needs to be truncated', 20)}")
    print(f"Percentage: {calculate_percentage(25, 100):.1f}%")
    print(f"Rating stars: {get_rating_stars(4.5)}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
