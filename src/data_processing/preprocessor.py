"""
文本预处理模块
包含清洗、分词、词形还原等功能
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy
import logging

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    文本预处理器
    支持多种预处理方法
    """

    def __init__(self, method='nltk', lowercase=True, remove_punct=True,
                 remove_stopwords=True, lemmatize=True):
        """
        初始化预处理器

        Args:
            method: 'nltk' 或 'spacy'
            lowercase: 是否转小写
            remove_punct: 是否移除标点
            remove_stopwords: 是否移除停用词
            lemmatize: 是否词形还原
        """
        self.method = method
        self.lowercase = lowercase
        self.remove_punct = remove_punct
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize

        # 初始化工具
        if method == 'nltk':
            # 下载必要的NLTK数据
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)

            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords', quiet=True)

            try:
                nltk.data.find('corpora/wordnet')
            except LookupError:
                nltk.download('wordnet', quiet=True)

            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()

        elif method == 'spacy':
            try:
                self.nlp = spacy.load('en_core_web_sm')
            except OSError:
                logger.warning("spaCy model not found. Please run: python -m spacy download en_core_web_sm")
                raise

    def clean_text(self, text):
        """
        基础文本清洗

        Args:
            text: 输入文本

        Returns:
            str: 清洗后的文本
        """
        if not isinstance(text, str):
            return ""

        # 移除URL
        text = re.sub(r'http\S+|www.\S+', '', text)

        # 移除HTML标签
        text = re.sub(r'<.*?>', '', text)

        # 移除邮箱
        text = re.sub(r'\S+@\S+', '', text)

        # 移除多余空白
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def process_nltk(self, text):
        """
        使用NLTK进行预处理

        Args:
            text: 输入文本

        Returns:
            str: 处理后的文本
        """
        # 基础清洗
        text = self.clean_text(text)

        # 转小写
        if self.lowercase:
            text = text.lower()

        # 分词
        tokens = word_tokenize(text)

        # 移除标点
        if self.remove_punct:
            tokens = [token for token in tokens if token not in string.punctuation]

        # 移除停用词
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]

        # 词形还原
        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]

        return ' '.join(tokens)

    def process_spacy(self, text):
        """
        使用spaCy进行预处理

        Args:
            text: 输入文本

        Returns:
            str: 处理后的文本
        """
        # 基础清洗
        text = self.clean_text(text)

        # 转小写
        if self.lowercase:
            text = text.lower()

        # spaCy处理
        doc = self.nlp(text)

        tokens = []
        for token in doc:
            # 移除标点
            if self.remove_punct and token.is_punct:
                continue

            # 移除停用词
            if self.remove_stopwords and token.is_stop:
                continue

            # 词形还原
            if self.lemmatize:
                tokens.append(token.lemma_)
            else:
                tokens.append(token.text)

        return ' '.join(tokens)

    def process(self, text):
        """
        处理文本（根据初始化时选择的方法）

        Args:
            text: 输入文本

        Returns:
            str: 处理后的文本
        """
        if self.method == 'nltk':
            return self.process_nltk(text)
        elif self.method == 'spacy':
            return self.process_spacy(text)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def process_batch(self, texts, show_progress=True):
        """
        批量处理文本

        Args:
            texts: 文本列表
            show_progress: 是否显示进度条

        Returns:
            list: 处理后的文本列表
        """
        if show_progress:
            from tqdm import tqdm
            return [self.process(text) for text in tqdm(texts, desc="Processing texts")]
        else:
            return [self.process(text) for text in texts]


class SentimentPreprocessor(TextPreprocessor):
    """
    专门用于情感分析的预处理器
    保留否定词和情感相关的标点
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # 否定词不应该被移除
        self.negation_words = {
            'not', 'no', 'never', 'neither', 'nobody', 'nothing',
            'nowhere', 'none', "don't", "doesn't", "didn't", "won't",
            "wouldn't", "shouldn't", "couldn't", "can't", "isn't",
            "aren't", "wasn't", "weren't", "hasn't", "haven't", "hadn't"
        }

        # 从停用词中移除否定词
        if self.method == 'nltk':
            self.stop_words = self.stop_words - self.negation_words

    def clean_text(self, text):
        """
        情感分析的文本清洗
        保留感叹号等情感标点
        """
        text = super().clean_text(text)

        # 保留感叹号和问号的情感信息
        text = re.sub(r'!+', ' EXCLAMATION ', text)
        text = re.sub(r'\?+', ' QUESTION ', text)

        return text


def main():
    """
    测试预处理器
    """
    # 测试文本
    test_texts = [
        "This restaurant is AMAZING!!! The food was delicious and the service was great!",
        "I don't like this place. The food wasn't good at all.",
        "Check out our website: http://example.com for more info!",
        "Contact us at info@restaurant.com"
    ]

    print("="*60)
    print("NLTK预处理器测试")
    print("="*60)

    # NLTK预处理器
    nltk_processor = TextPreprocessor(method='nltk')
    for text in test_texts:
        processed = nltk_processor.process(text)
        print(f"\n原文: {text}")
        print(f"处理后: {processed}")

    print("\n" + "="*60)
    print("情感分析预处理器测试")
    print("="*60)

    # 情感分析预处理器
    sentiment_processor = SentimentPreprocessor(method='nltk')
    for text in test_texts:
        processed = sentiment_processor.process(text)
        print(f"\n原文: {text}")
        print(f"处理后: {processed}")

    # 测试spaCy（如果可用）
    try:
        print("\n" + "="*60)
        print("spaCy预处理器测试")
        print("="*60)

        spacy_processor = TextPreprocessor(method='spacy')
        for text in test_texts:
            processed = spacy_processor.process(text)
            print(f"\n原文: {text}")
            print(f"处理后: {processed}")
    except:
        print("\nspaCy model not available. Skipping spaCy test.")


if __name__ == '__main__':
    main()
